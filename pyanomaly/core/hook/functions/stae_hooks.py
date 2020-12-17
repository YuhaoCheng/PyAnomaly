"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
# import os
# import pickle
# import cv2
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
# from torch.utils.data import DataLoader
from collections import OrderedDict
# import matplotlib.pyplot as plt
# from tsnecuda import TSNE
# from scipy.ndimage import gaussian_filter1d

from ..abstract import EvaluateHook
from pyanomaly.datatools.evaluate.utils import reconstruction_loss
# from pyanomaly.datatools.abstract.readers import GroundTruthLoader
# from pyanomaly.core.utils import tsne_vis, tensorboard_vis_images, save_score_results
from pyanomaly.core.utils import tensorboard_vis_images, save_score_results

from ..hook_registry import HOOK_REGISTRY

__all__ = ['STAEEvaluateHook']

@HOOK_REGISTRY.register()
class STAEEvaluateHook(EvaluateHook): 
    def evaluate(self, current_step):
        """STAE evaluation method. 

        Evaluate the model base on some methods.
        Args:
            current_step: The current step at present
        Returns:
            results: The magnitude of the method based on this evaluation metric
        """
        # self.trainer.set_requires_grad(self.trainer.STAE, False)
        # self.trainer.STAE.eval()
        self.engine.set_all(False) # eval mode
        tb_writer = self.engine.kwargs['writer_dict']['writer']
        global_steps = self.engine.kwargs['writer_dict']['global_steps_{}'.format(self.engine.kwargs['model_type'])]
        frame_num = self.engine.config.DATASET.val.sampled_clip_length
        # clip_step = self.engine.config.DATASET.val.clip_step
        # psnr_records=[]
        score_records=[]
        # total = 0
        num_videos = 0
        random_video_sn = torch.randint(0, len(self.engine.val_dataset_keys), (1,))
        # import ipdb; ipdb.set_trace()
        # calc the score for the test dataset

        for sn, video_name in enumerate(self.engine.val_dataset_keys):
            num_videos += 1
            # need to improve
            # dataset = self.trainer.test_dataset_dict[video_name]
            # dataloader = self.engine.val_dataloaders_dict[video_name]
            dataloader = self.engine.val_dataloaders_dict['general_dataset_dict'][video_name]
            # len_dataset = dataset.pics_len
            len_dataset = dataloader.dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            # test_iters = len_dataset // clip_step
            test_counter = 0

            # data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))

            # psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            # for clip_sn, (test_input, anno, meta) in enumerate(data_loader):
            for clip_sn, (test_input, anno, meta) in enumerate(dataloader):
                test_input = test_input.cuda()
                # test_target = data[:,:,16:,:,:].cuda()
                time_len = test_input.shape[2]
                # import ipdb; ipdb.set_trace()
                output, _ = self.engine.STAE(test_input)
                # import ipdb; ipdb.set_trace()
                clip_score = reconstruction_loss(output, test_input)
                clip_score = clip_score.tolist()

                if (frame_num + test_counter) > len_dataset:
                    temp = test_counter + frame_num - len_dataset
                    scores[test_counter:len_dataset] = clip_score[temp:]
                else:
                    scores[test_counter:(frame_num + test_counter)] = clip_score

                test_counter += 1

                if sn == random_video_sn and (clip_sn in vis_range):
                    vis_objects = OrderedDict({
                        'stae_eval_clip': test_input.detach(),
                        'stae_eval_clip_hat': output.detach()
                    })
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.engine.normalize.param['val'])
                
                if test_counter >= test_iters:
                    # scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    smin = min(scores)
                    normal_scores = np.array([(1.0 - np.divide(s-smin, smax)) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    score_records.append(normal_scores)
                    # print(f'finish test video set {video_name}')
                    logger.info(f'Finish testing the video:{video_name}')
                    break
        # import ipdb; ipdb.set_trace()

        # self.engine.pkl_path = save_score_results(self.engine.config, self.engine.logger, verbose=self.engine.verbose, config_name=self.engine.config_name, current_step=current_step, time_stamp=self.engine.kwargs["time_stamp"], score=score_records)
        self.engine.pkl_path = save_score_results(score_records, self.engine.config, self.engine.logger, verbose=self.engine.verbose, config_name=self.engine.config_name, current_step=current_step, time_stamp=self.engine.kwargs["time_stamp"])
        # results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        
        # for result_path in self.engine.pkl_path:
        #     results = self.engine.evaluate_function.compute({'val':{'default':result_path}})
        results = self.engine.evaluate_function.compute({'val': self.engine.pkl_path})
        
        self.engine.logger.info(results)
        tb_writer.add_text(f'{self.engine.config.MODEL.name}: AUC of ROC curve', f'auc is {results.avg_value}', global_steps)

        return results.avg_value

