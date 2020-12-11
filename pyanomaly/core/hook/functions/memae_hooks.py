"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
from tsnecuda import TSNE
from scipy.ndimage import gaussian_filter1d

from ..abstract import EvaluateHook
from pyanomaly.datatools.evaluate.utils import reconstruction_loss
from pyanomaly.datatools.abstract.readers import GroundTruthLoader
from pyanomaly.core.utils import tsne_vis, save_score_results, tensorboard_vis_images

from ..hook_registry import HOOK_REGISTRY

__all__ = ['MemAEEvaluateHook']

@HOOK_REGISTRY.register()
class MemAEEvaluateHook(EvaluateHook):
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.set_requires_grad(self.trainer.MemAE, False)
        self.trainer.MemAE.eval()
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]
        frame_num = self.trainer.config.DATASET.val.clip_length
        clip_step = self.trainer.config.DATASET.val.clip_step
        psnr_records=[]
        score_records=[]
        # total = 0
        num_videos = 0
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))
        # calc the score for the test dataset
        for sn, video_name in enumerate(self.trainer.test_dataset_keys):
            num_videos += 1
            # need to improve
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            # test_iters = len_dataset // clip_step
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            # scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = torch.zeros(len_dataset)
            # scores = [0.0 for i in range(len_dataset)]
            for clip_sn, (test_input, anno, meta) in enumerate(data_loader):
                test_target = test_input.cuda()
                time_len = test_input.shape[2]
                output, _ = self.trainer.MemAE(test_target)
                clip_score = reconstruction_loss(output, test_target)
                
                # scores[test_counter*time_len:(test_counter + 1)*time_len] = clip_score.squeeze(0)
                if (frame_num + test_counter) > len_dataset:
                    temp = test_counter + frame_num - len_dataset
                    scores[test_counter:len_dataset] = clip_score[temp:]
                else:
                    scores[test_counter:(frame_num + test_counter)] = clip_score
                        
                test_counter += 1

                if sn == random_video_sn and (clip_sn in vis_range):
                    vis_objects = OrderedDict({
                        'memae_eval_clip': test_target.detach(),
                        'memae_eval_clip_hat': output.detach()
                    })
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.normalize.param['val'])
                
                if test_counter >= test_iters:
                    # import ipdb; ipdb.set_trace()
                    # scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    smin = min(scores)
                    # normal_scores = np.array([(1.0 - np.divide(s-smin, smax)) for s in scores])
                    normal_scores = (1.0 - torch.div(scores-smin, smax-smin)).detach().cpu().numpy()
                    normal_scores = np.clip(normal_scores, 0, None)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break
        
        self.trainer.pkl_path = save_score_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('amc: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc
