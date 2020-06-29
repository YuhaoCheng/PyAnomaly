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

from .abstract.abstract_hook import HookBase
from pyanomaly.datatools.evaluate.utils import reconstruction_loss
from pyanomaly.datatools.evaluate.gtloader import GroundTruthLoader
# from lib.datatools.evaluate import eval_api
from pyanomaly.core.utils import tsne_vis, tensorboard_vis_images, save_results

HOOKS = ['STAEEvaluateHook']

class STAEEvaluateHook(HookBase):
    def after_step(self, current_step):
        acc = 0.0
        if current_step % self.trainer.eval_step == 0 and current_step != 0:
            with torch.no_grad():
                acc = self.evaluate(current_step)
                if acc > self.trainer.accuarcy:
                    self.trainer.accuarcy = acc
                    # save the model & checkpoint
                    self.trainer.save(current_step, best=True)
                elif current_step % self.trainer.save_step == 0 and current_step != 0:
                    # save the checkpoint
                    self.trainer.save(current_step)
                    self.trainer.logger.info('LOL==>the accuracy is not imporved in epcoh{} but save'.format(current_step))
                else:
                    pass
        else:
            pass
    
    def inference(self):
        # import ipdb; ipdb.set_trace()
        self.trainer.set_requires_grad(self.trainer.STAE, False)
        acc = self.evaluate(0)
        self.trainer.logger.info(f'The inference metric is:{acc:.3f}')
    
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.STAE.eval()
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]
        frame_num = self.trainer.config.DATASET.test_clip_length
        clip_step = self.trainer.config.DATASET.test_clip_step
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

            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            for clip_sn, (data, _) in enumerate(data_loader):
                test_input = data.cuda()
                # test_target = data[:,:,16:,:,:].cuda()
                time_len = test_input.shape[2]
                # import ipdb; ipdb.set_trace()
                output, _ = self.trainer.STAE(test_input)
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
                    vis_objects = OrderedDict()
                    vis_objects['stae_eval_clip'] = test_input.detach()
                    vis_objects['stae_eval_clip_hat'] = output.detach()
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.val_normalize, mean=self.trainer.val_mean, std=self.trainer.val_std)
                
                if test_counter >= test_iters:
                    # scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    smin = min(scores)
                    normal_scores = np.array([(1.0 - np.divide(s-smin, smax)) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break
        
        self.trainer.pkl_path = save_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('amc: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc


def get_stae_hooks(name):
    if name in HOOKS:
        t = eval(name)()
    else:
        raise Exception('The hook is not in amc_hooks')
    return t