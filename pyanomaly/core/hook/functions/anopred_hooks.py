"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import numpy as np
import torch
import os
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.core.utils import flow_batch_estimate, tensorboard_vis_images, save_score_results
from pyanomaly.datatools.evaluate.utils import simple_diff, find_max_patch, amc_score, calc_w

from ..abstract import EvaluateHook
from ..hook_registry import HOOK_REGISTRY

__all__ = ['AnoPredEvaluateHook']

@HOOK_REGISTRY.register()
class AnoPredEvaluateHook(EvaluateHook):
    
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        # self.trainer.set_requires_grad(self.trainer.F, False)
        # self.trainer.set_requires_grad(self.trainer.G, False)
        # self.trainer.set_requires_grad(self.trainer.D, False)
        # self.trainer.G.eval()
        # self.trainer.D.eval()
        # self.trainer.F.eval()
        self.engine.set_all(False) # eval mode
        tb_writer = self.engine.kwargs['writer_dict']['writer']
        global_steps = self.engine.kwargs['writer_dict']['global_steps_{}'.format(self.engine.kwargs['model_type'])]
        frame_num = self.engine.config.DATASET.val.clip_length
        psnr_records=[]
        score_records=[]
        # total = 0

        # for dirs in video_dirs:
        random_video_sn = torch.randint(0, len(self.engine.val_dataset_keys), (1,))
        for sn, video_name in enumerate(self.engine.val_dataset_keys):

            # need to improve
            dataset = self.engine.val_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            
            for frame_sn, (test_input, anno, meta) in enumerate(data_loader):
                test_target = test_input[:, :, -1, :, :].cuda()
                test_input = test_input[:, :, :-1, :, :].reshape(test_input.shape[0], -1, test_input.shape[-2],test_input.shape[-1]).cuda()

                g_output = self.engine.G(test_input)
                test_psnr = psnr_error(g_output.detach(), test_target, hat=True)
                test_psnr = test_psnr.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr
                scores[test_counter+frame_num-1]=test_psnr

                test_counter += 1
                # total+=1
                if sn == random_video_sn and (frame_sn in vis_range):
                    vis_objects = OrderedDict({
                        'anopred_eval_frame': test_target.detach(),
                        'anopred_eval_frame_hat': g_output.detach()
                    })
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.engine.normalize.param['val'])
                
                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    scores[:frame_num-1]=(scores[frame_num-1],)
                    smax = max(scores)
                    smin = min(scores)
                    normal_scores = np.array([np.divide(s-smin, smax-smin) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    # print(f'finish test video set {video_name}')
                    break

        self.engine.pkl_path = save_score_results(self.engine.config, self.engine.logger, verbose=self.engine.verbose, config_name=self.engine.config_name, current_step=current_step, time_stamp=self.engine.kwargs["time_stamp"],score=score_records, psnr=psnr_records)
        results = self.engine.evaluate_function(self.engine.pkl_path, self.engine.logger, self.engine.config, self.engine.config.DATASET.score_type)
        self.engine.logger.info(results)
        tb_writer.add_text('anopcn: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc
        
# def get_anopred_hooks(name):
#     if name in HOOKS:
#         t = eval(name)()
#     else:
#         raise Exception('The hook is not in amc_hooks')
#     return t
