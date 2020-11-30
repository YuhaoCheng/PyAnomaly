import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from ..abstract import EvaluateHook

from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.core.utils import flow_batch_estimate, tensorboard_vis_images, save_results, vis_optical_flow
from pyanomaly.datatools.evaluate.utils import simple_diff, find_max_patch, amc_score, calc_w

from ..hook_registry import HOOK_REGISTRY

# HOOKS = ['AMCEvaluateHook']
__all__ = ['AMCEvaluateHook']

@HOOK_REGISTRY.register()
class AMCEvaluateHook(EvaluateHook):
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.set_requires_grad(self.trainer.F, False)
        self.trainer.set_requires_grad(self.trainer.G, False)
        self.trainer.set_requires_grad(self.trainer.D, False)
        self.trainer.D.eval()
        self.trainer.G.eval()
        self.trainer.F.eval()
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]
        frame_num = self.trainer.config.DATASET.test_clip_length
        psnr_records=[]
        score_records=[]
        # score_records_w=[]
        w_dict = OrderedDict()
        # total = 0
        # calc the scores for the training set
        for video_name in self.trainer.test_dataset_keys_w:
            dataset = self.trainer.test_dataset_dict_w[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1 
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            scores = [0.0 for i in range(len_dataset)]

            for data, anno, meta in data_loader:
                input_data_test = data[:, :, 0, :, :].cuda()
                target_test = data[:, :, 1, :, :].cuda()
                output_flow_G, output_frame_G = self.trainer.G(input_data_test)
                gtFlowEstim = torch.cat([input_data_test, target_test], 1)
                gtFlow_vis, gtFlow = flow_batch_estimate(self.trainer.F, gtFlowEstim, self.trainer.normalize.param['val'], 
                                                         output_format=self.trainer.config.DATASET.optical_format, optical_size=self.trainer.config.DATASET.optical_size)
                diff_appe, diff_flow = simple_diff(target_test, output_frame_G, gtFlow, output_flow_G)
                patch_score_appe, patch_score_flow, _, _ = find_max_patch(diff_appe, diff_flow)
                scores[test_counter+frame_num-1] = [patch_score_appe, patch_score_flow]
                test_counter += 1
                if test_counter >= test_iters:
                    scores[:frame_num-1] = [scores[frame_num-1]]
                    scores = torch.tensor(scores)
                    frame_w =  torch.mean(scores[:,0])
                    flow_w = torch.mean(scores[:,1])
                    w_dict[video_name] = [len_dataset, frame_w, flow_w]
                    print(f'finish calc the scores of training set {video_name} in step:{current_step}')
                    break
        wf, wi = calc_w(w_dict)
        # wf , wi = 1.0, 1.0
        tb_writer.add_text('weight of train set', f'w_f:{wf:.3f}, w_i:{wi:.3f}', global_steps)
        print(f'wf:{wf}, wi:{wi}')
        num_videos = 0
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))
        # calc the score for the test dataset
        for sn, video_name in enumerate(self.trainer.test_dataset_keys):
            num_videos += 1
            # need to improve
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            for frame_sn, (data, anno, meta) in enumerate(data_loader):
                test_input = data[:, :, 0, :, :].cuda()
                test_target = data[:, :, 1, :, :].cuda()

                g_output_flow, g_output_frame = self.trainer.G(test_input)
                gt_flow_esti_tensor = torch.cat([test_input, test_target], 1)
                flow_gt_vis, flow_gt = flow_batch_estimate(self.trainer.F, gt_flow_esti_tensor, self.trainer.param['val'], 
                                                          output_format=self.trainer.config.DATASET.optical_format, optical_size=self.trainer.config.DATASET.optical_size)
                test_psnr = psnr_error(g_output_frame, test_target)
                score, _, _ = amc_score(test_target, g_output_frame, flow_gt, g_output_flow, wf, wi)
                test_psnr = test_psnr.tolist()
                score = score.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr
                scores[test_counter+frame_num-1] = score
                test_counter += 1

                if sn == random_video_sn and (frame_sn in vis_range):
                    temp = vis_optical_flow(g_output_flow.detach(), output_format=self.trainer.config.DATASET.optical_format, output_size=(g_output_flow.shape[-2], g_output_flow.shape[-1]), 
                                            normalize=self.trainer.normalize.param['val'])
                    vis_objects = OrderedDict({
                        'amc_eval_frame': test_target.detach(),
                        'amc_eval_frame_hat': g_output_frame.detach(),
                        'amc_eval_flow': flow_gt_vis.detach(),
                        'amc_eval_flow_hat': temp 
                    })
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.normalize.param['val'])
                
                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    # import ipdb; ipdb.set_trace()
                    scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    normal_scores = np.array([np.divide(s, smax) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break
        
        self.trainer.pkl_path = save_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records, psnr=psnr_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('AMC: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc

    
# def get_amc_hooks(name):
#     if name in HOOKS:
#         t = eval(name)()
#     else:
#         raise Exception('The hook is not in amc_hooks')
#     return t
