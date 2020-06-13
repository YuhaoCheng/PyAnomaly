import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from lib.datatools.evaluate.utils import psnr_error
from lib.core.utils import flow_batch_estimate
from .abstract.abstract_hook import HookBase
# from lib.datatools.evaluate import eval_api
# from lib.datatools.evaluate.amc_utils import calc_anomaly_score_one_frame
from lib.datatools.evaluate.utils import simple_diff, find_max_patch, amc_score, calc_w

HOOKS = ['AMCEvaluateHook']

class AMCEvaluateHook(HookBase):
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
        acc = self.evaluate(0)
        self.trainer.logger.info(f'The inference metric is:{acc:.3f}')
    
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
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

            for test_input in data_loader:
                # import ipdb; ipdb.set_trace()
                target_test = test_input[:, :, 1, :, :].cuda()
                input_data_test = test_input[:, :, 0, :, :].cuda()
                output_flow_G, output_frame_G = self.trainer.G(input_data_test)
                gtFlowEstim = torch.cat([input_data_test, target_test], 1)
                gtFlow, _ = flow_batch_estimate(self.trainer.F, gtFlowEstim)
                # score = amc_score(test_input, g_output_frame, flow_gt, g_output_flow)
                diff_appe, diff_flow = simple_diff(input_data_test, output_frame_G, gtFlow, output_flow_G)
                patch_score_appe, patch_score_flow, _, _ = find_max_patch(diff_appe, diff_flow)
                # score_appe = torch.mean(max_patch_appe)
                # score_flow = torch.mean(max_patch_flow)
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
            # scores = [0.0 for i in range(len_dataset)]
            for frame_sn, test_input in enumerate(data_loader):
                test_target = test_input[:, :, 1, :, :].cuda()
                test_input = test_input[:, :, 0, :, :].cuda()

                g_output_flow, g_output_frame = self.trainer.G(test_input)
                gt_flow_esti_tensor = torch.cat([test_input, test_target], 1)
                # flow_gt,_ = self.trainer.batch_estimate(gt_flow_esti_tensor)
                flow_gt,_ = flow_batch_estimate(self.trainer.F, gt_flow_esti_tensor)
                # score = amc_score(test_input, g_output_frame, flow_gt, g_output_flow)
                test_psnr = psnr_error(g_output_frame, test_target)
                score, _, _ = amc_score(test_input, g_output_frame, flow_gt, g_output_flow, wf, wi)
                # import ipdb; ipdb.set_trace()
                test_psnr = test_psnr.tolist()
                score = score.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr
                scores[test_counter+frame_num-1] = score
                test_counter += 1

                if sn == random_video_sn and (frame_sn in vis_range):
                    self.add_images(test_target, flow_gt, g_output_frame, g_output_flow, tb_writer, global_steps)
                
                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    # import ipdb; ipdb.set_trace()
                    scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    normal_scores = np.array([np.divide(s, smax) for s in scores])
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break
        
        result_dict = {'dataset': self.trainer.config.DATASET.name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': [], 'score':score_records, 'num_videos':num_videos}
        if not os.path.exists(self.trainer.config.TEST.result_output):
            os.mkdir(self.trainer.config.TEST.result_output)
        result_path = os.path.join(self.trainer.config.TEST.result_output, f'{self.trainer.verbose}_cfg#{self.trainer.config_name}#step{current_step}@{self.trainer.kwargs["time_stamp"]}_results.pkl')
        with open(result_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
        
        # results = eval_api.evaluate('compute_auc_score', result_path, self.trainer.logger, self.trainer.config)
        results = self.trainer.evaluate_function(result_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('amc: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc

    def add_images(self, frame, flow, frame_hat, flow_hat, writer, global_steps):
        frame = self.verse_normalize(frame.detach())
        frame_hat = self.verse_normalize(frame_hat.detach())
        flow = self.verse_normalize(flow.detach())
        flow_hat = self.verse_normalize(flow_hat.detach())
        
        writer.add_images('eval_frame', frame, global_steps)
        writer.add_images('eval_frame_hat', frame_hat, global_steps)
        writer.add_images('eval_flow', flow, global_steps)
        writer.add_images('eval_flow_hat', flow_hat, global_steps)
    
    def verse_normalize(self, image_tensor):
        std = self.trainer.config.ARGUMENT.val.normal.std
        mean = self.trainer.config.ARGUMENT.val.normal.mean
        if len(mean) == 0 and len(std) == 0:
            return image_tensor
        else:
            for i in range(len(std)):
                image_tensor[:,i,:,:] = image_tensor[:,i,:,:] * std[i] + mean[i]
            return image_tensor
    
def get_amc_hooks(name):
    if name in HOOKS:
        t = eval(name)()
    else:
        raise Exception('The hook is not in amc_hooks')
    return t
