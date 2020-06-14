import numpy as np
import torch
import os
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from lib.datatools.evaluate.utils import psnr_error
from .abstract.abstract_hook import HookBase
# from lib.datatools.evaluate import eval_api
# from lib.datatools.evaluate.amc_utils import calc_anomaly_score_one_frame
from lib.datatools.evaluate.utils import simple_diff, find_max_patch, amc_score, calc_w

HOOKS = ['AnoPredEvaluateHook']

class AnoPredEvaluateHook(HookBase):
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
        total = 0

        # for dirs in video_dirs:
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))
        for sn, video_name in enumerate(self.trainer.test_dataset_keys):

            # need to improve
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            
            for frame_sn, test_input in enumerate(data_loader):
                test_target = test_input[:, :, -1, :, :].cuda()
                test_input = test_input[:, :, :-1, :, :].reshape(test_input.shape[0], -1, test_input.shape[-2],test_input.shape[-1]).cuda()

                g_output = self.trainer.G(test_input)
                test_psnr = psnr_error(g_output, test_target)
                test_psnr = test_psnr.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr
                scores[test_counter+frame_num-1]=test_psnr

                test_counter += 1
                total+=1
                if sn == random_video_sn and (frame_sn in vis_range):
                    self.add_images(test_target, g_output, tb_writer, global_steps)
                
                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    scores[:frame_num-1]=(scores[frame_num-1],)
                    smax = max(scores)
                    smin = min(scores)
                    normal_scores = np.array([np.divide(s-smin, smax-smin) for s in scores])
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    # print(f'finish test video set {video_name}')
                    break

        result_dict = {'dataset': self.trainer.config.DATASET.name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': [], 'score':score_records, 'num_videos':len(psnr_records)}
        result_path = os.path.join(self.trainer.config.TEST.result_output, f'{self.trainer.verbose}_cfg#{self.trainer.config_name}#step{current_step}@{self.trainer.kwargs["time_stamp"]}_results.pkl')
        with open(result_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
        
        # results = eval_api.evaluate('compute_auc_score', result_path, self.trainer.logger, self.trainer.config)
        results = self.trainer.evaluate_function(result_path, self.trainer.logger, self.trainer.config)
        self.trainer.logger.info(results)
        return results.auc

    def add_images(self, frame, frame_hat, writer, global_steps):
        writer.add_images('ano_pred_frame', frame.detach(), global_steps)
        writer.add_images('ano_pred_frame_hat', frame_hat.detach(), global_steps)
        
def get_anopred_hooks(name):
    if name in HOOKS:
        t = eval(name)()
    else:
        raise Exception('The hook is not in amc_hooks')
    return t
