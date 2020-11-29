import numpy as np
import torch
import os
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from pyanomaly.datatools.evaluate.utils import psnr_error
from .abstract_hook import EvaluateHook
from pyanomaly.core.utils import save_results, tensorboard_vis_images
from ..hook_registry import HOOK_REGISTRY

__all__ = ['AnoPCNEvaluateHook']

@HOOK_REGISTRY.register()
class AnoPCNEvaluateHook(EvaluateHook):
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.set_requires_grad(self.trainer.F, False)
        self.trainer.set_requires_grad(self.trainer.G, False)
        self.trainer.set_requires_grad(self.trainer.D, False)
        self.trainer.G.eval()
        self.trainer.D.eval()
        self.trainer.F.eval()
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]

        frame_num = self.trainer.config.DATASET.test_clip_length
        psnr_records=[]
        score_records=[]
        total = 0

        # for dirs in video_dirs:
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))

        for sn, video_name in enumerate(self.trainer.test_dataset_keys):
            # _temp_test_folder = os.path.join(self.testing_data_folder, dir)

            # need to improve
            # dataset = AvenueTestOld(_temp_test_folder, clip_length=frame_num)
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            # import ipdb; ipdb.set_trace()
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            # for test_input, _ in data_loader:
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            for frame_sn, (test_input, anno, meta) in enumerate(data_loader):
                test_target = test_input[:, :, -1, :, :].cuda()
                test_input = test_input[:, :, :-1, :, :].cuda()

                _, g_output = self.trainer.G(test_input, test_target)
                test_psnr = psnr_error(g_output, test_target, hat=False)
                test_psnr = test_psnr.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr
                scores[test_counter+frame_num-1]=test_psnr
                
                if sn == random_video_sn and (frame_sn in vis_range):
                    vis_objects = OrderedDict({
                        'anopcn_eval_frame': test_target.detach(),
                        'anopcn_eval_frame_hat': g_output.detach()
                    })
                    # vis_objects['anopcn_eval_frame'] = test_target.detach()
                    # vis_objects['anopcn_eval_frame_hat'] = g_output.detach()
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.normalize.param['val'])
                test_counter += 1
                total+=1

                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    scores[:frame_num-1]=(scores[frame_num-1],)
                    smax = max(scores)
                    smin = min(scores)
                    normal_scores = np.array([np.divide(s-smin, smax-smin) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break

        self.trainer.pkl_path = save_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records, psnr=psnr_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('anopcn: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc

        
# def get_anopcn_hooks(name):
#     if name in HOOKS:
#         t = eval(name)()
#     else:
#         raise Exception('The hook is not in amc_hooks')
#     return t
