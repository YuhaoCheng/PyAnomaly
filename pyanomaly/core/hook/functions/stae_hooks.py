"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict

from ..abstract import EvaluateHook
from ..hook_registry import HOOK_REGISTRY

from pyanomaly.datatools.evaluate.utils import reconstruction_loss
from pyanomaly.core.utils import tensorboard_vis_images, save_score_results

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
        # Set basic things
        self.engine.set_all(False) # eval mode
        tb_writer = self.engine.kwargs['writer_dict']['writer']
        global_steps = self.engine.kwargs['writer_dict']['global_steps_{}'.format(self.engine.kwargs['model_type'])]
        frame_num = self.engine.config.DATASET.val.sampled_clip_length
        score_records=[]
        # num_videos = 0
        random_video_sn = torch.randint(0, len(self.engine.val_dataset_keys), (1,))

        # calc the score for the test dataset
        for sn, video_name in enumerate(self.engine.val_dataset_keys):
            # num_videos += 1
            # need to improve
            dataloader = self.engine.val_dataloaders_dict['general_dataset_dict'][video_name]
            len_dataset = dataloader.dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            # test_iters = len_dataset // clip_step
            test_counter = 0

            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))

            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            for clip_sn, (test_input, anno, meta) in enumerate(dataloader):
                test_input = test_input.cuda()
                # test_target = data[:,:,16:,:,:].cuda()
                time_len = test_input.shape[2]
                output, _ = self.engine.STAE(test_input)
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
                    logger.info(f'Finish testing the video:{video_name}')
                    break
        
        # Compute the metrics based on the model's results
        self.engine.pkl_path = save_score_results(score_records, self.engine.config, self.engine.logger, verbose=self.engine.verbose, config_name=self.engine.config_name, current_step=current_step, time_stamp=self.engine.kwargs["time_stamp"])
        results = self.engine.evaluate_function.compute({'val': self.engine.pkl_path})        
        self.engine.logger.info(results)

        # Write the metric into the tensorboard
        tb_writer.add_text(f'{self.engine.config.MODEL.name}: AUC of ROC curve', f'auc is {results.avg_value}', global_steps)

        return results.avg_value

