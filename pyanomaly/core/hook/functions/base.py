"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import os
import pickle
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

from ..abstract import HookBase
from ..hook_registry import HOOK_REGISTRY

__all__ = ['VisScoreHook']

@HOOK_REGISTRY.register()
class VisScoreHook(HookBase):
    """Visualize the scores.
    The method is used to visualize the score of the each video and save in the tensorboard
    """
    def _vis_score_function(self, result_path, verbose, writer, global_steps):
        """Visualization function.
        The function to visualize the score and save in the tensorboard.
        Args:
            result_path(str): The file's name stores the result
            verbose(str): Comments.
            writer(): The tensorboard writer
            global_steps(int): The steps of trianing process
        """
        with open(result_path, 'rb') as reader:
            results = pickle.load(reader)
        scores = results['score']

        gt = self.engine.evaluate_function.gt_dict['val']
        # plt the figure
        for video_id in range(len(scores)):
            fig = plt.figure()
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.6)
            ax2 = fig.add_subplot(1,2,1)
            ax2.plot([i for i in range(len(scores[video_id]))], scores[video_id])
            ax2.set_ylabel(f'score_{verbose}')
            ax3 = fig.add_subplot(1,2,2)
            ax3.plot([i for i in range(len(gt[video_id]))], gt[video_id])
            ax3.set_ylabel('GT')
            ax3.set_xlabel('frames')
            writer.add_figure(f'verbose_{self.engine.verbose}_{verbose}_{self.engine.config_name}_{self.engine.kwargs["time_stamp"]}_vis{video_id}', fig, global_steps)  
        
    def after_step(self, current_step):
        writer = self.engine.kwargs['writer_dict']['writer']
        global_steps = self.engine.kwargs['writer_dict']['global_steps_{}'.format(self.engine.kwargs['model_type'])]

        if not os.path.exists(self.engine.config.LOG.vis_dir):
            os.mkdir(self.engine.config.LOG.vis_dir)
        
        if current_step % self.engine.steps.param['eval'] == 0 and current_step != 0:
            for key, item in self.engine.pkl_path.items():
                logger.info(f'Vis the results in {item}')
                self._vis_score_function(item, key, writer, global_steps)
            
            self.engine.logger.info(f'^^^^Finish vis @{current_step}')

        
