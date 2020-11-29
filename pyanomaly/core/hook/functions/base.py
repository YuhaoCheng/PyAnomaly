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

from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.datatools.abstract.readers import GroundTruthLoader
from pyanomaly.core.utils import tsne_vis

from .abstract_hook import HookBase
from ..hook_registry import HOOK_REGISTRY

__all__ = ['VisScoreHook', 'TSNEHook']

@HOOK_REGISTRY.register()
class VisScoreHook(HookBase):
    def after_step(self, current_step):
        writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]

        if not os.path.exists(self.trainer.config.LOG.vis_dir):
            os.mkdir(self.trainer.config.LOG.vis_dir)
        
        if current_step % self.trainer.steps.param['eval'] == 0 and current_step != 0:
            with open(self.trainer.pkl_path, 'rb') as reader:
                results = pickle.load(reader)
            print(f'Vis The results in {self.trainer.pkl_path}')
            sigma = self.trainer.config.DATASET.smooth.guassian_sigma[0]
            psnrs = results['psnr']
            smooth_psnrs = results[f'psnr_smooth_{sigma}']
            scores = results['score']
            smooth_scores = results[f'score_smooth_{sigma}']
            gt_loader = GroundTruthLoader(self.trainer.config)
            gt = gt_loader()
            if psnrs == []:
                for i in range(len(scores)):
                    psnrs.append(np.zeros(shape=(scores[i].shape[0],)))
                    smooth_psnrs.append(np.zeros(shape=(scores[i].shape[0],)))
            elif scores == []:
                for i in range(len(psnrs)):
                    scores.append(np.zeros(shape=(psnrs[i].shape[0],)))
                    smooth_scores.append(np.zeros(shape=(scores[i].shape[0],)))
            else:
                assert len(psnrs) == len(scores), 'the number of psnr and score is not equal'
            
            assert len(gt) == len(psnrs) == len(scores), f'the number of gt {len(gt)}, psnrs {len(psnrs)}, scores {len(scores)}'
            
            # plt the figure
            for video_id in range(len(psnrs)):
                assert len(psnrs[video_id]) == len(scores[video_id]) == len(gt[video_id]), f'video_id:{video_id},the number of gt {len(gt)}, psnrs {len(psnrs)}, scores {len(scores)}'
                fig = plt.figure()
                fig.tight_layout()
                fig.subplots_adjust(wspace=0.6)
                ax1 = fig.add_subplot(2,3,1)
                ax1.plot([i for i in range(len(psnrs[video_id]))], psnrs[video_id])
                ax1.set_ylabel('psnr')
                ax2 = fig.add_subplot(2,3,2)
                ax2.plot([i for i in range(len(scores[video_id]))], scores[video_id])
                ax2.set_ylabel('score')
                ax3 = fig.add_subplot(2,3,3)
                ax3.plot([i for i in range(len(gt[video_id]))], gt[video_id])
                ax3.set_ylabel('GT')
                ax3.set_xlabel('frames')
                ax4 = fig.add_subplot(2,3,4)
                ax4.plot([i for i in range(len(smooth_scores[video_id]))], smooth_scores[video_id])
                ax4.set_ylabel(f'Guassian Smooth score{sigma}')
                ax4.set_xlabel('frames')
                ax5 = fig.add_subplot(2,3,5)
                ax5.plot([i for i in range(len(smooth_psnrs[video_id]))], smooth_psnrs[video_id])
                ax5.set_ylabel(f'Guassian Smooth PSNR{sigma}')
                writer.add_figure(f'verbose_{self.trainer.verbose}_{self.trainer.config_name}_{self.trainer.kwargs["time_stamp"]}_vis{video_id}', fig, global_steps)
            
        
            self.trainer.logger.info(f'^^^^Finish vis @{current_step}')

@HOOK_REGISTRY.register()
class TSNEHook(HookBase):
    def after_step(self, current_step):
        writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.tainer.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        if not os.path.exists(self.trainer.config.LOG.vis_dir):
            os.mkdir(self.trainer.config.LOG.vis_dir)
        
        if current_step % self.trainer.config.TRAIN.eval_step == 0:
            vis_path = os.path.join(self.trainer.config.LOG.vis_dir, f'{self.trainer.config.DATASET.name}_tsne_model:{self.trainer.config.MODEL.name}_step:{current_step}.jpg')
            feature, feature_labels = self.trainer.analyze_feature
            tsne_vis(feature, feature_labels, vis_path)
            image = cv2.imread(vis_path)
            image = image[:,:,[2,1,0]]
            writer.add_image(str(vis_path), image, global_step=global_steps)


# def get_base_hooks(name):
#     if name in HOOKS:
#         t = eval(name)()
#     else:
#         raise Exception('The hook is not in amc_hooks')
#     return t

        
