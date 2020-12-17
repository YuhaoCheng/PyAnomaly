'''
The eval api of the anomaly detection.
We will store the results in these structure and use off-line style to compute the metric
results {
           'dataset': the name of dataset
           'psnr': the psnr of each testing videos,
        }
'''
import numpy as np
import os
import argparse
import pickle
from sklearn import metrics
import json
from collections import OrderedDict

from .utils import load_pickle_results
from ..abstract import GroundTruthLoader, AbstractEvalMethod
from ..tools import RecordResult
from ..datatools_registry import EVAL_METHOD_REGISTRY

__all__ = ['ScoreAUCMetrics']

@EVAL_METHOD_REGISTRY.register()
class ScoreAUCMetrics(AbstractEvalMethod):
    def __init__(self, cfg, is_training) -> None:
        super(ScoreAUCMetrics, self).__init__(cfg)
        self.gt_loader = GroundTruthLoader()
        # self.dataset_name = cfg.DATASET.name
        # self.gt_path = cfg.DATASET.gt_path
        self.optimal_resulst = RecordResult()
        self.decidable_idx = self.dataset_params.decidable_idx
        self.decidable_idx_back = self.dataset_params.decidable_idx
        if is_training:
            self.parts = ['train', 'val']
        else:
            self.parts = ['val']
            
        if self.dataset_params.score_type == 'normal':
            self.pos_label = 0
        elif self.dataset_params.score_type == 'abnormal':
            self.pos_label = 1
        else:
            raise Exception(f'Not support the score type:{self.dataset_params.score_type}')

        self.gt_dict = self.load_ground_truth()
        self.result_type = None

    def load_ground_truth(self):
        gt_dict = OrderedDict()
        for part in self.parts:
            #=====================Need to change, temporal=========================
            if part == 'train':
                continue
            #=======================================================================
            gt_path = self.dataset_params[part]['gt_path']
            data_path = self.dataset_params[part]['data_path']
            # import ipdb; ipdb.set_trace()
            gt = self.gt_loader.read(self.dataset_name, gt_path, data_path)
            gt_dict[part] = gt
        # pass
        return gt_dict
    
    def load_results(self, result_file):
        '''
        results' format:
        {
          'dataset': the name of dataset
          'psnr': the psnr of each testing videos,  # will be deprecated in the future, only keep the 'score' key
          'flow': [], 
          'names': [], 
          'diff_mask': [], 
          'score': the score of each testing videos
          'num_videos': the number of the videos
        }
        '''
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        
        dataset_name = results['dataset']
        num_videos = results['num_videos']

        score_records = list()
        # psnr_records = list()  # need to change, image, if you have another key, you need to change the code here. Not very elegant 
        # if self.dataset_params.smooth.guassian:
        #     for sigma in self.dataset_params.smooth.guassian_sigma:
        #         score_records.append(results[f'score_smooth_{sigma}']) # need to improve, at present, the guassian process is in the resluts is in the file, 
        #                                                                # and facing the same problem which if you add new key, you will change the code here
        #         # if len(results['psnr']) == 0:
        #         #     psnr_records = [results[f'psnr_smooth_{self.dataset_params.smooth.guassian_sigma[0]}']]
        #         # else:
        #         #     psnr_records.append(results[f'psnr_smooth_{sigma}'])
        # else:
        #     score_records.append(results['score'])
            # psnr_records.append(results['psnr'])
        score_records.append(results['score'])

        assert dataset_name == self.dataset_name, f'The dataset are not match, Result:{dataset_name}, cfg:{self.dataset_name}'

        # return psnr_records, score_records, num_videos
        return score_records, num_videos
        

    def eval_method(self, result, gt, verbose):
        import ipdb; ipdb.set_trace()
        temp_result = result[0]
        assert len(temp_result) == len(gt)
        temp_video_num = len(gt)
        for i in range(temp_video_num):
            # fpr, tpr, thresholds = metrics.roc_curve(gt, result, pos_label=self.pos_label)
            fpr, tpr, thresholds = metrics.roc_curve(gt[i], temp_result[i], pos_label=self.pos_label)
            auc = metrics.auc(fpr, tpr)
            results = RecordResult(fpr, tpr, thresholds, auc, self.dataset_name, self._result_name, verbose)

        return results

    def compute(self, result_file_dict):
        """Compute the metrics.
        Load the results stored in the file, compute the results and return the optimal result.
        Args:
            result_file_dict: The dictionary to store the results' files
            For example:
            {'train':{'description1':sigma0_result_file, 'description2':sigma1_result_file, .....}, 
            'val':{'description1':sigma0_result_file, 'description2':sigma1_result_file}
            }
        """
        for part in self.parts:
            #=====================Need to change, temporal=========================
            if part == 'train':
                continue
            #=======================================================================
            gt = self.gt_dict[part]
            result_file = result_file_dict[part]
            # import ipdb; ipdb.set_trace()
            for key, item in result_file.items():
                self._result_name = item
                # score_records, num_videos = self.load_results(result_file)
                score_records, num_videos = self.load_results(item)
                assert num_videos == len(gt), f'the number of saved videos does not match the ground truth, {num_videos} != {len(gt)}'
                temp_result = self.eval_method(score_records, gt, str(key))
                if temp_result > self.optimal_resulst:
                    self.optimal_resulst = temp_result
        
        return self.optimal_resulst
