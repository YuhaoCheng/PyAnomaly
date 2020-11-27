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


def get_scores_labels(loss_file, cfg):
    '''
    base the psnr to get the scores of each videos
    '''
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, _, gt, _ = load_pickle_results(loss_file=loss_file, cfg=cfg)

    # the number of videos
    num_videos = len(psnr_records)

    # how many frames to ignore at first
    DECIDABLE_IDX = cfg.DATASET.decidable_idx

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        psnr_single_video = psnr_records[i]
        psnr_min = psnr_single_video.min()
        psnr_max = psnr_single_video.max()

        if cfg.DATASET.score_normalize:
            psnr_single_video -= psnr_min  # distances = (distance - min) / (max - min)
            psnr_single_video /= psnr_max
            # distance = 1 - distance

        scores = np.concatenate((scores[:], psnr_single_video[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:]), axis=0)
    return dataset, scores, labels


def precision_recall_auc(loss_file, cfg):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(loss_file=sub_loss_file, cfg=cfg)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        auc = metrics.auc(recall, precision)

        results = RecordResult(recall, precision, thresholds, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


def compute_eer(loss_file, cfg):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult(auc=np.inf)
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(loss_file=sub_loss_file, cfg=cfg)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        eer = cal_eer(fpr, tpr)

        results = RecordResult(fpr, tpr, thresholds, eer, dataset, sub_loss_file)

        if optimal_results > results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results

def average_psnr(loss_file, cfg):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    max_avg_psnr = -np.inf
    max_file = ''
    for file in loss_file_list:
        _, psnr_records, _, gt, _ = load_pickle_results(file, cfg)

        psnr_records = np.concatenate(psnr_records, axis=0)
        avg_psnr = np.mean(psnr_records)
        if max_avg_psnr < avg_psnr:
            max_avg_psnr = avg_psnr
            max_file = file
        print('{}, average psnr = {}'.format(file, avg_psnr))

    print('max average psnr file = {}, psnr = {}'.format(max_file, max_avg_psnr))


def calculate_psnr(loss_file, logger, cfg):
    optical_result = compute_auc_score(loss_file, logger, cfg)
    print('##### optimal result and model = {}'.format(optical_result))

    mean_psnr = []
    for file in os.listdir(loss_file):
        file = os.path.join(loss_file, file)
        dataset, psnr_records, _, gt, _ = load_pickle_results(file, cfg)

        psnr_records = np.concatenate(psnr_records, axis=0)
        gt = np.concatenate(gt, axis=0)

        mean_normal_psnr = np.mean(psnr_records[gt == 0])
        mean_abnormal_psnr = np.mean(psnr_records[gt == 1])
        mean = np.mean(psnr_records)
        print('mean normal psrn = {}, mean abnormal psrn = {}, mean = {}'.format(
            mean_normal_psnr,
            mean_abnormal_psnr,
            mean)
        )
        mean_psnr.append(mean)
    print('max mean psnr = {}'.format(np.max(mean_psnr)))


def compute_auc_psnr(loss_file, logger, cfg, score_type='normal'):
    '''
    For psnr, score_type is always 'normal', means that the higher PSNR, the higher normality 
    '''
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    DECIDABLE_IDX = cfg.DATASET.decidable_idx
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, score_records, gt, num_videos = load_pickle_results(loss_file=sub_loss_file, cfg=cfg)

        # the number of videos
        assert num_videos == len(score_records), 'The num of video is not equal'

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]
            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)
        '''
        Normalization is in the process of calculate the scores, instead of beforing getting the AUC
        '''
        # if cfg.DATASET.score_normalize:
        #     smin = scores.min()
        #     smax = scores.max()
        #     scores = scores - scores.min()  # scores = (scores - min) / (max - min)
        #     scores = scores / (smax - smin)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, thresholds, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    logger.info(f'##### optimal result and model = {optimal_results}')
    return optimal_results


def compute_auc_score(loss_file, logger, cfg, score_type='normal'):
    '''
    score_type:
        normal--> pos_label=0
        abnormal --> pos_label=1
        in dataset, 0 means normal, 1 means abnormal
    '''
    def get_results(score_record, sigma, pos_label):
        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)

        # video normalization
        for i in range(num_videos):
            score_one_video = score_record[i]
            l = len(score_one_video)
            score_one_video = np.clip(score_one_video, 0, None)
            scores = np.concatenate((scores, score_one_video[DECIDABLE_IDX:l-DECIDABLE_IDX_BACK]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:l-DECIDABLE_IDX_BACK]), axis=0)
        
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
        auc = metrics.auc(fpr, tpr)
        results = RecordResult(fpr, tpr, thresholds, auc, dataset, sub_loss_file, sigma)

        return results
        
    if score_type == 'normal':
        pos_label = 0
    elif score_type == 'abnormal':
        pos_label =1
    else:
        raise Exception('Error in score_type')
    
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    DECIDABLE_IDX = cfg.DATASET.decidable_idx
    DECIDABLE_IDX_BACK = cfg.DATASET.decidable_idx
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, score_records, gt, num_videos = load_pickle_results(loss_file=sub_loss_file, cfg=cfg)

        assert num_videos == len(score_records[0]), 'The num of video is not equal'

        # scores = np.array([], dtype=np.float32)
        # labels = np.array([], dtype=np.int8)
        
        # # video normalization
        # for i in range(num_videos):
        #     score_one_video = score_records[i]
        #     score_one_video = np.clip(score_one_video, 0, None)
        #     scores = np.concatenate((scores, score_one_video[DECIDABLE_IDX:]), axis=0)
        #     labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)
        # '''
        # Normalization is in the process of calculate the scores, instead of beforing getting the AUC
        # '''
        # # if cfg.DATASET.score_normalize:
        # #     smin = scores.min()
        # #     smax = scores.max()
        # #     scores = scores - smin  # scores = (scores - min) / (max - min)
        # #     scores = scores / (smax - smin)
        # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
        # auc = metrics.auc(fpr, tpr)

        # results = RecordResult(fpr, tpr, thresholds, auc, dataset, sub_loss_file)
        if cfg.DATASET.smooth.guassian:
            for index, sigma in enumerate(cfg.DATASET.smooth.guassian_sigma):
                score_record = score_records[index]
                results = get_results(score_record, sigma, pos_label)
                if optimal_results < results:
                    optimal_results = results
        else:
            results = get_results(score_records[0], 0, pos_label)
            if optimal_results < results:
                optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    logger.info(f'##### optimal result and model = {optimal_results}')
    return optimal_results

# use this dict to store the evaluate functions
eval_functions = {
    'compute_auc_psnr': compute_auc_psnr,
    'compute_auc_score': compute_auc_score,
    'compute_eer': compute_eer,
    'precision_recall_auc': precision_recall_auc,
    'calculate_psnr': calculate_psnr,
    # 'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr
}


# '''
# Functions for testing the evaluation functions
# '''
# def evaluate(eval_type, save_file, logger, cfg):
#     assert eval_type in eval_functions, f'there is no type of evaluation {eval_type}, please check {eval_functions.keys()}'
    
#     eval_func = eval_functions[eval_type]
#     optimal_results = eval_func(save_file, logger, cfg)
#     return optimal_results


@EVAL_METHOD_REGISTRY.register()
class AUCMetrics(AbstractEvalMethod):
    def __init__(self, cfg, is_training) -> None:
        super(AUCMetrics, self).__init__(cfg)
        self.gt_loader = GroundTruthLoader()
        # self.dataset_name = cfg.DATASET.name
        # self.gt_path = cfg.DATASET.gt_path
        self.optimal_resulst = RecordResult()
        self.decidable_idx = self.datasets_params.decidable_idx
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
            gt_path = self.dataset_params[part]['gt_path']
            data_path = self.dataset_params[part]['data_path']
            gt = self.gt_loader.read(self.dataset_name, gt_path, data_path)
            gt_dict[part] = gt
        # pass
        return gt_dict
    
    def load_results(self, result_file):
        '''
        results' format:
        {
          'dataset': the name of dataset
          'psnr': the psnr of each testing videos,
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
        psnr_records = list()  # need to change, image, if you have another key, you need to change the code here. Not very elegant 
        if self.dataset_params.smooth.guassian:
            for sigma in self.dataset_params.smooth.guassian_sigma:
                score_records.append(results[f'score_smooth_{sigma}']) # need to improve, at present, the guassian process is in the resluts is in the file, 
                                                                       # and facing the same problem which if you add new key, you will change the code here
                if len(results['psnr']) == 0:
                    psnr_records = [results[f'psnr_smooth_{self.dataset_params.smooth.guassian_sigma[0]}']]
                else:
                    psnr_records.append(results[f'psnr_smooth_{sigma}'])
        else:
            score_records.append(results['score'])
            psnr_records.append(results['psnr'])
        
        assert dataset_name == self.dataset_name, f'The dataset are not match, Result:{dataset_name}, cfg:{self.dataset_name}'

        return psnr_records, score_records, num_videos
        

    def eval_method(self, result, gt):
        fpr, tpr, thresholds = metrics.roc_curve(gt, result, pos_label=self.pos_label)
        auc = metrics.auc(fpr, tpr)
        results = RecordResult(fpr, tpr, thresholds, auc, self.dataset_name, self._result_name, 0, self.pos_label)

        pass
        return results

    def compute(self, result_file_dict):
        '''
        result_file_dict = {'train':......., 'val':..........}
        '''
        for part in self.parts:
            gt = self.gt_dict[part]
            result_file = result_file_dict[part]
            self._result_name = result_file
            psnr_records, score_records, num_videos = self.load_results(result_file)
            assert num_videos == len(gt), f'the number of saved videos does not match the ground truth, {num_videos} != {len(gt)}'
            temp_result = self.eval_method(score_records, gt)
            if temp_result > self.optimal_resulst:
                self.optimal_resulst = temp_result
        
        pass
        return self.optimal_resulst

@EVAL_METHOD_REGISTRY.register()
class ScoreAUC(AUCMetrics):
    def __init__(self, cfg, is_training) -> None:
        super(ScoreAUC, self).__init__(cfg, is_training)
        self.set_result_type_score

    def set_result_type_score(self):
        self.result_type = 'score'


@EVAL_METHOD_REGISTRY.register()
class PSNRAUC(AUCMetrics):
    def __init__(self, cfg, is_training) -> None:
        super(PSNRAUC, self).__init__(cfg, is_training)
        self.set_result_type_score

    def set_result_type_score(self):
        self.result_type = 'psnr'