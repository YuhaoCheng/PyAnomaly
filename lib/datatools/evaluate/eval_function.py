'''
The eval api of the anomaly detection.
We will store the results in these structure and use off-line style to compute the metric
results {
           'dataset': the name of dataset
           'psnr': the psnr of each testing videos,
        }
'''
import numpy as np
# import scipy.io as scio
import os
import argparse
import pickle
from sklearn import metrics
import json

from .utils import load_pickle_results
from .gtloader import GroundTruthLoader, RecordResult


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


def calculate_score(loss_file, logger, cfg):
    if not os.path.isdir(loss_file):
        loss_file_path = loss_file
    else:
        optical_result = compute_auc_score(loss_file, logger, cfg)
        loss_file_path = optical_result.loss_file
        print('##### optimal result and model = {}'.format(optical_result))
    dataset, psnr_records, _, gt, num_videos = load_pickle_results(loss_file=loss_file_path, cfg=cfg)

    # the number of videos
    # num_videos = len(psnr_records)
    DECIDABLE_IDX = cfg.DATASET.decidable_idx

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        distance = (distance - distance.min()) / (distance.max() - distance.min())

        scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

    mean_normal_scores = np.mean(scores[labels == 0])
    mean_abnormal_scores = np.mean(scores[labels == 1])
    print('mean normal scores = {}, mean abnormal scores = {}, '
          'delta = {}'.format(mean_normal_scores, mean_abnormal_scores, mean_normal_scores - mean_abnormal_scores))

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
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, score_records, gt, num_videos = load_pickle_results(loss_file=sub_loss_file, cfg=cfg)

        assert num_videos == len(score_records), 'The num of video is not equal'

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        
        # video normalization
        for i in range(num_videos):
            score_one_video = score_records[i]
            score_one_video = np.clip(score_one_video, 0, None)
            scores = np.concatenate((scores, score_one_video[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)
        '''
        Normalization is in the process of calculate the scores, instead of beforing getting the AUC
        '''
        # if cfg.DATASET.score_normalize:
        #     smin = scores.min()
        #     smax = scores.max()
        #     scores = scores - smin  # scores = (scores - min) / (max - min)
        #     scores = scores / (smax - smin)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, thresholds, auc, dataset, sub_loss_file)

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
    'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr
}


'''
Functions for testing the evaluation functions
'''
def evaluate(eval_type, save_file, logger, cfg):
    assert eval_type in eval_functions, f'there is no type of evaluation {eval_type}, please check {eval_functions.keys()}'
    
    eval_func = eval_functions[eval_type]
    optimal_results = eval_func(save_file, logger, cfg)
    return optimal_results


