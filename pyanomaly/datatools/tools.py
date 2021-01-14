"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

class RecordResult(object):
    def __init__(self, dataset=None, loss_file=None, metric_name='AUC', verbose=None):
        self.value = 0
        self.sum_value = 0
        self.avg_value = -np.inf
        self.count = 0
        self.metric_name = metric_name
        self.dataset = dataset
        self.loss_file = loss_file
        self.verbose = verbose

    def __lt__(self, other):
        return self.avg_value < other.avg_value

    def __gt__(self, other):
        return self.avg_value > other.avg_value

    def __str__(self):
        return f'dataset = {self.dataset}, loss file = {self.loss_file}, metric_name = {self.metric_name}, value = {self.avg_value:.3f}'
    
    def update(self, value):
        self.value = value
        self.sum_value += value
        self.count += 1
        self.avg_value = self.sum_value / self.count

def collect_fn(batch):
    """
    image_b, image_a, image, image_f, label, detection_result = batch
    """
    # max_detection = max(list(map(lambda x: len(x[5]), batch)))
    max_detection = max(list(map(lambda x: len(x), batch)))
    for i in range(len(batch)):
        batch[i] = list(batch[i]) # because the element in the batch is a tuple
        dummy = torch.zeros((1,5), dtype=batch[i][5].dtype)
        temp = batch[i][5]
        # make the detection to the same length in order to stack the
        while temp.size(0) < max_detection:
            temp = torch.cat((temp, dummy))
        batch[i][5] = temp
    
    return default_collate(batch)

def collect_fn_local(batch):
    """
    image_b, image_a, image, image_f, crop_objects = batch
    """
    max_detection = max(list(map(lambda x: len(x[4]), batch)))
    for i in range(len(batch)):
        batch[i] = list(batch[i]) # because the element in the batch is a tuple
        dummy = torch.zeros((1,128,64), dtype=batch[i][4][0].dtype)
        temp = batch[i][4]
        # make the detection to the same length in order to stack the
        while temp.size(0) < max_detection:
        # while len(temp) < max_detection:
            temp = torch.cat((temp, dummy))
            # temp.append(dummy)
        batch[i][4] = temp
    
    return default_collate(batch)

