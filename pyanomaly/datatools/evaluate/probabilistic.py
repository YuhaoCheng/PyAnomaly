import torch
import torch.nn.functional as F
import numpy as np
class AnomalyProbabilistic(object):
    def __init__(self,score_type='normal', ignore_frames=1):
        self.score_list = []
        self.score_type = score_type
        self.ignore_frames = ignore_frames
    def update(self, score):
        self.score_list.append(score)
    
    def output(self):
        max_score = [max(self.score_list)]
        self.score_list = max_score * self.ignore_frames + self.score_list 
        tensor = torch.Tensor(self.score_list)
        probability = F.softmax(tensor)
        return probability
    
    def __call__(self, confidence=0.5):
        prob = self.output()
        one = torch.ones(1)
        zero = torch.zeros(1)
        label = torch.where(torch.gt(prob,confidence), prob, zero)
        label = torch.where(torch.eq(label, zero), label, one)
        return label

