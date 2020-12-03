"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import torch.nn as nn
from ..loss_registry import LOSS_REGISTRY

__all__ = ['MemLoss', 'get_expand_loss']

@LOSS_REGISTRY.register()
class MemLoss(nn.Module):
    def __init__(self):
        super(MemLoss, self).__init__()
        # self.l_num =2
    def forward(self, att_weights):
        att_weights = att_weights + (att_weights == 0).float() * 1.0
        x = torch.mean(-att_weights * att_weights.log())
        # import ipdb; ipdb.set_trace()
        # print(f'the memae loss is{x}')
        return x

LOSSDICT ={
    'mem_loss': MemLoss().cuda()
}
EXPAND_LOSS = LOSSDICT.keys()

def get_expand_loss(loss_name, cfg):
    assert loss_name in LOSSDICT.keys(), f'The loss name: {loss_name} is not support'
    print(f'Using the loss:{loss_name}')
    loss_function = LOSSDICT[loss_name]
    return loss_function