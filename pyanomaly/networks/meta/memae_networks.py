"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import torch.nn as nn
import math
# from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

import torchsnooper
from ..model_registry import META_ARCH_REGISTRY

__all__ = ['AutoEncoderCov3DMem']

class MemoryModule3D(nn.Module):
    def __init__(self, mem_dim, fea_dim, hard_shrink=True, lam=1.0):
        super(MemoryModule3D, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.hard_shrink = hard_shrink
        if hard_shrink:
            self.shrink_thres = lam / self.mem_dim
        self.memory = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.cos_similarity = nn.CosineSimilarity(dim=2, )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.memory)
    
    def hard_shrink_relu(self, input, lambd=0, epsilon=1e-15):
        output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output
    
    # @torchsnooper.snoop()
    def forward(self, z):
        with torch.autograd.set_detect_anomaly(True):
            N, C, D, H, W = z.size() # C=256
            z = z.permute(0, 2, 3, 4, 1) 
            z = z.reshape(-1, C) # [N*D*H*W, C]
            ex_mem = self.memory.unsqueeze(0).repeat(z.shape[0], 1, 1) # the shape of memory is to be [N*D*H*W, M, C]
            ex_z = z.unsqueeze(1).repeat(1,self.mem_dim, 1) # ex_z is to be [N*D*H*W, M, C]
            w_logit = self.cos_similarity(ex_z, ex_mem)
            w = F.softmax(w_logit, dim=1)
            if self.hard_shrink:
                w_hat = self.hard_shrink_relu(w, lambd=self.shrink_thres)
            else:
                w_hat = F.relu(w)
            
            w_hat = F.normalize(w_hat, p=1, dim=0)
            mem_trans = self.memory.permute(1,0)
            z_hat = F.linear(w_hat, mem_trans)
            z_output = z_hat.reshape(N,D,H,W,C).permute(0,4,1,2,3)
            return z_output, w_hat
        

@META_ARCH_REGISTRY.register()
class AutoEncoderCov3DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()
        self.chnum_in = chnum_in
        self.mem_dim = mem_dim
        channel_size = 32
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, channel_size*3, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel_size*3, channel_size*4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel_size*4, channel_size*8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel_size*8, channel_size*8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mem_rep = MemoryModule3D(mem_dim=self.mem_dim, fea_dim=channel_size*8, hard_shrink=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(channel_size*8, channel_size*8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(channel_size*8, channel_size*4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(channel_size*4, channel_size*3, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(channel_size*3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(channel_size*3, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1))
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.ConvTranspose3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def forward(self, x):
        f = self.encoder(x)
        z_hat, w_hat = self.mem_rep(f)
        # import ipdb; ipdb.set_trace()

        output = self.decoder(z_hat)
        return output, w_hat
