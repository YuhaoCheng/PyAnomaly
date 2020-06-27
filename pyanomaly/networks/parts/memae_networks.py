'''
The network of the 'Memoryizing Normality to detect anomaly: memory-augmented deep Autoencoder for Unsupervised anomaly detection(iccv2019)'
'''
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

import torchsnooper

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, hard_shrink=True, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.hard_shrink = hard_shrink
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
    
    def hard_shrink_relu(self, input, lambd=0, epsilon=1e-12):
        output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def forward(self, input):
        N, C, D, H, W = input.size()
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.hard_shrink:
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        else:
            att_weight = F.ReLU(att_weight)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}

# relu based hard shrinkage function, only works for positive values


class MemoryModule3D(nn.Module):
    def __init__(self, mem_dim, fea_dim, hard_shrink=True):
        super(MemoryModule3D, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.hard_shrink = hard_shrink
        if hard_shrink:
            self.shrink_thres = 1.0 / self.mem_dim
        self.memory = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.cos_similarity = nn.CosineSimilarity(dim=2, )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.memory)
    
    def hard_shrink_relu(self, input, lambd=0, epsilon=1e-15):
        output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output
    
    def forward(self, z):
        N, C, D, H, W = z.size() # C=256
        z = z.reshape(N, C, -1)  # z is to be [N, C, D*H*W]
        ex_mem = self.memory.unsqueeze(0).repeat(N, 1, 1) # the shape of memory is to be [N, M, C]
        ex_mem = ex_mem.unsqueeze(-1).repeat(1, 1, 1, z.shape[-1])
        ex_z = z.unsqueeze(1).repeat(1,self.mem_dim, 1, 1) # ex_z is to be [N, M, C, D*H*W]
        w_logit = self.cos_similarity(ex_z, ex_mem)
        w = F.softmax(w_logit, dim=1)
        if self.hard_shrink:
            w_hat = self.hard_shrink_relu(w, lambd=self.shrink_thres)
        else:
            w_hat = F.ReLU(w)
        
        # import ipdb; ipdb.set_trace()
        w_hat = F.normalize(w_hat, p=1, dim=1)
        z_hat = torch.matmul(w_hat.permute(0,2,1), self.memory)
        z_output = z_hat.permute(0,2,1).reshape(N,C,D,H,W)
        return z_output, w_hat
        


class AutoEncoderCov3DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()
        print('AutoEncoderCov3DMem')
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
        output = self.decoder(z_hat)
        # import ipdb; ipdb.set_trace()
        return output, w_hat

def get_model_memae(cfg):
    model_dict = OrderedDict()
    model_dict['MemAE'] = AutoEncoderCov3DMem(cfg.DATASET.channel_num, 2000)
    return model_dict