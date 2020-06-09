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
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # import ipdb; ipdb.set_trace()

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # import ipdb; ipdb.set_trace()
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
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
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class AutoEncoderCov3DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0):
        super(AutoEncoderCov3DMem, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        # feature_num = 128
        # feature_num_2 = 96
        # feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, 96, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=256, shrink_thres =shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(128, self.chnum_in, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1))
        )
    # @torchsnooper.snoop()
    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        print(f'max att:{att.max()}')
        # return {'output': output, 'att': att}
        # import ipdb; ipdb.set_trace()
        return output, att

# class AutoEncoderCov3D(nn.Module):
#     def __init__(self, chnum_in):
#         super(AutoEncoderCov3D, self).__init__()
#         self.chnum_in = chnum_in # input channel number is 1;
#         feature_num = 128
#         feature_num_2 = 96
#         feature_num_x2 = 256
#         self.encoder = nn.Sequential(
#             nn.Conv3d(self.chnum_in, feature_num_2, (3,3,3), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(feature_num_2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
#             nn.BatchNorm3d(feature_num),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(feature_num, feature_num_x2, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
#             nn.BatchNorm3d(feature_num_x2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(feature_num_x2, feature_num_x2, (3,3,3), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(feature_num_x2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(feature_num_x2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(feature_num_x2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
#             nn.BatchNorm3d(feature_num),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(feature_num, feature_num_2, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
#             nn.BatchNorm3d(feature_num_2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1))
#         )

#     def forward(self, x):
#         f = self.encoder(x)
#         out = self.decoder(f)
#         return out

def get_model_memae(cfg):
    model_dict = OrderedDict()
    model_dict['MemAE'] = AutoEncoderCov3DMem(cfg.DATASET.channel_num, 2000)
    return model_dict