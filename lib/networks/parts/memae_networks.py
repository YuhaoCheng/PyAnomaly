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

# class MemoryUnit(nn.Module):
#     def __init__(self, mem_dim, fea_dim, bias=True, shrink_thres=0.0005):
#         super(MemoryUnit, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
#         self.bias = bias
#         self.shrink_thres= shrink_thres
#         # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 3. / math.sqrt(self.weight.size(1))
#         # self.weight.data.uniform_(-stdv, stdv)
#         nn.init.kaiming_uniform_(self.weight.data)
#         if self.bias:
#             self.bias_data = Parameter(torch.Tensor(self.mem_dim))
#             self.bias_data = nn.init.kaiming_uniform(stdv)
#         else:
#             self.bias_data = None
#         # import ipdb; ipdb.set_trace()

#     def forward(self, input):
#         # att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxN) = TxN
#         # import ipdb; ipdb.set_trace()
#         att_weight = F.linear(input, self.weight, bias=self.bias_data)
#         att_weight = torch.div(att_weight, (input.norm() * self.weight.norm()))
#         att_weight = F.softmax(att_weight, dim=1)  # TxN
#         # ReLU based shrinkage, hard shrinkage for positive value
#         if(self.shrink_thres>0):
#             att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
#             # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
#             # normalize???
#             import ipdb; ipdb.set_trace()
#             att_weight = F.normalize(att_weight, p=1, dim=1)
#             # att_weight = F.softmax(att_weight, dim=1)
#             # att_weight = self.hard_sparse_shrink_opt(att_weight)
#         mem_trans = self.weight.permute(1, 0)  # Mem^T, NxC
#         # import ipdb; ipdb.set_trace()
#         output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxN) x (NxC) = TxC
#         return {'output': output, 'att': att_weight}  # output, att_weight

#     def extra_repr(self):
#         return 'mem_dim={}, fea_dim={}'.format(
#             self.mem_dim, self.fea_dim is not None
#         )


# # NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
# class MemModule(nn.Module):
#     def __init__(self, mem_dim, fea_dim, shrink_thres=0.0005, device='cuda'):
#         super(MemModule, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.shrink_thres = shrink_thres
#         self.linear = nn.Sequential(
#             nn.Linear(65536, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, self.fea_dim)
#         )
#         # self.Linear1 = nn.Linear(65536, 2048)
#         # self.Linear2 = nn.Linear(2048, 1024)
#         # self.Linear3 = nn.Linear(1024, self.fea_dim)
#         # self.upLinear1 = nn.Linear(self.fea_dim, 1024)
#         # self.upLinear2 = nn.Linear(1024, 2048)
#         # self.upLinear3 = nn.Linear(2048, 65536)
#         self.upLinear = nn.Sequential(
#             nn.Linear(self.fea_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, 65536)
#         )
#         self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

#     def forward(self, input):
#         x = input.reshape(input.shape[0], -1)
#         # x = self.Linear1(x)
#         # x = self.Linear2(x)
#         # x = self.Linear3(x)
#         x = self.linear(x)
#         y_and = self.memory(x)

#         y = y_and['output']
#         att = y_and['att']
#         # y = self.upLinear1(y)
#         # y = self.upLinear2(y)
#         # y = self.upLinear3(y)
#         y = self.upLinear(y)

#         y = y.reshape(input.shape)

#         return {'output': y, 'att': att}

# # relu based hard shrinkage function, only works for positive values
# def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
#     output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
#     return output

# class AutoEncoderCov3DMem(nn.Module):
#     def __init__(self, chnum_in, mem_dim, shrink_thres=0.0005):
#         super(AutoEncoderCov3DMem, self).__init__()
#         self.chnum_in = chnum_in
#         # feature_num = 128
#         # feature_num_2 = 96
#         # feature_num_x2 = 256
#         self.encoder = nn.Sequential(
#             nn.Conv3d(self.chnum_in, 96, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(96),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(96, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(128),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=256, shrink_thres =shrink_thres)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose3d(128, self.chnum_in, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1))
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
#             if isinstance(m, nn.ConvTranspose3d):
#                 nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
#     # @torchsnooper.snoop()
#     def forward(self, x):
#         f = self.encoder(x)
#         # import ipdb; ipdb.set_trace()
#         res_mem = self.mem_rep(f)
#         f = res_mem['output']
#         att = res_mem['att']
#         output = self.decoder(f)
#         print(f'max att:{att.max()}')
#         # return {'output': output, 'att': att}
#         # import ipdb; ipdb.set_trace()
#         return output, att
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

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
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
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=feature_num_x2, shrink_thres =shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1))
        )

    def forward(self, x):
        f = self.encoder(x)
        import ipdb; ipdb.set_trace()
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}

def get_model_memae(cfg):
    model_dict = OrderedDict()
    model_dict['MemAE'] = AutoEncoderCov3DMem(cfg.DATASET.channel_num, 2000)
    return model_dict