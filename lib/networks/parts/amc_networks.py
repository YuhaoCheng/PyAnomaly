"""
reference to the implementation in the Pytorch
"""
import sys
# sys.path.append('/export/home/chengyh/Anomaly_DA')
import torch
import torch.nn as nn
import torchsnooper
import torch.nn.functional as F
from collections import OrderedDict
from lib.networks.auxiliary.flownet2.models import FlowNet2
from lib.networks.parts.base.commonness import Conv2dLeakly, ConcatDeconv2d, Deconv2d, BasicConv2d, Inception
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         # return F.relu(x, inplace=True)
#         return x 

# class Conv2dLeakly(nn.Module):
#     def __init__(self, c_in, c_out, bn_flag=True, **kwargs):
#         super(Conv2dLeakly, self).__init__()
#         self.bn_flag = bn_flag
#         self.conv = nn.Conv2d(c_in, c_out, **kwargs)
        
#         self.bn = nn.BatchNorm2d(c_out)
#     # @torchsnooper.snoop()
#     def forward(self,x):
#         x = self.conv(x)
#         if self.bn_flag:
#             x = self.bn(x)
#         return F.leaky_relu_(x)

# class ConcatDeconv2d(nn.Module):
#     def __init__(self, c_in, c_out, dropout_prob):
#         '''
#         use the conv_tranpose to enlarge the feature into two times
#         '''
#         super(ConcatDeconv2d, self).__init__()
#         self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
#         self.bn = nn.BatchNorm2d(c_out)
#         self.dropout = nn.Dropout2d(p=dropout_prob)
#         self.reduce_channel = nn.Conv2d(c_out*2, c_out, kernel_size=1)
#     def forward(self, x1, x2):
#         x1 = self.conv_transpose(x1)
#         x1 = self.dropout(x1)
#         x1 = F.relu_(x1)
#         x2 = torch.cat([x1,x2], dim=1)
#         x2 = self.reduce_channel(x2)
#         # import ipdb; ipdb.set_trace()
#         return x2

# class Deconv2d(nn.Module):
#     def __init__(self, c_in, c_out, dropout_prob):
#         '''
#         use the conv_tranpose to enlarge the feature into two times
#         '''
#         super(Deconv2d, self).__init__()
#         self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=(4,4), stride=2, padding=1)
#         self.bn = nn.BatchNorm2d(c_out)
#         self.dropout = nn.Dropout2d(p=dropout_prob)
    
#     def forward(self,x):
#         x = self.conv_transpose(x)
#         x = self.dropout(x)

#         return F.relu_(x)

class AMCGenerator(nn.Module):
    def __init__(self, c_in, c_out, dropout_prob=0,bilinear=True):
        super(AMCGenerator, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear

        # common encoder
        self.inception = Inception(c_in, 64)
        self.h1 = Conv2dLeakly(64, 64, bias=False, kernel_size=3, stride=1, padding=1)
        self.h2 = Conv2dLeakly(64, 128, bias=True, kernel_size=4, stride=2, padding=1)
        self.h3 = Conv2dLeakly(128, 256, bias=True, kernel_size=4, stride=2, padding=1)
        self.h4 = Conv2dLeakly(256, 512, bias=True, kernel_size=4, stride=2, padding=1)
        self.h5 = Conv2dLeakly(512, 512, bias=True, kernel_size=4, stride=2, padding=1)
        # unet for optical flow, decoder
        self.h4fl = ConcatDeconv2d(512, 512, dropout_prob)
        self.h3fl = ConcatDeconv2d(512, 256, dropout_prob)
        self.h2fl = ConcatDeconv2d(256, 128, dropout_prob)
        self.h1fl = ConcatDeconv2d(128, 64, dropout_prob)
        self.conv_fl = BasicConv2d(64,3,kernel_size=3, stride=1, padding=1)
        # decoder for frame
        self.h4fr = Deconv2d(512, 512, dropout_prob)
        self.h3fr = Deconv2d(512, 256, dropout_prob)
        self.h2fr = Deconv2d(256, 128, dropout_prob)
        self.h1fr = Deconv2d(128, 64, dropout_prob)
        self.conv_fr = BasicConv2d(64,3, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out') 

    # @torchsnooper.snoop()
    def forward(self, x):
        # common encoder
        x = self.inception(x)
        x1 = self.h1(x)
        x2 = self.h2(x1)
        x3 = self.h3(x2)
        x4 = self.h4(x3)
        x5 = self.h5(x4)
        
        # unet for optical flow
        fl4 = self.h4fl(x5, x4)
        fl3 = self.h3fl(fl4,x3)
        fl2 = self.h2fl(fl3, x2)
        fl1 = self.h1fl(fl2, x1)
        out_flow = self.conv_fl(fl1)

        # for frame
        fr4 = self.h4fr(x5)
        fr3 = self.h3fr(fr4)
        fr2 = self.h2fr(fr3)
        fr1 = self.h1fr(fr2)
        out_frame = self.conv_fr(fr1)

        return out_flow, out_frame

class AMCDiscriminiator(nn.Module):
    def __init__(self, c_in, filters):
        super(AMCDiscriminiator, self).__init__()
        self.conv1 = nn.Conv2d(c_in, filters, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(filters, filters*2, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(filters*2)
        self.conv3 = nn.Conv2d(filters*2, filters*4, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(filters*4)
        self.conv4 = nn.Conv2d(filters*4, filters*8, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(filters*8)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu_(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu_(x)
        x = self.conv4(x)
        x = self.bn4(x)
        # x_sigmod = F.sigmoid(x)
        
        # return x_sigmod, x
        return x

# class Inception(nn.Module):
#     def __init__(self, c_in, c_out, max_filter_size=7):
#         super(Inception, self).__init__()
#         assert max_filter_size % 2 == 1 and max_filter_size < 8
#         self.n_branch = (max_filter_size + 1 ) // 2
#         assert c_out % self.n_branch == 0
#         nf_branch = c_out // self.n_branch
#         # 1x1 
#         self.branch1 = BasicConv2d(in_channels=c_in, out_channels=nf_branch, kernel_size=1)
#         # 3x3
#         self.branch2 = Inception3x3(in_channels=c_in, out_channels=nf_branch)
#         # 5x5
#         self.branch3 = Inception5x5(in_channels=c_in, out_channels=nf_branch)
#         # 7x7
#         self.branch4 = Inception7x7(in_channels=c_in, out_channels=nf_branch)

#     def forward(self, x):
#         out1 = self.branch1(x)
#         if self.n_branch == 1:
#             return out1
#         out2 = self.branch2(x)
#         if self.n_branch == 2:
#             return torch.cat([out1, out2], dim=1)
#         out3 = self.branch3(x)
#         if self.n_branch == 3:
#             return torch.cat([out1, out2, out3], dim=1)
#         out4 = self.branch4(x)
#         if self.n_branch == 4:
#             return torch.cat([out1, out2, out3, out4], dim=1)

#         # return x
        

# class Inception3x3(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Inception3x3, self).__init__()
#         self.s3_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
#         self.s3_1n = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s3_n1 = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
#     def forward(self,x):
#         x = self.s3_11(x)
#         x = self.s3_1n(x)
#         x = self.s3_n1(x)

#         return x

# class Inception5x5(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Inception5x5, self).__init__()
#         self.s5_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
#         self.s5_1n_a = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s5_n1_a = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
#         self.s5_1n_b = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s5_n1_b = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
#     def forward(self,x):
#         x = self.s5_11(x)
#         x = self.s5_1n_a(x)
#         x = self.s5_n1_a(x)
#         x = self.s5_1n_b(x)
#         x = self.s5_n1_b(x)

#         return x
# class Inception7x7(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Inception7x7, self).__init__()
#         self.s7_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
#         self.s7_1n_a = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s7_n1_a = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
#         self.s7_1n_b = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s7_n1_b = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
#         self.s7_1n_c = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
#         self.s7_n1_c = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
#     def forward(self, x):
#         x = self.s7_11(x)
#         x = self.s7_1n_a(x)
#         x = self.s7_n1_a(x)
#         x = self.s7_1n_b(x)
#         x = self.s7_n1_b(x)
#         x = self.s7_1n_c(x)
#         x = self.s7_n1_c(x)

#         return x

def get_model_amc(cfg):
    from collections import namedtuple
    temp = namedtuple('Args', ['fp16', 'rgb_max'])
    args = temp(False, 1.0)
    generator_model = AMCGenerator(c_in=3, c_out=3, dropout_prob=0.3)
    discriminator_model = AMCDiscriminiator(c_in=6, filters=64)
    flow_model = FlowNet2(args)
    flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path)['state_dict'])
    model_dict = OrderedDict()
    model_dict['Generator'] = generator_model
    model_dict['Discriminator'] = discriminator_model
    model_dict['FlowNet'] = flow_model
    return model_dict


if __name__ == '__main__':
    # model = GeneratorUnet(3,3)
    model = AMCDiscriminiator(c_in=6, filters=64)
    temp = torch.rand((8,6,128,192))
    output = model(temp)
    import ipdb; ipdb.set_trace()