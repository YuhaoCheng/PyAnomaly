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

class AMCGenerator(nn.Module):
    def __init__(self, c_in, opticalflow_channel_num=2, image_channel_num=3, dropout_prob=0, bilinear=True):
        super(AMCGenerator, self).__init__()
        self.c_in = c_in
        self.bilinear = bilinear

        # common encoder
        self.inception = Inception(c_in, 64)
        self.h1 = Conv2dLeakly(64, 64, bn_flag=False, kernel_size=3, stride=1, padding=1)
        self.h2 = Conv2dLeakly(64, 128, bn_flag=True, kernel_size=4, stride=2, padding=1)
        self.h3 = Conv2dLeakly(128, 256, bn_flag=True, kernel_size=4, stride=2, padding=1)
        self.h4 = Conv2dLeakly(256, 512, bn_flag=True, kernel_size=4, stride=2, padding=1)
        self.h5 = Conv2dLeakly(512, 512, bn_flag=True, kernel_size=4, stride=2, padding=1)
        # unet for optical flow, decoder
        self.h4fl = ConcatDeconv2d(512, 256, dropout_prob)
        self.h3fl = ConcatDeconv2d(768, 256, dropout_prob)
        self.h2fl = ConcatDeconv2d(512, 128, dropout_prob)
        self.h1fl = ConcatDeconv2d(256, 64, dropout_prob)
        self.conv_fl = BasicConv2d(128, opticalflow_channel_num,kernel_size=3, stride=1, padding=1)
        
        # decoder for frame
        self.h4fr = Deconv2d(512, 256, dropout_prob)
        self.h3fr = Deconv2d(256, 256, dropout_prob)
        self.h2fr = Deconv2d(256, 128, dropout_prob)
        self.h1fr = Deconv2d(128, 64, dropout_prob)
        self.conv_fr = BasicConv2d(64, image_channel_num, kernel_size=3, stride=1, padding=1)

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

        self._init_weights()
    
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
        x_sigmod = F.sigmoid(x)
        
        return x_sigmod

def get_model_amc(cfg):
    from collections import namedtuple
    temp = namedtuple('Args', ['fp16', 'rgb_max'])
    args = temp(False, 1.0)
    generator_model = AMCGenerator(c_in=3, opticalflow_channel_num=2, image_channel_num=cfg.DATASET.channel_num, dropout_prob=0.7)
    discriminator_model = AMCDiscriminiator(c_in=5, filters=64)
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