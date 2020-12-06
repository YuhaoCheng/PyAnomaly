"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torchsnooper
from ..model_registry import META_ARCH_REGISTRY
from pyanomaly.networks.meta.base.commonness import (
    PixelDiscriminator, 
    DoubleConv, 
    Down, 
    Up, 
    OutConv,  
    BasicConv2d
)

__all__ = ['AnoPredGeneratorUnet']


@META_ARCH_REGISTRY.register()
class AnoPredGeneratorUnet(nn.Module):
    def __init__(self, c_in, c_out, bilinear=False):
        super(GeneratorUnet, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear

        self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        # self.inter = DoubleConv(256, 512)
        self.up1 = Up(768, 512, 256, self.bilinear)
        self.up2 = Up(384,256,128, self.bilinear)
        self.up3 = Up(192,128,64, self.bilinear)
        self.output = BasicConv2d(64, self.c_out, kernel_size=3, padding=1)
    
    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4 = self.inter(x3)
        # import ipdb; ipdb.set_trace()
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.output(x)
        # return x
        return torch.tanh(x)
