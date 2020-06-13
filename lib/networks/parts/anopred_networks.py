# import sys
# sys.path.append('../../')
import torch
import torch.nn as nn
from collections import OrderedDict
import torchsnooper
# from lib.networks.parts.unet.unet_parts import *
from lib.networks.parts.base.commonness import DoubleConv, Down, Up, OutConv, PixelDiscriminator, BasicConv2d
from lib.networks.auxiliary.flownet2.models import FlowNet2

class GeneratorUnet(nn.Module):
    def __init__(self, c_in, c_out, bilinear=False):
        super(GeneratorUnet, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear

        # self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128,256)
        self.inter = DoubleConv(256, 512)
        self.up1 = Up(768, 512, 256, self.bilinear)
        self.up2 = Up(384,256,128, self.bilinear)
        self.up3 = Up(180,128,64, self.bilinear)
        self.output = BasicConv2d(64, self.c_out, kernel_size=3, padding=1)
    
    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.inter(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.output(x)
        # return x
        return torch.tanh(x)

def get_model_ano_pred(cfg):
    from collections import namedtuple
    temp = namedtuple('Args', ['fp16', 'rgb_max'])
    args = temp(False, 1.0)
    flow_model = FlowNet2(args)
    if cfg.MODEL.name == 'anopred':
        flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path)['state_dict'])
    for n,p in flow_model.named_parameters():
        p.requires_grad = False
    generator_model = GeneratorUnet(12,3) # 4*3 =12
    discriminator_model = PixelDiscriminator(3, cfg.MODEL.discriminator_channels, use_norm=False)
    model_dict = OrderedDict()
    # model_dict = {'Generator':generator_model,'Discriminator':discriminator_model, 'FlowNet':flow_model}
    model_dict['Generator'] = generator_model
    model_dict['Discriminator'] = discriminator_model
    model_dict['FlowNet'] = flow_model

    return model_dict


if __name__ == '__main__':
    model = GeneratorUnet(3,3)
    temp = torch.rand((2,3,256,256))
    output = model(temp)