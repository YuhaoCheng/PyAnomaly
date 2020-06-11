# import sys
# sys.path.append('../../')
import torch
import torch.nn as nn
from collections import OrderedDict
import torchsnooper
# from lib.networks.parts.unet.unet_parts import *
from lib.networks.parts.base.commonness import DoubleConv, Down, Up, OutConv, PixelDiscriminator
from lib.networks.auxiliary.flownet2.models import FlowNet2

class GeneratorUnet(nn.Module):
    def __init__(self, c_in, c_out, bilinear=True):
        super(GeneratorUnet, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear

        self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256,256)
        self.down3 = Down(256,512)
        self.up1 = Up(512, 256, self.bilinear)
        # self.up2 = Up(384,128, self.bilinear)
        self.up2 = Up(256,128, self.bilinear)
        # self.up3 = Up(192,64, self.bilinear)
        self.up3 = Up(128,64, self.bilinear)
        self.output = OutConv(64, self.c_out)
    
    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.output(x)
        # return x
        return torch.sigmoid(x)

# # [128,256,512,512]
# class TestPixelDiscriminator(nn.Module):
#     """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

#     def __init__(self, input_nc, num_filters, use_norm=False,norm_layer=nn.BatchNorm2d):
#         """Construct a 1x1 PatchGAN discriminator

#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#         """
#         '''
#         different from ano_pred with norm here
#         '''


#         super(PixelDiscriminator, self).__init__()
#         if use_norm:
#             if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#                 use_bias = norm_layer.func != nn.InstanceNorm2d
#             else:
#                 use_bias = norm_layer != nn.InstanceNorm2d
#         else:
#             use_bias=True

#         self.conv1 = nn.Conv2d(input_nc, num_filters[0],kernel_size=4,padding=2,stride=2)
#         self.lkr1 = nn.LeakyReLU(0.1)
#         self.conv2 = nn.Conv2d(num_filters[0], num_filters[1],kernel_size=4,padding=2,stride=2,bias=use_bias)
#         self.lkr2 = nn.LeakyReLU(0.1)
#         self.conv3 = nn.Conv2d(num_filters[1], num_filters[2],kernel_size=4,padding=2,stride=2,bias=use_bias)
#         self.lkr3 = nn.LeakyReLU(0.1)
#         self.conv4 = nn.Conv2d(num_filters[2], num_filters[3],kernel_size=4,padding=2,stride=2,bias=use_bias)
#         self.lkr4 = nn.LeakyReLU(0.1)
#         self.conv5 = nn.Conv2d(num_filters[3], 1,kernel_size=4,padding=2,stride=2,bias=use_bias)
#         # self.net=[]
#         # self.net.append(nn.Conv2d(input_nc,num_filters[0],kernel_size=4,padding=2,stride=2))
#         # self.net.append(nn.LeakyReLU(0.1))
#         # if use_norm:
#         #     for i in range(1,len(num_filters)-1):
#         #         self.net.extend([nn.Conv2d(num_filters[i-1],num_filters[i],4,2,2,bias=use_bias),
#         #                          nn.LeakyReLU(0.1),
#         #                          norm_layer(num_filters[i])])
#         # else :
#         #     for i in range(1,len(num_filters)-1):
#         #         self.net.extend([nn.Conv2d(num_filters[i-1],num_filters[i],4,2,2,bias=use_bias),
#         #                          nn.LeakyReLU(0.1)])
#         # self.net.append(nn.Conv2d(num_filters[-1],1,4,1,2))
#         # # self.net = [
#         # #     nn.Conv2d(input_nc, num_filters[0], kernel_size=1, stride=1, padding=0),
#         # #     nn.LeakyReLU(0.2, True),
#         # #     nn.Conv2d(num_filters[0], num_filters[1], kernel_size=1, stride=1, padding=0, bias=use_bias),
#         # #     norm_layer(num_filters[1]),
#         # #     nn.LeakyReLU(0.2, True),
#         # #     nn.Conv2d(num_filters[1], 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

#         # self.net = nn.Sequential(*self.net)

#     def forward(self, input):
#         """Standard forward."""
#         with torch.autograd.set_detect_anomaly(True):
#             x1 = self.conv1(input)
#             x2 = self.lkr1(x1)
#             x3 = self.conv2(x2)
#             x4 = self.lkr2(x3)
#             x5 = self.conv3(x4)
#             x6 = self.lkr3(x5)
#             x7 = self.conv4(x5)
#             x8 = self.lkr4(x7)
#             output = self.conv5(x8)
#         return output

def get_model_ano_pred(cfg):
    from collections import namedtuple
    temp = namedtuple('Args', ['fp16', 'rgb_max'])
    args = temp(False, 1.0)
    # flow_model = Network()
    # if cfg.MODEL.name == 'ano_pred':
    #     flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path))
    # else:
    #     raise Exception('Not correct model name in ano_pred')
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