import torch
import torch.nn as nn
import torchsnooper
from collections import OrderedDict, namedtuple
from pyanomaly.networks.parts.pcn_parts.pcm import PCM
from pyanomaly.networks.parts.pcn_parts.erm import ERM
from pyanomaly.networks.parts.base.commonness import PixelDiscriminator, NLayerDiscriminator
from pyanomaly.networks.parts.base.commonness import DoubleConv, Down, Up, OutConv,  BasicConv2d



class UNet(nn.Module):
    def __init__(self, c_in, c_out, bilinear=False, output=False):
        super(UNet, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear
        self.output = output
        self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        self.up1 = Up(768, 512, 256, self.bilinear)
        self.up2 = Up(384,256,128, self.bilinear)
        self.up3 = Up(192,128,64, self.bilinear)
        if output:
            self.output_conv = nn.Conv2d(64, self.c_out, 3, padding=1)

    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        if self.output:
            x = self.output(x)
        return x



class AnoPcn(nn.Module):
    def __init__(self, cfg):
        super(AnoPcn, self).__init__()
        self.pcm = PCM()
        self.erm = ERM(3,3)
    
    # @torchsnooper.snoop()
    def forward(self, x, target):
        # input是video_clip
        prediction = self.pcm(x)
        pe = torch.sub(target,prediction) # pe = prediction error
        re = self.erm(pe) # re = recontruction error
        result = torch.add(prediction, re)
        
        return prediction, result

def get_model_anopcn(cfg):
    if cfg.ARGUMENT.train.normal.use:
        rgb_max = 1.0
    else:
        rgb_max = 255.0
    if cfg.MODEL.flownet == 'flownet2':
        from collections import namedtuple
        from pyanomaly.networks.auxiliary.flownet2.models import FlowNet2
        temp = namedtuple('Args', ['fp16', 'rgb_max'])
        args = temp(False, rgb_max)
        flow_model = FlowNet2(args)
        flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path)['state_dict'])
    elif cfg.MODEL.flownet == 'liteflownet':
        from pyanomaly.networks.auxiliary.liteflownet.models import LiteFlowNet
        flow_model = LiteFlowNet()
        flow_model.load_state_dict({strKey.replace('module', 'net'): weight for strKey, weight in torch.load(cfg.MODEL.flow_model_path).items()})
    else:
        raise Exception('Not support optical flow methods')

    generator_model = AnoPcn(cfg)
    discriminator_model = PixelDiscriminator(3, cfg.MODEL.discriminator_channels, use_norm=False)
    # discriminator_model = NLayerDiscriminator(3)
    
    model_dict = OrderedDict()
    model_dict['Generator'] = generator_model
    model_dict['Discriminator'] = discriminator_model
    model_dict['FlowNet'] = flow_model
    
    return model_dict