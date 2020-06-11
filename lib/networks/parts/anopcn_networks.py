import torch
import torch.nn as nn
import torchsnooper
from collections import OrderedDict, namedtuple
from lib.networks.parts.pcn_parts.pcm import PCM
from lib.networks.parts.pcn_parts.erm import ERM
from lib.networks.auxiliary.flownet2.models import FlowNet2
from lib.networks.parts.amc_networks import AMCDiscriminiator
from lib.networks.parts.base.commonness import PixelDiscriminator

class AnoPcn(nn.Module):
    def __init__(self, cfg):
        super(AnoPcn, self).__init__()
        self.pcm = PCM()
        self.erm = ERM()
    def _init_weights(self):
        pass
    # @torchsnooper.snoop()
    def forward(self, x, target):
        # input是video_clip
        prediction = self.pcm(x)
        pe = torch.sub(target,prediction) # pe = prediction error
        re = self.erm(pe) # re = recontruction error
        result = torch.add(prediction, re)
        
        return result

def get_model_anopcn(cfg):
    print(f'The config file is{cfg}')
    temp = namedtuple('Args', ['fp16', 'rgb_max'])
    args = temp(False, 255.)
    generator_model = AnoPcn(cfg)
    # discriminator_model = AMCDiscriminiator(c_in=6, filters=64)
    discriminator_model = PixelDiscriminator(3, cfg.MODEL.discriminator_channels, use_norm=False)
    flow_model = FlowNet2(args)
    model_dict = OrderedDict()
    model_dict['Generator'] = generator_model
    model_dict['Discriminator'] = discriminator_model
    model_dict['FlowNet'] = flow_model
    return model_dict