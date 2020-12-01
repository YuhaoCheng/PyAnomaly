"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torchsnooper
from ..model_registry import META_ARCH_REGISTRY

__all__=['STAutoEncoderCov3D', 'get_model_stae']

@META_ARCH_REGISTRY.register()
class STAutoEncoderCov3D(nn.Module):
    def __init__(self, cfg):
        super(STAutoEncoderCov3D, self).__init__()
        self.chnum_in = cfg.DATASET.channel_num # input channel number is 1;
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, 32, (3,3,3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(32, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(48, 64, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 64, (3,3,3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        )
        self.decoder_reconstruction = nn.Sequential(
            nn.ConvTranspose3d(64, 48, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(48, 32, (3,3,3), stride=(2,2,2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, 32, (3,3,3), stride=(2,2,2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, self.chnum_in, (3,3,3), stride=(1,1,1), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.decoder_prediction = nn.Sequential(
            nn.ConvTranspose3d(64, 48, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(48, 32, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, 32, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, self.chnum_in, (3,3,3), stride=(1,1,1), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # @torchsnooper.snoop()
    def forward(self, x):
        f = self.encoder(x)
        out_reconstruction = self.decoder_reconstruction(f)
        out_prediction = self.decoder_prediction(f)
        # import ipdb; ipdb.set_trace()
        return out_reconstruction, out_prediction
