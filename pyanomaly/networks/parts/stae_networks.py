import torch
import torch.nn as nn
from collections import OrderedDict
class STAutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super(STAutoEncoderCov3D, self).__init__()
        self.chnum_in = chnum_in # input channel number is 1;
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
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        )
        self.decoder_reconstruction = nn.Sequential(
            nn.ConvTranspose3d(64, 48, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(48, 32, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, 32, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, self.chnum_in, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
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
            nn.ConvTranspose3d(32, self.chnum_in, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.encoder(x)
        out_reconstruction = self.decoder_reconstruction(f)
        out_prediction = self.decoder_prediction(f)
        return out_reconstruction, out_prediction

def get_model_stae(cfg):
    model_dict = OrderedDict()
    stae = STAutoEncoderCov3D(cfg.DATASET.channel_num)
    model_dict['STAE'] = stae
    return model_dict