import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
from pyanomaly.networks.parts.base.commonness import DoubleConv, Down, Up, OutConv,  BasicConv2d

class ERM(nn.Module):
    def __init__(self, c_in, c_out, bilinear=False):
        super(ERM, self).__init__()
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
        self.output = nn.Conv2d(64, self.c_out, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.ConvTranspose2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
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
        return x

if __name__ == '__main__':
    temp = torch.randn([8, 3, 256, 256])
    model = ERM()
    output = model(temp)
    import ipdb; ipdb.set_trace()