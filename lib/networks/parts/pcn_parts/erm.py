import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

class Conv2dLeakly(nn.Module):
    def __init__(self, c_in, c_out, bn_flag=True, **kwargs):
        super(Conv2dLeakly, self).__init__()
        self.bn_flag = bn_flag
        self.conv = nn.Conv2d(c_in, c_out, **kwargs)
        
        self.bn = nn.BatchNorm2d(c_out)
    # @torchsnooper.snoop()
    def forward(self,x):
        x = self.conv(x)
        if self.bn_flag:
            x = self.bn(x)
        return F.leaky_relu_(x)

class BasicConv2d(nn.Module):
    def __init__(self, c_in, c_out, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(c_out, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # return F.relu(x, inplace=True)
        return x

class ConcatDeconv2d(nn.Module):
    def __init__(self, c_in, c_out, dropout_prob=0):
        '''
        use the conv_tranpose to enlarge the feature into two times
        '''
        super(ConcatDeconv2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.reduce_channel = nn.Conv2d(c_out*2, c_out, kernel_size=1)
        
    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        x1 = self.dropout(x1)
        x1 = F.relu_(x1)
        x2 = torch.cat([x1,x2], dim=1)
        x2 = self.reduce_channel(x2)
        # import ipdb; ipdb.set_trace()
        return x2

class ERM(nn.Module):
    def __init__(self):
        super(ERM, self).__init__()
        # downsample
        self.h1 = Conv2dLeakly(c_in=3,   c_out=64, kernel_size=3, stride=1, padding=1)
        self.h2 = Conv2dLeakly(c_in=64,  c_out=128, kernel_size=3, stride=2, padding=1)
        self.h3 = Conv2dLeakly(c_in=128, c_out=256, kernel_size=3, stride=2, padding=1)
        self.h4 = Conv2dLeakly(c_in=256, c_out=512, kernel_size=3, stride=2, padding=1)
        # upsample
        self.uh3 = ConcatDeconv2d(c_in=512, c_out=256)
        self.uh2 = ConcatDeconv2d(c_in=256, c_out=128)
        self.uh1 = ConcatDeconv2d(c_in=128, c_out=64)
        self.fr = BasicConv2d(c_in=64, c_out=3, kernel_size=3, stride=1, padding=1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.ConvTranspose2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.h1(x)
        x2 = self.h2(x1)
        x3 = self.h3(x2)
        x4 = self.h4(x3)

        ux3 = self.uh3(x4, x3)
        ux2 = self.uh2(ux3, x2)
        ux1 = self.uh1(ux2, x1)
        # import ipdb; ipdb.set_trace()
        result = self.fr(ux1)
        return result


if __name__ == '__main__':
    temp = torch.randn([8, 3, 256, 256])
    model = ERM()
    output = model(temp)
    import ipdb; ipdb.set_trace()