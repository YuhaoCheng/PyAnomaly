import torch
import torch.nn as nn
import torchsnooper
import torch.nn.functional as F
import functools
from pyanomaly.networks.model_registry import META_ARCH_REGISTRY

class BasicConv2d(nn.Module):
    '''
    The basic convaolution with bn
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # return F.relu(x, inplace=True)
        return x 

class Conv2dLeakly(nn.Module):
    '''
    The basic convolution with leakly relu
    conv-bn-leakyReLU
    '''
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

class ConcatDeconv2dReduce(nn.Module):
    '''
    the outpuct channel is the c_out, c_out <= c_in
    '''
    def __init__(self, c_in, c_out, dropout_prob):
        '''
        use the conv_tranpose to enlarge the feature into two times
        '''
        super(ConcatDeconv2dReduce, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.reduce_channel = nn.Conv2d(c_out*2, c_out, kernel_size=1)
    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        x1 = self.dropout(x1)
        x1 = F.relu_(x1)
        x2 = torch.cat([x1,x2], dim=1)
        x2 = self.reduce_channel(x2)
        # import ipdb; ipdb.set_trace()
        return x2

class ConcatDeconv2d(nn.Module):
    def __init__(self, c_in, c_out, dropout_prob=0):
        '''
        use the conv_tranpose to enlarge the feature into two times
        '''
        super(ConcatDeconv2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        x1 = self.dropout(x1)
        x1 = F.relu(x1)
        x2 = torch.cat([x1,x2], dim=1)
        # x2 = self.reduce_channel(x2)
        # import ipdb; ipdb.set_trace()
        return x2


class Deconv2d(nn.Module):
    def __init__(self, c_in, c_out, dropout_prob):
        '''
        use the conv_tranpose to enlarge the feature into two times
        '''
        super(Deconv2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(c_in, c_out, kernel_size=(4,4), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout2d(p=dropout_prob)
    
    def forward(self,x):
        x = self.conv_transpose(x)
        x = self.dropout(x)

        return F.relu_(x)

'''
parts of U-Net: DoubleConv, Down, Up, OutConv
'''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, a, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_channels, in_channels//2,1))
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(a, a, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Inception3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception3x3, self).__init__()
        self.s3_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.s3_1n = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s3_n1 = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
    def forward(self,x):
        x = self.s3_11(x)
        x = self.s3_1n(x)
        x = self.s3_n1(x)

        return x

class Inception5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception5x5, self).__init__()
        self.s5_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.s5_1n_a = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s5_n1_a = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        self.s5_1n_b = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s5_n1_b = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
    def forward(self,x):
        x = self.s5_11(x)
        x = self.s5_1n_a(x)
        x = self.s5_n1_a(x)
        x = self.s5_1n_b(x)
        x = self.s5_n1_b(x)

        return x
class Inception7x7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception7x7, self).__init__()
        self.s7_11 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.s7_1n_a = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s7_n1_a = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        self.s7_1n_b = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s7_n1_b = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        self.s7_1n_c = BasicConv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        self.s7_n1_c = BasicConv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
    def forward(self, x):
        x = self.s7_11(x)
        x = self.s7_1n_a(x)
        x = self.s7_n1_a(x)
        x = self.s7_1n_b(x)
        x = self.s7_n1_b(x)
        x = self.s7_1n_c(x)
        x = self.s7_n1_c(x)

        return x


class Inception(nn.Module):
    def __init__(self, c_in, c_out, max_filter_size=7):
        super(Inception, self).__init__()
        assert max_filter_size % 2 == 1 and max_filter_size < 8
        self.n_branch = (max_filter_size + 1 ) // 2
        assert c_out % self.n_branch == 0
        nf_branch = c_out // self.n_branch
        # 1x1 
        self.branch1 = BasicConv2d(in_channels=c_in, out_channels=nf_branch, kernel_size=1)
        # 3x3
        self.branch2 = Inception3x3(in_channels=c_in, out_channels=nf_branch)
        # 5x5
        self.branch3 = Inception5x5(in_channels=c_in, out_channels=nf_branch)
        # 7x7
        self.branch4 = Inception7x7(in_channels=c_in, out_channels=nf_branch)

    def forward(self, x):
        out1 = self.branch1(x)
        if self.n_branch == 1:
            return out1
        out2 = self.branch2(x)
        if self.n_branch == 2:
            return torch.cat([out1, out2], dim=1)
        out3 = self.branch3(x)
        if self.n_branch == 3:
            return torch.cat([out1, out2, out3], dim=1)
        out4 = self.branch4(x)
        if self.n_branch == 4:
            return torch.cat([out1, out2, out3, out4], dim=1)

        # return x

@META_ARCH_REGISTRY.register()        
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, num_filters, use_norm=False,norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        '''
        different from ano_pred with norm here
        '''


        super(PixelDiscriminator, self).__init__()
        if use_norm:
            if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
                use_bias = norm_layer.func != nn.InstanceNorm2d
            else:
                use_bias = norm_layer != nn.InstanceNorm2d
        else:
            use_bias=True

        self.net=[]
        self.net.append(nn.Conv2d(input_nc,num_filters[0],kernel_size=4,padding=2,stride=2))
        self.net.append(nn.LeakyReLU(0.1))
        if use_norm:
            for i in range(1,len(num_filters)-1):
                self.net.extend([nn.Conv2d(num_filters[i-1],num_filters[i],4,2,2,bias=use_bias),
                                 nn.LeakyReLU(0.1),
                                 norm_layer(num_filters[i])])
        else :
            for i in range(1,len(num_filters)-1):
                self.net.extend([nn.Conv2d(num_filters[i-1],num_filters[i],4,2,2,bias=use_bias),
                                 nn.LeakyReLU(0.1)])
        self.net.append(nn.Conv2d(num_filters[-1],1,4,1,2))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        with torch.autograd.set_detect_anomaly(True):
            output = self.net(input)
        return output



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
