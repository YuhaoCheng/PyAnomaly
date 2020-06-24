import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
# from torch.autograd import Variable
from .convolution_lstm import ConvLSTMCell
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

class SingleStampConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(SingleStampConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)  # 是len(hidden_channels)
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        # for each sequence, we need to clear the internal_state
        self.internal_state = list()
    
    # @torchsnooper.snoop()
    def forward(self, input, step):
        x = input  # the input is a single image, shape is N C H W
        for i in range(self.num_layers):
            if step == 0 and i==0:
                self.internal_state = list() # 清空state中的状态，因为换到下一个video clip了
            name = 'cell{}'.format(i)

            if step == 0:
                # all cells are initialized in the first step
                bsize, _, height, width = x.size()
                (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                self.internal_state.append((h, c))
            
            # do forward
            (h, c) = self.internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            self.internal_state[i] = (x, new_c)
        return x, new_c

class PEP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PEP, self).__init__()
        # downsample
        self.h1 = Conv2dLeakly(c_in=in_channels,   c_out=64, kernel_size=3, stride=1, padding=1)
        self.h2 = Conv2dLeakly(c_in=64,  c_out=128, kernel_size=3, stride=2, padding=1)
        self.h3 = Conv2dLeakly(c_in=128, c_out=256, kernel_size=3, stride=2, padding=1)
        self.h4 = Conv2dLeakly(c_in=256, c_out=512, kernel_size=3, stride=2, padding=1)
        # upsample
        self.uh3 = ConcatDeconv2d(c_in=512, c_out=256)
        self.uh2 = ConcatDeconv2d(c_in=256, c_out=128)
        self.uh1 = ConcatDeconv2d(c_in=128, c_out=out_channels)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.ConvTranspose2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def forward(self, x):
        x1 = self.h1(x)
        x2 = self.h2(x1)
        x3 = self.h3(x2)
        x4 = self.h4(x3)

        ux3 = self.uh3(x4, x3)
        ux2 = self.uh2(ux3, x2)
        ux1 = self.uh1(ux2, x1)

        return ux1

class PCM(nn.Module):
    def __init__(self):
        super(PCM, self).__init__()
        self.convlstm = SingleStampConvLSTM(input_channels=64, hidden_channels=[128, 128, 64], kernel_size=3)
        self.pep = PEP(in_channels=3, out_channels=64)
        self.fr = Conv2dLeakly(c_in=64, c_out=3,kernel_size=3, stride=1, padding=1)
    
    def _init_weights(self):
        pass
    
    # @torchsnooper.snoop()
    def forward(self, video_clip):
        # the video_clip is N C D H W
        len_video = video_clip.shape[2]
        frames = torch.chunk(video_clip, len_video, 2)
        for time_stamp in range(len_video):
            # print(time_stamp)
            frame = frames[time_stamp].squeeze(2)
            if time_stamp == 0:
                E = torch.zeros_like(frame)
            else:
                E = torch.sub(frame, temp)
            R = self.pep(E)
            x, _ = self.convlstm(R, time_stamp)
            # import ipdb; ipdb.set_trace()
            Ihat = self.fr(x)
            temp = Ihat.detach()
            if time_stamp == len_video-1: # 最后一个
                result = Ihat
        
        return result


if __name__ == '__main__':
    data = torch.randn([8,3,4,128,192]).cuda()
    label = torch.randn([8,3,128,192]).cuda()
    model = PCM().cuda()
    result = model(data)
    loss = nn.L1Loss()
    l = loss(result, label)
    l.backward()
    import ipdb; ipdb.set_trace()