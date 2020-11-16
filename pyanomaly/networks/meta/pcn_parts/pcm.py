import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
from pyanomaly.networks.parts.base.commonness import Conv2dLeakly
from pyanomaly.networks.parts.base.commonness import DoubleConv, Down, Up, OutConv,  BasicConv2d
from .convolution_lstm import ConvLSTMCell

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
    def __init__(self, c_in, c_out, bilinear=False):
        super(PEP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bilinear = bilinear

        self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        self.up1 = Up(768, 512, 256, self.bilinear)
        self.up2 = Up(384,256,128, self.bilinear)
        self.up3 = Up(192,128,64, self.bilinear)

    # @torchsnooper.snoop()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x



class PCM(nn.Module):
    def __init__(self):
        super(PCM, self).__init__()
        self.convlstm = SingleStampConvLSTM(input_channels=64, hidden_channels=[128, 64], kernel_size=3)
        self.pep = PEP(c_in=3, c_out=64, bilinear=True)
        # self.fr = Conv2dLeakly(c_in=64, c_out=3,kernel_size=3, stride=1, padding=1)
        self.fr = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.ConvTranspose2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    # @torchsnooper.snoop()
    def forward(self, video_clip):
        # the video_clip is [N C D H W]
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
            Ihat = self.fr(x)
            # import ipdb; ipdb.set_trace()
            # temp = Ihat.detach()
            temp = Ihat
            if time_stamp == len_video-1: # 最后一个
                result = Ihat
        
        return result


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        # the part to initalize the convLSTM
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        
        self.pep = PEP(c_in=3, c_out=64, bilinear=False)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                print(name)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)



if __name__ == '__main__':
    data = torch.randn([8,3,4,128,192]).cuda()
    label = torch.randn([8,3,128,192]).cuda()
    model = PCM().cuda()
    result = model(data)
    loss = nn.L1Loss()
    l = loss(result, label)
    l.backward()
    import ipdb; ipdb.set_trace()