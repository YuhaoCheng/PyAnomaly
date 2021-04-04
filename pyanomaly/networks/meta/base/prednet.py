# -*- coding: utf-8 -*-

'''
PredNet in PyTorch.
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def hard_sigmoid(x):
    '''
    - hard sigmoid function by zcr.
    - Computes element-wise hard sigmoid of x.
    - what is hard sigmoid?
        Segment-wise linear approximation of sigmoid. Faster than sigmoid.
        Returns 0. if x < -2.5, 1. if x > 2.5. In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.
    - See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    '''
    slope = 0.2
    shift = 0.5
    x = (slope * x) + shift
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

def get_activationFunc(act_str):
    act = act_str.lower()
    if act == 'relu':
        # return nn.ReLU(True)
        return nn.ReLU()
    elif act == 'tanh':
        # return F.tanh
        return nn.Tanh()
    # elif act == 'hard_sigmoid':
    #     return hard_sigmoid
    else:
        raise(RuntimeError('cannot obtain the activation function named %s' % act_str))

def batch_flatten(x):
    '''
    equal to the `batch_flatten` in keras.
    x is a Variable in pytorch
    '''
    shape = [*x.size()]
    dim = np.prod(shape[1:])
    dim = int(dim)      # 不加这步的话, dim是<class 'numpy.int64'>类型, 不能在view中用. 加上这步转成<class 'int'>类型.
    return x.view(-1, dim)



class PredNet(nn.Module):
    """
    PredNet realized by zcr.
    
    Args:
        stack_sizes:
            - Number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            - Length of stack_size (i.e. len(stack_size) and we use `num_layers` to denote it) is the number of layers in the architecture.
            - First element is the number of channels in the input.
            - e.g., (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and
              has 16 and 32 channels in the second and third layers, respectively.
            - 下标为(lay + 1)的值即为pytorch中第lay个卷积层的out_channels参数. 例如上述16对应到lay 0层(即输入层)的A和Ahat的out_channels是16.
        R_stack_sizes:
            - Number of channels in the representation (R) modules.
            - Length must equal length of stack_sizes, but the number of channels per layer can be different.
            - 即pytorch中卷积层的out_channels参数.
        A_filter_sizes:
            - Filter sizes for the target (A) modules. (except the target (A) in lowest layer (i.e., input image))
            - Has length of len(stack_sizes) - 1.
            - e.g., (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of
              the errors (E) from the layer below (followed by max-pooling)
            - 即pytorch中卷积层的kernel_size.
        Ahat_filter_sizes:
            - Filter sizes for the prediction (Ahat) modules.
            - Has length equal to length of stack_sizes.
            - e.g., (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution
              of the representation (R) modules at each layer.
            - 即pytorch中卷积层的kernel_size.
        R_filter_sizes:
            - Filter sizes for the representation (R) modules.
            - Has length equal to length of stack_sizes.
            - Corresponds to the filter sizes for all convolutions in the LSTM.
            - 即pytorch中卷积层的kernel_size.
        pixel_max:
            - The maximum pixel value.
            - Used to clip the pixel-layer prediction.
        error_activation:
            - Activation function for the error (E) units.
        A_activation:
            - Activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation:
            - Activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation:
            - Activation function for the gates in the LSTM.
        output_mode:
            - Either 'error', 'prediction', 'all' or layer specification (e.g., R2, see below).
            - Controls what is outputted by the PredNet.
                - if 'error':
                    The mean response of the error (E) units of each layer will be outputted.
                    That is, the output shape will be (batch_size, num_layers).
                - if 'prediction':
                    The frame prediction will be outputted.
                - if 'all':
                    The output will be the frame prediction concatenated with the mean layer errors.
                    The frame prediction is flattened before concatenation.
                    Note that nomenclature of 'all' means all TYPE of the output (i.e., `error` and `prediction`), but should not be confused with returning all of the layers of the model.
                - For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                    e.g., to return the features of the LSTM "representational" units in the lowest layer, output_mode should be specificied as 'R0'.
                    The possible unit types are 'R', 'Ahat', 'A', and 'E' corresponding to the 'representation', 'prediction', 'target', and 'error' units respectively.
        extrap_start_time:
            - Time step for which model will start extrapolating.
            - Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        data_format:
            - 'channels_first': (channel, Height, Width)
            - 'channels_last' : (Height, Width, channel)

    """
    def __init__(self, stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes,
                 pixel_max = 1.0, error_activation = 'relu', A_activation = 'relu', LSTM_activation = 'tanh',
                 LSTM_inner_activation = 'hard_sigmoid', output_mode = 'error',
                 extrap_start_time = None, data_format = 'channels_last', return_sequences = False):
        super(PredNet, self).__init__()
        self.stack_sizes = stack_sizes
        self.num_layers  = len(stack_sizes)
        assert len(R_stack_sizes) == self.num_layers
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filter_sizes) == self.num_layers - 1
        self.A_filter_sizes = A_filter_sizes
        assert len(Ahat_filter_sizes) == self.num_layers
        self.Ahat_filter_sizes = Ahat_filter_sizes
        assert len(R_filter_sizes) == self.num_layers
        self.R_filter_sizes = R_filter_sizes

        self.pixel_max = pixel_max
        self.error_activation = error_activation
        self.A_activation = A_activation
        self.LSTM_activation = LSTM_activation
        self.LSTM_inner_activation = LSTM_inner_activation

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.num_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_NO = int(self.output_mode[-1])    # suppose the number of layers is < 10
        else:
            self.output_layer_type = None
            self.output_layer_NO = None

        self.extrap_start_time = extrap_start_time
        assert data_format in ['channels_first', 'channels_last']
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.channel_axis = -3
            self.row_axis = -2
            self.col_axis = -1
        else:
            self.channel_axis = -1
            self.row_axis = -3
            self.col_axis = -2

        self.return_sequences = return_sequences

        self.make_layers()


    def get_initial_states(self, input_shape):
        '''
        input_shape is like: (batch_size, timeSteps, Height, Width, 3)
                         or: (batch_size, timeSteps, 3, Height, Width)
        '''
        init_height = input_shape[self.row_axis]     # equal to `init_nb_rows` in original version
        init_width  = input_shape[self.col_axis]     # equal to `init_nb_cols` in original version

        base_initial_state = np.zeros(input_shape)
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = np.sum(base_initial_state, axis = non_channel_axis)
        base_initial_state = np.sum(base_initial_state, axis = 1)   # (batch_size, 3)

        initial_states = []
        states_to_pass = ['R', 'c', 'E']    # R is `representation`, c is Cell state in LSTM, E is `error`.
        layerNum_to_pass = {sta: self.num_layers for sta in states_to_pass}
        if self.extrap_start_time is not None:
            states_to_pass.append('Ahat')   # pass prediction in states so can use as actual for t+1 when extrapolating
            layerNum_to_pass['Ahat'] = 1

        for sta in states_to_pass:
            for lay in range(layerNum_to_pass[sta]):
                downSample_factor = 2 ** lay            # 下采样缩放因子
                row = init_height // downSample_factor
                col = init_width  // downSample_factor
                if sta in ['R', 'c']:
                    stack_size = self.R_stack_sizes[lay]
                elif sta == 'E':
                    stack_size = self.stack_sizes[lay] * 2
                elif sta == 'Ahat':
                    stack_size = self.stack_sizes[lay]
                output_size = stack_size * row * col    # flattened size
                reducer = np.zeros((input_shape[self.channel_axis], output_size))   # (3, output_size)
                initial_state = np.dot(base_initial_state, reducer)                 # (batch_size, output_size)

                if self.data_format == 'channels_first':
                    output_shape = (-1, stack_size, row, col)
                else:
                    output_shape = (-1, row, col, stack_size)
                # initial_state = torch.from_numpy(np.reshape(initial_state, output_shape)).float().cuda()
                initial_state = Variable(torch.from_numpy(np.reshape(initial_state, output_shape)).float().cuda(), requires_grad = True)
                initial_states += [initial_state]

        if self.extrap_start_time is not None:
            # initial_states += [torch.IntTensor(1).zero_().cuda()]   # the last state will correspond to the current timestep
            initial_states += [Variable(torch.IntTensor(1).zero_().cuda())]   # the last state will correspond to the current timestep
        return initial_states


    # def compute_output_shape(self, input_shape):
    #     if self.output_mode == 'prediction':
    #         out_shape = input_shape[2:]
    #     elif self.output_mode == 'error':   # error模式输出为各层误差,每层一个标量
    #         out_shape = (self.num_layers,)
    #     elif self.output_mode == 'all':
    #         out_shape = (np.prod(input_shape[2:]) + self.num_layers,)   # np.prod 元素逐个相乘
    #     else:
    #         if self.output_layer_type == 'R':
    #             stack_str = 'R_stack_sizes'
    #         else:
    #             stack_str = 'stack_sizes'

    #         if self.output_layer_type == 'E':
    #             stack_multi = 2
    #         else:
    #             stack_multi = 1

    #         out_stack_size = stack_multi * getattr(self, stack_str)[self.output_layer_NO]
    #         layer_out_row = input_shape[self.row_axis] / (2 ** self.output_layer_NO)
    #         layer_out_col = input_shape[self.col_axis] / (2 ** self.output_layer_NO)
    #         if self.data_format == 'channels_first':
    #             out_shape = (out_stack_size, layer_out_row, layer_out_col)
    #         else:
    #             out_shape = (layer_out_row, layer_out_col, out_stack_size)

    #         if self.return_sequences:
    #             return (input_shape[0], input_shape[1]) + out_shape    # input_shape[1] is the timesteps
    #         else:
    #             return (input_shape[0],) + out_shape


    def isNotTopestLayer(self, layerIndex):
        '''judge if the layerIndex is not the topest layer.'''
        return layerIndex < self.num_layers - 1


    def make_layers(self):
        '''
        equal to the `build` method in original version.
        '''
        # i: input, f: forget, c: cell, o: output
        self.conv_layers = {item: [] for item in ['i', 'f', 'c', 'o', 'A', 'Ahat']}
        lstm_list = ['i', 'f', 'c', 'o']

        for item in sorted(self.conv_layers.keys()):
            for lay in range(self.num_layers):
                downSample_factor = 2 ** lay        # 下采样缩放因子
                if item == 'Ahat':
                    in_channels = self.R_stack_sizes[lay]   # 因为Ahat是对R的输出进行卷积, 所以输入Ahat的channel数就是相同层中R的输出channel数.
                    self.conv_layers['Ahat'].append(nn.Conv2d(in_channels = in_channels,
                                                              out_channels = self.stack_sizes[lay],
                                                              kernel_size = self.Ahat_filter_sizes[lay],
                                                              stride = (1, 1),
                                                              padding = int((self.Ahat_filter_sizes[lay] - 1) / 2)    # the `SAME` mode (i.e.,(kernel_size - 1) / 2)
                                                              ))
                    act = 'relu' if lay == 0 else self.A_activation
                    self.conv_layers['Ahat'].append(get_activationFunc(act))

                elif item == 'A':
                    if self.isNotTopestLayer(lay):   # 这里只是控制一下层数(比其他如Ahat等少一层)
                        # NOTE: 这里是从第二层(lay = 1)开始构建A的(因为整个网络的最低一层(layer0)的A就是原始图像(可以将layer0的A视为一个`恒等层`, 即输入图像, 输出原封不动的图像))
                        in_channels = self.R_stack_sizes[lay] * 2   # A卷积层输入特征数(in_channels)是对应层E的特征数,E包含(Ahat-A)和(A-Ahat)两部分,故x2. [从paper的Fig.1左图来看, E是Ahat的输出和A进行相减, 之后拼接.]
                        self.conv_layers['A'].append(nn.Conv2d(in_channels = in_channels,
                                                               out_channels = self.stack_sizes[lay + 1],
                                                               kernel_size = self.A_filter_sizes[lay],
                                                               stride = (1, 1),
                                                               padding = int((self.A_filter_sizes[lay] - 1) / 2)    # the `SAME` mode
                                                               ))
                        self.conv_layers['A'].append(get_activationFunc(self.A_activation))

                elif item in lstm_list:     # 构建R模块
                    # R的输入特征数(in_channels): 同层的E、同层上一时刻的R(即R_t-1)、 同时刻上层的R(即R_l+1)这三者的特征数之和.
                    # 如果该R模块位于顶层, 则没有来自上层的R. 其中:
                    # - stack_sizes[lay] * 2 表示的是同层E的channel数 (因为E是将同层的A和Ahat在channel这一维度上拼接得到的, 故x2)
                    # - R_stack_sizes[lay] 表示的是同层上一时刻的R的channel数
                    # - R_stack_sizes[lay + 1] 表示的是同时刻上层的R的channel数
                    in_channels = self.stack_sizes[lay] * 2 + self.R_stack_sizes[lay]
                    if self.isNotTopestLayer(lay):
                        in_channels += self.R_stack_sizes[lay + 1]
                    # for j in lstm_list:     # 严重的bug! 赶紧注释掉...下面的向前缩进4个空格...
                    # LSTM中的i,f,c,o的非线性激活函数层放在forward中实现. (因为这里i,f,o要用hard_sigmoid函数, Keras中LSTM默认就是hard_sigmoid, 但是pytorch中需自己实现)
                    # act = self.LSTM_activation if j == 'c' else self.LSTM_inner_activation
                    # act = get_activationFunc(act)
                    self.conv_layers[item].append(nn.Conv2d(in_channels = in_channels,
                                                         out_channels = self.R_stack_sizes[lay],
                                                         kernel_size = self.R_filter_sizes[lay],
                                                         stride = (1, 1),
                                                         padding = int((self.R_filter_sizes[lay] - 1) / 2)    # the `SAME` mode
                                                         ))

        for name, layerList in self.conv_layers.items():
            self.conv_layers[name] = nn.ModuleList(layerList)
            setattr(self, name, self.conv_layers[name])

        # see the source code in:
        #     [PyTorch]: http://pytorch.org/docs/master/_modules/torch/nn/modules/upsampling.html
        #     [Keras  ]: keras-master/keras/layers/convolution.py/`class UpSampling2D(Layer)`
        # self.upSample = nn.Upsample(size = (2, 2), mode = 'nearest')  # 是错误的! pytorch中的scale_factor参数对应到keras中的size参数.
        self.upSample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        # see the source code in:
        #     [PyTorch]: http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxPool2d
        #     [Keras  ]: keras-master/keras/layers/pooling.py/``
        # `pool_size` in Keras is equal to `kernel_size` in pytorch.
        # [TODO] padding here is not very clear. Is `0` here is the `SAME` mode in Keras?
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)


    def step(self, A, states):
        '''
        这个step函数是和原代码中的`step`函数是等价的. 是PredNet的核心逻辑所在.
        类比于标准LSTM的实现方式, 这个step函数的角色相当于LSTMCell, 而下面的forward函数相当于LSTM类.

        Args:
            A: 4D tensor with the shape of (batch_size, 3, Height, Width). 就是从A_withTimeStep按照时间步抽取出来的数据.
            states 和 `forward`函数的`initial_states`的形式完全相同, 只是后者是初始化的PredNet状态, 而这里的states是在timesteps内运算时的PredNet参数.
        '''
        n = self.num_layers
        R_current = states[       :    (n)]
        c_current = states[    (n):(2 * n)]
        E_current = states[(2 * n):(3 * n)]

        if self.extrap_start_time is not None:
            timestep = states[-1]
            if timestep >= self.t_extrap:   # if past self.extrap_start_time, the previous prediction will be treated as the actual.
                A = states[-2]
            else:
                A = A

        R_list = []
        c_list = []
        E_list = []

        # Update R units starting from the top.
        for lay in reversed(range(self.num_layers)):
            inputs = [R_current[lay], E_current[lay]]   # 如果是顶层, R_l的输入只有两个: E_l^t, R_l^(t-1). 即没有高层的R模块的输入项.
            if self.isNotTopestLayer(lay):              # 如果不是顶层,R_l的输入就有三个: E_l^t, R_l^(t-1), R_(l+1)^t. R_up即为R_(l+1)^t
                inputs.append(R_up)
            
            inputs = torch.cat(inputs, dim = self.channel_axis)
            if not isinstance(inputs, Variable):        # 第一个时间步内inputs还是Tensor类型, 但是过一遍网络之后, 以后的时间步中就都是Variable类型了.
                inputs = Variable(inputs, requires_grad = True)

            # print(lay, type(inputs), inputs.size())   # 正确的情况下, 举例如下:
            # lay3: torch.Size([8, 576, 16, 20])  [576 = 384(E_l^t) + 192(R_l^(t-1))]
            # lay2: torch.Size([8, 480, 32, 40])  [480 = 192(E_l^t) +  96(R_l^(t-1)) + 192(R_(l+1)^t)]
            # lay1: torch.Size([8, 240, 64, 80])  [240 =  96(E_l^t) +  48(R_l^(t-1)) +  96(R_(l+1)^t)]
            # lay0: torch.Size([8, 57, 160, 128]) [ 57 =   6(E_l^t) +   3(R_l^(t-1)) +  48(R_(l+1)^t)]

            # see https://github.com/huggingface/torchMoji/blob/master/torchmoji/lstm.py
            in_gate     = hard_sigmoid(self.conv_layers['i'][lay](inputs))
            forget_gate = hard_sigmoid(self.conv_layers['f'][lay](inputs))
            cell_gate   = F.tanh(self.conv_layers['c'][lay](inputs))
            out_gate    = hard_sigmoid(self.conv_layers['o'][lay](inputs))

            # print(forget_gate.size())       # torch.Size([8, 192, 16, 20])
            # print(c_current[lay].size())    # torch.Size([8, 192, 16, 20])
            # print(in_gate.size())           # torch.Size([8, 192, 16, 20])
            # print(cell_gate.size())         # torch.Size([8, 192, 16, 20])
            # print(type(forget_gate))        # <class 'torch.autograd.variable.Variable'>
            # print(type(c_current[lay]))     # <class 'torch.cuda.FloatTensor'>
            # print(type(Variable(c_current[lay])))     # <class 'torch.autograd.variable.Variable'>
            # print(type(in_gate))            # <class 'torch.autograd.variable.Variable'>
            # print(type(cell_gate))          # <class 'torch.autograd.variable.Variable'>
            if not isinstance(c_current[lay], Variable):
                c_current[lay] = Variable(c_current[lay], requires_grad = True)
            c_next = (forget_gate * c_current[lay]) + (in_gate * cell_gate)     # 对应元素相乘
            R_next = out_gate * F.tanh(c_next)      # `R_next` here相当于标准LSTM中的hidden state. 这个就是视频的表征.

            c_list.insert(0, c_next)
            R_list.insert(0, R_next)

            if lay > 0:
                # R_up = self.upSample(R_next).data     # 注意: 这里出来的是Variable, 上面要append到inputs列表里的都是FloatTensor, 所以这里需要变成Tensor形式, 即加个`.data`
                R_up = self.upSample(R_next)            # NOTE: 这个就是困扰好久, 导致loss.backward()报错的原因: torch.cat()中将Tensor和Variable混用导致的错误!
                # print(R_up.size())  # lay3: torch.Size([8, 192, 32, 40])


        # Update feedforward path starting from the bottom.
        for lay in range(self.num_layers):
            Ahat = self.conv_layers['Ahat'][2 * lay](R_list[lay])   # Ahat是R的卷积, 故将同层同时刻的R输入. 这里千万注意: 每个`lay`其实对应的是两个组件: 卷积层+非线性激活层, 所以这里需要用(2 * lay)来索引`lay`对应的卷积层, 用(2 * lay + 1)来索引`lay`对应的非线性激活函数层. 下面对A的处理也是一样.
            Ahat = self.conv_layers['Ahat'][2 * lay + 1](Ahat)      # 勿忘非线性激活.下面对A的处理也是一样.
            if lay == 0:
                # Ahat = torch.min(Ahat, self.pixel_max)            # 错误(keras中的表示方式)
                Ahat[Ahat > self.pixel_max] = self.pixel_max        # passed through a saturating non-linearity set at the maximum pixel value
                frame_prediction = Ahat                             # 最低一层的Ahat即为预测输出帧图像
                # if self.output_mode == 'prediction':
                #     break
            
            # print('&' * 10, lay)
            # print('Ahat', Ahat.size())  # torch.Size([batch_size, 3, 128, 160])
            # print('A', A.size())        # 原来A0直接用的是从dataloader中加载出来的数据, 所以打印的是torch.Size([batch_size, 10, 3, 128, 160]), 这就是问题所在: dataloader返回的数据是(batch_size, timesteps, (image_shape)), 而实际上在RNN中用的是将每个时间步分开的. 现在将核心逻辑解耦出来形成`step`函数, A0就变成torch.Size([batch_size, 3, 128, 160])这个维度了.
            # print('&' * 20)
            
            # compute errors
            if self.error_activation.lower() == 'relu':
                E_up   = F.relu(Ahat - A)
                E_down = F.relu(A - Ahat)
            elif self.error_activation.lower() == 'tanh':
                E_up   = F.tanh(Ahat - A)
                E_down = F.tanh(A - Ahat)
            else:
                raise(RuntimeError('cannot obtain the activation function named %s' % self.error_activation))
            
            E_list.append(torch.cat((E_up, E_down), dim = self.channel_axis))

            # 如果是想要获取特定的层中特定模块的输出:
            if self.output_layer_NO == lay:
                if   self.output_layer_type == 'A':
                    output = A
                elif self.output_layer_type == 'Ahat':
                    output = Ahat
                elif self.output_layer_type == 'R':
                    output = R_list[lay]
                elif self.output_layer_type == 'E':
                    output = E_list[lay]

            if self.isNotTopestLayer(lay):
                A = self.conv_layers['A'][2 * lay](E_list[lay])     # 对E进行卷积+池化之后, 得到同时刻上一层的A, 如果该层已经是最顶层了, 就不用了
                A = self.conv_layers['A'][2 * lay + 1](A)           # 勿忘非线性激活.
                A = self.pool(A)    # target for next layer
        

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for lay in range(self.num_layers):
                    layer_error = torch.mean(batch_flatten(E_list[lay]), dim = -1, keepdim = True)     # batch_flatten函数是zcr依照Kears中同名函数实现的. 第0维是batch_size维度, 将除此维度之外的维度拉平
                    all_error = layer_error if lay == 0 else torch.cat((all_error, layer_error), dim = -1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = torch.cat((batch_flatten(frame_prediction), all_error), dim = -1)

        states = R_list + c_list + E_list
        if self.extrap_start_time is not None:
            states += [frame_prediction, (timestep + 1)]
        return output, states


    def forward(self, A0_withTimeStep, initial_states):
        '''
        A0_withTimeStep is the input from dataloader. Its shape is: (batch_size, timesteps, 3, Height, Width).
            说白了, 这个A0_withTimeStep就是dataloader加载出来的原始图像, 即最底层(layer 0)的A, 只不过在batch_size和timestep两个维度扩展了.
        initial_states  is a list of pytorch-tensors. 这个states参数其实就是初始状态, 因为这个forword函数本身是不被循环执行的.

        NOTE: 这个foward函数目的是为了实现原Keras版本的 `step` 函数, 但是和后者不太一样.  因为原代码的PredNet类是
              继承了Keras中的`Recurrent`类, 所以貌似该父类就实现了将dataloader(即原代码中的SequenceGenerator)加载
              的数据(batch_size, timesteps, 3, H, W)分解为(batch_size, 3, H, W), 然后循环timesteps次求解.
              而这里的forward需要自己实现循环timesteps次. 这里的A的shape就是从dataloader中来的5D tensor (batch_size, timesteps, 3, Height, Width),
              原代码中step函数的输入`x`的shape是4D tensor (batch_size, 3, Height, Width).
        '''

        # 默认是batch_fist == True的, 即第一维是batch_size, 第二维是timesteps.
        # A0_withTimeStep = A0_withTimeStep.transpose(0, 1)   # (b, t, c, h, w) -> (t, b, c, h, w)
        A0_withTimeStep = A0_withTimeStep.permute(2,0,1,3,4)   # (b, c, t, h, w) -> (t, b, c, h, w)

        num_timesteps = A0_withTimeStep.size()[0]

        hidden_states = initial_states    # 赋值为hidden_states是为了在下面的循环中可以无痛使用
        output_list = []                  # output需要保留下来: `error`模式下需要按照layer和timestep进行加权得到最终的loss; `prediction`模式下需要输出每个时间步的预测图像(如timestep为10的话, 输出10个图像)
        for t in range(num_timesteps):
            '''
                原本的LSTM(或普通RNN)是要两重循环的:
                for lay in range(num_layers):
                    for t in range(num_timesteps):
                        pass
                但是正如原Keras版本的代码中脚注部分说的那样: PredNet虽然设定了层数, 但其实实现的时候是用
                一个超级层(`super layer`)实现, 即本身就是一层. 所以这里就没有for lay循环了.
            '''
            A0 = A0_withTimeStep[t, ...]
            output, hidden_states = self.step(A0, hidden_states)
            output_list.append(output)
            # hidden_states 不需要保留,只需让其在时间步内进行`长江后浪推前浪`式的迭代即可.

        if self.output_mode == 'error':
            '''进行按照layer和timestep的加权. 不同于原代码中加Dense layer的方式, 这里加权操作可以直接写在PredNet模型里(就这个if语句里), 也可以将所有时间步中每层的error返回, 在main函数中进行计算. zcr选择后者(和原代码保持一致)'''
            # print(len(output_list))             # 10, 即timestep数
            # print('output: ', output_list)      # 每个时间步的`error`是(batch_size, num_layer)的矩阵, 类型是Variable. [torch.cuda.FloatTensor of size 8x4 (GPU 0)] 根据这个来进行按照layer和timestep的加权, 即可实现loss的计算! (按照layer进行两种加权, 即可得到所谓的`L_0`和`L_all`的两类loss)
            # print('Got the `error` list with the length of len(timeSteps) and shape of each element in this list is: (batch_size, num_layer).')
            return output_list
        elif self.output_mode == 'prediction':
            return output_list  # 此时的output_list是timestep个预测帧图像
        elif self.output_mode != 'all':
            raise(RuntimeError('Kidding? Unknown output mode!'))


if __name__ == '__main__':
    n_channels = 3
    img_height = 128
    img_width  = 160

    stack_sizes       = (n_channels, 48, 96, 192)
    R_stack_sizes     = stack_sizes
    A_filter_sizes    = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes    = (3, 3, 3, 3)

    prednet = PredNet(stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes,
                      output_mode = 'error', return_sequences = True)
