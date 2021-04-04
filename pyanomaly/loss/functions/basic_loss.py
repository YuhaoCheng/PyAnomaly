"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from collections import namedtuple
from ..loss_registry import LOSS_REGISTRY

__all__ = ['L2Loss', 'IntensityLoss', 'GradientLoss', 'Adversarial_Loss', 
           'Discriminate_Loss', 'AMCDiscriminateLoss', 'AMCGenerateLoss', 
           'GANLoss', 'WeightedPredLoss', 'MSELoss', 'CrossEntropyLoss', 'MemLoss']

def get_loss_args(loss_cfg):
    """
    Generate the loss functions' args based on the configuration file

    Args:
        loss_cfg: An 2-d list. Each item of the list is a list containing a couple of configuration.
    
    Returns:
        A namedtuple contians the configuration
    
    Examples::
        >>> loss_args = get_loss_args([['size_average', None], ['reduce', None], ['reduction', 'mean']])
        >>> loss_args
        >>> namedtuple(size_average=None, reduce=None, reduction='mean')
    """
    args_name = []
    args_value = []
    for config in loss_cfg:
        args_name.append(config[0])
        args_value.append(config[1])
    loss_args_template = namedtuple('LossArgs', args_name)
    return loss_args_template._make(args_value)

def pad_same(in_dim, ks, stride, dilation=1):
    """
    Refernces:
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
    """
    assert stride > 0
    assert dilation >= 1
    effective_ks = (ks - 1) * dilation + 1
    out_dim = (in_dim + stride - 1) // stride
    p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

    padding_before = p // 2
    padding_after = p - padding_before
    return padding_before, padding_after

def conv2d_samepad(in_dim, in_ch, out_ch, ks, stride, dilation=1, bias=True):
    pad_before, pad_after = pad_same(in_dim, ks, stride, dilation)
    if pad_before == pad_after:
        return [nn.Conv2d(in_ch, out_ch, ks, stride, pad_after, dilation, bias=bias)]
    else:
        return [nn.ZeroPad2d((pad_before, pad_after, pad_before, pad_after)),
                nn.Conv2d(in_ch, out_ch, ks, stride, 0, dilation, bias=bias)]


@LOSS_REGISTRY.register()
class L2Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(L2Loss, self).__init__()
        self.eps = eps
    
    def forward(self, gen, gt):
        error = torch.mean((gen - gt)**2)
        x = torch.sqrt(error + self.eps)
        if torch.isnan(x):
            import ipdb; ipdb.set_trace()
        return x

@LOSS_REGISTRY.register()
class IntensityLoss(nn.Module):
    def __init__(self, loss_cfg=None):
        super(IntensityLoss, self).__init__()
        self.l_num =2
    def forward(self, gen_frames, gt_frames):
        # x = torch.mean(torch.abs(gen_frames - gt_frames)**self.l_num)
        return torch.mean(torch.pow(torch.abs(gen_frames - gt_frames), self.l_num))

@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    def __init__(self, loss_cfg=None):
        super(GradientLoss, self).__init__()
        # channels = gen_frames.size(1)
        channels = 3
        pos = torch.tensor(np.identity(channels), dtype=torch.float32)
        neg = (-1 * pos) + 1 -1 
        self.alpha = 1
        self.filter_x = torch.unsqueeze(torch.stack([neg, pos], dim=0), 0).permute(2,3,0,1).cuda()        # c_out, c_in, h, w
        self.filter_y = torch.stack([torch.unsqueeze(neg, dim=0), torch.unsqueeze(pos, dim=0)]).permute(2,3,0,1).cuda()
        # strides = [1,1]

    def forward(self, gen_frames, gt_frames):

        gen_frames_x = torch.nn.ZeroPad2d((1,0,0,0))(gen_frames)
        gen_frames_y = torch.nn.ZeroPad2d((0,0,1,0))(gen_frames)
        gt_frames_x = torch.nn.ZeroPad2d((1,0,0,0))(gt_frames)
        gt_frames_y = torch.nn.ZeroPad2d((0,0,1,0))(gt_frames)
        
        gen_dx = torch.abs(torch.nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(torch.nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(torch.nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(torch.nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gen_dx - gt_dx)
        grad_diff_y = torch.abs(gen_dy - gt_dy)
        # import ipdb; ipdb.set_trace()
        # return gen_dx, gen_dy
        return torch.mean(torch.pow(grad_diff_x, self.alpha)+ torch.pow(grad_diff_y, self.alpha))

@LOSS_REGISTRY.register()
class Adversarial_Loss(nn.Module):
    def __init__(self, loss_cfg):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)

@LOSS_REGISTRY.register()
class Discriminate_Loss(nn.Module):
    def __init__(self, loss_cfg):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

@LOSS_REGISTRY.register()
class AMCDiscriminateLoss(nn.Module):
    def __init__(self, loss_cfg):
        super(AMCDiscriminateLoss, self).__init__()
        self.t1 = nn.BCELoss()
        
    def forward(self, outputs, labels):
        return self.t1(outputs, labels)

@LOSS_REGISTRY.register()
class AMCGenerateLoss(nn.Module):
    def __init__(self, loss_cfg):
        super(AMCGenerateLoss, self).__init__()
        self.t1 = nn.BCELoss()
    def forward(self, fake_outputs, fake):
        return self.t1(fake_outputs, fake)

@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    def __init__(self, loss_cfg):
        """ Initialize the GANLoss class.

        Parameters:
            loss_cfg: A list of configuration. Template: [['gan_mode', vanilla], ['target_real_label', 1.0], ['target_fake_label', 0.0]]
                      gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
                      target_real_label (bool) - - label for a real image
                      target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        loss_args = get_loss_args(loss_cfg)
        self.register_buffer('real_label', torch.tensor(loss_args.target_real_label))
        self.register_buffer('fake_label', torch.tensor(loss_args.target_fake_label))
        self.gan_mode = loss_args.gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            loss = -prediction.mean() if target_is_real else prediction.mean()
        return loss

@LOSS_REGISTRY.register()
class WeightedPredLoss(nn.Module):
    def __init__(self):
        super(WeightedPredLoss, self).__init__()
        # pass
    
    def forward(self, x, target):
        pred_len = target.shape[2]
        weight = [i * 1.0 for i in range(pred_len, 0, -1)]
        weighted_error = [torch.mean(torch.pow(x[:,:,i,:,:] - target[:,:,i,:,:], 2)) * weight[i] for i in range(pred_len)]
        error = sum(weighted_error)
        # error /= pred_len ** 2
        # import ipdb; ipdb.set_trace()

        error /= pred_len ** 2
        return error

@LOSS_REGISTRY.register()
class MSELoss(nn.MSELoss):
    '''
    loss_cfg = [['size_average', None], ['reduce', None], ['reduction', 'mean']]
    '''
    def __init__(self, loss_cfg):
        loss_args = get_loss_args(loss_cfg)
        if len(loss_args) == 0:
            super(MSELoss, self).__init__()
        else:
            super(MSELoss, self).__init__(loss_args)

@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    '''
    loss_cfg = [['weight', None], ['size_average', None], ['ignore_index', -100], ['reduce', None], ['reduction', 'mean']]
    '''
    def __init__(self, loss_cfg):
        loss_args = get_loss_args(loss_cfg)
        if len(loss_args) == 0:
            super(CrossEntropyLoss, self).__init__()
        else:
            super(CrossEntropyLoss, self).__init__(loss_args)

@LOSS_REGISTRY.register()
class L1Loss(nn.L1Loss):
    '''
    loss_cfg = [['size_average', None], ['reduce', None], ['reduction', 'mean']]
    '''
    def __init__(self, loss_cfg):
        loss_args = get_loss_args(loss_cfg)
        if len(loss_args) == 0:
            super(L1Loss, self).__init__()
        else:
            super(L1Loss, self).__init__(loss_args)

@LOSS_REGISTRY.register()
class MemLoss(nn.Module):
    def __init__(self):
        super(MemLoss, self).__init__()
        # self.l_num =2
    def forward(self, att_weights):
        att_weights = att_weights + (att_weights == 0).float() * 1.0
        # import ipdb; ipdb.set_trace()
        # print(f'the memae loss is{x}')
        return torch.mean(-att_weights * att_weights.log())
