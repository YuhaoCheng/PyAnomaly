import torch
import numpy as np
import torch.nn as nn
from torch import gt

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


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, gen, gt):
        # x = torch.mean(torch.sqrt(((gen - gt)**2 + (1e-8))))
        x = torch.sqrt(((gen - gt)**2 + (1e-8)))
        import ipdb; ipdb.set_trace()
        if torch.isnan(x):
            import ipdb; ipdb.set_trace()
        return x

class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()
        self.l_num =2
    def forward(self, gen_frames, gt_frames):
        x = torch.mean(torch.pow(torch.abs(gen_frames - gt_frames), self.l_num))
        # x = torch.mean(torch.abs(gen_frames - gt_frames)**self.l_num)
        return x

class GradientLoss(nn.Module):
    def __init__(self):
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


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)

class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

class AMCDiscriminateLoss1(nn.Module):
    def __init__(self):
        super(AMCDiscriminateLoss1, self).__init__()
        self.t1 = nn.BCELoss()
        
    def forward(self, outputs, labels):
        loss  = self.t1(outputs, labels) 
        return loss

class AMCDiscriminateLoss2(nn.Module):
    def __init__(self):
        super(AMCDiscriminateLoss2, self).__init__()
        self.t1 = nn.BCELoss()
        
    def forward(self, outputs, labels):
        loss  = self.t1(outputs, labels) 
        return loss

class AMCGenerateLoss(nn.Module):
    def __init__(self):
        super(AMCGenerateLoss, self).__init__()
        self.t1 = nn.BCELoss()
    def forward(self, fake_outputs, fake):
        loss  = self.t1(fake_outputs, fake)
        return loss

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
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
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class WeightedPredLoss(nn.Module):
    def __init__(self):
        super(WeightedPredLoss, self).__init__()
    
    def forward(self, x, target):
        error = 0
        pred_len = target.shape[2]
        weight = [i * 1.0 for i in range(pred_len, 0, -1)]
        weighted_error = [torch.mean(torch.pow(x[:,:,i,:,:] - target[:,:,i,:,:], 2)) * weight[i] for i in range(pred_len)]
        for item in weighted_error:
            error += item
        # error /= pred_len ** 2
        # import ipdb; ipdb.set_trace()
        
        error /= pred_len ** 2
        return error



LOSSDICT ={
    'mse': nn.MSELoss(reduction='mean').cuda(),
    'cross': nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False).cuda(),
    'g_adverserial_loss': Adversarial_Loss().cuda(),
    'd_adverserial_loss': Discriminate_Loss().cuda(),
    'opticalflow_loss': nn.L1Loss().cuda(),
    'opticalflow_loss_sqrt': L2Loss().cuda(),
    'gradient_loss':GradientLoss().cuda(),
    'intentsity_loss': IntensityLoss().cuda(),
    'amc_d_adverserial_loss_1': AMCDiscriminateLoss1().cuda(),
    'amc_d_adverserial_loss_2': AMCDiscriminateLoss2().cuda(),
    'amc_g_adverserial_loss': AMCGenerateLoss().cuda(),
    'gan_loss': GANLoss(gan_mode='vanilla').cuda(),
    'gan_loss_mse': GANLoss(gan_mode='lsgan').cuda(),
    'A_loss': IntensityLoss().cuda(),
    'B_loss': IntensityLoss().cuda(),
    'C_loss': IntensityLoss().cuda(),
    # 'rec_loss': nn.MSELoss(reduction='mean').cuda(),
    'rec_loss': L2Loss().cuda(),
    'weighted_pred_loss': WeightedPredLoss().cuda()
}

def get_basic_loss(loss_name, cfg):
    assert loss_name in LOSSDICT.keys(), f'The loss name: {loss_name} is not support'
    print(f'Using the loss:{loss_name}')
    loss_function = LOSSDICT[loss_name]
    return loss_function

