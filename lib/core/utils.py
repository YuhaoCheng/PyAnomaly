'''
Some useful tools in training process
'''
# import sys
# sys.path.append('/export/home/chengyh/Anomaly_DA')
import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as tf
from tsnecuda import TSNE
from lib.utils.flow_utils import flow2img
from skimage.measure import compare_ssim as ssim
from collections import OrderedDict
import matplotlib.pyplot as plt
class AverageMeter(object):
    """
    Computes and store the average the current value
    """
    def __init__(self):
        self.val = 0  # current value 
        self.avg = 0  # avage value
        self.sum = 0  
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def modelparallel(model):
    '''
    To make the model parallel 
    '''
    if isinstance(model, OrderedDict):
        print('The model is a OrderedDict')
    elif isinstance(model, torch.nn.Module):
        print('The model is a nn.Module')
    else:
        raise Exception('Not support the model')

def modeldist(model):
    '''
    To make the model in dist
    '''
    assert torch.distributed.is_initialized(), 'Not init the dist'
    if isinstance(model, OrderedDict):
        print('The model is OrderedDict')
    elif isinstance(model, torch.nn.Module):
        print('The model is nn.Module')
    else:
        raise Exception('Not support the model')

def grid_crop(bottom, bbox, object_size=(64,64)):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    # args:
    # bbox: [N, 4] = [[x1,x2,y1,y2], ....]   
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = bbox.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1) # C
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = int(rois.size(0) / batch_size)
    x1 = bbox[:, 0::4] 
    y1 = bbox[:, 1::4] 
    x2 = bbox[:, 2::4] 
    y2 = bbox[:, 3::4] 

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    # zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    zero = rois.new_zeros(bbox.size(0), 1)
    # import ipdb; ipdb.set_trace()
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)
    # import ipdb; ipdb.set_trace()
    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, object_size[0], object_size[1])))
    # import ipdb; ipdb.set_trace()
    bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W).contiguous().view(-1, D, H, W)
    crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def multi_obj_grid_crop(bottom, bbox, object_size=(64,64)):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    # args:
    # bbox: [N, 4] = [[x1,x2,y1,y2], ....]   N=object number
    # bottom: [1, C, H , W] 
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    # import ipdb; ipdb.set_trace()
    bottom = bottom.repeat(bbox.size(0), 1, 1, 1)
    rois = bbox.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1) # C
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = int(rois.size(0) / batch_size)
    x1 = bbox[:, 0::4] 
    y1 = bbox[:, 1::4] 
    x2 = bbox[:, 2::4] 
    y2 = bbox[:, 3::4] 

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    # zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    zero = rois.new_zeros(bbox.size(0), 1)
    # import ipdb; ipdb.set_trace()
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)
    # import ipdb; ipdb.set_trace()
    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, object_size[0], object_size[1])))
    # import ipdb; ipdb.set_trace()
    bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W).contiguous().view(-1, D, H, W)
    crops = F.grid_sample(bottom, grid)
    
    return crops, grid

# def image_gradient(x):
#     '''
#     Args:
#         x: the image with bs. [bs,1,h,w]
#     Returns:
#         dx: [bs,1,h,w] the gradient on x-axis
#         dy: [bs,1,h,w] the gradient on y-axis
#     '''
#     # tf.image.image_gradients(image) in tensorflow
#     h_x = x.size()[-2]
#     w_x = x.size()[-1]
#     # gradient step=1
#     l = x
#     r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
#     t = x
#     b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

#     dx, dy = torch.abs(r - l), torch.abs(b - t)
#     # dx will always have zeros in the last column, r-l
#     # dy will always have zeros in the last row,    b-t
#     dx[:, :, :, -1] = 0
#     dy[:, :, -1, :] = 0

#     return dx, dy

def image_gradient(image):
    '''
    Args:
        x: the image with bs. [bs,c,h,w]
    Returns:
        dx: [bs,c,h,w] the gradient on x-axis
        dy: [bs,c,h,w] the gradient on y-axis
    '''
    channel = image.size()[1]
    pos = torch.eye(channel, device=image.device, dtype=torch.float32)
    neg = (-1 * pos) + 1 -1
    filter_x = torch.unsqueeze(torch.stack([neg, pos], dim=0), 0).permute(2,3,0,1)
    filter_y = torch.stack([torch.unsqueeze(neg, dim=0), torch.unsqueeze(pos, dim=0)]).permute(2,3,0,1)
    image_x = torch.nn.ZeroPad2d((1,0,0,0))(image)
    image_y = torch.nn.ZeroPad2d((0,0,1,0))(image)
    dx = torch.abs(torch.nn.functional.conv2d(image_x, filter_x))
    dy = torch.abs(torch.nn.functional.conv2d(image_y, filter_y))

    return dx, dy

def frame_gradient(x):
    '''
    The input is a video clip and get the gradient on the image 
    Args:
        x: the video with bs. [bs, d, 1, h, w]
    
    Returns:
        dx: [bs, d, 1, h, w] the gradient on x-axis
        dy: [bs, d, 1, h, w] the gradient on y-axis
    '''
    video_length = x.size(1)
    dx = list()
    dy = list()
    for i in range(video_length):
        temp_dx, temp_dy = image_gradient(x[:,i,:,:,:])
        # dx.append(temp_dx.unsqueeze_(1))
        dx.append(temp_dx)
        dy.append(temp_dy)
    
    dx = torch.stack(dx, dim=1)
    dy = torch.stack(dy, dim=1)
    gradient = dx + dy
    return dx, dy, gradient

def flow_batch_estimate(flow_model, tensor_batch, scale=64.0, output_format='xym', optical_size=None, normalize=True, mean=[], std=[]):
    '''
    output_format:
        general: u,v
        xym: u,v,mag
        hsv:
        rgb:
    '''
    flow_model.eval()
    # the tensor have been changed into [0.0, 1.0] and [b,c,h,w]
    if optical_size is not None:
        intHeight = optical_size[0]
        intWidth = optical_size[1]
    else:
        intHeight = tensor_batch.size(2)
        intWidth = tensor_batch.size(3)

    tensorFirst = tensor_batch[:, :3, :, :]
    tensorSecond = tensor_batch[:, 3:, :, :]
    # import ipdb; ipdb.set_trace()
    # intPreprocessedWidth = int(math.floor(math.ceil(intWidth / scale)*scale))
    # intPreprocessedHeight = int(math.floor(math.ceil(intHeight / scale)*scale))

    # tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    # tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    # # import ipdb; ipdb.set_trace()
    # new_temp = torch.stack([tensorPreprocessedFirst, tensorPreprocessedSecond], dim=2)
    # # result = self.F(tensorPreprocessedFirst, tensorPreprocessedSecond)
    # result = flow_model(new_temp)
    new_temp = torch.stack([tensorFirst, tensorSecond], dim=2)
    result = flow_model(new_temp)
    # optical_flow = torch.nn.functional.interpolate(input=result,size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    # optical_flow = torch.nn.functional.interpolate(input=result,size=(tensor_batch.size(2), tensor_batch.size(3)), mode='bilinear', align_corners=False)

    # optical_flow[:,0,:,:] *= float(intWidth) / float(intPreprocessedWidth)
    # optical_flow[:,1,:,:] *= float(intHeight) / float(intPreprocessedHeight)

    # temp = optical_flow.detach().cpu().permute(0,2,3,1).numpy()
    temp = result.detach().cpu().permute(0,2,3,1).numpy()
    # import ipdb; ipdb.set_trace()
    temp_list = list()
    for i in range(temp.shape[0]):
        np_image = flow2img(temp[i], output_format)
        temp_image = torch.from_numpy(np_image.transpose((2, 0, 1)))
        if normalize:
            temp_image = temp_image / 255.0
            if (len(mean)!=0) and (len(std) != 0):
                temp_image = tf.normalize(temp_image, mean=mean, std=std)
        temp_list.append(temp_image)
    optical_flow_image = torch.stack(temp_list, 0).cuda()
    # return optical_flow_image, optical_flow
    return optical_flow_image, result


def tsne_vis(feature, feature_labels, vis_path):
    feature_np = feature.detach().cpu().numpy()
    feature_embeddings = TSNE().fit_transform(feature_np)
    vis_x = feature_embeddings[:, 0]
    vis_y = feature_embeddings[:, 1]
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(vis_x, vis_y, c=feature_labels, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(vis_path)


def training_vis_images(vis_objects, writer, global_step, normalize=True, mean=None, std=None):

    # def verse_normalize(image_tensor):
    #     std = self.trainer.config.ARGUMENT.val.normal.std
    #     mean = self.trainer.config.ARGUMENT.val.normal.mean
    #     if len(mean) == 0 and len(std) == 0:
    #         return image_tensor
    #     else:
    #         for i in range(len(std)):
    #             image_tensor[:,i,:,:] = image_tensor[:,i,:,:] * std[i] + mean[i]
    #         return image_tensor
    
    vis_keys = list(vis_objects.keys())
    # import ipdb; ipdb.set_trace()
    for vis_key in vis_keys:
        temp = vis_objects[vis_key]
        if len(temp.shape) == 5:
            batch_num = temp.shape[0]
            for i in range(batch_num):
                temp_one = temp[i,:,:,:,:].permute(1,0,2,3)
                writer.add_images(vis_key+f'video_{i}in{batch_num}', temp_one, global_step)
        elif len(temp.shape) == 4:
            writer.add_images(vis_key+'_batch', temp, global_step)
        elif len(temp.shape) == 3:
            writer.add_image(vis_key+'_image', temp, global_step)


def get_batch_dets(det_model, batch_image):
        """
        Use the detecron2
        """
        image_list = list()
        batch_size = batch_image.size(0)
        images = torch.chunk(batch_image, batch_size, dim=0)
        for image in images:
            image_list.append({"image":image.squeeze_(0).mul(255).byte()[[2,0,1],:,:]})
        outputs = det_model(image_list)
        
        bboxs = []
        frame_objects = OrderedDict()
        max_objects = 0
        min_objects = 1000
        for frame_id, out in enumerate(outputs):
            temp = out['instances'].pred_boxes.tensor.detach()
            temp.requires_grad = False
            frame_objects[frame_id] = temp.size(0)
            if frame_objects[frame_id] > max_objects:
                max_objects = frame_objects[frame_id]
            if frame_objects[frame_id] < min_objects:
                min_objects = frame_objects[frame_id]
            bboxs.append(temp)
        
        return bboxs



if __name__ == '__main__':
    path = '/export/home/chengyh/data/COCO/MSCOCO/images/test2017/000000000019.jpg'
    from PIL import Image
    import torchvision.transforms.functional as tf
    pil_image = Image.open(path)
    tensor_image = tf.to_tensor(pil_image).unsqueeze(0)
    dx, dy = image_gradient(tensor_image)
    dx_image = tf.to_pil_image(dx.squeeze(0).cpu())
    dy_image = tf.to_pil_image(dy.squeeze(0).cpu())
    import ipdb; ipdb.set_trace()
