"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from collections import OrderedDict
import imgaug.augmenters as iaa
import logging
logger = logging.getLogger(__name__)

class AugmentAPI(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._default_transform_train = iaa.Sequential([iaa.Identity()])
        self._default_transform_val = iaa.Sequential([iaa.Identity()])
        self.normalize = True
        self._use_default_train = False
        self._use_default_val = False
        self.train_aug_cfg = self.cfg.get('ARGUMENT')['train']
        self.val_aug_cfg = self.cfg.get('ARGUMENT')['val']

    def build(self):
        aug_dict = OrderedDict()

        if self.train_aug_cfg.use:
            del self.train_aug_cfg['use']
            train_aug = self._compose_transforms(self.train_aug_cfg, 'train')
        else:
            train_aug = None
        if self.val_aug_cfg.use:
            del self.val_aug_cfg['use']
            val_aug = self._compose_transforms(self.val_aug_cfg, 'val')
        else:
            val_aug = None
        
        aug_dict['train_aug'] = train_aug
        aug_dict['val_aug'] = val_aug

        return aug_dict

    def _compose_transforms(self, transforms_cfg, phase):
        aug_functions = list()
        used_transforms = []
        for transform_name in transforms_cfg.keys():
            if transform_name == 'resize' and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Resize({"height": transforms_cfg[transform_name].height, "width": transforms_cfg[transform_name].width}, interpolation='linear', name='resize'))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'fliplr'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Fliplr(transforms_cfg[transform_name].p, name='fliplr'))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'flipud'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Flipud(transforms_cfg[transform_name].p, name='flipud'))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'rotate'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Rotate(transforms_cfg[transform_name].degrees, name='rotate'))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'JpegCompression'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.JpegCompression(compression=(transforms_cfg[transform_name].low, transforms_cfg[transform_name].high)))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'GaussianBlur'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.GaussianBlur(sigma=(transforms_cfg[transform_name].low, transforms_cfg[transform_name].high)))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            elif transform_name == 'CropToFixedSize'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.CropToFixedSize(width=transforms_cfg[transform_name].width, height=transforms_cfg[transform_name].height, position=transforms_cfg[transform_name].position))
                # logger.info(f'{transform_name} is used in {phase}')
                used_transforms.append(str(transform_name))
                continue
            else:
                # logger.info(f'{transform_name} is not support in augment build')
                # aug_functions.append(iaa.Noop())
                continue
        if len(aug_functions) == 0:
            logger.info(f'Not use any transforms in {phase}')
            aug_functions.append(iaa.Noop())
        else:
            # import ipdb; ipdb.set_trace()
            message = ','.join(used_transforms)
            logger.info(f'{message} is used in {phase}')
        iaa_seq = iaa.Sequential(aug_functions, name=f'{self.cfg.DATASET.name}_{phase}_iaa_seq')
        
        return iaa_seq
    

