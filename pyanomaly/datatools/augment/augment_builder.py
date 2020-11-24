# import torchvision.transforms as T
# import torchvision.transforms.functional as tf
# from ..abstract.aug_builder import UtilsBuilder
from collections import OrderedDict
import imgaug.augmenters as iaa
import logging
logger = logging.getLogger(__name__)

# class AugmentBuilder(UtilsBuilder):
class AugmentAPI(object):
    def __init__(self, cfg):
        # super(AugmentBuilder, self).__init__(cfg)
        self.cfg = cfg
        # self.logger = logger
        self._default_transform_train = iaa.Sequential([iaa.Identity()])
        self._default_transform_val = iaa.Sequential([iaa.Identity()])
        self.normalize = True
        self._use_default_train = False
        self._use_default_val = False
        self.train_aug_cfg = self.cfg.get('ARGUMENT')['train']
        self.test_aug_cfg = self.cfg.get('ARGUMENT')['val']
        # import ipdb; ipdb.set_trace()

    # def _get_node(self, flag):
    #     self.flag = flag
    #     if flag == 'train':
    #         if not self.cfg.ARGUMENT.train.use:
    #             self._use_default_train = True
    #             # self.logger.info(f'Not use the augment in {flag}')
    #             logger.info(f'Not use the augment in {flag}')
    #         else:
    #             self._node = OrderedDict(self.cfg.ARGUMENT.train)
    #     elif flag == 'val':
    #         if not self.cfg.ARGUMENT.val.use:
    #             self._use_default_val = True
    #             logger.info(f'Not use the augment in {flag}')
    #         else:
    #             self._node = OrderedDict(self.cfg.ARGUMENT.val)
    #     else:
    #         raise Exception(f'Wrong flag name:{flag}')

    def build(self):
        # not use the augment
        # if self._use_default_train:
        #     self._use_default_train = False # change to the init value, in order multi __call__
        #     return self._default_transform_train
        # if self._use_default_val:
        #     self._use_default_val = True # change to the init value, in order multi __call__
        #     return self._default_transform_val
        
        # used augments
        # self._train_transforms = OrderedDict()
        # self._test_transforms = OrderedDict()
        aug_dict = OrderedDict()

        # del self._node['use']
        if self.train_aug_cfg.use:
            del self.train_aug_cfg['use']
            train_aug = self._compose_transforms(self.train_aug_cfg, 'train')
            # for name in  self.train_aug_cfg.keys():
            #     if  self.train_aug_cfg[name].use:
            #         self._transforms[name] = self._node[name]
        else:
            train_aug = None
        # final_transform = self._compose_transforms()
        if self.test_aug_cfg.use:
            del self.test_aug_cfg['use']
            test_aug = self._compose_transforms(self.test_aug_cfg, 'test')
            # for name in  self.train_aug_cfg.keys():
            #     if  self.train_aug_cfg[name].use:
            #         self._transforms[name] = self._node[name]
        else:
            test_aug = None
        # final_transform = self._compose_transforms()

        # logger.info(f'Used IAA in {self.flag}are:{final_transform}')
        aug_dict['train_aug'] = train_aug
        aug_dict['test_aug'] = test_aug

        return aug_dict

    def _compose_transforms(self, transforms_cfg, phase):
        aug_functions = list()
        for transform_name in transforms_cfg.keys():
            if transform_name == 'resize' and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Resize({"height": transforms_cfg[transform_name].height, "width": transforms_cfg[transform_name].width}, interpolation='linear', name='resize'))
                continue
            elif transform_name == 'fliplr'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Fliplr(transforms_cfg[transform_name].p, name='fliplr'))
                continue
            elif transform_name == 'flipud'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Flipud(transforms_cfg[transform_name].p, name='flipud'))
                continue
            elif transform_name == 'rotate'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.Rotate(transforms_cfg[transform_name].degrees, name='rotate'))
                continue
            elif transform_name == 'JpegCompression'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.JpegCompression(compression=(transforms_cfg[transform_name].low, transforms_cfg[transform_name].high)))
                continue
            elif transform_name == 'GaussianBlur'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.GaussianBlur(sigma=(transforms_cfg[transform_name].low, transforms_cfg[transform_name].high)))
                continue
            elif transform_name == 'CropToFixedSize'and transforms_cfg[transform_name].use:
                aug_functions.append(iaa.CropToFixedSize(width=transforms_cfg[transform_name].width, height=transforms_cfg[transform_name].height, position=transforms_cfg[transform_name].position))
                continue
            else:
                logger.info(f'{transform_name} is not support in augment build')
                aug_functions.append(iaa.Noop())
                continue
        iaa_seq = iaa.Sequential(aug_functions, name=f'{self.cfg.DATASET.name}_{phase}_iaa_seq')
        
        return iaa_seq
    
    # def _compose_transforms(self, transforms_cfg):
    #     self.aug_functions = list()
    #     for transform_name in self._transforms.keys():
    #         if transform_name == 'resize':
    #             self.aug_functions.append(iaa.Resize({"height": self._transforms[transform_name].height, "width": self._transforms[transform_name].width}, interpolation='linear', name='resize'))
    #             continue
    #         elif transform_name == 'fliplr':
    #             self.aug_functions.append(iaa.Fliplr(self._transforms[transform_name].p, name='fliplr'))
    #             continue
    #         elif transform_name == 'flipud':
    #             self.aug_functions.append(iaa.Flipud(self._transforms[transform_name].p, name='flipud'))
    #             continue
    #         elif transform_name == 'rotate':
    #             self.aug_functions.append(iaa.Rotate(self._transforms[transform_name].degrees, name='rotate'))
    #             continue
    #         elif transform_name == 'JpegCompression':
    #             self.aug_functions.append(iaa.JpegCompression(compression=(self._transforms[transform_name].low, self._transforms[transform_name].high)))
    #             continue
    #         elif transform_name == 'GaussianBlur':
    #             self.aug_functions.append(iaa.GaussianBlur(sigma=(self._transforms[transform_name].low, self._transforms[transform_name].high)))
    #             continue
    #         elif transform_name == 'CropToFixedSize':
    #             self.aug_functions.append(iaa.CropToFixedSize(width=self._transforms[transform_name].width, height=self._transforms[transform_name].height, position=self._transforms[transform_name].position))
    #             continue
    #         else:
    #             logger.info(f'{transform_name} is not support in augment build')
    #             self.aug_functions.append(iaa.Noop())
    #             continue
    #     iaa_seq = iaa.Sequential(self.aug_functions, name=f'{self.cfg.DATASET.name}_{self.flag}_iaa_seq')
        
    #     return iaa_seq
    


# class AugmentAPI(AugmentBuilder):
#     def __init__(self,cfg, logger):
#         super(AugmentAPI, self).__init__(cfg, logger)
    
#     def add(self, extra_aug):
#         '''
#         add the extra aug into the aug
#         '''
#         print('add the extra_aug')

#     def __call__(self, flag='train') -> dict:
#         super(AugmentAPI, self)._get_node(flag)
#         # import ipdb; ipdb.set_trace()
#         t = super(AugmentAPI, self)._build()
#         return t

