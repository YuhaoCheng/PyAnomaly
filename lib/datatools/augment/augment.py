import torchvision.transforms as T
import torchvision.transforms.functional as tf
from ..abstract.aug_builder import UtilsBuilder
from collections import OrderedDict
import imgaug.augmenters as iaa


class AugmentBuilder(UtilsBuilder):
    def __init__(self, cfg, logger):
        super(AugmentBuilder, self).__init__(cfg)
        self.logger = logger
        self._default_transform_train = iaa.Sequential([iaa.Identity()])
        self._default_transform_val = iaa.Sequential([iaa.Identity()])
        self.normalize = True
        self._use_default_train = False
        self._use_default_val = False

    def _get_node(self, flag):
        self.flag = flag
        if flag == 'train':
            if not self.cfg.ARGUMENT.train.use:
                self._use_default_train = True
                self.logger.info(f'Not use the augment in {flag}')
            else:
                self._node = OrderedDict(self.cfg.ARGUMENT.train)
        elif flag == 'val':
            if not self.cfg.ARGUMENT.val.use:
                self._use_default_val = True
                self.logger.info(f'Not use the augment in {flag}')
            else:
                self._node = OrderedDict(self.cfg.ARGUMENT.val)
        else:
            raise Exception(f'Wrong flag name:{flag}')

    def _build(self):
        # not use the augment
        if self._use_default_train:
            self._use_default_train = False # change to the init value, in order multi __call__
            return self._default_transform_train
        if self._use_default_val:
            self._use_default_val = True # change to the init value, in order multi __call__
            return self._default_transform_val
        
        # used augments
        self._transforms = OrderedDict()
        del self._node['use']
        for name in self._node.keys():
            if self._node[name].use:
                self._transforms[name] = self._node[name]
        final_transform = self._compose_transforms()

        self.logger.info(f'Used IAA in {self.flag}are:{final_transform}')

        return final_transform

    def _compose_transforms(self):
        self.aug_functions = list()
        for transform_name in self._transforms.keys():
            if transform_name == 'resize':
                self.aug_functions.append(iaa.Resize({"height": self._transforms[transform_name].height, "width": self._transforms[transform_name].width}, name='resize'))
                continue
            elif transform_name == 'fliplr':
                self.aug_functions.append(iaa.Fliplr(self._transforms[transform_name].p, name='fliplr'))
                continue
            elif transform_name == 'flipud':
                self.aug_functions.append(iaa.Flipud(self._transforms[transform_name].p, name='flipud'))
                continue
            elif transform_name == 'rotate':
                self.aug_functions.append(iaa.Rotate(self._transforms[transform_name].degrees, name='rotate'))
                continue
            elif transform_name == 'JpegCompression':
                self.aug_functions.append(iaa.JpegCompression(compression=(self._transforms[transform_name].low, self._transforms[transform_name].high)))
                continue
            elif transform_name == 'GaussianBlur':
                self.aug_functions.append(iaa.GaussianBlur(sigma=(self._transforms[transform_name].low, self._transforms[transform_name].high)))
                continue
            elif transform_name == 'CropToFixedSize':
                self.aug_functions.append(iaa.CropToFixedSize(width=self._transforms[transform_name].width, height=self._transforms[transform_name].height, position=self._transforms[transform_name].position))
                continue
            else:
                self.logger.info(f'{transform_name} is not support in augment build')
                self.aug_functions.append(iaa.Noop())
                continue
        iaa_seq = iaa.Sequential(self.aug_functions, name=f'{self.cfg.DATASET.name}_{self.flag}_iaa_seq')
        
        return iaa_seq

