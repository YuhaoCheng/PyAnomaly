"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import os
import torch.nn as nn
import torch
import torchsnooper
from collections import OrderedDict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from ..model_registry import META_ARCH_REGISTRY
__all__ = ['CAE', 'get_model_ocae']

@META_ARCH_REGISTRY.register()
class CAE(nn.Module):
    def __init__(self, c_in):
        super(CAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=(3,3), padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, c_in, kernel_size=(3,3), padding=1)
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    # @torchsnooper.snoop()
    def forward(self,x):
        # import ipdb; ipdb.set_trace()
        x = (0.3 * x[:,0] + 0.59 * x[:,1] + 0.11 * x[:,2]).unsqueeze(1)
        latent_feature = self.encoder(x)
        output = self.decoder(latent_feature)
        return latent_feature, output, x

def get_model_ocae(cfg):
    model_dict = OrderedDict()
    model_dict['A'] = CAE(c_in=1)
    model_dict['B'] = CAE(c_in=1)
    model_dict['C'] = CAE(c_in=1)
    if cfg.MODEL.name == 'ocae':
        import detectron2
        from detectron2.engine import DefaultPredictor
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.modeling import build_model
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    else:
        raise Exception('Not the correct the model name')
    detector_cfg = get_cfg()
    file_name = cfg.MODEL.detector_config
    detector_cfg.merge_from_file(model_zoo.get_config_file(file_name))
    detector_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detector_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8
    det_model = build_model(detector_cfg)
    DetectionCheckpointer(det_model).load(cfg.MODEL.detector_model_path)
    det_model.train(False)
    model_dict['Detector'] = det_model

    model_dict['OVR'] = OneVsRestClassifier(LinearSVC(random_state = 0), n_jobs=16)

    return model_dict

