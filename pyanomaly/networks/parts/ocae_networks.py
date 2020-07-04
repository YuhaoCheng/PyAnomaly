import os
import torch.nn as nn
import torch
import torchsnooper
from collections import OrderedDict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class CAE(nn.Module):
    def __init__(self):
        super(CAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, kernel_size=(3,3),padding=1,stride=1),
            nn.MaxPool2d(2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=(3,3), padding=1)
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    # @torchsnooper.snoop()
    def forward(self,x):
        x = (0.3 * x[:,0] + 0.59 * x[:,1] + 0.11 * x[:,2]).unsqueeze(1)
        latent_feature = self.encoder(x)
        # import ipdb; ipdb.set_trace()
        x = self.decoder(latent_feature)
        return latent_feature, x

def get_model_ocae(cfg):
    model_dict = OrderedDict()
    model_dict['A'] = CAE()
    model_dict['B'] = CAE()
    model_dict['C'] = CAE()
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

