import torch
import torch.nn as nn
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo

from pyanomaly.networks.model_registry import AUX_ARCH_REGISTRY

@AUX_ARCH_REGISTRY.register()
class Detector(nn.Module):
    """The detector module.
    """
    def __init__(self, cfg):
        super(Detector, self).__init__()
        auxiliary_cfg = cfg.MODEL.auxiliary.detector
        detector_cfg = get_cfg()
        file_name = auxiliary_cfg.detector_config
        detector_cfg.merge_from_file(model_zoo.get_config_file(file_name))
        detector_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        detector_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8

        self.det_model = build_model(detector_cfg)
        
        DetectionCheckpointer(self.det_model).load(auxiliary_cfg.detector_model_path)
        self.det_model.train(False)
    
    def forward(self, input):
        return self.det_model(input)
    
