from torch import mode
from collections import OrderedDict
import torch

from .model_registry import *
from .meta import( 
    STAutoEncoderCov3D,
    AMCGenerator,
    AMCDiscriminiator,
    AnoPcn,
    AnoPredGeneratorUnet,
    AutoEncoderCov3DMem,
    CAE
)

class ModelAPI(object):
    # MODEL_TYPE_1 = []
    MODEL_TYPE = ['e2e', 'me2e', 'ae2e', 'ame2e']
    def __init__(self, cfg, logger):
        # self.model_name = cfg.MODEL.name
        # super(ModelAPI, self).__init__(cfg)
        self.cfg = cfg
        self.logger = logger

    def __call__(self):
        # model = super(ModelAPI, self).build()
        # 1. Decide the model type: a. trainable one model(e2e); b. trainable multi models(me2e); c. trainable one model + auxiliary(ae2e); d. trainable multi models + + auxiliary(ame2e)
        self.logger.info('The name is ' +f'\033[1;31m {self.cfg.MODEL.name} \033[0m')
        self.logger.info('The model type is' + f'\033[1;31m {self.cfg.MODEL.type} \033[0m')
        model_name = self.cfg.MODEL.name
        model_parts = self.cfg.MODEL.parts
        # model_type = self.cfg.MODEL.type2
        model_type = self.cfg.MODEL.type
        if model_type in ModelAPI.MODEL_TYPE:
            model = OrderedDict()
            self.logger.info('Model Dict')
            # 2. get the model based on the registry
            _model_parts = list(model_parts[i:i+2] for i in range(0, len(model_parts), 2))
            for couple in _model_parts:
                model_dict_key = couple[0].split('_')
                if model_dict_key[0] == 'auxiliary':
                    model_dict_value = AUX_ARCH_REGISTRY.get(couple[1])(self.cfg)
                elif model_dict_key[0] == 'meta':
                    model_dict_value = META_ARCH_REGISTRY.get(couple[1])(self.cfg)
                elif model_dict_key[0] == 'base':
                    model_dict_value = BASE_ARCH_REGISTRY.get(couple[1])(self.cfg)
                else:
                    raise Exception('Wrong model in line62')
                # 3. set the grad requirement  --move to the Trainer, for the convenience
                # 4. get the final model 
                model[model_dict_key[1]] = model_dict_value
        # elif model_type in ModelAPI.MODEL_TYPE_2:
        #     model = OrderedDict()
        #     self.logger.info('nn.Module')
        #     if model_parts[0] is not None:
        #         raise Exception('Wrong config in model.parts')
        #     # 2. get the model based on the registry
        #     model = META_ARCH_REGISTRY.get(model_parts[1])(self.cfg)
        else:
            raise Exception(f'Not support Model Type, we only support: {ModelAPI.MODEL_TYPE}')
        
        return model
