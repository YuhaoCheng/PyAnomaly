from torch import mode
# from .abstract.abstract_model_builder import ModelBuilder
from collections import OrderedDict
import torch

# class ModelAPI(ModelBuilder):
#     def __init__(self, cfg, logger):
#         # self.model_name = cfg.MODEL.name
#         super(ModelAPI, self).__init__(cfg)
#         self.logger = logger

#     def __call__(self):
#         model = super(ModelAPI, self).build()
#         self.logger.info('The name is ' +f'\033[1;31m {self.cfg.MODEL.name} \033[0m')
#         if isinstance(model, OrderedDict):
#             self.logger.info('Make the model is the ' + '\033[1;31m OrderedDict \033[0m')
#             message = 'The model keys are: '
#             for key in model.keys():
#                 temp = f'\033[1;31m {key} \033[0m ,'
#                 message += temp
#             self.logger.info(message)
#         elif isinstance(model, torch.nn.Module):
#             self.logger.info('Make the model is the ' + ' \033[1:31m nn.Module \033[0m')
#         else:
#             raise Exception('No supprot model type')

#         return model
from .model_registry import *
from .meta import( 
    STAutoEncoderCov3D
)
class ModelAPI(object):
    MODEL_TYPE_1 = ['e2e']
    MODEL_TYPE_2 = ['me2e', 'ae2e', 'ame2e']
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
        model_type = self.cfg.MODEL.type2
        if model_type in ModelAPI.MODEL_TYPE_1:
            model = OrderedDict()
            self.logger.info('Model Dict')
            # 2. get the model based on the registry
            _model_parts = list(model_parts[i:i+2] for i in range(0, len(model_parts), 2))
            for couple in _model_parts:
                model_dict_key = couple[0]
                if model_dict_key == 'auxiliary':
                    model_dict_value = AUX_ARCH_REGISTRY.get(model_dict_key)(self.cfg)
                elif model_dict_key == 'meta':
                    model_dict_value = META_ARCH_REGISTRY.get(model_dict_key)(self.cfg)
                elif model_dict_key == 'base':
                    model_dict_value = BASE_ARCH_REGISTRY.get(model_dict_key)(self.cfg)
                else:
                    raise Exception('Wrong model in line62')
                # 3. set the grad requirement
                # 4. get the final model 
                model[model_dict_key] = model_dict_value
        elif model_type in ModelAPI.MODEL_TYPE_2:
            self.logger.info('nn.Module')
            if model_parts is not None:
                raise Exception('Wrong config in model.parts')
            model = META_ARCH_REGISTRY.get(model_name)(self.cfg)
            # return model
        else:
            raise Exception(f'Not support Model Type, we only support: {ModelAPI.MODEL_TYPE_1} and {ModelAPI.MODEL_TYPE_2}')
        
        return model
