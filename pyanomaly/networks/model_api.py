from .abstract.abstract_model_builder import ModelBuilder
from collections import OrderedDict
import torch

class ModelAPI(ModelBuilder):
    def __init__(self, cfg, logger):
        # self.model_name = cfg.MODEL.name
        super(ModelAPI, self).__init__(cfg)
        self.logger = logger

    def __call__(self):
        model = super(ModelAPI, self).build()
        self.logger.info('The name is ' +f'\033[1;31m {self.cfg.MODEL.name} \033[0m')
        if isinstance(model, OrderedDict):
            self.logger.info('Make the model is the ' + '\033[1;31m OrderedDict \033[0m')
            message = 'The model keys are: '
            for key in model.keys():
                temp = f'\033[1;31m {key} \033[0m ,'
                message += temp
            self.logger.info(message)
        elif isinstance(model, torch.nn. Module):
            self.logger.info('Make the model is the ' + ' \033[1:31m nn.Module \033[0m')
        else:
            raise Exception('No supprot model type')

        return model
