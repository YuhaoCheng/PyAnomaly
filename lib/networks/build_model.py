from .build.abstract import ModelBuilder
from collections import OrderedDict
import torch
from colorama import init,Fore
init(autoreset=True)

class ModelAPI(ModelBuilder):
    def __init__(self, cfg, logger):
        # self.model_name = cfg.MODEL.name
        super(ModelAPI, self).__init__(cfg)
        self.logger = logger

    def __call__(self):
        model = super(ModelAPI, self).build()
        self.logger.info('the name is' +Fore.RED +f'{self.cfg.MODEL.name}')
        if isinstance(model, OrderedDict):
            self.logger.info('Make the model is the'+ Fore.RED + ' OrderedDict')
            message = 'The model keys are: '
            for key in model.keys():
                temp = Fore.RED + str(key) + Fore.GREEN + ','
                message += temp
            self.logger.info(message)
        elif isinstance(model, torch.nn. Module):
            self.logger.info('Make the model is the' + Fore.RED + 'nn.Module')
        else:
            raise Exception('No supprot model type')

        return model
