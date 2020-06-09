from collections import OrderedDict
# from .losscatalog import LossCatalog
from lib.loss.functions.basic_loss import get_basic_loss
from lib.loss.functions.expand_loss import get_expand_loss, EXPAND_LOSS
# class LossBuilder_old(object):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.loss_list = self.cfg.TRAIN.loss
#         self.loss_coefficient_list = self.cfg.TRAIN.loss_coefficients
#         assert len(self.loss_list) == len(self.loss_coefficient_list), 'The lengths of the loss and the coefficient are not equal!!'

#     def build(self):
#         loss_dict = OrderedDict()
#         loss_coefficient_dict = OrderedDict()
#         for index,loss_name in enumerate(self.loss_list):
#             if not loss_name in LossCatalog._REGISTERED.keys():
#                 raise Exception(f'The loss function named: {loss_name} is not supported!')
#             else:
#                 loss_dict[loss_name] = LossCatalog.get(loss_name, self.cfg)
#                 loss_coefficient_dict[loss_name] = self.loss_coefficient_list[index]

#         return loss_dict, loss_coefficient_dict

class LossBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_list = self.cfg.TRAIN.loss
        self.loss_coefficient_list = self.cfg.TRAIN.loss_coefficients
        assert len(self.loss_list) == len(self.loss_coefficient_list), 'The lengths of the loss and the coefficient are not equal!!'
    
    def _get_loss(self, loss_name, cfg):
        if loss_name in EXPAND_LOSS:
            return get_expand_loss(loss_name, cfg)
        else:
            return get_basic_loss(loss_name, cfg)

    def build(self):
        loss_dict = OrderedDict()
        loss_coefficient_dict = OrderedDict()
        for index,loss_name in enumerate(self.loss_list):
            loss_dict[loss_name] = self._get_loss(loss_name, self.cfg)
            loss_coefficient_dict[loss_name] = self.loss_coefficient_list[index]

        return loss_dict, loss_coefficient_dict