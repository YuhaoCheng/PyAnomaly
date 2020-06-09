from .build.abstract import LossBuilder

class LossAPI(LossBuilder):
    def __init__(self, cfg, logger):
        super(LossAPI, self).__init__(cfg)
        self.logger = logger
    def __call__(self):
        loss_dict, loss_lamada = super(LossAPI, self).build()
        self.logger.info(f'the loss names:{loss_dict.keys()}')
        self.logger.info(f'the loss lamada:{loss_lamada}')
        return loss_dict, loss_lamada