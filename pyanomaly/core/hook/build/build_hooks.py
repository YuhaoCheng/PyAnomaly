from .hookcatalog import HookCatalog

class HookAPI(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.train_hook_names = cfg.MODEL.hooks.train
        self.val_hook_names = cfg.MODEL.hooks.val
        # self.eval_hook_names = cfg.MODEL.eval_hooks
        self.logger = logger
    def __call__(self, mode):
        if mode == 'train':
            hook_names = self.train_hook_names
        elif mode == 'val':
            hook_names = self.val_hook_names
        else:
            raise Exception('Not support hook mode')
        self.logger.info(f'{mode}*********use hooks:{hook_names}**********')
        hooks = []
        for name in hook_names:
            # prefix = name.split('.')[0]
            hook_name = name.split('.')[1]
            temp = HookCatalog.get(name, hook_name)
            hooks.append(temp)
        self.logger.info(f'build:{hooks}')

        return hooks

