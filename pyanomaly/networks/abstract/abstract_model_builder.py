from .modelcatalog import ModelCatalog
class ModelBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def build(self):
        model = ModelCatalog.get(self.cfg.MODEL.name, self.cfg)
        return model
        
