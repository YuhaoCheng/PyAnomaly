from .datacatalog import DatasetCatalog
from .avenue_ped_shanghai import get_avenue_ped_shanghai

def register_builtin_dataset():
    DatasetCatalog.register('avenue', lambda cfg, flag, aug: get_avenue_ped_shanghai(cfg, flag, aug))
    DatasetCatalog.register('shanghai', lambda cfg, flag, aug: get_avenue_ped_shanghai(cfg, flag, aug))
    DatasetCatalog.register('ped2', lambda cfg, flag, aug: get_avenue_ped_shanghai(cfg, flag, aug))
    


register_builtin_dataset()