"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
    Registry for dataset classes
"""
DATASET_FACTORY_REGISTRY = Registry("DATASET_FACTORY")
DATASET_FACTORY_REGISTRY.__doc__ = """
    Registry for dataset factory classes
"""
EVAL_METHOD_REGISTRY = Registry("EVAL_METHOD")
EVAL_METHOD_REGISTRY.__doc__ = """
    Registry for eval method classes
"""