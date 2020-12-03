"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
    Registry for loss function classes
"""
