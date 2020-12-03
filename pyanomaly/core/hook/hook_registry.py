"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from fvcore.common.registry import Registry

HOOK_REGISTRY = Registry("HOOK")
HOOK_REGISTRY.__doc__ = """
    Registry for hook functional classes
"""
