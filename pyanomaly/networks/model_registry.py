"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from fvcore.common.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
    Registry for meta-architectures, i.e. the whole model.
    The registered object will be called with `obj(cfg)`
    and expected to return a `nn.Module` object.
"""

BASE_ARCH_REGISTRY = Registry("BASE_ARCH")
BASE_ARCH_REGISTRY.__doc__ = """
    Registry for base-architectures, i.e. the backbone model.
    The registered object will be called with `obj(cfg)`
    and expected to return a `nn.Module` object.
"""

AUX_ARCH_REGISTRY = Registry("AUX_ARCH")
AUX_ARCH_REGISTRY.__doc__="""
    Registry for auxiliary-architectures, i.e. the backbone model.
    The registered object will be called with `obj(cfg)`
    and expected to return a `nn.Module` object.
"""