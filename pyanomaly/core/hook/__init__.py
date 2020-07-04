from .abstract.hookcatalog import HookCatalog
from .amc_hooks import get_amc_hooks
from .anopcn_hooks import get_anopcn_hooks
from .anopred_hooks import get_anopred_hooks
from .base import get_base_hooks
from .memae_hooks import get_memae_hooks
from .ocae_hooks import get_ocae_hooks
from .stae_hooks import get_stae_hooks

from .hooks_api import HookAPI

__all__ = ['HookAPI']

def register_hooks():
    HookCatalog.register('base.VisScoreHook', lambda name:get_base_hooks(name))
    HookCatalog.register('base.TSNEHook', lambda name:get_base_hooks(name))
    HookCatalog.register('amc.AMCEvaluateHook', lambda name:get_amc_hooks(name))
    HookCatalog.register('anopcn.AnoPCNEvaluateHook', lambda name:get_anopcn_hooks(name))
    HookCatalog.register('anopred.AnoPredEvaluateHook', lambda name:get_anopred_hooks(name))
    HookCatalog.register('oc.ClusterHook', lambda name:get_ocae_hooks(name))
    HookCatalog.register('oc.OCEvaluateHook', lambda name:get_ocae_hooks(name))
    HookCatalog.register('stae.STAEEvaluateHook', lambda name:get_stae_hooks(name))
    HookCatalog.register('memae.MemAEEvaluateHook', lambda name:get_memae_hooks(name))

register_hooks()
