from .hookcatalog import HookCatalog
# from ..ae_baseline_hooks import get_ae_baseline_hooks
# from ..aicn_hooks import get_aicn_hooks
from ..amc_hooks import get_amc_hooks
from ..anopcn_hooks import get_anopcn_hooks
from ..anopred_hooks import get_anopred_hooks
from ..base import get_base_hooks
# from ..discriminative_hooks import get_discriminative_hooks
# from ..gan_baseline_hooks import get_gan_baseline_hooks
# from ..itae_hooks import get_itae_hooks
# from ..lsa_hooks import get_lsa_hooks
# from ..ltr_hooks import get_ltr_hooks
from ..memae_hooks import get_memae_hooks
# from ..mlad_hooks import get_mlad_hooks
# from ..mpen_rnn_hooks import get_mpen_rnn_hooks
from ..oc_hooks import get_oc_hooks
# from ..occ_hooks import get_occ_hooks
# from ..srnn_ae_hooks import get_srnn_ae_hooks
# from ..srnn_hooks import get_srnn_hooks
from ..stae_hooks import get_stae_hooks
# from ..twostage_hooks import get_twostage_hooks
# from ..unmasking_hooks import get_unmasking_hooks

from .build_hooks import HookAPI

__all__ = ['HookAPI']

def register_hooks():
    HookCatalog.register('base.VisScoreHook', lambda name:get_base_hooks(name))
    HookCatalog.register('base.TSNEHook', lambda name:get_base_hooks(name))
    HookCatalog.register('amc.AMCEvaluateHook', lambda name:get_amc_hooks(name))
    HookCatalog.register('anopcn.AnoPCNEvaluateHook', lambda name:get_anopcn_hooks(name))
    HookCatalog.register('anopred.AnoPredEvaluateHook', lambda name:get_anopred_hooks(name))
    HookCatalog.register('oc.ClusterHook', lambda name:get_oc_hooks(name))
    HookCatalog.register('oc.OCEvaluateHook', lambda name:get_oc_hooks(name))
    HookCatalog.register('stae.STAEEvaluateHook', lambda name:get_stae_hooks(name))
    HookCatalog.register('memae.MemAEEvaluateHook', lambda name:get_memae_hooks(name))

register_hooks()
