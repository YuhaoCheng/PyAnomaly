from .modelcatalog import ModelCatalog

# all the method to get the model named as 'get_model_xxxx()'
from ..parts.anopred_networks import get_model_ano_pred
from ..parts.anopcn_networks import get_model_anopcn
from ..parts.amc_networks import get_model_amc
from ..parts.ocae_networks import get_model_ocae
from ..parts.stae_networks import get_model_stae
from ..parts.ae_baseline_netowrks import get_model_ae_baseline
from ..parts.gan_baseline_networks import get_model_gan_baseline
from ..parts.srnn_networks import get_model_srnn
from ..parts.srnn_ae_networks import get_model_srnn_ae
from ..parts.occ_networks import get_model_occ
from ..parts.mpen_rnn_networks import get_model_mpen_rnn
from ..parts.mlad_networks import get_model_mlad
from ..parts.memae_networks import get_model_memae
from ..parts.ltr_networks import get_model_ltr
from ..parts.lsa_networks import get_model_lsa
from ..parts.itae_networks import get_model_itae
from ..parts.aicn_networks import get_model_aicn

def register_models():
    ModelCatalog.register('anopred', lambda cfg: get_model_ano_pred(cfg))
    ModelCatalog.register('anopcn', lambda cfg: get_model_anopcn(cfg))
    ModelCatalog.register('amc', lambda cfg: get_model_amc(cfg))
    ModelCatalog.register('ocae', lambda cfg: get_model_ocae(cfg))
    ModelCatalog.register('ae_baseline', lambda cfg: get_model_ae_baseline(cfg))
    ModelCatalog.register('stae', lambda cfg: get_model_stae(cfg))
    ModelCatalog.register('gan_baseline', lambda cfg: get_model_gan_baseline(cfg))
    ModelCatalog.register('srnn', lambda cfg: get_model_srnn(cfg))
    ModelCatalog.register('srnn_ae', lambda cfg: get_model_srnn_ae(cfg))
    ModelCatalog.register('occ', lambda cfg: get_model_occ(cfg))
    ModelCatalog.register('mpen_rnn', lambda cfg: get_model_mpen_rnn(cfg))
    ModelCatalog.register('mlad', lambda cfg: get_model_mlad(cfg))
    ModelCatalog.register('memae', lambda cfg: get_model_memae(cfg))
    ModelCatalog.register('ltr', lambda cfg: get_model_ltr(cfg))
    ModelCatalog.register('lsa', lambda cfg: get_model_lsa(cfg))
    ModelCatalog.register('itae', lambda cfg: get_model_itae(cfg))


register_models()