# from .losscatalog import LossCatalog
# from ..functions.basic_loss import get_loss

# def register_loss_functions():
#     LossCatalog.register('mse', lambda cfg: get_loss('mse', cfg))
#     LossCatalog.register('cross', lambda cfg: get_loss('cross', cfg))
#     LossCatalog.register('g_adverserial_loss', lambda cfg: get_loss('g_adv', cfg))
#     LossCatalog.register('d_adverserial_loss', lambda cfg: get_loss('d_adv', cfg))
#     LossCatalog.register('opticalflow_loss', lambda cfg: get_loss('flow_loss', cfg))
#     LossCatalog.register('gradient_loss', lambda cfg: get_loss('gd_loss', cfg))
#     LossCatalog.register('intentsity_loss', lambda cfg: get_loss('int_loss', cfg))
#     LossCatalog.register('amc_d_adverserial_loss_1', lambda cfg: get_loss('amc_d_adverserial_loss_1', cfg))
#     LossCatalog.register('amc_d_adverserial_loss_2', lambda cfg: get_loss('amc_d_adverserial_loss_2', cfg))
#     LossCatalog.register('amc_g_adverserial_loss', lambda cfg: get_loss('amc_g_adverserial_loss', cfg))
#     LossCatalog.register('A_loss', lambda cfg: get_loss('A_loss', cfg))
#     LossCatalog.register('B_loss', lambda cfg: get_loss('B_loss', cfg))
#     LossCatalog.register('C_loss', lambda cfg: get_loss('C_loss', cfg))


# register_loss_functions()