"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from fvcore.common.config import CfgNode as CN

__all__ = ['update_config'] 
"""
This the default configuration of the whole prohect.
"""
config = CN()
config.DESCROPTION = 'This the description of the configuration defaults. If you have some information related to the configuration file, please fullfil this item'
# configure the system related matters, such as gpus, cudnn and so on
config.SYSTEM = CN()
# config.SYSTEM.multigpus = False  # to determine whether use multi gpus to train or test(data parallel) # will be deprecated in the future 
# config.SYSTEM.num_gpus = 1    # decide the num_gpus  # will be deprecated in the future 
# Configure the number of gpus, and whether use the  parallell training 
config.SYSTEM.gpus = [0]

config.SYSTEM.cudnn = CN()
config.SYSTEM.cudnn.benchmark = True
config.SYSTEM.cudnn.deterministic = False
config.SYSTEM.cudnn.enable = True

# about use the distributed
config.SYSTEM.distributed = CN()
config.SYSTEM.distributed.use = False
# configure the log things
config.LOG = CN()
config.LOG.log_output_dir = './output/log' # log 
config.LOG.tb_output_dir = './output/tensorboard' # tensorboard log output dir
config.LOG.vis_dir = './output/vis'

# configure the dataset 
config.DATASET = CN()
config.DATASET.factory = 'VideoAnomalyDatasetFactory'
config.DATASET.num_workers = 16
config.DATASET.name = ''
config.DATASET.seed = 2020
config.DATASET.read_format = 'opencv'
config.DATASET.image_format = 'jpg'
config.DATASET.channel_num = 3 # 1: grayscale image | 2: optical flow | 3: RGB or other 3 channel image
config.DATASET.channel_name = 'rgb' # 'gray' | 'uv' | 'rgb' | ....
config.DATASET.optical_format = 'Y' # the format of the optical 
config.DATASET.optical_size = [384, 512] # the size of image before estimating the optical flow, H*W
config.DATASET.train = CN()
config.DATASET.train.data_path = ''
config.DATASET.train.clip_length = 5 # the total clip length, including frames not be sampled
config.DATASET.train.sampled_clip_length = 5 # the real used frame, most time it equals to clip_length
config.DATASET.train.frame_step = 1  # frame sample frequency
config.DATASET.train.clip_step = 1   # clip sample frequency
config.DATASET.train.gt_path = ''   # the path of the label file, not containing the name of the label file such as 'keypoints.json'
config.DATASET.train.execute_test = False   # Testing the model on the train data
config.DATASET.val = CN()
config.DATASET.val.data_path = ''
config.DATASET.val.clip_length = 5
config.DATASET.val.sampled_clip_length = 5
config.DATASET.val.frame_step = 1
config.DATASET.val.clip_step = 1
config.DATASET.val.gt_path = ''
config.DATASET.number_of_class = 1 # use in changing the label to one hot
config.DATASET.score_normalize = False
config.DATASET.score_type = 'normal' # 'normal' | 'abnormal'
config.DATASET.decidable_idx = 1 # The front decidable frame idx
config.DATASET.decidable_idx_back = 1 # The back decidable frame idx
config.DATASET.smooth = CN()
config.DATASET.smooth.guassian = False
config.DATASET.smooth.guassian_sigma = [10]
config.DATASET.mini_dataset = CN() 
config.DATASET.mini_dataset.samples = 2
config.DATASET.evaluate_function = CN()
config.DATASET.evaluate_function.name = ''
config.DATASET.evaluate_function.result_type = 'score'  # 'score' | 'psnr' | .....

# ****************configure the augment of the data*************************
config.AUGMENT = CN()
#========================Train Augment===================
config.AUGMENT.train = CN()
config.AUGMENT.train.use = False
config.AUGMENT.train.resize = CN()
config.AUGMENT.train.resize.use = False
config.AUGMENT.train.resize.height = 32
config.AUGMENT.train.resize.width = 32
config.AUGMENT.train.grayscale = CN()
config.AUGMENT.train.grayscale.use = False
config.AUGMENT.train.fliplr = CN()
config.AUGMENT.train.fliplr.use = False
config.AUGMENT.train.fliplr.p = 1.0
config.AUGMENT.train.flipud = CN()
config.AUGMENT.train.flipud.use = False
config.AUGMENT.train.flipud.p = 1.0
config.AUGMENT.train.rote = CN()
config.AUGMENT.train.rote.use = False
config.AUGMENT.train.rote.degrees = [0,0]
config.AUGMENT.train.JpegCompression = CN()
config.AUGMENT.train.JpegCompression.use = False
config.AUGMENT.train.JpegCompression.high = 100
config.AUGMENT.train.JpegCompression.low = 80
config.AUGMENT.train.GaussianBlur = CN()
config.AUGMENT.train.GaussianBlur.use = False
config.AUGMENT.train.GaussianBlur.high = 0.3
config.AUGMENT.train.GaussianBlur.low = 0.03
config.AUGMENT.train.CropToFixedSize = CN()
config.AUGMENT.train.CropToFixedSize.use = False
config.AUGMENT.train.CropToFixedSize.height = 256
config.AUGMENT.train.CropToFixedSize.width = 256
config.AUGMENT.train.CropToFixedSize.position = 'center' # uniform | normal | center | ...
#-------------------Normal------------------------
config.AUGMENT.train.normal = CN()
config.AUGMENT.train.normal.use = False
config.AUGMENT.train.normal.mean = [0.485, 0.456, 0.406]
config.AUGMENT.train.normal.std = [0.229, 0.224, 0.225]
#========================Val Augment===================
config.AUGMENT.val = CN()
config.AUGMENT.val.use = False
config.AUGMENT.val.resize = CN()
config.AUGMENT.val.resize.use = False
config.AUGMENT.val.resize.height = 32
config.AUGMENT.val.resize.width = 32
config.AUGMENT.val.grayscale = CN()
config.AUGMENT.val.grayscale.use = False
config.AUGMENT.val.fliplr = CN()
config.AUGMENT.val.fliplr.use = False
config.AUGMENT.val.fliplr.p = 1.0
config.AUGMENT.val.flipud = CN()
config.AUGMENT.val.flipud.use = False
config.AUGMENT.val.flipud.p = 1.0
config.AUGMENT.val.rote = CN()
config.AUGMENT.val.rote.use = False
config.AUGMENT.val.rote.degrees = [0,0]
config.AUGMENT.val.JpegCompression = CN()
config.AUGMENT.val.JpegCompression.use = False
config.AUGMENT.val.JpegCompression.high = 100
config.AUGMENT.val.JpegCompression.low = 80
config.AUGMENT.val.GaussianBlur = CN()
config.AUGMENT.val.GaussianBlur.use = False
config.AUGMENT.val.GaussianBlur.high = 0.3
config.AUGMENT.val.GaussianBlur.low = 0.03
config.AUGMENT.val.CropToFixedSize = CN()
config.AUGMENT.val.CropToFixedSize.use = False
config.AUGMENT.val.CropToFixedSize.height = 256
config.AUGMENT.val.CropToFixedSize.width = 256
config.AUGMENT.val.CropToFixedSize.position = 'center' # uniform | normal | center | ...
#-------------------Normal------------------------
config.AUGMENT.val.normal = CN()
config.AUGMENT.val.normal.use = False
config.AUGMENT.val.normal.mean = [0.485, 0.456, 0.406]
config.AUGMENT.val.normal.std = [0.229, 0.224, 0.225]
# *************************************************************************

# configure the model related things
config.MODEL = CN()
config.MODEL.name = ''   # the name of the network, such as resnet

# a. trainable one model(e2e); b. trainable multi models(me2e); c. trainable one model + auxiliary(ae2e); d. trainable multi models  + auxiliary(ame2e)
config.MODEL.type = 'e2e'

# must be even, the 0-th is name in dict, 1-th is the model named registered in the registry; e.g. ['meta_Backbone', 'Resnet18', 'meta_Head', 'ps'], 
#                                                                                                  ['meta_Generator', 'AMCGenerator', 'meta_Discriminiator', 'AMCDiscriminiator', 'auxiliary_Flownet', 'FlowNet2']
# The 0-th's format is 'registryName_NameInEngine', for example, 'meta_Generator' means the registry is 'meta' and the attribute named 'self.Generator' refers to it. 
# if the model type is e2e, the 0-th is ''(None)
config.MODEL.parts = ['','resnet18']

config.MODEL.hooks = CN()  # determine the hooks use in the training
config.MODEL.hooks.train = []  # determine the hooks use in the training
config.MODEL.hooks.val = []  # determine the hooks use in the training
config.MODEL.discriminator_channels = []
config.MODEL.pretrain_model = ''

# This part defines the auxiliary of the whole model, most of time these models are frozen
config.MODEL.auxiliary = CN()
config.MODEL.auxiliary.optical_flow = CN()
config.MODEL.auxiliary.optical_flow.require_grad = False
config.MODEL.auxiliary.optical_flow.rgb_max = 255.0
config.MODEL.auxiliary.optical_flow.fp16 = False
config.MODEL.auxiliary.optical_flow.name = 'flownet2' # the flownet type 'flownet2' | 'liteflownet'
config.MODEL.auxiliary.optical_flow.model_path = ''
config.MODEL.auxiliary.detector = CN()
config.MODEL.auxiliary.detector.require_grad = False
config.MODEL.auxiliary.detector.name = 'detectron2'
config.MODEL.auxiliary.detector.config = ''
config.MODEL.auxiliary.detector.model_path = ''

# configure the training process
#-----------------basic-----------------
config.TRAIN = CN()
config.TRAIN.engine_name = ''
config.TRAIN.batch_size = 2
config.TRAIN.start_step = 0
config.TRAIN.max_steps = 20000  # epoch * len(dataset)
config.TRAIN.dynamic_steps = [0, 50, 100]
config.TRAIN.log_step = 5  # the step to print the info
config.TRAIN.vis_step = 100  # the step to vis of the training results
# =========================================Will be deprecated in the future, because we don't need the minieval anymore===================================================
config.TRAIN.mini_eval_step = 100 # the step to exec the light-weight eval
# ============================================================================================================================
config.TRAIN.eval_step = 100 # the step to use the evaluate function
config.TRAIN.save_step = 500  # the step to save the model
config.TRAIN.epochs = 1
# configure the resume
config.TRAIN.resume = CN()
config.TRAIN.resume.use = False
config.TRAIN.resume.checkpoint_path = ''
# configure the freezing layers
config.TRAIN.finetune = CN()
config.TRAIN.finetune.use = False
config.TRAIN.finetune.layer_list = []

# must be 4-times, the 0-th is name in dict, 1-th is the coefficicent of this loss, 2-th is the model named registered in the registry, 3-th is the params of the loss functions
# e.g. ['loss_GeneratorLoss_cuda', 1.0, 'Adversarial_Loss', [], 'loss_Discriminiator_cuda', 1.0, 'Discriminate_Loss', []]
# The 0-th's format is 'registryName_NameInEngine_DeviceType', for example, 'loss_GeneratorLoss_cuda' means the registry is 'loss' and the attribute named 'self.GeneratorLoss' refers to it, the device is cuda. 
# If 3-th is null, the loss will use the default setting. If not null, it will be depended on the designing
# if 3-th is not null, it will be named as 'loss_cfg' and pass to the Loss class; each item is the configuration, the first is name of the args, the second is the value of the args. 
config.TRAIN.losses = ['loss_MSE_cuda', 0.5, 'MSELoss', [['size_average', None], ['reduce', None], ['reduction', 'mean']], 
                       'loss_Cross_cuda', 0.5, 'CrossEntropyLoss', []]  
                    #    ['weight', None], ['size_average', None], ['ignore_index', -100], ['reduce', None], ['reduction', 'mean']

config.TRAIN.mode = 'general' # general | adversarial | ....
#===============General Mode Settings==============
config.TRAIN.general = CN()
#---------------Optimizer configure---------------
config.TRAIN.general.optimizer = CN()
config.TRAIN.general.optimizer.include = ['A', 'B', 'C']
config.TRAIN.general.optimizer.name = 'adam'
config.TRAIN.general.optimizer.lrs = [1e-3]
config.TRAIN.general.optimizer.betas = [0.9, 0.999]
config.TRAIN.general.optimizer.momentum = 0.9
config.TRAIN.general.optimizer.weight_decay = 0.0001
config.TRAIN.general.optimizer.nesterov = False
config.TRAIN.general.optimizer.mode = 'all'  # all | individual
# config.TRAIN.general.optimizer.output_name = ['optimizer_abc'] # Will be discarded in the future
#-----------------Scheduler configure--------------
config.TRAIN.general.scheduler = CN()
config.TRAIN.general.scheduler.use = False
config.TRAIN.general.scheduler.name = 'none'
config.TRAIN.general.scheduler.step_size = 30 # the numebr of the iter, should be len(dataset) * want_epochs
config.TRAIN.general.scheduler.steps = [10000, 20000] # the numebr of the iter, should be len(dataset) * want_epochs
config.TRAIN.general.scheduler.gamma = 0.1
config.TRAIN.general.scheduler.T_max = 300 # use ine the cosine annealing LR
config.TRAIN.general.scheduler.eta_min = 0
config.TRAIN.general.scheduler.warmup_factor = 0.001
config.TRAIN.general.scheduler.warmup_iters = 1000
config.TRAIN.general.scheduler.warmup_method = 'linear' # 'linear' | 'constant'
#==================Adversarial Mode Setting==================
config.TRAIN.adversarial = CN()
#---------------Optimizer configure---------------
config.TRAIN.adversarial.optimizer = CN()
config.TRAIN.adversarial.optimizer.include = ['Generator', 'Discriminator']
config.TRAIN.adversarial.optimizer.name = 'adam'
config.TRAIN.adversarial.optimizer.lrs = [1e-2, 1e-2] # g_lr,d_lr
config.TRAIN.adversarial.optimizer.betas = [0.9, 0.999]
config.TRAIN.adversarial.optimizer.weight_decay = 0.0001
config.TRAIN.adversarial.optimizer.mode = 'individual'  # all: all the model parts use one optimizer | individual: each model part uses one optimizer 
#-----------------Scheduler configure--------------
config.TRAIN.adversarial.scheduler = CN()
config.TRAIN.adversarial.scheduler.use = False
config.TRAIN.adversarial.scheduler.name = 'none'
config.TRAIN.adversarial.scheduler.step_size = 30 # the numebr of the iter, should be len(dataset) * want_epochs
config.TRAIN.adversarial.scheduler.steps = [1000,2000] # the numebr of the iter, should be len(dataset) * want_epochs
config.TRAIN.adversarial.scheduler.gamma = 0.1
config.TRAIN.adversarial.scheduler.T_max = 300 # use ine the cosine annealing LR
config.TRAIN.adversarial.scheduler.eta_min = 0
config.TRAIN.adversarial.scheduler.warmup_factor = 0.001
config.TRAIN.adversarial.scheduler.warmup_iters = 5000
config.TRAIN.adversarial.scheduler.warmup_method = 'linear' # 'linear' | 'constant'
#----------------Train save configure------------
# config.TRAIN.split = ''
config.TRAIN.model_output = './output/models' # use save the final model
config.TRAIN.checkpoint_output = './output/checkpoint' # use to save the intermediate results, including lr, optimizer, state_dict...
config.TRAIN.pusedo_data_path = ''
#-------------------cluster setting--------------
config.TRAIN.cluster = CN()
config.TRAIN.cluster.k = 10

# configure the val process, equals to the TEST. 
config.VAL = CN()
config.VAL.engine_name = ''
config.VAL.log_step = 5  # the step to print the info
config.VAL.vis_step = 100  # the step to vis of the training results
config.VAL.path = '' # if not use the data in the DATASET.val.data_path
config.VAL.batch_size = 1
config.VAL.model_file = ''
config.VAL.result_output = './output/results'

# configure the service function
config.SERVICE = CN()
config.SERVICE.use = False
config.SERVICE.engine_name = 'BaseService'

def _get_cfg_defaults():
    """
    Get the config template.
    NOT USE IN OTHER FILES!!
    """
    return config.clone()


def update_config(yaml_path, opts):
    """
    Make the template update based on the yaml file.
    """
    print('=>Merge the config with {}\t'.format(yaml_path))
    cfg = _get_cfg_defaults()
    cfg.merge_from_file(yaml_path)
    cfg.merge_from_list(opts)
    cfg.freeze()

    return cfg
