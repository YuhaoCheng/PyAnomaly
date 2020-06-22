import os
import torch
import importlib
from pathlib import Path
from collections import OrderedDict

from lib.config.config import update_config
from lib.utils.cmd import parse_args
from lib.utils.system import system_setup
from lib.utils.utils import create_logger, get_tensorboard

from lib.networks.build_model import ModelAPI
from lib.loss.build_loss import LossAPI
from lib.core.optimizer.build_optimizer import OptimizerAPI
from lib.core.scheduler.build_scheduler import SchedulerAPI
from lib.core.hook.build.build_hooks import HookAPI
from lib.datatools.build_augment import AugmentAPI
from lib.datatools.build_datasets import DataAPI
from lib.datatools.build_evaluate import EvaluateAPI

def main(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name):
    
    # the system setting
    system_setup(args, cfg, logger)
    
    # get the model structure
    ma = ModelAPI(cfg, logger)
    model_dict = ma()

    # get the loss function dict
    la = LossAPI(cfg, logger)
    loss_function_dict, loss_lamada = la()
    
    # import ipdb; ipdb.set_trace()
    # get the optimizer
    oa = OptimizerAPI(cfg, logger)
    optimizer_dict = oa(model_dict)
    
    # get the the scheduler
    sa = SchedulerAPI(cfg, logger)
    lr_scheduler_dict = sa(optimizer_dict)
    
    # get data augment
    aa = AugmentAPI(cfg, logger)

    # get the train augment
    train_augment = aa(flag='train')
    
    # get the val augment 
    val_augment = aa(flag='val')

    # build the dataAPI, can use the cfg to get the dataloader
    da = DataAPI(cfg)
    #  Get the train dataloader
    train_dataloader = da(flag='train', aug=train_augment)
    
    # Get the validation dataloader
    valid_dataloder = da(flag='val', aug=val_augment)

    # Get the test datasets
    test_dataset_dict, test_dataset_keys = da(flag='test', aug=val_augment)
    test_dataset_dict_w, test_dataset_keys_w = da(flag='train_w', aug=val_augment)
    
    # Get the cluster dataset
    cluster_dataset_dict, cluster_dataset_keys = da(flag='cluster_train', aug=train_augment)

    # get the evaluate function
    ea = EvaluateAPI(cfg, logger)
    evaluate_function = ea(cfg.DATASET.evaluate_function_type)

    # Add the Summary writer 
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # build hook
    ha = HookAPI(cfg, logger)
    hooks = ha('train')

    
    # instance the trainer
    core = importlib.import_module(f'lib.core.{cfg.MODEL.name}')
    logger.info(f'Build the trainer in {core}')
    trainer = core.Trainer(model_dict, train_dataloader, valid_dataloder, optimizer_dict, loss_function_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
                            pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, 
                            loss_lamada=loss_lamada,test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
                            test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w,
                            cluster_dataset_dict=cluster_dataset_dict, cluster_dataset_keys=cluster_dataset_keys,
                            hooks=hooks, evaluate_function=evaluate_function,
                            # lr_shechulder_g=lr_shechulder_g, lr_shechulder_d=lr_shechulder_d
                            lr_scheduler_dict=lr_scheduler_dict
                            )
    
    trainer.run(cfg.TRAIN.start_step, cfg.TRAIN.max_steps)
    
    logger.info('Finish Training')

    model_result_path = trainer.result_path
    print(f'The model path is {model_result_path}')
   

def inference(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name):
    
    # the system setting
    system_setup(args, cfg, logger)
    
    # get the model structure
    ma = ModelAPI(cfg, logger)
    model_dict = ma()
    
    # get the saved model path
    model_path = args.inference_model
    # get data augment
    aa = AugmentAPI(cfg, logger)

    # get the val augment 
    val_augment = aa(flag='val')

    # build the dataAPI, can use the cfg to get the dataloader
    da = DataAPI(cfg)
    
    # Get the validation dataloader
    valid_dataloder = da(flag='val', aug=val_augment)

    # Get the test datasets  ?
    test_dataset_dict, test_dataset_keys = da(flag='test', aug=val_augment)
    test_dataset_dict_w, test_dataset_keys_w = da(flag='train_w', aug=val_augment)

    # Add the Summary writer 
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # build hook
    ha = HookAPI(cfg, logger)
    hooks = ha('val')

    # get the evaluate function
    ea = EvaluateAPI(cfg, logger)
    evaluate_function = ea(cfg.DATASET.evaluate_function_type)

    # instance the inference
    core = importlib.import_module(f'lib.core.{cfg.MODEL.name}')
    logger.info(f'Build the inference in {core}')
    inference = core.Inference(model_dict, model_path, valid_dataloder, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
                            pretrain=False, verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, 
                            test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
                            test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w, 
                            hooks=hooks,evaluate_function=evaluate_function
                            )
    
    inference.run()
    
    logger.info('Finish Inference')

if __name__ == '__main__':
    args = parse_args()
    # Get the root path of the project
    root_path = Path(args.project_path)
    
    # Get the config yaml file and upate 
    cfg_path = root_path /'experiments'/ args.cfg_folder / args.cfg_name
    cfg = update_config(cfg_path, args.opts)
    # get the logger
    logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name = create_logger(root_path, cfg, args.cfg_name, phase='train', verbose=args.verbose)
    logger.info('^_^==> Use the following tensorboard:{}'.format(tensorboard_log_dir))
    logger.info('@_@==> Use the following config in path: {}'.format(cfg_path))
    logger.info(f'the configure name is {cfg_name}, the content is:\n{cfg}')
    
    # decide the spcific function: train or inference
    if not args.inference:
        main(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name)
        logger.info('Finish training the whole process!!!')
    elif args.inference:
        inference(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name)
