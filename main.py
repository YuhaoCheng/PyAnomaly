from pathlib import Path
from pyanomaly.config import update_config
from pyanomaly.utils.cmd import parse_args
from pyanomaly.utils.system import system_setup
from pyanomaly.utils.utils import create_logger, get_tensorboard

from pyanomaly import (
    ModelAPI,
    LossAPI,
    OptimizerAPI,
    SchedulerAPI,
    EngineAPI,
    HookAPI,
    DataAPI,
    EvaluateAPI
)

def train(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=True):
    
    # the system setting
    system_setup(args, cfg, logger)
    
    # get the model structure
    ma = ModelAPI(cfg, logger)
    model_dict = ma()

    # get the loss function dict
    la = LossAPI(cfg, logger)
    loss_function_dict, loss_lamada = la()
    
    # get the optimizer
    oa = OptimizerAPI(cfg, logger)
    optimizer_dict = oa(model_dict)

    # get the the scheduler
    sa = SchedulerAPI(cfg, logger)
    lr_scheduler_dict = sa(optimizer_dict)
    
   
    da = DataAPI(cfg, is_training)
    dataloaders_dict = da()
   
    # get the evaluate function
    ea = EvaluateAPI(cfg, logger)
    evaluate_function = ea(cfg.DATASET.evaluate_function_type)

    # Add the Summary writer 
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # build hook
    ha = HookAPI(cfg)
    hooks = ha(is_training)

    # =================================================Need to change to use the registry================================================================================================
    # # instance the trainer
    # core = importlib.import_module(f'pyanomaly.core.{cfg.MODEL.name}')
    # logger.info(f'Build the trainer in {core}')
    # # trainer = core.Trainer(model_dict, train_dataloader, valid_dataloder, optimizer_dict, loss_function_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
    # #                         pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, 
    # #                         loss_lamada=loss_lamada,test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
    # #                         test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w,
    # #                         cluster_dataset_dict=cluster_dataset_dict, cluster_dataset_keys=cluster_dataset_keys,
    # #                         hooks=hooks, evaluate_function=evaluate_function,
    # #                         lr_scheduler_dict=lr_scheduler_dict
    # #                         ) 
    # trainer = core.Trainer(model_dict, dataloaders_dict, optimizer_dict, loss_function_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
    #                         pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, loss_lamada=loss_lamada,
    #                         # test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
    #                         # test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w,
    #                         # cluster_dataset_dict=cluster_dataset_dict, cluster_dataset_keys=cluster_dataset_keys,
    #                         # dataset_dict=dataset_dict, 
    #                         hooks=hooks, evaluate_function=evaluate_function,
    #                         lr_scheduler_dict=lr_scheduler_dict
    #                         )
    engine_api = EngineAPI(cfg, True)
    engine = engine_api.build()
    trainer = engine(model_dict, dataloaders_dict, optimizer_dict, loss_function_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
                    pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, loss_lamada=loss_lamada,
                    hooks=hooks, evaluate_function=evaluate_function,
                    lr_scheduler_dict=lr_scheduler_dict
                    )
    # ===================================================================================================================================================================================

    # import ipdb; ipdb.set_trace()
    trainer.run(cfg.TRAIN.start_step, cfg.TRAIN.max_steps)
    
    logger.info('Finish Training')

    model_result_path = trainer.result_path

    logger.info(f'The model path is {model_result_path}')
   

def inference(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=False):
    
    # the system setting
    system_setup(args, cfg, logger)
    
    # get the model structure
    ma = ModelAPI(cfg, logger)
    model_dict = ma()
    
    # get the saved model path
    model_path = args.inference_model
    # get data augment
    # aa = AugmentAPI(cfg, logger)

    # # get the val augment 
    # val_augment = aa(flag='val')

    # build the dataAPI, can use the cfg to get the dataloader
    da = DataAPI(cfg)
    dataloaders_dict = da(cfg, is_training)
    # Get the validation dataloader
    # valid_dataloder = da(flag='val', aug=val_augment)

    # # Get the test datasets  ?
    # test_dataset_dict, test_dataset_keys = da(flag='test', aug=val_augment)
    # test_dataset_dict_w, test_dataset_keys_w = da(flag='train_w', aug=val_augment)


    # Add the Summary writer
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # build hook
    ha = HookAPI(cfg)
    # hooks = ha('val')
    hooks = ha(is_training)

    # get the evaluate function
    ea = EvaluateAPI(cfg, logger)
    evaluate_function = ea(cfg.DATASET.evaluate_function_type)

    # instance the inference
    # core = importlib.import_module(f'pyanomaly.core.{cfg.MODEL.name}')
    # logger.info(f'Build the inference in {core}')
    # inference = core.Inference(model_dict, model_path, valid_dataloder, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
    #                         pretrain=False, verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, 
    #                         test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
    #                         test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w, 
    #                         hooks=hooks,evaluate_function=evaluate_function
    #                         )
    engine_api = EngineAPI(cfg, False)
    engine = engine_api.build()
    inference = engine(model_dict, model_path, dataloaders_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
                            pretrain=False, verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, 
                            # test_dataset_dict=test_dataset_dict, test_dataset_keys=test_dataset_keys, 
                            # test_dataset_dict_w=test_dataset_dict_w, test_dataset_keys_w=test_dataset_keys_w, 
                            hooks=hooks,evaluate_function=evaluate_function
                            )
    inference.run()
    
    logger.info('Finish Inference')

if __name__ == '__main__':
    args = parse_args()
    # Get the root path of the project
    root_path = Path(args.project_path)
    
    # Get the config yaml file and upate 
    cfg_path = root_path /'configuration'/ args.cfg_folder / args.cfg_name
    cfg = update_config(cfg_path, args.opts)
    
    # decide the spcific function: train or inference
    if not args.inference:
        # get the logger
        logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name = create_logger(root_path, cfg, args.cfg_name, phase='train', verbose=args.verbose)
        logger.info(f'^_^==> Use the following tensorboard:{tensorboard_log_dir}')
        logger.info(f'@_@==> Use the following config in path: {cfg_path}')
        logger.info(f'the configure name is {cfg_name}, the content is:\n{cfg}')
        train(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=True)
        logger.info('Finish training the whole process!!!')
    elif args.inference:
        # get the logger
        logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name = create_logger(root_path, cfg, args.cfg_name, phase='inference', verbose=args.verbose)
        logger.info(f'^_^==> Use the following tensorboard:{tensorboard_log_dir}')
        logger.info(f'@_@==> Use the following config in path: {cfg_path}')
        logger.info(f'the configure name is {cfg_name}, the content is:\n{cfg}')
        inference(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=False)
        logger.info('Finish inference the whole process!!!')
