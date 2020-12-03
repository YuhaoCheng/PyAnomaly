from pathlib import Path
from pyanomaly.config import update_config

from pyanomaly.utils import(
    parse_args,
    system_setup, 
    create_logger, 
    get_tensorboard
)

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

def main(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training):
    
    # the system setting
    system_setup(args, cfg)
    
    # get the model structure
    ma = ModelAPI(cfg)
    model_dict = ma()

    # get the loss function dict
    # la = LossAPI(cfg, logger)
    la = LossAPI(cfg)
    loss_function_dict, loss_lamada = la()
    # import ipdb; ipdb.set_trace()
    # get the optimizer
    oa = OptimizerAPI(cfg)
    optimizer_dict = oa(model_dict)

    # get the the scheduler
    sa = SchedulerAPI(cfg)
    lr_scheduler_dict = sa(optimizer_dict)
    
    # get the data loaders
    da = DataAPI(cfg, is_training)
    dataloaders_dict = da()

    # import ipdb; ipdb.set_trace()
    
    # get the evaluate function
    ea = EvaluateAPI(cfg, is_training)
    evaluate_function = ea()

    # Add the Summary writer 
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # Build hook
    ha = HookAPI(cfg)
    hooks = ha(is_training)

    # Get the engine
    engine_api = EngineAPI(cfg, True)
    engine = engine_api.build()
    trainer = engine(model_dict, dataloaders_dict, optimizer_dict, loss_function_dict, logger, cfg, parallel=cfg.SYSTEM.multigpus, 
                    pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, loss_lamada=loss_lamada,
                    hooks=hooks, evaluate_function=evaluate_function,
                    lr_scheduler_dict=lr_scheduler_dict
                    )

    
    trainer.run(cfg.TRAIN.start_step, cfg.TRAIN.max_steps)
    
    
    model_result_path = trainer.result_path

    logger.info(f'The model path is {model_result_path}')
   

if __name__ == '__main__':
    args = parse_args()
    # Get the root path of the project
    root_path = Path(args.project_path)
    
    # Get the config yaml file and upate 
    cfg_path = root_path /'configuration'/ args.cfg_folder / args.cfg_name
    cfg = update_config(cfg_path, args.opts)
    phase = 'train' if args.train else 'inference'

    logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name = create_logger(root_path, cfg, args.cfg_name, phase=phase, verbose=args.verbose)  # dataset_name, model_name, cfg_name, time_stamp will decide the all final name such that of the model, results, tensorboard, log.
    logger.info(f'^_^==> Use the following tensorboard:{tensorboard_log_dir}')
    logger.info(f'@_@==> Use the following config in path: {cfg_path}')
    logger.info(f'the configure name is {cfg_name}, the content is:\n{cfg}')
    main(args, cfg, logger, final_output_dir, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=args.train)
    logger.info(f'Finish {phase} the whole process!!!')
