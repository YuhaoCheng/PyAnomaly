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

def main(args, cfg, logger, writer_dict, cfg_name, time_stamp, log_file_name, is_training):
    """Main function to execute the training or inference process.

    The function indicates the corret order to use the APIs. If you want to use these API, please follow the order shown in this function.

    Args:
        args: The arguments from the command line.
        cfg: The configuration object.
        logger: The logger object to print the log information in the files and screen.
        writer_dict: The tensorboard writer dictionary. 
        cfg_name: The configuration file's name.
        time_stamp: The time string is when the main process starts.
        log_file_nam: The name of the log file.
        is_training: Indicate whether the process is training or not
    
    Returns:
        None

    """
    
    # the system setting
    parallel_flag = system_setup(args, cfg)
    
    # get the model structure
    ma = ModelAPI(cfg)
    model_dict = ma()

    # get the loss function dict
    la = LossAPI(cfg)
    loss_function_dict, loss_lamada = la()

    # get the optimizer
    oa = OptimizerAPI(cfg)
    optimizer_dict = oa(model_dict)

    # get the the scheduler
    sa = SchedulerAPI(cfg)
    lr_scheduler_dict = sa(optimizer_dict)
    
    # get the data loaders
    da = DataAPI(cfg, is_training)
    dataloaders_dict = da()

    # get the evaluate function
    ea = EvaluateAPI(cfg, is_training)
    evaluate_function = ea()

    # # Add the Summary writer 
    # writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)

    # Build hook
    ha = HookAPI(cfg)
    hooks = ha(is_training)

    # Get the engine
    engine_api = EngineAPI(cfg, is_training)
    engine = engine_api.build()
    trainer = engine(model_dict=model_dict, dataloaders_dict=dataloaders_dict, optimizer_dict=optimizer_dict, loss_function_dict=loss_function_dict, logger=logger, config=cfg, parallel=parallel_flag, 
                    pretrain=False,verbose=args.verbose, time_stamp=time_stamp, model_type=cfg.MODEL.name, writer_dict=writer_dict, config_name=cfg_name, loss_lamada=loss_lamada,
                    hooks=hooks, evaluate_function=evaluate_function,
                    lr_scheduler_dict=lr_scheduler_dict
                    )

    # import ipdb; ipdb.set_trace()
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
    # Get the Summary writer 
    writer_dict = get_tensorboard(tensorboard_log_dir, time_stamp, cfg.MODEL.name, log_file_name)
    # main(args, cfg, logger, tensorboard_log_dir, cfg_name, time_stamp, log_file_name, is_training=args.train)
    main(args, cfg, logger, writer_dict, cfg_name, time_stamp, log_file_name, is_training=args.train)
    logger.info(f'Finish {phase} the whole process!!!')
