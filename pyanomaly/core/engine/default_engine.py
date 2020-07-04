import torch
from pyanomaly.core.utils import AverageMeter
from .utils import engine_save_checkpoint
from .utils import engine_save_model
from .abstract import AbstractTrainer, AbstractInference

class DefaultTrainer(AbstractTrainer):
    ''' __init__ template 
    the inf trainer 
    '''
    def __init__(self, *defaults, **kwargs):
        '''
        Args:
            defaults(tuple): the default will have:
                0->model: the model of the experiment 
                1->train_dataloader: the dataloader   
                2->val_dataloader: the dataloader     
                3->optimizer: the optimizer of the network
                4->loss_function: the loss function of the model
                5->logger: the logger of the whole training process
                6->config: the config object of the whole process

            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model

        '''
        # logger & config
        self.logger = defaults[5]
        self.config = defaults[6]
        
        # basic things
        if kwargs['parallele']:
            self.model = self.data_parallel(defaults[0])
        else:
            self.model = defaults[0].cuda()
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self.val_dataloader = defaults[2]
        self.optimizer = defaults[3]
        self.loss_function = defaults[4]

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_basic = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.kwargs = kwargs
        self.result_path = '.'
        self.log_step = 5  # how many steps to print the information
        self.eval_step = 5 # how many steps to use the val dataset to test the model
        self.save_step = 5

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
    
    def load_pretrain(self):
        model_path = self.config.MODEL.pretrain_model
        if  model_path is '':
            self.logger.info('=>Not have the pre-train model! Training from the scratch')
        else:
            self.logger.info('=>Loading the model in {}'.format(model_path))
            pretrain_model = torch.load(model_path)
            if 'epoch' in pretrain_model.keys():
                self.logger.info('(|_|) ==> Use the check point file')
                self.model.load_state_dict(pretrain_model['model_state_dict'])
            else:
                self.logger.info('(+_+) ==> Use the model file')
                self.model.load_state_dict(pretrain_model['state_dict'])

    
    def resume(self):
        self.logger.info('=> Resume the previous training')
        checkpoint_path = self.config.RESUME.checkpoint_path
        self.logger.info('=> Load the checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    def fine_tune(self):
        layer_list = self.config.FINETUNE.layer_list
        self.logger.info('=> Freeze layers except start with:{}'.format(layer_list))
        for n, p in self.model.named_parameters():
            parts = n.split('.')
            # consider the data parallel situation
            if parts[0] == 'module':
                if parts[1] not in layer_list:
                    p.requires_grad = False
                if p.requires_grad:
                    print(n)
            else:
                if parts[0] not in layer_list:
                    p.requires_grad = False
                if p.requires_grad:
                    print(n)
        self.logger.info('Finish Setting freeze layers')
    
    def data_parallel(self, model):
        '''
        Data parallel the model
        '''
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in range(self.config.SYSTEM.num_gpus)]
        model_parallel = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        return model_parallel
    
    '''
    Run the whole process:
    1. print the log information ( before_step)
    2. execute training process (train)
    3. evaluate(including the validation and test) -
                                                   |  --> (after_step)
    4. save model                                  -
    '''
    
    def before_step(self, current_step):
        pass

    
    def after_step(self, current_step):
        # acc = 0.0
        for h in self._hooks:
            h.after_step(current_step)
        
        # if (current_step % self.eval_step != 0) or current_step == 0:
        #     self.mini_eval(current_step)
        #     return

    
    def after_train(self):
        for h in self._hooks:
            h.after_train()
        
        self.save(self.config.TRAIN.max_steps)

    def save(self, current_epoch, best=False):
        '''
        self.saved_model: is the model or a dict of combination of models
        self.saved_optimizer: is the optimizer or a dict of combination of optimizers
        self.saved_loss: the loss  or a dict of the combination  of loss 
        '''
        if best:
            engine_save_checkpoint(self.config, self.kwargs['config_name'], self.saved_model, current_epoch, self.saved_loss, self.saved_optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, flag='best', verbose=(self.kwargs['model_type'] + '#' + self.verbose),best=best)
            self.result_path = engine_save_model(self.config, self.kwargs['config_name'], self.saved_model, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=best)
        else:
            engine_save_checkpoint(self.config, self.kwargs['config_name'], self.saved_model, current_epoch, self.saved_loss, self.saved_optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=best)

    
    
    def train(self,current_step):
        raise Exception('Need to implement the train function!!')
    
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official

        Returns:
            metric: the metric 
        '''
        raise Exception('Need to implement the evaluation function, return the score')
    
        
 
class DefaultInference(AbstractInference):
    '''__init__ template
    '''
    def __init__(self, *defaults,**kwargs):
        '''
         Args:
            defaults(tuple): the default will have:
                0->model: the model of the experiment
                1->model_path: the path of the model path
                2->val_dataloader: the dataloader to inference
                3->logger: the logger of the whole process
                4->config: the config object of the whole process
            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model
                mode(str): 'dataset' -> the data will use the dataloder to pass in(dicard, becasue we will use the dataset to get all I need)
        '''
        self.logger = defaults[3]
        self.config = defaults[4]
        self.model_path = defaults[1]
        if kwargs['parallel']:
            self.model = self.data_parallel(defaults[0])
        else:
            self.model = defaults[0]

        self.load()

        self.verbose = kwargs['verbose']
        self.kwargs = kwargs
        self.mode = kwargs['mode']

        self.metric = 0.0

    def load(self):
        if type(self.model) == type(dict()):
            for k, v in self.model.items():
                temp = torch.load(self.model_path)
                if k[0] == 'F':
                    continue
                self.model[k].load_state_dict(temp[k[0]])
        else:
            self.model.load_state_dict(torch.load(self.model_path))

    def data_parallel(self, model):
        '''
        Data parallel the model
        '''
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        return model_parallel
    
    def inference(self, current_step):
        if self.mode == 'dataset':
            metric = self.evaluate()
        elif self.mode == 'other':
            self.get_result()
        else:
            raise Exception('Wrong inference mode')
    

    
    def get_result(self):
        '''
        Get the results for one image
        
        '''
        raise Exception('Need to implement the get_result function, return the score')
    
    def extract_feature(self):
        '''
        Get the feature of input
        '''
        pass

    def save(self):
        '''
        Save the results or the feature
        '''
        pass