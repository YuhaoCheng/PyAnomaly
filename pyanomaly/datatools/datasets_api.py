# from .dataclass.dataset_builder import DatasetBuilder
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from .sampler.inf_sampler import TrainSampler
from .sampler.dist_inf_sampler import DistTrainSampler
# from .dataclass.datacatalog import DatasetCatalog
from .abstract.abstract_datasets_builder import AbstractBuilder
from .datasets_registry import DATASET_FACTORY_REGISTRY
from .dataclass import *
from .augment import AugmentAPI
import logging
logger = logging.getLogger(__name__)

BUILTIN = ['avenue', 'shanghai', 'vad', 'ped1', 'ped2', 'dota']

class DataAPI(AbstractBuilder):
    _name = 'DatasetAPI'
    def __init__(self, cfg, is_training):
        # super(DatasetBuilder, self).__init__(cfg)
        self.seed = cfg.DATASET.seed
        self.cfg = cfg
        # self.aug = aug
        self.is_training = is_training
        aug_api = AugmentAPI(cfg)
        aug_dict = aug_api.build()
        self.factory = DATASET_FACTORY_REGISTRY.get(self.cfg.DATASET.factory)(self.cfg, aug_dict, self.is_training)
        # print(f'The dataclass register in {DatasetBuilder._name} are: {DatasetCatalog._REGISTERED}')

    def build(self):
        '''
        flag: the type of the dataset
        train--> use to train, all data, inf sampler
        val--> use to val, all data, no-inf sampler
        test--> use to test, dataset for each video, no-inf sampler
        '''
        # build the dataset
        # self.flag = flag
        dataset = self._build_dataset()
        # build the sampler
        # self._data_len = len(dataset)

        # if flag == 'train' or flag == 'mini':
        #     sampler = self._build_sampler()
        #     batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, self.cfg.TRAIN.batch_size, drop_last=True)
        #     dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
        # elif flag == 'val':
        #     dataloader = DataLoader(dataset, batch_size=self.cfg.VAL.batch_size, shuffle=False, pin_memory=True)
        # elif flag == 'test':
        #     dataloader = dataset    #  the dataset is the dict of dataset, need to imporve in the future
        # elif flag == 'train_w' or flag == 'cluster_train':
        #     dataloader = dataset
        # else:
        #     raise Exception('No supprot dataset')
        if self.is_training:
            dataset_dict = dataset['train_dataset_dict']
            batch_size = self.cfg.TRAIN.batch_size
        else:
            dataset_dict = dataset['test_dataset_dict']
            batch_size = self.cfg.TEST.batch_size
        
        import ipdb; ipdb.set_trace()
        dataloader_dict = OrderedDict()

        for key in dataset_dict:
            temp = dataset_dict[key]
            temp_data_len = len(temp)
            sampler = self._build_sampler(temp_data_len)
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
            dataloader_dict[key] = dataloader
        
        return dataloader_dict
    
    def _build_dataset(self):
        
        # if self.cfg.DATASET.name in BUILTIN:
        #     dataset = DatasetCatalog.get(self.cfg.DATASET.name, self.cfg, self.flag, aug)
        # else:
        #     raise Exception('no implement')
        dataset = self.factory()
        return dataset
    
    def _build_sampler(self, _data_len):
        if self.cfg.SYSTEM.distributed.use:
            sampler = DistTrainSampler(_data_len)
        else:
            sampler = TrainSampler(_data_len, self.seed)
        return sampler
    
    def __call__(self):
        dataloader = self.build()
        return dataloader


# class DataAPI(DatasetBuilder):
#     def __init__(self, cfg):
#         super(DataAPI, self).__init__(cfg)
    
#     def information(self):
#         print('no information')
    
#     def __call__(self, flag, aug):
#         data = super(DataAPI, self).build(flag=flag, aug=aug)
#         return data