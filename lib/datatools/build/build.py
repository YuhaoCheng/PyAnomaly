import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .abstract_build import AbstractBuilder

from ..sampler.inf_sampler import TrainSampler
from ..sampler.dist_inf_sampler import DistTrainSampler
from ..dataclass.datacatalog import DatasetCatalog
BUILTIN = ['avenue', 'shanghai', 'vad', 'ped1', 'ped2']

class DatasetBuilder(AbstractBuilder):
    _name = 'DatasetBuilder'
    def __init__(self, cfg):
        super(DatasetBuilder, self).__init__(cfg)
        self.seed = cfg.DATASET.seed
        
        # print(f'The dataclass register in {DatasetBuilder._name} are: {DatasetCatalog._REGISTERED}')

    def build(self, flag='train',aug=None):
        '''
        flag: the type of the dataset
        train--> use to train, all data, inf sampler
        val--> use to val, all data, no-inf sampler
        test--> use to test, dataset for each video, no-inf sampler
        '''
        # build the dataset
        self.flag = flag
        dataset = self._build_dataset(aug)
        
        # build the sampler
        self._data_len = len(dataset)
        if flag == 'train' or flag == 'mini':
            sampler = self._build_sampler()
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, self.cfg.TRAIN.batch_size, drop_last=True)
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
        elif flag == 'val':
            dataloader = DataLoader(dataset, batch_size=self.cfg.VAL.batch_size, shuffle=False, pin_memory=True)
        elif flag == 'test':
            dataloader = dataset    #  the dataset is the dict of dataset, need to imporve in the future
        elif flag == 'train_w' or flag == 'cluster_train':
            dataloader = dataset
        else:
            raise Exception('No supprot dataset')

        return dataloader
    
    def _build_dataset(self, aug):
        
        if self.cfg.DATASET.name in BUILTIN:
            dataset = DatasetCatalog.get(self.cfg.DATASET.name, self.cfg, self.flag, aug)
        else:
            raise Exception('no implement')

        return dataset
    
    def _build_sampler(self):
        if self.cfg.SYSTEM.distributed.use:
            sampler = DistTrainSampler(self._data_len)
        else:
            sampler = TrainSampler(self._data_len, self.seed)
        return sampler


