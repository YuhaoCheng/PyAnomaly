import abc
import six

@six.add_metaclass(abc.ABCMeta)
class AbstractDatasetFactory(object):
    def __init__(self, cfg, aug, is_training=True) -> None:
        self.cfg = cfg
        self.model_name = cfg.MODEL.name
        self.dataset_name = cfg.DATASET.name
        self.dataset_params = self.cfg.DATASET
        self.aug = aug
        self.is_training = is_training

        if self.is_training:
            self.phase = 'train'
        else:
            self.phase = 'val'
    
    @abc.abstractclassmethod
    def _produce_train_dataset(self):
        """
        Aim to produce the dataset used for the training process
        """
        pass
    
    @abc.abstractclassmethod
    def _produce_test_dataset(self):
        pass
    
    @abc.abstractclassmethod
    def _build(self):
        pass

@six.add_metaclass(abc.ABCMeta)
class GetWDataset(object):

    @abc.abstractclassmethod
    def _produce_w_dataset(self):
        pass

    @abc.abstractclassmethod
    def _jude_need_w(self):
        """
        base the need of the process to decide whether need the w dataset
        
        """
        pass


@six.add_metaclass(abc.ABCMeta)
class GetClusterDataset(object):

    @abc.abstractclassmethod
    def _produce_cluster_dataset(self):
        """
        Produce the 
        """
        pass

    @abc.abstractclassmethod
    def _jude_need_cluster(self):
        """
        base the need of the process to decide whether need the cluster dataset
        """
        pass

