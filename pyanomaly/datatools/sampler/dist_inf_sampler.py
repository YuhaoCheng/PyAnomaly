from torch.utils.data.sampler import Sampler
import itertools
import torch
import torch.distributed as dist
import numpy as np
import pyanomaly.datatools.sampler.common as comm
# def get_world_size() -> int:
#     if not dist.is_available():
#         return 1
#     if not dist.is_initialized():
#         return 1
#     return dist.get_world_size()

# def get_rank() -> int:
#     if not dist.is_available():
#         return 0
#     if not dist.is_initialized():
#         return 0
#     return dist.get_rank()

class DistTrainSampler(Sampler):
    """refer https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py

    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """
    def __init__(self, size, shuffle=True, seed=None, start=0):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            start(int): where to start to generate the data
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            # seed = np.random.RandomState(2020)
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        # self._start = start

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._inf_indices(), start, None, self._world_size)
    
    def _inf_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

class InferenceSampler(Sampler):
    """refer https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """
    def __init__(self, size, start=0):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._locak_indeices = range(start, self._size)
    
    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)