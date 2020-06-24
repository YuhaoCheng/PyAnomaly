import itertools
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data.sampler import Sampler


class TrainSampler(Sampler):
    '''
    Generate the infinite stream, not support dist
    However, the inference not use this!!
    '''
    def __init__(self, size, seed, start=0, shuffle=True):
        self._size = size
        self._seed = seed
        self._shuffle = shuffle
        self._start = start
    
    def __iter__(self):
        yield from itertools.islice(self._inf_indices(), self._start, None)

    def _inf_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        '''
        inf generate the indices of each single dataset
        '''
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size) 


if __name__ == '__main__':
    cifar100_path = '/export/home/chengyh/data/cifar-100'
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    aug_train = T.Compose([T.Resize((32,32)),
                             T.RandomHorizontalFlip(p=0.5),
                             T.RandomRotation(degrees=10),
                             T.ToTensor(),
                             normalize
                    ])
    train_dataset = torchvision.datasets.CIFAR100(root=cifar100_path, transform=aug_train, download=False)
    size = len(train_dataset)
    seed = 2020
    sampler = TrainSampler(size, seed)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 600, drop_last=True
        )
        # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler
        )
    print(size)
    diter = iter(data_loader)
    for i in range(100):
        print(i)
        diter.next()
        import ipdb; ipdb.set_trace()
    
