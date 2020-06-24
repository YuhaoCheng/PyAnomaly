from .image_dataset import AbstractImageDataset
from .tools import np_load_frame, LoadImage

class AbstractVideoDataset(AbstractImageDataset):
    _name = 'AbstractVideoDataset'
    def __init__(self, dataset_folder, clip_length=1, transforms=None, cfg=None):
        super(AbstractVideoDataset, self).__init__(dataset_folder, transforms)
        self.cfg = cfg
        self.image_loader = LoadImage(read_format=self.cfg.DATASET.read_format,transforms=self.transforms)

    def setup(self):
        # self.image_loder
        pass 
    
    def get_image(self, image_name):
        '''
        Get one single image
        '''
        pass

    def _get_frames(self, indice):
        pass

    def __getitem__(self, indice):
        pass
    