from torch.utils.data import Dataset
class AbstractImageDataset(Dataset):
    _name = 'AbstractImageDataset'
    def __init__(self, dataset_folder, transforms):
        self.dir = dataset_folder
        self.transforms = transforms
    
    def get_image(self, image_name):
        '''
        Get one single image
        '''
        pass

    def aug_image(self):
        '''
        Use the transforms to augment one single image
        '''    
        pass
    
    def aug_batch_image(self):
        '''
        Use the transforms to augment one batch image
        '''
        pass

    def __getitem__(self, indice):
        raise Exception(f'No inplement at {AbstractImageDataset._name}')
    
    def __len__(self):
        raise Exception(f'No implement at {AbstractImageDataset._name}')

if __name__ == '__main__':
    temp = AbstractImageDataset(123, 123)