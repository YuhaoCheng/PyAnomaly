from .build.build import DatasetBuilder

class DataAPI(DatasetBuilder):
    def __init__(self, cfg):
        super(DataAPI, self).__init__(cfg)
    
    def information(self):
        print('no information')
    
    def __call__(self, flag, aug):
        data = super(DataAPI, self).build(flag=flag, aug=aug)
        return data