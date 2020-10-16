from fvcore.common import flops_count
###
# time, flops 
class Statistic(object):
    def __init__(self, model=None, mode='cuda', input_size=None, test_iters=3):
        '''
        Args:
            model: the test model
            mode: cpu or cuda
            input_size: the size of the input data. [NCHW] or [NCDHW]
            test_iters: the loops to test and average
        '''
        assert model is None, 'The model is None, please check it'
        assert input_size is None, 'The input size is none, please check it'
        self.model = model
        self.input_size = input_size
        print('The init the statistic of ')