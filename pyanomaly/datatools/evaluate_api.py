from .evaluate.eval_function import eval_functions

class EvaluateAPI(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
    
    def __call__(self, eval_function_type):
        assert eval_function_type in eval_functions, f'there is no type of evaluation {eval_function_type}, please check {eval_functions.keys()}'
        self.logger.info(f'==> Using the eval function: {eval_function_type}')
        t = eval_functions[eval_function_type]
        return t

        