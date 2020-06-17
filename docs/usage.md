## Content
- [Quick Start](#Quick-Start)
- [Support](#Support-Method)
- [Build Components](#build-components)
- [Tools](#tools)
- [Model Zoo](#model-zoo)

## Quick Start
The toolbox uses the shell file to start the training or inference process, and the parameters are as follows:

```shell
-m MODEL: The name of the method
-d DATASET: The name of the dataset
-p PROJRECT_PATH(ABSOLUTE)
-g GPUS(e.g. 0,1)
-c CONFIG_NAME
-v VERBOSE
-im(only in inference.sh) INFERENCE MODEL
```

The name list of methods and datasets are in the [MODEL ZOO](./model_zoo.md)

Example: 

Train: use the `amc` in `avenue`

```shell
cd $PATH/TO/ROOT
sh train.sh -m amc -d avenue -p PATH/TO/ANOMALY -g 0 -c default.yaml -v train_test
```

Inference: use the `amc` in `avenue`

```shell
cd $PATH/TO/ROOT
sh inference.sh -m amc -d avenue -p PATH/TO/ANOMALY -g 0 -c default.yaml /
-im MODEL_PATH -v inference_test
```

## Support
### Methods

| Method Name | `-m`      |
| ----------- | --------- |
| [STAE]()    | `stae`    |
| [AMC]()     | `amc`     |
| [OCAE]()    | `ocae`    |
| [AnoPred]() | `anopred` |
| [AnoPCN]()  | `anopcn`  |
| [MemAE]()   | `memae`   |

### Dataset

| Dataset Name            | `-d`       |
| ----------------------- | ---------- |
| [UCSD Ped2]()           | `ped2`     |
| [Avenue]()              | `avenue`   |
| [ShanghaiTech Campus]() | `shanghai` |

######  Note: More support will come in the future, if you have the interests, please refer to the [PLAN](./plan.md)


## Build Components

The original intention is to promote the development of anomaly detection, so almost everything in PyAnomaly can be built by researchers and engineers. We will introduce the simple steps to build your own components. For some details, please refer to the codes. 

### Dataset

We provide the abstract dataset class, and when users want to build the dataset, please inherit the abstract class. Meanwhile, we also provide the video and image reader tools that users can use it easily.

For example, if users want to build the dataset named `Example`, they should follow the steps:

1. Make a Python file named`example.py`  in `lib/datatools/dataclass`and contains the following things:

   ```python
   from lib.datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
   from lib.datatools.abstract.tools import ImageLoader, VideoLoader
   class Example(AbstractVideoAnomalyDataset):
       def custom_step(self):
           '''
           Step up the image loader or video loder
           '''
       def _get_frames(self):
           '''
           Step up the functions to get the frames
           '''
   ...
   ...
   def get_example(cfg, flag, aug):
       t = Example()
   ```

2. Open the `__init__.py`  in `lib/datatools/dataclass` and write the following things:

   ```python
   from .example import get_example
   def register_builtin_dataset():
       ...
   	DatasetCatalog.register('example', lambda cfg, flag, aug: get_example(cfg, flag, aug))
       ...
   ```

### Hooks

For example, users want to make a hook named `Example.ExampleTestHook`

1. Make a Python file named `example_hooks.py` in `lib/core/hook`and code the followings:

   ```python
   from .abstract.abstract_hook import HookBase
   HOOKS = ['ExampleTestHook']
   class ExampleTestHook(HookBase):
       def before_train(self):
           '''
           functions
           '''
       def after_train(self):
           '''
           functions
           '''
       def after_step(self):
           '''
           functions
           '''
       def before_step(self):
           '''
           functions
           '''
   
   def get_example_hooks(name):
       if name in HOOKS:
           t = eval(name)()
       else:
           raise Exception('The hook is not in amc_hooks')
       return t
   
   ```

2. Open the `__init__.py`  in `lib/core/hook/build` and add the following things:

   ```python
   from ..example_hooks import get_example_hooks
   
   def register_hooks():
       ...
       HookCatalog.register('Example.ExampleTestHook', lambda name:get_example_hooks(name))
       ...
   ```

### Loss Functions

Please refer to the codes

### Evaluation functions

Please refer to the codes

### Networks

For example, if you want to build a model named `Example`.

1. Make a Python file named `example_networks.py` in `lib/core/networks/parts`and code the followings:

   ```python
   ...
   class Example():
       '''
       the example networks
       '''
   def get_model_example():
       ...
       model_dict['example'] = Example()
       ...
       return model_dict
   ```

   

2. Open the `__init__.py`  in `/lib/networks/build/` and add the following things:

   ```python
   from ..parts.example_networks import get_model_example
   def register_model():
       ...
       ModelCatalog.register('example', lambda cfg: get_model_example(cfg))
       ...
   ```

   

## Tools

During build this tool, we make some code tools to read data, make loggers, and so on. We will introduce these things here and tell the users how to use them. Some of them are referred to other repo with some modification, and we will cite the original version of them. Details are in [TOOLS](./tools.md)


## [Model Zoo](./model_zoo.md)

