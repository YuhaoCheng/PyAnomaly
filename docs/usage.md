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
-f(only in inference.sh) INFERENCE MODEL
```

The name list of methods and datasets are in the [MODEL ZOO](./model_zoo.md)

Example: 

Train: use the `amc` in `avenue`

```shell
cd $PATH/TO/ROOT
sh ./script/train.sh -m amc -d avenue -p PATH/TO/ANOMALY -g 0 -c default.yaml -v train_test
```

Inference: use the `amc` in `avenue`

```shell
cd $PATH/TO/ROOT
sh ./script/inference.sh -m amc -d avenue -p PATH/TO/ANOMALY -g 0 -c default.yaml /
-f MODEL_FILE -v inference_test
```

## Support

This part  introduces the present supported methods and the datasets in our project. The method's type is based on the taxonomy shown in the [PyAnomaly: A Pytorch-based Toolkit for Video Anomaly Detection](https://dl.acm.org/doi/10.1145/3394171.3414540). 
### Methods

| Method Name | Method Type | Brief Introduction | `-m` parameter     |
| ----------- | --------- | --------- | --------- |
| [STAE](https://dl.acm.org/doi/abs/10.1145/3123266.3123451) | Reconstruction + Prediction | It uses the diﬀerences between the reconstructed and the original to decide whether the frame is an anomaly | `stae`    |
| [AMC](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf) | Reconstruction | A method uses GAN to reconstruct the optical ﬂow and image at the same time. | `amc`     |
| [OCAE](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf) | Reconstruction | A method mainly uses encoded features from the objects in the scene for construction. | `ocae`    |
| [AnoPred](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf) | Prediction | A method predicts the frames based on the historical information, and judges whether anomaly based on the diﬀerences between the predicted frames and the original ones. | `anopred` |
| [AnoPCN](https://dl.acm.org/doi/10.1145/3343031.3350899) | Reconstruction + Prediction | A method uses the RNN to reconstruct and predict at the same time. | `anopcn`  |
| [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf) | Reconstruction | A method uses the memory mechanism to reconstruct the frames. | `memae`   |

### Dataset

| Dataset Name            |  Brief Introduction    | `-d` parameter |
| ----------------------- | ---- | ---------- |
| [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) | Ped2 contains 16 normal training videos and 12 testing videos | `ped2`     |
| [Avenue](https://dl.acm.org/doi/10.1109/ICCV.2013.338) | It contains 16 training and 21 testing video clips | `avenue`   |
| [ShanghaiTech Campus](https://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf) | It contains 330 training videos and 107 testing videos, which is captured from 13 diﬀerent scenes. | `shanghai` |

######  Note: More support will come in the future, if you have the interests, please refer to the [PLAN](./plan.md)


## Build Components

The original intention is to promote the development of anomaly detection, so almost everything in [PyAnomaly](https://github.com/YuhaoCheng/PyAnomaly) can be built by researchers and engineers. We will introduce the simple steps to build your own components. For some details, please refer to the codes. 

### Dataset

We provide the abstract dataset class, and when users want to build the dataset, please inherit the abstract class. Meanwhile, we also provide the video and image reader tools that users can use it easily.

For example, if users want to build the dataset named `Example`, they should follow the steps:

1. Make a Python file named`example.py`  in `pyanomaly/datatools/dataclass`and contains the following things:

   ```python
   from ..abstract.video_dataset import FrameLevelVideoDataset
   from ..datatools_registry import DATASET_REGISTRY
   
   @DATASET_REGISTRY.register()
   class ExampleDataset(FrameLevelVideoDataset):
       def custom_step(self):
           '''
           Step up the image loader or video loder
           '''
   ```

2. Open the `dataset_factory.py`  in `pyanomaly/datatools/dataclass` and write the following things on the top lines:

   ```python
   from .example import ExampleDataset
   ```

### Dataset Factory

As shown in above, we use the Factory Designing Pattern to produce the instance of dataset class. And, specifically, we also allow the developers to use another factory instead of `VideoAnomalyDatasetFactory`. For example, if users want to build a factory named `Example`, they should follow the steps:

1. Make a Python file named `example_factory.py` in `pyanomaly/datatools/dataclass`  which contains the following things:

   ```python
   from ..datatools_registry import DATASET_FACTORY_REGISTRY
   from ..datatools_registry import DATASET_REGISTRY
   from ..abstract import AbstractDatasetFactory
   from DATASET_CLASS_FILES import *
   
   @DATASET_FACTORY_REGISTRY.register()
   class ExampleFactory(AbstractDatasetFactory):
       def _produce_train_dataset(self):
           '''
           Produce the dataset for training process
           '''
           pass
       def _produce_test_dataset(self):
           '''
           Produce the dataset for inference produce
           '''
           pass
       def _buid(self):
           '''
           Call the self._produce_train_dataset and self._produce_test_dataset
           '''
           pass
   ```

2. Open `__init__.py` in `pyanomaly/datatools/dataclass` and write the following things:

   ```python
   from .example_factory import ExampleFactory
   ```

### Networks

For example, if you want to build a model named `Example`.

1. Make a Python file named `example_networks.py` in `pyanomaly/core/networks/meta`and code the followings:

   ```python
   import torch
   import torch.nn as nn
   from ..model_registry import META_ARCH_REGISTRY
   
   @META_ARCH_REGISTRY.register()
   class ExampleModule(nn.Module):
       '''
       the example networks
       '''
   ```

2. Open the `__init__.py`  in `pyanomaly/networks/meta` and add the following things:

   ```python
   from .example_networks import ExampleModule
   ```

3. Open the `model_api.py`  in `pyanomaly/networks` and add the following things:

   ```python
   from .meta import ExampleModule
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

### Engine 

 ```

 ```




### Loss Functions

Please refer to the codes

### Evaluation functions

Please refer to the codes


## Tools

During build this tool, we make some code tools to read data, make loggers, and so on. We will introduce these things here and tell the users how to use them. Some of them are referred to other repo with some modification, and we will cite the original version of them. Details are in [TOOLS](./tools.md)

## Model Zoo

In order to help researchers and engineers, we put some trained models on the Model Zoo.  For the details information, please refer to the [Model Zoo](./model_zoo.md).

