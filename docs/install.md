### Installation

#### System Requirement
- Linux

#### Packages Requirements

The requirement packages are as follows, and if users have some problems in installation, please submit an issue, and we will solve it ASAP.

- PyTorch >= 1.3
- [Detectron2](https://github.com/facebookresearch/detectron2)
  - we use this package to get the object detection. 
- [Flownet2](https://github.com/NVIDIA/flownet2-pytorch)
  - Installing this means that users should install the dependency packages of Flownet2 and build the layers.
  - Users should go the `scripts` and execute `sh install.sh` 
- `cupy`
  - `cupy` must match the CUDA version, if having the problem, please try this: `pip install cupy-cudaxxx`. `xxx` is the version of the CUDA, e.g. cuda101 is CUDA 10.1
- `tensorboard`
- torchvison
- `colorama`
- `torchsnooper`
  - This package is in [Link](https://github.com/zasdfgbnm/TorchSnooper). This is a package  that help users to print the tensors'  shape during the debug process

#### Install PyAnomaly

```shell
cd ROOT/PATH
git clone https://github.com/YuhaoCheng/PyAnomaly.git
cd PyAnomaly
```

#### Dataset Preparing
1. Make the `data` folder in `PyAnomaly`:

   ```shell
   cd PyAnomaly
   mkdir data
   ```

2. Download the data into the `data` folder:

   - If you only need the following dataset:
     - [Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html): is a widely-used dataset in video anomaly detection, it contains two parts: Ped1 and Ped2. Ped1 contains 34 normal training videos and 36 testing videos, and Ped2 contains 16 normal training videos and 12 testing videos. As a result of that most methods only use the Ped2 part, we also use the Ped2 part in our tools to demonstrate the accuracy of reproduced methods
     - [Avenue](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf):contains anomalies such as strange actions, wrong directions, and abnormal objects, and it also is a widely-used dataset in video anomaly detection. It contains 16 training and 21 testing video clips
     - [ShanghaiTech Campus](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf): is a very challenging dataset whose anomalies are diverse and realistic. It contains normal 330 training videos and 107 testing videos, which is captured from 13 different scenes
    You can choose to download the data from this [link](https://github.com/StevenLiuWen/ano_pred_cvpr2018), which is provided by previous researchers 
     

3. Change the structure of the data:

   - Please change the structure of the data into the following structure.

   ```
   --data
   |--dataset1
     |--training
       |--frames
         |--video1
           |--1.jpg
           |--2.jpg
           ...
         |--video2
     |--testing
       |--frames
       
   ```

   - Please put the annotations in the path which is written in the configuration file.
   - Please check the data path in the configuration file.
