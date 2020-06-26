### Introduction
- Please download the model in the table
- Please change the `inference model` path when you use these models
- Please follow the configurations in the `experiments`
- The results in parenthesis are proposed in the original papers

|   Method    | Ped2 | Avenue | Shanghai |
| :---------: | :--: | :----: | :------: |
|   STAE[1]   | [90.1]()(91.2) | [75.2]()(77.1) | - |
|  MemAE[2]   | [92.3]()(94.1) | [82.9]()(83.3) | [70.0]() (71.2) |
|   OCAE[5]   | [95.3]()(97.8) | [90.0](https://drive.google.com/file/d/13cF0XyM-hJiN9fWG1diZIjPQ7N6KMBKt/view?usp=sharing)(90.4) | [81.3]() (84.9) |
|  AnoPCN[7]  | [94.2]()(96.8) | [87.1](https://drive.google.com/file/d/1au9eFZ5CJzEcoJhKCPkO6Or7211URPDu/view?usp=sharing)(86.2) | [72.1]() (73.6) |
|   AMC[13]   | [97.4](https://drive.google.com/file/d/1spUSv_o5RIHc3x2NXTCVAQvpOz7vXoZ6/view?usp=sharing)(96.2) | [83.5](https://drive.google.com/file/d/1BuaPqsGvUxOb0Vo-b-uE4xYhRDaZ6J30/view?usp=sharing) (86.9) | - |
| AnoPred[14] | [93.9]()(96.4) | [86.7](https://drive.google.com/file/d/1BaqiRyjOTudF5ja25O6YlgQl8S7LrIFz/view?usp=sharing) (85.1) | [69.9]() (72.8) |

Note: We found that different GPU type may inference the results. 

### Details

#### Optical Flow

- Some methods using other optical flow methods to get the optical in video

- We choose to use the different optical flow methods implemented in PyTorch

  - [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)
  - [LiteFlowNet](https://github.com/sniklaus/pytorch-liteflownet)

- We use the pre-trained models in these methods, which can be download from their GitHub repo. 

  | Method  | Optical Flow Method | Ped2                                                         | Avenue | Shanghai |
  | ------- | ------------------- | ------------------------------------------------------------ | ------ | -------- |
  | AMC     | FlowNet2            | [97.4](https://drive.google.com/file/d/1spUSv_o5RIHc3x2NXTCVAQvpOz7vXoZ6/view?usp=sharing) |        |          |
  | AMC     | LiteFlowNet         |                                                              |        |          |
  | AnoPred | FlowNet2            |                                                              |        |          |
  | AnoPred | LiteFlowNet         |                                                              |        |          |
  | AnoPCN  | FlowNet2            |                                                              |        |          |
  | AnoPCN  | LiteFlowNet         |                                                              |        |          |

  