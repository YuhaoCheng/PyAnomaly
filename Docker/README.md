## Docker 

### Introduction

- In order to make it easy to use, we create the docker image file
- `.condarc` is the channels of `conda`, we choose `tuna` source. Users can change it. 
- If you want to change to the default sources, please run use this [condarc](https://drive.google.com/file/d/1OIqaTbebbIs94Ku_9wraU-BT7g5F_5q8/view?usp=sharing) to replace the original one in the `Docker `folder

### Docker Hub

We have put the docker image on the docker hub, so you can use the following command to pull the docker image

```
docker pull roviocyh/pytorch16-cu101:latest
```

However,  in order to make the docker docker general, we do not install some requirements of this project

### Build Usage

#### Method1

##### Build the docker image

```shell
$PATH = path/to/PyAnomaly
cd docker
nvidia-docker build -t pyanomaly:test .
```

##### Start the docker container

```shell
nvidia-docker run -t -i pyanomaly:test /bin/bash
```

- The `data` folder  and `pretrained_model`should be the location in the host

- We clone the `PyAnomaly` in `pyanomaly_docker` in docker, so the `data` should be `.../pyanomaly_docker/data` 

- Users can use this to start the docker and mount the volumes:

  ```shell
  DATA_PATH=/path/to/dataset
  PRETRAIN_MODEL_PATH=/path/to/pretrained/model
  nvidia-docker run -t -i -v $DATA_PATH:/home/appuser/pyanomaly_docker/data -v $PRETRAIN_MODEL_PATH:/home/appuser/pyanomaly_docker/pretrained_model pyanomaly:test /bin/bash
  ```

##### Use in docker 

```
bash ./script/train.sh -p /home/appuser/pyanomaly_docker -d DATASET_NAME -m MODEL_NAME
```

#### Method2

We provide the quick start shell scripts in `script` folder, users can use these shell to start the docker quickly

##### Build the docker image

```shell
$PATH = path/to/PyAnomaly
cd docker
nvidia-docker build -t pyanomaly:test .
```

##### Start the docker container

```
sh ./script/run_docker.sh
```

##### Execute the docker image

```
sh ./script/exec_docker.sh
```

#### Use in docker 

```
bash ./script/train.sh -p /home/appuser/pyanomaly_docker -d DATASET_NAME -m MODEL_NAME
```



#### Note

- Changes in docker container can't be saved, expect users commit them
- We not suggest users to make changes in Docker, which will make the Docker image larger and break the rules of building good images