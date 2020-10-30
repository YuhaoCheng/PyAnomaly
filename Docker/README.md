# Docker 

## Introduction

- In order to make it easy to use, we create the docker image file
- `.condarc` is the channels of `conda`, we choose `tuna` source. Users can change it. 
- If you want to change to the default sources, please run use this [condarc](https://drive.google.com/file/d/1OIqaTbebbIs94Ku_9wraU-BT7g5F_5q8/view?usp=sharing) to replace the original one in the `Docker `folder



## Quick Start

We provide a function to quickly use the docker image and container

```
sh dockerTools.sh --help
```



## Get the docker image

### Docker Hub

We have put the docker image on the docker hub, so you can use the following command to pull the docker image

```docker
docker pull roviocyh/pytorch16-cu101:latest
```

However,  in order to make the docker docker general, we do not install some requirements of this project. If you want to directly use the project in your application ,please use the following command to pull the docker image and  build the container directly. 

```docker
docker pull roviocyh/pyanomaly:latest
```

if you use this command,  you can skip the building image process. 

### Build image locally 

```shell
$PATH = path/to/PyAnomaly
cd docker
nvidia-docker build -t pyanomaly:test .
```

## Run the docker container

### Start the container 
```
sh ./script/run_docker.sh
```

### Execute the docker image

```
sh ./script/exec_docker.sh
```

### Use in docker 

```
bash ./script/train.sh -p /home/appuser/pyanomaly_docker -d DATASET_NAME -m MODEL_NAME
```

## Note

- Changes in docker container can't be saved, expect users commit them
- We not suggest users to make changes in Docker, which will make the Docker image larger and break the rules of building good images