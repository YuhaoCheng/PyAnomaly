### Docker 

#### Introduction

- In order to make it easy to use, we create the docker image file

- `.condarc` is the channels of `conda`, we choose `tuna` source. Users can change it. 

#### Usage

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

- The `data` folder and `output` folder should be the location in the host

- We clone the `PyAnomaly` in `pyanomaly_repo` in docker, so the `data` should be `.../pyanomaly_repo/data` and the `output` should be `.../pyanomaly_repo/output`

- Users can use this to start the docker and mount the volumes:

  ```shell
  DATA_PATH=/path/to/dataset
  OUTPUT_PATH=/path/to/output
  nvidia-docker run -t -i -v $DATA_PATH:/home/appuser/pyanomaly_repo/data -v $OUTPUT_PATH:/home/appuser/pyanomaly_repo/output pyanomaly:test /bin/bash
  ```

##### Use in docker 

```
conda activate pyanomaly
```



#### Note

- Changes in docker container can't be saved, expect users commit them
- We not suggest users to make changes in Docker, which will make the Docker image larger and break the rules of building good images