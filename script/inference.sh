#! /bin/bash
# Initialize the paramaters
project_path='/export/home/chengyh/PyAnomaly'
model="amc"
dataset="ped2"
verbose='default'
config=""
inference_model='./final_model/avenue_amc_cfg@amc_avenue#2020-03-02-22-05#0.835^amc#amc.pth'
GPUS="0"
MULTIGPUS=False
func() {
    echo "Usage:"
    echo "inference.sh [-m MODEL] [-d DATASET] [-p PROJRCT] [-g GPUS] [-c CONFIG] [-v VERBOESE] [-f INFERENCE MODEL]"
    echo "Description:"
    echo "MODEL,the name of model."
    echo "DATASET,the name of dataset."
    echo "PROJRCT,the path of project."
    echo "GPUS, the ids of gpus."
    echo "CONFIG, the name of configure file."
    echo "VERBOSE, the verbose of training of inference."
    echo "INFERENCE MODEL, the inference model."
    exit -1
}
# NUM_GPUS=1
#-----------------useage-------------------------
# sh xxx.sh GPUS project_path configure_name verbose
#------------------------------------------------
while getopts "m:d:p:g:c:v:f:" opt
do
    case $opt in 
    m) 
    model=$OPTARG
    echo "model:"$model
    ;;
    d)
    dataset=$OPTARG
    echo "dataset":$dataset
    ;;
    p)
    project_path=$OPTARG
    echo "project_path:"$project_path
    ;;
    g)
    GPUS=$OPTARG
    echo "GPUS:"$GPU
    ;;
    c)
    config=$OPTARG
    echo "config:"$config
    ;;
    v)
    verobose=$OPTARG
    echo "verbose:"$verobose
    ;;
    f)
    inference_model=$OPTARG
    echo "inference_model:"$inference_model
    ;;
    h)
    func
    ;;
    ?)
    echo "Unknow args"
    func
    exit 1
    ;;
    esac
done

GPUS_LIST=(${GPUS//,/ })
NUM_GPUS=${#GPUS_LIST[@]}

if test $NUM_GPUS -gt 1
then
    MULTIGPUS=True
    echo 'Use multi gpus, the num of gpus:'$NUM_GPUS
else
    echo 'Use single gpus, the device id is:'$GPUS
fi

if test -n "$config" # $config is not null, it is true 
then
    echo "the config file is:" $config
else
    config=$dataset'_default.yaml'
    echo "Using the default config:"$config
fi

cfg_folder=$model'/'$dataset # e.g amc/avenue
echo 'The cfg folder is:'$cfg_folder
echo "Using the config:"$config

echo "Using the project path:"$project_path
echo "Using the inference model:"$inference_model
echo "Using the verbose:"$verbose
# Go to the main file location
cd $project_path


CUDA_VISIBLE_DEVICES=$GPUS python main.py --project_path $project_path --cfg_folder $cfg_folder --cfg_name $config --verbose $verbose  --inference_model $inference_model\
                                                SYSTEM.multigpus $MULTIGPUS \
                                                SYSTEM.num_gpus $NUM_GPUS 