#! /bin/bash
project_path='/export/home/chengyh/PyAnomaly'
model="amc"
dataset="ped2"
verbose='default'
config=""
GPUS="0"
MULTIGPUS=False
func() {
    echo "Usage:"
    echo "train.sh [-m MODEL] [-d DATASET] [-p PROJRCT] [-g GPUS] [-c CONFIG] [-v VERBOESE]"
    echo "Description:"
    echo "MODEL,the name of model."
    echo "DATASET,the name of dataset."
    echo "PROJRCT,the path of project."
    echo "GPUS, the ids of gpus."
    echo "CONFIG, the name of configure file."
    echo "VERBOSE, the verbose of training of inference."
    exit -1
}

#-----------------useage-------------------------
# sh train.sh [-m MODEL] [-d DATASET] [-p PROJRCT] [-g GPUS] [-c CONFIG] [-v VERBOESE]
#------------------------------------------------
while getopts "m:d:p:g:c:v:h" opt
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
    echo "GPUS:"$GPUS
    ;;
    c)
    config=$OPTARG
    echo "config:"$config
    ;;
    v)
    verobose=$OPTARG
    echo "verbose:"$verobose
    ;;
    h)
    func
    ;;
    ?)
    func
    echo "Unknow args"
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

echo "Using the project path:"$project_path
cfg_folder=$model'/'$dataset # e.g amc/avenue
echo 'The cfg folder is:'$cfg_folder

if test -n "$config" # $config is not null, it is true 
then
    echo "the config file is:" $config
else
    config=$dataset'_default.yaml'
    echo "Using the default config:"$config
fi

echo "Using the verbose:"$verbose

# Go to the main file location
cd $project_path

CUDA_VISIBLE_DEVICES=$GPUS python main.py --project_path $project_path --cfg_folder $cfg_folder --cfg_name $config --verbose $verbose  \
                                                SYSTEM.multigpus $MULTIGPUS \
                                                SYSTEM.num_gpus $NUM_GPUS \
                                                # MODEL.flow_model_path $project_path'/pretrained_model/FlowNet2_checkpoint.pth.tar' \
                                                # MODEL.discriminator_channels [128,256,512,512]
