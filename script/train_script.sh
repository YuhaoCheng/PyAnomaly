#! /bin/bash
project_path='/export/home/chengyh/Anomaly_DA'
verbose='default'
MULTIGPUS=False
# NUM_GPUS=1
#-----------------useage-------------------------
# sh xxx.sh GPUS project_path configure_name verbose
#------------------------------------------------
GPUS=$1
# echo $GPUS
GPUS_LIST=(${GPUS//,/ })
NUM_GPUS=${#GPUS_LIST[@]}

if test $NUM_GPUS -gt 1
then
    MULTIGPUS=True
    echo 'Use multi gpus, the num of gpus:'$NUM_GPUS
else
    echo 'Use single gpus, the device id is:'$GPUS
fi

model=$2
echo 'The model name is:'$model

dataset=$3
cfg_folder=$model'/'$dataset # e.g amc/avenue
echo 'The cfg folder is:'$cfg_folder

if test -n "$4" # $2 is not null, it is true 
then
    project_path=$4
    echo "The project path is:" $project_path
else
    echo "Using the default project path:"$project_path
fi

cd $project_path

if test -n "$5" # $2 is not null, it is true 
then
    cfg_name=$5
    echo "the config file is:" $cfg_name
else
    cfg_name=$dataset'_default.yaml'
    echo "Using the default config:"$cfg_name
fi

if test -n "$6"
then
    verbose=$6
    echo "The verbose is:"$verbose
else
    echo "Using the default verbose:"$verbose
fi

CUDA_VISIBLE_DEVICES=$GPUS python main.py --project_path $project_path --cfg_folder $cfg_folder --cfg_name $cfg_name --verbose $verbose  \
                                                SYSTEM.multigpus $MULTIGPUS \
                                                SYSTEM.num_gpus $NUM_GPUS \
                                                MODEL.flow_model_path $project_path'/pretrained_model/FlowNet2_checkpoint.pth.tar' \
                                                MODEL.discriminator_channels [128,256,512,512]
