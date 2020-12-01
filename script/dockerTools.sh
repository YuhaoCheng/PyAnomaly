#! /bin/bash
func() {
    echo "Usage:"
    echo "dockerTools.sh [-p] [-b] [-r] [-e]"
    echo "Description:"
    echo "-p, pull the docker image"
    echo "-b, build the docker image"
    echo "-r, run the docker container in the daemon"
    echo "-e, execute the docker container"
    exit -1
}

while getopts "p:b:r:e" opt
do
    case $opt in 
    p) 
    data_path=$OPTARG
    echo "data path:"$data_path
    ;;
    mp)
    model_path=$OPTARG
    echo "model path":$model_path
    ;;
    n)
    name=$OPTARG
    echo "name":$name
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
