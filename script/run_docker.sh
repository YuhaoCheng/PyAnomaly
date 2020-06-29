#! /bin/bash
data_path="/export/home/chengyh/data"
model_path="/export/home/chengyh/pretrained_model"
name="pyanomaly"
func() {
    echo "Usage:"
    echo "run_docekr.sh [-dp DATA PATH] [-mp MODE PATH] [-n NAME]"
    echo "Description:"
    echo "DATA PATH, the path of the data."
    echo "MODE PATH, the path of pretrain models path."
    exit -1
}

while getopts "dp:mp:h" opt
do
    case $opt in 
    dp) 
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

nvidia-docker run -d --name $name -v $data_path:/home/appuser/pyanomaly_docker/data -v $model_path:/home/appuser/pyanomaly_docker/pretrained_model pyanomaly:test /bin/bash