#! /bin/bash
name="pyanomaly"
func() {
    echo "Usage:"
    echo "exec_docekr.sh [-n NAME]"
    echo "Description:"
    echo "NAME, the name of the docker container"
    exit -1
}

while getopts "n:h" opt
do
    case $opt in 
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

nvidia-docker exec -ti $name /bin/bash