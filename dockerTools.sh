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

