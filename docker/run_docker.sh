#!/bin/sh
echo $0

if [ "$1" = "" ]
then
    echo "Input container name : ./run.sh <container_name>"
    exit 1
fi

docker run --runtime=nvidia -ti --rm --ipc host --net host\
	--shm-size=20g \
	--name $1 \
	-v $HOME:$HOME \
	-v /media:/media \
	-v /raid:/raid \
	-v /work:/work \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-u $(id -u):$(id -g) \
	-h docker \
	-w $HOME \
	kaggle_rsna:reproducebility /bin/bash
