#!/bin/bash

cd docker
docker build . --tag=aa274 \
	--build-arg uid=$UID \
	--build-arg user=$USER \
	--build-arg gid=$(id -g) &&
docker rmi $(docker images -qa -f 'dangling=true')
