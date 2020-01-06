#!/bin/bash

directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
network=aa274_net
num_ports=64

# Base Docker command
cmd=( \
	docker run -it --rm --init --privileged \
	--mount type=bind,src=$directory/catkin_ws,dst=/home/$USER/catkin_ws \
	--mount type=bind,src=$directory/.ros,dst=/home/$USER/.ros \
	--net $network \
	--user $USER)

# Parse arguments
while [[ $# -ge 2 && $1 == -* ]]; do
	if [[ $1 == "--display" ]]; then
		# Display argument is not passed to Docker
		display=$2
		shift
		shift
		continue
	elif [[ $1 == "--vncport" ]]; then
		# vncport argument is not passed to Docker
		vncport=$2
		shift
		shift
		continue
	elif [[ $1 == "--rosmaster" ]]; then
		# rosmaster argument is not passed to Docker
		rosmaster=$2
		shift
		shift
		continue
	elif [[ $1 == "--rosport" ]]; then
		# rosport argument is not passed to Docker
		rosport=$2
		shift
		shift
		continue
	fi

	# Add arguments to Docker command
	cmd+=($1 $2)
	shift
	shift
done

# Display options
if [[ -n $display ]]; then

	# Set vncport to default (display + 5900) if display > 0
	if [[ -z $vncport ]]; then
		vncport=$(($display + 5900))
	fi

	# Set VNC settings
	if [[ -n $vncport ]]; then
		cmd+=(-p $vncport:$vncport)

		# Prompt VNC password
		printf "\nRunning a VNC instance at localhost:${vncport} with the password created below.\n\n"
		while [[ ${#password} -lt 6 ]]; do
			read -p "VNC password (6-8 characters): " -s password
		done
		printf "\n\n"

		cmd+=( \
			--env "VNCPASSWD=$password" \
			--env "VNCPORT=$vncport")
	fi
	cmd+=(--env "DISPLAY=:$display")
elif [[ -n $DISPLAY ]]; then
	if [[ $(uname) == 'Darwin' ]]; then
		cmd+=(--env "DISPLAY=host.docker.internal:0")
	else
		cmd+=(--env "DISPLAY=$DISPLAY")
	fi
fi

# Set rosport to default (11311)
if [[ -z $rosport ]]; then
	rosport=11311
fi

# Set rosmaster to default (master)
if [[ -z $rosmaster ]]; then
	rosmaster="master"
fi

# Configure ROS ports
if [[ "$rosmaster" != "master" ]]; then
	if [[ $(uname) == "Linux" ]]; then
		rosmaster=$(ping -c 1 $rosmaster | head -2 | tail -1 | cut -d ' ' -f 4 | cut -d ':' -f 1)
		cmd+=(--env "ROS_IP=$(hostname -I | grep -oP '192.168.1.\d+')")
	elif [[ $(uname) == "Darwin" ]]; then
		ping -c1 $HOSTNAME
		if [[ "$?" == 0 ]]; then
			cmd+=(--env "ROS_HOSTNAME=$HOSTNAME")
		else
			cmd+=(--env "ROS_IP=$(ifconfig | awk '/inet / {print $2}' | grep 192.168.1 | head)")
		fi
	else
		cmd+=(--env "ROS_HOSTNAME=$HOSTNAME")
	fi
	num_containers=$(docker container ls | grep aa274 | wc -l)
	port_mapping_start=$((32768 + num_containers * num_ports))
	port_mapping_end=$((port_mapping_start + num_ports - 1))
	port_mapping_range="$port_mapping_start-$port_mapping_end"
	cmd_ports=(--publish "$port_mapping_range:$port_mapping_range")
	cmd_ports+=(--env "PORT_MAPPING_START=$port_mapping_start")
	cmd_ports+=(--env "PORT_MAPPING_END=$port_mapping_end")
fi
cmd+=(--env "ROS_MASTER_URI=http://$rosmaster:$rosport")

# Application specific options
args="$@"
if [[ $* == *roscore* ]]; then
	# Name the roscore container 'master' so that other nodes can reach it
	cmd+=(--name $rosmaster)
elif [[ $* == *jupyter* ]]; then
	cmd+=(--publish 8888:8888)
	args+=(--ip=0.0.0.0)
fi

# Run command
echo ${cmd[@]} ${cmd_ports[@]}
echo ${args[@]}
${cmd[@]} ${cmd_ports[@]} aa274 ${args[@]}
while [ $? -eq 125 ]; do
	port_mapping_start=$((port_mapping_start + num_ports))
	port_mapping_end=$((port_mapping_start + num_ports - 1))
	port_mapping_range="$port_mapping_start-$port_mapping_end"
	cmd_ports=(--publish "$port_mapping_range:$port_mapping_range")
	cmd_ports+=(--env "PORT_MAPPING_START=$port_mapping_start")
	cmd_ports+=(--env "PORT_MAPPING_END=$port_mapping_end")
	echo ${cmd[@]} ${cmd_ports[@]}
	${cmd[@]} ${cmd_ports[@]} aa274 ${args[@]}
done
