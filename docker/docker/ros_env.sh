#!/bin/bash

# Limit local port range available to ROS
if [[ ! -z $PORT_MAPPING_START ]]; then
	sudo sysctl -w net.ipv4.ip_local_port_range="$PORT_MAPPING_START $PORT_MAPPING_END"
fi

# Set ROS build mode
if [[ -z $build ]]; then
	build=devel
fi

# Setup ros environment
source /opt/ros/kinetic/setup.bash
if [[ -f ~/catkin_ws/$build/setup.bash ]]; then
	source ~/catkin_ws/$build/setup.bash
fi

# Export environment variables
if [[ -z $ROS_MASTER_URI ]]; then
	export ROS_MASTER_URI=http://master:11311
fi
export TURTLEBOT3_MODEL=burger
