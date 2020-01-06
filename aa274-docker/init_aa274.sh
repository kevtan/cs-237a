#!/bin/bash

# Create Catkin workspace
mkdir -p .ros
mkdir -p catkin_ws/src
touch catkin_ws/.bashrc
cd catkin_ws/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone https://github.com/StanfordASL/asl_turtlebot.git

# Create docker network
docker network create --driver bridge aa274_net

# Download latest ros image
docker pull osrf/ros:kinetic-desktop-full
