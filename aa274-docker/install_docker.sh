#!/bin/bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Create Docker group to avoid need for sudo
sudo groupadd docker
sudo usermod -aG docker $USER

echo "\nDocker successfully installed. Please restart the computer for changes to take effect.\n"
