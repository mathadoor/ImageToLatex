#!/bin/bash

# Uninstall older versions of docker
sudo apt-get remove -y docker docker-engine docker.io containerd run

# Set up the repository
# Update the apt package index and install packages to allow apt to use a repository over HTTPS
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker’s official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
# Update the package index
sudo apt-get update

# Install Docker Engine, contained, and Docker Compose
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify Docker Engine Installation is Successful
sudo docker run hello-world

# Add Local User to Sudo Group
# Add Docker group if it doesn’t exist already
sudo groupadd docker

# Add the connected user $USER to the docker group
sudo gpasswd -a $USER docker
