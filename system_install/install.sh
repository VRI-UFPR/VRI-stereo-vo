#/bin/bash

distro=$(lsb_release -si)
if [ "$distro" != "Ubuntu" ]; then
    echo "Unsupported distribution '$distro'. Skipping docker installation..."
else
    # Remove old conflicting packages
    for pkg in docker.io docker-doc docker-compose podman-docker containerd runc
    do 
        sudo apt-get remove $pkg
    done

    # Add Docker's official GPG key
    sudo apt-get update -y
    sudo apt-get install ca-certificates curl gnupg -y
    sudo install -m 0755 -d /etc/apt/keyrings -y
    curl -fsSL https://download.docker.com/linux/raspbian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up Docker's Apt repository
    echo \
        "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/raspbian \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y

    # Install Docker Engine
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-compose -y

    # Post-installation steps
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
fi

# Build image
default_install_type="nano"
read -p "Select installation type (nano or desktop): " install_type
install_type=${install_type:-$default_install_type}

if [ "$install_type" == "nano" ]; then
    docker build -f Dockerfile.nano -t vri-ufpr/stereo-vo:latest .
elif [ "$install_type" == "desktop" ]; then
    docker build -f Dockerfile.desktop -t vri-ufpr/stereo-vo:latest .
else
    echo "Invalid installation type"
fi
