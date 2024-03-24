# Install nvidia-docker:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.5/install-guide.html#installing-with-yum-or-dnf
sudo yum-config-manager --disable amzn2-graphics
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker ssm-user
# docker run --runtime=nvidia -it nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 bash