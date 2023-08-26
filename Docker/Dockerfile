FROM nvidia/cudagl:11.4.2-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND noninteractive

# Basic dependencies and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    less \
    emacs \
    tmux \
    bash-completion \
    command-not-found \
    software-properties-common \
    xsel \
    xdg-user-dirs \
    python-pip \
    python-protobuf \
    python-apt \
    python-pexpect \
    pcl-tools \
    libomp5 \
    vulkan-utils


# Install tmux, vim and git
RUN apt-get install -y tmux vim git

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["bash"]
