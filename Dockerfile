FROM osrf/ros:noetic-desktop

ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV TORCH_CUDA_ARCH_LIST="10.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    fluxbox \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-image-transport-plugins \
    ros-noetic-compressed-image-transport \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    tqdm \
    gnupg \
    pycryptodomex \
    rosbags


WORKDIR /ros_ws

# Automate environment sourcing for all new terminals
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]