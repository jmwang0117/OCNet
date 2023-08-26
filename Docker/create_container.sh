docker create -t --name robot --runtime nvidia  -e NVIDIA_VISIBLE_DEVICES=all \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e CYCLONEDDS_URI=$CYCLONEDDS_URI \
    -e XAUTHORITY=$XAUTHORITY \
    --network host \
    --ipc host \
    --privileged \
    --shm-size 8G \
    -v /dev:/dev \
    -v /home/jmwang/datasets:/root/datasets \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $XAUTHORITY:$XAUTHORITY \
    robot bash
