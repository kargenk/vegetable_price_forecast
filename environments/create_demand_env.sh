IMAGE_NAME=$(whoami)_demand:latest
CONTAINER_NAME="kg_demand"

# dockerイメージのビルド
docker build ./ \
    --tag ${IMAGE_NAME} \
    --build-arg USERNAME=$USER \
    --build-arg GROUPNAME=$USER \
    --build-arg UID=$(id -u $USER) \
    --build-arg GID=$(id -g $USER) \
    --build-arg PASSWORD=testpass

# dockerコンテナの立ち上げ
docker run \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --net=host \
    -h `hostname` \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/$(whoami)/demand_forecasting/:/home/$USER/demand_forecasting \
    -it -d --shm-size=32gb ${IMAGE_NAME} /bin/zsh