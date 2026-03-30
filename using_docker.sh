#!/bin/bash

# RF-Solver-Edit Docker Build and Run Script
#
# Usage:
#   ./using_docker.sh            # build + run (default)
#   ./using_docker.sh -nc        # build without cache + run
#   ./using_docker.sh -nb        # skip build, run only
#   ./using_docker.sh -r         # reattach to existing container (fast)

# Log file
LOGFILE="$(dirname "$0")/using_docker.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging: terminal gets color, log file gets plain text
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# Check for options
NO_CACHE=""
SKIP_BUILD=false
REATTACH=false

for arg in "$@"; do
    case $arg in
        -nc)
            NO_CACHE="--no-cache"
            echo -e "${YELLOW}Building without cache...${NC}"
            ;;
        -nb)
            SKIP_BUILD=true
            echo -e "${YELLOW}Skipping build...${NC}"
            ;;
        -r)
            REATTACH=true
            echo -e "${YELLOW}Reattaching to existing container...${NC}"
            ;;
    esac
done

# Reattach to existing container
if [ "$REATTACH" = true ]; then
    existing=$(docker ps -a --filter "name=${docker_name}_" --filter "status=exited" --format "{{.Names}}" | head -1)
    if [ -z "$existing" ]; then
        existing=$(docker ps --filter "name=${docker_name}_" --format "{{.Names}}" | head -1)
        if [ -z "$existing" ]; then
            echo -e "${RED}No existing container found. Run without -r first.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Attaching to running container: ${existing}${NC}"
        exec 1>/dev/tty 2>/dev/tty
        docker exec -it ${existing} bash
    else
        echo -e "${GREEN}Restarting container: ${existing}${NC}"
        exec 1>/dev/tty 2>/dev/tty
        docker start -ai ${existing}
    fi
    exit 0
fi

if [ "$SKIP_BUILD" = false ] && [ -z "$NO_CACHE" ]; then
    echo -e "${YELLOW}Building with cache...${NC}"
fi

docker_name=rf-solver-edit
container_name=${docker_name}_$(date +%y%m%d_%H%M%S)
dir_cur=/workspace/${PWD##*/}
dir_data=/data/dataset

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  RF-Solver-Edit Docker Container${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${YELLOW}Docker image: ${docker_name}${NC}"

####################################################################################
#   docker build
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}Building docker image...${NC}"

    if [ -n "$NO_CACHE" ]; then
        docker build --platform linux/amd64 --force-rm --shm-size=64g ${NO_CACHE} --build-arg CACHEBUST=$(date +%s) -t ${docker_name} -f Dockerfile .
    else
        docker build --platform linux/amd64 --force-rm --shm-size=64g -t ${docker_name} -f Dockerfile .
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker build failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}Build complete${NC}"
else
    echo -e "${YELLOW}Build skipped due to -nb option${NC}"
fi

echo -e "${YELLOW}Data mount:   ${dir_data} -> /data${NC}"
echo -e "${YELLOW}Work dir:     ${dir_cur}${NC}"
echo ""
echo -e "${GREEN}Entering container...${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Restore stdout/stderr (stop logging to using_docker.log before interactive shell)
exec 1>/dev/tty 2>/dev/tty

#   docker run
# Remove old container with same name if exists
docker rm -f ${container_name} 2>/dev/null

docker run -it --name ${container_name} \
    --shm-size=64g \
    --gpus all \
    -e QT_DEBUG_PLUGINS=1 \
    --net=host \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    --privileged \
    -w ${dir_cur} \
    -v ${dir_data}:/data \
    -v $PWD:${dir_cur} \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v /etc/sudoers.d:/etc/sudoers.d:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /data/kevin_workspace/.cache/huggingface:/root/.cache/huggingface \
    ${docker_name} bash
