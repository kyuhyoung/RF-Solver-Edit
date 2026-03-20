#!/bin/bash

# RF-Solver-Edit Conda Environment Setup Script
#
# Usage:
#   ./using_conda.sh            # create env + activate
#   ./using_conda.sh -r         # activate existing env (fast)

set -e

# Log file
LOGFILE="$(dirname "$0")/using_conda.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

ENV_NAME=rfsolver
ENV_PATH=/data/kevin_workspace/envs/rfsolver

# Check for options
REUSE=false
for arg in "$@"; do
    case $arg in
        -r)
            REUSE=true
            ;;
    esac
done

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo -e "${RED}conda not found. Install Miniconda/Anaconda first.${NC}"
    exit 1
fi

# Source conda for subshell
eval "$(conda shell.bash hook)"

if [ "$REUSE" = true ]; then
    if [ -d "${ENV_PATH}" ]; then
        echo -e "${GREEN}Activating existing environment: ${ENV_PATH}${NC}"
        exec 1>/dev/tty 2>/dev/tty
        SHELL_NAME=$(basename "$SHELL")
        if [ "$SHELL_NAME" = "fish" ]; then
            exec fish -C "conda activate ${ENV_PATH}; echo -e '(${ENV_NAME}) environment active. Run ./usage.sh'"
        else
            exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active. Run ./usage.sh${NC}'")
        fi
    else
        echo -e "${RED}Environment not found at ${ENV_PATH}. Run without -r first.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  RF-Solver-Edit Conda Environment Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

####################################################################################
#   Create conda environment
if [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Environment already exists at ${ENV_PATH}. Removing...${NC}"
    conda env remove -p ${ENV_PATH} -y
fi

echo -e "${YELLOW}Creating conda environment: ${ENV_PATH} (python 3.10)${NC}"
mkdir -p "$(dirname ${ENV_PATH})"
conda create -p ${ENV_PATH} python=3.10 -y

conda activate ${ENV_PATH}

####################################################################################
#   Install dependencies
echo -e "${YELLOW}Installing PyTorch (CUDA 11.8)...${NC}"
pip install --no-cache-dir \
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --no-cache-dir \
    einops==0.8.0 \
    accelerate==0.34.2 \
    transformers==4.41.2 \
    huggingface-hub==0.24.6 \
    diffusers \
    sentencepiece \
    opencv-python \
    fire \
    rasterio \
    scipy \
    matplotlib \
    omegaconf \
    timm \
    invisible-watermark \
    safetensors \
    requests \
    tqdm

####################################################################################
#   System dependencies check
echo -e "${YELLOW}Checking system libraries...${NC}"
MISSING=""
dpkg -s gdal-bin &>/dev/null || MISSING="$MISSING gdal-bin"
dpkg -s libgdal-dev &>/dev/null || MISSING="$MISSING libgdal-dev"
dpkg -s libgl1-mesa-glx &>/dev/null || MISSING="$MISSING libgl1-mesa-glx"

if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}Installing missing system packages:${MISSING}${NC}"
    sudo apt-get update && sudo apt-get install -y $MISSING
else
    echo -e "${GREEN}All system libraries present${NC}"
fi

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${YELLOW}To activate:  conda activate ${ENV_NAME}${NC}"
echo -e "${YELLOW}Then run:     ./usage.sh${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Drop into activated shell
exec 1>/dev/tty 2>/dev/tty
SHELL_NAME=$(basename "$SHELL")
if [ "$SHELL_NAME" = "fish" ]; then
    exec fish -C "conda activate ${ENV_PATH}; echo -e '(${ENV_NAME}) environment active. Run ./usage.sh'"
else
    exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active. Run ./usage.sh${NC}'")
fi
