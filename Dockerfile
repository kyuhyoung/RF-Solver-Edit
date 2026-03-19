FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip

# Install PyTorch (cu118)
RUN pip install --no-cache-dir \
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
RUN pip install --no-cache-dir \
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

WORKDIR /workspace

CMD ["/bin/bash"]
