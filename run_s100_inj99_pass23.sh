#!/bin/bash
# s100_inj99 pass2 and pass3 on GPU 7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_s100_inj99_pass23.log"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT="Clean high resolution satellite image with sharp details and reduced noise"

GPU=7
TAG="s100_inj99"

echo "============================================="
echo "  ${TAG} pass2 + pass3 on GPU $GPU"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Pass 2
P1="${OUTPUT_DIR}/${TAG}_pass1.tif"
P2="${OUTPUT_DIR}/${TAG}_pass2.tif"
echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass2 started"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
    --input "$P1" --output "$P2" \
    --num_steps 75 --inject 74 --guidance 4 \
    --tile_size 4096 --overlap 0 --no_stretch \
    --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
    > "${LOG_DIR}/${TAG}_pass2.log" 2>&1 && \
echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass2 done" || \
{ echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass2 FAILED"; exit 1; }

# Pass 3
P3="${OUTPUT_DIR}/${TAG}_pass3.tif"
echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass3 started"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
    --input "$P2" --output "$P3" \
    --num_steps 75 --inject 74 --guidance 4 \
    --tile_size 4096 --overlap 0 --no_stretch \
    --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
    > "${LOG_DIR}/${TAG}_pass3.log" 2>&1 && \
echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass3 done" || \
echo "[$(date '+%H:%M:%S')] GPU $GPU: ${TAG}_pass3 FAILED"

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lh "$P1" "$P2" "$P3" 2>/dev/null
