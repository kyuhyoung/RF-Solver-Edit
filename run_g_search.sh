#!/bin/bash
# s75_inj73 guidance search: g1.5, g3.0, g4.0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_g_search.log"
mkdir -p "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

run_job() {
    local gpu=$1 guidance=$2
    local tag="s75_inj73_g${guidance}"
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 75 --inject 73 --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --gamma 0.7 \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "${LOG_DIR}/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done"
}

echo "============================================="
echo "  s75_inj73 guidance search: g1.5, g3.0, g4.0"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Batch 1: g1.5 and g3.0 simultaneously
run_job 7 1.5 &
run_job 6 3.0 &
wait

# Batch 2: g4.0
run_job 7 4.0
wait

echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lh "$OUTPUT_DIR"/s75_inj73_g*.tif
