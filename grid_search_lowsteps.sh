#!/bin/bash
# Low steps exploration: inject = steps-1, 3pass each
# GPU 2-7, all 6 combos in parallel

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_lowsteps.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PROMPT_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT_PASS="Clean high resolution satellite image with sharp details and reduced noise"

run_full_pipeline() {
    local gpu=$1 steps=$2 inject=$3
    local tag="s${steps}_inj${inject}"

    # Pass 1
    local p1="${OUTPUT_DIR}/${tag}_pass1.tif"
    if [ ! -f "$p1" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$INPUT" --output "$p1" \
            --num_steps "$steps" --inject "$inject" --guidance 4 \
            --tile_size 4096 --overlap 0 --gamma 0.6 \
            --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
            > "${LOG_DIR}/${tag}_pass1.log" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 done" || \
        { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 FAILED"; return 1; }
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag}_pass1 (exists)"
    fi

    # Pass 2
    local p2="${OUTPUT_DIR}/${tag}_pass2.tif"
    if [ ! -f "$p2" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$p1" --output "$p2" \
            --num_steps 75 --inject 74 --guidance 4 \
            --tile_size 4096 --overlap 0 --no_stretch \
            --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
            > "${LOG_DIR}/${tag}_pass2.log" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 done" || \
        { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 FAILED"; return 1; }
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag}_pass2 (exists)"
    fi

    # Pass 3
    local p3="${OUTPUT_DIR}/${tag}_pass3.tif"
    if [ ! -f "$p3" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$p2" --output "$p3" \
            --num_steps 75 --inject 74 --guidance 4 \
            --tile_size 4096 --overlap 0 --no_stretch \
            --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
            > "${LOG_DIR}/${tag}_pass3.log" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 done" || \
        { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 FAILED"; return 1; }
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag}_pass3 (exists)"
    fi
}

echo "============================================="
echo "  Low Steps Exploration: 6 combos × 3pass"
echo "  inject = steps-1, gamma=0.6, g4"
echo "  GPUs: 2,3,4,5,6,7"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

run_full_pipeline 2 20 19 &
run_full_pipeline 3 30 29 &
run_full_pipeline 4 40 39 &
run_full_pipeline 5 50 49 &
run_full_pipeline 6 60 59 &
run_full_pipeline 7 70 69 &

wait

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lht "$OUTPUT_DIR"/s{20,30,40,50,60,70}_inj*_pass*.tif 2>/dev/null
