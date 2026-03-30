#!/bin/bash
# Retry failed steps=50 grid search jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search"
LOG_DIR="${SCRIPT_DIR}/output/grid_search/logs"

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

# Failed s50 jobs (skip s50_inj49_g0.5 which succeeded)
JOBS=(
    "50:45:0.5"
    "50:45:1.0"
    "50:45:2.0"
    "50:47:0.5"
    "50:47:1.0"
    "50:47:2.0"
    "50:49:1.0"
    "50:49:2.0"
)

echo "============================================="
echo "  Retry: ${#JOBS[@]} failed s50 jobs"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

RUNNING_PIDS=()
RUNNING_GPUS=()

wait_for_gpu() {
    while true; do
        for idx in "${!RUNNING_PIDS[@]}"; do
            pid=${RUNNING_PIDS[$idx]}
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                local gpu=${RUNNING_GPUS[$idx]}
                unset 'RUNNING_PIDS[$idx]'
                unset 'RUNNING_GPUS[$idx]'
                RUNNING_PIDS=("${RUNNING_PIDS[@]}")
                RUNNING_GPUS=("${RUNNING_GPUS[@]}")
                echo "$gpu"
                return
            fi
        done
        sleep 2
    done
}

gpu=0
for job in "${JOBS[@]}"; do
    IFS=: read -r steps inject guidance <<< "$job"
    tag="s${steps}_inj${inject}_g${guidance}"
    outfile="${OUTPUT_DIR}/${tag}.tif"
    logfile="${LOG_DIR}/${tag}.log"

    if [ ${#RUNNING_PIDS[@]} -ge 8 ]; then
        gpu=$(wait_for_gpu)
    else
        gpu=${#RUNNING_PIDS[@]}
    fi

    echo "[$(date '+%H:%M:%S')] GPU $gpu: steps=$steps inject=$inject guidance=$guidance"

    CUDA_VISIBLE_DEVICES=$gpu python3 -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "$outfile" \
        --num_steps "$steps" \
        --inject "$inject" \
        --guidance "$guidance" \
        --tile_size 4096 \
        --overlap 0 \
        --gamma 0.7 \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 &

    RUNNING_PIDS+=($!)
    RUNNING_GPUS+=($gpu)
done

echo ""
echo "All jobs launched. Waiting for completion..."
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lh "$OUTPUT_DIR"/s50*.tif 2>/dev/null
