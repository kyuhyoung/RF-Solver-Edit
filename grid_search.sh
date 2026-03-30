#!/bin/bash
# Grid search for RF-Solver-Edit parameters
# Runs up to 8 jobs in parallel (one per GPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search"
LOG_DIR="${SCRIPT_DIR}/output/grid_search/logs"
GRID_LOG="${SCRIPT_DIR}/grid_search.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Redirect all stdout/stderr to grid_search.log (unbuffered) and terminal
exec > >(stdbuf -oL tee -a "$GRID_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

# Grid parameters
STEPS_LIST=(25 50)
GUIDANCE_LIST=(0.5 1.0 2.0)
# inject = steps - offset
INJECT_OFFSETS=(1 3 5)

# Build job list
JOBS=()
for steps in "${STEPS_LIST[@]}"; do
    for offset in "${INJECT_OFFSETS[@]}"; do
        inject=$((steps - offset))
        for guidance in "${GUIDANCE_LIST[@]}"; do
            JOBS+=("${steps}:${inject}:${guidance}")
        done
    done
done

echo "============================================="
echo "  Grid Search: ${#JOBS[@]} combinations"
echo "============================================="
echo ""
for j in "${JOBS[@]}"; do
    IFS=: read -r s i g <<< "$j"
    echo "  steps=$s  inject=$i  guidance=$g"
done
echo ""
echo "Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

GPU_ORDER=(7 6 5 4 3 2 1 0)
RUNNING_PIDS=()
RUNNING_GPUS=()

find_free_gpu() {
    # Find first available GPU in preferred order (7,6,5,...,0)
    # Checks both our own jobs and other processes using nvidia-smi
    local used_gpus=""
    # Our own running jobs
    for idx in "${!RUNNING_PIDS[@]}"; do
        if kill -0 "${RUNNING_PIDS[$idx]}" 2>/dev/null; then
            used_gpus="${used_gpus} ${RUNNING_GPUS[$idx]}"
        fi
    done
    # Other processes on GPUs (from nvidia-smi)
    local nv_used
    nv_used=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$nv_used" -gt 0 ]; then
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
        for g in "${GPU_ORDER[@]}"; do
            local mem
            mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if [ "$mem" -gt 100 ]; then
                used_gpus="${used_gpus} ${g}"
            fi
        done
    fi
    for g in "${GPU_ORDER[@]}"; do
        if ! echo "$used_gpus" | grep -qw "$g"; then
            echo "$g"
            return
        fi
    done
    echo "-1"
}

wait_for_gpu() {
    # Wait until any one of our jobs finishes, clean it up, then find free GPU
    while true; do
        for idx in "${!RUNNING_PIDS[@]}"; do
            pid=${RUNNING_PIDS[$idx]}
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                unset 'RUNNING_PIDS[$idx]'
                unset 'RUNNING_GPUS[$idx]'
                RUNNING_PIDS=("${RUNNING_PIDS[@]}")
                RUNNING_GPUS=("${RUNNING_GPUS[@]}")
                local gpu
                gpu=$(find_free_gpu)
                echo "$gpu"
                return
            fi
        done
        sleep 2
    done
}

for job in "${JOBS[@]}"; do
    IFS=: read -r steps inject guidance <<< "$job"
    tag="s${steps}_inj${inject}_g${guidance}"
    outfile="${OUTPUT_DIR}/${tag}.tif"
    logfile="${LOG_DIR}/${tag}.log"

    # Find a free GPU
    gpu=$(find_free_gpu)
    if [ "$gpu" = "-1" ]; then
        gpu=$(wait_for_gpu)
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

# Wait for all remaining jobs
echo ""
echo "All jobs launched. Waiting for completion..."
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done

echo ""
echo "============================================="
echo "  Grid search complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: $OUTPUT_DIR"
echo "============================================="
echo ""
ls -lh "$OUTPUT_DIR"/*.tif 2>/dev/null
