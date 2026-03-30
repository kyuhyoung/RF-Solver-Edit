#!/bin/bash
# Full grid search: guidance × gamma with s75_inj73
# Then steps sweep with best params
# Runs 3 jobs in parallel (GPU 7, 6, 5)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_full"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_full.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

GPU_LIST=(7 6 5)
NUM_GPUS=${#GPU_LIST[@]}

# Slot-based GPU management: each slot holds a PID
SLOT_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    SLOT_PIDS+=(0)
done

get_free_slot() {
    # Return first slot whose process has finished (or never started)
    while true; do
        for ((i=0; i<NUM_GPUS; i++)); do
            local pid=${SLOT_PIDS[$i]}
            if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                if [ "$pid" != "0" ]; then
                    wait "$pid" 2>/dev/null
                fi
                echo "$i"
                return
            fi
        done
        sleep 5
    done
}

launch_job() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4 gamma=$5
    local tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "$outfile" \
        --num_steps "$steps" \
        --inject "$inject" \
        --guidance "$guidance" \
        --tile_size 4096 \
        --overlap 0 \
        --gamma "$gamma" \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

# =============================================
# Phase 1: guidance × gamma (s75_inj73 fixed)
# =============================================
GUIDANCE_LIST=(5 7 8 9 10 12 15 20 30 40 70 100)
GAMMA_LIST=(0.5 0.6 0.7 0.8 0.9 1.0)

JOBS=()
for guidance in "${GUIDANCE_LIST[@]}"; do
    for gamma in "${GAMMA_LIST[@]}"; do
        JOBS+=("75:73:${guidance}:${gamma}")
    done
done

# Phase 2: steps sweep (will use best params, but pre-generate with g4.0 gamma0.7 as baseline)
STEPS_SWEEP=(
    "50:49:4.0:0.7"
    "100:97:4.0:0.7"
)
for job in "${STEPS_SWEEP[@]}"; do
    JOBS+=("$job")
done

echo "============================================="
echo "  Full Grid Search: ${#JOBS[@]} combinations"
echo "  Phase 1: guidance(${#GUIDANCE_LIST[@]}) × gamma(${#GAMMA_LIST[@]}) = $((${#GUIDANCE_LIST[@]} * ${#GAMMA_LIST[@]}))"
echo "  Phase 2: steps sweep = ${#STEPS_SWEEP[@]}"
echo "  Parallel GPUs: ${GPU_ORDER[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

SKIPPED=0
for job in "${JOBS[@]}"; do
    IFS=: read -r steps inject guidance gamma <<< "$job"
    tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    outfile="${OUTPUT_DIR}/${tag}.tif"

    # Skip if output already exists
    if [ -f "$outfile" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag} (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    launch_job "$gpu" "$steps" "$inject" "$guidance" "$gamma" &
    SLOT_PIDS[$slot]=$!
done

# Wait for all remaining
echo ""
echo "All jobs launched ($SKIPPED skipped). Waiting for completion..."
for ((i=0; i<NUM_GPUS; i++)); do
    pid=${SLOT_PIDS[$i]}
    if [ "$pid" != "0" ]; then
        wait "$pid" 2>/dev/null
    fi
done

echo ""
echo "============================================="
echo "  Grid search complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: $OUTPUT_DIR"
echo "============================================="
echo ""
echo "Results:"
ls -lhS "$OUTPUT_DIR"/*.tif 2>/dev/null
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
