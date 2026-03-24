#!/bin/bash
# Fine grid search: inject × guidance with gamma=0.8 fixed
# Then steps sweep with best params

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="${SCRIPT_DIR}/input/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_fine"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_fine.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

GPU_LIST=(7 6 5)
NUM_GPUS=${#GPU_LIST[@]}

SLOT_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    SLOT_PIDS+=(0)
done

get_free_slot() {
    while true; do
        for ((i=0; i<NUM_GPUS; i++)); do
            pid=${SLOT_PIDS[$i]}
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
# Phase 1: inject × guidance (gamma=0.8 fixed)
# =============================================
INJECT_LIST=(71 72 73 74)
GUIDANCE_LIST=(3 3.5 4 4.5 5 5.5 6 6.5 7)
GAMMA="0.8"

JOBS=()
for inject in "${INJECT_LIST[@]}"; do
    for guidance in "${GUIDANCE_LIST[@]}"; do
        JOBS+=("75:${inject}:${guidance}:${GAMMA}")
    done
done

# Phase 2: steps sweep (g5, gamma=0.8)
JOBS+=("50:49:5:${GAMMA}")
JOBS+=("100:97:5:${GAMMA}")

echo "============================================="
echo "  Fine Grid Search: ${#JOBS[@]} combinations"
echo "  Phase 1: inject(${#INJECT_LIST[@]}) × guidance(${#GUIDANCE_LIST[@]}) = $((${#INJECT_LIST[@]} * ${#GUIDANCE_LIST[@]}))"
echo "  Phase 2: steps sweep = 2"
echo "  Gamma: ${GAMMA} (fixed)"
echo "  Parallel GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

SKIPPED=0
for job in "${JOBS[@]}"; do
    IFS=: read -r steps inject guidance gamma <<< "$job"
    tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    outfile="${OUTPUT_DIR}/${tag}.tif"

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
