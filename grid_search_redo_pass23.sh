#!/bin/bash
# Redo pass2/pass3 with SAME params as pass1 (not s75_inj74)
# Reuses existing pass1 files

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_redo"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_redo_pass23.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT="Clean high resolution satellite image with sharp details and reduced noise"

GPU_LIST=(7 6 5 4)
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

run_pass23() {
    local gpu=$1 steps=$2 inject=$3
    local tag="s${steps}_inj${inject}"
    local p1="${INPUT_DIR}/${tag}_pass1.tif"
    local p2="${OUTPUT_DIR}/${tag}_pass2.tif"
    local p3="${OUTPUT_DIR}/${tag}_pass3.tif"

    if [ ! -f "$p1" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag} (pass1 missing)"
        return 1
    fi

    # Pass 2: same steps/inject as pass1
    if [ ! -f "$p2" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$p1" --output "$p2" \
            --num_steps "$steps" --inject "$inject" --guidance 4 \
            --tile_size 4096 --overlap 0 --no_stretch \
            --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
            > "${LOG_DIR}/${tag}_pass2.log" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 done" || \
        { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass2 FAILED"; return 1; }
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag}_pass2 (exists)"
    fi

    # Pass 3: same steps/inject as pass1
    if [ ! -f "$p3" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$p2" --output "$p3" \
            --num_steps "$steps" --inject "$inject" --guidance 4 \
            --tile_size 4096 --overlap 0 --no_stretch \
            --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
            > "${LOG_DIR}/${tag}_pass3.log" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 done" || \
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass3 FAILED"
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${tag}_pass3 (exists)"
    fi
}

# Priority order: most promising first
COMBOS=(
    "100:98"
    "100:99"
    "80:79"
    "90:89"
    "100:97"
    "70:69"
    "60:59"
    "50:49"
)

echo "============================================="
echo "  Redo pass2/pass3 with SAME params as pass1"
echo "  ${#COMBOS[@]} combos × pass2+pass3"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

for combo in "${COMBOS[@]}"; do
    IFS=: read -r steps inject <<< "$combo"

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    run_pass23 "$gpu" "$steps" "$inject" &
    SLOT_PIDS[$slot]=$!
done

# Wait for all
echo ""
echo "All jobs launched. Waiting for completion..."
for ((i=0; i<NUM_GPUS; i++)); do
    pid=${SLOT_PIDS[$i]}
    if [ "$pid" != "0" ]; then
        wait "$pid" 2>/dev/null
    fi
done

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
