#!/bin/bash
# 192 combos: steps(4) × gap(3) × guidance(4) × pass(4)
# 4 GPUs × 2 per GPU = 8 slots

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_192"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_192.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PASS="Clean high resolution satellite image with sharp details and reduced noise"

GPU_SLOTS=(7 7 6 6 5 5 0 0)
NUM_SLOTS=${#GPU_SLOTS[@]}

SLOT_PIDS=()
for ((i=0; i<NUM_SLOTS; i++)); do
    SLOT_PIDS+=(0)
done

get_free_slot() {
    while true; do
        for ((i=0; i<NUM_SLOTS; i++)); do
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

run_pipeline() {
    local gpu=$1 steps=$2 gap=$3 guidance=$4 max_pass=$5
    local inject=$((steps - gap))
    local tag="s${steps}_inj${inject}_g${guidance}"

    # Pass 1
    local p1="${OUTPUT_DIR}/${tag}_pass1.tif"
    local log1="${LOG_DIR}/${tag}_pass1.log"
    if [ ! -f "$p1" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 started"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$INPUT" --output "$p1" \
            --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
            --tile_size 4096 --overlap 0 \
            --src_prompt "$SRC" --tar_prompt "$TAR" \
            > "$log1" 2>&1 || \
        { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 FAILED"; return 1; }
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 done"
    fi

    # Pass 2+
    local prev="$p1"
    for ((p=2; p<=max_pass; p++)); do
        local pN="${OUTPUT_DIR}/${tag}_pass${p}.tif"
        local logN="${LOG_DIR}/${tag}_pass${p}.log"
        if [ ! -f "$pN" ]; then
            echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass${p} started"
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
                --input "$prev" --output "$pN" \
                --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
                --tile_size 4096 --overlap 0 --no_stretch \
                --src_prompt "$SRC_PASS" --tar_prompt "$TAR_PASS" \
                > "$logN" 2>&1 || \
            { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass${p} FAILED"; return 1; }
            echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass${p} done"
        fi
        prev="$pN"
    done
}

STEPS_LIST=(40 60 80 100)
GAP_LIST=(1 3 5)
GUIDANCE_LIST=(2 4 6 8)
MAX_PASS=4

# Build pipeline list
PIPELINES=()
for s in "${STEPS_LIST[@]}"; do
    for gap in "${GAP_LIST[@]}"; do
        for g in "${GUIDANCE_LIST[@]}"; do
            PIPELINES+=("${s}:${gap}:${g}")
        done
    done
done

echo "============================================="
echo "  192 Combo Grid Search"
echo "  ${#PIPELINES[@]} pipelines × ${MAX_PASS} passes = $((${#PIPELINES[@]} * MAX_PASS)) total"
echo "  Steps: ${STEPS_LIST[*]}"
echo "  Gap: ${GAP_LIST[*]}"
echo "  Guidance: ${GUIDANCE_LIST[*]}"
echo "  Pass: 1-${MAX_PASS}"
echo "  GPU slots: ${GPU_SLOTS[*]} (${NUM_SLOTS} slots)"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

for pipeline in "${PIPELINES[@]}"; do
    IFS=: read -r steps gap guidance <<< "$pipeline"
    inject=$((steps - gap))
    tag="s${steps}_inj${inject}_g${guidance}"

    # Skip if final pass already exists
    final="${OUTPUT_DIR}/${tag}_pass${MAX_PASS}.tif"
    if [ -f "$final" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP pipeline ${tag} (pass${MAX_PASS} exists)"
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_SLOTS[$slot]}

    run_pipeline "$gpu" "$steps" "$gap" "$guidance" "$MAX_PASS" &
    SLOT_PIDS[$slot]=$!
done

# Wait for all
echo ""
echo "All pipelines launched. Waiting for completion..."
for ((i=0; i<NUM_SLOTS; i++)); do
    pid=${SLOT_PIDS[$i]}
    if [ "$pid" != "0" ]; then
        wait "$pid" 2>/dev/null
    fi
done

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""
echo "Total files:"
ls "$OUTPUT_DIR"/*.tif 2>/dev/null | wc -l
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
