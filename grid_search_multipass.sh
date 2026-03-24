#!/bin/bash
# Multi-pass exploration: 3pass, 4pass, and pass2 guidance variations

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

PASS1_FILE="${SCRIPT_DIR}/output/grid_search_steps/s100_inj97_g4_gamma0.8.tif"
PASS2_FILE="${SCRIPT_DIR}/output/grid_search_2pass/pass2_s75_inj74_g4.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_multipass"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_multipass.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT="Clean high resolution satellite image with sharp details and reduced noise"

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

run_job() {
    local gpu=$1 input=$2 output=$3 steps=$4 inject=$5 guidance=$6 tag=$7
    local logfile="${LOG_DIR}/${tag}.log"

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" \
        --output "$output" \
        --num_steps "$steps" \
        --inject "$inject" \
        --guidance "$guidance" \
        --tile_size 4096 \
        --overlap 0 \
        --no_stretch \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

echo "============================================="
echo "  Multi-pass Exploration"
echo "  Parallel GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# --- Batch 1: 3pass + pass2 guidance variations (~35min) ---
# 3pass: pass2 result -> s75_inj74_g4
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "$PASS2_FILE" "${OUTPUT_DIR}/pass3_s75_inj74_g4.tif" 75 74 4 "pass3_s75_inj74_g4" &
SLOT_PIDS[$slot]=$!

# pass2 with g3
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "$PASS1_FILE" "${OUTPUT_DIR}/pass2_s75_inj74_g3.tif" 75 74 3 "pass2_s75_inj74_g3" &
SLOT_PIDS[$slot]=$!

# pass2 with g5
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "$PASS1_FILE" "${OUTPUT_DIR}/pass2_s75_inj74_g5.tif" 75 74 5 "pass2_s75_inj74_g5" &
SLOT_PIDS[$slot]=$!

# --- Batch 2: more pass2 variations + 4pass (~35min) ---
# pass2 with g2
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "$PASS1_FILE" "${OUTPUT_DIR}/pass2_s75_inj74_g2.tif" 75 74 2 "pass2_s75_inj74_g2" &
SLOT_PIDS[$slot]=$!

# pass2 with g6
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "$PASS1_FILE" "${OUTPUT_DIR}/pass2_s75_inj74_g6.tif" 75 74 6 "pass2_s75_inj74_g6" &
SLOT_PIDS[$slot]=$!

# 4pass: pass3 result -> s75_inj74_g4 (pass3 must be done by now from batch 1)
slot=$(get_free_slot)
gpu=${GPU_LIST[$slot]}
run_job "$gpu" "${OUTPUT_DIR}/pass3_s75_inj74_g4.tif" "${OUTPUT_DIR}/pass4_s75_inj74_g4.tif" 75 74 4 "pass4_s75_inj74_g4" &
SLOT_PIDS[$slot]=$!

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
echo "  Results in: $OUTPUT_DIR"
echo "============================================="
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
