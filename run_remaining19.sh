#!/bin/bash
# Remaining 19 combos: C (pass4 × 7) + D (s125/150/175 × 4pass)
# GPU 0, 6, 7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
PIPELINE_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_5combos"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_remaining19.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PROMPT_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT_PASS="Clean high resolution satellite image with sharp details and reduced noise"

GPU_LIST=(7 6 0)
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

run_pass1() {
    local gpu=$1 steps=$2 inject=$3
    local tag="s${steps}_inj${inject}_pass1"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"
    if [ -f "$outfile" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "$outfile" \
        --num_steps "$steps" --inject "$inject" --guidance 4 \
        --tile_size 4096 --overlap 0 \
        --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"; return 1; }
}

run_passN() {
    local gpu=$1 input=$2 steps=$3 inject=$4 pass=$5
    local tag="s${steps}_inj${inject}_pass${pass}"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"
    if [ -f "$outfile" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    if [ ! -f "$input" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag} (input missing: $input)"; return 1; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" --output "$outfile" \
        --num_steps "$steps" --inject "$inject" --guidance 4 \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"; return 1; }
}

run_pass4_from_pipeline() {
    local gpu=$1 steps=$2 inject=$3
    local input="${PIPELINE_DIR}/s${steps}_inj${inject}_pass3.tif"
    run_passN "$gpu" "$input" "$steps" "$inject" 4
}

run_full_4pass() {
    local gpu=$1 steps=$2 inject=$3
    # pass1
    run_pass1 "$gpu" "$steps" "$inject" || return 1
    # pass2
    run_passN "$gpu" "${OUTPUT_DIR}/s${steps}_inj${inject}_pass1.tif" "$steps" "$inject" 2 || return 1
    # pass3
    run_passN "$gpu" "${OUTPUT_DIR}/s${steps}_inj${inject}_pass2.tif" "$steps" "$inject" 3 || return 1
    # pass4
    run_passN "$gpu" "${OUTPUT_DIR}/s${steps}_inj${inject}_pass3.tif" "$steps" "$inject" 4 || return 1
}

echo "============================================="
echo "  Remaining 19 combos"
echo "  C: pass4 × 7, D: s125/150/175 × 4pass"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# ===== Group C: pass4 from existing pass3 (7 runs, independent) =====
echo ""
echo "===== Group C: pass4 runs ====="

PASS4_COMBOS=("50:49" "60:59" "70:69" "80:79" "85:82" "95:92" "100:96")

for combo in "${PASS4_COMBOS[@]}"; do
    IFS=: read -r steps inject <<< "$combo"
    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}
    run_pass4_from_pipeline "$gpu" "$steps" "$inject" &
    SLOT_PIDS[$slot]=$!
done

# Wait for all C jobs before starting D (to free GPUs)
for ((i=0; i<NUM_GPUS; i++)); do
    pid=${SLOT_PIDS[$i]}
    if [ "$pid" != "0" ]; then
        wait "$pid" 2>/dev/null
    fi
    SLOT_PIDS[$i]=0
done

echo ""
echo "===== Group C complete at $(date '+%H:%M:%S') ====="

# ===== Group D: s125/150/175 full 4-pass chains =====
echo ""
echo "===== Group D: s125/150/175 4-pass chains ====="

# 3 chains on 3 GPUs simultaneously
(run_full_4pass 7 125 121) &
PID7=$!
(run_full_4pass 6 150 146) &
PID6=$!
(run_full_4pass 0 175 170) &
PID0=$!

wait $PID7 $PID6 $PID0

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""
echo "Results:"
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null | head -30
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
