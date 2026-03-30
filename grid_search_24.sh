#!/bin/bash
# 24 combo grid search: pass extensions + infill + infill pass2
# gamma=0.8 forced, g varies per combo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_24"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_24.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PROMPT_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT_PASS="Clean high resolution satellite image with sharp details and reduced noise"

GPU_LIST=(6 5 4)
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
    local gpu=$1 steps=$2 inject=$3 guidance=$4 gamma=$5
    local tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"

    if [ -f "$outfile" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag} (exists)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "$outfile" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --gamma "$gamma" --force_gamma \
        --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"; return 1; }
}

run_passN() {
    local gpu=$1 input=$2 steps=$3 inject=$4 guidance=$5 gamma=$6 pass=$7
    local base_tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    local tag="${base_tag}_pass${pass}"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"

    if [ -f "$outfile" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag} (exists)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" --output "$outfile" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"; return 1; }
}

# Full pipeline: pass1 -> pass2 -> pass3 -> (pass4)
run_pipeline() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4 gamma=$5 max_pass=$6
    local base_tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"

    # Pass 1
    run_pass1 "$gpu" "$steps" "$inject" "$guidance" "$gamma"

    # Pass 2..N
    local prev="${OUTPUT_DIR}/${base_tag}.tif"
    for ((p=2; p<=max_pass; p++)); do
        local cur="${OUTPUT_DIR}/${base_tag}_pass${p}.tif"
        if [ ! -f "$prev" ]; then
            echo "[$(date '+%H:%M:%S')] GPU $gpu: SKIP ${base_tag}_pass${p} (prev missing)"
            return 1
        fi
        run_passN "$gpu" "$prev" "$steps" "$inject" "$guidance" "$gamma" "$p"
        prev="$cur"
    done
}

echo "============================================="
echo "  24 Combo Grid Search"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# =============================================
# Group 1: Pass extensions (sequential chains, run on dedicated GPUs)
# =============================================
echo ""
echo "===== Group 1: Pass extensions ====="

# s75_inj72_g6_gamma0.8 pass2->pass3->pass4
slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
run_pipeline "$gpu" 75 72 6 0.8 4 &
SLOT_PIDS[$slot]=$!

# s75_inj72_g5_gamma0.8 pass2->pass3
slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
run_pipeline "$gpu" 75 72 5 0.8 3 &
SLOT_PIDS[$slot]=$!

# s90_inj87_g5_gamma0.8 pass2
slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
run_pipeline "$gpu" 90 87 5 0.8 2 &
SLOT_PIDS[$slot]=$!

# =============================================
# Group 2: Infill single pass (parallel)
# =============================================
INFILL_PASS1=(
    "80:77:4:0.8"
    "80:77:5:0.8"
    "80:77:6:0.8"
    "85:82:4:0.8"
    "85:82:5:0.8"
    "85:82:6:0.8"
    "90:87:4:0.8"
    "90:87:6:0.8"
    "95:92:4:0.8"
    "95:92:5:0.8"
    "95:92:6:0.8"
    "100:97:3:0.8"
    "100:97:6:0.8"
)

for combo in "${INFILL_PASS1[@]}"; do
    IFS=: read -r steps inject guidance gamma <<< "$combo"
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass1 "$gpu" "$steps" "$inject" "$guidance" "$gamma" &
    SLOT_PIDS[$slot]=$!
done

# =============================================
# Group 3: Infill + pass2/pass3 (depends on group 2)
# =============================================
INFILL_MULTI=(
    "80:77:5:0.8:2"
    "85:82:5:0.8:2"
    "90:87:5:0.8:3"
    "95:92:5:0.8:2"
    "100:97:5:0.8:2"
)

for combo in "${INFILL_MULTI[@]}"; do
    IFS=: read -r steps inject guidance gamma max_pass <<< "$combo"
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pipeline "$gpu" "$steps" "$inject" "$guidance" "$gamma" "$max_pass" &
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
echo "  Results in: $OUTPUT_DIR"
echo "============================================="
echo ""
echo "Results:"
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
