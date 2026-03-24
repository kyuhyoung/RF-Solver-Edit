#!/bin/bash
# Full pipeline grid search: 1pass × 2pass × 3pass
# gamma=0.6, g4 fixed
# 1pass: vary steps/inject
# 2pass/3pass: fixed s75_inj74_g4 --no_stretch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="${SCRIPT_DIR}/input/fused_top_naive.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_pipeline.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PROMPT_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT_PASS="Clean high resolution satellite image with sharp details and reduced noise"

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

run_pass1() {
    local gpu=$1 steps=$2 inject=$3
    local tag="s${steps}_inj${inject}"
    local outfile="${OUTPUT_DIR}/${tag}_pass1.tif"
    local logfile="${LOG_DIR}/${tag}_pass1.log"

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 started"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "$outfile" \
        --num_steps "$steps" \
        --inject "$inject" \
        --guidance 4 \
        --tile_size 4096 \
        --overlap 0 \
        --gamma 0.6 \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag}_pass1 FAILED"
}

run_passN() {
    local gpu=$1 input=$2 output=$3 tag=$4
    local logfile="${LOG_DIR}/${tag}.log"

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" \
        --output "$output" \
        --num_steps 75 \
        --inject 74 \
        --guidance 4 \
        --tile_size 4096 \
        --overlap 0 \
        --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" \
        --tar_prompt "$TAR_PROMPT_PASS" \
        > "$logfile" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

wait_all() {
    for ((i=0; i<NUM_GPUS; i++)); do
        pid=${SLOT_PIDS[$i]}
        if [ "$pid" != "0" ]; then
            wait "$pid" 2>/dev/null
        fi
        SLOT_PIDS[$i]=0
    done
}

# 1pass grid: steps × inject
STEPS_LIST=(80 85 90 95 100)
OFFSETS=(2 3 4)

COMBOS=()
for s in "${STEPS_LIST[@]}"; do
    for off in "${OFFSETS[@]}"; do
        inj=$((s - off))
        COMBOS+=("${s}:${inj}")
    done
done

echo "============================================="
echo "  Pipeline Grid Search"
echo "  ${#COMBOS[@]} combos × 3 passes = $((${#COMBOS[@]} * 3)) total runs"
echo "  1pass: gamma=0.6, g4, vary steps/inject"
echo "  2pass/3pass: s75_inj74_g4 --no_stretch"
echo "  Parallel GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# ========== Phase 1: 1pass ==========
echo ""
echo "===== Phase 1: 1pass (${#COMBOS[@]} combos) ====="

for combo in "${COMBOS[@]}"; do
    IFS=: read -r steps inject <<< "$combo"
    tag="s${steps}_inj${inject}"
    outfile="${OUTPUT_DIR}/${tag}_pass1.tif"

    if [ -f "$outfile" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag}_pass1 (already exists)"
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    run_pass1 "$gpu" "$steps" "$inject" &
    SLOT_PIDS[$slot]=$!
done

wait_all
echo ""
echo "===== Phase 1 complete at $(date '+%H:%M:%S') ====="

# ========== Phase 2: 2pass ==========
echo ""
echo "===== Phase 2: 2pass (${#COMBOS[@]} combos) ====="

for combo in "${COMBOS[@]}"; do
    IFS=: read -r steps inject <<< "$combo"
    tag="s${steps}_inj${inject}"
    pass1_file="${OUTPUT_DIR}/${tag}_pass1.tif"
    pass2_file="${OUTPUT_DIR}/${tag}_pass2.tif"

    if [ -f "$pass2_file" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag}_pass2 (already exists)"
        continue
    fi

    if [ ! -f "$pass1_file" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag}_pass2 (pass1 missing)"
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    run_passN "$gpu" "$pass1_file" "$pass2_file" "${tag}_pass2" &
    SLOT_PIDS[$slot]=$!
done

wait_all
echo ""
echo "===== Phase 2 complete at $(date '+%H:%M:%S') ====="

# ========== Phase 3: 3pass ==========
echo ""
echo "===== Phase 3: 3pass (${#COMBOS[@]} combos) ====="

for combo in "${COMBOS[@]}"; do
    IFS=: read -r steps inject <<< "$combo"
    tag="s${steps}_inj${inject}"
    pass2_file="${OUTPUT_DIR}/${tag}_pass2.tif"
    pass3_file="${OUTPUT_DIR}/${tag}_pass3.tif"

    if [ -f "$pass3_file" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag}_pass3 (already exists)"
        continue
    fi

    if [ ! -f "$pass2_file" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag}_pass3 (pass2 missing)"
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    run_passN "$gpu" "$pass2_file" "$pass3_file" "${tag}_pass3" &
    SLOT_PIDS[$slot]=$!
done

wait_all
echo ""
echo "===== Phase 3 complete at $(date '+%H:%M:%S') ====="

echo ""
echo "============================================="
echo "  Pipeline grid search complete at $(date '+%Y-%m-%d %H:%M:%S')"
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
