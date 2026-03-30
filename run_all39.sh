#!/bin/bash
# All remaining 39 combos in one script
# B(6) + D(7) + E(12) + F(5) + G(3) + H(3) + I(3) = 39

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
PIPELINE_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_5combos"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_all39.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PASS="Clean high resolution satellite image with sharp details and reduced noise"

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

wait_all() {
    for ((i=0; i<NUM_GPUS; i++)); do
        pid=${SLOT_PIDS[$i]}
        if [ "$pid" != "0" ]; then
            wait "$pid" 2>/dev/null
        fi
        SLOT_PIDS[$i]=0
    done
}

# pass1 with gamma (force_gamma)
run_pass1_gamma() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4 gamma=$5
    local tag="s${steps}_inj${inject}_g${guidance}_gamma${gamma}"
    local out="${OUTPUT_DIR}/${tag}.tif"
    local log="${LOG_DIR}/${tag}.log"
    if [ -f "$out" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "$out" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --gamma "$gamma" --force_gamma \
        --src_prompt "$SRC" --tar_prompt "$TAR" \
        > "$log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

# pass1 without gamma
run_pass1() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4
    local tag="s${steps}_inj${inject}_g${guidance}"
    local out="${OUTPUT_DIR}/${tag}.tif"
    local log="${LOG_DIR}/${tag}.log"
    if [ -f "$out" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "$out" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 \
        --src_prompt "$SRC" --tar_prompt "$TAR" \
        > "$log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

# passN (no_stretch)
run_passN() {
    local gpu=$1 input=$2 steps=$3 inject=$4 guidance=$5 pass=$6 tag_prefix=$7
    local tag="${tag_prefix}_pass${pass}"
    local out="${OUTPUT_DIR}/${tag}.tif"
    local log="${LOG_DIR}/${tag}.log"
    if [ -f "$out" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    if [ ! -f "$input" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag} (input missing)"; return 1; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" --output "$out" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PASS" --tar_prompt "$TAR_PASS" \
        > "$log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"; return 1; }
}

# pass4 from pipeline pass3
run_pass4_pipeline() {
    local gpu=$1 steps=$2 inject=$3
    local input="${PIPELINE_DIR}/s${steps}_inj${inject}_pass3.tif"
    local tag="s${steps}_inj${inject}"
    run_passN "$gpu" "$input" "$steps" "$inject" 4 4 "$tag"
}

# pass2 from pipeline pass1
run_pass2_from_pipeline() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4
    local input="${PIPELINE_DIR}/s${steps}_inj${inject}_pass1.tif"
    local tag="pass2_s${steps}_inj${inject}_g${guidance}"
    local out="${OUTPUT_DIR}/${tag}.tif"
    local log="${LOG_DIR}/${tag}.log"
    if [ -f "$out" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag}"; return 0; fi
    if [ ! -f "$input" ]; then echo "[$(date '+%H:%M:%S')] SKIP ${tag} (input missing)"; return 1; fi
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$input" --output "$out" \
        --num_steps "$steps" --inject "$inject" --guidance "$guidance" \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PASS" --tar_prompt "$TAR_PASS" \
        > "$log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

# full 4-pass chain (pass1 without gamma)
run_full_4pass() {
    local gpu=$1 steps=$2 inject=$3
    local tag="s${steps}_inj${inject}"
    # pass1
    run_pass1 "$gpu" "$steps" "$inject" 4
    # pass2
    run_passN "$gpu" "${OUTPUT_DIR}/${tag}_g4.tif" "$steps" "$inject" 4 2 "${tag}"
    # pass3
    run_passN "$gpu" "${OUTPUT_DIR}/${tag}_pass2.tif" "$steps" "$inject" 4 3 "${tag}"
    # pass4
    run_passN "$gpu" "${OUTPUT_DIR}/${tag}_pass3.tif" "$steps" "$inject" 4 4 "${tag}"
}

echo "============================================="
echo "  All 39 remaining combos"
echo "  GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# ===== B: gamma infill inj=72 (6) =====
echo ""
echo "===== B: s75_inj72 gamma infill (6) ====="
for g in 5.5 6 6.5; do
    for gamma in 0.7 0.9; do
        slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
        run_pass1_gamma "$gpu" 75 72 "$g" "$gamma" &
        SLOT_PIDS[$slot]=$!
    done
done
wait_all

# ===== D: pass4 from pipeline (7) =====
echo ""
echo "===== D: pass4 (7) ====="
for combo in "50:49" "60:59" "70:69" "80:79" "85:82" "95:92" "100:96"; do
    IFS=: read -r steps inject <<< "$combo"
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass4_pipeline "$gpu" "$steps" "$inject" &
    SLOT_PIDS[$slot]=$!
done
wait_all

# ===== F: s70_inj67 gamma0.8 (5) =====
echo ""
echo "===== F: s70_inj67 gamma0.8 (5) ====="
for g in 4.5 5 5.5 6 6.5; do
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass1_gamma "$gpu" 70 67 "$g" 0.8 &
    SLOT_PIDS[$slot]=$!
done
wait_all

# ===== G: s80_inj77 gamma0.8 (3) =====
echo ""
echo "===== G: s80_inj77 gamma0.8 (3) ====="
for g in 4.5 5.5 6.5; do
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass1_gamma "$gpu" 80 77 "$g" 0.8 &
    SLOT_PIDS[$slot]=$!
done
wait_all

# ===== H: pass2_s70_inj69 (3) =====
echo ""
echo "===== H: pass2_s70_inj69 (3) ====="
for g in 4.5 5 5.5; do
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass2_from_pipeline "$gpu" 70 69 "$g" &
    SLOT_PIDS[$slot]=$!
done
wait_all

# ===== I: pass2_s80_inj79 (3) =====
echo ""
echo "===== I: pass2_s80_inj79 (3) ====="
for g in 4.5 5 5.5; do
    slot=$(get_free_slot); gpu=${GPU_LIST[$slot]}
    run_pass2_from_pipeline "$gpu" 80 79 "$g" &
    SLOT_PIDS[$slot]=$!
done
wait_all

# ===== E: s125/150/175 full 4-pass chains (12) =====
echo ""
echo "===== E: s125/150/175 4-pass chains (12) ====="
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
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null | head -40
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
