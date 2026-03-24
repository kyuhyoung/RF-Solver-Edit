#!/bin/bash
# 2-pass grid search
# Pass 1 (fixed): s100_inj97_g4_gamma0.8 (already generated)
# Pass 2 (search): vary steps, inject, guidance with --no_stretch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

PASS1_FILE="${SCRIPT_DIR}/output/grid_search_steps/s100_inj97_g4_gamma0.8.tif"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_2pass"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/grid_search_2pass.log"
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

launch_job() {
    local gpu=$1 steps=$2 inject=$3 guidance=$4
    local tag="pass2_s${steps}_inj${inject}_g${guidance}"
    local outfile="${OUTPUT_DIR}/${tag}.tif"
    local logfile="${LOG_DIR}/${tag}.log"

    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$PASS1_FILE" \
        --output "$outfile" \
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

# Pass 2 parameter grid
# guidance fixed at 4, explore inject widely
# s30: inject 20~29 (67%~97%)
# s50: inject 35~49 (70%~98%)
# s75: inject 55~74 (73%~99%)

G=4
JOBS=()
for inj in 20 22 24 25 26 27 28 29; do
    JOBS+=("30:${inj}:${G}")
done
for inj in 35 38 41 43 45 47 48 49; do
    JOBS+=("50:${inj}:${G}")
done
for inj in 55 60 65 68 70 72 73 74; do
    JOBS+=("75:${inj}:${G}")
done

echo "============================================="
echo "  2-Pass Grid Search: ${#JOBS[@]} combinations"
echo "  Pass 1 (fixed): s100_inj97_g4_gamma0.8"
echo "  Pass 2: vary steps/inject/guidance"
echo "  Parallel GPUs: ${GPU_LIST[*]}"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

SKIPPED=0
for job in "${JOBS[@]}"; do
    IFS=: read -r steps inject guidance <<< "$job"
    tag="pass2_s${steps}_inj${inject}_g${guidance}"
    outfile="${OUTPUT_DIR}/${tag}.tif"

    if [ -f "$outfile" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag} (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    slot=$(get_free_slot)
    gpu=${GPU_LIST[$slot]}

    launch_job "$gpu" "$steps" "$inject" "$guidance" &
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
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null
echo ""
echo "Failed jobs:"
for f in "$LOG_DIR"/*.log; do
    if ! grep -q "Done!" "$f" 2>/dev/null; then
        echo "  $(basename "$f")"
    fi
done
