#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTDIR="${SCRIPT_DIR}/output/grid_search_5combos"
LOGDIR="${OUTDIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_6gamma.log"
mkdir -p "$OUTDIR" "$LOGDIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

echo "============================================="
echo "  6 Gamma Combos (2+2+2)"
echo "  GPUs: 0, 6, 7"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

run_one() {
    local gpu=$1 g=$2 gamma=$3
    local tag="s75_inj73_g${g}_gamma${gamma}"
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} started"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "${OUTDIR}/${tag}.tif" \
        --num_steps 75 --inject 73 --guidance "$g" \
        --tile_size 4096 --overlap 0 --gamma "$gamma" --force_gamma \
        --src_prompt "$SRC" --tar_prompt "$TAR" \
        > "${LOGDIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ${tag} FAILED"
}

# GPU 7: g5.5, g6 gamma0.7
(run_one 7 5.5 0.7 && run_one 7 6 0.7) &

# GPU 6: g6.5 gamma0.7, g5.5 gamma0.9
(run_one 6 6.5 0.7 && run_one 6 5.5 0.9) &

# GPU 0: g6 gamma0.9, g6.5 gamma0.9
(run_one 0 6 0.9 && run_one 0 6.5 0.9) &

wait

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lht "$OUTDIR"/s75_inj73*.tif 2>/dev/null
