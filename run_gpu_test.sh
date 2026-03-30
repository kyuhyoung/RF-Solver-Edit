#!/bin/bash
# Test: 2 RF-Solver processes on same GPU 7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
OUTDIR="${SCRIPT_DIR}/output/gpu_test"
RUN_LOG="${SCRIPT_DIR}/run_gpu_test.log"
mkdir -p "$OUTDIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

GPU=7

echo "============================================="
echo "  GPU Test: 2 processes on GPU $GPU"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Process A: s30_inj29_g5
(
    echo "[$(date '+%H:%M:%S')] GPU $GPU [A]: s30_inj29_g5 started"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "${OUTDIR}/test_A_s30_inj29_g5.tif" \
        --num_steps 30 --inject 29 --guidance 5 \
        --tile_size 4096 --overlap 0 \
        --src_prompt "$SRC" --tar_prompt "$TAR" \
        > "${OUTDIR}/test_A.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $GPU [A]: done" || \
    echo "[$(date '+%H:%M:%S')] GPU $GPU [A]: FAILED"
) &
PID_A=$!

# Process B: s30_inj29_g6
(
    echo "[$(date '+%H:%M:%S')] GPU $GPU [B]: s30_inj29_g6 started"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" --output "${OUTDIR}/test_B_s30_inj29_g6.tif" \
        --num_steps 30 --inject 29 --guidance 6 \
        --tile_size 4096 --overlap 0 \
        --src_prompt "$SRC" --tar_prompt "$TAR" \
        > "${OUTDIR}/test_B.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU $GPU [B]: done" || \
    echo "[$(date '+%H:%M:%S')] GPU $GPU [B]: FAILED"
) &
PID_B=$!

wait $PID_A $PID_B

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""
echo "GPU memory peak (check test_A.log / test_B.log for OOM):"
grep -l "OutOfMemory\|FAILED" "${OUTDIR}/test_A.log" "${OUTDIR}/test_B.log" 2>/dev/null || echo "No OOM detected"
echo ""
ls -lh "${OUTDIR}"/test_*.tif 2>/dev/null || echo "No output files (both failed?)"
