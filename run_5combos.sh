#!/bin/bash
# 5 combos: pass1_s75_inj74_g5, pass3/4_s75_inj74_g5, s90_inj87_pass4, s40_inj39_pass4
# No gamma (gamma=1.0 or auto-detect will skip)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/data/kevin_workspace/.cache/huggingface
PYTHON=/data/kevin_workspace/envs/rfsolver/bin/python3

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
PIPELINE_DIR="${SCRIPT_DIR}/output/grid_search_pipeline"
MULTI_DIR="${SCRIPT_DIR}/output/grid_search_multipass"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_search_5combos"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${SCRIPT_DIR}/run_5combos.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
SRC_PROMPT_PASS="Satellite image with noise, blurring, and low resolution"
TAR_PROMPT_PASS="Clean high resolution satellite image with sharp details and reduced noise"

echo "============================================="
echo "  5 Combos Run"
echo "  GPUs: 0, 6, 7"
echo "  Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# GPU 7: #1 pass1_s75_inj74_g5 (independent, single pass from original)
(
    tag="pass1_s75_inj74_g5"
    echo "[$(date '+%H:%M:%S')] GPU 7: ${tag} started"
    CUDA_VISIBLE_DEVICES=7 $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 75 --inject 74 --guidance 5 \
        --tile_size 4096 --overlap 0 \
        --src_prompt "$SRC_PROMPT" --tar_prompt "$TAR_PROMPT" \
        > "${LOG_DIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU 7: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU 7: ${tag} FAILED"
) &
PID_GPU7=$!

# GPU 6: #2 pass3_s75_inj74_g5 -> #3 pass4_s75_inj74_g5 (sequential)
(
    # pass3: input = pass2_s75_inj74_g5 from multipass dir
    PASS2_FILE="${MULTI_DIR}/pass2_s75_inj74_g5.tif"
    tag="pass3_s75_inj74_g5"
    echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} started"
    CUDA_VISIBLE_DEVICES=6 $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$PASS2_FILE" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 75 --inject 74 --guidance 5 \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "${LOG_DIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} done" || \
    { echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} FAILED"; exit 1; }

    # pass4: input = pass3 result
    tag="pass4_s75_inj74_g5"
    echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} started"
    CUDA_VISIBLE_DEVICES=6 $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "${OUTPUT_DIR}/pass3_s75_inj74_g5.tif" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 75 --inject 74 --guidance 5 \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "${LOG_DIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU 6: ${tag} FAILED"
) &
PID_GPU6=$!

# GPU 0: #4 s90_inj87_pass4 -> #5 s40_inj39_pass4 (independent, sequential on same GPU)
(
    # s90_inj87_pass4: input = s90_inj87_pass3 from pipeline dir
    tag="s90_inj87_pass4"
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} started"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "${PIPELINE_DIR}/s90_inj87_pass3.tif" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 90 --inject 87 --guidance 4 \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "${LOG_DIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} FAILED"

    # s40_inj39_pass4: input = s40_inj39_pass3 from pipeline dir
    tag="s40_inj39_pass4"
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} started"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "${PIPELINE_DIR}/s40_inj39_pass3.tif" \
        --output "${OUTPUT_DIR}/${tag}.tif" \
        --num_steps 40 --inject 39 --guidance 4 \
        --tile_size 4096 --overlap 0 --no_stretch \
        --src_prompt "$SRC_PROMPT_PASS" --tar_prompt "$TAR_PROMPT_PASS" \
        > "${LOG_DIR}/${tag}.log" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} done" || \
    echo "[$(date '+%H:%M:%S')] GPU 0: ${tag} FAILED"
) &
PID_GPU0=$!

wait $PID_GPU7 $PID_GPU6 $PID_GPU0

echo ""
echo "============================================="
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
ls -lht "$OUTPUT_DIR"/*.tif 2>/dev/null
