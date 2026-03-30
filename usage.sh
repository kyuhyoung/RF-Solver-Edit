#!/bin/bash

# RF-Solver-Edit usage script for TOP (TIFF/GeoTIFF) satellite image restoration
#
# Usage:
#   ./usage.sh                                     # default input
#   ./usage.sh <input_top_file>                    # output: input_rfsolver.tif
#   ./usage.sh <input_top_file> [output_top_file]  # custom output
#
# Options:
#   --num_steps    ODE steps for inversion & denoising  (default: 25)
#   --inject       Feature sharing steps                (default: 20, higher=more structure preservation)
#   --guidance     Guidance scale for denoising          (default: 2.0)
#   --tile_size    Tile size for processing              (default: 1024)
#   --overlap      Overlap between tiles                 (default: 128)
#   --gamma        Gamma correction                      (default: 0.7)
#   --offload      CPU offload for low-memory GPUs

set -e

# Pick first free GPU in order 7,6,5,4,3,2,1,0
pick_free_gpu() {
    for g in 7 6 5 4 3 2 1 0; do
        local mem
        mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ "$mem" -le 100 ]; then
            echo "$g"
            return
        fi
    done
    echo "7"  # fallback
}
export CUDA_VISIBLE_DEVICES=$(pick_free_gpu)
export HF_HOME=/data/kevin_workspace/.cache/huggingface

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Log file
LOGFILE="${SCRIPT_DIR}/usage.log"
exec > >(tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# Default input
DEFAULT_INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

INPUT_FILE="${1:-$DEFAULT_INPUT}"
if [ $# -gt 0 ]; then shift; fi

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

INPUT_BASENAME=$(basename "$INPUT_FILE")
INPUT_STEM="${INPUT_BASENAME%.*}"

# Determine output file
OUTPUT_FILE=""
EXTRA_ARGS=()
if [ $# -gt 0 ] && [[ ! "$1" == --* ]]; then
    OUTPUT_FILE="$1"
    shift
fi
EXTRA_ARGS=("$@")

OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

# Default parameters
DEFAULT_NUM_STEPS="25"
DEFAULT_INJECT="24"
DEFAULT_GUIDANCE="1.0"
DEFAULT_TILE_SIZE="4096"
DEFAULT_OVERLAP="0"
DEFAULT_GAMMA="0.7"
DEFAULT_SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
DEFAULT_TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

# Check if user provided these args
HAS_NUM_STEPS=false
HAS_INJECT=false
HAS_GUIDANCE=false
HAS_TILE_SIZE=false
HAS_OVERLAP=false
HAS_GAMMA=false
HAS_SRC_PROMPT=false
HAS_TAR_PROMPT=false
for arg in "${EXTRA_ARGS[@]}"; do
    case $arg in
        --num_steps) HAS_NUM_STEPS=true ;;
        --inject) HAS_INJECT=true ;;
        --guidance) HAS_GUIDANCE=true ;;
        --tile_size) HAS_TILE_SIZE=true ;;
        --overlap) HAS_OVERLAP=true ;;
        --gamma) HAS_GAMMA=true ;;
        --src_prompt) HAS_SRC_PROMPT=true ;;
        --tar_prompt) HAS_TAR_PROMPT=true ;;
    esac
done

CMD_ARGS=()
if [ "$HAS_NUM_STEPS" = false ]; then CMD_ARGS+=(--num_steps "$DEFAULT_NUM_STEPS"); fi
if [ "$HAS_INJECT" = false ]; then CMD_ARGS+=(--inject "$DEFAULT_INJECT"); fi
if [ "$HAS_GUIDANCE" = false ]; then CMD_ARGS+=(--guidance "$DEFAULT_GUIDANCE"); fi
if [ "$HAS_TILE_SIZE" = false ]; then CMD_ARGS+=(--tile_size "$DEFAULT_TILE_SIZE"); fi
if [ "$HAS_OVERLAP" = false ]; then CMD_ARGS+=(--overlap "$DEFAULT_OVERLAP"); fi
if [ "$HAS_GAMMA" = false ]; then CMD_ARGS+=(--gamma "$DEFAULT_GAMMA"); fi
if [ "$HAS_SRC_PROMPT" = false ]; then CMD_ARGS+=(--src_prompt "$DEFAULT_SRC_PROMPT"); fi
if [ "$HAS_TAR_PROMPT" = false ]; then CMD_ARGS+=(--tar_prompt "$DEFAULT_TAR_PROMPT"); fi

# Determine actual parameter values for filename postfix
ACTUAL_INJECT="$DEFAULT_INJECT"
ACTUAL_GUIDANCE="$DEFAULT_GUIDANCE"
ACTUAL_STEPS="$DEFAULT_NUM_STEPS"
for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        --inject) ACTUAL_INJECT="${EXTRA_ARGS[$((i+1))]}" ;;
        --guidance) ACTUAL_GUIDANCE="${EXTRA_ARGS[$((i+1))]}" ;;
        --num_steps) ACTUAL_STEPS="${EXTRA_ARGS[$((i+1))]}" ;;
    esac
done

PARAM_POSTFIX="inj${ACTUAL_INJECT}_g${ACTUAL_GUIDANCE}_s${ACTUAL_STEPS}"

if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_STEM}_rfsolver_${PARAM_POSTFIX}.tif"
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  RF-Solver-Edit Satellite Image Restoration${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${YELLOW}Input:    ${INPUT_FILE}${NC}"
echo -e "${YELLOW}Output:   ${OUTPUT_FILE}${NC}"
echo -e "${YELLOW}Inject:   ${ACTUAL_INJECT} (higher=more structure preservation)${NC}"
echo -e "${YELLOW}Guidance: ${ACTUAL_GUIDANCE}${NC}"
echo -e "${YELLOW}Steps:    ${ACTUAL_STEPS}${NC}"
echo ""

# FLUX model auto-downloads from HuggingFace on first run
# (cached in ~/.cache/huggingface, mounted from host)

python3 -u "${SCRIPT_DIR}/refine_geotiff.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    "${CMD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Done!${NC}"
echo -e "${GREEN}  Output: ${OUTPUT_FILE}${NC}"
echo -e "${GREEN}==================================================${NC}"
