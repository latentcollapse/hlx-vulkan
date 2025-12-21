#!/bin/bash
# build_shaders.sh - Compile all GLSL shaders to SPIR-V
# Requires: glslc from Vulkan SDK

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHADER_DIR="${SCRIPT_DIR}/shader"
OUT_DIR="${SHADER_DIR}/spv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}╔══════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  HLX Transformer Shader Compilation      ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════╝${NC}"
echo ""

# Check for glslc
if ! command -v glslc &> /dev/null; then
    echo -e "${RED}Error: glslc not found. Install Vulkan SDK.${NC}"
    echo "  Ubuntu: apt install vulkan-tools glslang-tools"
    echo "  Arch: pacman -S vulkan-tools glslang"
    exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# Compilation function
compile_shader() {
    local src="$1"
    local name=$(basename "$src" .glsl)
    local dst="${OUT_DIR}/${name}.spv"
    
    printf "  %-30s" "${name}.glsl"
    
    if glslc --target-env=vulkan1.2 \
             -fshader-stage=compute \
             -O \
             -Werror \
             "$src" \
             -o "$dst" 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

echo "Compiling shaders..."
echo ""

FAILED=0

# Core kernels
for shader in \
    gemm \
    gemm_backward \
    layernorm_forward \
    layernorm_backward \
    softmax_forward \
    softmax_backward \
    gelu_forward \
    gelu_backward \
    embedding_forward \
    embedding_backward \
    cross_entropy_forward \
    cross_entropy_backward \
    adam_update \
    elementwise \
    reduce_sum \
    reduce_final
do
    if [ -f "${SHADER_DIR}/${shader}.glsl" ]; then
        compile_shader "${SHADER_DIR}/${shader}.glsl" || ((FAILED++))
    else
        printf "  %-30s" "${shader}.glsl"
        echo -e "${YELLOW}SKIP (not found)${NC}"
    fi
done

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All shaders compiled successfully!${NC}"
    echo ""
    echo "Output directory: ${OUT_DIR}"
    echo ""
    ls -la "${OUT_DIR}"/*.spv 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}'
else
    echo -e "${RED}${FAILED} shader(s) failed to compile.${NC}"
    exit 1
fi
