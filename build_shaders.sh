#!/bin/bash
# build_shaders.sh - Compile GLSL to SPIR-V
# Requires: glslc from Vulkan SDK

set -euo pipefail

SHADER_DIR="$(dirname "$0")/shader"
OUT_DIR="${SHADER_DIR}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Compiling gradient kernel shaders...${NC}"

# Check for glslc
if ! command -v glslc &> /dev/null; then
    echo -e "${RED}Error: glslc not found. Install Vulkan SDK.${NC}"
    exit 1
fi

# Compile each shader
compile_shader() {
    local src="$1"
    local dst="$2"
    
    echo -n "  Compiling $(basename "$src")... "
    
    if glslc --target-env=vulkan1.2 \
             -fshader-stage=compute \
             -O \
             "$src" \
             -o "$dst" 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        exit 1
    fi
}

compile_shader "${SHADER_DIR}/gradient_forward.glsl" "${OUT_DIR}/forward.spv"
compile_shader "${SHADER_DIR}/gradient_backward.glsl" "${OUT_DIR}/backward.spv"
compile_shader "${SHADER_DIR}/gradient_reduce.glsl" "${OUT_DIR}/reduce.spv"

echo ""
echo -e "${GREEN}All shaders compiled successfully!${NC}"
echo ""
echo "Output files:"
ls -la "${OUT_DIR}"/*.spv 2>/dev/null || echo "  (no .spv files found)"
