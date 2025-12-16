#!/bin/bash
# HLX Demo Cube - Shader Compilation Script
# CONTRACT_1000: hlx-demo-cube
#
# Usage: bash shaders/build_shaders.sh
#
# This script compiles GLSL shaders to SPIR-V using glslc (preferred)
# or glslangValidator as fallback.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/compiled"

echo "HLX Demo Cube - Shader Compilation"
echo "==================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find compiler
if command -v glslc &> /dev/null; then
    COMPILER="glslc"
    echo "Using glslc compiler"
elif command -v glslangValidator &> /dev/null; then
    COMPILER="glslangValidator"
    echo "Using glslangValidator compiler"
else
    echo "ERROR: No GLSL compiler found!"
    echo "Install one of:"
    echo "  - Arch Linux: sudo pacman -S shaderc"
    echo "  - Ubuntu: sudo apt install vulkan-tools"
    echo "  - macOS: brew install shaderc"
    exit 1
fi

# Compile vertex shader
echo ""
echo "Compiling cube.vert..."
if [ "$COMPILER" = "glslc" ]; then
    glslc "$SCRIPT_DIR/cube.vert" -o "$OUTPUT_DIR/cube.vert.spv"
else
    glslangValidator -V "$SCRIPT_DIR/cube.vert" -o "$OUTPUT_DIR/cube.vert.spv"
fi

# Compile fragment shader
echo "Compiling cube.frag..."
if [ "$COMPILER" = "glslc" ]; then
    glslc "$SCRIPT_DIR/cube.frag" -o "$OUTPUT_DIR/cube.frag.spv"
else
    glslangValidator -V "$SCRIPT_DIR/cube.frag" -o "$OUTPUT_DIR/cube.frag.spv"
fi

# Verify SPIR-V magic number
echo ""
echo "Verifying SPIR-V binaries..."

verify_spirv() {
    local file="$1"
    local magic=$(xxd -l 4 -p "$file" 2>/dev/null | sed 's/\(..\)/\1 /g')
    if [ "$magic" = "03 02 23 07 " ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        echo "  [OK] $(basename "$file"): $size bytes, magic=0x07230203"
        return 0
    else
        echo "  [FAIL] $(basename "$file"): Invalid magic: $magic"
        return 1
    fi
}

verify_spirv "$OUTPUT_DIR/cube.vert.spv"
verify_spirv "$OUTPUT_DIR/cube.frag.spv"

echo ""
echo "==================================="
echo "Compilation successful!"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.spv
echo ""
echo "Next step: cargo build --release --bin hlx_demo_cube"
