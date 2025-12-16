#!/bin/bash

# Shader compilation script for HLX compute particles demo
# Compiles GLSL shaders to SPIR-V using glslc (Vulkan SDK)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILED_DIR="$SCRIPT_DIR/compiled"

# Create compiled output directory
mkdir -p "$COMPILED_DIR"

echo "HLX Compute Particles - Shader Compilation"
echo "=========================================="
echo "Source directory: $SCRIPT_DIR"
echo "Output directory: $COMPILED_DIR"
echo ""

# Check for glslc compiler
if ! command -v glslc &> /dev/null; then
    echo "ERROR: glslc not found"
    echo "Please install Vulkan SDK: https://vulkan.lunarg.com"
    exit 1
fi

echo "Using glslc: $(glslc --version)"
echo ""

# Compile compute shader
echo "Compiling particle.comp..."
glslc -fshader-stage=compute \
    "$SCRIPT_DIR/particle.comp" \
    -o "$COMPILED_DIR/particle.comp.spv" \
    -O -g

echo "  ✓ particle.comp.spv ($(stat -f%z "$COMPILED_DIR/particle.comp.spv" 2>/dev/null || stat -c%s "$COMPILED_DIR/particle.comp.spv") bytes)"

# Compile vertex shader
echo "Compiling particle.vert..."
glslc -fshader-stage=vertex \
    "$SCRIPT_DIR/particle.vert" \
    -o "$COMPILED_DIR/particle.vert.spv" \
    -O -g

echo "  ✓ particle.vert.spv ($(stat -f%z "$COMPILED_DIR/particle.vert.spv" 2>/dev/null || stat -c%s "$COMPILED_DIR/particle.vert.spv") bytes)"

# Compile fragment shader
echo "Compiling particle.frag..."
glslc -fshader-stage=fragment \
    "$SCRIPT_DIR/particle.frag" \
    -o "$COMPILED_DIR/particle.frag.spv" \
    -O -g

echo "  ✓ particle.frag.spv ($(stat -f%z "$COMPILED_DIR/particle.frag.spv" 2>/dev/null || stat -c%s "$COMPILED_DIR/particle.frag.spv") bytes)"

echo ""
echo "Shader compilation complete!"
echo ""
echo "Generated files:"
ls -lh "$COMPILED_DIR/"*.spv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Verification:"
file "$COMPILED_DIR"/*.spv

echo ""
echo "✓ All shaders compiled successfully"
