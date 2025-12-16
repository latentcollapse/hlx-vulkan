#!/bin/bash

# Build script for HLX N-body shaders
# Compiles GLSL shaders to SPIR-V format

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPILED_DIR="$SCRIPT_DIR/compiled"

# Create compiled output directory
mkdir -p "$COMPILED_DIR"

echo "Building HLX N-body shaders..."
echo "=============================="

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Please install Vulkan SDK."
    echo "  Ubuntu/Debian: sudo apt install vulkan-tools"
    echo "  Fedora: sudo dnf install vulkan-tools"
    echo "  macOS: brew install vulkan-sdk"
    exit 1
fi

# Compile compute shader
if [ -f "$SCRIPT_DIR/nbody.comp" ]; then
    echo "Compiling: nbody.comp"
    glslc -fshader-stage=compute "$SCRIPT_DIR/nbody.comp" \
        -o "$COMPILED_DIR/nbody.comp.spv"
    echo "  -> $COMPILED_DIR/nbody.comp.spv"
else
    echo "Warning: nbody.comp not found"
fi

# Compile vertex shader
if [ -f "$SCRIPT_DIR/sphere.vert" ]; then
    echo "Compiling: sphere.vert"
    glslc -fshader-stage=vertex "$SCRIPT_DIR/sphere.vert" \
        -o "$COMPILED_DIR/sphere.vert.spv"
    echo "  -> $COMPILED_DIR/sphere.vert.spv"
else
    echo "Warning: sphere.vert not found"
fi

# Compile fragment shader
if [ -f "$SCRIPT_DIR/sphere.frag" ]; then
    echo "Compiling: sphere.frag"
    glslc -fshader-stage=fragment "$SCRIPT_DIR/sphere.frag" \
        -o "$COMPILED_DIR/sphere.frag.spv"
    echo "  -> $COMPILED_DIR/sphere.frag.spv"
else
    echo "Warning: sphere.frag not found"
fi

echo ""
echo "Shader compilation complete!"
echo "Output directory: $COMPILED_DIR"
echo ""

# List compiled shaders
echo "Compiled shaders:"
ls -lh "$COMPILED_DIR"/*.spv 2>/dev/null || echo "  (No SPIR-V files generated)"
