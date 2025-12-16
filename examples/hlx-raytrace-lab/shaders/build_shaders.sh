#!/bin/bash
# HLX Ray Tracing Lab - Shader Compilation Script
# Compiles GLSL ray tracing shaders to SPIR-V

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILED_DIR="${SCRIPT_DIR}/compiled"

# Ensure compiled directory exists
mkdir -p "${COMPILED_DIR}"

# Check for glslc compiler
if ! command -v glslc &> /dev/null; then
    echo "ERROR: glslc not found. Install Vulkan SDK (glslang tools)"
    echo "  Ubuntu: sudo apt install glslang-tools"
    echo "  Fedora: sudo dnf install glslang-tools"
    exit 1
fi

echo "Compiling ray tracing shaders..."

# Compile raygen shader
echo "  Compiling raytrace.rgen..."
glslc -fshader-stage=raygen "${SCRIPT_DIR}/raytrace.rgen" \
    -o "${COMPILED_DIR}/raytrace.rgen.spv"

# Compile closest-hit shader
echo "  Compiling raytrace.rchit..."
glslc -fshader-stage=closesthit "${SCRIPT_DIR}/raytrace.rchit" \
    -o "${COMPILED_DIR}/raytrace.rchit.spv"

# Compile miss shader
echo "  Compiling raytrace.rmiss..."
glslc -fshader-stage=miss "${SCRIPT_DIR}/raytrace.rmiss" \
    -o "${COMPILED_DIR}/raytrace.rmiss.spv"

echo ""
echo "✓ Ray tracing shaders compiled successfully"
echo "  Raygen:      ${COMPILED_DIR}/raytrace.rgen.spv"
echo "  Closest-hit: ${COMPILED_DIR}/raytrace.rchit.spv"
echo "  Miss:        ${COMPILED_DIR}/raytrace.rmiss.spv"
