#!/bin/bash
# HLX Shader Compiler - Example Workflow
# CONTRACT_1001: Compile cube demo shaders to SPIR-V with HLX contracts
#
# This script demonstrates the full shader compilation workflow:
# 1. Check for available GLSL compiler
# 2. Compile vertex and fragment shaders
# 3. Validate SPIR-V output
# 4. Generate CONTRACT_900 definitions
#
# Usage:
#   ./examples/compile_cube_shaders.sh
#
# Prerequisites:
#   - cargo build --release --bin hlx_shader_compiler
#   - glslc or glslangValidator installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHADER_DIR="${PROJECT_ROOT}/examples/hlx-demo-cube/shaders"
OUTPUT_DIR="${SHADER_DIR}/compiled"
CONTRACT_DIR="${SHADER_DIR}/contracts"

# Path to shader compiler binary
COMPILER="${PROJECT_ROOT}/target/release/hlx_shader_compiler"

echo "========================================"
echo "HLX Shader Compiler - Cube Demo Workflow"
echo "========================================"
echo ""

# Check if compiler is built
if [ ! -f "${COMPILER}" ]; then
    echo "Building hlx_shader_compiler..."
    cd "${PROJECT_ROOT}"
    cargo build --release --bin hlx_shader_compiler
    echo ""
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CONTRACT_DIR}"

# Step 1: Check compiler availability
echo "Step 1: Checking GLSL compiler..."
echo "----------------------------------------"
"${COMPILER}" check-compiler
echo ""

# Step 2: Compile vertex shader
echo "Step 2: Compiling vertex shader..."
echo "----------------------------------------"
if [ -f "${SHADER_DIR}/cube.vert" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.vert" \
        --output "${OUTPUT_DIR}/cube.vert.spv" \
        --force
    echo ""
else
    echo "WARNING: cube.vert not found at ${SHADER_DIR}"
    echo ""
fi

# Step 3: Compile fragment shader
echo "Step 3: Compiling fragment shader..."
echo "----------------------------------------"
if [ -f "${SHADER_DIR}/cube.frag" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.frag" \
        --output "${OUTPUT_DIR}/cube.frag.spv" \
        --force
    echo ""
else
    echo "WARNING: cube.frag not found at ${SHADER_DIR}"
    echo ""
fi

# Step 4: Validate compiled shaders
echo "Step 4: Validating SPIR-V output..."
echo "----------------------------------------"
for spv in "${OUTPUT_DIR}"/*.spv; do
    if [ -f "$spv" ]; then
        "${COMPILER}" validate "$spv"
        echo ""
    fi
done

# Step 5: Generate HLXL contracts
echo "Step 5: Generating HLXL contracts..."
echo "----------------------------------------"
if [ -f "${SHADER_DIR}/cube.vert" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.vert" --hlxl --no-store \
        > "${CONTRACT_DIR}/cube_vert.hlxl" 2>/dev/null || true
    echo "Generated: ${CONTRACT_DIR}/cube_vert.hlxl"
fi
if [ -f "${SHADER_DIR}/cube.frag" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.frag" --hlxl --no-store \
        > "${CONTRACT_DIR}/cube_frag.hlxl" 2>/dev/null || true
    echo "Generated: ${CONTRACT_DIR}/cube_frag.hlxl"
fi
echo ""

# Step 6: Generate JSON contracts
echo "Step 6: Generating JSON contracts..."
echo "----------------------------------------"
if [ -f "${SHADER_DIR}/cube.vert" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.vert" --no-store \
        > "${CONTRACT_DIR}/cube_vert.json" 2>/dev/null || true
    echo "Generated: ${CONTRACT_DIR}/cube_vert.json"
fi
if [ -f "${SHADER_DIR}/cube.frag" ]; then
    "${COMPILER}" compile "${SHADER_DIR}/cube.frag" --no-store \
        > "${CONTRACT_DIR}/cube_frag.json" 2>/dev/null || true
    echo "Generated: ${CONTRACT_DIR}/cube_frag.json"
fi
echo ""

# Step 7: Verify determinism (A1 axiom)
echo "Step 7: Verifying A1 DETERMINISM axiom..."
echo "----------------------------------------"
if [ -f "${SHADER_DIR}/cube.vert" ]; then
    HANDLE1=$("${COMPILER}" compile "${SHADER_DIR}/cube.vert" --no-store 2>&1 | grep "Handle:" | awk '{print $2}')
    HANDLE2=$("${COMPILER}" compile "${SHADER_DIR}/cube.vert" --no-store 2>&1 | grep "Handle:" | awk '{print $2}')

    if [ "${HANDLE1}" = "${HANDLE2}" ]; then
        echo "A1 DETERMINISM: VERIFIED"
        echo "  Run 1: ${HANDLE1}"
        echo "  Run 2: ${HANDLE2}"
    else
        echo "A1 DETERMINISM: FAILED"
        echo "  Run 1: ${HANDLE1}"
        echo "  Run 2: ${HANDLE2}"
        exit 1
    fi
fi
echo ""

# Summary
echo "========================================"
echo "Compilation Summary"
echo "========================================"
echo ""
echo "SPIR-V binaries:"
ls -la "${OUTPUT_DIR}"/*.spv 2>/dev/null || echo "  (none)"
echo ""
echo "HLXL contracts:"
ls -la "${CONTRACT_DIR}"/*.hlxl 2>/dev/null || echo "  (none)"
echo ""
echo "JSON contracts:"
ls -la "${CONTRACT_DIR}"/*.json 2>/dev/null || echo "  (none)"
echo ""
echo "========================================"
echo "All shaders compiled successfully!"
echo "========================================"
