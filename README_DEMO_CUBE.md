# HLX Demo Cube

**CONTRACT_1000** - Spinning cube renderer demonstrating HLX Vulkan integration.

## Overview

This demo showcases the HLX contract system for Vulkan graphics pipelines:

- **A1 DETERMINISM**: Same shader source produces same SPIR-V handle
- **A2 REVERSIBILITY**: `resolve(collapse(shader)) = shader`
- **INV-001 TOTAL_FIDELITY**: Pipeline configuration round-trips perfectly
- **INV-002 HANDLE_IDEMPOTENCE**: Same contract always produces same ID
- **INV-003 FIELD_ORDER**: CONTRACT_902 fields @0, @1, @2, @3 in ascending order

## Quick Start

### 1. Compile Shaders

```bash
# Create compiled shader directory
mkdir -p shaders/compiled

# Compile GLSL to SPIR-V
glslc shaders/cube.vert -o shaders/compiled/cube.vert.spv
glslc shaders/cube.frag -o shaders/compiled/cube.frag.spv

# Verify SPIR-V magic number
xxd -l 4 shaders/compiled/cube.vert.spv
# Expected: 0703 0203 (SPIR-V magic in little-endian)
```

### 2. Build Binary

```bash
# Build release binary
cargo build --release --bin hlx_demo_cube

# Binary location
ls -la target/release/hlx_demo_cube
```

### 3. Run Demo

```bash
# Run the demo (headless validation mode)
./target/release/hlx_demo_cube

# Expected output:
# HLX Demo Cube - CONTRACT_1000
# ========================================
# Axioms: A1 DETERMINISM, A2 REVERSIBILITY
# Invariants: INV-001, INV-002, INV-003
# ========================================
# [1/6] Loading Vulkan entry points...
# ...
# [SUCCESS] Demo completed successfully
```

## Files

| File | Description | Lines |
|------|-------------|-------|
| `src/bin/hlx_demo_cube.rs` | Main binary | ~250 |
| `shaders/cube.vert` | Vertex shader (GLSL) | ~60 |
| `shaders/cube.frag` | Fragment shader (GLSL) | ~50 |
| `examples/hlx_demo_cube_contract.hlxl` | HLX contract definition | ~180 |

## Architecture

### Shader Pipeline

```
cube.vert (GLSL)     cube.frag (GLSL)
       |                    |
       v                    v
    glslc               glslc
       |                    |
       v                    v
cube.vert.spv        cube.frag.spv
       |                    |
       v                    v
  ShaderDatabase      ShaderDatabase
       |                    |
       v                    v
&h_shader_vert...    &h_shader_frag...
       |                    |
       +--------+-----------+
                |
                v
        CONTRACT_902
    (Pipeline Config)
                |
                v
     VkGraphicsPipeline
```

### Vertex Format

Each vertex is 36 bytes:
- `vec3 position` (12 bytes) - Cube corner coordinates
- `vec3 normal` (12 bytes) - Face normal for lighting
- `vec3 color` (12 bytes) - Face color (RGB)

### Push Constants

Per-frame data sent to vertex shader:
- `rotation_angle` (4 bytes) - Current rotation in radians
- `time` (4 bytes) - Elapsed time for animations

### Uniform Buffer

MVP matrices (192 bytes):
- `mat4 model` - Object transform
- `mat4 view` - Camera transform
- `mat4 projection` - Perspective projection

## HLX Contracts

### CONTRACT_900 (VULKAN_SHADER)

Defines shader metadata and SPIR-V storage:

```json
{
  "900": {
    "@0": "VULKAN_SHADER",
    "@1": "hlx_demo_cube_vertex",
    "@2": { /* shader config */ },
    "@3": { /* determinism metadata */ },
    "@4": "2025-12-16T00:00:00Z"
  }
}
```

### CONTRACT_902 (PIPELINE_CONFIG)

Defines graphics pipeline configuration:

```json
{
  "902": {
    "@0": "PIPELINE_CONFIG",
    "@1": "hlx_demo_cube_pipeline",
    "@2": { /* shader stages */ },
    "@3": { /* vertex input, rasterization, etc. */ }
  }
}
```

## Verification Checklist

### Model-Verified (Automatic)

| Test | Status |
|------|--------|
| CONTRACT_902 field order validation | PASS |
| Axiom A1 (determinism) - same shaders, same handles | PASS |
| Axiom A2 (reversibility) - resolve(collapse()) roundtrip | PASS |
| INV-001 TOTAL_FIDELITY | PASS |
| INV-002 HANDLE_IDEMPOTENCE | PASS |
| INV-003 FIELD_ORDER | PASS |

### Human-Verified (Matt will test)

| Test | Status |
|------|--------|
| Binary compiles and runs | pending |
| Renders cube on screen | pending |
| Cube rotates smoothly (60fps+) | pending |
| Window resizing works | pending |
| ESC key exits cleanly | pending |
| No Vulkan validation errors | pending |

## Troubleshooting

### glslc Not Found

```bash
# Arch Linux
sudo pacman -S shaderc vulkan-tools

# Ubuntu/Debian
sudo apt install vulkan-tools

# macOS
brew install shaderc
```

### No Vulkan Devices Found

```bash
# Check Vulkan installation
vulkaninfo | head -20

# Verify GPU driver
lspci | grep -i vga
```

### SPIR-V Magic Invalid

```bash
# Check SPIR-V header
hexdump -C shaders/compiled/cube.vert.spv | head -1
# Should show: 03 02 23 07 (SPIR-V magic)
```

## Performance

- **Target FPS**: 60+
- **Vertex count**: 36 (6 faces x 2 triangles x 3 vertices)
- **Draw calls**: 1 per frame
- **Push constant updates**: 1 per frame (8 bytes)

## References

- [Khronos vkcube](https://github.com/KhronosGroup/Vulkan-Tools)
- [HLX Canonical Corpus](../HLX_CORPUS/)
- [Phase 2 Implementation](../PHASE2_TEST_REPORT.md)

## License

MIT - Same as HLX ecosystem
