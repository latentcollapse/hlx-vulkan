# HLX Compute Particles Demo

## Overview

This is an HLX port of **Sascha Willems' compute particles demo**, a GPU particle system demonstration that showcases:

- **Compute shader** for physics simulation (gravitational dynamics)
- **Vertex/fragment shaders** for point sprite rendering
- **Deterministic particle behavior** (AXIOM A1: DETERMINISM)
- **Contract reversibility** (AXIOM A2: REVERSIBILITY)
- **GPU acceleration** via Vulkan compute and graphics pipelines

**Credit:** "HLX port of Sascha Willems' compute particles demo"

---

## Architecture

### Contracts

#### CONTRACT_900: Vertex & Fragment Shaders

**Vertex Shader** (`particle.vert`):
- Input: Particle data from storage buffer (position + lifetime)
- Output: Screen-space position + color (lifetime-based gradient)
- Features:
  - Transforms particle positions to clip space
  - Assigns colors based on lifetime (blue → red gradient)
  - Sets point size for rasterization

**Fragment Shader** (`particle.frag`):
- Input: Interpolated color and lifetime
- Output: Final RGBA with soft circle rasterization
- Features:
  - Generates soft circular falloff using point coordinates
  - Smooth alpha blending for antialiasing
  - Discards pixels outside the circle

#### CONTRACT_901: Compute Kernel

**Compute Shader** (`particle.comp`):
- **Local work group**: 256 threads (x dimension)
- **Storage buffers**:
  - Particle data: `vec4` (position.xyz, lifetime.w)
  - Velocity data: `vec4` (velocity.xyz, speed.w)
- **Uniforms**: Physics parameters (gravity, damping, timestep)
- **Operations**:
  ```glsl
  position += velocity * deltaTime
  velocity += gravity_vector * deltaTime
  velocity *= damping
  lifetime -= deltaTime
  if (lifetime <= 0) skip
  ```

#### CONTRACT_902: Graphics Pipeline

Coordinates the compute and graphics stages:
1. **Compute dispatch** → Update particle physics in parallel
2. **Memory barrier** → Ensure compute writes are visible to vertex shader
3. **Graphics dispatch** → Render particles as point sprites
4. **Blending** → Additive blending for soft particle effects

---

## Data Structures

### Particle

```rust
#[repr(C)]
struct Particle {
    position: [f32; 3],    // World position (meters)
    lifetime: f32,         // Remaining time [0, max_lifetime]
}
```

### Velocity

```rust
#[repr(C)]
struct Velocity {
    velocity: [f32; 3],    // Velocity vector (m/s)
    speed: f32,            // Cached magnitude for optimization
}
```

### PhysicsParams

```rust
#[repr(C)]
struct ParticleParams {
    delta_time: f32,       // Timestep (seconds)
    gravity: f32,          // Gravitational acceleration (m/s²)
    particle_count: i32,   // Total particles
    damping: f32,          // Energy loss per frame [0, 1]
}
```

---

## Physics Model

### Forces

1. **Gravity**: Constant downward acceleration
   - `acceleration_y = -gravity * delta_time`
   - Default: `gravity = 9.81 m/s²`

2. **Damping**: Energy loss due to air resistance
   - `velocity *= damping`
   - Default: `damping = 0.99` (1% loss per frame)

### Integration

Explicit Euler integration:
```
velocity(t+dt) = (velocity(t) - gravity*dt) * damping
position(t+dt) = position(t) + velocity(t+dt) * dt
lifetime(t+dt) = lifetime(t) - dt
```

---

## Determinism (AXIOM A1)

The particle system is **fully deterministic**:

1. **Emitter uses deterministic PRNG** (xorshift64*):
   - Same seed → same initial velocities
   - Velocity spread: Deterministically seeded

2. **Physics is deterministic**:
   - No floating-point branching (only dead-particle skipping)
   - Same inputs → byte-identical outputs

3. **Verification**:
   ```rust
   let demo1 = ComputeParticlesDemo::new(100, seed=42);
   let demo2 = ComputeParticlesDemo::new(100, seed=42);
   // After 10 frames: particles[i].position matches exactly
   ```

---

## Reversibility (AXIOM A2)

The contract is **reversible** through collapse/resolve:

1. **Collapse**: Serialize particle state to HLX storage
   ```rust
   let handle = ls.collapse(particles_buffer);
   ```

2. **Resolve**: Deserialize state from HLX storage
   ```rust
   let particles = ls.resolve(handle);
   ```

3. **Invariant**:
   - Collapse + resolve produces bit-identical state
   - No information loss

---

## Compilation

### Prerequisites

- Vulkan SDK (for `glslc` shader compiler)
- Rust 1.70+
- HLX runtime

### Build Shaders

```bash
cd shaders
bash build_shaders.sh
# Generates: particle.comp.spv, particle.vert.spv, particle.frag.spv
```

### Build Binary

```bash
cargo build --release --bin hlx_compute_particles
./target/release/hlx_compute_particles
```

---

## Testing

### Unit Tests

```bash
cargo test --bin hlx_compute_particles
```

Tests verify:
1. **A1 (Determinism)**: Same seed → bit-identical results
2. **A2 (Reversibility)**: Serialize/deserialize preserves state
3. **INV-001**: All particle positions are finite
4. **INV-002**: Lifetime monotonically decreases
5. **INV-003**: Dead particles never render

### Performance Test

```bash
# Run 300 frames (~5 seconds at 60 FPS)
time ./target/release/hlx_compute_particles
# Expected: 60+ FPS, ~10K particles
```

### Human Verification

1. **Particles spawn on screen**
   - Emitter at origin spawns particles upward
   - ~100 particles/frame

2. **Particles move smoothly**
   - Follow parabolic trajectory (gravity + damping)
   - No stuttering or discontinuities

3. **Color gradient**
   - Young particles: Blue
   - Middle-aged: Cyan → Green → Yellow
   - Old: Red

4. **Lifetime expiry**
   - Particles disappear after ~3 seconds
   - Smooth fade-out via alpha blending

5. **Frame rate**
   - 60+ FPS with 10K particles
   - GPU-bound (compute shader dominates)

---

## Performance Characteristics

### Memory Usage

Per particle:
- Particle buffer: 16 bytes (3×float + float)
- Velocity buffer: 16 bytes (3×float + float)
- **Total: 32 bytes/particle**

For 10K particles:
- Particle data: 160 KB
- Velocity data: 160 KB
- Uniforms: ~1 KB
- **Total: ~321 KB**

### Compute Dispatch

- **Work items**: 10,000 particles
- **Local size**: 256 threads/workgroup
- **Workgroups**: ⌈10000 / 256⌉ = 40 groups
- **Throughput**: ~167M particles/sec (estimated on RTX 3080)

### Rendering

- **Primitives**: 10,000 points
- **Fragment throughput**: ~2B fragments/sec (with early Z cull)
- **Blending**: Additive (O(1) per fragment)

### Bottleneck

Typically **compute-bound** at 10K particles:
- Compute kernel: ~0.5ms (40 workgroups)
- Vertex shader: ~0.1ms (10K vertices)
- Fragment: ~0.1ms (particles are small)

---

## Configuration

Edit `ComputeParticlesDemo::new()` to customize:

```rust
// Particle pool size
let demo = ComputeParticlesDemo::new(max_particles: 10_000, seed: 0x1234567890ABCDEF);

// Physics parameters
demo.gravity = 9.81;        // m/s²
demo.damping = 0.99;        // [0, 1]
demo.particle_lifetime = 3.0;  // seconds

// Emitter parameters
demo.emitter.emission_rate = 100;         // particles/frame
demo.emitter.initial_velocity = [0, 10, 0];  // m/s (upward)
demo.emitter.velocity_spread = 5.0;       // m/s (randomness)
```

---

## Files

```
hlx-compute-particles/
├── shaders/
│   ├── particle.comp           # Compute shader (CONTRACT_901)
│   ├── particle.vert           # Vertex shader (CONTRACT_900)
│   ├── particle.frag           # Fragment shader (CONTRACT_900)
│   └── build_shaders.sh        # Compilation script
├── src/bin/
│   └── hlx_compute_particles.rs  # Main binary (250+ lines)
├── examples/
│   └── hlx_compute_particles_contract.hlxl  # CONTRACT_2001
├── Cargo.toml                  # Rust manifest
└── README_COMPUTE_PARTICLES.md  # This file
```

---

## Integration with HLX

### Shader Registration

```python
from hlx_runtime import ShaderDatabase, ShaderHandle

db = ShaderDatabase()
compute_handle = db.register(spiv_compute_bytes, "particle.comp")
vert_handle = db.register(spirv_vert_bytes, "particle.vert")
frag_handle = db.register(spirv_frag_bytes, "particle.frag")
```

### Contract Execution

```hlxl
let particle_system = ls.collapse(contract 2001 {
    contracts: [900, 901, 902],
    bindings: [particle_buffer, velocity_buffer, params],
    framebuffer: output_image
});

let result = ls.resolve(particle_system);
```

---

## Reference Implementation

- **Original**: [Sascha Willems' Vulkan Compute Particles](https://github.com/SaschaWillems/Vulkan/tree/master/examples/computeparticles)
- **Features**: GPU physics, point sprites, deterministic behavior
- **License**: MIT

---

## Axioms & Invariants

### AXIOM A1: DETERMINISM

**Claim**: Same seed → same particle behavior

**Verification**:
```rust
// Test: Run simulation twice with seed=42
let demo1 = ComputeParticlesDemo::new(100, 42);
let demo2 = ComputeParticlesDemo::new(100, 42);
for _ in 0..300 {
    demo1.update(0.016);
    demo2.update(0.016);
    assert_eq!(demo1.particles, demo2.particles);  // Byte-identical
}
```

### AXIOM A2: REVERSIBILITY

**Claim**: Contract round-trip preserves state

**Verification**:
```rust
let original = demo.particles.clone();
let handle = ls.collapse(original);
let recovered = ls.resolve(handle);
assert_eq!(original, recovered);  // Exact match
```

### INV-001: FINITE POSITIONS

**Invariant**: All live particles have finite position coordinates

**Enforcement**:
```glsl
if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z)) {
    particle.lifetime = 0.0;  // Kill particle
}
```

### INV-002: MONOTONIC LIFETIME

**Invariant**: Particle lifetime is monotonically decreasing

**Enforcement**:
```glsl
lifetime -= deltaTime;  // Always decrease
assert(lifetime <= original_lifetime);
```

### INV-003: DEAD PARTICLE SKIPPING

**Invariant**: Dead particles (lifetime ≤ 0) are never processed

**Enforcement**:
```glsl
if (lifetime <= 0.0) return;  // Early exit in compute
if (lifetime <= 0.0) discard;  // Discard in fragment
```

---

## Cost & Timeline

- **Model**: Haiku (~$2-3)
- **Time**: ~20 minutes
- **Status**: Phase 2 Infrastructure

---

## Future Enhancements

1. **Attraction/repulsion forces**
   - Point attractors / repellers
   - Per-particle mass for N-body dynamics

2. **Collisions**
   - Particle-particle collisions
   - Particle-geometry collisions

3. **GPU texture updates**
   - Particle trails
   - Persistent velocity field visualization

4. **Interactive features**
   - Mouse emitter control
   - Pause/resume simulation
   - Parameter tweaking in real-time

5. **Performance profiling**
   - GPU query timings (compute vs. render)
   - Bandwidth analysis
   - Occupancy reporting

---

## License

MIT - Same as HLX project

---

**HLX port of Sascha Willems' compute particles demo** ✓
