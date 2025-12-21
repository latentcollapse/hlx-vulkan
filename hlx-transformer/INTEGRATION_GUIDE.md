# Integration Guide

This guide explains how to integrate the HLX Transformer components with your existing Vulkan infrastructure.

## Prerequisites

Your codebase should have these components:
- `Buffer` - GPU memory management
- `ComputePipeline` - Compute pipeline creation and dispatch
- `ShaderModule` - SPIR-V loading
- `Device` - Vulkan device wrapper
- `CommandBufferPool` - Command buffer management

## Step 1: Copy Files

```bash
# Copy shaders
cp -r hlx-transformer/shader/*.glsl your_project/shader/

# Copy Rust modules
cp hlx-transformer/src/tensor.rs your_project/src/
cp hlx-transformer/src/transformer_config.rs your_project/src/
cp hlx-transformer/src/gemm_kernel.rs your_project/src/
cp hlx-transformer/src/attention_kernel.rs your_project/src/
cp hlx-transformer/src/transformer_layer.rs your_project/src/
cp hlx-transformer/src/bin/train_transformer.rs your_project/src/bin/
```

## Step 2: Update lib.rs Stubs

The provided `lib.rs` contains stub modules. Replace these with your actual implementations:

### Buffer Module
```rust
// Replace this stub:
pub mod buffer {
    pub struct Buffer;
    // ...
}

// With your actual implementation (import from your crate):
pub use crate::buffer::Buffer;
```

### ComputePipeline Module
```rust
// Replace the stub with your actual ComputePipeline

// Your implementation should provide:
impl ComputePipeline {
    pub fn new(
        device: Arc<Device>,
        shader: &ShaderModule,
        bindings: &DescriptorBindings,
        push_constant_size: u32,
    ) -> Result<Self, VulkanErrorKind>;
    
    pub fn allocate_descriptor_set(&self) -> Result<vk::DescriptorSet, VulkanErrorKind>;
    
    pub fn update_descriptor_set(
        &self,
        set: vk::DescriptorSet,
        bindings: &[(u32, vk::Buffer)],
    ) -> Result<(), VulkanErrorKind>;
    
    pub fn record_dispatch<T>(
        &self,
        cmd_buffer: vk::CommandBuffer,
        desc_set: vk::DescriptorSet,
        push_constants: Option<&T>,
        groups: (u32, u32, u32),
    );
}
```

### Buffer Requirements
```rust
impl Buffer {
    pub fn new(
        device: Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Self, VulkanErrorKind>;
    
    pub fn buffer(&self) -> vk::Buffer;
    
    pub fn upload_data<T>(&self, data: &[T]) -> Result<(), VulkanErrorKind>;
    
    pub fn download_data<T>(&self, data: &mut [T]) -> Result<(), VulkanErrorKind>;
}
```

## Step 3: Compile Shaders

```bash
cd your_project
./build_shaders.sh
```

This produces `.spv` files in `shader/spv/`.

## Step 4: Load Shaders

```rust
use std::fs;

fn load_shader(device: &Arc<Device>, name: &str) -> Result<ShaderModule, Error> {
    let spv_path = format!("shader/spv/{}.spv", name);
    let spv_bytes = fs::read(&spv_path)?;
    ShaderModule::new(device.clone(), &spv_bytes, "main".to_string())
}

// Load all shaders
let gemm_shader = load_shader(&device, "gemm")?;
let gemm_backward_shader = load_shader(&device, "gemm_backward")?;
let layernorm_forward_shader = load_shader(&device, "layernorm_forward")?;
// ... etc
```

## Step 5: Create Kernels

```rust
use hlx_transformer::*;

// GEMM kernel
let gemm = GemmKernel::new(
    device.clone(),
    &gemm_spv,
    &gemm_backward_spv,
)?;

// Test it:
let a = Tensor::from_f32(&a_data, &[512, 256], device.clone())?;
let b = Tensor::from_f32(&b_data, &[256, 512], device.clone())?;
let c = Tensor::zeros(&[512, 512], DType::F32, device.clone())?;

// Record to command buffer
gemm.record_forward(
    cmd_buffer,
    a.buffer(),
    b.buffer(),
    c.buffer(),
    None,  // no bias
    512, 256, 512,
)?;
```

## Step 6: Build the Model

```rust
let config = TransformerConfig::tiny();
config.validate()?;

let model = TransformerModel::new(config, batch_size, device.clone())?;
model.summary();
```

## Step 7: Wire Up Training Loop

See `src/bin/train_transformer.rs` for the complete training loop structure.

Key integration points:

```rust
// 1. Load corpus
let examples = load_corpus(&corpus_path);

// 2. Tokenize
let tokenizer = CharTokenizer::new();
let batches = create_batches(&examples, &tokenizer, batch_size, max_seq_len);

// 3. Training loop
for epoch in 1..=num_epochs {
    for batch in &batches {
        // Forward
        let logits = model.forward(&batch.input_ids)?;
        
        // Loss
        let loss = cross_entropy_forward(&logits, &batch.target_ids)?;
        
        // Backward
        let grad = cross_entropy_backward(&logits, &batch.target_ids)?;
        model.backward(&grad)?;
        
        // Update
        optimizer.step()?;
    }
    
    // Checkpoint
    if epoch % 10 == 0 {
        save_checkpoint(&model, epoch)?;
    }
}
```

## Descriptor Binding Numbers

**CRITICAL**: Ensure your Rust bindings match the GLSL bindings:

| Shader | Binding 0 | Binding 1 | Binding 2 | Binding 3 | Binding 4 |
|--------|-----------|-----------|-----------|-----------|-----------|
| gemm | A | B | C | bias | - |
| gemm_backward | input1 | input2 | output | - | - |
| layernorm_forward | input | output | gamma | beta | stats |
| softmax_forward | input | output | - | - | - |
| cross_entropy | logits | targets | losses | softmax | - |
| adam_update | params | grads | m | v | - |

## Push Constant Layouts

All push constants use `#[repr(C)]` for correct memory layout:

```rust
// GEMM
#[repr(C)]
struct GemmPushConstants {
    m: u32,
    k: u32,
    n: u32,
    use_bias: u32,
}  // 16 bytes

// LayerNorm
#[repr(C)]
struct LayerNormPushConstants {
    num_positions: u32,
    d_model: u32,
    eps: f32,
}  // 12 bytes (padded to 16)

// Adam
#[repr(C)]
struct AdamPushConstants {
    num_params: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_t: f32,
    beta2_t: f32,
}  // 28 bytes (padded to 32)
```

## Memory Barriers

Insert barriers between dependent dispatches:

```rust
fn record_barrier(device: &Device, cmd_buffer: vk::CommandBuffer) {
    let memory_barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .build();
    
    unsafe {
        device.cmd_pipeline_barrier(
            cmd_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }
}
```

## Common Issues

### 1. Wrong Buffer Size
```
VALIDATION ERROR: Buffer size mismatch
```
**Fix**: Ensure buffer size = numel * sizeof(f32)

### 2. Descriptor Set Not Updated
```
Access to uninitialized descriptor
```
**Fix**: Call `update_descriptor_set` before dispatch

### 3. Missing Barrier
```
Race condition / non-deterministic output
```
**Fix**: Add barrier after every dispatch before dependent reads

### 4. Push Constant Alignment
```
Push constant data not aligned
```
**Fix**: Use `#[repr(C)]` and verify struct size matches GLSL layout

## Testing Integration

```rust
#[test]
fn test_gemm_integration() {
    // Create device
    let device = create_test_device();
    
    // Load shader
    let spv = include_bytes!("shader/spv/gemm.spv");
    let gemm = GemmKernel::new(device.clone(), spv, backward_spv)?;
    
    // Create test matrices
    let a = vec![1.0f32; 512 * 256];
    let b = vec![1.0f32; 256 * 512];
    
    let a_tensor = Tensor::from_f32(&a, &[512, 256], device.clone())?;
    let b_tensor = Tensor::from_f32(&b, &[256, 512], device.clone())?;
    let c_tensor = Tensor::zeros(&[512, 512], DType::F32, device.clone())?;
    
    // Execute
    let cmd = command_pool.allocate()?;
    gemm.record_forward(cmd, a_tensor.buffer(), b_tensor.buffer(), c_tensor.buffer(), None, 512, 256, 512)?;
    submit_and_wait(cmd)?;
    
    // Verify
    let c_data = c_tensor.to_f32()?;
    assert!((c_data[0] - 256.0).abs() < 1e-3);  // 1×1 dot product of 256 ones
}
```

## Next Steps

1. Run unit tests for each kernel
2. Test determinism (3 identical runs)
3. Run full training on tiny model
4. Scale up to small/medium models
5. Compare loss curve to CUDA baseline
