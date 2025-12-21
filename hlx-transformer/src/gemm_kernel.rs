//! General Matrix Multiply (GEMM) Kernel
//!
//! Provides GPU-accelerated matrix multiplication with optional fused bias.
//! Foundation for all linear operations in the transformer.

use std::sync::Arc;
use ash::vk;

use crate::buffer::Buffer;
use crate::compute::{ComputePipeline, DescriptorBindingBuilder};
use crate::device::Device;
use crate::error::VulkanErrorKind;
use crate::shader::ShaderModule;
use crate::tensor::Tensor;

// =============================================================================
// CONSTANTS
// =============================================================================

const TILE_SIZE: u32 = 16;

// =============================================================================
// PUSH CONSTANTS
// =============================================================================

/// Push constants for GEMM forward pass
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GemmPushConstants {
    pub m: u32,        // Rows of A
    pub k: u32,        // Cols of A = Rows of B
    pub n: u32,        // Cols of B
    pub use_bias: u32, // 1 if bias should be added
}

/// Push constants for GEMM backward pass
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GemmBackwardPushConstants {
    pub m: u32,    // Original forward M
    pub k: u32,    // Original forward K
    pub n: u32,    // Original forward N
    pub mode: u32, // 0 = compute dA, 1 = compute dB
}

// =============================================================================
// GEMM KERNEL
// =============================================================================

/// GEMM kernel for matrix multiplication
///
/// Supports:
/// - C = A × B
/// - C = A × B + bias
pub struct GemmKernel {
    forward_pipeline: ComputePipeline,
    backward_pipeline: ComputePipeline,
    device: Arc<Device>,
}

impl GemmKernel {
    /// Creates a new GEMM kernel
    pub fn new(
        device: Arc<Device>,
        forward_spv: &[u8],
        backward_spv: &[u8],
    ) -> Result<Self, VulkanErrorKind> {
        // Forward pipeline bindings
        let forward_bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(0)  // Matrix A
            .add_storage_buffer(1)  // Matrix B
            .add_storage_buffer(2)  // Matrix C (output)
            .add_storage_buffer(3)  // Bias (optional)
            .build();
        
        let forward_shader = ShaderModule::new(
            device.clone(),
            forward_spv,
            "main".to_string(),
        )?;
        
        let forward_pipeline = ComputePipeline::new(
            device.clone(),
            &forward_shader,
            &forward_bindings,
            std::mem::size_of::<GemmPushConstants>() as u32,
        )?;
        
        // Backward pipeline bindings
        let backward_bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(0)  // Input 1
            .add_storage_buffer(1)  // Input 2
            .add_storage_buffer(2)  // Output
            .build();
        
        let backward_shader = ShaderModule::new(
            device.clone(),
            backward_spv,
            "main".to_string(),
        )?;
        
        let backward_pipeline = ComputePipeline::new(
            device.clone(),
            &backward_shader,
            &backward_bindings,
            std::mem::size_of::<GemmBackwardPushConstants>() as u32,
        )?;
        
        Ok(Self {
            forward_pipeline,
            backward_pipeline,
            device,
        })
    }
    
    /// Records forward pass: C = A × B (+ bias)
    ///
    /// # Arguments
    /// * `cmd_buffer` - Command buffer to record into
    /// * `a` - Matrix A (M × K)
    /// * `b` - Matrix B (K × N)
    /// * `c` - Output matrix C (M × N)
    /// * `bias` - Optional bias vector (N,)
    /// * `m`, `k`, `n` - Matrix dimensions
    pub fn record_forward(
        &self,
        cmd_buffer: vk::CommandBuffer,
        a: vk::Buffer,
        b: vk::Buffer,
        c: vk::Buffer,
        bias: Option<vk::Buffer>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), VulkanErrorKind> {
        let desc_set = self.forward_pipeline.allocate_descriptor_set()?;
        
        // Create dummy bias buffer if not provided
        let bias_buffer = bias.unwrap_or(a);  // Reuse A as placeholder
        let use_bias = if bias.is_some() { 1 } else { 0 };
        
        self.forward_pipeline.update_descriptor_set(
            desc_set,
            &[
                (0, a),
                (1, b),
                (2, c),
                (3, bias_buffer),
            ],
        )?;
        
        let push_constants = GemmPushConstants { m, k, n, use_bias };
        
        // Dispatch: one workgroup per TILE_SIZE × TILE_SIZE block of C
        let num_groups_x = (n + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m + TILE_SIZE - 1) / TILE_SIZE;
        
        self.forward_pipeline.record_dispatch(
            cmd_buffer,
            desc_set,
            Some(&push_constants),
            (num_groups_x, num_groups_y, 1),
        );
        
        Ok(())
    }
    
    /// Records backward pass for gradient w.r.t. A
    ///
    /// dA = dC × B^T
    pub fn record_backward_a(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dc: vk::Buffer,    // Gradient of C (M × N)
        b: vk::Buffer,     // Matrix B (K × N)
        da: vk::Buffer,    // Output: gradient of A (M × K)
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), VulkanErrorKind> {
        let desc_set = self.backward_pipeline.allocate_descriptor_set()?;
        
        self.backward_pipeline.update_descriptor_set(
            desc_set,
            &[
                (0, dc),
                (1, b),
                (2, da),
            ],
        )?;
        
        let push_constants = GemmBackwardPushConstants { m, k, n, mode: 0 };
        
        // Output is (M × K)
        let num_groups_x = (k + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m + TILE_SIZE - 1) / TILE_SIZE;
        
        self.backward_pipeline.record_dispatch(
            cmd_buffer,
            desc_set,
            Some(&push_constants),
            (num_groups_x, num_groups_y, 1),
        );
        
        Ok(())
    }
    
    /// Records backward pass for gradient w.r.t. B
    ///
    /// dB = A^T × dC
    pub fn record_backward_b(
        &self,
        cmd_buffer: vk::CommandBuffer,
        a: vk::Buffer,     // Matrix A (M × K)
        dc: vk::Buffer,    // Gradient of C (M × N)
        db: vk::Buffer,    // Output: gradient of B (K × N)
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), VulkanErrorKind> {
        let desc_set = self.backward_pipeline.allocate_descriptor_set()?;
        
        self.backward_pipeline.update_descriptor_set(
            desc_set,
            &[
                (0, a),
                (1, dc),
                (2, db),
            ],
        )?;
        
        let push_constants = GemmBackwardPushConstants { m, k, n, mode: 1 };
        
        // Output is (K × N)
        let num_groups_x = (n + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (k + TILE_SIZE - 1) / TILE_SIZE;
        
        self.backward_pipeline.record_dispatch(
            cmd_buffer,
            desc_set,
            Some(&push_constants),
            (num_groups_x, num_groups_y, 1),
        );
        
        Ok(())
    }
    
    /// Inserts a memory barrier for compute-to-compute synchronization
    pub fn record_barrier(&self, cmd_buffer: vk::CommandBuffer) {
        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        
        unsafe {
            self.device.handle().cmd_pipeline_barrier(
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
}

// =============================================================================
// LINEAR LAYER
// =============================================================================

/// Linear layer: y = x @ W^T + b
///
/// Wraps GEMM kernel with weight and bias tensors.
pub struct LinearLayer {
    /// Weight matrix (out_features, in_features)
    pub weight: Tensor,
    /// Bias vector (out_features,) - optional
    pub bias: Option<Tensor>,
    /// Weight gradient
    pub weight_grad: Tensor,
    /// Bias gradient
    pub bias_grad: Option<Tensor>,
    /// Input features
    pub in_features: u32,
    /// Output features
    pub out_features: u32,
}

impl LinearLayer {
    /// Creates a new linear layer with Xavier initialization
    pub fn new(
        in_features: u32,
        out_features: u32,
        use_bias: bool,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        use crate::transformer_config::{xavier_uniform, zeros};
        
        let weight = xavier_uniform(&[out_features, in_features], device.clone())?;
        let weight_grad = zeros(&[out_features, in_features], device.clone())?;
        
        let (bias, bias_grad) = if use_bias {
            (
                Some(zeros(&[out_features], device.clone())?),
                Some(zeros(&[out_features], device)?),
            )
        } else {
            (None, None)
        };
        
        Ok(Self {
            weight,
            bias,
            weight_grad,
            bias_grad,
            in_features,
            out_features,
        })
    }
    
    /// Returns total parameter count
    pub fn param_count(&self) -> usize {
        let weight_count = (self.in_features * self.out_features) as usize;
        let bias_count = if self.bias.is_some() { self.out_features as usize } else { 0 };
        weight_count + bias_count
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_push_constants_size() {
        assert_eq!(std::mem::size_of::<GemmPushConstants>(), 16);
        assert_eq!(std::mem::size_of::<GemmBackwardPushConstants>(), 16);
    }
    
    #[test]
    fn test_dispatch_groups() {
        // 512 × 256 matrix with 16×16 tiles
        let m = 512u32;
        let n = 256u32;
        
        let groups_x = (n + TILE_SIZE - 1) / TILE_SIZE;
        let groups_y = (m + TILE_SIZE - 1) / TILE_SIZE;
        
        assert_eq!(groups_x, 16);  // 256 / 16
        assert_eq!(groups_y, 32);  // 512 / 16
    }
}
