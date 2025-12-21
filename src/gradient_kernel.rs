//! Dual-Pass Gradient Kernel for Deterministic ML Training
//!
//! This module provides GPU-accelerated gradient computation via three compute shaders:
//! - **Forward pass**: Computes activations, stores intermediate state
//! - **Backward pass**: Computes gradients, accumulates to per-workgroup staging
//! - **Reduce pass**: Deterministic fixed-order summation of workgroup gradients
//!
//! # Determinism Guarantee
//!
//! The two-phase reduction strategy eliminates cross-workgroup atomic operations,
//! ensuring bit-identical results across runs. This is critical for reproducible
//! ML training.
//!
//! # Example
//!
//! ```rust,ignore
//! let kernel = GradientKernel::new(
//!     device,
//!     memory_properties,
//!     &forward_spv,
//!     &backward_spv,
//!     &reduce_spv,
//!     input_size,
//!     output_size,
//!     batch_size,
//! )?;
//!
//! kernel.full_pass(&mut cmd_pool, input, output, output_grad, queue)?;
//! ```

use std::sync::Arc;
use ash::{vk, Device};

use crate::buffer::Buffer;
use crate::compute::{ComputePipeline, CommandBufferPool, DescriptorBindingBuilder};
use crate::shader::ShaderModule;
use crate::error::VulkanErrorKind;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Workgroup size for forward and backward passes
/// 256 threads = optimal warp utilization on modern GPUs
const WORKGROUP_SIZE: u32 = 256;

// =============================================================================
// PUSH CONSTANTS
// =============================================================================

/// Push constants for backward and reduce passes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GradientPushConstants {
    pub num_workgroups: u32,
    pub param_size: u32,
}

// =============================================================================
// UNIFORM BUFFER
// =============================================================================

/// Parameters uniform buffer layout (std140 alignment)
/// Must match shader layout exactly
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GradientParameters {
    pub input_size: u32,
    pub output_size: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
}

// =============================================================================
// GRADIENT KERNEL
// =============================================================================

/// Dual-pass gradient kernel with deterministic execution
///
/// Manages three compute pipelines (forward, backward, reduce) and all
/// intermediate buffers required for gradient-based training.
pub struct GradientKernel {
    // --- Compute Pipelines ---
    /// Forward pass: input → activation → output
    forward_pipeline: ComputePipeline,
    /// Backward pass: output_grad → input_grad + workgroup staging
    backward_pipeline: ComputePipeline,
    /// Reduce pass: workgroup staging → param_grad (deterministic)
    reduce_pipeline: ComputePipeline,

    // --- Intermediate Buffers ---
    /// Stores activations from forward pass for backward pass consumption
    activation_buffer: Buffer,
    /// Per-workgroup gradient staging (eliminates cross-workgroup atomics)
    workgroup_grad_buffer: Buffer,
    /// Input gradients (propagates to previous layer)
    input_grad_buffer: Buffer,
    /// Final accumulated parameter gradient
    param_grad_buffer: Buffer,
    /// Uniform buffer for parameters
    params_buffer: Buffer,
    /// Learnable weight buffer (single f32 for v1)
    weight_buffer: Buffer,

    // --- Descriptor Sets ---
    /// Pre-allocated descriptor set for forward pass
    forward_descriptor_set: vk::DescriptorSet,
    /// Pre-allocated descriptor set for backward pass
    backward_descriptor_set: vk::DescriptorSet,
    /// Pre-allocated descriptor set for reduce pass
    reduce_descriptor_set: vk::DescriptorSet,

    // --- Configuration ---
    /// Number of input elements
    input_size: u32,
    /// Number of output elements
    output_size: u32,
    /// Batch size (fixed at construction)
    batch_size: u32,
    /// Number of workgroups for forward/backward dispatch
    num_workgroups: u32,

    // --- Device Reference ---
    device: Arc<Device>,
}

impl GradientKernel {
    /// Creates a new gradient kernel with pre-allocated buffers and pipelines
    ///
    /// # Arguments
    ///
    /// * `device` - Vulkan device reference
    /// * `memory_properties` - Physical device memory properties for allocation
    /// * `forward_spv` - Compiled SPIR-V for forward pass
    /// * `backward_spv` - Compiled SPIR-V for backward pass
    /// * `reduce_spv` - Compiled SPIR-V for reduction pass
    /// * `input_size` - Number of input tensor elements
    /// * `output_size` - Number of output tensor elements
    /// * `batch_size` - Fixed batch size for this kernel
    ///
    /// # Returns
    ///
    /// Configured `GradientKernel` ready for execution
    pub fn new(
        device: Arc<Device>,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        forward_spv: &[u8],
        backward_spv: &[u8],
        reduce_spv: &[u8],
        input_size: u32,
        output_size: u32,
        batch_size: u32,
    ) -> Result<Self, VulkanErrorKind> {
        // Validate inputs
        debug_assert!(input_size > 0, "input_size must be positive");
        debug_assert!(output_size > 0, "output_size must be positive");
        debug_assert!(batch_size > 0, "batch_size must be positive");

        // Calculate dispatch dimensions
        let num_workgroups = (input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // --- Create Shader Modules ---
        let forward_shader = ShaderModule::new(
            &device,
            forward_spv,
            "main".to_string(),
        )?;
        let backward_shader = ShaderModule::new(
            &device,
            backward_spv,
            "main".to_string(),
        )?;
        let reduce_shader = ShaderModule::new(
            &device,
            reduce_spv,
            "main".to_string(),
        )?;

        // --- Create Descriptor Layouts ---

        // Forward pass bindings: input(0), output(1), activations(2), params(6), weight(8)
        let forward_bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(0)  // input_data (read)
            .add_storage_buffer(1)  // output_data (write)
            .add_storage_buffer(2)  // activations (write)
            .add_uniform_buffer(6)  // parameters
            .add_storage_buffer(8)  // weight (read)
            .build();

        // Backward pass bindings: input(0), activations(2), input_grad(3),
        //                         output_grad(4), workgroup_grads(5), params(6), weight(8)
        let backward_bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(0)  // input_data (read)
            .add_storage_buffer(2)  // activations (read)
            .add_storage_buffer(3)  // input_grad (write)
            .add_storage_buffer(4)  // output_grad (read)
            .add_storage_buffer(5)  // workgroup_grads (write)
            .add_uniform_buffer(6)  // parameters
            .add_storage_buffer(8)  // weight (read)
            .build();

        // Reduce pass bindings: workgroup_grads(5), param_grad(7)
        let reduce_bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(5)  // workgroup_grads (read)
            .add_storage_buffer(7)  // param_grad (write)
            .build();

        // --- Create Pipelines ---
        
        // Push constant size for backward and reduce
        let push_constant_size = std::mem::size_of::<GradientPushConstants>() as u32;

        let forward_pipeline = ComputePipeline::new(
            device.clone(),
            &forward_shader,
            &forward_bindings,
            0,  // No push constants for forward
        )?;

        let backward_pipeline = ComputePipeline::new(
            device.clone(),
            &backward_shader,
            &backward_bindings,
            push_constant_size,
        )?;

        let reduce_pipeline = ComputePipeline::new(
            device.clone(),
            &reduce_shader,
            &reduce_bindings,
            push_constant_size,
        )?;

        // --- Allocate Buffers ---

        let element_size = std::mem::size_of::<f32>() as u64;

        // Activation buffer: stores forward pass output for backward consumption
        // Size: input_size elements
        let activation_buffer = Buffer::new(
            device.clone(),
            input_size as u64 * element_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            memory_properties,
        )?;

        // Workgroup gradient staging: one slot per workgroup
        // Eliminates cross-workgroup atomics for determinism
        let workgroup_grad_buffer = Buffer::new(
            device.clone(),
            num_workgroups as u64 * element_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            memory_properties,
        )?;

        // Input gradient buffer: same size as input
        let input_grad_buffer = Buffer::new(
            device.clone(),
            input_size as u64 * element_size,
            vk::BufferUsageFlags::STORAGE_BUFFER 
                | vk::BufferUsageFlags::TRANSFER_SRC,  // May need to read back
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            memory_properties,
        )?;

        // Parameter gradient buffer: single value for v1 (scalar model)
        // HOST_VISIBLE for CPU readback
        let param_grad_buffer = Buffer::new(
            device.clone(),
            element_size,  // Single f32 for v1
            vk::BufferUsageFlags::STORAGE_BUFFER 
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL 
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            memory_properties,
        )?;

        // Parameters uniform buffer
        let params_buffer = Buffer::new(
            device.clone(),
            std::mem::size_of::<GradientParameters>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER 
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL 
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            memory_properties,
        )?;

        // Upload initial parameters
        let params = GradientParameters {
            input_size,
            output_size,
            batch_size,
            learning_rate: 0.001,  // Default, can be updated
        };
        params_buffer.upload_data(&[params])?;

        // Weight buffer: single f32, HOST_VISIBLE for CPU updates
        let weight_buffer = Buffer::new(
            device.clone(),
            element_size,  // Single f32
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            memory_properties,
        )?;

        // Initialize weight to 1.0
        weight_buffer.upload_data(&[1.0f32])?;

        // --- Allocate Descriptor Sets ---
        let forward_descriptor_set = forward_pipeline.allocate_descriptor_set()?;
        let backward_descriptor_set = backward_pipeline.allocate_descriptor_set()?;
        let reduce_descriptor_set = reduce_pipeline.allocate_descriptor_set()?;

        Ok(Self {
            forward_pipeline,
            backward_pipeline,
            reduce_pipeline,
            activation_buffer,
            workgroup_grad_buffer,
            input_grad_buffer,
            param_grad_buffer,
            params_buffer,
            weight_buffer,
            forward_descriptor_set,
            backward_descriptor_set,
            reduce_descriptor_set,
            input_size,
            output_size,
            batch_size,
            num_workgroups,
            device,
        })
    }

    /// Updates the learning rate parameter
    pub fn set_learning_rate(&self, lr: f32) -> Result<(), VulkanErrorKind> {
        let params = GradientParameters {
            input_size: self.input_size,
            output_size: self.output_size,
            batch_size: self.batch_size,
            learning_rate: lr,
        };
        self.params_buffer.upload_data(&[params])
    }

    /// Records forward pass commands into a command buffer
    ///
    /// # Arguments
    ///
    /// * `cmd_buffer` - Active command buffer
    /// * `input_buffer` - Input tensor (GPU buffer)
    /// * `output_buffer` - Output tensor (GPU buffer)
    ///
    /// # Side Effects
    ///
    /// Writes activations to internal `activation_buffer` for backward pass
    pub fn record_forward_pass(
        &self,
        cmd_buffer: vk::CommandBuffer,
        input_buffer: vk::Buffer,
        output_buffer: vk::Buffer,
    ) -> Result<(), VulkanErrorKind> {
        // Update descriptor set with current buffers
        self.forward_pipeline.update_descriptor_set(
            self.forward_descriptor_set,
            &[
                (0, input_buffer),
                (1, output_buffer),
                (2, self.activation_buffer.buffer),
                (6, self.params_buffer.buffer),
                (8, self.weight_buffer.buffer),
            ],
        )?;

        // Record dispatch
        self.forward_pipeline.record_dispatch(
            cmd_buffer,
            self.forward_descriptor_set,
            None,  // No push constants
            (self.num_workgroups, 1, 1),
        );

        // Memory barrier: forward writes → backward reads
        // Ensures activations are visible before backward pass
        self.record_compute_barrier(
            cmd_buffer,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );

        Ok(())
    }

    /// Records backward pass commands into a command buffer
    ///
    /// Must be called after `record_forward_pass` on the same input.
    ///
    /// # Arguments
    ///
    /// * `cmd_buffer` - Active command buffer
    /// * `input_buffer` - Same input buffer used in forward pass
    /// * `output_grad_buffer` - Gradient from loss/upstream layer
    pub fn record_backward_pass(
        &self,
        cmd_buffer: vk::CommandBuffer,
        input_buffer: vk::Buffer,
        output_grad_buffer: vk::Buffer,
    ) -> Result<(), VulkanErrorKind> {
        // Update descriptor set
        self.backward_pipeline.update_descriptor_set(
            self.backward_descriptor_set,
            &[
                (0, input_buffer),
                (2, self.activation_buffer.buffer),
                (3, self.input_grad_buffer.buffer),
                (4, output_grad_buffer),
                (5, self.workgroup_grad_buffer.buffer),
                (6, self.params_buffer.buffer),
                (8, self.weight_buffer.buffer),
            ],
        )?;

        // Push constants
        let push_constants = GradientPushConstants {
            num_workgroups: self.num_workgroups,
            param_size: 1,  // Single parameter for v1
        };

        // Convert push constants to bytes
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push_constants as *const GradientPushConstants as *const u8,
                std::mem::size_of::<GradientPushConstants>(),
            )
        };

        // Record backward dispatch
        self.backward_pipeline.record_dispatch(
            cmd_buffer,
            self.backward_descriptor_set,
            Some(push_bytes),
            (self.num_workgroups, 1, 1),
        );

        // Memory barrier: backward writes → reduce reads
        self.record_compute_barrier(
            cmd_buffer,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );

        Ok(())
    }

    /// Records reduction pass commands into a command buffer
    ///
    /// Must be called after `record_backward_pass`.
    /// Sums per-workgroup gradients in deterministic fixed order.
    pub fn record_reduce_pass(
        &self,
        cmd_buffer: vk::CommandBuffer,
    ) -> Result<(), VulkanErrorKind> {
        // Update descriptor set
        self.reduce_pipeline.update_descriptor_set(
            self.reduce_descriptor_set,
            &[
                (5, self.workgroup_grad_buffer.buffer),
                (7, self.param_grad_buffer.buffer),
            ],
        )?;

        // Push constants
        let push_constants = GradientPushConstants {
            num_workgroups: self.num_workgroups,
            param_size: 1,
        };

        // Convert push constants to bytes
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push_constants as *const GradientPushConstants as *const u8,
                std::mem::size_of::<GradientPushConstants>(),
            )
        };

        // Record reduce dispatch (single thread for determinism)
        self.reduce_pipeline.record_dispatch(
            cmd_buffer,
            self.reduce_descriptor_set,
            Some(push_bytes),
            (1, 1, 1),  // Single workgroup, single thread
        );

        // Memory barrier: reduce writes → host reads
        self.record_compute_barrier(
            cmd_buffer,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::HOST_READ,
        );

        Ok(())
    }

    /// Executes a complete forward + backward + reduce pass
    ///
    /// High-level API that handles command buffer lifecycle.
    ///
    /// # Arguments
    ///
    /// * `cmd_pool` - Command buffer pool for allocation
    /// * `input_buffer` - Input tensor
    /// * `output_buffer` - Output tensor
    /// * `output_grad_buffer` - Gradient from loss
    /// * `queue` - Vulkan queue for submission
    pub fn full_pass(
        &self,
        cmd_pool: &mut CommandBufferPool,
        input_buffer: vk::Buffer,
        output_buffer: vk::Buffer,
        output_grad_buffer: vk::Buffer,
        queue: vk::Queue,
    ) -> Result<(), VulkanErrorKind> {
        // Begin command buffer
        let cmd_buffer = cmd_pool.begin_command_buffer()?;

        // Record all passes
        self.record_forward_pass(cmd_buffer, input_buffer, output_buffer)?;
        self.record_backward_pass(cmd_buffer, input_buffer, output_grad_buffer)?;
        self.record_reduce_pass(cmd_buffer)?;

        // End and submit
        cmd_pool.end_command_buffer(cmd_buffer)?;
        cmd_pool.submit_and_wait(cmd_buffer, queue)?;

        Ok(())
    }

    /// Reads back the accumulated parameter gradient from GPU
    pub fn read_param_gradient(&self) -> Result<f32, VulkanErrorKind> {
        let mut result = [0.0f32; 1];
        self.param_grad_buffer.download_data(&mut result)?;
        Ok(result[0])
    }

    /// Reads the current weight value
    pub fn read_weight(&self) -> Result<f32, VulkanErrorKind> {
        let mut result = [0.0f32; 1];
        self.weight_buffer.download_data(&mut result)?;
        Ok(result[0])
    }

    /// Updates the weight value (for gradient descent)
    pub fn set_weight(&self, weight: f32) -> Result<(), VulkanErrorKind> {
        self.weight_buffer.upload_data(&[weight])
    }

    /// Applies gradient descent: weight -= learning_rate * gradient
    /// Returns the new weight value
    pub fn apply_gradient_update(&self, learning_rate: f32) -> Result<f32, VulkanErrorKind> {
        let gradient = self.read_param_gradient()?;
        let current_weight = self.read_weight()?;
        let new_weight = current_weight - learning_rate * gradient;
        self.set_weight(new_weight)?;
        Ok(new_weight)
    }

    /// Reads back input gradients from GPU
    pub fn read_input_gradients(&self) -> Result<Vec<f32>, VulkanErrorKind> {
        let mut result = vec![0.0f32; self.input_size as usize];
        self.input_grad_buffer.download_data(&mut result)?;
        Ok(result)
    }

    /// Returns the internal activation buffer (for testing/debugging)
    pub fn activation_buffer(&self) -> &Buffer {
        &self.activation_buffer
    }

    /// Returns configuration metadata
    pub fn config(&self) -> (u32, u32, u32, u32) {
        (self.input_size, self.output_size, self.batch_size, self.num_workgroups)
    }

    // --- Internal Helpers ---

    /// Records a compute pipeline barrier
    fn record_compute_barrier(
        &self,
        cmd_buffer: vk::CommandBuffer,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}

impl Drop for GradientKernel {
    fn drop(&mut self) {
        // Buffers and pipelines implement Drop via RAII
        // Nothing manual needed here - Arc<Device> handles cleanup order
    }
}

// =============================================================================
// VALIDATION MODE
// =============================================================================

#[cfg(feature = "validate_determinism")]
impl GradientKernel {
    /// Validates that gradient computation is deterministic
    ///
    /// Runs the full pass multiple times with identical input and verifies
    /// bit-identical output. Panics on non-deterministic execution.
    ///
    /// # Arguments
    ///
    /// * `cmd_pool` - Command buffer pool
    /// * `input_buffer` - Input tensor
    /// * `output_buffer` - Output tensor (will be overwritten)
    /// * `output_grad_buffer` - Gradient input
    /// * `queue` - Vulkan queue
    /// * `iterations` - Number of runs to compare (recommend 3-5)
    pub fn validate_determinism(
        &self,
        cmd_pool: &mut CommandBufferPool,
        input_buffer: vk::Buffer,
        output_buffer: vk::Buffer,
        output_grad_buffer: vk::Buffer,
        queue: vk::Queue,
        iterations: usize,
    ) -> Result<(), String> {
        let mut results: Vec<f32> = Vec::with_capacity(iterations);

        for i in 0..iterations {
            self.full_pass(
                cmd_pool,
                input_buffer,
                output_buffer,
                output_grad_buffer,
                queue,
            ).map_err(|e| format!("Iteration {} failed: {:?}", i, e))?;

            let param_grad = self.read_param_gradient()
                .map_err(|e| format!("Failed to read gradient: {:?}", e))?;

            results.push(param_grad);

            if i > 0 {
                // Bitwise comparison
                let prev_bits = results[i - 1].to_bits();
                let curr_bits = param_grad.to_bits();

                if prev_bits != curr_bits {
                    return Err(format!(
                        "Determinism violation at iteration {}: \
                         prev={} (0x{:08x}), curr={} (0x{:08x})",
                        i, results[i - 1], prev_bits, param_grad, curr_bits
                    ));
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that GradientPushConstants has correct size for Vulkan
    #[test]
    fn test_push_constant_layout() {
        assert_eq!(
            std::mem::size_of::<GradientPushConstants>(),
            8,  // 2 * u32
            "Push constants must be 8 bytes"
        );
    }

    /// Test that GradientParameters matches std140 layout
    #[test]
    fn test_parameters_layout() {
        assert_eq!(
            std::mem::size_of::<GradientParameters>(),
            16,  // 4 * 4 bytes
            "Parameters struct must be 16 bytes for std140"
        );
    }

    /// Test workgroup calculation
    #[test]
    fn test_workgroup_calculation() {
        // Exact multiple
        assert_eq!((256 + 255) / 256, 1);
        assert_eq!((512 + 255) / 256, 2);
        
        // Ceiling division
        assert_eq!((257 + 255) / 256, 2);
        assert_eq!((1 + 255) / 256, 1);
        assert_eq!((1000 + 255) / 256, 4);
    }

    // Integration tests require a Vulkan device and are marked #[ignore]
    // Run with: cargo test -- --ignored

    #[test]
    #[ignore]
    fn test_gradient_kernel_creation() {
        // Requires Vulkan device - placeholder for integration testing
        // Would create device, load shaders, instantiate GradientKernel
    }

    #[test]
    #[ignore]
    fn test_determinism_3_runs() {
        // Requires Vulkan device
        // Would run full_pass 3 times, compare results bitwise
    }

    #[test]
    #[ignore]
    fn test_gradient_correctness_vs_numeric() {
        // Requires Vulkan device
        // Would compare analytic gradient vs finite difference
    }
}
