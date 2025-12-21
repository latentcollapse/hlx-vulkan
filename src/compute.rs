//! Vulkan compute pipeline infrastructure for ML workloads
//!
//! Provides compute-specific abstractions separate from graphics pipelines.
//! Designed for deterministic ML training with explicit memory barriers,
//! descriptor management, and command buffer pooling.
//!
//! # Architecture
//!
//! - `ComputePipeline`: Wraps VkPipeline (compute), descriptor sets, and layout
//! - `DescriptorPool`: Manages descriptor allocation for tensor buffers
//! - `ComputeDispatch`: Encapsulates a single compute kernel dispatch with barriers
//!
//! # Example
//!
//! ```rust,no_run
//! use hlx_vulkan::compute::ComputePipeline;
//!
//! // Create compute pipeline from SPIR-V shader
//! let pipeline = ComputePipeline::new(
//!     device.clone(),
//!     &shader_module,
//!     &descriptor_set_layouts,
//!     push_constant_size,
//! )?;
//!
//! // Dispatch compute kernel
//! pipeline.dispatch(
//!     command_buffer,
//!     &descriptor_sets,
//!     &push_constants,
//!     (groups_x, groups_y, groups_z),
//! )?;
//! ```

use ash::{vk, Device};
use std::sync::Arc;
use std::ffi::CString;

use crate::error::VulkanErrorKind;
use crate::shader::ShaderModule;

/// Command buffer pool for efficient reuse
///
/// Maintains a pool of command buffers that can be reset and reused
/// across multiple compute dispatches. Reduces allocation overhead.
pub struct CommandBufferPool {
    /// Vulkan command pool handle
    command_pool: vk::CommandPool,

    /// Pre-allocated command buffers
    command_buffers: Vec<vk::CommandBuffer>,

    /// Index of next available command buffer
    next_buffer_index: usize,

    /// Reference to device
    device: Arc<Device>,
}

impl CommandBufferPool {
    /// Create a new command buffer pool.
    ///
    /// # Arguments
    ///
    /// * `device` - Logical device
    /// * `queue_family` - Queue family index for compute operations
    /// * `initial_count` - Number of command buffers to pre-allocate
    pub fn new(
        device: Arc<Device>,
        queue_family: u32,
        initial_count: u32,
    ) -> Result<Self, VulkanErrorKind> {
        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER); // Allow individual buffer reset

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create command pool: {:?}",
                    e
                ))
            })?;

        // Allocate initial command buffers
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(initial_count);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to allocate command buffers: {:?}",
                    e
                ))
            })?;

        log::info!(
            "Created CommandBufferPool: {} buffers, queue_family={}",
            initial_count,
            queue_family
        );

        Ok(Self {
            command_pool,
            command_buffers,
            next_buffer_index: 0,
            device,
        })
    }

    /// Get a command buffer and begin recording.
    ///
    /// Returns a command buffer ready for recording commands.
    /// Automatically cycles through the pool (round-robin).
    pub fn begin_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanErrorKind> {
        let cmd_buffer = self.command_buffers[self.next_buffer_index];

        // Reset command buffer for reuse
        unsafe {
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| {
                    VulkanErrorKind::InitializationFailed(format!(
                        "Failed to reset command buffer: {:?}",
                        e
                    ))
                })?;
        }

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(|e| {
                    VulkanErrorKind::InitializationFailed(format!(
                        "Failed to begin command buffer: {:?}",
                        e
                    ))
                })?;
        }

        // Advance to next buffer (round-robin)
        self.next_buffer_index = (self.next_buffer_index + 1) % self.command_buffers.len();

        log::debug!("Command buffer recording started");
        Ok(cmd_buffer)
    }

    /// End command buffer recording.
    pub fn end_command_buffer(&self, cmd_buffer: vk::CommandBuffer) -> Result<(), VulkanErrorKind> {
        unsafe {
            self.device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| {
                    VulkanErrorKind::InitializationFailed(format!(
                        "Failed to end command buffer: {:?}",
                        e
                    ))
                })?;
        }

        log::debug!("Command buffer recording ended");
        Ok(())
    }

    /// Submit command buffer to queue and wait for completion.
    ///
    /// This is a blocking submit - useful for testing and debugging.
    /// Production code should use async submission with fences.
    pub fn submit_and_wait(
        &self,
        cmd_buffer: vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Result<(), VulkanErrorKind> {
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd_buffer));

        unsafe {
            self.device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .map_err(|e| {
                    VulkanErrorKind::InitializationFailed(format!(
                        "Failed to submit command buffer: {:?}",
                        e
                    ))
                })?;

            // Wait for queue to finish (blocking)
            self.device
                .queue_wait_idle(queue)
                .map_err(|e| {
                    VulkanErrorKind::InitializationFailed(format!(
                        "Failed to wait for queue: {:?}",
                        e
                    ))
                })?;
        }

        log::debug!("Command buffer submitted and executed");
        Ok(())
    }

}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        log::debug!("Dropping CommandBufferPool");
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Compute pipeline with descriptor management
///
/// Wraps VkPipeline (compute type), VkPipelineLayout, and descriptor sets.
/// Designed for ML workloads with explicit synchronization and deterministic execution.
pub struct ComputePipeline {
    /// Vulkan compute pipeline handle
    pub pipeline: vk::Pipeline,

    /// Pipeline layout (descriptor set layouts + push constants)
    pub layout: vk::PipelineLayout,

    /// Descriptor set layouts (for binding input/output buffers)
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,

    /// Descriptor pool (for allocating descriptor sets)
    pub descriptor_pool: vk::DescriptorPool,

    /// Binding type map: binding index → descriptor type
    /// Used by update_descriptor_set to use the correct type
    binding_types: std::collections::HashMap<u32, vk::DescriptorType>,

    /// Reference to device (needed for cleanup)
    device: Arc<Device>,
}

impl ComputePipeline {
    /// Create a new compute pipeline from a compute shader.
    ///
    /// # Arguments
    ///
    /// * `device` - Logical device (Arc-wrapped for safe sharing)
    /// * `shader` - Compiled SPIR-V compute shader module
    /// * `descriptor_bindings` - Array of descriptor bindings (e.g., buffer bindings)
    /// * `push_constant_size` - Size of push constants in bytes (0 if none)
    ///
    /// # Returns
    ///
    /// ComputePipeline ready for dispatch operations
    ///
    /// # Errors
    ///
    /// Returns `VulkanErrorKind::InitializationFailed` if pipeline creation fails.
    pub fn new(
        device: Arc<Device>,
        shader: &ShaderModule,
        descriptor_bindings: &[vk::DescriptorSetLayoutBinding],
        push_constant_size: u32,
    ) -> Result<Self, VulkanErrorKind> {
        log::info!(
            "Creating ComputePipeline: entry={}, bindings={}, push_constants={} bytes",
            shader.entry_point,
            descriptor_bindings.len(),
            push_constant_size
        );

        // Create descriptor set layout from bindings
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device, descriptor_bindings)?;

        // Create pipeline layout (descriptor sets + push constants)
        let pipeline_layout = Self::create_pipeline_layout(
            &device,
            &[descriptor_set_layout],
            push_constant_size,
        )?;

        // Create descriptor pool (for allocating descriptor sets)
        let descriptor_pool = Self::create_descriptor_pool(&device, descriptor_bindings)?;

        // Build binding type map
        let binding_types: std::collections::HashMap<u32, vk::DescriptorType> = descriptor_bindings
            .iter()
            .map(|b| (b.binding, b.descriptor_type))
            .collect();

        // Create compute pipeline
        let pipeline = Self::create_compute_pipeline(&device, shader, pipeline_layout)?;

        log::info!("ComputePipeline created successfully");

        Ok(Self {
            pipeline,
            layout: pipeline_layout,
            descriptor_set_layouts: vec![descriptor_set_layout],
            descriptor_pool,
            binding_types,
            device,
        })
    }

    /// Create a descriptor set layout from bindings.
    ///
    /// Descriptor bindings specify how buffers (tensors) are bound to shader.
    fn create_descriptor_set_layout(
        device: &Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::DescriptorSetLayout, VulkanErrorKind> {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create descriptor set layout: {:?}",
                    e
                ))
            })?;

        log::debug!("Created descriptor set layout with {} bindings", bindings.len());
        Ok(layout)
    }

    /// Create pipeline layout (descriptor sets + push constants).
    ///
    /// Pipeline layout defines the interface between CPU and GPU:
    /// - Descriptor sets: Large data (buffers, images)
    /// - Push constants: Small, frequently-changing parameters
    fn create_pipeline_layout(
        device: &Device,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_size: u32,
    ) -> Result<vk::PipelineLayout, VulkanErrorKind> {
        let push_constant_ranges = if push_constant_size > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_size)]
        } else {
            vec![]
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create pipeline layout: {:?}",
                    e
                ))
            })?;

        log::debug!(
            "Created pipeline layout: {} descriptor sets, {} push constant bytes",
            set_layouts.len(),
            push_constant_size
        );
        Ok(layout)
    }

    /// Create descriptor pool for allocating descriptor sets.
    ///
    /// Pool size is based on the number of bindings and expected usage.
    /// For ML workloads, we need both storage buffers (tensors) and uniform buffers (params).
    fn create_descriptor_pool(
        device: &Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::DescriptorPool, VulkanErrorKind> {
        // Count descriptor types from bindings
        let mut storage_count = 0u32;
        let mut uniform_count = 0u32;

        for binding in bindings {
            match binding.descriptor_type {
                vk::DescriptorType::STORAGE_BUFFER => storage_count += binding.descriptor_count,
                vk::DescriptorType::UNIFORM_BUFFER => uniform_count += binding.descriptor_count,
                _ => {}
            }
        }

        // Build pool sizes (only include types we actually need)
        let mut pool_sizes = Vec::new();

        if storage_count > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: storage_count.max(16),
            });
        }

        if uniform_count > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: uniform_count.max(4),
            });
        }

        // Fallback if no bindings
        if pool_sizes.is_empty() {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 16,
            });
        }

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(16);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create descriptor pool: {:?}",
                    e
                ))
            })?;

        log::debug!(
            "Created descriptor pool: {} storage buffers, {} uniform buffers",
            storage_count,
            uniform_count
        );
        Ok(pool)
    }

    /// Create the actual compute pipeline from shader module.
    ///
    /// This is the core compilation step that produces executable GPU code.
    fn create_compute_pipeline(
        device: &Device,
        shader: &ShaderModule,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, VulkanErrorKind> {
        // Convert entry point to CString (required by Vulkan API)
        let entry_point_cstr = CString::new(shader.entry_point.as_str())
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Invalid entry point name: {}",
                    e
                ))
            })?;

        // Compute shader stage
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.handle)
            .name(&entry_point_cstr);

        // Compute pipeline creation
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(), // No cache for now (add later for optimization)
                &[pipeline_info],
                None,
            )
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to create compute pipeline: {:?}",
                e.1 // .1 is the error, .0 is the partial result
            ))
        })?;

        log::debug!("Created compute pipeline");
        Ok(pipeline[0])
    }

    /// Allocate a descriptor set from the pool.
    ///
    /// Descriptor sets bind buffers (tensors) to shader bindings.
    /// This must be called before dispatch to bind input/output buffers.
    pub fn allocate_descriptor_set(&self) -> Result<vk::DescriptorSet, VulkanErrorKind> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&self.descriptor_set_layouts);

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to allocate descriptor set: {:?}",
                    e
                ))
            })?;

        Ok(descriptor_sets[0])
    }

    /// Update descriptor set with buffer bindings.
    ///
    /// Binds actual GPU buffers to the descriptor set for shader access.
    /// Automatically uses the correct descriptor type (STORAGE_BUFFER or UNIFORM_BUFFER)
    /// based on the pipeline's binding configuration.
    ///
    /// # Arguments
    ///
    /// * `descriptor_set` - The descriptor set to update
    /// * `bindings` - Array of (binding_index, buffer) pairs
    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        bindings: &[(u32, vk::Buffer)],
    ) -> Result<(), VulkanErrorKind> {
        // First, create all buffer infos
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|(_, buffer)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        // Then create write descriptor sets referencing the buffer infos
        // Use the correct descriptor type from the binding_types map
        let write_descriptor_sets: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .enumerate()
            .map(|(idx, (binding, _))| {
                // Look up the correct descriptor type, default to STORAGE_BUFFER
                let descriptor_type = self
                    .binding_types
                    .get(binding)
                    .copied()
                    .unwrap_or(vk::DescriptorType::STORAGE_BUFFER);

                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(descriptor_type)
                    .buffer_info(std::slice::from_ref(&buffer_infos[idx]))
            })
            .collect();

        unsafe {
            self.device.update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        log::debug!("Updated descriptor set with {} buffer bindings", bindings.len());
        Ok(())
    }

    /// Record a compute dispatch command.
    ///
    /// This records the actual kernel execution into a command buffer.
    /// Must be called within an active command buffer recording.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - Command buffer to record into
    /// * `descriptor_set` - Descriptor set with bound buffers
    /// * `push_constants` - Optional push constant data (small parameters)
    /// * `group_count` - Workgroup dispatch dimensions (x, y, z)
    pub fn record_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        push_constants: Option<&[u8]>,
        group_count: (u32, u32, u32),
    ) {
        unsafe {
            // Bind compute pipeline
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            // Bind descriptor sets (buffers)
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0, // first_set
                &[descriptor_set],
                &[], // dynamic offsets
            );

            // Push constants (if any)
            if let Some(constants) = push_constants {
                self.device.cmd_push_constants(
                    command_buffer,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0, // offset
                    constants,
                );
            }

            // Dispatch compute workgroups
            self.device.cmd_dispatch(
                command_buffer,
                group_count.0,
                group_count.1,
                group_count.2,
            );
        }

        log::debug!(
            "Recorded compute dispatch: workgroups=({}, {}, {})",
            group_count.0,
            group_count.1,
            group_count.2
        );
    }

    /// Insert a memory barrier for deterministic execution ordering.
    ///
    /// Critical for ML training: ensures gradient computations happen in
    /// deterministic order without race conditions.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - Command buffer to insert barrier into
    /// * `src_access` - Source access mask (what previous operations did)
    /// * `dst_access` - Destination access mask (what next operations need)
    pub fn record_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER, // src_stage
                vk::PipelineStageFlags::COMPUTE_SHADER, // dst_stage
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[], // buffer barriers
                &[], // image barriers
            );
        }

        log::debug!("Recorded memory barrier for deterministic ordering");
    }

}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        log::debug!("Dropping ComputePipeline");
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            for layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(*layout, None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

/// Builder for common descriptor binding patterns in ML workloads
pub struct DescriptorBindingBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

impl DescriptorBindingBuilder {
    /// Create a new descriptor binding builder
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Add a storage buffer binding (for tensors)
    ///
    /// Storage buffers support both read and write access from compute shaders.
    /// Used for input tensors, output tensors, and gradient accumulation.
    pub fn add_storage_buffer(mut self, binding: u32) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        self
    }

    /// Add a uniform buffer binding (for read-only parameters)
    ///
    /// Uniform buffers are read-only and cached differently.
    /// Use for model weights, hyperparameters, etc.
    pub fn add_uniform_buffer(mut self, binding: u32) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        self
    }

    /// Build the final binding array
    pub fn build(self) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
        self.bindings
    }
}

impl Default for DescriptorBindingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_binding_builder() {
        let bindings = DescriptorBindingBuilder::new()
            .add_storage_buffer(0) // Input tensor
            .add_storage_buffer(1) // Output tensor
            .add_uniform_buffer(2) // Parameters
            .build();

        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[1].binding, 1);
        assert_eq!(bindings[2].binding, 2);
        assert_eq!(bindings[0].descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(bindings[2].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    }
}
