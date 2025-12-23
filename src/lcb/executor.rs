//! LC-B Executor
//!
//! Executes LC-B instruction batches on the GPU.
//! Routes contract IDs to appropriate Vulkan compute kernels.

use std::sync::Arc;
use ash::{vk, Device};

use crate::lcb::parser::{LCBBatch, LCBInstruction, TensorData, DType, contracts};

/// GPU execution context
pub struct LCBExecutor {
    device: Arc<Device>,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    
    // Compute pipelines
    gemm_pipeline: vk::Pipeline,
    layernorm_pipeline: vk::Pipeline,
    gelu_pipeline: vk::Pipeline,
    softmax_pipeline: vk::Pipeline,
    cross_entropy_pipeline: vk::Pipeline,
    
    // Pipeline layout and descriptor resources
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

/// Result of executing an LC-B batch
pub struct ExecutionResult {
    pub outputs: Vec<TensorData>,
}

impl LCBExecutor {
    /// Create a new executor with initialized GPU resources
    pub fn new(
        device: Arc<Device>,
        compute_queue: vk::Queue,
        queue_family_index: u32,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        shader_dir: &std::path::Path,
    ) -> Result<Self, String> {
        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        
        let command_pool = unsafe {
            device.create_command_pool(&pool_info, None)
        }.map_err(|e| format!("Failed to create command pool: {:?}", e))?;
        
        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        
        let command_buffers = unsafe {
            device.allocate_command_buffers(&alloc_info)
        }.map_err(|e| format!("Failed to allocate command buffer: {:?}", e))?;
        
        let command_buffer = command_buffers[0];
        
        // Create fence
        let fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)
        }.map_err(|e| format!("Failed to create fence: {:?}", e))?;
        
        // Create descriptor set layout (8 storage buffers)
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..8)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&layout_info, None)
        }.map_err(|e| format!("Failed to create descriptor set layout: {:?}", e))?;
        
        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(32);
        
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
        }.map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?;
        
        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256),
        ];
        
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(32)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None)
        }.map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?;
        
        // Load and create pipelines
        let load_pipeline = |name: &str| -> Result<vk::Pipeline, String> {
            let path = shader_dir.join(format!("{}.spv", name));
            let spv = std::fs::read(&path)
                .map_err(|e| format!("Failed to load {}: {}", name, e))?;
            
            // Create shader module
            let code: Vec<u32> = spv.chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
            let shader_module = unsafe {
                device.create_shader_module(&module_info, None)
            }.map_err(|e| format!("Failed to create shader module: {:?}", e))?;
            
            // Create pipeline
            let entry_point = std::ffi::CString::new("main").unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_point);
            
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);
            
            let pipelines = unsafe {
                device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            }.map_err(|e| format!("Failed to create pipeline: {:?}", e.1))?;
            
            // Cleanup shader module (pipeline keeps reference)
            unsafe {
                device.destroy_shader_module(shader_module, None);
            }
            
            Ok(pipelines[0])
        };
        
        let gemm_pipeline = load_pipeline("gemm")?;
        let layernorm_pipeline = load_pipeline("layernorm_forward")?;
        let gelu_pipeline = load_pipeline("gelu_forward")?;
        let softmax_pipeline = load_pipeline("softmax_forward")?;
        let cross_entropy_pipeline = load_pipeline("cross_entropy_forward")?;
        
        Ok(Self {
            device,
            compute_queue,
            command_pool,
            command_buffer,
            fence,
            memory_properties,
            gemm_pipeline,
            layernorm_pipeline,
            gelu_pipeline,
            softmax_pipeline,
            cross_entropy_pipeline,
            pipeline_layout,
            descriptor_pool,
            descriptor_set_layout,
        })
    }
    
    /// Execute an LC-B batch and return results
    pub fn execute(&mut self, batch: &LCBBatch) -> Result<ExecutionResult, String> {
        let mut outputs = Vec::new();
        
        for instruction in &batch.instructions {
            let result = self.execute_instruction(instruction)?;
            outputs.extend(result);
        }
        
        Ok(ExecutionResult { outputs })
    }
    
    fn execute_instruction(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        match instr.contract_id {
            contracts::GEMM => self.execute_gemm(instr),
            contracts::LAYERNORM => self.execute_layernorm(instr),
            contracts::GELU => self.execute_gelu(instr),
            contracts::SOFTMAX => self.execute_softmax(instr),
            contracts::CROSS_ENTROPY => self.execute_cross_entropy(instr),
            _ => Err(format!("Unknown contract ID: {}", instr.contract_id)),
        }
    }
    
    /// Execute GEMM: C = A @ B
    fn execute_gemm(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        if instr.tensors.len() != 2 {
            return Err(format!("GEMM expects 2 tensors, got {}", instr.tensors.len()));
        }
        
        let a = &instr.tensors[0];
        let b = &instr.tensors[1];
        
        // Validate shapes
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err("GEMM requires 2D tensors".to_string());
        }
        
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];
        
        if b.shape[0] != k {
            return Err(format!(
                "GEMM dimension mismatch: A is {}x{}, B is {}x{}",
                m, k, b.shape[0], b.shape[1]
            ));
        }
        
        // Allocate output buffer
        let output_size = m * n;
        let mut output_data = vec![0.0f32; output_size];
        
        // Create GPU buffers
        let a_buffer = self.create_buffer_with_data(a)?;
        let b_buffer = self.create_buffer_with_data(b)?;
        let c_buffer = self.create_buffer((output_size * 4) as u64)?;
        
        // Allocate descriptor set
        let desc_set = self.allocate_descriptor_set()?;
        
        // Update descriptor set
        self.update_descriptor_set(desc_set, &[
            (0, a_buffer.0, (a.data.len()) as u64),
            (1, b_buffer.0, (b.data.len()) as u64),
            (2, c_buffer.0, (output_size * 4) as u64),
        ]);
        
        // Push constants
        #[repr(C)]
        struct GemmPush {
            m: u32,
            k: u32,
            n: u32,
            use_bias: u32,
        }
        
        let push = GemmPush {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            use_bias: 0,
        };
        
        // Record and execute
        self.execute_pipeline(
            self.gemm_pipeline,
            desc_set,
            &push,
            ((n + 15) / 16) as u32,
            ((m + 15) / 16) as u32,
            1,
        )?;
        
        // Download result
        self.download_buffer(c_buffer.1, &mut output_data)?;
        
        // Cleanup
        self.destroy_buffer(a_buffer);
        self.destroy_buffer(b_buffer);
        self.destroy_buffer(c_buffer);
        self.free_descriptor_set(desc_set)?;
        
        Ok(vec![TensorData::from_f32(&output_data, vec![m, n])])
    }
    
    /// Execute LayerNorm
    fn execute_layernorm(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        if instr.tensors.is_empty() {
            return Err("LayerNorm requires at least 1 tensor".to_string());
        }
        
        let x = &instr.tensors[0];
        let eps = instr.scalars.get("eps").copied().unwrap_or(1e-5);
        
        if x.shape.len() != 2 {
            return Err("LayerNorm expects 2D tensor [batch, features]".to_string());
        }
        
        let num_rows = x.shape[0];
        let row_size = x.shape[1];
        let total_size = num_rows * row_size;
        
        // Allocate output
        let mut output_data = vec![0.0f32; total_size];
        
        // Create buffers
        let x_buffer = self.create_buffer_with_data(x)?;
        let y_buffer = self.create_buffer((total_size * 4) as u64)?;
        let stats_buffer = self.create_buffer((num_rows * 2 * 4) as u64)?; // mean + inv_std

        // Gamma and beta (ones and zeros if not provided)
        let gamma = if instr.tensors.len() > 1 {
            instr.tensors[1].clone()
        } else {
            TensorData::from_f32(&vec![1.0f32; row_size], vec![row_size])
        };

        let beta = if instr.tensors.len() > 2 {
            instr.tensors[2].clone()
        } else {
            TensorData::from_f32(&vec![0.0f32; row_size], vec![row_size])
        };

        let gamma_buffer = self.create_buffer_with_data(&gamma)?;
        let beta_buffer = self.create_buffer_with_data(&beta)?;

        // Descriptor set - MATCH SHADER BINDING ORDER!
        // binding 0: input, 1: output, 2: gamma, 3: beta, 4: stats
        let desc_set = self.allocate_descriptor_set()?;
        self.update_descriptor_set(desc_set, &[
            (0, x_buffer.0, (x.data.len()) as u64),
            (1, y_buffer.0, (total_size * 4) as u64),
            (2, gamma_buffer.0, (gamma.data.len()) as u64),
            (3, beta_buffer.0, (beta.data.len()) as u64),
            (4, stats_buffer.0, (num_rows * 2 * 4) as u64),
        ]);
        
        // Push constants
        #[repr(C)]
        struct LayerNormPush {
            num_rows: u32,
            row_size: u32,
            eps: f32,
            _pad: u32,
        }
        
        let push = LayerNormPush {
            num_rows: num_rows as u32,
            row_size: row_size as u32,
            eps,
            _pad: 0,
        };
        
        self.execute_pipeline(
            self.layernorm_pipeline,
            desc_set,
            &push,
            num_rows as u32,
            1,
            1,
        )?;
        
        // Download result
        self.download_buffer(y_buffer.1, &mut output_data)?;
        
        // Cleanup
        self.destroy_buffer(x_buffer);
        self.destroy_buffer(y_buffer);
        self.destroy_buffer(stats_buffer);
        self.destroy_buffer(gamma_buffer);
        self.destroy_buffer(beta_buffer);
        self.free_descriptor_set(desc_set)?;
        
        Ok(vec![TensorData::from_f32(&output_data, x.shape.clone())])
    }
    
    /// Execute GELU activation
    fn execute_gelu(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        if instr.tensors.is_empty() {
            return Err("GELU requires 1 tensor".to_string());
        }
        
        let x = &instr.tensors[0];
        let num_elements = x.num_elements();
        let mut output_data = vec![0.0f32; num_elements];
        
        let x_buffer = self.create_buffer_with_data(x)?;
        let y_buffer = self.create_buffer((num_elements * 4) as u64)?;
        
        let desc_set = self.allocate_descriptor_set()?;
        self.update_descriptor_set(desc_set, &[
            (0, x_buffer.0, x.data.len() as u64),
            (1, y_buffer.0, (num_elements * 4) as u64),
        ]);
        
        #[repr(C)]
        struct GeluPush {
            num_elements: u32,
        }
        
        let push = GeluPush { num_elements: num_elements as u32 };
        
        self.execute_pipeline(
            self.gelu_pipeline,
            desc_set,
            &push,
            ((num_elements + 255) / 256) as u32,
            1,
            1,
        )?;
        
        self.download_buffer(y_buffer.1, &mut output_data)?;
        
        self.destroy_buffer(x_buffer);
        self.destroy_buffer(y_buffer);
        self.free_descriptor_set(desc_set)?;
        
        Ok(vec![TensorData::from_f32(&output_data, x.shape.clone())])
    }
    
    /// Execute Softmax
    fn execute_softmax(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        if instr.tensors.is_empty() {
            return Err("Softmax requires 1 tensor".to_string());
        }
        
        let x = &instr.tensors[0];
        
        if x.shape.len() != 2 {
            return Err("Softmax expects 2D tensor".to_string());
        }
        
        let num_rows = x.shape[0];
        let row_size = x.shape[1];
        let total_size = num_rows * row_size;
        let mut output_data = vec![0.0f32; total_size];
        
        let x_buffer = self.create_buffer_with_data(x)?;
        let y_buffer = self.create_buffer((total_size * 4) as u64)?;
        
        let desc_set = self.allocate_descriptor_set()?;
        self.update_descriptor_set(desc_set, &[
            (0, x_buffer.0, x.data.len() as u64),
            (1, y_buffer.0, (total_size * 4) as u64),
        ]);
        
        #[repr(C)]
        struct SoftmaxPush {
            num_rows: u32,
            row_size: u32,
        }
        
        let push = SoftmaxPush {
            num_rows: num_rows as u32,
            row_size: row_size as u32,
        };
        
        self.execute_pipeline(
            self.softmax_pipeline,
            desc_set,
            &push,
            num_rows as u32,
            1,
            1,
        )?;
        
        self.download_buffer(y_buffer.1, &mut output_data)?;
        
        self.destroy_buffer(x_buffer);
        self.destroy_buffer(y_buffer);
        self.free_descriptor_set(desc_set)?;
        
        Ok(vec![TensorData::from_f32(&output_data, x.shape.clone())])
    }
    
    /// Execute Cross-Entropy Loss
    fn execute_cross_entropy(&mut self, instr: &LCBInstruction) -> Result<Vec<TensorData>, String> {
        if instr.tensors.len() < 2 {
            return Err("CrossEntropy requires 2 tensors (logits, targets)".to_string());
        }
        
        let logits = &instr.tensors[0];
        let targets = &instr.tensors[1];
        
        if logits.shape.len() != 2 {
            return Err("CrossEntropy logits must be 2D [batch, vocab]".to_string());
        }
        
        let num_positions = logits.shape[0];
        let vocab_size = logits.shape[1];
        
        // Output: per-position losses
        let mut losses = vec![0.0f32; num_positions];
        let mut softmax_out = vec![0.0f32; num_positions * vocab_size];
        
        let logits_buffer = self.create_buffer_with_data(logits)?;
        let targets_buffer = self.create_buffer_with_data(targets)?;
        let losses_buffer = self.create_buffer((num_positions * 4) as u64)?;
        let softmax_buffer = self.create_buffer((num_positions * vocab_size * 4) as u64)?;
        
        let desc_set = self.allocate_descriptor_set()?;
        self.update_descriptor_set(desc_set, &[
            (0, logits_buffer.0, logits.data.len() as u64),
            (1, targets_buffer.0, targets.data.len() as u64),
            (2, losses_buffer.0, (num_positions * 4) as u64),
            (3, softmax_buffer.0, (num_positions * vocab_size * 4) as u64),
        ]);
        
        #[repr(C)]
        struct CEPush {
            num_positions: u32,
            vocab_size: u32,
            ignore_index: u32,
        }

        // Note: negative f32 values saturate to 0 when cast to u32 in Rust
        // So we treat any negative ignore_index as "ignore nothing" (0xFFFFFFFF)
        let ignore_index = instr.scalars.get("ignore_index")
            .map(|&v| if v < 0.0 { 0xFFFFFFFF } else { v as u32 })
            .unwrap_or(0xFFFFFFFF);  // Default: ignore nothing

        let push = CEPush {
            num_positions: num_positions as u32,
            vocab_size: vocab_size as u32,
            ignore_index,
        };
        
        self.execute_pipeline(
            self.cross_entropy_pipeline,
            desc_set,
            &push,
            num_positions as u32,
            1,
            1,
        )?;
        
        self.download_buffer(losses_buffer.1, &mut losses)?;
        self.download_buffer(softmax_buffer.1, &mut softmax_out)?;

        // Compute mean loss
        let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;

        self.destroy_buffer(logits_buffer);
        self.destroy_buffer(targets_buffer);
        self.destroy_buffer(losses_buffer);
        self.destroy_buffer(softmax_buffer);
        self.free_descriptor_set(desc_set)?;

        // Return mean loss as scalar
        Ok(vec![
            TensorData::from_f32(&[mean_loss], vec![1]),
        ])
    }
    
    // =========================================================================
    // Helper methods
    // =========================================================================
    
    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, String> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0 
                && self.memory_properties.memory_types[i as usize].property_flags.contains(properties) {
                return Ok(i);
            }
        }
        Err("No suitable memory type".to_string())
    }
    
    fn create_buffer(&self, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None)
        }.map_err(|e| format!("Create buffer failed: {:?}", e))?;
        
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type);
        
        let memory = unsafe {
            self.device.allocate_memory(&alloc_info, None)
        }.map_err(|e| format!("Allocate memory failed: {:?}", e))?;
        
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0)
        }.map_err(|e| format!("Bind memory failed: {:?}", e))?;
        
        Ok((buffer, memory))
    }
    
    fn create_buffer_with_data(&self, tensor: &TensorData) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let (buffer, memory) = self.create_buffer(tensor.data.len() as u64)?;
        
        unsafe {
            let ptr = self.device.map_memory(memory, 0, tensor.data.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Map memory failed: {:?}", e))?;
            std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), ptr as *mut u8, tensor.data.len());
            self.device.unmap_memory(memory);
        }
        
        Ok((buffer, memory))
    }
    
    fn download_buffer(&self, memory: vk::DeviceMemory, data: &mut [f32]) -> Result<(), String> {
        let size = (data.len() * 4) as u64;
        unsafe {
            let ptr = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Map memory failed: {:?}", e))?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, data.as_mut_ptr(), data.len());
            self.device.unmap_memory(memory);
        }
        Ok(())
    }
    
    fn destroy_buffer(&self, buffer: (vk::Buffer, vk::DeviceMemory)) {
        unsafe {
            self.device.destroy_buffer(buffer.0, None);
            self.device.free_memory(buffer.1, None);
        }
    }
    
    fn allocate_descriptor_set(&self) -> Result<vk::DescriptorSet, String> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        
        let sets = unsafe {
            self.device.allocate_descriptor_sets(&alloc_info)
        }.map_err(|e| format!("Allocate descriptor set failed: {:?}", e))?;
        
        Ok(sets[0])
    }
    
    fn free_descriptor_set(&self, set: vk::DescriptorSet) -> Result<(), String> {
        unsafe {
            self.device.free_descriptor_sets(self.descriptor_pool, &[set])
        }.map_err(|e| format!("Free descriptor set failed: {:?}", e))
    }
    
    fn update_descriptor_set(&self, set: vk::DescriptorSet, buffers: &[(u32, vk::Buffer, u64)]) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers.iter()
            .map(|(_, buf, size)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buf)
                    .offset(0)
                    .range(*size)
            })
            .collect();
        
        let writes: Vec<vk::WriteDescriptorSet> = buffers.iter()
            .zip(buffer_infos.iter())
            .map(|((binding, _, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }
    
    fn execute_pipeline<T>(
        &mut self,
        pipeline: vk::Pipeline,
        descriptor_set: vk::DescriptorSet,
        push_constants: &T,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), String> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        
        unsafe {
            self.device.begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| format!("Begin command buffer failed: {:?}", e))?;
            
            self.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            
            let push_bytes = std::slice::from_raw_parts(
                push_constants as *const T as *const u8,
                std::mem::size_of::<T>(),
            );
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            
            self.device.cmd_dispatch(self.command_buffer, groups_x, groups_y, groups_z);
            
            self.device.end_command_buffer(self.command_buffer)
                .map_err(|e| format!("End command buffer failed: {:?}", e))?;
            
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)
                .map_err(|e| format!("Queue submit failed: {:?}", e))?;
            
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("Wait for fences failed: {:?}", e))?;
            
            self.device.reset_fences(&[self.fence])
                .map_err(|e| format!("Reset fences failed: {:?}", e))?;
            
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("Reset command buffer failed: {:?}", e))?;
        }
        
        Ok(())
    }
}

impl Drop for LCBExecutor {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            
            self.device.destroy_pipeline(self.gemm_pipeline, None);
            self.device.destroy_pipeline(self.layernorm_pipeline, None);
            self.device.destroy_pipeline(self.gelu_pipeline, None);
            self.device.destroy_pipeline(self.softmax_pipeline, None);
            self.device.destroy_pipeline(self.cross_entropy_pipeline, None);
            
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_data_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(&data, vec![2, 2]);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
