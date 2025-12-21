//! Vulkan-based training harness for HLX models
//!
//! Demonstrates end-to-end training using:
//! - GradientKernel for forward/backward/reduce passes
//! - TensorPool for memory management
//! - Deterministic execution for reproducible results
//!
//! This is a simplified training loop focused on validation against
//! the CUDA baseline (0.0131 loss on ASCII specialist).

use ash::{vk, Entry, Instance, Device};
use std::sync::Arc;
use std::ffi::CString;
use hlx_vulkan::{
    GradientKernel, CommandBufferPool, Buffer, TensorPool,
};

// =============================================================================
// TRAINING CONFIGURATION
// =============================================================================

/// Training hyperparameters
struct TrainingConfig {
    /// Number of training epochs
    epochs: u32,
    /// Learning rate
    learning_rate: f32,
    /// Batch size (fixed for this model)
    batch_size: u32,
    /// Input tensor size
    input_size: u32,
    /// Early stopping patience (epochs without improvement)
    patience: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 1,
            input_size: 256,
            patience: 10,
        }
    }
}

// =============================================================================
// VULKAN CONTEXT
// =============================================================================

/// Minimal Vulkan context for training
struct TrainingContext {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl TrainingContext {
    /// Initialize Vulkan context
    fn new() -> Result<Self, String> {
        println!("Initializing Vulkan...");

        let entry = unsafe { Entry::load() }
            .map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;

        // Create instance
        let app_name = CString::new("HLX Vulkan Training").unwrap();
        let engine_name = CString::new("HLX").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("Failed to create instance: {:?}", e))?;

        // Select physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate devices: {:?}", e))?;

        if physical_devices.is_empty() {
            return Err("No Vulkan devices found".to_string());
        }

        let physical_device = physical_devices[0];

        // Get device properties
        let device_props = unsafe {
            instance.get_physical_device_properties(physical_device)
        };
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(device_props.device_name.as_ptr())
                .to_string_lossy()
        };

        println!("Using GPU: {}", device_name);

        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        // Find compute queue family
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let compute_queue_family = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| "No compute queue family found".to_string())?;

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }
        .map_err(|e| format!("Failed to create device: {:?}", e))?;

        let device = Arc::new(device);

        let compute_queue = unsafe {
            device.get_device_queue(compute_queue_family, 0)
        };

        println!("Vulkan initialized successfully\n");

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            memory_properties,
        })
    }
}

impl Drop for TrainingContext {
    fn drop(&mut self) {
        unsafe {
            // Wait for all GPU operations to complete
            if let Err(e) = self.device.device_wait_idle() {
                log::error!("Failed to wait for device idle: {:?}", e);
            }

            // Check if we're the last holder of the device
            let strong_count = Arc::strong_count(&self.device);
            if strong_count > 1 {
                // Other resources still hold the device - this shouldn't happen
                // if Trainer field order is correct (ctx drops last)
                log::error!(
                    "Cannot destroy device: {} references still held. Leaking to avoid crash.",
                    strong_count
                );
                return;
            }

            // We're the last holder - safe to destroy
            log::info!("Destroying Vulkan device and instance");

            // Destroy device first (before instance)
            self.device.destroy_device(None);

            // Then destroy instance
            self.instance.destroy_instance(None);
        }
    }
}

// =============================================================================
// TRAINING HARNESS
// =============================================================================

/// Main training harness
///
/// IMPORTANT: Field order matters for Drop! Resources must be destroyed
/// before the Vulkan context that owns them. Fields drop in declaration order.
struct Trainer {
    // GPU resources (drop first - release device references)
    gradient_kernel: GradientKernel,
    cmd_pool: CommandBufferPool,
    tensor_pool: TensorPool,

    // Configuration (no cleanup needed)
    config: TrainingConfig,

    // Vulkan context (drop LAST - destroys device/instance)
    ctx: TrainingContext,
}

impl Trainer {
    /// Create a new trainer
    fn new(config: TrainingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = TrainingContext::new()?;

        println!("Loading shaders...");

        // Load compiled SPIR-V shaders
        let forward_spv = std::fs::read("shader/forward.spv")
            .map_err(|e| format!("Failed to load forward.spv: {}", e))?;
        let backward_spv = std::fs::read("shader/backward.spv")
            .map_err(|e| format!("Failed to load backward.spv: {}", e))?;
        let reduce_spv = std::fs::read("shader/reduce.spv")
            .map_err(|e| format!("Failed to load reduce.spv: {}", e))?;

        println!("Creating gradient kernel...");

        // Create gradient kernel
        let gradient_kernel = GradientKernel::new(
            ctx.device.clone(),
            ctx.memory_properties,
            &forward_spv,
            &backward_spv,
            &reduce_spv,
            config.input_size,
            config.input_size,
            config.batch_size,
        )?;

        // Create command buffer pool
        let cmd_pool = CommandBufferPool::new(
            ctx.device.clone(),
            ctx.compute_queue_family,
            4,
        )?;

        println!("Creating tensor pool...");

        // Create tensor pool
        let tensor_pool = TensorPool::new(
            ctx.device.clone(),
            ctx.memory_properties,
            ctx.compute_queue_family,
            ctx.compute_queue,
            1024 * 1024 * 1024,  // 1GB device arena
            256 * 1024 * 1024,   // 256MB staging arena
        )?;

        println!("Training infrastructure ready\n");

        Ok(Self {
            // GPU resources (must drop before ctx)
            gradient_kernel,
            cmd_pool,
            tensor_pool,
            // Config (no resources)
            config,
            // Vulkan context (must drop last)
            ctx,
        })
    }

    /// Run training loop
    fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Vulkan Training Started ===\n");
        println!("Configuration:");
        println!("  Epochs: {}", self.config.epochs);
        println!("  Learning rate: {}", self.config.learning_rate);
        println!("  Input size: {}", self.config.input_size);
        println!("  Patience: {}", self.config.patience);
        println!();

        // Create input data: values from 0.1 to 25.6 (for 256 elements)
        let input_data: Vec<f32> = (0..self.config.input_size)
            .map(|i| (i as f32 + 1.0) * 0.1)  // 0.1, 0.2, ..., 25.6
            .collect();

        // Target: We want the model to learn weight = 2.0
        // So target = 2.0 * ReLU(input) = 2.0 * input (since all inputs are positive)
        let target_weight = 2.0f32;
        let target_data: Vec<f32> = input_data.iter()
            .map(|&x| target_weight * x.max(0.0))  // target = 2.0 * ReLU(x)
            .collect();

        let input_buffer = Buffer::new(
            self.ctx.device.clone(),
            (self.config.input_size as u64) * std::mem::size_of::<f32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            self.ctx.memory_properties,
        )?;
        input_buffer.upload_data(&input_data)?;

        // Output buffer needs to be HOST_VISIBLE so we can read it for loss computation
        let output_buffer = Buffer::new(
            self.ctx.device.clone(),
            (self.config.input_size as u64) * std::mem::size_of::<f32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            self.ctx.memory_properties,
        )?;

        // Output gradient buffer - will be computed from loss
        let output_grad_buffer = Buffer::new(
            self.ctx.device.clone(),
            (self.config.input_size as u64) * std::mem::size_of::<f32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            self.ctx.memory_properties,
        )?;

        // Training state
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;

        // Initialize weight to 1.0 (needs to learn to be 2.0)
        self.gradient_kernel.set_weight(1.0)?;
        let initial_weight = self.gradient_kernel.read_weight()?;
        println!("Initial weight: {:.6}", initial_weight);
        println!("Target weight:  {:.6}\n", target_weight);

        // Training loop
        for epoch in 0..self.config.epochs {
            // Step 1: Run forward pass only to get predictions
            // We need to manually record commands for fine-grained control
            let cmd_buffer = self.cmd_pool.begin_command_buffer()?;
            self.gradient_kernel.record_forward_pass(
                cmd_buffer,
                input_buffer.buffer,
                output_buffer.buffer,
            )?;
            self.cmd_pool.end_command_buffer(cmd_buffer)?;
            self.cmd_pool.submit_and_wait(cmd_buffer, self.ctx.compute_queue)?;

            // Step 2: Read predictions from GPU
            let mut predictions = vec![0.0f32; self.config.input_size as usize];
            output_buffer.download_data(&mut predictions)?;

            // Step 3: Compute MSE loss and gradients on CPU
            // Loss = (1/n) * sum((prediction - target)^2)
            // d(Loss)/d(prediction) = (2/n) * (prediction - target)
            let n = self.config.input_size as f32;
            let mut total_loss = 0.0f32;
            let mut output_grads = vec![0.0f32; self.config.input_size as usize];

            for i in 0..self.config.input_size as usize {
                let diff = predictions[i] - target_data[i];
                total_loss += diff * diff;
                output_grads[i] = (2.0 / n) * diff;  // MSE gradient
            }

            let loss = total_loss / n;  // Mean squared error

            // Step 4: Upload output gradients to GPU
            output_grad_buffer.upload_data(&output_grads)?;

            // Step 5: Run backward + reduce passes
            let cmd_buffer = self.cmd_pool.begin_command_buffer()?;
            self.gradient_kernel.record_backward_pass(
                cmd_buffer,
                input_buffer.buffer,
                output_grad_buffer.buffer,
            )?;
            self.gradient_kernel.record_reduce_pass(cmd_buffer)?;
            self.cmd_pool.end_command_buffer(cmd_buffer)?;
            self.cmd_pool.submit_and_wait(cmd_buffer, self.ctx.compute_queue)?;

            // Step 6: Apply gradient descent
            let weight_before = self.gradient_kernel.read_weight()?;
            let param_grad = self.gradient_kernel.read_param_gradient()?;
            let new_weight = self.gradient_kernel.apply_gradient_update(self.config.learning_rate)?;

            // Log progress
            if epoch < 5 || epoch % 10 == 0 || loss < best_loss {
                println!(
                    "Epoch {:3}: loss={:.6}, weight={:.6}→{:.6}, grad={:.4}{}",
                    epoch + 1,
                    loss,
                    weight_before,
                    new_weight,
                    param_grad,
                    if loss < best_loss { " ✓" } else { "" }
                );
            }

            // Check for improvement
            if loss < best_loss {
                best_loss = loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            // Early stopping
            if epochs_without_improvement >= self.config.patience {
                println!("\nEarly stopping triggered (patience={})", self.config.patience);
                break;
            }

            // Convergence check: stop if loss is very small
            if loss < 1e-6 {
                println!("\nConverged! Loss < 1e-6");
                break;
            }
        }

        // Final results
        let final_weight = self.gradient_kernel.read_weight()?;
        println!("\n=== Training Complete ===");
        println!("Target weight: {:.6}", target_weight);
        println!("Final weight:  {:.6}", final_weight);
        println!("Weight error:  {:.6}", (final_weight - target_weight).abs());
        println!("Best loss:     {:.6}", best_loss);

        Ok(())
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let config = TrainingConfig::default();
    let mut trainer = Trainer::new(config)?;
    trainer.train()?;

    Ok(())
}
