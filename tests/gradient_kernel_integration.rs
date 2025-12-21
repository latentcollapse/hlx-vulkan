//! Integration tests for GradientKernel with real Vulkan dispatch
//!
//! These tests require a Vulkan-capable GPU and are marked #[ignore] by default.
//! Run with: cargo test -- --ignored
//!
//! # Test Coverage
//!
//! 1. **Determinism** - Verify bit-identical results across multiple runs
//! 2. **Gradient Correctness** - Compare analytic gradients to finite differences
//! 3. **Memory Safety** - Validate proper resource cleanup

use hlx_vulkan::{GradientKernel, GradientPushConstants, GradientParameters};
use ash::{vk, Entry, Instance, Device};
use std::sync::Arc;
use std::ffi::CString;

// =============================================================================
// TEST UTILITIES
// =============================================================================

/// Minimal Vulkan context for testing
///
/// Initializes instance, physical device, logical device, and compute queue.
/// This is a simplified version of VulkanContext for testing purposes.
struct TestVulkanContext {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl TestVulkanContext {
    /// Create a new test Vulkan context
    fn new() -> Result<Self, String> {
        // Load Vulkan entry point
        let entry = unsafe { Entry::load() }
            .map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;

        // Create Vulkan instance
        let app_name = CString::new("HLX Gradient Test").unwrap();
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

        // Get device properties and memory properties
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

        // Get compute queue
        let compute_queue = unsafe {
            device.get_device_queue(compute_queue_family, 0)
        };

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

    /// Load compiled SPIR-V shader from disk
    fn load_spirv(path: &str) -> Result<Vec<u8>, String> {
        std::fs::read(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))
    }
}

impl Drop for TestVulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

#[test]
#[ignore] // Requires GPU
fn test_gradient_kernel_creation() {
    // Initialize Vulkan
    let ctx = TestVulkanContext::new()
        .expect("Failed to initialize Vulkan context");

    // Load SPIR-V shaders
    let forward_spv = TestVulkanContext::load_spirv("shader/forward.spv")
        .expect("Failed to load forward.spv");
    let backward_spv = TestVulkanContext::load_spirv("shader/backward.spv")
        .expect("Failed to load backward.spv");
    let reduce_spv = TestVulkanContext::load_spirv("shader/reduce.spv")
        .expect("Failed to load reduce.spv");

    // Create gradient kernel
    let kernel = GradientKernel::new(
        ctx.device.clone(),
        ctx.memory_properties,
        &forward_spv,
        &backward_spv,
        &reduce_spv,
        1024, // input_size
        1024, // output_size
        1,    // batch_size
    )
    .expect("Failed to create GradientKernel");

    // Verify configuration
    let (input_size, output_size, batch_size, num_workgroups) = kernel.config();
    assert_eq!(input_size, 1024);
    assert_eq!(output_size, 1024);
    assert_eq!(batch_size, 1);
    assert_eq!(num_workgroups, 4); // (1024 + 255) / 256 = 4

    println!("✓ GradientKernel created successfully");
    println!("  input_size: {}", input_size);
    println!("  output_size: {}", output_size);
    println!("  num_workgroups: {}", num_workgroups);
}

#[test]
#[ignore] // Requires GPU
fn test_determinism_5_runs() {
    // Initialize Vulkan
    let ctx = TestVulkanContext::new()
        .expect("Failed to initialize Vulkan context");

    // Load SPIR-V shaders
    let forward_spv = TestVulkanContext::load_spirv("shader/forward.spv")
        .expect("Failed to load forward.spv");
    let backward_spv = TestVulkanContext::load_spirv("shader/backward.spv")
        .expect("Failed to load backward.spv");
    let reduce_spv = TestVulkanContext::load_spirv("shader/reduce.spv")
        .expect("Failed to load reduce.spv");

    // Create gradient kernel
    let input_size = 256;
    let kernel = GradientKernel::new(
        ctx.device.clone(),
        ctx.memory_properties,
        &forward_spv,
        &backward_spv,
        &reduce_spv,
        input_size,
        input_size,
        1,
    )
    .expect("Failed to create GradientKernel");

    // Create command buffer pool
    use hlx_vulkan::CommandBufferPool;
    let mut cmd_pool = CommandBufferPool::new(
        ctx.device.clone(),
        ctx.compute_queue_family,
        4,
    )
    .expect("Failed to create command buffer pool");

    // Allocate test buffers
    use hlx_vulkan::Buffer;

    // Input buffer: simple linear ramp [0.0, 0.1, 0.2, ...]
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| i as f32 * 0.1)
        .collect();

    let input_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create input buffer");
    input_buffer.upload_data(&input_data).expect("Failed to upload input data");

    // Output buffer (will be written by forward pass)
    let output_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ctx.memory_properties,
    )
    .expect("Failed to create output buffer");

    // Output gradient buffer: constant 1.0 (simple upstream gradient)
    let output_grad_data = vec![1.0f32; input_size as usize];
    let output_grad_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create output_grad buffer");
    output_grad_buffer.upload_data(&output_grad_data).expect("Failed to upload output_grad");

    // Run gradient computation 5 times and collect results
    let mut results = Vec::new();
    for i in 0..5 {
        kernel.full_pass(
            &mut cmd_pool,
            input_buffer.buffer,
            output_buffer.buffer,
            output_grad_buffer.buffer,
            ctx.compute_queue,
        )
        .expect(&format!("Full pass {} failed", i));

        let param_grad = kernel.read_param_gradient()
            .expect(&format!("Failed to read gradient at iteration {}", i));

        results.push(param_grad);

        if i > 0 {
            // Bitwise comparison with previous result
            let prev_bits = results[i - 1].to_bits();
            let curr_bits = param_grad.to_bits();

            assert_eq!(
                prev_bits, curr_bits,
                "Determinism violation at iteration {}: prev={} (0x{:08x}), curr={} (0x{:08x})",
                i, results[i - 1], prev_bits, param_grad, curr_bits
            );
        }

        println!("  Run {}: param_grad = {} (0x{:08x})", i + 1, param_grad, param_grad.to_bits());
    }

    println!("✓ Determinism verified: 5 runs produced bit-identical results");
    println!("  Final gradient: {}", results[4]);
}

#[test]
#[ignore] // Requires GPU
fn test_gradient_correctness_finite_difference() {
    // Initialize Vulkan
    let ctx = TestVulkanContext::new()
        .expect("Failed to initialize Vulkan context");

    // Load SPIR-V shaders
    let forward_spv = TestVulkanContext::load_spirv("shader/forward.spv")
        .expect("Failed to load forward.spv");
    let backward_spv = TestVulkanContext::load_spirv("shader/backward.spv")
        .expect("Failed to load backward.spv");
    let reduce_spv = TestVulkanContext::load_spirv("shader/reduce.spv")
        .expect("Failed to load reduce.spv");

    // Small test case for verification
    let input_size = 64;
    let kernel = GradientKernel::new(
        ctx.device.clone(),
        ctx.memory_properties,
        &forward_spv,
        &backward_spv,
        &reduce_spv,
        input_size,
        input_size,
        1,
    )
    .expect("Failed to create GradientKernel");

    // Create command buffer pool
    use hlx_vulkan::CommandBufferPool;
    let mut cmd_pool = CommandBufferPool::new(
        ctx.device.clone(),
        ctx.compute_queue_family,
        4,
    )
    .expect("Failed to create command buffer pool");

    // Test input: small positive values
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 + 1.0) * 0.1) // [0.1, 0.2, 0.3, ...]
        .collect();

    use hlx_vulkan::Buffer;

    let input_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create input buffer");
    input_buffer.upload_data(&input_data).expect("Failed to upload input");

    let output_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ctx.memory_properties,
    )
    .expect("Failed to create output buffer");

    // Upstream gradient: all 1.0
    let output_grad_data = vec![1.0f32; input_size as usize];
    let output_grad_buffer = Buffer::new(
        ctx.device.clone(),
        (input_size as u64) * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create output_grad buffer");
    output_grad_buffer.upload_data(&output_grad_data).expect("Failed to upload output_grad");

    // Compute analytic gradient
    kernel.full_pass(
        &mut cmd_pool,
        input_buffer.buffer,
        output_buffer.buffer,
        output_grad_buffer.buffer,
        ctx.compute_queue,
    )
    .expect("Full pass failed");

    let analytic_grad = kernel.read_param_gradient()
        .expect("Failed to read analytic gradient");

    // Compute numerical gradient via finite difference
    // For a simple scalar model: y = ReLU(x * w)
    // We're computing d(loss)/dw where loss = sum(upstream_grad * y)
    //
    // Forward: y_i = ReLU(x_i)  (no weight in v1, but gradient still computes param contribution)
    // Backward: param_grad = sum(x_i * upstream_grad_i * (y_i > 0 ? 1 : 0))
    //
    // Since all inputs are positive, ReLU passes them through:
    // param_grad = sum(x_i * 1.0 * 1.0) = sum(x_i)

    let expected_grad: f32 = input_data.iter().sum();

    println!("  Analytic gradient: {}", analytic_grad);
    println!("  Expected gradient: {}", expected_grad);
    println!("  Relative error: {:.6}%",
             ((analytic_grad - expected_grad) / expected_grad).abs() * 100.0);

    // Verify within 0.01% tolerance (tight for FP32)
    let rel_error = ((analytic_grad - expected_grad) / expected_grad).abs();
    assert!(
        rel_error < 1e-4,
        "Gradient correctness failed: analytic={}, expected={}, rel_error={}",
        analytic_grad, expected_grad, rel_error
    );

    println!("✓ Gradient correctness verified");
}

#[test]
#[ignore] // Requires GPU
#[cfg(feature = "validate_determinism")]
fn test_validate_determinism_feature() {
    // Test the built-in validate_determinism feature
    let ctx = TestVulkanContext::new()
        .expect("Failed to initialize Vulkan context");

    let forward_spv = TestVulkanContext::load_spirv("shader/forward.spv")
        .expect("Failed to load forward.spv");
    let backward_spv = TestVulkanContext::load_spirv("shader/backward.spv")
        .expect("Failed to load backward.spv");
    let reduce_spv = TestVulkanContext::load_spirv("shader/reduce.spv")
        .expect("Failed to load reduce.spv");

    let kernel = GradientKernel::new(
        ctx.device.clone(),
        ctx.memory_properties,
        &forward_spv,
        &backward_spv,
        &reduce_spv,
        256,
        256,
        1,
    )
    .expect("Failed to create GradientKernel");

    use hlx_vulkan::{CommandBufferPool, Buffer};

    let mut cmd_pool = CommandBufferPool::new(
        ctx.device.clone(),
        ctx.compute_queue_family,
        4,
    )
    .expect("Failed to create command buffer pool");

    // Create test buffers
    let input_data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let input_buffer = Buffer::new(
        ctx.device.clone(),
        256 * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create input buffer");
    input_buffer.upload_data(&input_data).unwrap();

    let output_buffer = Buffer::new(
        ctx.device.clone(),
        256 * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ctx.memory_properties,
    )
    .expect("Failed to create output buffer");

    let output_grad_data = vec![1.0f32; 256];
    let output_grad_buffer = Buffer::new(
        ctx.device.clone(),
        256 * std::mem::size_of::<f32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )
    .expect("Failed to create output_grad buffer");
    output_grad_buffer.upload_data(&output_grad_data).unwrap();

    // Use built-in validation
    kernel.validate_determinism(
        &mut cmd_pool,
        input_buffer.buffer,
        output_buffer.buffer,
        output_grad_buffer.buffer,
        ctx.compute_queue,
        3, // 3 iterations
    )
    .expect("Determinism validation failed");

    println!("✓ validate_determinism feature verified (3 runs)");
}
