//! GEMM Kernel Validation Test
//!
//! Tests the Vulkan GEMM kernel against NumPy ground truth.
//! Simple 2×2 matrix multiplication to verify correctness.

use ash::{vk, Entry, Instance, Device};
use std::sync::Arc;
use std::ffi::CString;
use hlx_vulkan::{GemmKernel, CommandBufferPool, Tensor};

// =============================================================================
// VULKAN CONTEXT
// =============================================================================

/// Minimal Vulkan context for testing
struct TestContext {
    _entry: Entry,
    instance: Instance,
    device: Arc<Device>,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl TestContext {
    fn new() -> Result<Self, String> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;

        let app_name = CString::new("GEMM Test").unwrap();
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

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate devices: {:?}", e))?;

        if physical_devices.is_empty() {
            return Err("No Vulkan devices found".to_string());
        }

        let physical_device = physical_devices[0];

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

        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let compute_queue_family = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| "No compute queue family found".to_string())?;

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

        Ok(Self {
            _entry: entry,
            instance,
            device,
            compute_queue,
            compute_queue_family,
            memory_properties,
        })
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// =============================================================================
// MAIN TEST
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("=== GEMM Kernel Validation Test ===\n");

    // Initialize Vulkan
    println!("Initializing Vulkan...");
    let ctx = TestContext::new()?;
    let device = ctx.device.clone();

    // Load GEMM shaders
    println!("Loading GEMM shaders...");
    let forward_spv = std::fs::read("shader/spv/gemm.spv")?;
    let backward_spv = std::fs::read("shader/spv/gemm_backward.spv")?;

    let gemm = GemmKernel::new(device.clone(), &forward_spv, &backward_spv)?;

    // Test case: C = A × B
    // A = [[1, 2],     B = [[5, 6],
    //      [3, 4]]          [7, 8]]
    //
    // Expected C = [[1*5 + 2*7, 1*6 + 2*8],   = [[19, 22],
    //               [3*5 + 4*7, 3*6 + 4*8]]      [43, 50]]
    println!("\nTest case: 2×2 matrix multiplication");
    println!("A = [[1, 2],");
    println!("     [3, 4]]");
    println!("B = [[5, 6],");
    println!("     [7, 8]]");
    println!("Expected C = [[19, 22],");
    println!("              [43, 50]]\n");

    let m = 2u32;
    let k = 2u32;
    let n = 2u32;

    // Create test matrices (row-major layout)
    // Note: For now, we'll use Buffer directly since Tensor::zeros has a broken memory_properties hack
    use hlx_vulkan::Buffer;
    use ash::vk;

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    // Create buffers manually with correct memory properties
    let a_buffer = Buffer::new(
        device.clone(),
        (m * k * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )?;
    a_buffer.upload_data(&a_data)?;

    let b_buffer = Buffer::new(
        device.clone(),
        (k * n * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )?;
    b_buffer.upload_data(&b_data)?;

    let c_buffer = Buffer::new(
        device.clone(),
        (m * n * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ctx.memory_properties,
    )?;
    let zeros = vec![0.0f32; (m * n) as usize];
    c_buffer.upload_data(&zeros)?;

    println!("Created buffers:");
    println!("  A: {}×{} = {} elements", m, k, m*k);
    println!("  B: {}×{} = {} elements", k, n, k*n);
    println!("  C: {}×{} = {} elements", m, n, m*n);

    // Create command pool
    let mut cmd_pool = CommandBufferPool::new(device.clone(), ctx.compute_queue_family, 1)?;

    // Run GEMM: C = A × B
    println!("\nRunning GEMM kernel...");
    let cmd_buffer = cmd_pool.begin_command_buffer()?;

    gemm.record_forward(
        cmd_buffer,
        a_buffer.buffer(),
        b_buffer.buffer(),
        c_buffer.buffer(),
        None,  // No bias
        m,
        k,
        n,
    )?;

    cmd_pool.end_command_buffer(cmd_buffer)?;
    cmd_pool.submit_and_wait(cmd_buffer, ctx.compute_queue)?;

    // Download results
    println!("Downloading results...");
    let mut c_result = vec![0.0f32; (m * n) as usize];
    c_buffer.download_data(&mut c_result)?;

    println!("\nGot result:");
    println!("  C[0,0] = {}", c_result[0]);
    println!("  C[0,1] = {}", c_result[1]);
    println!("  C[1,0] = {}", c_result[2]);
    println!("  C[1,1] = {}", c_result[3]);

    // Expected results (computed by hand)
    let expected = vec![19.0, 22.0, 43.0, 50.0];

    println!("\nExpected:");
    println!("  C[0,0] = {}", expected[0]);
    println!("  C[0,1] = {}", expected[1]);
    println!("  C[1,0] = {}", expected[2]);
    println!("  C[1,1] = {}", expected[3]);

    // Validate
    println!("\nValidating...");
    let mut all_match = true;
    let tolerance = 1e-5;

    for i in 0..4 {
        let diff = (c_result[i] - expected[i]).abs();
        let matches = diff < tolerance;

        if !matches {
            println!("  ❌ Element {}: got {}, expected {}, diff = {}",
                i, c_result[i], expected[i], diff);
            all_match = false;
        } else {
            println!("  ✅ Element {}: {} (diff = {:.2e})", i, c_result[i], diff);
        }
    }

    println!("\n{}", "=".repeat(50));
    if all_match {
        println!("✅ GEMM KERNEL VALIDATION PASSED");
        println!("All elements match within tolerance ({})", tolerance);
        println!("\nThe Vulkan GEMM kernel produces correct results!");
    } else {
        println!("❌ GEMM KERNEL VALIDATION FAILED");
        println!("Some elements differ from expected values");
        println!("\nThe GEMM shader may have bugs or incorrect memory layout.");
    }
    println!("{}\n", "=".repeat(50));

    Ok(())
}
