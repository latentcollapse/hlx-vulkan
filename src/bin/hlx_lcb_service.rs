//! HLX LC-B Service
//!
//! Unix socket service that receives LC-B batches from Python,
//! executes them on the GPU, and returns results.
//!
//! Usage:
//!   hlx_lcb_service [--socket /tmp/hlx_vulkan.sock]

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Import from the library crate
use hlx_vulkan::lcb::parser::{LCBParser, TensorData, serialize_tensor};
use hlx_vulkan::lcb::executor::LCBExecutor;

/// Service configuration
struct ServiceConfig {
    socket_path: PathBuf,
    shader_dir: PathBuf,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/hlx_vulkan.sock"),
            shader_dir: PathBuf::from("shader/spv"),
        }
    }
}

fn main() {
    let config = parse_args();
    
    println!("╔══════════════════════════════════════════╗");
    println!("║     HLX LC-B GPU Service                 ║");
    println!("╚══════════════════════════════════════════╝\n");
    
    // Initialize Vulkan and executor
    let executor = match init_executor(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to initialize GPU executor: {}", e);
            std::process::exit(1);
        }
    };
    
    // Wrap in Arc for signal handler
    let mut executor = executor;
    
    // Setup signal handling for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        println!("\n📛 Received shutdown signal...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");
    
    // Remove existing socket if present
    if config.socket_path.exists() {
        std::fs::remove_file(&config.socket_path).ok();
    }
    
    // Bind socket
    let listener = match UnixListener::bind(&config.socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind socket: {}", e);
            std::process::exit(1);
        }
    };
    
    // Set non-blocking for graceful shutdown
    listener.set_nonblocking(true).expect("Cannot set non-blocking");
    
    println!("🚀 Listening on {:?}", config.socket_path);
    println!("   Press Ctrl+C to stop\n");
    
    let mut request_count = 0u64;
    
    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                request_count += 1;
                if let Err(e) = handle_connection(stream, &mut executor, request_count) {
                    eprintln!("Request {}: Error: {}", request_count, e);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No pending connections, sleep briefly
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => {
                eprintln!("Accept error: {}", e);
            }
        }
    }
    
    // Cleanup
    println!("\n🧹 Cleaning up...");
    std::fs::remove_file(&config.socket_path).ok();
    println!("✅ Service stopped. Processed {} requests.", request_count);
}

fn init_executor(config: &ServiceConfig) -> Result<LCBExecutor, String> {
    use ash::{vk, Entry, Instance, Device};
    use std::sync::Arc;
    use std::ffi::CString;
    
    // Load Vulkan
    let entry = unsafe { Entry::load() }
        .map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;
    
    // Create instance
    let app_name = CString::new("HLX LC-B Service").unwrap();
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
    
    // Get physical device
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("Failed to enumerate devices: {:?}", e))?;
    
    if physical_devices.is_empty() {
        return Err("No Vulkan devices found".to_string());
    }
    
    let physical_device = physical_devices[0];
    
    let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe {
        std::ffi::CStr::from_ptr(device_props.device_name.as_ptr()).to_string_lossy()
    };
    println!("🖥️  Using GPU: {}", device_name);
    
    let memory_properties = unsafe {
        instance.get_physical_device_memory_properties(physical_device)
    };
    
    // Find compute queue family
    let queue_family_props = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };
    
    let queue_family_index = queue_family_props
        .iter()
        .enumerate()
        .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| "No compute queue family".to_string())?;
    
    // Create logical device
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);
    
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));
    
    let device = unsafe {
        instance.create_device(physical_device, &device_create_info, None)
    }.map_err(|e| format!("Failed to create device: {:?}", e))?;
    
    let device = Arc::new(device);
    let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    
    // Create executor
    LCBExecutor::new(
        device,
        compute_queue,
        queue_family_index,
        memory_properties,
        &config.shader_dir,
    )
}

fn handle_connection(
    mut stream: UnixStream,
    executor: &mut LCBExecutor,
    request_id: u64,
) -> Result<(), String> {
    let start = std::time::Instant::now();
    
    // Set blocking for this connection
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    
    // Read request length
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).map_err(|e| format!("Read length failed: {}", e))?;
    let request_len = u32::from_le_bytes(len_buf) as usize;
    
    // Read request body
    let mut request_buf = vec![0u8; request_len];
    stream.read_exact(&mut request_buf).map_err(|e| format!("Read body failed: {}", e))?;
    
    // Parse LC-B batch
    let mut parser = LCBParser::new(&request_buf);
    let batch = parser.parse()?;
    
    println!("📥 Request {}: {} instructions, {} bytes", 
        request_id, batch.instructions.len(), request_len);
    
    // Execute on GPU
    let result = executor.execute(&batch)?;
    
    // Serialize response
    let mut response = Vec::new();
    
    // Number of output tensors
    response.extend_from_slice(&(result.outputs.len() as u32).to_le_bytes());
    
    // Each tensor
    for tensor in &result.outputs {
        let serialized = serialize_tensor(tensor);
        response.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
        response.extend_from_slice(&serialized);
    }
    
    // Send response
    stream.write_all(&(response.len() as u32).to_le_bytes())
        .map_err(|e| format!("Write response length failed: {}", e))?;
    stream.write_all(&response)
        .map_err(|e| format!("Write response failed: {}", e))?;
    
    let elapsed = start.elapsed();
    println!("📤 Request {}: {} outputs, {} bytes, {:.2}ms",
        request_id, result.outputs.len(), response.len(), elapsed.as_secs_f64() * 1000.0);
    
    Ok(())
}

fn parse_args() -> ServiceConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = ServiceConfig::default();
    let mut i = 1;
    
    while i < args.len() {
        match args[i].as_str() {
            "--socket" | "-s" => {
                i += 1;
                if i < args.len() {
                    config.socket_path = PathBuf::from(&args[i]);
                }
            }
            "--shader-dir" | "-d" => {
                i += 1;
                if i < args.len() {
                    config.shader_dir = PathBuf::from(&args[i]);
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    
    config
}

fn print_help() {
    println!("HLX LC-B GPU Service\n");
    println!("Receives LC-B batches over Unix socket, executes on GPU, returns results.\n");
    println!("Usage: hlx_lcb_service [OPTIONS]\n");
    println!("Options:");
    println!("  -s, --socket <path>      Socket path (default: /tmp/hlx_vulkan.sock)");
    println!("  -d, --shader-dir <path>  Shader directory (default: shader/spv)");
    println!("  -h, --help               Show this help");
}
