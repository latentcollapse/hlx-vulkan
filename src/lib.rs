//! HLX Vulkan Compute Backend
//!
//! Provides Python bindings for Vulkan compute operations.
//! Integrates with HLX content-addressed storage for deterministic
//! shader caching and memoized execution.
//!
//! # Architecture
//!
//! This crate provides a thin Rust layer over Vulkan (via ash) that:
//! 1. Initializes Vulkan instance, device, and compute queue
//! 2. Loads SPIR-V shaders with content-addressed caching
//! 3. Exposes a clean Python API via PyO3
//!
//! # Example (Python)
//!
//! ```python
//! from hlx_vulkan import VulkanContext
//!
//! ctx = VulkanContext()
//! print(f"GPU: {ctx.device_name}")
//!
//! shader_id = ctx.load_shader(spirv_bytes, "main")
//! assert ctx.is_shader_cached(shader_id)
//!
//! ctx.cleanup()
//! ```

use pyo3::prelude::*;

mod context;
mod error;
mod shader;
mod validation;
mod pipeline;
mod buffer;
mod compute;
mod gradient_kernel;
mod tensor_buffer;

// Transformer modules (from Opus)
pub mod tensor;
mod transformer_config;
mod gemm_kernel;
mod attention_kernel;
mod transformer_layer;

pub use context::VulkanContext;
pub use error::VulkanErrorKind;
pub use pipeline::{GraphicsPipeline, create_simple_render_pass};
pub use buffer::Buffer;
pub use compute::{ComputePipeline, CommandBufferPool, DescriptorBindingBuilder};
pub use gradient_kernel::{GradientKernel, GradientPushConstants, GradientParameters};
pub use tensor_buffer::{TensorBuffer, TensorPool, MemoryArena, ArenaStats, PoolStats};

// Transformer exports
pub use tensor::Tensor;
pub use transformer_config::TransformerConfig;
pub use gemm_kernel::{GemmKernel, LinearLayer};
pub use attention_kernel::MultiHeadAttention;
pub use transformer_layer::{TransformerLayer, TransformerModel};

/// HLX Vulkan Python module
///
/// Exposes VulkanContext and related types to Python.
#[pymodule]
fn hlx_vulkan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging on module load (respects RUST_LOG env var)
    env_logger::try_init().ok();

    // Register the main class
    m.add_class::<VulkanContext>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "HLX Vulkan compute backend with content-addressed shader caching")?;

    log::info!("hlx_vulkan module initialized (version {})", env!("CARGO_PKG_VERSION"));

    Ok(())
}
