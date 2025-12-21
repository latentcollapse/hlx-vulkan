//! Error types for HLX Vulkan operations
//!
//! Provides typed errors that translate cleanly to Python exceptions.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

/// Enumeration of all possible Vulkan-related errors.
///
/// Each variant provides a descriptive message that will be
/// surfaced to Python as a RuntimeError.
#[derive(Error, Debug)]
pub enum VulkanErrorKind {
    /// Failed to initialize Vulkan (instance, device, etc.)
    #[error("Vulkan initialization failed: {0}")]
    InitializationFailed(String),

    /// No GPU with compute capability found
    #[error("No suitable GPU found with compute queue support")]
    NoSuitableDevice,

    /// Failed to create VkShaderModule
    #[error("Shader creation failed: {0}")]
    ShaderCreationFailed(String),

    /// SPIR-V binary is invalid
    #[error("Invalid SPIR-V: {0}")]
    InvalidSpirv(String),

    /// Requested handle not in cache
    #[error("Handle not found in cache: {0}")]
    CacheMiss(String),

    /// Vulkan API returned an error
    #[error("Vulkan API error: {0:?}")]
    VulkanApi(ash::vk::Result),

    /// Entry point loading failed
    #[error("Failed to load Vulkan entry point: {0}")]
    EntryLoadFailed(String),

    /// Failed to create graphics pipeline
    #[error("Pipeline creation failed: {0}")]
    PipelineCreationFailed(String),

    /// Failed to create buffer
    #[error("Buffer creation failed: {0}")]
    BufferCreationFailed(String),

    /// Buffer not found in cache
    #[error("Buffer not found: {0}")]
    BufferNotFound(String),

    /// Generic error (Opus transformer API compatibility)
    #[error("{0}")]
    Other(String),
}

/// Convert VulkanErrorKind to Python exception
impl From<VulkanErrorKind> for PyErr {
    fn from(err: VulkanErrorKind) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Convert ash::vk::Result to VulkanErrorKind
impl From<ash::vk::Result> for VulkanErrorKind {
    fn from(result: ash::vk::Result) -> Self {
        VulkanErrorKind::VulkanApi(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VulkanErrorKind::InitializationFailed("test".to_string());
        assert!(err.to_string().contains("initialization failed"));
    }

    #[test]
    fn test_invalid_spirv_error() {
        let err = VulkanErrorKind::InvalidSpirv("bad magic".to_string());
        assert!(err.to_string().contains("Invalid SPIR-V"));
    }
}
