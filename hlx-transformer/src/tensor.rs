//! Tensor Abstraction for Vulkan ML
//!
//! Provides shape-aware tensor operations on top of raw GPU buffers.
//! All tensors are row-major with shape (batch, seq_len, d_model) convention.

use std::sync::Arc;
use ash::vk;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::VulkanErrorKind;

// =============================================================================
// DATA TYPES
// =============================================================================

/// Supported tensor data types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,  // For token indices
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::U32 => 4,
        }
    }
}

// =============================================================================
// TENSOR
// =============================================================================

/// GPU tensor with shape information
///
/// Wraps a Vulkan buffer with metadata for shape-aware operations.
/// Memory layout is always row-major (C-contiguous).
#[derive(Clone)]
pub struct Tensor {
    /// Underlying GPU buffer
    buffer: Buffer,
    /// Shape dimensions, e.g., [batch, seq_len, d_model]
    shape: Vec<u32>,
    /// Strides for indexing (computed from shape)
    strides: Vec<u32>,
    /// Data type
    dtype: DType,
    /// Device reference
    device: Arc<Device>,
}

impl Tensor {
    /// Creates a new tensor filled with zeros
    pub fn zeros(
        shape: &[u32],
        dtype: DType,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let numel = shape.iter().product::<u32>() as usize;
        let size_bytes = numel * dtype.size_bytes();
        
        let memory_properties = device.memory_properties();
        let buffer = Buffer::new(
            device.clone(),
            size_bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            memory_properties,
        )?;
        
        // Zero-initialize
        let zeros = vec![0u8; size_bytes];
        buffer.upload_bytes(&zeros)?;
        
        let strides = Self::compute_strides(shape);
        
        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            strides,
            dtype,
            device,
        })
    }
    
    /// Creates a tensor from f32 data
    pub fn from_f32(
        data: &[f32],
        shape: &[u32],
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let numel = shape.iter().product::<u32>() as usize;
        assert_eq!(data.len(), numel, "Data length must match shape");
        
        let mut tensor = Self::zeros(shape, DType::F32, device)?;
        tensor.buffer.upload_data(data)?;
        Ok(tensor)
    }
    
    /// Creates a tensor from u32 data (for token indices)
    pub fn from_u32(
        data: &[u32],
        shape: &[u32],
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let numel = shape.iter().product::<u32>() as usize;
        assert_eq!(data.len(), numel, "Data length must match shape");
        
        let mut tensor = Self::zeros(shape, DType::U32, device)?;
        tensor.buffer.upload_data(data)?;
        Ok(tensor)
    }
    
    /// Downloads tensor data as f32 vector
    pub fn to_f32(&self) -> Result<Vec<f32>, VulkanErrorKind> {
        assert_eq!(self.dtype, DType::F32, "Tensor must be f32");
        let numel = self.numel();
        let mut data = vec![0.0f32; numel];
        self.buffer.download_data(&mut data)?;
        Ok(data)
    }
    
    /// Downloads tensor data as u32 vector
    pub fn to_u32(&self) -> Result<Vec<u32>, VulkanErrorKind> {
        assert_eq!(self.dtype, DType::U32, "Tensor must be u32");
        let numel = self.numel();
        let mut data = vec![0u32; numel];
        self.buffer.download_data(&mut data)?;
        Ok(data)
    }
    
    /// Returns the tensor shape
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }
    
    /// Returns number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Returns total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<u32>() as usize
    }
    
    /// Returns size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
    
    /// Returns the underlying Vulkan buffer handle
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.buffer()
    }
    
    /// Returns reference to the Buffer wrapper
    pub fn buffer_ref(&self) -> &Buffer {
        &self.buffer
    }
    
    /// Returns the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    /// Returns the device
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
    
    /// Creates a view with a new shape (must have same numel)
    pub fn reshape(&self, new_shape: &[u32]) -> Self {
        let old_numel: u32 = self.shape.iter().product();
        let new_numel: u32 = new_shape.iter().product();
        assert_eq!(old_numel, new_numel, "Reshape must preserve element count");
        
        Self {
            buffer: self.buffer.clone(),
            shape: new_shape.to_vec(),
            strides: Self::compute_strides(new_shape),
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
    
    /// Computes row-major strides from shape
    fn compute_strides(shape: &[u32]) -> Vec<u32> {
        let mut strides = vec![1u32; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    
    /// Returns a shape string for debugging
    pub fn shape_str(&self) -> String {
        format!("[{}]", self.shape.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "))
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={}, dtype={:?})", self.shape_str(), self.dtype)
    }
}

// =============================================================================
// TENSOR SHAPE UTILITIES
// =============================================================================

/// Validates that two tensors have compatible shapes for matrix multiplication
pub fn validate_matmul_shapes(a: &Tensor, b: &Tensor) -> Result<Vec<u32>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err("Matrices must have at least 2 dimensions".to_string());
    }
    
    let a_m = a_shape[a_shape.len() - 2];
    let a_k = a_shape[a_shape.len() - 1];
    let b_k = b_shape[b_shape.len() - 2];
    let b_n = b_shape[b_shape.len() - 1];
    
    if a_k != b_k {
        return Err(format!(
            "Incompatible shapes for matmul: A has K={}, B has K={}",
            a_k, b_k
        ));
    }
    
    // Output shape: batch dims + (M, N)
    let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
    out_shape.push(a_m);
    out_shape.push(b_n);
    
    Ok(out_shape)
}

/// Validates shapes for element-wise operations (broadcasting)
pub fn validate_broadcast_shapes(a: &Tensor, b: &Tensor) -> Result<Vec<u32>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    let max_len = a_shape.len().max(b_shape.len());
    let mut out_shape = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let a_dim = if i < a_shape.len() { 
            a_shape[a_shape.len() - 1 - i] 
        } else { 
            1 
        };
        let b_dim = if i < b_shape.len() { 
            b_shape[b_shape.len() - 1 - i] 
        } else { 
            1 
        };
        
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                a_shape, b_shape
            ));
        }
        
        out_shape.push(a_dim.max(b_dim));
    }
    
    out_shape.reverse();
    Ok(out_shape)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_strides() {
        // Shape [2, 3, 4] should have strides [12, 4, 1]
        let strides = Tensor::compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
        
        // Shape [5] should have strides [1]
        let strides = Tensor::compute_strides(&[5]);
        assert_eq!(strides, vec![1]);
        
        // Shape [2, 3] should have strides [3, 1]
        let strides = Tensor::compute_strides(&[2, 3]);
        assert_eq!(strides, vec![3, 1]);
    }
    
    #[test]
    fn test_validate_matmul_shapes() {
        // (2, 3) × (3, 4) → (2, 4)
        let result = validate_matmul_shapes_raw(&[2, 3], &[3, 4]);
        assert_eq!(result.unwrap(), vec![2, 4]);
        
        // (batch=5, 2, 3) × (3, 4) → (5, 2, 4)
        let result = validate_matmul_shapes_raw(&[5, 2, 3], &[3, 4]);
        assert_eq!(result.unwrap(), vec![5, 2, 4]);
        
        // Incompatible: (2, 3) × (4, 5)
        let result = validate_matmul_shapes_raw(&[2, 3], &[4, 5]);
        assert!(result.is_err());
    }
    
    fn validate_matmul_shapes_raw(a: &[u32], b: &[u32]) -> Result<Vec<u32>, String> {
        if a.len() < 2 || b.len() < 2 {
            return Err("Need 2D".to_string());
        }
        let a_k = a[a.len() - 1];
        let b_k = b[b.len() - 2];
        if a_k != b_k {
            return Err("K mismatch".to_string());
        }
        let mut out = a[..a.len() - 2].to_vec();
        out.push(a[a.len() - 2]);
        out.push(b[b.len() - 1]);
        Ok(out)
    }
}
