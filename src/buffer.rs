//! Vulkan buffer management
//!
//! Handles creation, data upload, and lifecycle of GPU buffers.
//! Supports vertex buffers, uniform buffers, and CPU→GPU transfer.

use ash::{vk, Device};
use std::sync::Arc;
use crate::error::VulkanErrorKind;

/// Wrapper around VkBuffer and VkDeviceMemory
///
/// Manages both the buffer and its backing GPU memory.
/// Handles data upload via memory mapping.
#[derive(Clone)]
pub struct Buffer {
    /// Vulkan buffer handle
    pub buffer: vk::Buffer,

    /// Vulkan device memory backing the buffer
    pub memory: vk::DeviceMemory,

    /// Size of the buffer in bytes
    pub size: vk::DeviceSize,

    /// Reference to device (needed for cleanup and memory operations)
    device: Arc<Device>,
}

impl Buffer {
    /// Create a new GPU buffer with specified usage and memory properties.
    ///
    /// # Arguments
    ///
    /// * `device` - Logical device (wrapped in Arc)
    /// * `size` - Size in bytes
    /// * `usage` - Buffer usage flags (VERTEX_BUFFER, UNIFORM_BUFFER, etc.)
    /// * `properties` - Memory property flags (HOST_VISIBLE, HOST_COHERENT, etc.)
    /// * `memory_properties` - Physical device memory properties
    ///
    /// # Returns
    ///
    /// Buffer instance ready for data upload
    pub fn new(
        device: Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Self, VulkanErrorKind> {
        // Create VkBuffer
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create VkBuffer: {:?}",
                    e
                ))
            })?;

        log::debug!("Created VkBuffer: size={} bytes, usage={:?}", size, usage);

        // Get buffer memory requirements
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find suitable memory type
        let memory_type_index =
            find_memory_type(&memory_properties, mem_requirements.memory_type_bits, properties)
                .ok_or_else(|| {
                    VulkanErrorKind::InitializationFailed(
                        "No suitable memory type found for buffer".to_string(),
                    )
                })?;

        // Allocate device memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to allocate VkDeviceMemory: {:?}",
                    e
                ))
            })?;

        log::debug!(
            "Allocated VkDeviceMemory: size={} bytes, memory_type_index={}",
            mem_requirements.size,
            memory_type_index
        );

        // Bind buffer to memory
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to bind buffer to memory: {:?}",
                    e
                ))
            })?;

        log::debug!("Bound VkBuffer to VkDeviceMemory");

        Ok(Self {
            buffer,
            memory,
            size,
            device,
        })
    }

    /// Upload data to the buffer from CPU memory.
    ///
    /// Memory must be HOST_VISIBLE for this to work.
    /// Automatically handles mapping/unmapping.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to upload
    ///
    /// # Returns
    ///
    /// Ok if upload succeeds
    pub fn upload_data<T: Copy>(&self, data: &[T]) -> Result<(), VulkanErrorKind> {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if data_size > self.size {
            return Err(VulkanErrorKind::InitializationFailed(format!(
                "Data size {} exceeds buffer size {}",
                data_size, self.size
            )));
        }

        // Map memory to CPU address space
        let mapped_ptr = unsafe {
            self.device
                .map_memory(self.memory, 0, data_size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to map buffer memory: {:?}",
                e
            ))
        })?;

        // Copy data to mapped memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                data_size as usize,
            );
        }

        // Unmap memory
        unsafe {
            self.device.unmap_memory(self.memory);
        }

        log::debug!("Uploaded {} bytes to buffer", data_size);

        Ok(())
    }

    /// Download data from the buffer to CPU memory.
    ///
    /// Memory must be HOST_VISIBLE for this to work.
    /// Automatically handles mapping/unmapping.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice to receive the data
    ///
    /// # Returns
    ///
    /// Ok if download succeeds
    pub fn download_data<T: Copy>(&self, data: &mut [T]) -> Result<(), VulkanErrorKind> {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if data_size > self.size {
            return Err(VulkanErrorKind::InitializationFailed(format!(
                "Request size {} exceeds buffer size {}",
                data_size, self.size
            )));
        }

        // Map memory to CPU address space
        let mapped_ptr = unsafe {
            self.device
                .map_memory(self.memory, 0, data_size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to map buffer memory: {:?}",
                e
            ))
        })?;

        // Copy data from mapped memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped_ptr as *const u8,
                data.as_mut_ptr() as *mut u8,
                data_size as usize,
            );
        }

        // Unmap memory
        unsafe {
            self.device.unmap_memory(self.memory);
        }

        log::debug!("Downloaded {} bytes from buffer", data_size);

        Ok(())
    }

    /// Get the buffer handle for use in commands
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Get the buffer size in bytes
    pub fn len(&self) -> vk::DeviceSize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    // =============================================================================
    // ADAPTER LAYER FOR OPUS TRANSFORMER CODE
    // =============================================================================

    /// Adapter method: Returns buffer handle (Opus API compatibility)
    ///
    /// Opus code expects `buffer()` method instead of `handle()`.
    /// This is a thin wrapper to make the APIs compatible.
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Adapter method: Upload raw bytes (Opus API compatibility)
    ///
    /// Opus code expects `upload_bytes(&[u8])` instead of `upload_data<T>(&[T])`.
    /// This is a thin wrapper that casts to u8 slice.
    pub fn upload_bytes(&self, data: &[u8]) -> Result<(), VulkanErrorKind> {
        self.upload_data(data)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        log::debug!("Destroying buffer (size={})", self.size);
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Find suitable memory type for buffer allocation.
///
/// Searches through available memory types for one that:
/// 1. Is supported by the buffer (type_filter)
/// 2. Has all required properties
///
/// # Arguments
///
/// * `memory_properties` - Physical device memory properties
/// * `type_filter` - Bitmask of supported memory types from buffer
/// * `properties` - Required memory properties
///
/// # Returns
///
/// Memory type index if found, None otherwise
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..memory_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (memory_properties.memory_types[i as usize].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_memory_type() {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 2;
        props.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        props.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        // Should find type 1 (HOST_VISIBLE | HOST_COHERENT)
        let result = find_memory_type(
            &props,
            0b11, // Both types supported
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_find_memory_type_no_match() {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 1;
        props.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        // Should not find HOST_VISIBLE memory
        let result = find_memory_type(
            &props,
            0b1,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        assert_eq!(result, None);
    }
}
