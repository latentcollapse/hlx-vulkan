//! Content-Addressed Tensor Buffer Management
//!
//! Provides efficient GPU memory management for ML workloads with:
//! - **Batch allocation**: Single VkDeviceMemory → multiple VkBuffer bindings
//! - **Content addressing**: SHA-256 hashing for automatic deduplication
//! - **Memory pooling**: Arena allocator with reuse across training iterations
//!
//! # Architecture
//!
//! ```text
//! TensorPool
//!   ├── MemoryArena (1GB VkDeviceMemory)
//!   │     ├── TensorBuffer [0..256MB]
//!   │     ├── TensorBuffer [256MB..512MB]
//!   │     └── TensorBuffer [512MB..768MB]
//!   └── ContentAddressMap
//!         └── SHA256 → TensorBuffer (deduplication)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! let pool = TensorPool::new(device, memory_properties, 1024 * 1024 * 1024)?; // 1GB
//!
//! // Allocate tensor (content-addressed)
//! let data = vec![1.0f32; 1024];
//! let tensor = pool.allocate(&data)?;
//!
//! // Second allocation with same data returns cached tensor
//! let tensor2 = pool.allocate(&data)?;
//! assert_eq!(tensor.hash(), tensor2.hash());
//! ```

use ash::{vk, Device};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use sha2::{Sha256, Digest};

use crate::error::VulkanErrorKind;
use crate::buffer;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default arena size: 1GB
const DEFAULT_ARENA_SIZE: u64 = 1024 * 1024 * 1024;

/// Minimum allocation granularity (256 bytes, matches most GPU alignment)
const ALLOCATION_GRANULARITY: u64 = 256;

// =============================================================================
// TENSOR BUFFER
// =============================================================================

/// A single tensor buffer with content-addressed identity
///
/// Wraps a VkBuffer that is sub-allocated from a larger memory arena.
/// Each tensor has a unique SHA-256 hash of its contents for deduplication.
#[derive(Clone)]
pub struct TensorBuffer {
    /// Vulkan buffer handle
    pub buffer: vk::Buffer,

    /// Offset within the parent memory arena
    pub offset: u64,

    /// Size of this tensor in bytes
    pub size: u64,

    /// SHA-256 hash of tensor contents (content addressing)
    hash: [u8; 32],

    /// Reference to parent arena (keeps memory alive)
    arena: Arc<MemoryArena>,
}

impl TensorBuffer {
    /// Get the content hash of this tensor
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Get the buffer handle for use in descriptor sets
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Get the size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the offset within the parent memory
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Upload data to this tensor buffer
    ///
    /// # Safety
    /// Only call this if the parent memory is HOST_VISIBLE
    pub fn upload_data<T: Copy>(&self, data: &[T]) -> Result<(), VulkanErrorKind> {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if data_size > self.size {
            return Err(VulkanErrorKind::InitializationFailed(format!(
                "Data size {} exceeds tensor size {}",
                data_size, self.size
            )));
        }

        // Map memory at the correct offset
        let mapped_ptr = unsafe {
            self.arena.device.map_memory(
                self.arena.memory,
                self.offset,
                data_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to map tensor memory: {:?}",
                e
            ))
        })?;

        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                data_size as usize,
            );
        }

        // Unmap
        unsafe {
            self.arena.device.unmap_memory(self.arena.memory);
        }

        log::debug!("Uploaded {} bytes to tensor at offset {}", data_size, self.offset);
        Ok(())
    }

    /// Download data from this tensor buffer
    ///
    /// # Safety
    /// Only call this if the parent memory is HOST_VISIBLE
    pub fn download_data<T: Copy>(&self, data: &mut [T]) -> Result<(), VulkanErrorKind> {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if data_size > self.size {
            return Err(VulkanErrorKind::InitializationFailed(format!(
                "Request size {} exceeds tensor size {}",
                data_size, self.size
            )));
        }

        // Map memory at the correct offset
        let mapped_ptr = unsafe {
            self.arena.device.map_memory(
                self.arena.memory,
                self.offset,
                data_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to map tensor memory: {:?}",
                e
            ))
        })?;

        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped_ptr as *const u8,
                data.as_mut_ptr() as *mut u8,
                data_size as usize,
            );
        }

        // Unmap
        unsafe {
            self.arena.device.unmap_memory(self.arena.memory);
        }

        log::debug!("Downloaded {} bytes from tensor at offset {}", data_size, self.offset);
        Ok(())
    }
}

// =============================================================================
// MEMORY ARENA
// =============================================================================

/// Batch memory allocator - single VkDeviceMemory with multiple VkBuffer bindings
///
/// Manages a large contiguous block of GPU memory and sub-allocates tensors
/// from it. This reduces allocation overhead and improves memory locality.
pub struct MemoryArena {
    /// Vulkan device memory (single large allocation)
    memory: vk::DeviceMemory,

    /// Total size of the arena in bytes
    total_size: u64,

    /// Memory property flags (DEVICE_LOCAL, HOST_VISIBLE, etc.)
    properties: vk::MemoryPropertyFlags,

    /// Free list: (offset, size) pairs of available regions
    free_list: Mutex<Vec<(u64, u64)>>,

    /// Reference to device
    device: Arc<Device>,
}

impl MemoryArena {
    /// Create a new memory arena
    ///
    /// # Arguments
    ///
    /// * `device` - Vulkan device
    /// * `size` - Total arena size in bytes
    /// * `properties` - Memory property flags
    /// * `memory_properties` - Physical device memory properties
    pub fn new(
        device: Arc<Device>,
        size: u64,
        properties: vk::MemoryPropertyFlags,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Arc<Self>, VulkanErrorKind> {
        // Find suitable memory type
        let memory_type_index = buffer::find_memory_type(
            &memory_properties,
            0xFFFFFFFF, // All types allowed
            properties,
        )
        .ok_or_else(|| {
            VulkanErrorKind::InitializationFailed(
                "No suitable memory type found for arena".to_string(),
            )
        })?;

        // Allocate device memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to allocate arena memory: {:?}",
                    e
                ))
            })?;

        log::info!(
            "Created MemoryArena: size={} bytes ({} MB), properties={:?}",
            size,
            size / (1024 * 1024),
            properties
        );

        // Initialize free list with entire arena
        let free_list = Mutex::new(vec![(0, size)]);

        Ok(Arc::new(Self {
            memory,
            total_size: size,
            properties,
            free_list,
            device,
        }))
    }

    /// Allocate a tensor buffer from this arena
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes (will be rounded up to ALLOCATION_GRANULARITY)
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// (VkBuffer, offset) on success
    fn allocate_region(
        self: &Arc<Self>,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, u64), VulkanErrorKind> {
        // Round up to allocation granularity
        let aligned_size = ((size + ALLOCATION_GRANULARITY - 1) / ALLOCATION_GRANULARITY)
            * ALLOCATION_GRANULARITY;

        // Find free region (first-fit strategy)
        let mut free_list = self.free_list.lock().unwrap();

        let allocation = free_list
            .iter()
            .enumerate()
            .find(|(_, (_, free_size))| *free_size >= aligned_size);

        let (offset, remaining_size) = if let Some((idx, &(free_offset, free_size))) = allocation {
            // Remove this free block
            free_list.remove(idx);

            // If there's leftover space, add it back to free list
            let remaining = free_size - aligned_size;
            if remaining > 0 {
                free_list.push((free_offset + aligned_size, remaining));
            }

            (free_offset, remaining)
        } else {
            return Err(VulkanErrorKind::InitializationFailed(
                format!("Arena out of memory: requested {} bytes", aligned_size),
            ));
        };

        // Create VkBuffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(aligned_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create buffer in arena: {:?}",
                    e
                ))
            })?;

        // Bind buffer to arena memory at offset
        unsafe {
            self.device
                .bind_buffer_memory(buffer, self.memory, offset)
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to bind buffer to arena: {:?}",
                e
            ))
        })?;

        log::debug!(
            "Allocated {} bytes at offset {} (remaining: {} bytes free)",
            aligned_size,
            offset,
            remaining_size
        );

        Ok((buffer, offset))
    }

    /// Free a tensor buffer region (returns to free list)
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset of the region to free
    /// * `size` - Size of the region to free
    fn free_region(&self, offset: u64, size: u64) {
        let mut free_list = self.free_list.lock().unwrap();

        // Add back to free list
        free_list.push((offset, size));

        // TODO: Merge adjacent free regions (coalescing)
        // For now, simple free list without merging

        log::debug!("Freed {} bytes at offset {}", size, offset);
    }

    /// Get arena statistics
    pub fn stats(&self) -> ArenaStats {
        let free_list = self.free_list.lock().unwrap();
        let free_bytes: u64 = free_list.iter().map(|(_, size)| size).sum();
        let used_bytes = self.total_size - free_bytes;

        ArenaStats {
            total_bytes: self.total_size,
            used_bytes,
            free_bytes,
            num_allocations: free_list.len(),
        }
    }
}

impl Drop for MemoryArena {
    fn drop(&mut self) {
        log::info!("Destroying MemoryArena (size={} bytes)", self.total_size);
        unsafe {
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Arena memory statistics
#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub num_allocations: usize,
}

// =============================================================================
// TENSOR POOL
// =============================================================================

/// Content-addressed tensor pool with automatic deduplication
///
/// Manages tensor allocation from memory arenas with SHA-256 content addressing.
/// Identical tensors (by content) share the same GPU allocation.
pub struct TensorPool {
    /// Memory arena for GPU-local tensors
    device_arena: Arc<MemoryArena>,

    /// Memory arena for host-visible staging
    staging_arena: Arc<MemoryArena>,

    /// Transfer command pool for staging uploads
    transfer_pool: vk::CommandPool,

    /// Transfer queue for staging uploads
    transfer_queue: vk::Queue,

    /// Content address map: SHA256 → TensorBuffer
    content_map: Mutex<HashMap<[u8; 32], TensorBuffer>>,

    /// Reference to device
    device: Arc<Device>,
}

impl TensorPool {
    /// Create a new tensor pool
    ///
    /// # Arguments
    ///
    /// * `device` - Vulkan device
    /// * `memory_properties` - Physical device memory properties
    /// * `transfer_queue_family` - Queue family index for transfer operations
    /// * `transfer_queue` - Transfer queue handle
    /// * `device_arena_size` - Size of GPU-local arena (default: 1GB)
    /// * `staging_arena_size` - Size of staging arena (default: 256MB)
    pub fn new(
        device: Arc<Device>,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        transfer_queue_family: u32,
        transfer_queue: vk::Queue,
        device_arena_size: u64,
        staging_arena_size: u64,
    ) -> Result<Self, VulkanErrorKind> {
        // Create device-local arena (GPU memory)
        let device_arena = MemoryArena::new(
            device.clone(),
            device_arena_size,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            memory_properties,
        )?;

        // Create staging arena (host-visible memory for uploads)
        let staging_arena = MemoryArena::new(
            device.clone(),
            staging_arena_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            memory_properties,
        )?;

        // Create transfer command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let transfer_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to create transfer command pool: {:?}",
                    e
                ))
            })?;

        log::info!(
            "Created TensorPool: device={} MB, staging={} MB",
            device_arena_size / (1024 * 1024),
            staging_arena_size / (1024 * 1024)
        );

        Ok(Self {
            device_arena,
            staging_arena,
            transfer_pool,
            transfer_queue,
            content_map: Mutex::new(HashMap::new()),
            device,
        })
    }

    /// Allocate a tensor with content addressing
    ///
    /// If a tensor with identical content exists, returns the cached version.
    /// Otherwise, allocates new memory and uploads the data.
    ///
    /// # Arguments
    ///
    /// * `data` - Tensor data (will be hashed for content addressing)
    ///
    /// # Returns
    ///
    /// TensorBuffer (either new or cached)
    pub fn allocate<T: Copy>(&self, data: &[T]) -> Result<TensorBuffer, VulkanErrorKind> {
        // Compute content hash
        let hash = Self::compute_hash(data);

        // Check if we already have this tensor
        {
            let content_map = self.content_map.lock().unwrap();
            if let Some(tensor) = content_map.get(&hash) {
                log::debug!("Content address hit: reusing existing tensor (hash={:x})",
                           u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]));
                return Ok(tensor.clone());
            }
        }

        // Cache miss - allocate new tensor
        let size = (data.len() * std::mem::size_of::<T>()) as u64;

        let (buffer, offset) = self.device_arena.allocate_region(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let tensor = TensorBuffer {
            buffer,
            offset,
            size,
            hash,
            arena: self.device_arena.clone(),
        };

        // Upload data via staging buffer (DEVICE_LOCAL requires staging)
        self.upload_via_staging(&tensor, data)?;

        // Cache the tensor
        {
            let mut content_map = self.content_map.lock().unwrap();
            content_map.insert(hash, tensor.clone());
        }

        log::info!("Allocated new tensor: {} bytes, hash={:x}",
                   size,
                   u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]));

        Ok(tensor)
    }

    /// Upload data to device-local tensor via staging buffer
    ///
    /// # Pipeline
    /// 1. Allocate temporary staging buffer (HOST_VISIBLE)
    /// 2. Copy data to staging buffer
    /// 3. Record vkCmdCopyBuffer command
    /// 4. Submit to transfer queue and wait
    /// 5. Free staging buffer
    fn upload_via_staging<T: Copy>(
        &self,
        tensor: &TensorBuffer,
        data: &[T],
    ) -> Result<(), VulkanErrorKind> {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if data_size > tensor.size {
            return Err(VulkanErrorKind::InitializationFailed(format!(
                "Data size {} exceeds tensor size {}",
                data_size, tensor.size
            )));
        }

        // Allocate temporary staging buffer
        let (staging_buffer, staging_offset) = self.staging_arena.allocate_region(
            data_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        // Copy data to staging buffer
        let mapped_ptr = unsafe {
            self.device.map_memory(
                self.staging_arena.memory,
                staging_offset,
                data_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to map staging memory: {:?}",
                e
            ))
        })?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                data_size as usize,
            );
        }

        unsafe {
            self.device.unmap_memory(self.staging_arena.memory);
        }

        // Record transfer command
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| {
                VulkanErrorKind::InitializationFailed(format!(
                    "Failed to allocate transfer command buffer: {:?}",
                    e
                ))
            })?;

        let cmd_buffer = cmd_buffers[0];

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(cmd_buffer, &begin_info)
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to begin transfer command buffer: {:?}",
                e
            ))
        })?;

        // Copy command: staging → device
        let copy_region = vk::BufferCopy::default()
            .src_offset(staging_offset)
            .dst_offset(tensor.offset)
            .size(data_size);

        unsafe {
            self.device.cmd_copy_buffer(
                cmd_buffer,
                staging_buffer,
                tensor.buffer,
                &[copy_region],
            );
        }

        // End recording
        unsafe {
            self.device.end_command_buffer(cmd_buffer)
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to end transfer command buffer: {:?}",
                e
            ))
        })?;

        // Submit and wait
        let cmd_buffer_slice = [cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffer_slice);

        unsafe {
            self.device
                .queue_submit(self.transfer_queue, std::slice::from_ref(&submit_info), vk::Fence::null())
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to submit transfer command: {:?}",
                e
            ))
        })?;

        // Wait for transfer to complete
        unsafe {
            self.device.queue_wait_idle(self.transfer_queue)
        }
        .map_err(|e| {
            VulkanErrorKind::InitializationFailed(format!(
                "Failed to wait for transfer queue: {:?}",
                e
            ))
        })?;

        // Free command buffer
        unsafe {
            self.device
                .free_command_buffers(self.transfer_pool, &[cmd_buffer]);
        }

        // Free staging buffer
        self.staging_arena.free_region(staging_offset, data_size);

        log::debug!(
            "Uploaded {} bytes via staging buffer (offset={})",
            data_size,
            staging_offset
        );

        Ok(())
    }

    /// Compute SHA-256 hash of tensor data
    fn compute_hash<T: Copy>(data: &[T]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        hasher.update(bytes);
        hasher.finalize().into()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let content_map = self.content_map.lock().unwrap();
        let arena_stats = self.device_arena.stats();

        PoolStats {
            arena: arena_stats,
            cached_tensors: content_map.len(),
        }
    }

    /// Clear the content address cache (does not free GPU memory)
    pub fn clear_cache(&self) {
        let mut content_map = self.content_map.lock().unwrap();
        content_map.clear();
        log::info!("Cleared tensor cache");
    }
}

impl Drop for TensorPool {
    fn drop(&mut self) {
        log::info!("Destroying TensorPool");
        unsafe {
            self.device.destroy_command_pool(self.transfer_pool, None);
        }
    }
}

/// Tensor pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub arena: ArenaStats,
    pub cached_tensors: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_granularity() {
        // 100 bytes should round up to 256
        let size = 100u64;
        let aligned = ((size + ALLOCATION_GRANULARITY - 1) / ALLOCATION_GRANULARITY)
            * ALLOCATION_GRANULARITY;
        assert_eq!(aligned, 256);

        // 256 bytes should stay 256
        let size = 256u64;
        let aligned = ((size + ALLOCATION_GRANULARITY - 1) / ALLOCATION_GRANULARITY)
            * ALLOCATION_GRANULARITY;
        assert_eq!(aligned, 256);

        // 257 bytes should round up to 512
        let size = 257u64;
        let aligned = ((size + ALLOCATION_GRANULARITY - 1) / ALLOCATION_GRANULARITY)
            * ALLOCATION_GRANULARITY;
        assert_eq!(aligned, 512);
    }

    #[test]
    fn test_hash_consistency() {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data3 = vec![1.0f32, 2.0, 3.0, 5.0]; // Different

        let hash1 = TensorPool::compute_hash(&data1);
        let hash2 = TensorPool::compute_hash(&data2);
        let hash3 = TensorPool::compute_hash(&data3);

        assert_eq!(hash1, hash2, "Identical data should have same hash");
        assert_ne!(hash1, hash3, "Different data should have different hash");
    }
}
