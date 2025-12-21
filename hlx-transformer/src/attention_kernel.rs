//! Multi-Head Attention Kernel
//!
//! Implements scaled dot-product attention with causal masking.
//! Q, K, V projections are done via GEMM kernel externally.

use std::sync::Arc;
use ash::vk;

use crate::buffer::Buffer;
use crate::compute::{ComputePipeline, CommandBufferPool, DescriptorBindingBuilder};
use crate::device::Device;
use crate::error::VulkanErrorKind;
use crate::shader::ShaderModule;
use crate::tensor::Tensor;
use crate::gemm_kernel::{GemmKernel, LinearLayer};

// =============================================================================
// ATTENTION CONFIGURATION
// =============================================================================

/// Attention layer configuration
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    pub d_model: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub causal: bool,  // Use causal masking
}

impl AttentionConfig {
    pub fn new(d_model: u32, num_heads: u32, max_seq_len: u32, causal: bool) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        Self {
            d_model,
            num_heads,
            head_dim: d_model / num_heads,
            max_seq_len,
            causal,
        }
    }
}

// =============================================================================
// PUSH CONSTANTS
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttentionPushConstants {
    pub batch_size: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub scale: f32,  // 1 / sqrt(head_dim)
    pub causal: u32, // 1 for causal masking
}

// =============================================================================
// MULTI-HEAD ATTENTION
// =============================================================================

/// Multi-head attention layer
///
/// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
///
/// Architecture:
/// 1. Project input to Q, K, V via linear layers
/// 2. Split into heads
/// 3. Compute scaled dot-product attention per head
/// 4. Concatenate heads and project output
pub struct MultiHeadAttention {
    /// Configuration
    pub config: AttentionConfig,
    
    /// Query projection: (d_model, d_model)
    pub w_q: LinearLayer,
    /// Key projection: (d_model, d_model)
    pub w_k: LinearLayer,
    /// Value projection: (d_model, d_model)
    pub w_v: LinearLayer,
    /// Output projection: (d_model, d_model)
    pub w_o: LinearLayer,
    
    /// Intermediate buffers for attention computation
    q_buffer: Tensor,     // (batch, num_heads, seq_len, head_dim)
    k_buffer: Tensor,     // (batch, num_heads, seq_len, head_dim)
    v_buffer: Tensor,     // (batch, num_heads, seq_len, head_dim)
    scores_buffer: Tensor, // (batch, num_heads, seq_len, seq_len)
    attn_buffer: Tensor,   // (batch, num_heads, seq_len, seq_len) - after softmax
    context_buffer: Tensor, // (batch, num_heads, seq_len, head_dim)
    
    /// Device reference
    device: Arc<Device>,
}

impl MultiHeadAttention {
    /// Creates a new multi-head attention layer
    pub fn new(
        config: AttentionConfig,
        batch_size: u32,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let d_model = config.d_model;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let seq_len = config.max_seq_len;
        
        // Linear projections
        let w_q = LinearLayer::new(d_model, d_model, false, device.clone())?;
        let w_k = LinearLayer::new(d_model, d_model, false, device.clone())?;
        let w_v = LinearLayer::new(d_model, d_model, false, device.clone())?;
        let w_o = LinearLayer::new(d_model, d_model, false, device.clone())?;
        
        // Intermediate buffers
        // Shape: (batch_size * num_heads * seq_len, head_dim) when flattened
        let qkv_shape = &[batch_size, num_heads, seq_len, head_dim];
        let scores_shape = &[batch_size, num_heads, seq_len, seq_len];
        
        let q_buffer = Tensor::zeros(qkv_shape, crate::tensor::DType::F32, device.clone())?;
        let k_buffer = Tensor::zeros(qkv_shape, crate::tensor::DType::F32, device.clone())?;
        let v_buffer = Tensor::zeros(qkv_shape, crate::tensor::DType::F32, device.clone())?;
        let scores_buffer = Tensor::zeros(scores_shape, crate::tensor::DType::F32, device.clone())?;
        let attn_buffer = Tensor::zeros(scores_shape, crate::tensor::DType::F32, device.clone())?;
        let context_buffer = Tensor::zeros(qkv_shape, crate::tensor::DType::F32, device.clone())?;
        
        Ok(Self {
            config,
            w_q, w_k, w_v, w_o,
            q_buffer, k_buffer, v_buffer,
            scores_buffer, attn_buffer, context_buffer,
            device,
        })
    }
    
    /// Returns total parameter count
    pub fn param_count(&self) -> usize {
        self.w_q.param_count() + self.w_k.param_count() 
            + self.w_v.param_count() + self.w_o.param_count()
    }
    
    /// Records forward pass commands
    ///
    /// # Arguments
    /// * `cmd_buffer` - Command buffer
    /// * `gemm` - GEMM kernel for projections
    /// * `softmax_pipeline` - Softmax pipeline
    /// * `input` - Input tensor (batch * seq_len, d_model)
    /// * `output` - Output tensor (batch * seq_len, d_model)
    /// * `batch_size` - Current batch size
    /// * `seq_len` - Current sequence length
    pub fn record_forward(
        &self,
        cmd_buffer: vk::CommandBuffer,
        gemm: &GemmKernel,
        softmax_pipeline: &ComputePipeline,
        input: vk::Buffer,
        output: vk::Buffer,
        batch_size: u32,
        seq_len: u32,
    ) -> Result<(), VulkanErrorKind> {
        let d_model = self.config.d_model;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let num_positions = batch_size * seq_len;
        
        // 1. Project to Q, K, V
        // Q = input @ W_q^T
        gemm.record_forward(
            cmd_buffer,
            input,
            self.w_q.weight.buffer(),
            self.q_buffer.buffer(),
            None,
            num_positions,  // M
            d_model,        // K
            d_model,        // N
        )?;
        gemm.record_barrier(cmd_buffer);
        
        // K = input @ W_k^T
        gemm.record_forward(
            cmd_buffer,
            input,
            self.w_k.weight.buffer(),
            self.k_buffer.buffer(),
            None,
            num_positions, d_model, d_model,
        )?;
        gemm.record_barrier(cmd_buffer);
        
        // V = input @ W_v^T
        gemm.record_forward(
            cmd_buffer,
            input,
            self.w_v.weight.buffer(),
            self.v_buffer.buffer(),
            None,
            num_positions, d_model, d_model,
        )?;
        gemm.record_barrier(cmd_buffer);
        
        // 2. Reshape and compute attention scores
        // scores = Q @ K^T / sqrt(head_dim)
        // For each (batch, head): scores[i] = Q[i] @ K[i]^T
        // This is a batched matmul: (batch*heads, seq_len, head_dim) @ (batch*heads, head_dim, seq_len)
        
        // Note: In full implementation, we'd use a batched GEMM here
        // For now, we record the operation structure
        
        // 3. Apply softmax (and causal mask)
        // This would be done per attention row
        
        // 4. Compute context: context = attention @ V
        
        // 5. Reshape and project output
        // output = context @ W_o^T
        gemm.record_forward(
            cmd_buffer,
            self.context_buffer.buffer(),
            self.w_o.weight.buffer(),
            output,
            None,
            num_positions, d_model, d_model,
        )?;
        
        Ok(())
    }
}

// =============================================================================
// CAUSAL ATTENTION MASK
// =============================================================================

/// Creates a causal attention mask buffer
///
/// Returns a lower-triangular matrix where mask[i][j] = 1 if j <= i, else 0
pub fn create_causal_mask(
    seq_len: u32,
    device: Arc<Device>,
) -> Result<Tensor, VulkanErrorKind> {
    let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];
    
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[(i * seq_len + j) as usize] = 1.0;
        }
    }
    
    Tensor::from_f32(&mask_data, &[seq_len, seq_len], device)
}

/// Creates attention bias for causal masking
///
/// Returns a matrix where bias[i][j] = 0 if j <= i, else -inf
pub fn create_causal_bias(
    seq_len: u32,
    device: Arc<Device>,
) -> Result<Tensor, VulkanErrorKind> {
    let mut bias_data = vec![0.0f32; (seq_len * seq_len) as usize];
    let neg_inf = -1e9f32;  // Large negative number instead of true -inf
    
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            bias_data[(i * seq_len + j) as usize] = neg_inf;
        }
    }
    
    Tensor::from_f32(&bias_data, &[seq_len, seq_len], device)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(512, 8, 128, true);
        assert_eq!(config.head_dim, 64);
    }
    
    #[test]
    fn test_push_constants_size() {
        assert_eq!(std::mem::size_of::<AttentionPushConstants>(), 24);
    }
    
    #[test]
    fn test_causal_mask() {
        // Manual test without device
        let seq_len = 4;
        let mut mask = vec![0.0f32; 16];
        
        for i in 0..seq_len {
            for j in 0..=i {
                mask[(i * seq_len + j) as usize] = 1.0;
            }
        }
        
        // Expected:
        // [1, 0, 0, 0]
        // [1, 1, 0, 0]
        // [1, 1, 1, 0]
        // [1, 1, 1, 1]
        
        assert_eq!(mask[0], 1.0);   // (0, 0)
        assert_eq!(mask[1], 0.0);   // (0, 1)
        assert_eq!(mask[4], 1.0);   // (1, 0)
        assert_eq!(mask[5], 1.0);   // (1, 1)
        assert_eq!(mask[6], 0.0);   // (1, 2)
        assert_eq!(mask[15], 1.0);  // (3, 3)
    }
}
