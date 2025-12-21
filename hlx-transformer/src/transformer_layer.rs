//! Transformer Layer
//!
//! Pre-LN transformer block:
//! x = x + Attention(LayerNorm(x))
//! x = x + FFN(LayerNorm(x))

use std::sync::Arc;
use ash::vk;

use crate::device::Device;
use crate::error::VulkanErrorKind;
use crate::tensor::Tensor;
use crate::gemm_kernel::LinearLayer;
use crate::attention_kernel::{AttentionConfig, MultiHeadAttention};
use crate::transformer_config::{xavier_uniform, zeros, ones};

// =============================================================================
// LAYER NORM
// =============================================================================

/// Layer normalization parameters
pub struct LayerNorm {
    /// Scale parameter (gamma)
    pub gamma: Tensor,
    /// Shift parameter (beta)
    pub beta: Tensor,
    /// Gradients
    pub gamma_grad: Tensor,
    pub beta_grad: Tensor,
    /// Dimension
    pub d_model: u32,
    /// Epsilon
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d_model: u32, eps: f32, device: Arc<Device>) -> Result<Self, VulkanErrorKind> {
        Ok(Self {
            gamma: ones(&[d_model], device.clone())?,
            beta: zeros(&[d_model], device.clone())?,
            gamma_grad: zeros(&[d_model], device.clone())?,
            beta_grad: zeros(&[d_model], device)?,
            d_model,
            eps,
        })
    }
    
    pub fn param_count(&self) -> usize {
        2 * self.d_model as usize
    }
}

// =============================================================================
// FEEDFORWARD NETWORK
// =============================================================================

/// Feedforward network: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
pub struct FeedForward {
    /// First linear: d_model -> ffn_dim
    pub linear1: LinearLayer,
    /// Second linear: ffn_dim -> d_model
    pub linear2: LinearLayer,
    /// Intermediate buffer for GELU input
    pub intermediate: Tensor,
    /// Intermediate buffer for GELU output
    pub intermediate_gelu: Tensor,
}

impl FeedForward {
    pub fn new(
        d_model: u32,
        ffn_dim: u32,
        batch_size: u32,
        seq_len: u32,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let linear1 = LinearLayer::new(d_model, ffn_dim, true, device.clone())?;
        let linear2 = LinearLayer::new(ffn_dim, d_model, true, device.clone())?;
        
        let intermediate_shape = &[batch_size * seq_len, ffn_dim];
        let intermediate = Tensor::zeros(intermediate_shape, crate::tensor::DType::F32, device.clone())?;
        let intermediate_gelu = Tensor::zeros(intermediate_shape, crate::tensor::DType::F32, device)?;
        
        Ok(Self {
            linear1,
            linear2,
            intermediate,
            intermediate_gelu,
        })
    }
    
    pub fn param_count(&self) -> usize {
        self.linear1.param_count() + self.linear2.param_count()
    }
}

// =============================================================================
// TRANSFORMER LAYER
// =============================================================================

/// Single transformer layer (Pre-LN architecture)
///
/// Forward pass:
/// 1. x_norm1 = LayerNorm(x)
/// 2. attn_out = MultiHeadAttention(x_norm1)
/// 3. x = x + attn_out  (residual)
/// 4. x_norm2 = LayerNorm(x)
/// 5. ffn_out = FFN(x_norm2)
/// 6. x = x + ffn_out  (residual)
pub struct TransformerLayer {
    /// Layer index
    pub layer_idx: u32,
    /// Pre-attention layer norm
    pub norm1: LayerNorm,
    /// Multi-head attention
    pub attention: MultiHeadAttention,
    /// Pre-FFN layer norm
    pub norm2: LayerNorm,
    /// Feedforward network
    pub ffn: FeedForward,
    /// Intermediate buffers
    pub x_norm1: Tensor,  // After first layer norm
    pub attn_out: Tensor, // Attention output
    pub x_residual1: Tensor, // After first residual
    pub x_norm2: Tensor,  // After second layer norm
    pub ffn_out: Tensor,  // FFN output
}

impl TransformerLayer {
    pub fn new(
        layer_idx: u32,
        d_model: u32,
        num_heads: u32,
        ffn_dim: u32,
        max_seq_len: u32,
        batch_size: u32,
        eps: f32,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        let attention_config = AttentionConfig::new(d_model, num_heads, max_seq_len, true);
        
        let norm1 = LayerNorm::new(d_model, eps, device.clone())?;
        let attention = MultiHeadAttention::new(attention_config, batch_size, device.clone())?;
        let norm2 = LayerNorm::new(d_model, eps, device.clone())?;
        let ffn = FeedForward::new(d_model, ffn_dim, batch_size, max_seq_len, device.clone())?;
        
        // Intermediate buffers
        let buffer_shape = &[batch_size * max_seq_len, d_model];
        let x_norm1 = Tensor::zeros(buffer_shape, crate::tensor::DType::F32, device.clone())?;
        let attn_out = Tensor::zeros(buffer_shape, crate::tensor::DType::F32, device.clone())?;
        let x_residual1 = Tensor::zeros(buffer_shape, crate::tensor::DType::F32, device.clone())?;
        let x_norm2 = Tensor::zeros(buffer_shape, crate::tensor::DType::F32, device.clone())?;
        let ffn_out = Tensor::zeros(buffer_shape, crate::tensor::DType::F32, device)?;
        
        Ok(Self {
            layer_idx,
            norm1,
            attention,
            norm2,
            ffn,
            x_norm1,
            attn_out,
            x_residual1,
            x_norm2,
            ffn_out,
        })
    }
    
    pub fn param_count(&self) -> usize {
        self.norm1.param_count()
            + self.attention.param_count()
            + self.norm2.param_count()
            + self.ffn.param_count()
    }
}

// =============================================================================
// TRANSFORMER MODEL
// =============================================================================

/// Complete transformer model
pub struct TransformerModel {
    /// Model configuration
    pub config: crate::transformer_config::TransformerConfig,
    
    /// Token embedding table
    pub token_embedding: Tensor,
    pub token_embedding_grad: Tensor,
    
    /// Positional embedding table
    pub pos_embedding: Tensor,
    pub pos_embedding_grad: Tensor,
    
    /// Transformer layers
    pub layers: Vec<TransformerLayer>,
    
    /// Final layer norm
    pub final_norm: LayerNorm,
    
    /// Output projection (d_model -> vocab_size)
    pub output_projection: LinearLayer,
    
    /// Device reference
    pub device: Arc<Device>,
}

impl TransformerModel {
    pub fn new(
        config: crate::transformer_config::TransformerConfig,
        batch_size: u32,
        device: Arc<Device>,
    ) -> Result<Self, VulkanErrorKind> {
        config.validate().map_err(|e| {
            VulkanErrorKind::Other(e)
        })?;
        
        // Token embeddings
        let token_embedding = xavier_uniform(
            &[config.vocab_size, config.d_model],
            device.clone(),
        )?;
        let token_embedding_grad = zeros(
            &[config.vocab_size, config.d_model],
            device.clone(),
        )?;
        
        // Positional embeddings
        let pos_embedding = xavier_uniform(
            &[config.max_seq_len, config.d_model],
            device.clone(),
        )?;
        let pos_embedding_grad = zeros(
            &[config.max_seq_len, config.d_model],
            device.clone(),
        )?;
        
        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            layers.push(TransformerLayer::new(
                i,
                config.d_model,
                config.num_heads,
                config.ffn_dim,
                config.max_seq_len,
                batch_size,
                config.layer_norm_eps,
                device.clone(),
            )?);
        }
        
        // Final layer norm
        let final_norm = LayerNorm::new(config.d_model, config.layer_norm_eps, device.clone())?;
        
        // Output projection
        let output_projection = LinearLayer::new(
            config.d_model,
            config.vocab_size,
            false,  // No bias for output projection
            device.clone(),
        )?;
        
        Ok(Self {
            config,
            token_embedding,
            token_embedding_grad,
            pos_embedding,
            pos_embedding_grad,
            layers,
            final_norm,
            output_projection,
            device,
        })
    }
    
    /// Returns total parameter count
    pub fn param_count(&self) -> usize {
        let embedding = self.token_embedding.numel() + self.pos_embedding.numel();
        let layers: usize = self.layers.iter().map(|l| l.param_count()).sum();
        let final_norm = self.final_norm.param_count();
        let output = self.output_projection.param_count();
        
        embedding + layers + final_norm + output
    }
    
    /// Prints model summary
    pub fn summary(&self) {
        println!("=== Transformer Model Summary ===");
        println!("Config:");
        println!("  vocab_size:   {}", self.config.vocab_size);
        println!("  d_model:      {}", self.config.d_model);
        println!("  num_layers:   {}", self.config.num_layers);
        println!("  num_heads:    {}", self.config.num_heads);
        println!("  ffn_dim:      {}", self.config.ffn_dim);
        println!("  max_seq_len:  {}", self.config.max_seq_len);
        println!("");
        println!("Parameters:");
        println!("  Token embedding: {} ({:.2}M)", 
            self.token_embedding.numel(),
            self.token_embedding.numel() as f64 / 1e6);
        println!("  Pos embedding:   {} ({:.2}M)", 
            self.pos_embedding.numel(),
            self.pos_embedding.numel() as f64 / 1e6);
        println!("  Layers:          {} ({:.2}M per layer)",
            self.layers.len(),
            self.layers[0].param_count() as f64 / 1e6);
        println!("  Output proj:     {} ({:.2}M)",
            self.output_projection.param_count(),
            self.output_projection.param_count() as f64 / 1e6);
        println!("");
        println!("  TOTAL: {} ({:.2}M)",
            self.param_count(),
            self.param_count() as f64 / 1e6);
        println!("================================");
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer_config::TransformerConfig;
    
    #[test]
    fn test_layer_param_count() {
        // Manual calculation for tiny config
        let d_model = 256u32;
        let num_heads = 4u32;
        let ffn_dim = 1024u32;
        
        // LayerNorm: 2 * d_model = 512
        let ln_params = 2 * d_model as usize;
        
        // Attention: 4 * d_model^2 = 4 * 65536 = 262144
        let attn_params = 4 * (d_model * d_model) as usize;
        
        // FFN: d_model * ffn_dim + ffn_dim + ffn_dim * d_model + d_model
        // = 256 * 1024 + 1024 + 1024 * 256 + 256
        // = 262144 + 1024 + 262144 + 256 = 525568
        let ffn_params = (d_model * ffn_dim + ffn_dim + ffn_dim * d_model + d_model) as usize;
        
        let layer_params = 2 * ln_params + attn_params + ffn_params;
        
        println!("Expected layer params: {}", layer_params);
        // Should be around 789,760
    }
}
