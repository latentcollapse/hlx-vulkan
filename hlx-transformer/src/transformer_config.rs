//! Transformer Model Configuration and Architecture
//!
//! Defines the transformer architecture for Vulkan ML training.

use std::sync::Arc;

use crate::device::Device;
use crate::error::VulkanErrorKind;
use crate::tensor::Tensor;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Transformer model configuration
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: u32,
    /// Model dimension (embedding size)
    pub d_model: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Dimension per attention head (usually d_model / num_heads)
    pub head_dim: u32,
    /// Feedforward network hidden dimension (usually 4 * d_model)
    pub ffn_dim: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Dropout probability (0.0 for deterministic training)
    pub dropout: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

impl TransformerConfig {
    /// Creates a tiny configuration (~10M parameters)
    /// Good for testing and validation
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,    // Character-level
            d_model: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 64,       // 256 / 4
            ffn_dim: 1024,      // 4 * 256
            max_seq_len: 128,
            dropout: 0.0,       // Deterministic
            layer_norm_eps: 1e-5,
        }
    }
    
    /// Creates a small configuration (~50M parameters)
    /// Good for serious training experiments
    pub fn small() -> Self {
        Self {
            vocab_size: 256,    // Character-level
            d_model: 512,
            num_layers: 6,
            num_heads: 8,
            head_dim: 64,       // 512 / 8
            ffn_dim: 2048,      // 4 * 512
            max_seq_len: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }
    
    /// Creates a medium configuration (~100M parameters)
    /// Comparable to GPT-2 small
    pub fn medium() -> Self {
        Self {
            vocab_size: 256,
            d_model: 768,
            num_layers: 12,
            num_heads: 12,
            head_dim: 64,       // 768 / 12
            ffn_dim: 3072,      // 4 * 768
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }
    
    /// Computes total parameter count
    pub fn param_count(&self) -> usize {
        let embedding = self.vocab_size as usize * self.d_model as usize;
        let pos_embedding = self.max_seq_len as usize * self.d_model as usize;
        
        // Per layer:
        // - LayerNorm 1: 2 * d_model (gamma, beta)
        // - Attention: 4 * d_model * d_model (Q, K, V, O projections)
        // - LayerNorm 2: 2 * d_model
        // - FFN: d_model * ffn_dim + ffn_dim + ffn_dim * d_model + d_model
        let ln_params = 2 * self.d_model as usize;
        let attention_params = 4 * self.d_model as usize * self.d_model as usize;
        let ffn_params = 2 * self.d_model as usize * self.ffn_dim as usize 
                       + self.ffn_dim as usize + self.d_model as usize;
        let layer_params = 2 * ln_params + attention_params + ffn_params;
        
        let output_proj = self.d_model as usize * self.vocab_size as usize;
        
        embedding + pos_embedding 
            + self.num_layers as usize * layer_params 
            + ln_params  // Final layer norm
            + output_proj
    }
    
    /// Validates configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.num_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.d_model, self.num_heads
            ));
        }
        
        if self.head_dim != self.d_model / self.num_heads {
            return Err(format!(
                "head_dim ({}) should equal d_model / num_heads ({})",
                self.head_dim, self.d_model / self.num_heads
            ));
        }
        
        Ok(())
    }
}

// =============================================================================
// PARAMETER INITIALIZATION
// =============================================================================

/// Initializes parameters with Xavier/Glorot uniform distribution
pub fn xavier_uniform(shape: &[u32], device: Arc<Device>) -> Result<Tensor, VulkanErrorKind> {
    let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { shape[0] };
    let fan_out = if shape.len() >= 2 { shape[shape.len() - 1] } else { shape[0] };
    
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    
    // Generate uniform random values in [-limit, limit]
    let numel: usize = shape.iter().product::<u32>() as usize;
    let mut data = Vec::with_capacity(numel);
    
    // Simple LCG PRNG for deterministic initialization
    let mut seed: u64 = 42;
    for _ in 0..numel {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (seed as f64) / (u64::MAX as f64);  // [0, 1]
        let val = (2.0 * u - 1.0) as f32 * limit;   // [-limit, limit]
        data.push(val);
    }
    
    Tensor::from_f32(&data, shape, device)
}

/// Initializes parameters with zeros
pub fn zeros(shape: &[u32], device: Arc<Device>) -> Result<Tensor, VulkanErrorKind> {
    Tensor::zeros(shape, crate::tensor::DType::F32, device)
}

/// Initializes parameters with ones
pub fn ones(shape: &[u32], device: Arc<Device>) -> Result<Tensor, VulkanErrorKind> {
    let numel: usize = shape.iter().product::<u32>() as usize;
    let data = vec![1.0f32; numel];
    Tensor::from_f32(&data, shape, device)
}

// =============================================================================
// ADAM OPTIMIZER STATE
// =============================================================================

/// Adam optimizer hyperparameters
#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,      // Good default for transformers
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Per-parameter optimizer state
pub struct AdamState {
    /// First moment (momentum)
    pub m: Tensor,
    /// Second moment (adaptive learning rate)
    pub v: Tensor,
    /// Current timestep
    pub t: u32,
}

impl AdamState {
    pub fn new(param_shape: &[u32], device: Arc<Device>) -> Result<Self, VulkanErrorKind> {
        Ok(Self {
            m: Tensor::zeros(param_shape, crate::tensor::DType::F32, device.clone())?,
            v: Tensor::zeros(param_shape, crate::tensor::DType::F32, device)?,
            t: 0,
        })
    }
}

// =============================================================================
// LEARNING RATE SCHEDULE
// =============================================================================

/// Learning rate schedule with warmup and cosine decay
pub struct LRSchedule {
    pub base_lr: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
    pub min_lr: f32,
}

impl LRSchedule {
    pub fn new(base_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            min_lr: base_lr * 0.1,
        }
    }
    
    /// Computes learning rate for given step
    pub fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32 
                         / (self.total_steps - self.warmup_steps) as f32;
            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * decay
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_param_count() {
        let tiny = TransformerConfig::tiny();
        let count = tiny.param_count();
        println!("Tiny config: {} parameters ({:.2}M)", count, count as f64 / 1e6);
        assert!(count > 1_000_000);  // Should be > 1M
        assert!(count < 50_000_000); // Should be < 50M for tiny
        
        let small = TransformerConfig::small();
        let count = small.param_count();
        println!("Small config: {} parameters ({:.2}M)", count, count as f64 / 1e6);
        
        let medium = TransformerConfig::medium();
        let count = medium.param_count();
        println!("Medium config: {} parameters ({:.2}M)", count, count as f64 / 1e6);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = TransformerConfig::tiny();
        assert!(config.validate().is_ok());
        
        config.num_heads = 5;  // Invalid: 256 not divisible by 5
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_lr_schedule() {
        let schedule = LRSchedule::new(3e-4, 100, 1000);
        
        // Warmup: should increase
        assert!(schedule.get_lr(50) < schedule.get_lr(100));
        
        // After warmup: should decrease
        assert!(schedule.get_lr(200) > schedule.get_lr(800));
        
        // At start: should be near zero
        assert!(schedule.get_lr(0) < 1e-5);
        
        // At warmup end: should be base_lr
        assert!((schedule.get_lr(100) - 3e-4).abs() < 1e-6);
    }
}
