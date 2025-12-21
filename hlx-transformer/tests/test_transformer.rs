//! Integration tests for HLX Transformer
//!
//! These tests require a Vulkan-capable GPU to run.
//! Generate reference data first: python test_data/generate_references.py

use std::fs;

// =============================================================================
// TEST UTILITIES
// =============================================================================

/// Maximum absolute error between two float vectors
fn max_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Check if two vectors are approximately equal
fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    max_error(a, b) < tol
}

// =============================================================================
// UNIT TESTS (CPU-only, no Vulkan required)
// =============================================================================

#[test]
fn test_tensor_strides() {
    // Shape [2, 3, 4] should have strides [12, 4, 1]
    let shape = vec![2u32, 3, 4];
    let mut strides = vec![1u32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    assert_eq!(strides, vec![12, 4, 1]);
}

#[test]
fn test_matmul_shape_validation() {
    // (M, K) × (K, N) → (M, N)
    let a_shape = vec![512u32, 256];
    let b_shape = vec![256u32, 512];
    
    let a_k = a_shape[a_shape.len() - 1];
    let b_k = b_shape[b_shape.len() - 2];
    
    assert_eq!(a_k, b_k, "Inner dimensions must match");
    
    let out_shape = vec![a_shape[0], b_shape[1]];
    assert_eq!(out_shape, vec![512, 512]);
}

#[test]
fn test_gelu_reference() {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi = 0.7978845608028654f32;
    let gelu_coef = 0.044715f32;
    
    let gelu = |x: f32| {
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + gelu_coef * x3);
        x * 0.5 * (1.0 + inner.tanh())
    };
    
    // Test values
    assert!((gelu(0.0) - 0.0).abs() < 1e-6);
    assert!((gelu(1.0) - 0.8413).abs() < 1e-3);
    assert!((gelu(-1.0) - (-0.1587)).abs() < 1e-3);
}

#[test]
fn test_softmax_reference() {
    // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
    let x = vec![1.0f32, 2.0, 3.0];
    let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = x.iter().map(|v| (v - max_x).exp()).sum();
    let softmax: Vec<f32> = x.iter().map(|v| (v - max_x).exp() / exp_sum).collect();
    
    assert!((softmax[0] - 0.0900).abs() < 0.001);
    assert!((softmax[1] - 0.2447).abs() < 0.001);
    assert!((softmax[2] - 0.6652).abs() < 0.001);
    
    // Sum should be 1
    let sum: f32 = softmax.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_cross_entropy_reference() {
    // CE(softmax([1,2,3]), target=2) = -log(0.6652) ≈ 0.4076
    let logits = vec![1.0f32, 2.0, 3.0];
    let target = 2usize;
    
    let max_x = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|v| (v - max_x).exp()).sum();
    let log_softmax: Vec<f32> = logits.iter()
        .map(|v| v - max_x - exp_sum.ln())
        .collect();
    
    let loss = -log_softmax[target];
    assert!((loss - 0.4076).abs() < 0.001);
}

#[test]
fn test_adam_reference() {
    // Single step of Adam
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let lr = 0.001f32;
    let eps = 1e-8f32;
    
    let param = 1.0f32;
    let grad = 0.1f32;
    let m = 0.0f32;
    let v = 0.0f32;
    let t = 1;
    
    // Update moments
    let m_new = beta1 * m + (1.0 - beta1) * grad;
    let v_new = beta2 * v + (1.0 - beta2) * grad * grad;
    
    // Bias correction
    let m_hat = m_new / (1.0 - beta1.powi(t));
    let v_hat = v_new / (1.0 - beta2.powi(t));
    
    // Update
    let param_new = param - lr * m_hat / (v_hat.sqrt() + eps);
    
    // Should be slightly less than 1.0
    assert!(param_new < param);
    assert!((param_new - 0.999).abs() < 0.001);
}

#[test]
fn test_layernorm_reference() {
    // LayerNorm([1, 2, 3]) with gamma=1, beta=0
    let x = vec![1.0f32, 2.0, 3.0];
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std = (var + 1e-5).sqrt();
    
    let normalized: Vec<f32> = x.iter()
        .map(|v| (v - mean) / std)
        .collect();
    
    // Mean should be ~0, std should be ~1
    let norm_mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
    let norm_var: f32 = normalized.iter().map(|v| (v - norm_mean).powi(2)).sum::<f32>() / normalized.len() as f32;
    
    assert!(norm_mean.abs() < 1e-5);
    assert!((norm_var - 1.0).abs() < 0.1);  // Allow some tolerance
}

#[test]
fn test_config_param_count() {
    // Tiny config should have ~10M params
    // vocab=256, d_model=256, layers=4, heads=4, ffn=1024, seq=128
    
    let vocab_size = 256u32;
    let d_model = 256u32;
    let num_layers = 4u32;
    let num_heads = 4u32;
    let ffn_dim = 1024u32;
    let max_seq_len = 128u32;
    
    // Embeddings
    let embed_params = vocab_size * d_model + max_seq_len * d_model;
    
    // Per layer
    let ln_params = 2 * d_model;  // gamma, beta
    let attn_params = 4 * d_model * d_model;  // Q, K, V, O
    let ffn_params = d_model * ffn_dim + ffn_dim + ffn_dim * d_model + d_model;
    let layer_params = 2 * ln_params + attn_params + ffn_params;
    
    // Output
    let output_params = d_model * vocab_size;
    
    let total = embed_params + num_layers * layer_params + ln_params + output_params;
    
    println!("Estimated params: {} ({:.2}M)", total, total as f64 / 1e6);
    assert!(total > 5_000_000);  // >5M
    assert!(total < 20_000_000); // <20M
}

// =============================================================================
// GPU TESTS (require Vulkan)
// =============================================================================

#[test]
#[ignore]  // Run with: cargo test --ignored
fn test_gemm_gpu() {
    // TODO: Implement once Vulkan infrastructure is integrated
    // This test requires:
    // 1. Vulkan device initialization
    // 2. Shader loading
    // 3. Buffer creation
    // 4. Compute dispatch
    // 5. Result verification
}

#[test]
#[ignore]
fn test_determinism_gpu() {
    // TODO: Run same computation 3 times, verify bit-identical results
}

#[test]
#[ignore]
fn test_training_loop_gpu() {
    // TODO: Run 10 training steps, verify loss decreases
}
