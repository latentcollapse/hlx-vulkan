# Testing Guide

## Test Strategy

### 1. Unit Tests (Per Kernel)
Test each shader against a reference implementation.

### 2. Integration Tests
Test kernel composition (e.g., forward + backward).

### 3. Determinism Tests
Verify bit-identical results across runs.

### 4. End-to-End Tests
Overfit on single example, validate loss → 0.

---

## Unit Tests

### GEMM Test

```python
# reference_gemm.py - Generate test vectors
import numpy as np

np.random.seed(42)
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(256, 512).astype(np.float32)
C = A @ B

np.save('test_data/gemm_a.npy', A)
np.save('test_data/gemm_b.npy', B)
np.save('test_data/gemm_c_expected.npy', C)
```

```rust
#[test]
fn test_gemm_correctness() {
    let a = load_npy("test_data/gemm_a.npy")?;
    let b = load_npy("test_data/gemm_b.npy")?;
    let expected = load_npy("test_data/gemm_c_expected.npy")?;
    
    let c = gemm_kernel.forward(&a, &b)?;
    
    let max_error = c.iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(max_error < 1e-4, "Max error: {}", max_error);
}
```

### LayerNorm Test

```python
# reference_layernorm.py
import torch
import torch.nn as nn

x = torch.randn(4, 128, 256)  # batch=4, seq=128, d_model=256
ln = nn.LayerNorm(256)

# Use default gamma=1, beta=0
y = ln(x)

torch.save({'x': x, 'y': y}, 'test_data/layernorm.pt')
```

```rust
#[test]
fn test_layernorm_correctness() {
    let data = load_torch("test_data/layernorm.pt");
    let x = data.get("x");
    let expected = data.get("y");
    
    let gamma = vec![1.0f32; 256];
    let beta = vec![0.0f32; 256];
    
    let y = layernorm_kernel.forward(&x, &gamma, &beta, 1e-5)?;
    
    let max_error = compute_max_error(&y, &expected);
    assert!(max_error < 1e-5, "Max error: {}", max_error);
}
```

### Softmax Test

```python
# reference_softmax.py
import torch
import torch.nn.functional as F

# Test numerical stability with large values
x = torch.randn(16, 128) * 100  # Large values
y = F.softmax(x, dim=-1)

# Should sum to 1
assert torch.allclose(y.sum(dim=-1), torch.ones(16))

torch.save({'x': x, 'y': y}, 'test_data/softmax.pt')
```

```rust
#[test]
fn test_softmax_numerical_stability() {
    let data = load_torch("test_data/softmax.pt");
    let x = data.get("x");
    let expected = data.get("y");
    
    let y = softmax_kernel.forward(&x)?;
    
    // Check row sums
    for row in 0..16 {
        let row_sum: f32 = y[row * 128..(row + 1) * 128].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", row, row_sum);
    }
    
    let max_error = compute_max_error(&y, &expected);
    assert!(max_error < 1e-5);
}
```

### GELU Test

```python
# reference_gelu.py
import torch
import torch.nn.functional as F

x = torch.randn(1024)
y = F.gelu(x)

torch.save({'x': x, 'y': y}, 'test_data/gelu.pt')
```

### Cross-Entropy Test

```python
# reference_cross_entropy.py
import torch
import torch.nn.functional as F

logits = torch.randn(4, 128, 256)  # batch=4, seq=128, vocab=256
targets = torch.randint(0, 256, (4, 128))

# Per-position loss
loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1), reduction='none')
loss = loss.view(4, 128)

# Mean loss
mean_loss = loss.mean()

torch.save({
    'logits': logits,
    'targets': targets,
    'loss': loss,
    'mean_loss': mean_loss,
}, 'test_data/cross_entropy.pt')
```

---

## Gradient Tests

### Numerical Gradient Check

```rust
fn numerical_gradient<F>(f: F, x: &[f32], epsilon: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    
    for i in 0..x.len() {
        x_plus[i] = x[i] + epsilon;
        x_minus[i] = x[i] - epsilon;
        
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
        
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    
    grad
}

#[test]
fn test_gemm_gradient() {
    let a = random_tensor(&[32, 16]);
    let b = random_tensor(&[16, 32]);
    
    // Analytic gradient
    let c = gemm.forward(&a, &b);
    let dc = ones_like(&c);  // Gradient of sum
    let da_analytic = gemm.backward_a(&dc, &b);
    
    // Numerical gradient
    let da_numerical = numerical_gradient(
        |a_flat| {
            let a_tensor = Tensor::from_slice(a_flat, &[32, 16]);
            let c = gemm.forward(&a_tensor, &b);
            c.sum()
        },
        &a.to_vec(),
        1e-4,
    );
    
    let max_error = compute_max_error(&da_analytic, &da_numerical);
    assert!(max_error < 1e-3, "Gradient error: {}", max_error);
}
```

---

## Determinism Tests

### Bit-Exact Reproducibility

```rust
#[test]
fn test_full_forward_determinism() {
    let input = random_tensor_seeded(&[4, 128], 42);
    
    let mut results = Vec::new();
    
    for run in 0..5 {
        // Recreate everything from scratch
        let model = TransformerModel::new(config, device.clone())?;
        let output = model.forward(&input)?;
        results.push(output.to_vec()?);
    }
    
    // All results must be bit-identical
    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "Run {} differs from run 0", i
        );
    }
}
```

### Training Determinism

```rust
#[test]
fn test_training_determinism() {
    let mut final_losses = Vec::new();
    let mut final_weights = Vec::new();
    
    for run in 0..3 {
        // Train for 10 epochs with fixed seed
        let model = TransformerModel::new(config, device.clone())?;
        let optimizer = AdamOptimizer::new(&model)?;
        
        for epoch in 0..10 {
            let loss = train_epoch(&model, &optimizer, &batches)?;
        }
        
        final_losses.push(loss);
        final_weights.push(model.get_weights()?);
    }
    
    // All runs must produce identical results
    for i in 1..3 {
        assert_eq!(
            final_losses[0].to_bits(), 
            final_losses[i].to_bits(),
            "Loss differs at run {}", i
        );
        assert_eq!(
            final_weights[0],
            final_weights[i],
            "Weights differ at run {}", i
        );
    }
    
    println!("✓ Training is deterministic across {} runs", 3);
}
```

---

## End-to-End Tests

### Overfit Single Example

```rust
#[test]
fn test_overfit_single_example() {
    // Single training example
    let input = "Hello";
    let target = "⟨h e l l o⟩";
    
    let model = TransformerModel::new(TransformerConfig::micro(), device)?;
    let optimizer = AdamOptimizer::new(&model, AdamConfig {
        lr: 1e-3,  // Higher LR for faster convergence
        ..Default::default()
    })?;
    
    let tokens = tokenizer.encode(&format!("{}{}", input, target));
    
    let mut best_loss = f32::MAX;
    
    for step in 0..1000 {
        let loss = train_step(&model, &optimizer, &tokens)?;
        
        if loss < best_loss {
            best_loss = loss;
        }
        
        if loss < 0.01 {
            println!("Converged at step {} with loss {}", step, loss);
            break;
        }
    }
    
    assert!(best_loss < 0.1, "Failed to overfit. Final loss: {}", best_loss);
}
```

### Generate After Training

```rust
#[test]
fn test_generation_quality() {
    // Load trained checkpoint
    let model = TransformerModel::load("checkpoints/epoch_100.bin")?;
    
    let prompt = "Test";
    let generated = model.generate(&prompt, max_len: 50)?;
    
    // Should produce HLX-like output
    assert!(generated.contains("⟨"), "Missing HLX markers");
    assert!(generated.contains("⟩"), "Missing HLX markers");
    
    println!("Generated: {}", generated);
}
```

---

## Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_gemm_correctness

# With output
cargo test -- --nocapture

# Release mode (faster)
cargo test --release

# Determinism tests only
cargo test determinism

# Generate test data first
cd test_data
python generate_references.py
cd ..
cargo test
```

---

## Test Data Generation Script

```python
#!/usr/bin/env python3
# test_data/generate_references.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.makedirs('test_data', exist_ok=True)

print("Generating GEMM test data...")
np.random.seed(42)
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(256, 512).astype(np.float32)
C = A @ B
np.save('test_data/gemm_a.npy', A)
np.save('test_data/gemm_b.npy', B)
np.save('test_data/gemm_c.npy', C)

print("Generating LayerNorm test data...")
torch.manual_seed(42)
x = torch.randn(4, 128, 256)
ln = nn.LayerNorm(256)
y = ln(x)
torch.save({'x': x, 'y': y, 'gamma': ln.weight, 'beta': ln.bias}, 'test_data/layernorm.pt')

print("Generating Softmax test data...")
x = torch.randn(16, 128) * 100
y = F.softmax(x, dim=-1)
torch.save({'x': x, 'y': y}, 'test_data/softmax.pt')

print("Generating GELU test data...")
x = torch.randn(1024)
y = F.gelu(x)
torch.save({'x': x, 'y': y}, 'test_data/gelu.pt')

print("Generating Cross-Entropy test data...")
logits = torch.randn(4, 128, 256)
targets = torch.randint(0, 256, (4, 128))
loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1), reduction='none').view(4, 128)
torch.save({'logits': logits, 'targets': targets, 'loss': loss}, 'test_data/cross_entropy.pt')

print("Done! Test data saved to test_data/")
```

---

## Expected Test Results

| Test | Expected | Tolerance |
|------|----------|-----------|
| GEMM correctness | Max error < 1e-4 | FP32 precision |
| LayerNorm | Max error < 1e-5 | After eps |
| Softmax sum | = 1.0 | < 1e-5 |
| GELU | Max error < 1e-5 | |
| Cross-entropy | Max error < 1e-4 | |
| Determinism | Bit-exact | 0 |
| Overfit | Loss < 0.1 | 1000 steps |
