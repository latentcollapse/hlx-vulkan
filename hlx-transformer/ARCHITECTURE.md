# HLX Transformer Architecture

## Overview

This document describes the architecture of the HLX Transformer, a Vulkan-based transformer implementation designed for deterministic ML training.

## Design Principles

### 1. Determinism First

Every operation produces bit-identical results across runs:
- Fixed-order reductions (no cross-workgroup atomics)
- Explicit memory barriers
- Consistent floating-point ordering

### 2. Modular Kernels

Each operation is a standalone shader:
- Easy to test independently
- Can swap implementations without touching others
- Clear interface boundaries

### 3. Pre-LN Architecture

We use Pre-LayerNorm transformer blocks:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Benefits:
- More stable training (gradients flow through residual paths)
- No gradient explosion without warmup
- Standard in modern transformers (GPT-3, LLaMA)

## Component Breakdown

### GEMM (gemm.glsl, gemm_backward.glsl)

**Tiled matrix multiplication** with 16×16 blocking.

```
┌─────────────────────────────────────────┐
│  Shared Memory Tiles                    │
│  ┌─────────┐ ┌─────────┐               │
│  │ Tile A  │ │ Tile B  │               │
│  │ (16×16) │ │ (16×16) │               │
│  └─────────┘ └─────────┘               │
│       ↓          ↓                      │
│    Multiply & Accumulate                │
│       ↓                                 │
│  ┌─────────┐                           │
│  │ Tile C  │ → Write to global         │
│  └─────────┘                           │
└─────────────────────────────────────────┘
```

**Backward passes:**
- `dA = dC × B^T` (mode=0)
- `dB = A^T × dC` (mode=1)

### LayerNorm (layernorm_forward.glsl, layernorm_backward.glsl)

**Parallel reduction** for mean and variance:

```
Step 1: Each thread sums its elements → shared_sum[tid]
Step 2: Tree reduction in shared memory
Step 3: Thread 0 computes mean, variance
Step 4: All threads apply normalization
```

**Numerical stability:**
- Variance computed as `E[x²] - E[x]²`
- Epsilon = 1e-5 before sqrt

### Softmax (softmax_forward.glsl, softmax_backward.glsl)

**Three-pass algorithm** for numerical stability:

```
Pass 1: Find max (parallel reduction)
Pass 2: Compute sum of exp(x - max)
Pass 3: Normalize: exp(x - max) / sum
```

**Backward:**
```
d_input[i] = softmax[i] * (d_output[i] - dot(d_output, softmax))
```

### GELU (gelu_forward.glsl, gelu_backward.glsl)

**Approximation formula:**
```
GELU(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

Element-wise, highly parallel.

### Cross-Entropy Loss (cross_entropy_forward.glsl, cross_entropy_backward.glsl)

**Log-softmax trick** for stability:
```
log_prob[i] = logit[i] - max - log(sum(exp(logits - max)))
loss = -log_prob[target]
```

**Backward:**
```
d_logits = softmax - one_hot(target)
```

This elegant form is the foundation of language model training.

### Adam Optimizer (adam_update.glsl)

**Standard Adam** with bias correction:
```
m = β₁ * m + (1 - β₁) * grad
v = β₂ * v + (1 - β₂) * grad²
m_hat = m / (1 - β₁ᵗ)
v_hat = v / (1 - β₂ᵗ)
param -= lr * m_hat / (√v_hat + ε)
```

Hyperparameters:
- lr = 3e-4 (good default for transformers)
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

## Memory Layout

### Tensor Format

All tensors are **row-major** (C-contiguous):
- Shape: (batch, seq_len, d_model)
- Stride: [seq_len * d_model, d_model, 1]

### Buffer Types

| Buffer | Usage | Memory |
|--------|-------|--------|
| Weights | Read in forward, write in backward | DEVICE_LOCAL |
| Activations | Written in forward, read in backward | DEVICE_LOCAL |
| Gradients | Written in backward, read by optimizer | DEVICE_LOCAL |
| Staging | Per-workgroup gradient accumulation | DEVICE_LOCAL |
| Parameters | Uniform buffers | HOST_VISIBLE + DEVICE_LOCAL |

## Synchronization

### Barrier Strategy

```
Forward Pass          Backward Pass         Optimizer
    │                      │                    │
    ▼                      ▼                    ▼
[DISPATCH]            [DISPATCH]            [DISPATCH]
    │                      │                    │
[BARRIER]             [BARRIER]             [BARRIER]
SHADER_WRITE →        SHADER_WRITE →        SHADER_WRITE →
SHADER_READ           SHADER_READ           HOST_READ
    │                      │                    │
    ▼                      ▼                    ▼
Next layer            Prev layer            CPU reads
```

### Determinism Guarantees

1. **Within workgroup:** `barrier()` ensures all threads sync
2. **Cross workgroup:** Explicit `vkCmdPipelineBarrier`
3. **Reduction order:** Fixed iteration (0, 1, 2, ..., N-1)
4. **No atomics:** Per-workgroup staging eliminates races

## Model Configurations

### Tiny (~10M params)
```rust
TransformerConfig {
    vocab_size: 256,
    d_model: 256,
    num_layers: 4,
    num_heads: 4,
    ffn_dim: 1024,
    max_seq_len: 128,
}
```

### Small (~50M params)
```rust
TransformerConfig {
    vocab_size: 256,
    d_model: 512,
    num_layers: 6,
    num_heads: 8,
    ffn_dim: 2048,
    max_seq_len: 256,
}
```

### Medium (~100M params)
```rust
TransformerConfig {
    vocab_size: 256,
    d_model: 768,
    num_layers: 12,
    num_heads: 12,
    ffn_dim: 3072,
    max_seq_len: 512,
}
```

## Training Loop

```
for epoch in 1..=num_epochs:
    for batch in dataset:
        # Forward
        embeddings = embed(batch.tokens)
        for layer in model.layers:
            embeddings = layer.forward(embeddings)
        logits = output_projection(final_norm(embeddings))
        loss = cross_entropy(logits, batch.targets)
        
        # Backward
        grad_logits = cross_entropy_backward(logits, batch.targets)
        grad = output_projection.backward(grad_logits)
        grad = final_norm.backward(grad)
        for layer in reversed(model.layers):
            grad = layer.backward(grad)
        embed_backward(grad)
        
        # Update
        optimizer.step()
        
    # Checkpoint
    if epoch % 10 == 0:
        save_checkpoint(model, epoch)
```

## Integration Notes

### Existing HLX Components to Reuse

1. **ComputePipeline** - Pipeline management
2. **CommandBufferPool** - Command buffer reuse
3. **Buffer** - GPU memory management
4. **ShaderModule** - SPIR-V loading
5. **DescriptorBindingBuilder** - Descriptor layout

### What This Package Provides

1. **Shader implementations** - GLSL compute shaders
2. **Rust orchestration** - Kernel wrappers, model definition
3. **Configuration** - Model hyperparameters, optimizer settings
4. **Training utilities** - LR schedule, checkpointing

### Integration Steps

1. Copy shaders to your project's shader/ directory
2. Copy Rust modules to src/
3. Replace stub modules with your existing implementations
4. Build shaders with `./build_shaders.sh`
5. Run tests: `cargo test`

## Performance Characteristics

### Expected Throughput

| Operation | RTX 3090 | RTX 5060 (est) |
|-----------|----------|----------------|
| GEMM 512×512 | ~2 TFLOP/s | ~1.5 TFLOP/s |
| Softmax | ~500 GB/s | ~400 GB/s |
| LayerNorm | ~400 GB/s | ~300 GB/s |

### Memory Usage

| Model | Parameters | Activations | Total |
|-------|------------|-------------|-------|
| Tiny (10M) | 40 MB | ~100 MB | ~200 MB |
| Small (50M) | 200 MB | ~500 MB | ~1 GB |
| Medium (100M) | 400 MB | ~1 GB | ~2 GB |

## Future Work

- [ ] Flash Attention (memory efficient)
- [ ] Mixed Precision (FP16/BF16)
- [ ] Gradient Checkpointing
- [ ] Multi-GPU Support
- [ ] BPE Tokenization
