# HLX Transformer

A complete Vulkan-based transformer implementation for deterministic ML training.

## Features

- **Deterministic**: Bit-identical results across runs (no atomics, fixed-order reductions)
- **Modular**: Each kernel is independent and testable
- **Efficient**: Tiled GEMM, fused operations, minimal memory transfers
- **Scalable**: From 10M (micro) to 100M+ (medium) parameters

## Architecture

```
Input tokens
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Layer (×N)             │
│  ┌─────────────────────────────────┐│
│  │ LayerNorm → Attention → Residual││
│  │ LayerNorm → FFN → Residual      ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Output Projection (→ vocab_size)
    │
    ▼
Cross-Entropy Loss
```

## Model Configurations

| Config | Params | Layers | d_model | Heads | FFN | Max Seq |
|--------|--------|--------|---------|-------|-----|---------|
| Micro | ~10M | 4 | 256 | 4 | 1024 | 128 |
| Tiny | ~25M | 6 | 512 | 8 | 2048 | 128 |
| Small | ~50M | 6 | 512 | 8 | 2048 | 256 |
| Medium | ~100M | 12 | 768 | 12 | 3072 | 512 |

## Quick Start

### 1. Build Shaders

```bash
chmod +x build_shaders.sh
./build_shaders.sh
```

### 2. Run Tests

```bash
cargo test
```

### 3. Train

```bash
cargo run --release --bin train_transformer -- \
    --corpus /path/to/corpus.jsonl \
    --model-size tiny \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 3e-4
```

## Project Structure

```
hlx-transformer/
├── shader/                    # GLSL compute shaders
│   ├── gemm.glsl             # Matrix multiplication
│   ├── gemm_backward.glsl    # GEMM gradients
│   ├── layernorm_forward.glsl
│   ├── layernorm_backward.glsl
│   ├── softmax_forward.glsl
│   ├── softmax_backward.glsl
│   ├── gelu_forward.glsl
│   ├── gelu_backward.glsl
│   ├── embedding_forward.glsl
│   ├── embedding_backward.glsl
│   ├── attention_scores.glsl # Q @ K^T
│   ├── attention_output.glsl # Attn @ V
│   ├── cross_entropy_forward.glsl
│   ├── cross_entropy_backward.glsl
│   ├── adam_update.glsl
│   ├── elementwise.glsl
│   ├── reduce_sum.glsl
│   └── reduce_final.glsl
├── src/
│   ├── lib.rs                # Module exports
│   ├── tensor.rs             # Tensor abstraction
│   ├── transformer_config.rs # Model configuration
│   ├── gemm_kernel.rs        # GEMM kernel wrapper
│   ├── attention_kernel.rs   # Attention implementation
│   ├── transformer_layer.rs  # Full transformer layer
│   └── bin/
│       └── train_transformer.rs
├── Cargo.toml
├── build_shaders.sh
├── ARCHITECTURE.md
├── INTEGRATION_GUIDE.md
├── TESTING.md
└── README.md
```

## Shaders

| Shader | Purpose | Workgroup |
|--------|---------|-----------|
| gemm.glsl | Tiled matrix multiply | 16×16×1 |
| layernorm_forward.glsl | Layer normalization | 256×1×1 |
| softmax_forward.glsl | Numerically stable softmax | 256×1×1 |
| gelu_forward.glsl | GELU activation | 256×1×1 |
| cross_entropy_forward.glsl | Log-softmax + NLL loss | 256×1×1 |
| adam_update.glsl | Adam optimizer | 256×1×1 |

## Integration

This package is designed to integrate with existing HLX Vulkan infrastructure:

1. Copy shaders and Rust modules
2. Replace stub modules with your implementations
3. Build shaders with `./build_shaders.sh`
4. Follow INTEGRATION_GUIDE.md

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for details.

## Testing

See [TESTING.md](TESTING.md) for:
- Unit test procedures
- Gradient checking
- Determinism validation
- End-to-end tests

## Determinism

All operations guarantee bit-identical results:

- **GEMM**: Fixed tile iteration order
- **Reductions**: Tree reduction then single-thread final sum
- **No atomics**: Per-workgroup staging buffers
- **Explicit barriers**: Between all dependent dispatches

Verify with:
```bash
cargo run --release --bin train_transformer -- \
    --validate-determinism
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| GEMM | >1 TFLOP/s | 512×512 matrices |
| Training speed | <2s/epoch | 182 examples, batch=4 |
| Memory | <4GB VRAM | Tiny model |
| Loss | <0.05 @ 100 epochs | ASCII corpus |

## Target Hardware

- **GPU**: RTX 5060 (primary), any Vulkan 1.2 capable GPU
- **VRAM**: 4GB minimum, 8GB+ recommended
- **CPU**: Any modern x86_64

## Dependencies

- Rust 1.70+
- Vulkan SDK (for glslc shader compiler)
- ash 0.37 (Vulkan bindings)

## License

MIT

## Acknowledgments

- Design spec by Claude Sonnet (CLI)
- Implementation by Claude Opus (Web)
- Human oversight by Matt
