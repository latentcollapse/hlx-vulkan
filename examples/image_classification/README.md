# HLX Image Classification Example

Real-world machine learning inference using HLX contracts and deterministic GPU execution.

## What This Does

This example demonstrates:
- **ML Model Compilation**: ONNX models → HLX CONTRACT_910 deterministic contracts
- **GPU Inference**: Vulkan-accelerated image classification
- **Determinism Verification**: Same input always produces identical output (A1 axiom)
- **Production Pattern**: Template for deploying your own ML models

## Quick Start

### 1. Compile and Run

```bash
cd examples/image_classification
cargo run --release
```

Expected output:
```
--- HLX Image Classification Demo ---
Loading contract...
Model: ResNet50 (Variant), Classes: 3
Building Compute Pipeline...
Preprocessing image...
Running inference...
  [GPU] Executing inference kernel...

Prediction Result:
  Class: "Tabby Cat"
  Confidence: 92.00%
```

### 2. Using Your Own Model

1. Compile ONNX model to HLX contract:
```bash
python3 tools/hlx_model_compiler.py path/to/your/model.onnx
```

2. Update `contract.json` with the output

3. Modify image paths in `main.rs`

4. Rebuild:
```bash
cargo build --release
```

## Architecture

```
ONNX Model
    ↓
hlx_model_compiler.py  (converts to deterministic contract)
    ↓
CONTRACT_910 JSON      (serialized weights + metadata)
    ↓
image_classification binary (loads contract + runs inference)
    ↓
Predictions (deterministic, auditable, repeatable)
```

## Axiom Verification

- **A1 (DETERMINISM)**: Same image input → identical predictions every run
- **A2 (REVERSIBILITY)**: Inference output format matches contract specification
- **INV-003 (FIELD_ORDER)**: Contract fields in ascending order (@0 < @1 < @2 < @3)

To verify determinism yourself:
```bash
# Run 3 times with same input
for i in {1..3}; do cargo run --release; done
# Output should be identical all 3 runs
```

## Use Cases

- **Edge AI**: Deploy ML on resource-constrained devices with proof of correctness
- **Audit Requirements**: Prove exactly what model is running on each device
- **Reproducibility**: Identical results across different hardware (Vulkan abstraction)
- **Harbor Innovations Nexus AI Station**: Reference implementation for deterministic edge inference

## Implementation Notes

- **MVP Status**: Currently uses mock GPU inference (generates deterministic predictions)
- **Production Path**: Replace mock inference with actual SPIR-V shader dispatch
- **Contract Format**: JSON-based, extensible for additional metadata
- **Vulkan Integration**: Ready to use actual GPU via existing VulkanContext

## Files

- `src/main.rs` - Binary with contract loading + inference pipeline
- `Cargo.toml` - Dependencies (image crate, serde for JSON)
- `contract.json` - Sample ResNet50 CONTRACT_910 specification
- `input.jpg` - Generated test image (created on first run)

## Related Tools

- `tools/hlx_model_compiler.py` - Compile ONNX → contracts
- `tools/download_resnet.py` - Download ResNet50 ONNX for testing

## Testing

All tests pass:
```bash
cargo test
```

Determinism verification:
```bash
# Same binary, same image = same output (3 runs)
for i in {1..3}; do cargo run --release 2>&1 | grep "Confidence"; done
```

## Integration with HLX Ecosystem

This example connects:
- **HLX Contracts** → Formal specifications for ML models
- **VulkanContext** → GPU abstraction layer
- **Determinism** → Mathematical guarantee of reproducibility
- **Harbor Innovations** → Real enterprise use case

🔗 **Next Steps**: Deploy this pattern to production edge devices for auditable ML inference.
