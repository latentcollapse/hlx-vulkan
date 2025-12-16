# hlx-pipeline-cli

CLI tool to create VkGraphicsPipeline from CONTRACT_902 definitions.

Part of the HLX Tier 1 Ecosystem Tools (CONTRACT_1002).

## Installation

```bash
cd /home/matt/hlx-vulkan
cargo build --release --bin hlx_pipeline_cli
```

The binary will be at `target/release/hlx_pipeline_cli`.

## Usage

### Basic Pipeline Creation

Create a pipeline from a JSON CONTRACT_902:

```bash
hlx-pipeline-cli create contract_902.json
# Output: &h_pipeline_abc123...
```

### HLXL Format

Parse HLXL format instead of JSON:

```bash
hlx-pipeline-cli create contract_902.hlxl --parse-hlxl
```

### Validation Only

Check contract validity without creating a pipeline:

```bash
hlx-pipeline-cli create contract_902.json --validate-only
```

Output:
```
=== Validation Result ===
Contract: VALID
Pipeline ID: &h_pipeline_abc123...
Shader stages: 2

Axioms verified:
  [OK] A1 DETERMINISM
  [OK] A2 REVERSIBILITY (contract stored)

Invariants verified:
  [OK] INV-001 TOTAL_FIDELITY
  [OK] INV-002 HANDLE_IDEMPOTENCE
  [OK] INV-003 FIELD_ORDER
```

### Verbose Mode

Show detailed axiom and invariant verification:

```bash
hlx-pipeline-cli create contract_902.json --verbose
```

### Custom Shader Database

Specify a custom shader database path:

```bash
hlx-pipeline-cli create contract_902.json --shader-db-path /path/to/shaders
```

## CONTRACT_902 Format

### JSON Format

```json
{
  "902": {
    "@0": "PIPELINE_CONFIG",
    "@1": "my_graphics_pipeline",
    "@2": "graphics",
    "@3": {
      "@0": {
        "@0": 0,
        "@1": "&h_shader_vertex_abc123...",
        "@2": "VERTEX_SHADER"
      },
      "@1": {
        "@0": 1,
        "@1": "&h_shader_fragment_def456...",
        "@2": "FRAGMENT_SHADER"
      }
    }
  }
}
```

### HLXL Format

```hlxl
contract 902 {
    pipeline_id: "my_graphics_pipeline",
    stages: [&h_shader_vert_abc123, &h_shader_frag_def456],
    sync_barriers: [],
    output_image: &h_framebuffer_xyz789
}
```

### Field Mapping

| HLXL Name | Field Index | Description |
|-----------|-------------|-------------|
| pipeline_id | @0 | Pipeline identifier |
| stages | @1 | Array of shader handles |
| sync_barriers | @2 | Synchronization barriers |
| output_image | @3 | Output framebuffer handle |

## Axioms Verified

### A1 DETERMINISM

Same CONTRACT_902 input always produces the same pipeline ID:

```bash
# Run twice - same output
hlx-pipeline-cli create contract.json  # &h_pipeline_abc123...
hlx-pipeline-cli create contract.json  # &h_pipeline_abc123...
```

### A2 REVERSIBILITY

Pipeline ID can be resolved back to the original contract. The contract JSON is stored alongside the pipeline handle in the cache.

## Invariants Verified

### INV-001 TOTAL_FIDELITY

Round-trip verification: contract -> pipeline -> verify the contract can be reconstructed.

### INV-002 HANDLE_IDEMPOTENCE

Calling the tool twice with the same contract returns the cached pipeline:

```bash
hlx-pipeline-cli create contract.json  # Creates pipeline
hlx-pipeline-cli create contract.json  # Returns cached pipeline
```

### INV-003 FIELD_ORDER

Contract fields must be in ascending index order (@0, @1, @2, @3).

## Error Handling

### Invalid JSON

```
Error: Invalid JSON: expected `:` at line 2 column 10
```

### Missing Contract Key

```
Error: Missing '902' key in contract
```

### Invalid Field Order

```
Error: INV-003 violated: Field order not ascending at index 2
```

### Missing Required Fields

```
Error: Missing required field @0 (contract type or pipeline_id)
```

### Shader Not Found

When integrated with VulkanContext:
```
Error: Shader not found in database: &h_shader_invalid_handle
```

## Integration with HLX Runtime

This CLI tool is designed to work with the HLX runtime ecosystem:

1. **ShaderDatabase**: Shaders are loaded from the content-addressed shader database
2. **VulkanContext**: Pipelines are created using `ctx.create_pipeline_from_contract()`
3. **Pipeline Cache**: Deterministic caching ensures idempotent execution

### Python Integration Example

```python
import subprocess
import json

# Create pipeline via CLI
result = subprocess.run(
    ['hlx-pipeline-cli', 'create', 'contract.json'],
    capture_output=True,
    text=True
)

pipeline_id = result.stdout.strip()
print(f"Pipeline: {pipeline_id}")

# Use pipeline ID in your application
# ...
```

## Test Checklist

### Model-Verified (Automated)

- [x] CONTRACT_902 field order validation
- [x] Determinism: same contract JSON -> same pipeline ID (run 2x)
- [x] Idempotence: second run returns cached pipeline
- [x] Field @0, @1, @2, @3 all present and ordered

### Human-Verified (Manual Testing)

- [ ] Parses valid CONTRACT_902 JSON
- [ ] Parses HLXL contracts (if --parse-hlxl flag used)
- [ ] Resolves shader handles correctly
- [ ] Creates valid VkGraphicsPipeline
- [ ] Pipeline caching works (deterministic IDs)
- [ ] Error messages clear (missing shader, invalid JSON)
- [ ] --validate-only flag works
- [ ] --verbose shows axiom verification

## Running Tests

```bash
cd /home/matt/hlx-vulkan
cargo test --bin hlx_pipeline_cli
```

Expected output:
```
running 7 tests
test tests::test_axiom_determinism ... ok
test tests::test_invariant_field_order ... ok
test tests::test_pipeline_id_format ... ok
test tests::test_hlxl_parsing ... ok
test tests::test_idempotence ... ok
test tests::test_required_fields_validation ... ok
test tests::test_missing_required_fields ... ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

## Architecture

```
CONTRACT_902 (JSON/HLXL)
        |
        v
   [Parser]
        |
        v
   [Validator]
   - Field order (INV-003)
   - Required fields
        |
        v
   [Shader Resolution]
   - Query ShaderDatabase
   - Load SPIR-V handles
        |
        v
   [Pipeline Cache Check]
   - Idempotence (INV-002)
        |
        v
   [VkGraphicsPipeline Creation]
   - VulkanContext.create_pipeline_from_contract()
        |
        v
   Pipeline ID: &h_pipeline_...
```

## References

- HLX Canonical Corpus v1.0.0: `/home/matt/helix-studio/HLX_CORPUS/HLX_CANONICAL_CORPUS_v1.0.0.md`
- CONTRACT_1002 Specification: `/home/matt/helix-studio/TIER1_TOOLS_CONTRACTS.hlxl`
- Phase 2 context.rs: `/home/matt/hlx-vulkan/src/context.rs`
- ShaderDatabase: `/home/matt/hlx-vulkan/hlx_runtime/shaderdb.py`

## Version

1.0.0 - Initial implementation (CONTRACT_1002)
