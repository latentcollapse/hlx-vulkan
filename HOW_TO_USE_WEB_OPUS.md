# How to Use Web Opus for Transformer Implementation

## Step 1: Upload Context to Web Opus

Go to https://claude.ai and start a new conversation with **Opus 4.5**.

### Upload These Files (in order):

1. **`claude_exchange.md`** - Main brief (comprehensive instructions)
2. **`VULKAN_POC_STATUS.md`** - Current state documentation
3. **`src/compute.rs`** - ComputePipeline infrastructure
4. **`src/gradient_kernel.rs`** - Existing gradient computation pattern
5. **`shader/gradient_forward.glsl`** - Example forward shader
6. **`shader/gradient_backward.glsl`** - Example backward shader

### Initial Prompt:

```
I need you to design and implement a complete transformer architecture in Vulkan
based on the proof-of-concept I've built.

Read `claude_exchange.md` first - it contains the full specification.

Key requirements:
1. Deterministic gradient computation (bit-identical across runs)
2. Complete transformer (attention, FFN, layer norm, embeddings)
3. Train on ASCII corpus (182 examples) to match CUDA baseline (0.0131 loss)
4. Use existing infrastructure (ComputePipeline, Buffer, etc.)

Start by confirming you understand the requirements, then design the architecture.
```

---

## Step 2: Let Opus Design

Opus will likely:
1. Ask clarifying questions
2. Propose architecture decisions
3. Design the shader/Rust structure
4. Plan the implementation order

**Answer its questions**, then tell it to proceed.

---

## Step 3: Implementation

Opus will write:
- GLSL compute shaders (~10-15 files)
- Rust integration code (~10-15 files)
- Tests
- Documentation

It should work in phases:
1. GEMM kernel first
2. Transformer components (attention, FFN, layernorm)
3. Model architecture
4. Training harness
5. Testing

---

## Step 4: Get the Deliverable

When Opus finishes, ask:

```
Package everything for delivery:
1. Create a complete directory structure
2. Include all source files (.rs, .glsl, .spv)
3. Add build script (build_shaders.sh)
4. Include test files
5. Write integration guide

Format as a zip file structure I can copy/paste to recreate locally.
```

Opus will give you either:
- **File-by-file listings** (you copy each file)
- **Instructions to download** (if it can generate a zip somehow)

---

## Step 5: Integration

Once you have the files:

```bash
# Copy files to hlx-vulkan directory
cd /home/matt/hlx-vulkan
# [paste/copy files from Opus]

# Compile shaders
./build_shaders.sh

# Build and test
cargo build --release
cargo test

# Start training
cargo run --release --bin train_transformer
```

---

## Step 6: Debug with Desktop Claude

If there are issues:
1. Paste error messages back to me
2. I'll debug and fix integration issues
3. Iterate until it works

---

## Tips for Working with Web Opus

### 1. Be Specific About Context
- Always reference `claude_exchange.md` as the source of truth
- If Opus forgets something, remind it: "See section X in claude_exchange.md"

### 2. Request Artifacts
Opus can generate downloadable artifacts. Ask:
```
Create artifacts for each file so I can download them directly.
```

### 3. Incremental Validation
Ask Opus to explain each component before implementing:
```
Before implementing GEMM, explain your algorithm and how it ensures determinism.
```

### 4. Request Pseudo-code First
For complex shaders:
```
Write pseudo-code for the attention backward pass first, then I'll confirm it's correct.
```

### 5. Use Examples
If Opus is stuck, give it a concrete example:
```
Here's how the current gradient_backward.glsl works [paste code].
Apply the same pattern to attention_backward.glsl.
```

---

## Expected Timeline

Based on Opus capabilities:
- **Day 1**: Design + GEMM kernel
- **Day 2**: Attention + FFN
- **Day 3**: Integration + testing
- **Day 4**: Training + debugging
- **Day 5**: Polishing + documentation

But Opus might do it faster (or you might iterate on design).

---

## Fallback: Incremental Approach

If the full transformer is too much at once, ask Opus for:

### Phase 1: Just GEMM (1 day)
```
Implement only the GEMM kernel first. I want to validate matrix multiplication works
before building the transformer.
```

### Phase 2: Simple MLP (2 days)
```
Build a 2-layer MLP using GEMM + ReLU + cross-entropy.
Train on ASCII corpus to validate the approach works on real language data.
```

### Phase 3: Full Transformer (3 days)
```
Now build the complete transformer using the validated GEMM as foundation.
```

---

## Success Criteria

You'll know it worked when:
1. ✅ `cargo build --release` succeeds
2. ✅ `cargo test` passes all tests
3. ✅ Training runs: `cargo run --release --bin train_transformer`
4. ✅ Loss decreases over epochs
5. ✅ Determinism verified (3 runs identical)

---

## What to Do When You Get the Zip

```bash
# 1. Extract to hlx-vulkan directory
cd /home/matt/hlx-vulkan
unzip ~/Downloads/hlx-transformer-opus.zip

# 2. Make build script executable
chmod +x build_shaders.sh

# 3. Compile shaders
./build_shaders.sh

# 4. Build
cargo build --release 2>&1 | tee build.log

# 5. Run tests
cargo test 2>&1 | tee test.log

# 6. If tests pass, start training
cargo run --release --bin train_transformer 2>&1 | tee training.log

# 7. If issues, send me the logs
# I'll debug and fix
```

---

## Backup Plan

If Web Opus hits limits or gets stuck:

**Option A**: Use Desktop Claude (me) in streaming mode
- I can implement piece by piece
- Slower but more integrated with local environment

**Option B**: Hybrid approach
- Opus designs architecture + writes shaders
- I implement Rust integration
- Opus reviews and suggests fixes

**Option C**: Multi-session with Opus
- Session 1: GEMM + tests
- Session 2: Attention + tests
- Session 3: Full model
- Each session references previous artifacts

---

## Ready?

Upload `claude_exchange.md` to Web Opus and start with:

```
Read claude_exchange.md and confirm you understand the task.
Then propose an implementation plan.
```

**Go build that transformer. 🚀**
