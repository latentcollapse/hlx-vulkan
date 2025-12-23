#!/usr/bin/env python3
"""Debug Cross-Entropy contract"""

import sys
import numpy as np
sys.path.insert(0, '/home/matt/hlx-compiler/python')

from hlx_lcb_client import LCBClient, LCBBatchBuilder

print("=" * 70)
print("Debugging Cross-Entropy Contract")
print("=" * 70)

# Small test case for debugging
np.random.seed(42)
logits = np.random.randn(4, 5).astype(np.float32)
targets = np.array([0, 1, 2, 3], dtype=np.int32)

print("\nInput logits:")
print(logits)
print("\nTargets:", targets)

# Expected (NumPy) - using log-softmax like the shader
print("\n--- NumPy Computation ---")
max_logits = logits.max(axis=1, keepdims=True)
print("Max logits:", max_logits.flatten())

exp_logits = np.exp(logits - max_logits)
print("Exp(logits - max) sum:", exp_logits.sum(axis=1))

sum_exp = exp_logits.sum(axis=1, keepdims=True)
log_sum_exp = np.log(sum_exp)
print("Log sum exp:", log_sum_exp.flatten())

log_probs = logits - max_logits - log_sum_exp
print("\nLog probabilities:")
print(log_probs)

# Extract target log probs
target_log_probs = log_probs[range(4), targets]
print("\nTarget log probs:", target_log_probs)

# Negative log likelihood
per_position_losses = -target_log_probs
print("Per-position losses:", per_position_losses)

expected_mean_loss = per_position_losses.mean()
print(f"\nExpected mean loss: {expected_mean_loss:.6f}")

# GPU computation
print("\n--- GPU Computation ---")
batch = LCBBatchBuilder().add_cross_entropy(logits, targets).build()
client = LCBClient()
results = client.execute_batch(batch)

gpu_loss = results[0].item() if hasattr(results[0], 'item') else results[0][0]
print(f"GPU mean loss: {gpu_loss:.6f}")

print(f"\nError: {abs(gpu_loss - expected_mean_loss):.6e}")
print("=" * 70)
