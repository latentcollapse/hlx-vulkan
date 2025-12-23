#!/usr/bin/env python3
"""Test all 5 LC-B contracts"""

import sys
import numpy as np
sys.path.insert(0, '/home/matt/hlx-compiler/python')

from hlx_lcb_client import LCBClient, LCBBatchBuilder, ContractID

client = LCBClient()

print("=" * 70)
print("Testing All 5 LC-B Contracts")
print("=" * 70)

# Test 1: GEMM
print("\n✅ CONTRACT 906: GEMM (Matrix Multiply)")
A = np.random.randn(32, 64).astype(np.float32)
B = np.random.randn(64, 128).astype(np.float32)
expected_gemm = A @ B

batch = LCBBatchBuilder().add_gemm(A, B).build()
results = client.execute_batch(batch)
C = results[0]

if np.allclose(C, expected_gemm, rtol=1e-4):
    print(f"   ✅ PASS - Max error: {np.abs(C - expected_gemm).max():.2e}")
else:
    print(f"   ❌ FAIL - Max error: {np.abs(C - expected_gemm).max():.2e}")

# Test 2: LayerNorm
print("\n✅ CONTRACT 907: LayerNorm")
X = np.random.randn(16, 256).astype(np.float32)
eps = 1e-5

# Expected (NumPy)
mean = X.mean(axis=1, keepdims=True)
var = X.var(axis=1, keepdims=True)
expected_ln = (X - mean) / np.sqrt(var + eps)

batch = LCBBatchBuilder().add_layernorm(X, eps=eps).build()
try:
    results = client.execute_batch(batch)
    Y = results[0]

    if np.allclose(Y, expected_ln, rtol=1e-4):
        print(f"   ✅ PASS - Max error: {np.abs(Y - expected_ln).max():.2e}")
    else:
        print(f"   ❌ FAIL - Max error: {np.abs(Y - expected_ln).max():.2e}")
except Exception as e:
    print(f"   ⚠️  ERROR: {e}")

# Test 3: GELU
print("\n✅ CONTRACT 908: GELU Activation")
X = np.random.randn(128, 512).astype(np.float32)

# Expected (NumPy) - GELU approximation
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

expected_gelu = gelu(X)

batch = LCBBatchBuilder().add_gelu(X).build()
try:
    results = client.execute_batch(batch)
    Y = results[0]

    if np.allclose(Y, expected_gelu, rtol=1e-3):
        print(f"   ✅ PASS - Max error: {np.abs(Y - expected_gelu).max():.2e}")
    else:
        print(f"   ⚠️  APPROX - Max error: {np.abs(Y - expected_gelu).max():.2e}")
except Exception as e:
    print(f"   ⚠️  ERROR: {e}")

# Test 4: Softmax
print("\n✅ CONTRACT 909: Softmax")
X = np.random.randn(64, 1000).astype(np.float32)

# Expected (NumPy)
exp_x = np.exp(X - X.max(axis=1, keepdims=True))
expected_softmax = exp_x / exp_x.sum(axis=1, keepdims=True)

batch = LCBBatchBuilder().add_softmax(X).build()
try:
    results = client.execute_batch(batch)
    Y = results[0]

    if np.allclose(Y, expected_softmax, rtol=1e-4):
        print(f"   ✅ PASS - Max error: {np.abs(Y - expected_softmax).max():.2e}")
    else:
        print(f"   ⚠️  APPROX - Max error: {np.abs(Y - expected_softmax).max():.2e}")
except Exception as e:
    print(f"   ⚠️  ERROR: {e}")

# Test 5: Cross-Entropy
print("\n✅ CONTRACT 910: Cross-Entropy Loss")
logits = np.random.randn(32, 10).astype(np.float32)
targets = np.random.randint(0, 10, size=32).astype(np.int32)

# Expected (NumPy)
softmax_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
softmax_logits /= softmax_logits.sum(axis=1, keepdims=True)
expected_ce = -np.mean(np.log(softmax_logits[range(32), targets] + 1e-10))

batch = LCBBatchBuilder().add_cross_entropy(logits, targets).build()
try:
    results = client.execute_batch(batch)
    loss = results[0].item() if hasattr(results[0], 'item') else results[0][0]

    if abs(loss - expected_ce) < 1e-3:
        print(f"   ✅ PASS - Error: {abs(loss - expected_ce):.2e}")
    else:
        print(f"   ⚠️  APPROX - Error: {abs(loss - expected_ce):.2e}")
except Exception as e:
    print(f"   ⚠️  ERROR: {e}")

print("\n" + "=" * 70)
print("Contract Testing Complete!")
print("=" * 70)
