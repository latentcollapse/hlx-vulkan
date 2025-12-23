#!/usr/bin/env python3
"""Test LC-B GEMM contract via Python client"""

import sys
import numpy as np
sys.path.insert(0, '/home/matt/hlx-compiler/python')

from hlx_lcb_client import LCBClient, LCBBatchBuilder

print("=" * 60)
print("Testing HLX LC-B GEMM Contract")
print("=" * 60)

# Create test matrices
print("\n1. Creating test matrices...")
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 256).astype(np.float32)
print(f"   A shape: {A.shape}")
print(f"   B shape: {B.shape}")

# Compute expected result with NumPy
print("\n2. Computing expected result (NumPy)...")
expected = A @ B
print(f"   Expected shape: {expected.shape}")
print(f"   Expected sample: {expected[0, :5]}")

# Build LC-B batch
print("\n3. Building LC-B batch...")
batch = LCBBatchBuilder().add_gemm(A, B).build()
print(f"   Batch size: {len(batch)} bytes")

# Send to GPU
print("\n4. Sending to GPU via /tmp/hlx_vulkan.sock...")
try:
    client = LCBClient()
    results = client.execute_batch(batch)
    print(f"   ✅ Received {len(results)} result tensors")

    # Check result
    C = results[0]
    print(f"\n5. Verifying result...")
    print(f"   GPU result shape: {C.shape}")
    print(f"   GPU result sample: {C[0, :5]}")

    # Compare
    if np.allclose(C, expected, rtol=1e-4, atol=1e-5):
        print(f"\n✅ SUCCESS! GPU result matches NumPy")
        print(f"   Max error: {np.abs(C - expected).max():.2e}")
        print(f"   Mean error: {np.abs(C - expected).mean():.2e}")
    else:
        print(f"\n❌ MISMATCH!")
        print(f"   Max error: {np.abs(C - expected).max():.2e}")
        print(f"   Mean error: {np.abs(C - expected).mean():.2e}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
