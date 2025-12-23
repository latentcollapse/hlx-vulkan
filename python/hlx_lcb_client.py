"""
HLX LC-B Client

Python client for sending LC-B instruction batches to the Rust GPU executor.

Usage:
    from hlx_lcb_client import LCBClient, LCBBatchBuilder
    
    client = LCBClient()
    batch = LCBBatchBuilder().add_gemm(A, B).build()
    results = client.execute_batch(batch)
"""

import socket
import struct
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Using basic arrays.")


class DType(IntEnum):
    """Data types supported by LC-B"""
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT32 = 3


class ContractID(IntEnum):
    """LC-B Contract IDs"""
    GEMM = 906
    LAYERNORM = 907
    GELU = 908
    SOFTMAX = 909
    CROSS_ENTROPY = 910


def encode_leb128(value: int) -> bytes:
    """Encode unsigned integer as LEB128"""
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


def decode_leb128(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode LEB128 from bytes, returns (value, bytes_consumed)"""
    result = 0
    shift = 0
    consumed = 0
    
    while True:
        byte = data[offset + consumed]
        consumed += 1
        result |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            break
        shift += 7
    
    return result, consumed


class LCBClient:
    """Client for sending LC-B batches to Rust GPU executor"""
    
    def __init__(self, socket_path: str = "/tmp/hlx_vulkan.sock"):
        """
        Initialize LC-B client.
        
        Args:
            socket_path: Path to Unix socket where hlx_lcb_service is listening
        """
        self.socket_path = socket_path
    
    def execute_batch(self, batch_bytes: bytes) -> List[Any]:
        """
        Send LC-B batch and receive result tensors.
        
        Args:
            batch_bytes: Serialized LC-B batch from LCBBatchBuilder.build()
            
        Returns:
            List of result tensors (numpy arrays if available, else raw bytes)
        """
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            
            # Send request
            sock.sendall(struct.pack('<I', len(batch_bytes)))
            sock.sendall(batch_bytes)
            
            # Receive response length
            resp_len_bytes = self._recv_exact(sock, 4)
            resp_len = struct.unpack('<I', resp_len_bytes)[0]
            
            # Receive response
            response = self._recv_exact(sock, resp_len)
            
            # Deserialize tensors
            return self._deserialize_tensors(response)
    
    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket"""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data
    
    def _deserialize_tensors(self, data: bytes) -> List[Any]:
        """Deserialize tensor list from response bytes"""
        offset = 0
        num_tensors = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        tensors = []
        for _ in range(num_tensors):
            tensor_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            tensor_bytes = data[offset:offset + tensor_len]
            offset += tensor_len
            
            tensor = self._parse_tensor(tensor_bytes)
            tensors.append(tensor)
        
        return tensors
    
    def _parse_tensor(self, data: bytes) -> Any:
        """Parse a single tensor from bytes"""
        offset = 0
        
        # dtype
        dtype_code = data[offset]
        offset += 1
        
        dtype_map = {
            0: 'float32',
            1: 'float16',
            2: 'int32',
            3: 'uint32',
        }
        dtype_str = dtype_map.get(dtype_code, 'float32')
        
        # ndim
        ndim, consumed = decode_leb128(data, offset)
        offset += consumed
        
        # shape
        shape = []
        for _ in range(ndim):
            dim, consumed = decode_leb128(data, offset)
            offset += consumed
            shape.append(dim)
        
        # data
        array_data = data[offset:]
        
        if HAS_NUMPY:
            np_dtype = getattr(np, dtype_str)
            return np.frombuffer(array_data, dtype=np_dtype).reshape(shape)
        else:
            # Return raw bytes and shape if numpy not available
            return {'dtype': dtype_str, 'shape': shape, 'data': array_data}


class LCBBatchBuilder:
    """Builder for constructing LC-B instruction batches"""
    
    def __init__(self):
        self.instructions = []
    
    def add_gemm(self, a, b) -> 'LCBBatchBuilder':
        """
        Add GEMM operation: C = A @ B
        
        Args:
            a: Input matrix A [M, K]
            b: Input matrix B [K, N]
            
        Returns:
            Self for chaining
        """
        self.instructions.append({
            'contract_id': ContractID.GEMM,
            'tensors': [self._to_tensor(a), self._to_tensor(b)],
            'scalars': {},
        })
        return self
    
    def add_layernorm(self, x, gamma=None, beta=None, eps: float = 1e-5) -> 'LCBBatchBuilder':
        """
        Add LayerNorm operation.
        
        Args:
            x: Input tensor [batch, features]
            gamma: Scale parameter (optional)
            beta: Shift parameter (optional)
            eps: Epsilon for numerical stability
            
        Returns:
            Self for chaining
        """
        tensors = [self._to_tensor(x)]
        if gamma is not None:
            tensors.append(self._to_tensor(gamma))
        if beta is not None:
            tensors.append(self._to_tensor(beta))
        
        self.instructions.append({
            'contract_id': ContractID.LAYERNORM,
            'tensors': tensors,
            'scalars': {'eps': eps},
        })
        return self
    
    def add_gelu(self, x) -> 'LCBBatchBuilder':
        """
        Add GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Self for chaining
        """
        self.instructions.append({
            'contract_id': ContractID.GELU,
            'tensors': [self._to_tensor(x)],
            'scalars': {},
        })
        return self
    
    def add_softmax(self, x) -> 'LCBBatchBuilder':
        """
        Add Softmax operation (along last dimension).
        
        Args:
            x: Input tensor [batch, classes]
            
        Returns:
            Self for chaining
        """
        self.instructions.append({
            'contract_id': ContractID.SOFTMAX,
            'tensors': [self._to_tensor(x)],
            'scalars': {},
        })
        return self
    
    def add_cross_entropy(self, logits, targets, ignore_index: int = -100) -> 'LCBBatchBuilder':
        """
        Add Cross-Entropy loss computation.
        
        Args:
            logits: Unnormalized logits [batch, vocab]
            targets: Target indices [batch]
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Self for chaining
        """
        self.instructions.append({
            'contract_id': ContractID.CROSS_ENTROPY,
            'tensors': [self._to_tensor(logits), self._to_tensor(targets)],
            'scalars': {'ignore_index': float(ignore_index)},
        })
        return self
    
    def build(self) -> bytes:
        """
        Serialize instructions to LC-B binary format.
        
        Returns:
            LC-B batch bytes ready to send to executor
        """
        buf = bytearray()
        
        # Magic
        buf.extend(b'LCB!')
        
        # Version
        buf.extend(encode_leb128(1))
        
        # Num instructions
        buf.extend(encode_leb128(len(self.instructions)))
        
        # Instructions
        for instr in self.instructions:
            # Contract ID
            buf.extend(encode_leb128(instr['contract_id']))
            
            # Tensors
            buf.extend(encode_leb128(len(instr['tensors'])))
            for tensor in instr['tensors']:
                buf.extend(self._serialize_tensor(tensor))
            
            # Scalars
            scalars = instr.get('scalars', {})
            buf.extend(encode_leb128(len(scalars)))
            for name, value in scalars.items():
                name_bytes = name.encode('utf-8')
                buf.extend(encode_leb128(len(name_bytes)))
                buf.extend(name_bytes)
                buf.extend(struct.pack('<f', float(value)))
        
        # SHA256 signature
        signature = hashlib.sha256(buf).digest()
        buf.extend(signature)
        
        return bytes(buf)
    
    def _to_tensor(self, arr) -> dict:
        """Convert array-like to tensor dict"""
        if HAS_NUMPY:
            arr = np.asarray(arr)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            return {
                'dtype': arr.dtype,
                'shape': list(arr.shape),
                'data': arr.tobytes(),
            }
        else:
            # Assume it's already a dict with dtype, shape, data
            return arr
    
    def _serialize_tensor(self, tensor: dict) -> bytes:
        """Serialize tensor to bytes"""
        buf = bytearray()
        
        # dtype
        dtype = tensor['dtype']
        if HAS_NUMPY:
            dtype_map = {
                np.dtype('float32'): 0,
                np.dtype('float16'): 1,
                np.dtype('int32'): 2,
                np.dtype('uint32'): 3,
            }
            buf.append(dtype_map.get(dtype, 0))
        else:
            dtype_map = {'float32': 0, 'float16': 1, 'int32': 2, 'uint32': 3}
            buf.append(dtype_map.get(str(dtype), 0))
        
        # ndim
        shape = tensor['shape']
        buf.extend(encode_leb128(len(shape)))
        
        # shape
        for dim in shape:
            buf.extend(encode_leb128(dim))
        
        # data
        buf.extend(tensor['data'])
        
        return bytes(buf)


# =============================================================================
# Example usage and tests
# =============================================================================

def example_gemm():
    """Example: GEMM operation"""
    if not HAS_NUMPY:
        print("Example requires numpy")
        return
    
    # Create matrices
    M, K, N = 128, 256, 512
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Build LC-B batch
    batch = LCBBatchBuilder().add_gemm(A, B).build()
    
    print(f"GEMM: ({M}×{K}) @ ({K}×{N}) → ({M}×{N})")
    print(f"LC-B batch size: {len(batch)} bytes")
    
    # Execute (requires running service)
    try:
        client = LCBClient()
        results = client.execute_batch(batch)
        C = results[0]
        
        # Verify
        C_expected = A @ B
        if np.allclose(C, C_expected, rtol=1e-4):
            print("✅ GEMM result verified!")
        else:
            print("❌ GEMM result mismatch")
            print(f"   Max error: {np.max(np.abs(C - C_expected))}")
    except FileNotFoundError:
        print("⚠️  Service not running (socket not found)")
    except ConnectionRefusedError:
        print("⚠️  Service not running (connection refused)")


def example_transformer_ops():
    """Example: Chain of transformer operations"""
    if not HAS_NUMPY:
        print("Example requires numpy")
        return
    
    batch_size = 32
    seq_len = 64
    d_model = 256
    vocab_size = 260
    
    # Create test data
    hidden = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    W = np.random.randn(d_model, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, size=(batch_size * seq_len,)).astype(np.uint32)
    
    # Build batch with multiple operations
    batch = (LCBBatchBuilder()
        .add_layernorm(hidden)
        .add_gemm(hidden, W)  # Project to vocab
        .add_cross_entropy(np.random.randn(batch_size * seq_len, vocab_size).astype(np.float32), targets)
        .build())
    
    print(f"Transformer ops: LayerNorm + GEMM + CrossEntropy")
    print(f"LC-B batch size: {len(batch)} bytes")
    
    try:
        client = LCBClient()
        results = client.execute_batch(batch)
        print(f"✅ Received {len(results)} result tensors")
        for i, r in enumerate(results):
            if HAS_NUMPY:
                print(f"   Result {i}: shape={r.shape}, dtype={r.dtype}")
    except (FileNotFoundError, ConnectionRefusedError):
        print("⚠️  Service not running")


if __name__ == "__main__":
    print("HLX LC-B Client Examples\n")
    print("=" * 50)
    print("Example 1: GEMM")
    print("=" * 50)
    example_gemm()
    print()
    print("=" * 50)
    print("Example 2: Transformer Operations")
    print("=" * 50)
    example_transformer_ops()
