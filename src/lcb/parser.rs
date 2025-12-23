//! LC-B Binary Format Parser
//!
//! Parses LC-B (Low-level Contract Binary) instruction batches.
//! Format:
//!   - Magic: "LCB!" (4 bytes)
//!   - Version: LEB128
//!   - Num instructions: LEB128
//!   - Instructions: [contract_id, tensors, scalars]...
//!   - SHA256 signature (32 bytes)

use std::collections::HashMap;

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    UInt32 = 3,
}

impl DType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(DType::Float32),
            1 => Some(DType::Float16),
            2 => Some(DType::Int32),
            3 => Some(DType::UInt32),
            _ => None,
        }
    }
    
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float16 => 2,
        }
    }
}

/// Tensor data extracted from LC-B
#[derive(Debug, Clone)]
pub struct TensorData {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorData {
    /// Number of elements in tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Get data as f32 slice (only valid if dtype is Float32)
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::Float32 {
            return None;
        }
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
    
    /// Get data as mutable f32 slice
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::Float32 {
            return None;
        }
        let ptr = self.data.as_mut_ptr() as *mut f32;
        let len = self.data.len() / 4;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }
    
    /// Create tensor from f32 data
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        Self {
            dtype: DType::Float32,
            shape,
            data: bytes.to_vec(),
        }
    }
}

/// A single LC-B instruction
#[derive(Debug, Clone)]
pub struct LCBInstruction {
    pub contract_id: u32,
    pub tensors: Vec<TensorData>,
    pub scalars: HashMap<String, f32>,
}

/// Contract IDs
pub mod contracts {
    pub const GEMM: u32 = 906;
    pub const LAYERNORM: u32 = 907;
    pub const GELU: u32 = 908;
    pub const SOFTMAX: u32 = 909;
    pub const CROSS_ENTROPY: u32 = 910;
}

/// Parsed LC-B batch
#[derive(Debug)]
pub struct LCBBatch {
    pub version: u32,
    pub instructions: Vec<LCBInstruction>,
}

/// Parser for LC-B binary format
pub struct LCBParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> LCBParser<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    
    /// Parse complete LC-B batch
    pub fn parse(&mut self) -> Result<LCBBatch, String> {
        // Verify minimum size (magic + version + count + signature)
        if self.data.len() < 4 + 1 + 1 + 32 {
            return Err("LC-B data too short".to_string());
        }
        
        // Check magic
        if &self.data[0..4] != b"LCB!" {
            return Err(format!("Invalid magic: expected 'LCB!', got {:?}", &self.data[0..4]));
        }
        self.pos = 4;
        
        // Verify SHA256 signature
        let content_len = self.data.len() - 32;
        let expected_sig = &self.data[content_len..];
        let actual_sig = sha256(&self.data[..content_len]);
        if expected_sig != actual_sig.as_slice() {
            return Err("SHA256 signature mismatch".to_string());
        }
        
        // Parse version
        let version = self.read_leb128()? as u32;
        if version != 1 {
            return Err(format!("Unsupported LC-B version: {}", version));
        }
        
        // Parse instruction count
        let num_instructions = self.read_leb128()? as usize;
        
        // Parse instructions
        let mut instructions = Vec::with_capacity(num_instructions);
        for i in 0..num_instructions {
            let instr = self.parse_instruction()
                .map_err(|e| format!("Instruction {}: {}", i, e))?;
            instructions.push(instr);
        }
        
        Ok(LCBBatch { version, instructions })
    }
    
    fn parse_instruction(&mut self) -> Result<LCBInstruction, String> {
        // Contract ID
        let contract_id = self.read_leb128()? as u32;
        
        // Tensors
        let num_tensors = self.read_leb128()? as usize;
        let mut tensors = Vec::with_capacity(num_tensors);
        
        for _ in 0..num_tensors {
            let tensor = self.parse_tensor()?;
            tensors.push(tensor);
        }
        
        // Scalars
        let num_scalars = self.read_leb128()? as usize;
        let mut scalars = HashMap::with_capacity(num_scalars);
        
        for _ in 0..num_scalars {
            let name_len = self.read_leb128()? as usize;
            let name = self.read_string(name_len)?;
            let value = self.read_f32()?;
            scalars.insert(name, value);
        }
        
        Ok(LCBInstruction { contract_id, tensors, scalars })
    }
    
    fn parse_tensor(&mut self) -> Result<TensorData, String> {
        // dtype (1 byte)
        if self.pos >= self.data.len() {
            return Err("Unexpected end of data while reading dtype".to_string());
        }
        let dtype_byte = self.data[self.pos];
        self.pos += 1;
        
        let dtype = DType::from_u8(dtype_byte)
            .ok_or_else(|| format!("Unknown dtype: {}", dtype_byte))?;
        
        // ndim
        let ndim = self.read_leb128()? as usize;
        
        // shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = self.read_leb128()? as usize;
            shape.push(dim);
        }
        
        // Calculate data size
        let num_elements: usize = shape.iter().product();
        let data_size = num_elements * dtype.size_bytes();
        
        // Read data
        if self.pos + data_size > self.data.len() - 32 { // -32 for signature
            return Err(format!(
                "Not enough data for tensor: need {} bytes, have {}",
                data_size,
                self.data.len() - 32 - self.pos
            ));
        }
        
        let data = self.data[self.pos..self.pos + data_size].to_vec();
        self.pos += data_size;
        
        Ok(TensorData { dtype, shape, data })
    }
    
    /// Read unsigned LEB128 encoded integer
    fn read_leb128(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0;
        
        loop {
            if self.pos >= self.data.len() {
                return Err("Unexpected end of data while reading LEB128".to_string());
            }
            
            let byte = self.data[self.pos];
            self.pos += 1;
            
            result |= ((byte & 0x7F) as u64) << shift;
            
            if byte & 0x80 == 0 {
                break;
            }
            
            shift += 7;
            if shift >= 64 {
                return Err("LEB128 overflow".to_string());
            }
        }
        
        Ok(result)
    }
    
    fn read_string(&mut self, len: usize) -> Result<String, String> {
        if self.pos + len > self.data.len() {
            return Err("Unexpected end of data while reading string".to_string());
        }
        
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        
        String::from_utf8(bytes.to_vec())
            .map_err(|e| format!("Invalid UTF-8: {}", e))
    }
    
    fn read_f32(&mut self) -> Result<f32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("Unexpected end of data while reading f32".to_string());
        }
        
        let bytes = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        
        Ok(f32::from_le_bytes(bytes))
    }
}

/// Simple SHA-256 implementation
fn sha256(data: &[u8]) -> [u8; 32] {
    use std::num::Wrapping;
    
    // Initial hash values
    let mut h: [Wrapping<u32>; 8] = [
        Wrapping(0x6a09e667), Wrapping(0xbb67ae85), Wrapping(0x3c6ef372), Wrapping(0xa54ff53a),
        Wrapping(0x510e527f), Wrapping(0x9b05688c), Wrapping(0x1f83d9ab), Wrapping(0x5be0cd19),
    ];
    
    // Round constants
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    
    // Padding
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    
    // Process blocks
    for chunk in padded.chunks(64) {
        let mut w = [Wrapping(0u32); 64];
        
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = Wrapping(u32::from_be_bytes([word[0], word[1], word[2], word[3]]));
        }
        
        for i in 16..64 {
            let s0 = (w[i-15].0.rotate_right(7)) ^ (w[i-15].0.rotate_right(18)) ^ (w[i-15].0 >> 3);
            let s1 = (w[i-2].0.rotate_right(17)) ^ (w[i-2].0.rotate_right(19)) ^ (w[i-2].0 >> 10);
            w[i] = w[i-16] + Wrapping(s0) + w[i-7] + Wrapping(s1);
        }
        
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = 
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        
        for i in 0..64 {
            let s1 = Wrapping(e.0.rotate_right(6) ^ e.0.rotate_right(11) ^ e.0.rotate_right(25));
            let ch = Wrapping((e.0 & f.0) ^ ((!e.0) & g.0));
            let temp1 = hh + s1 + ch + Wrapping(K[i]) + w[i];
            let s0 = Wrapping(a.0.rotate_right(2) ^ a.0.rotate_right(13) ^ a.0.rotate_right(22));
            let maj = Wrapping((a.0 & b.0) ^ (a.0 & c.0) ^ (b.0 & c.0));
            let temp2 = s0 + maj;
            
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        
        h[0] = h[0] + a;
        h[1] = h[1] + b;
        h[2] = h[2] + c;
        h[3] = h[3] + d;
        h[4] = h[4] + e;
        h[5] = h[5] + f;
        h[6] = h[6] + g;
        h[7] = h[7] + hh;
    }
    
    let mut result = [0u8; 32];
    for (i, &hash) in h.iter().enumerate() {
        result[i*4..i*4+4].copy_from_slice(&hash.0.to_be_bytes());
    }
    result
}

/// Serialize a tensor to LC-B format
pub fn serialize_tensor(tensor: &TensorData) -> Vec<u8> {
    let mut buf = Vec::new();
    
    // dtype (1 byte)
    buf.push(tensor.dtype as u8);
    
    // ndim (LEB128)
    write_leb128(&mut buf, tensor.shape.len() as u64);
    
    // shape (each dim as LEB128)
    for &dim in &tensor.shape {
        write_leb128(&mut buf, dim as u64);
    }
    
    // data (raw bytes)
    buf.extend_from_slice(&tensor.data);
    
    buf
}

/// Write unsigned LEB128
fn write_leb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_leb128_roundtrip() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, u32::MAX as u64];
        for &v in &test_values {
            let mut buf = Vec::new();
            write_leb128(&mut buf, v);
            
            let mut parser = LCBParser::new(&buf);
            let parsed = parser.read_leb128().unwrap();
            assert_eq!(v, parsed, "LEB128 roundtrip failed for {}", v);
        }
    }
    
    #[test]
    fn test_sha256() {
        // Test vector: empty string
        let hash = sha256(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
            0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
            0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
            0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash, expected);
    }
    
    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Int32.size_bytes(), 4);
        assert_eq!(DType::UInt32.size_bytes(), 4);
    }
    
    #[test]
    fn test_tensor_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorData::from_f32(&data, vec![2, 3]);
        
        assert_eq!(tensor.dtype, DType::Float32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.num_elements(), 6);
        
        let recovered = tensor.as_f32().unwrap();
        assert_eq!(recovered, data.as_slice());
    }
}
