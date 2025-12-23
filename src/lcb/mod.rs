//! LC-B (Low-level Contract Binary) Module
//!
//! This module provides:
//! - `parser`: Parse LC-B binary format
//! - `executor`: Execute LC-B instructions on GPU
//!
//! # LC-B Format
//!
//! ```text
//! +------------------+
//! | Magic: "LCB!"    | 4 bytes
//! +------------------+
//! | Version (LEB128) | 1+ bytes
//! +------------------+
//! | Num Instructions | LEB128
//! +------------------+
//! | Instructions...  | Variable
//! +------------------+
//! | SHA256 Signature | 32 bytes
//! +------------------+
//! ```
//!
//! # Supported Contracts
//!
//! | ID  | Name         | Description                    |
//! |-----|--------------|--------------------------------|
//! | 906 | GEMM         | Matrix multiplication C = A@B  |
//! | 907 | LayerNorm    | Layer normalization            |
//! | 908 | GELU         | GELU activation                |
//! | 909 | Softmax      | Softmax activation             |
//! | 910 | CrossEntropy | Cross-entropy loss             |

pub mod parser;
pub mod executor;

// Re-export commonly used types
pub use parser::{LCBParser, LCBBatch, LCBInstruction, TensorData, DType, contracts};
pub use executor::{LCBExecutor, ExecutionResult};
