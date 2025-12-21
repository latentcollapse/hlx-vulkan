#version 460 core

// =============================================================================
// EMBEDDING - FORWARD PASS
// =============================================================================
// Combines token embedding lookup with positional embedding
//
// Output = token_embedding[token_ids] + position_embedding[position_ids]
//
// Input: token_ids (batch, seq_len) as uint
// Output: embeddings (batch * seq_len, d_model)
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Token IDs: (batch * seq_len,) as uint32
layout(binding = 0, std430) readonly buffer TokenIDs {
    uint token_ids[];
};

// Token embedding table: (vocab_size, d_model)
layout(binding = 1, std430) readonly buffer TokenEmbedding {
    float token_embedding[];
};

// Positional embedding table: (max_seq_len, d_model)
layout(binding = 2, std430) readonly buffer PosEmbedding {
    float pos_embedding[];
};

// Output embeddings: (batch * seq_len, d_model)
layout(binding = 3, std430) writeonly buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint seq_len;
    uint d_model;
    uint vocab_size;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total_elements = params.batch_size * params.seq_len * params.d_model;
    
    if (gid >= total_elements) {
        return;
    }
    
    // Compute indices
    uint pos_in_seq = (gid / params.d_model) % params.seq_len;  // Position within sequence
    uint token_idx = gid / params.d_model;                       // Which token (flattened)
    uint embed_dim = gid % params.d_model;                       // Which embedding dimension
    
    // Lookup token ID
    uint token_id = token_ids[token_idx];
    
    // Bounds check token ID
    if (token_id >= params.vocab_size) {
        token_id = 0;  // Default to first token if out of bounds
    }
    
    // Lookup embeddings and combine
    float tok_emb = token_embedding[token_id * params.d_model + embed_dim];
    float pos_emb = pos_embedding[pos_in_seq * params.d_model + embed_dim];
    
    output_data[gid] = tok_emb + pos_emb;
}
