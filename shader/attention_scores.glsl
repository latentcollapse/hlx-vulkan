#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// BATCHED ATTENTION SCORES
// =============================================================================
// Computes attention scores: Scores = (Q @ K^T) / sqrt(head_dim)
// With optional causal masking.
//
// Input shapes (after head split):
//   Q: (batch * num_heads, seq_len, head_dim)
//   K: (batch * num_heads, seq_len, head_dim)
// Output:
//   Scores: (batch * num_heads, seq_len, seq_len)
//
// This is a batched matrix multiply with scaling and optional mask.
// =============================================================================

#define TILE_SIZE 16

// --- Buffer Bindings ---

// Q: (batch * num_heads * seq_len, head_dim)
layout(binding = 0, std430) readonly buffer Query {
    float Q[];
};

// K: (batch * num_heads * seq_len, head_dim)
layout(binding = 1, std430) readonly buffer Key {
    float K[];
};

// Output scores: (batch * num_heads * seq_len, seq_len)
layout(binding = 2, std430) writeonly buffer Scores {
    float scores[];
};

// Causal mask bias: (seq_len, seq_len) - add -inf for masked positions
layout(binding = 3, std430) readonly buffer CausalBias {
    float causal_bias[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint batch_heads;  // batch_size * num_heads
    uint seq_len;
    uint head_dim;
    float scale;       // 1.0 / sqrt(head_dim)
    uint use_causal;   // 1 for causal masking
} params;

// --- Workgroup Configuration ---
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// --- Shared Memory ---
shared float tile_Q[TILE_SIZE][TILE_SIZE];
shared float tile_K[TILE_SIZE][TILE_SIZE];

// --- Main Entry Point ---
void main() {
    // Each workgroup computes a TILE_SIZE × TILE_SIZE block of one attention matrix
    uint batch_head_idx = gl_WorkGroupID.z;  // Which (batch, head)
    uint query_tile = gl_WorkGroupID.y;      // Which query block
    uint key_tile = gl_WorkGroupID.x;        // Which key block
    
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;
    
    uint global_query = query_tile * TILE_SIZE + local_row;
    uint global_key = key_tile * TILE_SIZE + local_col;
    
    if (batch_head_idx >= params.batch_heads) {
        return;
    }
    
    // Base indices into Q and K for this batch/head
    uint q_base = batch_head_idx * params.seq_len * params.head_dim;
    uint k_base = batch_head_idx * params.seq_len * params.head_dim;
    uint out_base = batch_head_idx * params.seq_len * params.seq_len;
    
    float sum = 0.0;
    
    // Tiled matrix multiply: Q[query, :] @ K[key, :]^T
    uint num_tiles = (params.head_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; t++) {
        // Load Q tile
        uint q_row = global_query;
        uint q_col = t * TILE_SIZE + local_col;
        
        if (q_row < params.seq_len && q_col < params.head_dim) {
            tile_Q[local_row][local_col] = Q[q_base + q_row * params.head_dim + q_col];
        } else {
            tile_Q[local_row][local_col] = 0.0;
        }
        
        // Load K tile (transposed access)
        uint k_row = global_key;
        uint k_col = t * TILE_SIZE + local_row;
        
        if (k_row < params.seq_len && k_col < params.head_dim) {
            tile_K[local_row][local_col] = K[k_base + k_row * params.head_dim + k_col];
        } else {
            tile_K[local_row][local_col] = 0.0;
        }
        
        barrier();
        memoryBarrierShared();
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tile_Q[local_row][k] * tile_K[k][local_col];
        }
        
        barrier();
    }
    
    // Apply scaling
    sum *= params.scale;
    
    // Apply causal mask if enabled
    if (params.use_causal != 0 && global_query < params.seq_len && global_key < params.seq_len) {
        sum += causal_bias[global_query * params.seq_len + global_key];
    }
    
    // Write output
    if (global_query < params.seq_len && global_key < params.seq_len) {
        scores[out_base + global_query * params.seq_len + global_key] = sum;
    }
}
