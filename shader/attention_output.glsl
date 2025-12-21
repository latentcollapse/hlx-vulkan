#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// ATTENTION OUTPUT
// =============================================================================
// Computes: Output = Attention_weights @ V
//
// Input shapes:
//   Attention: (batch * num_heads, seq_len, seq_len) - after softmax
//   V: (batch * num_heads, seq_len, head_dim)
// Output:
//   Context: (batch * num_heads, seq_len, head_dim)
// =============================================================================

#define TILE_SIZE 16

// --- Buffer Bindings ---

// Attention weights: (batch * num_heads * seq_len, seq_len)
layout(binding = 0, std430) readonly buffer Attention {
    float attention[];
};

// V: (batch * num_heads * seq_len, head_dim)
layout(binding = 1, std430) readonly buffer Value {
    float V[];
};

// Output context: (batch * num_heads * seq_len, head_dim)
layout(binding = 2, std430) writeonly buffer Context {
    float context[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint batch_heads;  // batch_size * num_heads
    uint seq_len;
    uint head_dim;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// --- Shared Memory ---
shared float tile_A[TILE_SIZE][TILE_SIZE];
shared float tile_V[TILE_SIZE][TILE_SIZE];

// --- Main Entry Point ---
void main() {
    uint batch_head_idx = gl_WorkGroupID.z;
    uint query_tile = gl_WorkGroupID.y;
    uint dim_tile = gl_WorkGroupID.x;
    
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;
    
    uint global_query = query_tile * TILE_SIZE + local_row;
    uint global_dim = dim_tile * TILE_SIZE + local_col;
    
    if (batch_head_idx >= params.batch_heads) {
        return;
    }
    
    uint attn_base = batch_head_idx * params.seq_len * params.seq_len;
    uint v_base = batch_head_idx * params.seq_len * params.head_dim;
    uint out_base = batch_head_idx * params.seq_len * params.head_dim;
    
    float sum = 0.0;
    
    uint num_tiles = (params.seq_len + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; t++) {
        // Load attention tile
        uint a_row = global_query;
        uint a_col = t * TILE_SIZE + local_col;
        
        if (a_row < params.seq_len && a_col < params.seq_len) {
            tile_A[local_row][local_col] = attention[attn_base + a_row * params.seq_len + a_col];
        } else {
            tile_A[local_row][local_col] = 0.0;
        }
        
        // Load V tile
        uint v_row = t * TILE_SIZE + local_row;
        uint v_col = global_dim;
        
        if (v_row < params.seq_len && v_col < params.head_dim) {
            tile_V[local_row][local_col] = V[v_base + v_row * params.head_dim + v_col];
        } else {
            tile_V[local_row][local_col] = 0.0;
        }
        
        barrier();
        memoryBarrierShared();
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[local_row][k] * tile_V[k][local_col];
        }
        
        barrier();
    }
    
    if (global_query < params.seq_len && global_dim < params.head_dim) {
        context[out_base + global_query * params.head_dim + global_dim] = sum;
    }
}
