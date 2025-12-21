#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// EMBEDDING - BACKWARD PASS
// =============================================================================
// Accumulates gradients for token and positional embeddings
//
// d_token_embedding[token_id] += d_output (for each occurrence of token_id)
// d_pos_embedding[pos] += d_output (summed over batch)
//
// Uses per-workgroup staging for deterministic accumulation.
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Token IDs (same as forward)
layout(binding = 0, std430) readonly buffer TokenIDs {
    uint token_ids[];
};

// Gradient from upstream: (batch * seq_len, d_model)
layout(binding = 1, std430) readonly buffer GradOutput {
    float grad_output[];
};

// Per-token gradient staging
// Since multiple positions may have same token, we stage per-position
// then reduce. Size: (batch * seq_len, d_model)
layout(binding = 2, std430) writeonly buffer TokenGradStaging {
    float token_grad_staging[];
};

// Positional embedding gradient staging
// Size: (num_workgroups, seq_len, d_model) - sum over batch
layout(binding = 3, std430) writeonly buffer PosGradStaging {
    float pos_grad_staging[];
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
    uint wid = gl_WorkGroupID.x;
    uint total_elements = params.batch_size * params.seq_len * params.d_model;
    
    if (gid >= total_elements) {
        return;
    }
    
    // Compute indices
    uint batch_seq_idx = gid / params.d_model;  // Which (batch, seq) position
    uint pos_in_seq = batch_seq_idx % params.seq_len;
    uint embed_dim = gid % params.d_model;
    
    float grad = grad_output[gid];
    
    // Stage token gradient (will need scatter-reduce by token_id later)
    token_grad_staging[gid] = grad;
    
    // Stage position gradient (will need sum over batch later)
    // For simplicity, just copy - reduction happens in separate pass
    pos_grad_staging[gid] = grad;
}
