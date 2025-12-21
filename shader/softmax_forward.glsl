#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// SOFTMAX - FORWARD PASS
// =============================================================================
// Computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//
// Numerically stable: subtracts max before exponentiating.
// Applied along the last dimension (typically attention scores over keys).
//
// Input shape: (batch * num_heads * seq_len, seq_len) - flattened attention scores
// Each workgroup handles one row (one query's attention over all keys)
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Input scores
layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

// Output probabilities
layout(binding = 1, std430) writeonly buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_rows;   // batch * num_heads * seq_len
    uint row_size;   // seq_len (softmax dimension)
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared float shared_max[BLOCK_SIZE];
shared float shared_sum[BLOCK_SIZE];

// --- Main Entry Point ---
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    
    if (row >= params.num_rows) {
        return;
    }
    
    uint base_idx = row * params.row_size;
    
    // Step 1: Find maximum (for numerical stability)
    float local_max = -3.402823466e+38;  // -FLT_MAX
    
    for (uint i = tid; i < params.row_size; i += BLOCK_SIZE) {
        local_max = max(local_max, input_data[base_idx + i]);
    }
    
    shared_max[tid] = local_max;
    barrier();
    memoryBarrierShared();
    
    // Parallel max reduction
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        barrier();
        memoryBarrierShared();
    }
    
    float row_max = shared_max[0];
    
    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0;
    
    for (uint i = tid; i < params.row_size; i += BLOCK_SIZE) {
        local_sum += exp(input_data[base_idx + i] - row_max);
    }
    
    shared_sum[tid] = local_sum;
    barrier();
    memoryBarrierShared();
    
    // Parallel sum reduction
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        barrier();
        memoryBarrierShared();
    }
    
    float row_sum = shared_sum[0];
    float inv_sum = 1.0 / row_sum;
    
    // Step 3: Compute and write softmax output
    for (uint i = tid; i < params.row_size; i += BLOCK_SIZE) {
        float val = exp(input_data[base_idx + i] - row_max) * inv_sum;
        output_data[base_idx + i] = val;
    }
}
