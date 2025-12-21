#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// LAYER NORMALIZATION - FORWARD PASS
// =============================================================================
// Computes: output = (input - mean) / sqrt(variance + eps) * gamma + beta
//
// For input of shape (batch, seq_len, d_model), normalizes over d_model dimension.
// Each workgroup handles one (batch, seq_len) position.
//
// Determinism: Uses fixed-order reduction for mean and variance
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Input: (batch * seq_len, d_model) - flattened
layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

// Output: same shape as input
layout(binding = 1, std430) writeonly buffer Output {
    float output_data[];
};

// Gamma (scale): (d_model,)
layout(binding = 2, std430) readonly buffer Gamma {
    float gamma[];
};

// Beta (shift): (d_model,)
layout(binding = 3, std430) readonly buffer Beta {
    float beta[];
};

// Store mean and inv_std for backward pass
layout(binding = 4, std430) writeonly buffer Stats {
    float stats[];  // [mean, inv_std] per position, so size = 2 * batch * seq_len
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_positions;  // batch * seq_len
    uint d_model;        // normalization dimension
    float eps;           // epsilon for numerical stability
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory for Reduction ---
shared float shared_sum[BLOCK_SIZE];
shared float shared_sum_sq[BLOCK_SIZE];

// --- Main Entry Point ---
void main() {
    uint pos = gl_WorkGroupID.x;  // Which (batch, seq_len) position
    uint tid = gl_LocalInvocationID.x;
    
    if (pos >= params.num_positions) {
        return;
    }
    
    uint base_idx = pos * params.d_model;
    
    // Step 1: Compute partial sums for mean
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    
    // Each thread handles multiple elements if d_model > BLOCK_SIZE
    for (uint i = tid; i < params.d_model; i += BLOCK_SIZE) {
        float val = input_data[base_idx + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    
    barrier();
    memoryBarrierShared();
    
    // Step 2: Parallel reduction (deterministic - fixed tree order)
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        barrier();
        memoryBarrierShared();
    }
    
    // Step 3: Compute mean and variance (thread 0)
    float mean = shared_sum[0] / float(params.d_model);
    float variance = shared_sum_sq[0] / float(params.d_model) - mean * mean;
    float inv_std = 1.0 / sqrt(variance + params.eps);
    
    // Broadcast to shared memory for other threads
    if (tid == 0) {
        shared_sum[0] = mean;
        shared_sum_sq[0] = inv_std;
    }
    
    barrier();
    memoryBarrierShared();
    
    mean = shared_sum[0];
    inv_std = shared_sum_sq[0];
    
    // Step 4: Normalize and apply affine transform
    for (uint i = tid; i < params.d_model; i += BLOCK_SIZE) {
        float val = input_data[base_idx + i];
        float normalized = (val - mean) * inv_std;
        output_data[base_idx + i] = normalized * gamma[i] + beta[i];
    }
    
    // Step 5: Save stats for backward pass
    if (tid == 0) {
        stats[pos * 2] = mean;
        stats[pos * 2 + 1] = inv_std;
    }
}
