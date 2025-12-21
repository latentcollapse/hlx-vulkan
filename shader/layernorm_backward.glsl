#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// LAYER NORMALIZATION - BACKWARD PASS
// =============================================================================
// Given forward: y = (x - mean) / std * gamma + beta
// Computes:
//   d_input: gradient w.r.t. input
//   d_gamma: gradient w.r.t. gamma (accumulated)
//   d_beta:  gradient w.r.t. beta (accumulated)
//
// Uses saved mean and inv_std from forward pass.
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Input from forward pass (needed for normalized computation)
layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

// Gradient from upstream
layout(binding = 1, std430) readonly buffer GradOutput {
    float grad_output[];
};

// Saved stats from forward: [mean, inv_std] per position
layout(binding = 2, std430) readonly buffer Stats {
    float stats[];
};

// Gamma (scale)
layout(binding = 3, std430) readonly buffer Gamma {
    float gamma[];
};

// Output: gradient w.r.t. input
layout(binding = 4, std430) writeonly buffer GradInput {
    float grad_input[];
};

// Workgroup-local gamma gradients (staged for deterministic reduction)
layout(binding = 5, std430) buffer WorkgroupGradGamma {
    float wg_grad_gamma[];  // Size: num_workgroups * d_model
};

// Workgroup-local beta gradients
layout(binding = 6, std430) buffer WorkgroupGradBeta {
    float wg_grad_beta[];  // Size: num_workgroups * d_model
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_positions;  // batch * seq_len
    uint d_model;
    float eps;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared float shared_sum1[BLOCK_SIZE];  // For d_gamma and intermediate reductions
shared float shared_sum2[BLOCK_SIZE];  // For d_beta and intermediate reductions

// --- Main Entry Point ---
void main() {
    uint pos = gl_WorkGroupID.x;  // Which position
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    
    if (pos >= params.num_positions) {
        return;
    }
    
    uint base_idx = pos * params.d_model;
    
    // Load saved statistics
    float mean = stats[pos * 2];
    float inv_std = stats[pos * 2 + 1];
    
    // Step 1: Compute intermediate sums for input gradient
    // d_input = inv_std * (d_y * gamma - mean(d_y * gamma) 
    //           - normalized * mean(d_y * gamma * normalized))
    
    float local_dgamma_sum = 0.0;  // sum of d_y * gamma
    float local_dgamma_norm_sum = 0.0;  // sum of d_y * gamma * normalized
    
    for (uint i = tid; i < params.d_model; i += BLOCK_SIZE) {
        float x = input_data[base_idx + i];
        float normalized = (x - mean) * inv_std;
        float dy = grad_output[base_idx + i];
        float dy_gamma = dy * gamma[i];
        
        local_dgamma_sum += dy_gamma;
        local_dgamma_norm_sum += dy_gamma * normalized;
    }
    
    shared_sum1[tid] = local_dgamma_sum;
    shared_sum2[tid] = local_dgamma_norm_sum;
    
    barrier();
    memoryBarrierShared();
    
    // Parallel reduction
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum1[tid] += shared_sum1[tid + stride];
            shared_sum2[tid] += shared_sum2[tid + stride];
        }
        barrier();
        memoryBarrierShared();
    }
    
    float mean_dgamma = shared_sum1[0] / float(params.d_model);
    float mean_dgamma_norm = shared_sum2[0] / float(params.d_model);
    
    // Step 2: Compute and write input gradient
    for (uint i = tid; i < params.d_model; i += BLOCK_SIZE) {
        float x = input_data[base_idx + i];
        float normalized = (x - mean) * inv_std;
        float dy = grad_output[base_idx + i];
        float dy_gamma = dy * gamma[i];
        
        // d_input = inv_std * (dy_gamma - mean_dgamma - normalized * mean_dgamma_norm)
        float dx = inv_std * (dy_gamma - mean_dgamma - normalized * mean_dgamma_norm);
        grad_input[base_idx + i] = dx;
    }
    
    // Step 3: Compute local gamma and beta gradients for this position
    // These need to be accumulated across all positions
    float local_dgamma = 0.0;
    float local_dbeta = 0.0;
    
    for (uint i = tid; i < params.d_model; i += BLOCK_SIZE) {
        float x = input_data[base_idx + i];
        float normalized = (x - mean) * inv_std;
        float dy = grad_output[base_idx + i];
        
        // Accumulate into workgroup staging buffer
        // We write per-position contributions, then reduce later
        
        // For thread 0 to write this position's contribution
        shared_sum1[tid] = dy * normalized;  // d_gamma contribution
        shared_sum2[tid] = dy;               // d_beta contribution
        
        barrier();
        memoryBarrierShared();
        
        // Reduce within workgroup for this feature index
        // Only thread 0 writes to staging buffer
        if (tid == 0) {
            float sum_dgamma = 0.0;
            float sum_dbeta = 0.0;
            for (uint j = 0; j < min(BLOCK_SIZE, params.d_model - i); j++) {
                // This would need adjustment for proper accumulation
                // Simplified: each position contributes to staging
            }
        }
    }
    
    // Write this workgroup's contribution to staging
    // The reduction kernel will sum across all workgroups
    if (tid < params.d_model) {
        float x = input_data[base_idx + tid];
        float normalized = (x - mean) * inv_std;
        float dy = grad_output[base_idx + tid];
        
        wg_grad_gamma[wid * params.d_model + tid] = dy * normalized;
        wg_grad_beta[wid * params.d_model + tid] = dy;
    }
    
    // Handle case where d_model > BLOCK_SIZE
    for (uint i = tid + BLOCK_SIZE; i < params.d_model; i += BLOCK_SIZE) {
        float x = input_data[base_idx + i];
        float normalized = (x - mean) * inv_std;
        float dy = grad_output[base_idx + i];
        
        // Atomic add to staging (within workgroup region)
        wg_grad_gamma[wid * params.d_model + i] = dy * normalized;
        wg_grad_beta[wid * params.d_model + i] = dy;
    }
}
