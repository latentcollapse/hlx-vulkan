#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// SOFTMAX - BACKWARD PASS
// =============================================================================
// Given forward: y = softmax(x)
// Computes: d_input[i] = y[i] * (d_output[i] - sum(d_output * y))
//
// Uses the softmax output from forward pass.
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Softmax output from forward pass
layout(binding = 0, std430) readonly buffer SoftmaxOutput {
    float softmax_out[];
};

// Gradient from upstream
layout(binding = 1, std430) readonly buffer GradOutput {
    float grad_output[];
};

// Gradient w.r.t. input (output)
layout(binding = 2, std430) writeonly buffer GradInput {
    float grad_input[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_rows;
    uint row_size;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared float shared_sum[BLOCK_SIZE];

// --- Main Entry Point ---
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    
    if (row >= params.num_rows) {
        return;
    }
    
    uint base_idx = row * params.row_size;
    
    // Step 1: Compute sum(d_output * softmax_output)
    float local_sum = 0.0;
    
    for (uint i = tid; i < params.row_size; i += BLOCK_SIZE) {
        local_sum += grad_output[base_idx + i] * softmax_out[base_idx + i];
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
    
    float dot_product = shared_sum[0];
    
    // Step 2: Compute gradient
    // d_input[i] = softmax_out[i] * (grad_output[i] - dot_product)
    for (uint i = tid; i < params.row_size; i += BLOCK_SIZE) {
        float y = softmax_out[base_idx + i];
        float dy = grad_output[base_idx + i];
        grad_input[base_idx + i] = y * (dy - dot_product);
    }
}
