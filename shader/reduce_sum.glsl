#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// SUM REDUCTION
// =============================================================================
// Computes sum of elements in deterministic fixed order.
// Used for averaging loss values, computing means, etc.
//
// Two-phase reduction:
// 1. Each workgroup reduces its portion to shared memory
// 2. Thread 0 writes partial sum to staging buffer
// 3. Final single-threaded pass sums staging buffer
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

// Per-workgroup partial sums
layout(binding = 1, std430) writeonly buffer PartialSums {
    float partial_sums[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_elements;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared float shared_sum[BLOCK_SIZE];

// --- Main Entry Point ---
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    uint gid = gl_GlobalInvocationID.x;
    
    // Each thread sums its assigned elements
    float local_sum = 0.0;
    
    for (uint i = gid; i < params.num_elements; i += gl_NumWorkGroups.x * BLOCK_SIZE) {
        local_sum += input_data[i];
    }
    
    shared_sum[tid] = local_sum;
    barrier();
    memoryBarrierShared();
    
    // Parallel reduction within workgroup
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        barrier();
        memoryBarrierShared();
    }
    
    // Thread 0 writes workgroup's partial sum
    if (tid == 0) {
        partial_sums[wid] = shared_sum[0];
    }
}
