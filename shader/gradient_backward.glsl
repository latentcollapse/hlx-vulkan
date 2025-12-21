#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// GRADIENT BACKWARD PASS
// =============================================================================
// Entry point: main
// Purpose: Compute gradients, accumulate to per-workgroup staging buffer
// Workgroup: 256x1x1
// 
// CRITICAL: No cross-workgroup atomics. Each workgroup writes to its own
// region of workgroup_grads. Reduction kernel sums in fixed order later.
// This guarantees deterministic gradient accumulation.
// =============================================================================

// --- Buffer Bindings ---

// Input tensor (read, needed for param gradient computation)
layout(binding = 0, std430) readonly buffer InputTensor {
    float input_data[];
};

// Stored activations from forward pass (read)
layout(binding = 2, std430) readonly buffer ActivationBuffer {
    float activations[];
};

// Input gradients (write, propagates to previous layer)
layout(binding = 3, std430) writeonly buffer InputGradient {
    float input_grad[];
};

// Output gradients from upstream (read, e.g., from loss function)
layout(binding = 4, std430) readonly buffer OutputGradient {
    float output_grad[];
};

// Per-workgroup gradient staging buffer (write)
// Layout: workgroup_grads[workgroup_id * param_size + param_idx]
// This avoids cross-workgroup atomics for determinism
layout(binding = 5, std430) writeonly buffer WorkgroupGradients {
    float workgroup_grads[];
};

// Parameters
layout(binding = 6, std140) uniform Parameters {
    uint input_size;
    uint output_size;
    uint batch_size;
    float learning_rate;
};

// Learnable weight (binding 8) - needed for input gradient chain rule
layout(binding = 8, std430) readonly buffer WeightBuffer {
    float weight;
};

// Push constant for runtime values
layout(push_constant) uniform PushConstants {
    uint num_workgroups;
    uint param_size;
} pc;

// --- Workgroup Configuration ---
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory for Workgroup Accumulation ---
// Each thread contributes its gradient; thread 0 sums and writes
shared float local_grad_accumulator[256];

// --- Activation Derivative ---
// d/dx ReLU(x) = 1 if x > 0 else 0
float activation_derivative(float activated_value) {
    return activated_value > 0.0 ? 1.0 : 0.0;
}

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;

    // Initialize shared memory
    local_grad_accumulator[lid] = 0.0;

    // Bounds check
    if (gid < input_size) {
        // 1. Read upstream gradient (from loss or next layer)
        float upstream_grad = output_grad[gid];

        // 2. Read stored activation from forward pass
        float activated = activations[gid];

        // 3. Compute input gradient via chain rule
        //    For y = weight * ReLU(x):
        //    d(loss)/d(x) = d(loss)/d(y) * d(y)/d(x) = upstream_grad * weight * ReLU'(x)
        float activation_deriv = activation_derivative(activated);
        float input_gradient = upstream_grad * weight * activation_deriv;

        // 4. Write input gradient (for previous layer's backward pass)
        input_grad[gid] = input_gradient;

        // 5. Compute parameter (weight) gradient contribution
        //    For y = weight * ReLU(x):
        //    d(loss)/d(weight) = d(loss)/d(y) * d(y)/d(weight) = upstream_grad * ReLU(x)
        //    = upstream_grad * activated
        float param_grad_contribution = upstream_grad * activated;

        // Store in shared memory for workgroup reduction
        local_grad_accumulator[lid] = param_grad_contribution;
    }
    
    // Synchronize all threads in workgroup
    barrier();
    memoryBarrierShared();
    
    // Thread 0 performs workgroup-local reduction and writes to staging buffer
    if (lid == 0) {
        float workgroup_sum = 0.0;
        
        // Sum all contributions in this workgroup
        // Fixed iteration order guarantees determinism within workgroup
        for (uint i = 0; i < 256; i++) {
            workgroup_sum += local_grad_accumulator[i];
        }
        
        // Write to per-workgroup staging buffer
        // Each workgroup writes to its own slot, no contention
        // Reduction kernel will sum these in fixed order
        workgroup_grads[wid] = workgroup_sum;
    }
    
    // Final memory barrier to ensure staging buffer is visible to reduce kernel
    memoryBarrierBuffer();
    barrier();
}
