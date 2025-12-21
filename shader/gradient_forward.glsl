#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// GRADIENT FORWARD PASS
// =============================================================================
// Entry point: main
// Purpose: Compute forward activation, store for backward pass
// Workgroup: 256x1x1 (optimal warp utilization)
// =============================================================================

// --- Buffer Bindings (must match descriptor set layout in Rust) ---

// Input tensor (read-only)
layout(binding = 0, std430) readonly buffer InputTensor {
    float input_data[];
};

// Output tensor (write)
layout(binding = 1, std430) writeonly buffer OutputTensor {
    float output_data[];
};

// Activation storage (write, read by backward pass)
layout(binding = 2, std430) writeonly buffer ActivationBuffer {
    float activations[];
};

// Parameters (uniform block, std140 for compatibility)
layout(binding = 6, std140) uniform Parameters {
    uint input_size;
    uint output_size;
    uint batch_size;
    float learning_rate;
};

// Learnable weight (binding 8)
layout(binding = 8, std430) readonly buffer WeightBuffer {
    float weight;  // Single scalar weight for v1
};

// Target values for loss computation
layout(binding = 9, std430) readonly buffer TargetBuffer {
    float target_data[];
};

// --- Workgroup Configuration ---
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// --- Activation Function ---
// ReLU for v1; specialization constants for others in future
float activation_fn(float x) {
    return max(0.0, x);
}

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;

    // Bounds check (workgroup may exceed tensor size)
    if (gid >= input_size) {
        return;
    }

    // 1. Read input
    float input_val = input_data[gid];

    // 2. Compute activation
    float activated = activation_fn(input_val);

    // 3. Store activation for backward pass
    //    This is the critical state that backward needs
    activations[gid] = activated;

    // 4. Apply learnable weight: output = weight * activated
    float output_val = weight * activated;

    // 5. Write output
    output_data[gid] = output_val;

    // 6. Memory barrier for deterministic ordering
    //    Ensures all threads in workgroup complete writes before proceeding
    memoryBarrierBuffer();
    barrier();
}
