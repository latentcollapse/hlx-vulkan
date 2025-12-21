#version 460 core

// =============================================================================
// GRADIENT REDUCTION PASS
// =============================================================================
// Entry point: main
// Purpose: Sum per-workgroup gradients in deterministic fixed order
// Workgroup: 1x1x1 (intentionally single-threaded for determinism)
//
// WHY SINGLE-THREADED:
// Floating point addition is not associative: (a + b) + c != a + (b + c)
// To guarantee bit-identical results across runs, we must sum in the exact
// same order every time. A parallel reduction would introduce non-determinism.
//
// PERFORMANCE NOTE:
// With typical num_workgroups < 1000, this loop takes <0.1ms.
// Don't optimize prematurely - determinism is the priority.
// =============================================================================

// --- Buffer Bindings ---

// Per-workgroup gradient staging buffer (read)
// Written by backward pass, each workgroup's sum is at workgroup_grads[wid]
layout(binding = 5, std430) readonly buffer WorkgroupGradients {
    float workgroup_grads[];
};

// Final accumulated parameter gradient (write)
layout(binding = 7, std430) writeonly buffer ParamGradient {
    float param_grad[];
};

// Push constants for runtime configuration
layout(push_constant) uniform PushConstants {
    uint num_workgroups;  // Number of workgroups from backward pass
    uint param_size;      // Number of parameters (unused in simplified v1)
} pc;

// --- Workgroup Configuration ---
// Single thread per dispatch - determinism over parallelism
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    uint param_idx = gl_GlobalInvocationID.x;
    
    // For v1 simplified model: single parameter gradient
    // Future: param_idx indexes into multi-parameter gradient array
    if (param_idx > 0) {
        return;
    }
    
    // Deterministic fixed-order summation
    // CRITICAL: This loop order must never change for reproducibility
    float total_grad = 0.0;
    
    for (uint wid = 0; wid < pc.num_workgroups; wid++) {
        total_grad += workgroup_grads[wid];
    }
    
    // Write final accumulated gradient
    param_grad[param_idx] = total_grad;
}
