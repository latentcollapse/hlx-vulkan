#version 460 core

// =============================================================================
// FINAL SUM REDUCTION
// =============================================================================
// Sums partial results from workgroups in deterministic order.
// Single-threaded to guarantee reproducibility.
// =============================================================================

// --- Buffer Bindings ---

layout(binding = 0, std430) readonly buffer PartialSums {
    float partial_sums[];
};

layout(binding = 1, std430) writeonly buffer Output {
    float result;
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_partials;
    float scale;  // Optional scaling factor (e.g., 1/N for mean)
} params;

// --- Workgroup Configuration ---
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    // Single-threaded, fixed-order summation
    float sum = 0.0;
    
    for (uint i = 0; i < params.num_partials; i++) {
        sum += partial_sums[i];
    }
    
    result = sum * params.scale;
}
