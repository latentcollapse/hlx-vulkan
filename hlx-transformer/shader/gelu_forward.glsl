#version 460 core

// =============================================================================
// GELU ACTIVATION - FORWARD PASS
// =============================================================================
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Smoother than ReLU, better gradient flow for transformers.
// =============================================================================

#define BLOCK_SIZE 256

// Constants for GELU
const float SQRT_2_OVER_PI = 0.7978845608028654;  // sqrt(2/π)
const float GELU_COEF = 0.044715;

// --- Buffer Bindings ---

layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

layout(binding = 1, std430) writeonly buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_elements;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- GELU Function ---
float gelu(float x) {
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    return x * 0.5 * (1.0 + tanh(inner));
}

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    if (gid >= params.num_elements) {
        return;
    }
    
    output_data[gid] = gelu(input_data[gid]);
}
