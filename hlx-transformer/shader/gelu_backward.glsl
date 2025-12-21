#version 460 core

// =============================================================================
// GELU ACTIVATION - BACKWARD PASS
// =============================================================================
// GELU(x) = x * 0.5 * (1 + tanh(inner))
// where inner = sqrt(2/π) * (x + 0.044715 * x³)
//
// Derivative:
// d/dx GELU(x) = 0.5 * (1 + tanh(inner)) 
//              + 0.5 * x * sech²(inner) * sqrt(2/π) * (1 + 3 * 0.044715 * x²)
// =============================================================================

#define BLOCK_SIZE 256

const float SQRT_2_OVER_PI = 0.7978845608028654;
const float GELU_COEF = 0.044715;

// --- Buffer Bindings ---

// Input from forward pass (needed to compute derivative)
layout(binding = 0, std430) readonly buffer Input {
    float input_data[];
};

// Gradient from upstream
layout(binding = 1, std430) readonly buffer GradOutput {
    float grad_output[];
};

// Gradient w.r.t. input
layout(binding = 2, std430) writeonly buffer GradInput {
    float grad_input[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_elements;
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- GELU Derivative ---
float gelu_derivative(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    float tanh_inner = tanh(inner);
    
    // sech²(inner) = 1 - tanh²(inner)
    float sech2_inner = 1.0 - tanh_inner * tanh_inner;
    
    // d(inner)/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
    float d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEF * x2);
    
    // Full derivative
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner;
}

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    if (gid >= params.num_elements) {
        return;
    }
    
    float x = input_data[gid];
    float dy = grad_output[gid];
    grad_input[gid] = dy * gelu_derivative(x);
}
