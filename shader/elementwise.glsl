#version 460 core

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================
// Supports various element-wise ops via mode parameter:
//   0: add:   C = A + B
//   1: sub:   C = A - B
//   2: mul:   C = A * B
//   3: scale: C = A * scalar
//   4: add_scalar: C = A + scalar
//
// Used for residual connections, scaling, etc.
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

layout(binding = 0, std430) readonly buffer InputA {
    float A[];
};

layout(binding = 1, std430) readonly buffer InputB {
    float B[];
};

layout(binding = 2, std430) writeonly buffer Output {
    float C[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_elements;
    uint mode;       // Operation type
    float scalar;    // For scale/add_scalar modes
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    if (gid >= params.num_elements) {
        return;
    }
    
    float a = A[gid];
    float result;
    
    switch (params.mode) {
        case 0:  // add
            result = a + B[gid];
            break;
        case 1:  // sub
            result = a - B[gid];
            break;
        case 2:  // mul
            result = a * B[gid];
            break;
        case 3:  // scale
            result = a * params.scalar;
            break;
        case 4:  // add_scalar
            result = a + params.scalar;
            break;
        default:
            result = a;
    }
    
    C[gid] = result;
}
