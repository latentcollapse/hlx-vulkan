#version 460 core

// =============================================================================
// CROSS-ENTROPY LOSS - BACKWARD PASS
// =============================================================================
// Given: loss = -log(softmax(logits)[target])
// Gradient: d_logits[i] = softmax[i] - (1 if i == target else 0)
//
// This is the elegant form: gradient is simply (predicted - actual).
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Softmax output from forward pass
layout(binding = 0, std430) readonly buffer Softmax {
    float softmax_out[];
};

// Target token IDs
layout(binding = 1, std430) readonly buffer Targets {
    uint targets[];
};

// Gradient w.r.t. logits
layout(binding = 2, std430) writeonly buffer GradLogits {
    float grad_logits[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_positions;
    uint vocab_size;
    uint ignore_index;
    float scale;  // Usually 1.0 / num_valid_positions for averaging
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total_elements = params.num_positions * params.vocab_size;
    
    if (gid >= total_elements) {
        return;
    }
    
    uint pos = gid / params.vocab_size;
    uint vocab_idx = gid % params.vocab_size;
    uint target = targets[pos];
    
    // Check if this position should be ignored
    if (target == params.ignore_index || target >= params.vocab_size) {
        grad_logits[gid] = 0.0;
        return;
    }
    
    // Gradient = softmax - one_hot(target)
    float grad = softmax_out[gid];
    if (vocab_idx == target) {
        grad -= 1.0;
    }
    
    // Scale for averaging over valid positions
    grad_logits[gid] = grad * params.scale;
}
