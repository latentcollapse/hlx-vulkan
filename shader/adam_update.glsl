#version 460 core

// =============================================================================
// ADAM OPTIMIZER UPDATE
// =============================================================================
// Updates parameters using Adam algorithm:
//
// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
// v_t = beta2 * v_{t-1} + (1 - beta2) * grad²
// m_hat = m_t / (1 - beta1^t)  (bias correction)
// v_hat = v_t / (1 - beta2^t)
// param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// All operations are element-wise, highly parallel.
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Parameters to update (read-write)
layout(binding = 0, std430) buffer Parameters {
    float params_data[];
};

// Gradients (read-only)
layout(binding = 1, std430) readonly buffer Gradients {
    float grads[];
};

// First moment (momentum) - read-write
layout(binding = 2, std430) buffer Moment1 {
    float m[];
};

// Second moment (adaptive LR) - read-write
layout(binding = 3, std430) buffer Moment2 {
    float v[];
};

// Hyperparameters
layout(push_constant) uniform PushConstants {
    uint num_params;
    float lr;           // Learning rate
    float beta1;        // First moment decay (0.9)
    float beta2;        // Second moment decay (0.999)
    float eps;          // Numerical stability (1e-8)
    float beta1_t;      // beta1^t (precomputed)
    float beta2_t;      // beta2^t (precomputed)
} hp;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Main Entry Point ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    if (gid >= hp.num_params) {
        return;
    }
    
    // Load current values
    float param = params_data[gid];
    float grad = grads[gid];
    float m_prev = m[gid];
    float v_prev = v[gid];
    
    // Update moments
    float m_new = hp.beta1 * m_prev + (1.0 - hp.beta1) * grad;
    float v_new = hp.beta2 * v_prev + (1.0 - hp.beta2) * grad * grad;
    
    // Bias correction
    float m_hat = m_new / (1.0 - hp.beta1_t);
    float v_hat = v_new / (1.0 - hp.beta2_t);
    
    // Parameter update
    float update = hp.lr * m_hat / (sqrt(v_hat) + hp.eps);
    float param_new = param - update;
    
    // Write back
    params_data[gid] = param_new;
    m[gid] = m_new;
    v[gid] = v_new;
}
