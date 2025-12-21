#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// CROSS-ENTROPY LOSS - FORWARD PASS
// =============================================================================
// Computes: loss = -log(softmax(logits)[target])
//
// Uses log-softmax for numerical stability:
// log_softmax(x)[i] = x[i] - max(x) - log(sum(exp(x - max(x))))
// loss = -log_softmax(logits)[target]
//
// Input: logits (batch * seq_len, vocab_size)
//        targets (batch * seq_len,) as uint32
// Output: losses per position, then reduced to scalar
// =============================================================================

#define BLOCK_SIZE 256

// --- Buffer Bindings ---

// Logits: (num_positions, vocab_size)
layout(binding = 0, std430) readonly buffer Logits {
    float logits[];
};

// Target token IDs: (num_positions,)
layout(binding = 1, std430) readonly buffer Targets {
    uint targets[];
};

// Per-position losses (will be reduced later)
layout(binding = 2, std430) writeonly buffer Losses {
    float losses[];
};

// Also output softmax probabilities for backward pass
layout(binding = 3, std430) writeonly buffer Softmax {
    float softmax_out[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint num_positions;  // batch * seq_len
    uint vocab_size;
    uint ignore_index;   // Token ID to ignore (e.g., padding), use 0xFFFFFFFF for none
} params;

// --- Workgroup Configuration ---
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared float shared_max[BLOCK_SIZE];
shared float shared_sum[BLOCK_SIZE];

// --- Main Entry Point ---
void main() {
    uint pos = gl_WorkGroupID.x;  // Which position
    uint tid = gl_LocalInvocationID.x;
    
    if (pos >= params.num_positions) {
        return;
    }
    
    uint target = targets[pos];
    uint base_idx = pos * params.vocab_size;
    
    // Check if this position should be ignored
    if (target == params.ignore_index || target >= params.vocab_size) {
        // Write zero loss and uniform softmax
        if (tid == 0) {
            losses[pos] = 0.0;
        }
        for (uint i = tid; i < params.vocab_size; i += BLOCK_SIZE) {
            softmax_out[base_idx + i] = 1.0 / float(params.vocab_size);
        }
        return;
    }
    
    // Step 1: Find max logit
    float local_max = -3.402823466e+38;
    for (uint i = tid; i < params.vocab_size; i += BLOCK_SIZE) {
        local_max = max(local_max, logits[base_idx + i]);
    }
    
    shared_max[tid] = local_max;
    barrier();
    memoryBarrierShared();
    
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        barrier();
        memoryBarrierShared();
    }
    
    float max_logit = shared_max[0];
    
    // Step 2: Compute sum of exp(logits - max)
    float local_sum = 0.0;
    for (uint i = tid; i < params.vocab_size; i += BLOCK_SIZE) {
        local_sum += exp(logits[base_idx + i] - max_logit);
    }
    
    shared_sum[tid] = local_sum;
    barrier();
    memoryBarrierShared();
    
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        barrier();
        memoryBarrierShared();
    }
    
    float sum_exp = shared_sum[0];
    float log_sum_exp = log(sum_exp);
    
    // Step 3: Compute softmax and write
    for (uint i = tid; i < params.vocab_size; i += BLOCK_SIZE) {
        float prob = exp(logits[base_idx + i] - max_logit) / sum_exp;
        softmax_out[base_idx + i] = prob;
    }
    
    // Step 4: Compute loss for this position
    // loss = -(logit[target] - max - log_sum_exp) = -log_softmax[target]
    if (tid == 0) {
        float target_logit = logits[base_idx + target];
        float log_prob = target_logit - max_logit - log_sum_exp;
        losses[pos] = -log_prob;  // Negative log probability
    }
}
