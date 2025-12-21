#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// GEMM BACKWARD PASS
// =============================================================================
// Given forward: C = A × B
// Computes:
//   dA = dC × B^T  (gradient w.r.t. A)
//   dB = A^T × dC  (gradient w.r.t. B)
//
// This shader computes ONE of these based on the `mode` push constant:
//   mode = 0: Compute dA = dC × B^T
//   mode = 1: Compute dB = A^T × dC
//
// Reuses tiled multiplication for efficiency.
// =============================================================================

#define TILE_SIZE 16

// --- Buffer Bindings ---

// For mode=0 (compute dA):
//   input1 = dC (M × N)
//   input2 = B  (K × N) - will be transposed
//   output = dA (M × K)
//
// For mode=1 (compute dB):
//   input1 = A  (M × K) - will be transposed
//   input2 = dC (M × N)
//   output = dB (K × N)

layout(binding = 0, std430) readonly buffer Input1 {
    float input1[];
};

layout(binding = 1, std430) readonly buffer Input2 {
    float input2[];
};

layout(binding = 2, std430) writeonly buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint M;           // Original forward M
    uint K;           // Original forward K
    uint N;           // Original forward N
    uint mode;        // 0 = compute dA, 1 = compute dB
} params;

// --- Workgroup Configuration ---
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// --- Shared Memory ---
shared float tile_1[TILE_SIZE][TILE_SIZE];
shared float tile_2[TILE_SIZE][TILE_SIZE];

// --- Main Entry Point ---
void main() {
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;
    
    float sum = 0.0;
    
    if (params.mode == 0) {
        // Mode 0: dA = dC × B^T
        // dC is (M × N), B^T is (N × K), dA is (M × K)
        
        uint out_M = params.M;
        uint out_K = params.K;
        uint inner = params.N;  // Contract over N
        
        uint global_row = gl_WorkGroupID.y * TILE_SIZE + local_row;  // row in dA
        uint global_col = gl_WorkGroupID.x * TILE_SIZE + local_col;  // col in dA
        
        uint num_tiles = (inner + TILE_SIZE - 1) / TILE_SIZE;
        
        for (uint t = 0; t < num_tiles; t++) {
            // Load tile of dC (M × N)
            uint dc_row = global_row;
            uint dc_col = t * TILE_SIZE + local_col;
            
            if (dc_row < out_M && dc_col < inner) {
                tile_1[local_row][local_col] = input1[dc_row * inner + dc_col];
            } else {
                tile_1[local_row][local_col] = 0.0;
            }
            
            // Load tile of B^T (which is B transposed)
            // B is (K × N), so B^T[i][j] = B[j][i]
            // We want B^T row = (t * TILE_SIZE + local_row), col = global_col
            uint bt_row = t * TILE_SIZE + local_row;  // = column in B
            uint bt_col = global_col;                  // = row in B
            
            if (bt_row < inner && bt_col < out_K) {
                // B^T[bt_row][bt_col] = B[bt_col][bt_row]
                tile_2[local_row][local_col] = input2[bt_col * inner + bt_row];
            } else {
                tile_2[local_row][local_col] = 0.0;
            }
            
            barrier();
            memoryBarrierShared();
            
            for (uint k = 0; k < TILE_SIZE; k++) {
                sum += tile_1[local_row][k] * tile_2[k][local_col];
            }
            
            barrier();
        }
        
        // Write dA
        if (global_row < out_M && global_col < out_K) {
            output_data[global_row * out_K + global_col] = sum;
        }
        
    } else {
        // Mode 1: dB = A^T × dC
        // A^T is (K × M), dC is (M × N), dB is (K × N)
        
        uint out_K = params.K;
        uint out_N = params.N;
        uint inner = params.M;  // Contract over M
        
        uint global_row = gl_WorkGroupID.y * TILE_SIZE + local_row;  // row in dB (= col in A)
        uint global_col = gl_WorkGroupID.x * TILE_SIZE + local_col;  // col in dB
        
        uint num_tiles = (inner + TILE_SIZE - 1) / TILE_SIZE;
        
        for (uint t = 0; t < num_tiles; t++) {
            // Load tile of A^T (which is A transposed)
            // A is (M × K), so A^T[i][j] = A[j][i]
            uint at_row = global_row;                   // = col in A
            uint at_col = t * TILE_SIZE + local_col;   // = row in A
            
            if (at_row < out_K && at_col < inner) {
                // A^T[at_row][at_col] = A[at_col][at_row]
                tile_1[local_row][local_col] = input1[at_col * out_K + at_row];
            } else {
                tile_1[local_row][local_col] = 0.0;
            }
            
            // Load tile of dC (M × N)
            uint dc_row = t * TILE_SIZE + local_row;
            uint dc_col = global_col;
            
            if (dc_row < inner && dc_col < out_N) {
                tile_2[local_row][local_col] = input2[dc_row * out_N + dc_col];
            } else {
                tile_2[local_row][local_col] = 0.0;
            }
            
            barrier();
            memoryBarrierShared();
            
            for (uint k = 0; k < TILE_SIZE; k++) {
                sum += tile_1[local_row][k] * tile_2[k][local_col];
            }
            
            barrier();
        }
        
        // Write dB
        if (global_row < out_K && global_col < out_N) {
            output_data[global_row * out_N + global_col] = sum;
        }
    }
}
