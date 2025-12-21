#version 460 core
#extension GL_KHR_memory_scope_semantics : require

// =============================================================================
// GENERAL MATRIX MULTIPLY (GEMM)
// =============================================================================
// Computes: C = A × B + bias (optional)
// Where: A is (M × K), B is (K × N), C is (M × N)
//
// Uses tiled multiplication for memory efficiency:
// - Load TILE_SIZE × TILE_SIZE blocks into shared memory
// - Compute partial products
// - Accumulate in registers
//
// Determinism: Fixed iteration order, no atomics
// =============================================================================

// Tile size for shared memory blocking
#define TILE_SIZE 16

// --- Buffer Bindings ---

// Matrix A: (M × K), row-major
layout(binding = 0, std430) readonly buffer MatrixA {
    float A[];
};

// Matrix B: (K × N), row-major
layout(binding = 1, std430) readonly buffer MatrixB {
    float B[];
};

// Matrix C: (M × N), row-major, output
layout(binding = 2, std430) writeonly buffer MatrixC {
    float C[];
};

// Optional bias: (N,) - added to each row of C
layout(binding = 3, std430) readonly buffer Bias {
    float bias[];
};

// Parameters
layout(push_constant) uniform PushConstants {
    uint M;         // Rows of A, rows of C
    uint K;         // Cols of A, rows of B
    uint N;         // Cols of B, cols of C
    uint use_bias;  // 1 if bias should be added, 0 otherwise
} params;

// --- Workgroup Configuration ---
// Each workgroup computes a TILE_SIZE × TILE_SIZE block of C
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// --- Shared Memory ---
shared float tile_A[TILE_SIZE][TILE_SIZE];
shared float tile_B[TILE_SIZE][TILE_SIZE];

// --- Main Entry Point ---
void main() {
    // Global output coordinates
    uint global_row = gl_WorkGroupID.y * TILE_SIZE + gl_LocalInvocationID.y;
    uint global_col = gl_WorkGroupID.x * TILE_SIZE + gl_LocalInvocationID.x;
    
    // Local thread coordinates within tile
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;
    
    // Accumulator for dot product
    float sum = 0.0;
    
    // Number of tiles along K dimension
    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over tiles in K dimension
    // CRITICAL: Fixed iteration order for determinism
    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        uint a_row = global_row;
        uint a_col = t * TILE_SIZE + local_col;
        
        if (a_row < params.M && a_col < params.K) {
            tile_A[local_row][local_col] = A[a_row * params.K + a_col];
        } else {
            tile_A[local_row][local_col] = 0.0;
        }
        
        // Load tile of B into shared memory
        uint b_row = t * TILE_SIZE + local_row;
        uint b_col = global_col;
        
        if (b_row < params.K && b_col < params.N) {
            tile_B[local_row][local_col] = B[b_row * params.N + b_col];
        } else {
            tile_B[local_row][local_col] = 0.0;
        }
        
        // Synchronize to ensure tiles are loaded
        barrier();
        memoryBarrierShared();
        
        // Compute partial dot product for this tile
        // CRITICAL: Fixed iteration order within tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[local_row][k] * tile_B[k][local_col];
        }
        
        // Synchronize before loading next tiles
        barrier();
    }
    
    // Write result to global memory
    if (global_row < params.M && global_col < params.N) {
        float result = sum;
        
        // Add bias if enabled
        if (params.use_bias != 0) {
            result += bias[global_col];
        }
        
        C[global_row * params.N + global_col] = result;
    }
}
