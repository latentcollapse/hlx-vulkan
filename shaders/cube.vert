#version 450
// HLX Demo Cube - Vertex Shader
// CONTRACT_1000: hlx-demo-cube
//
// HLX Axioms Verified:
//   A1 DETERMINISM: Same input vertices -> same transformed output
//   A2 REVERSIBILITY: Transformation matrices are invertible
//
// This shader:
//   - Transforms cube vertices using MVP matrices from uniform buffer
//   - Applies per-frame rotation via push constants (CONTRACT_902 compliant)
//   - Passes vertex attributes to fragment shader for Phong lighting

// Push constant for per-frame rotation angle (efficient per-draw update)
layout(push_constant) uniform PushConstants {
    float rotation_angle;  // Radians, updated each frame
    float time;            // Time in seconds (optional effects)
} push;

// Uniform buffer for transformation matrices (binding 0 per CONTRACT_902)
layout(binding = 0) uniform Matrices {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

// Vertex inputs (36 bytes per vertex: 3*float pos + 3*float normal + 3*float color)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Outputs to fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;

// Rotation around Y axis (deterministic, no branching)
mat4 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c,    0.0,  s,    0.0,
        0.0,  1.0,  0.0,  0.0,
        -s,   0.0,  c,    0.0,
        0.0,  0.0,  0.0,  1.0
    );
}

void main() {
    // Apply push constant rotation to model matrix
    mat4 rotatedModel = ubo.model * rotateY(push.rotation_angle);

    // Transform position to world space
    vec4 worldPos = rotatedModel * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;

    // Transform normal to world space (using normal matrix)
    mat3 normalMatrix = transpose(inverse(mat3(rotatedModel)));
    fragNormal = normalize(normalMatrix * inNormal);

    // Pass through vertex color
    fragColor = inColor;

    // Final clip-space position
    gl_Position = ubo.projection * ubo.view * worldPos;
}
