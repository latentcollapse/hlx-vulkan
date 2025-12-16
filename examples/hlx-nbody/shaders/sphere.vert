#version 450

// Sphere vertex shader
// Renders N bodies as spheres with instanced geometry
// One instance per body

layout(binding = 0) uniform NBodyMatrices {
    mat4 view;
    mat4 projection;
} matrices;

// Storage buffer containing body positions and masses
layout(std430, binding = 2) buffer BodyData {
    vec4 positions[];
} bodyPos;

// Per-vertex sphere geometry (shared across all instances)
layout(location = 0) in vec3 inPosition;   // Sphere vertex position (local space)
layout(location = 1) in vec3 inNormal;     // Sphere surface normal

// Per-instance data
layout(location = 2) in uint inBodyId;     // Which body this is (used as instance index)

// Output to fragment shader
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outFragPos;
layout(location = 2) out vec3 outBodyColor;

void main() {
    // Load body data: position in xyz, mass in w
    vec4 bodyData = bodyPos.positions[inBodyId];
    vec3 bodyPos = bodyData.xyz;
    float bodyMass = bodyData.w;

    // Scale sphere based on mass (cube root for volume relationship)
    float sphereRadius = pow(bodyMass, 1.0 / 3.0) * 0.5;  // Scale factor 0.5

    // Transform sphere vertex to world space
    vec3 worldPos = bodyPos + inPosition * sphereRadius;

    // Transform normal to world space (no scaling needed for unit sphere)
    vec3 worldNormal = inNormal;

    // Pass world position and normal to fragment shader
    outFragPos = worldPos;
    outNormal = worldNormal;

    // Color based on mass: hotter (redder) for more massive bodies
    float colorIntensity = clamp(bodyMass / 10.0, 0.2, 1.0);
    outBodyColor = mix(vec3(0.5, 0.8, 1.0), vec3(1.0, 0.2, 0.2), colorIntensity);

    // Transform to clip space
    gl_Position = matrices.projection * matrices.view * vec4(worldPos, 1.0);
}
