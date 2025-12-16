#version 450

// Particle vertex shader - Sascha Willems compute particles demo port
// HLX vertex shader CONTRACT_900
// Renders computed particles as point primitives

layout(binding = 0, std430) buffer ParticleBuffer {
    vec4 particles[];  // xyz = position, w = lifetime
} particleBuffer;

layout(binding = 1) uniform RenderParams {
    mat4 projection;
    mat4 view;
    float pointSize;
} renderParams;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outLifetime;

void main() {
    // Get particle data from storage buffer
    vec4 particle = particleBuffer.particles[gl_VertexIndex];

    // Discard dead particles
    if (particle.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -1.0, 1.0);
        outColor = vec4(0.0);
        outLifetime = 0.0;
        return;
    }

    // Transform position to clip space
    vec3 worldPos = particle.xyz;
    gl_Position = renderParams.projection * renderParams.view * vec4(worldPos, 1.0);

    // Set point size for rasterization
    gl_PointSize = renderParams.pointSize;

    // Color based on lifetime (fade out as particle ages)
    float lifetimeNorm = particle.w;  // Normalized lifetime [0, 1]

    // Hue shift: blue (young) -> cyan -> green -> yellow -> red (old)
    vec3 colorMap;
    if (lifetimeNorm > 0.75) {
        // Blue to cyan
        colorMap = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0), (1.0 - lifetimeNorm) / 0.25);
    } else if (lifetimeNorm > 0.5) {
        // Cyan to green
        colorMap = mix(vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 1.0), (lifetimeNorm - 0.5) / 0.25);
    } else if (lifetimeNorm > 0.25) {
        // Green to yellow
        colorMap = mix(vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), (lifetimeNorm - 0.25) / 0.25);
    } else {
        // Yellow to red
        colorMap = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), lifetimeNorm / 0.25);
    }

    // Fade alpha as particle dies
    outColor = vec4(colorMap, lifetimeNorm);
    outLifetime = lifetimeNorm;
}
