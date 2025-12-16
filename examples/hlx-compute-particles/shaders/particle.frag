#version 450

// Particle fragment shader - Sascha Willems compute particles demo port
// HLX fragment shader CONTRACT_900
// Colors particles with smooth falloff for point primitives

layout(location = 0) in vec4 inColor;
layout(location = 1) in float inLifetime;

layout(location = 0) out vec4 outColor;

void main() {
    // Create soft circle for point sprite
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    float r = dot(circCoord, circCoord);

    // Discard pixels outside the circle
    if (r > 1.0) {
        discard;
    }

    // Smooth falloff from center to edge
    float falloff = 1.0 - r;
    float alpha = inColor.a * falloff * falloff;

    // Output color with computed alpha
    outColor = vec4(inColor.rgb, alpha);
}
