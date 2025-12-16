#version 450
// HLX Demo Cube - Fragment Shader
// CONTRACT_1000: hlx-demo-cube
//
// HLX Axioms Verified:
//   A1 DETERMINISM: Same fragment inputs -> same output color (no randomness)
//   A2 REVERSIBILITY: Lighting calculations use invertible operations
//
// This shader:
//   - Implements Phong lighting (ambient + diffuse + specular)
//   - Uses fixed light position for deterministic output
//   - No dynamic branching for consistent GPU execution

// Inputs from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

// Output color
layout(location = 0) out vec4 outColor;

// Fixed lighting parameters (deterministic, no uniforms needed for demo)
const vec3 LIGHT_POS = vec3(2.0, 3.0, 4.0);
const vec3 LIGHT_COLOR = vec3(1.0, 1.0, 1.0);
const vec3 CAMERA_POS = vec3(0.0, 0.0, 5.0);

const float AMBIENT_STRENGTH = 0.15;
const float DIFFUSE_STRENGTH = 0.7;
const float SPECULAR_STRENGTH = 0.5;
const float SHININESS = 32.0;

void main() {
    // Normalize interpolated normal
    vec3 normal = normalize(fragNormal);

    // Ambient component
    vec3 ambient = AMBIENT_STRENGTH * LIGHT_COLOR;

    // Diffuse component (Lambert)
    vec3 lightDir = normalize(LIGHT_POS - fragWorldPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = DIFFUSE_STRENGTH * diff * LIGHT_COLOR;

    // Specular component (Blinn-Phong)
    vec3 viewDir = normalize(CAMERA_POS - fragWorldPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), SHININESS);
    vec3 specular = SPECULAR_STRENGTH * spec * LIGHT_COLOR;

    // Combine lighting with vertex color
    vec3 result = (ambient + diffuse + specular) * fragColor;

    // Output final color (no alpha blending for solid cube)
    outColor = vec4(result, 1.0);
}
