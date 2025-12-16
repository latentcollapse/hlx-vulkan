#version 450

// Sphere fragment shader
// Implements Phong shading for realistic lighting
// Per-body color based on mass

layout(binding = 1) uniform LightingParams {
    vec3 lightPos;
    float ambientStrength;
    vec3 lightColor;
    float specularStrength;
    float shininess;
} lighting;

// Input from vertex shader
layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inFragPos;
layout(location = 2) in vec3 inBodyColor;

// Output color
layout(location = 0) out vec4 outColor;

void main() {
    // Normalize interpolated normal
    vec3 norm = normalize(inNormal);

    // Ambient component
    vec3 ambient = lighting.ambientStrength * lighting.lightColor;

    // Diffuse component
    vec3 lightDir = normalize(lighting.lightPos - inFragPos);
    float diffuseIntensity = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diffuseIntensity * lighting.lightColor;

    // Specular component
    vec3 viewDir = normalize(-inFragPos);  // Camera at origin
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), lighting.shininess);
    vec3 specular = lighting.specularStrength * spec * lighting.lightColor;

    // Combine with body color
    vec3 result = (ambient + diffuse + specular) * inBodyColor;

    // Output with full opacity
    outColor = vec4(result, 1.0);
}
