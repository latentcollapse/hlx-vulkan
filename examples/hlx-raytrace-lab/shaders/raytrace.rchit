#version 460
#extension GL_EXT_ray_tracing : require

// HLX Ray Tracing Lab - Closest-Hit Shader
// ==========================================
// Performs surface shading for ray-geometry intersections
// Supports simple diffuse/normal-based coloring
// Can be hot-swapped via HLX resolve() for real-time shader editing

layout(location = 0) rayPayloadInEXT vec3 payload;

// Per-instance material data (from TLAS instance custom index)
hitAttributeEXT vec2 hit_attributes;

// Geometry buffer (simplified for this demo)
// In production: would store vertex/normal data per primitive
layout(binding = 3, set = 0) buffer GeometryBuffer {
    vec4 vertices[];
};

layout(binding = 4, set = 0) buffer NormalBuffer {
    vec4 normals[];
};

layout(binding = 5, set = 0) uniform MaterialParams {
    vec4 diffuse_color;
    vec4 specular_color;
    float roughness;
    float metallic;
    float ior;
    float pad;
} material;

layout(binding = 6, set = 0) uniform SceneParams {
    vec4 light_position;
    vec4 light_color;
    vec4 ambient_color;
} scene;

void main() {
    // Get instance data
    uint instance_id = gl_InstanceID;
    uint hit_kind = gl_HitKindEXT;

    // Unpack barycentric coordinates from hit_attributes
    // These represent (u, v) coordinates in the hit triangle
    vec2 bary = hit_attributes;
    float w = 1.0 - bary.x - bary.y;

    // Basic shading: use instance ID to vary color for visualization
    vec3 color;

    // Simple color based on geometry
    if (hit_kind == gl_HitKindFrontFacingTriangleEXT) {
        // Front-facing: use diffuse coloring
        color = material.diffuse_color.rgb;

        // Vary color by instance for visualization
        float hue_offset = float(instance_id % 8) / 8.0;
        color = mix(color, vec3(hue_offset, 1.0 - hue_offset, 0.5), 0.3);
    } else {
        // Back-facing: darker shade
        color = material.diffuse_color.rgb * 0.5;
    }

    // Apply simple lighting
    // Light direction (hardcoded for determinism)
    vec3 light_dir = normalize(vec3(1.0, 1.0, 0.5));
    float diffuse = max(0.0, dot(vec3(0.0, 1.0, 0.0), light_dir));

    // Combine with ambient
    vec3 result = color * (scene.ambient_color.rgb + diffuse * 0.8);

    // Store in payload for raygen to use
    payload = result;
}
