#version 460
#extension GL_EXT_ray_tracing : require

// HLX Ray Tracing Lab - Miss Shader
// ==================================
// Generates background/sky color when rays don't hit geometry
// Provides deterministic environment for testing ray tracing
// Supports gradient sky based on ray direction

layout(location = 0) rayPayloadInEXT vec3 payload;

layout(binding = 7, set = 0) uniform SkyParams {
    vec4 sky_color_zenith;
    vec4 sky_color_horizon;
    vec4 ground_color;
    float horizon_softness;
} sky;

void main() {
    // Get ray direction from normalized launch position
    vec3 ray_dir = normalize(gl_WorldRayDirectionEXT);

    // Simple sky gradient: blend between zenith and horizon based on Y component
    float horizon_blend = smoothstep(0.0, sky.horizon_softness, ray_dir.y);

    vec3 sky_color;
    if (ray_dir.y > 0.0) {
        // Upper hemisphere: blend zenith to horizon
        sky_color = mix(
            sky.sky_color_horizon.rgb,
            sky.sky_color_zenith.rgb,
            horizon_blend
        );
    } else {
        // Lower hemisphere: ground color
        sky_color = sky.ground_color.rgb * (0.5 + 0.5 * abs(ray_dir.y));
    }

    // Store in payload
    payload = sky_color;
}
