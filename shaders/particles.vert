#version 460
// Instanced billboard vertex shader for particles.
//
// One instance per particle. Six vertices per instance form a screen-aligned
// quad whose centre is the particle position. The quad is sized in world
// units, then projected through the camera. The fragment shader draws a
// soft circular glow inside the quad.

struct Particle { vec4 pos_age; };

layout(set = 0, binding = 0, std430) readonly buffer Particles {
    Particle particles[];
};

layout(set = 0, binding = 1) uniform Camera {
    mat4  viewInv;
    mat4  projInv;
    vec4  cameraPos;
    vec4  domainParams;
    vec4  renderParams;
    vec4  heatmapParams;
} cam;

layout(push_constant) uniform PC {
    mat4 view;
    mat4 proj;
    float pointSize;   // world-space radius of each billboard
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec3 vWorldPos;

void main() {
    // Six-vertex fullscreen quad as two triangles via gl_VertexIndex.
    vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
        vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
    );
    vec2 corner = corners[gl_VertexIndex];

    Particle pt = particles[gl_InstanceIndex];
    vec3 worldCenter = pt.pos_age.xyz;

    // Camera basis in world space.
    vec3 right = vec3(pc.view[0][0], pc.view[1][0], pc.view[2][0]);
    vec3 up    = vec3(pc.view[0][1], pc.view[1][1], pc.view[2][1]);

    // Hide invalid particles (failed rejection sampling) by collapsing the quad.
    float scale = (pt.pos_age.w >= 0.0) ? pc.pointSize : 0.0;
    vec3 worldPos = worldCenter
                  + right * corner.x * scale
                  + up    * corner.y * scale;

    gl_Position = pc.proj * pc.view * vec4(worldPos, 1.0);
    vUv = corner;
    vWorldPos = worldCenter;
}