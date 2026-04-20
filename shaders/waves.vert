#version 460
// Instanced wireframe icosphere for emitted waves.
//
// Each wave instance carries an origin and a current radius; the vertex
// shader scales a unit sphere and offsets it. The fragment shader draws
// a thin radial sinusoidal pattern so the surface looks like an
// expanding wave rather than a solid bubble.

struct Wave {
    vec4 origin_radius;   // xyz = origin, w = current radius
    vec4 color_age;       // rgb = color, a = age in seconds
};

layout(set = 0, binding = 0, std430) readonly buffer Waves {
    Wave waves[];
};

layout(push_constant) uniform PC {
    mat4 view;
    mat4 proj;
} pc;

layout(location = 0) in vec3 inLocalPos;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vLocalPos;
layout(location = 2) flat out vec4 vColor;
layout(location = 3) flat out float vAge;

void main() {
    Wave w = waves[gl_InstanceIndex];
    vec3 worldPos = w.origin_radius.xyz + inLocalPos * w.origin_radius.w;
    gl_Position = pc.proj * pc.view * vec4(worldPos, 1.0);
    vNormal   = normalize(inLocalPos);
    vLocalPos = inLocalPos;
    vColor    = vec4(w.color_age.rgb, 1.0);
    vAge      = w.color_age.a;
}