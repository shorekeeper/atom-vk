#version 460
// Particle fragment: soft Gaussian glow whose hue encodes the local
// phase of the wavefunction at the particle position.

#define PI 3.14159265358979323846

layout(location = 0) in vec2 vUv;        // [-1, 1] within the quad
layout(location = 1) in vec3 vWorldPos;  // particle centre

layout(set = 0, binding = 1) uniform Camera {
    mat4  viewInv;
    mat4  projInv;
    vec4  cameraPos;
    vec4  domainParams;
    vec4  renderParams;
    vec4  heatmapParams;
} cam;

layout(set = 0, binding = 2) uniform sampler3D psiTex;

layout(location = 0) out vec4 outColor;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float r2 = dot(vUv, vUv);
    if (r2 > 1.0) discard;
    float falloff = exp(-r2 * 4.0);

    // Look up local psi to colour by phase, like the volume renderer does.
    float h = cam.domainParams.x;
    vec3 uvw = (vWorldPos + vec3(h)) / (2.0 * h);
    vec2 psi = texture(psiTex, uvw).rg;
    float phase = atan(psi.y, psi.x);
    float hue   = (phase + PI) / (2.0 * PI);
    vec3 col = hsv2rgb(vec3(hue, 0.7, 1.0));

    // Additive output. Alpha used by the blend equation; rgb modulated by
    // the Gaussian envelope.
    outColor = vec4(col * falloff, falloff);
}