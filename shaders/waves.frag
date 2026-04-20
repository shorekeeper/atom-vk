#version 460
// Wave fragment: thin shell with sinusoidal banding and view-angle fade.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vLocalPos;
layout(location = 2) flat in vec4 vColor;
layout(location = 3) flat in float vAge;

layout(location = 0) out vec4 outColor;

void main() {
    // Sinusoidal banding driven by age so the shell appears to oscillate.
    float band = 0.5 + 0.5 * sin(vAge * 12.0);
    // Soft fade at the silhouette so the shell looks volumetric.
    float facing = abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)));
    float fade   = 1.0 - smoothstep(0.0, 0.95, 1.0 - facing);
    float alpha  = 0.35 * band * fade;
    outColor = vec4(vColor.rgb * (0.6 + 0.4 * band), alpha);
}