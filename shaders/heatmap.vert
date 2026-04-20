#version 460
// Fullscreen triangle (same as raymarch.vert).
layout(location = 0) out vec2 ndc;

void main() {
    vec2 pos = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2)
    );
    ndc = pos * 2.0 - vec2(1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}