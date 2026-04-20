#version 460
// Fullscreen triangle vertex shader.
//
// Emits a single triangle that covers the entire viewport. The fragment shader
// performs the actual ray marching using the interpolated NDC coordinate.

layout(location = 0) out vec2 ndc;

void main() {
    // Standard fullscreen triangle from gl_VertexIndex in {0, 1, 2}.
    vec2 pos = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2)
    );
    ndc = pos * 2.0 - vec2(1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}