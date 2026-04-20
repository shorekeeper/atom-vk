#version 460
// Text overlay vertex shader.
//
// Emits one screen-space quad per instance; each instance corresponds to
// one glyph at a given pixel position with a given atlas UV rectangle.
// Six vertices per instance form a two-triangle quad via gl_VertexIndex.
// The pipeline consumes zero vertex attributes; all per-glyph data comes
// from the quad SSBO, which the host writes once per frame.

struct TextQuad {
    vec4 rect;      // xy = pixel position (top-left origin), zw = pixel size
    vec4 uv_rect;   // u0, v0, u1, v1 inside the font atlas
    vec4 color;     // rgba in [0, 1]
};

layout(set = 0, binding = 0, std430) readonly buffer Quads {
    TextQuad quads[];
};

layout(push_constant) uniform PC {
    vec2 screen_size;   // framebuffer extent in pixels
    vec2 _pad;
} pc;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vColor;

void main() {
    // CCW quad. Vulkan clip-space y points down, matching pixel coords,
    // so no flip is needed when going from pixel space to NDC.
    vec2 corners[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec2 c = corners[gl_VertexIndex];

    TextQuad q = quads[gl_InstanceIndex];
    vec2 px  = q.rect.xy + c * q.rect.zw;
    vec2 ndc = (px / pc.screen_size) * 2.0 - vec2(1.0);

    gl_Position = vec4(ndc, 0.0, 1.0);
    vUv         = mix(q.uv_rect.xy, q.uv_rect.zw, c);
    vColor      = q.color;
}