#version 460
// Text overlay fragment shader.
//
// Samples the monochrome font atlas. Glyph pixels are either 0 (background)
// or 1 (foreground) in the atlas texture, so a hard threshold with discard
// produces crisp 1-bit text without touching neighbouring pixels. For a
// soft AA look one could switch to alpha blending on the raw texel value;
// the 1-bit Unifont glyphs are already designed to be displayed without AA.

layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;

layout(set = 0, binding = 1) uniform sampler2D fontAtlas;

layout(location = 0) out vec4 outColor;

void main() {
    float a = texture(fontAtlas, vUv).r;
    if (a < 0.5) discard;
    outColor = vColor;
}