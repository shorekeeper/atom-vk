#version 460
// Heatmap (planar slice) visualization of the wavefunction volume.
//
// Implements the heatmap visualization subsystem described in section 13 of
// the architecture document: extracts a planar slice through the volumetric
// data and applies a color mapping. Three slice axes (XY/XZ/YZ), three color
// modes (density, real part, phase), and three scale modes (linear, log,
// power) are exposed through the heatmap_params uniform.
//
// Nodal surfaces are visible as zero-crossings of Re(psi) when the diverging
// color scheme is active and as the boundary between high-density regions
// when the density scheme is active. A contour overlay highlights the zero
// contour explicitly.

#define PI 3.14159265358979323846

layout(location = 0) in  vec2 ndc;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D psiTex;

layout(set = 0, binding = 1) uniform Camera {
    mat4  viewInv;
    mat4  projInv;
    vec4  cameraPos;
    // x = halfExtent, y = maxDensity, z = voxelSize, w = viewMode (unused here)
    vec4  domainParams;
    // x = stepSize, y = opacityScale, z = threshold, w = gamma
    vec4  renderParams;
    // x = sliceAxis (0=XY, 1=XZ, 2=YZ)
    // y = sliceOffset in [-1, 1] along the normal axis
    // z = colorMode (0=density, 1=real, 2=phase)
    // w = contourFlag (>0.5 enables zero-contour overlay)
    vec4  heatmapParams;
    // x = volume mip skip, y = coarse grid size
    // z = heatmap scale mode, w = viewport aspect ratio
    vec4  perfParams;
} cam;

// Polynomial viridis approximation. Sequential, perceptually uniform; used
// for non-negative quantities like |psi|^2.
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3( 0.27770,  0.00485,  0.32941);
    const vec3 c1 = vec3( 0.10570,  1.40400,  1.38484);
    const vec3 c2 = vec3(-0.33308,  0.21482,  0.09524);
    const vec3 c3 = vec3(-4.63421, -5.79911,-19.33244);
    const vec3 c4 = vec3( 6.22886, 14.17993, 56.69052);
    const vec3 c5 = vec3( 4.77638,-13.74510,-65.35303);
    const vec3 c6 = vec3(-5.43590,  4.64577, 26.31242);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// Cool-warm diverging colormap centred at zero. For signed quantities like
// Re(psi). Negative -> blue, zero -> near-white, positive -> red.
vec3 coolwarm(float t) {
    // t in [-1, 1]
    t = clamp(t, -1.0, 1.0);
    if (t >= 0.0) {
        return mix(vec3(0.95, 0.95, 0.95), vec3(0.71, 0.02, 0.15), t);
    } else {
        return mix(vec3(0.95, 0.95, 0.95), vec3(0.13, 0.20, 0.66), -t);
    }
}

// HSV cyclic colormap for phase. Returns full-saturation colors that wrap
// around the hue circle as the phase goes from -pi to +pi.
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Sample the volume at a given world-space position. Returns vec2(re, im).
vec2 samplePsi(vec3 worldPos) {
    float h = cam.domainParams.x;
    vec3 uvw = (worldPos + vec3(h)) / (2.0 * h);
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
        return vec2(0.0);
    return texture(psiTex, uvw).rg;
}

// Applies the selected scale to a non-negative normalized value.
float scaleUnsigned(float value, int scaleMode) {
    float x = max(value, 0.0);
    if (scaleMode == 0) return clamp(x, 0.0, 1.0);
    if (scaleMode == 1) {
        const float decades = 8.0;
        float logValue = log(max(x, 1e-30)) / log(10.0);
        return clamp((logValue + decades) / decades, 0.0, 1.0);
    }
    return pow(clamp(x, 0.0, 1.0), max(cam.renderParams.w, 1e-4));
}

// Applies the selected scale while preserving the sign.
float scaleSigned(float value, int scaleMode) {
    float magnitude = abs(value);
    float scaled = scaleUnsigned(magnitude, scaleMode);
    return value < 0.0 ? -scaled : scaled;
}

// Reduces a complex sample to the selected scalar and colormap coordinate.
void valueAndNorm(vec2 psi, int mode, int scaleMode,
                  out float value, out float normalized) {
    float maxDensity = max(cam.domainParams.y, 1e-30);
    if (mode == 0) {
        value = dot(psi, psi);
        normalized = scaleUnsigned(value / maxDensity, scaleMode);
    } else if (mode == 1) {
        value = psi.x;
        normalized = scaleSigned(
            value / sqrt(maxDensity), scaleMode);
    } else {
        float density = dot(psi, psi);
        value = density > 1e-20 ? atan(psi.y, psi.x) : 0.0;
        normalized = (value + PI) / (2.0 * PI);
    }
}

vec3 legendColor(float t, int colorMode) {
    if (colorMode == 0) return viridis(t);
    if (colorMode == 1) return coolwarm(t * 2.0 - 1.0);
    return hsv2rgb(vec3(t, 0.85, 1.0));
}

void main() {
    int axis      = int(cam.heatmapParams.x + 0.5);
    float offset  = cam.heatmapParams.y;
    int colorMode = int(cam.heatmapParams.z + 0.5);
    bool showCont = cam.heatmapParams.w > 0.5;
    int scaleMode = int(cam.perfParams.z + 0.5);
    float h = cam.domainParams.x;
    float aspect = max(cam.perfParams.w, 1e-6);

    // Preserve equal world-space scale along both displayed axes.
    vec2 sliceNdc = ndc;
    if (aspect >= 1.0) sliceNdc.x *= aspect;
    else sliceNdc.y /= aspect;
    bool insideSlice = all(lessThanEqual(abs(sliceNdc), vec2(1.0)));
    vec2 sliceCoord = sliceNdc * h;

    vec3 worldPos;
    if (axis == 0) {
        // XY plane, slice offset along Z.
        worldPos = vec3(sliceCoord.x, sliceCoord.y, offset * h);
    } else if (axis == 1) {
        // XZ plane, slice offset along Y.
        worldPos = vec3(sliceCoord.x, offset * h, sliceCoord.y);
    } else {
        // YZ plane, slice offset along X.
        worldPos = vec3(offset * h, sliceCoord.x, sliceCoord.y);
    }

    vec2 psi = insideSlice ? samplePsi(worldPos) : vec2(0.0);
    float v, nrm;
    valueAndNorm(psi, colorMode, scaleMode, v, nrm);

    vec3 color;
    if (colorMode == 0) {
        color = viridis(nrm);
        // Density also dims the color when extremely small to reveal the
        // bounding box edges.
        if (dot(psi, psi) < 1e-12) color = vec3(0.0);
    } else if (colorMode == 1) {
        color = coolwarm(nrm);
    } else {
        // Phase: keep hue but darken low-amplitude regions so noise does
        // not dominate near nodal surfaces.
        float amp = sqrt(dot(psi, psi));
        float ampNrm = clamp(amp / sqrt(cam.domainParams.y), 0.0, 1.0);
        vec3 hue = hsv2rgb(vec3(nrm, 0.85, 1.0));
        color = hue * pow(ampNrm, 0.4);
    }

    // Optional zero-contour overlay. Detect sign change of Re(psi) by
    // sampling four nearest neighbours and drawing a thin antialiased line.
    if (insideSlice && showCont && colorMode == 1) {
        float voxel = cam.domainParams.z;
        vec3 dxv, dyv;
        if (axis == 0)      { dxv = vec3(voxel, 0, 0); dyv = vec3(0, voxel, 0); }
        else if (axis == 1) { dxv = vec3(voxel, 0, 0); dyv = vec3(0, 0, voxel); }
        else                { dxv = vec3(0, voxel, 0); dyv = vec3(0, 0, voxel); }

        float v0 = samplePsi(worldPos        ).x;
        float vx = samplePsi(worldPos + dxv  ).x;
        float vy = samplePsi(worldPos + dyv  ).x;
        // Approximate distance to zero crossing using gradient magnitude.
        vec2  grad = vec2(vx - v0, vy - v0);
        float gmag = length(grad);
        if (gmag > 1e-6) {
            float dist = abs(v0) / gmag;
            float aa = 1.0 - smoothstep(0.0, 1.5, dist);
            color = mix(color, vec3(0.0), aa * 0.85);
        }
    }

    // Draw axes through the nucleus using screen-space derivatives.
    if (insideSlice) {
        float lineWidth = max(
            fwidth(sliceCoord.x), fwidth(sliceCoord.y)) * 0.65;
        float axisDist = min(abs(sliceCoord.x), abs(sliceCoord.y));
        float axisAlpha = 1.0 - smoothstep(
            lineWidth * 0.5, lineWidth * 1.5, axisDist);
        color = mix(color, vec3(0.4), axisAlpha * 0.6);
    } else {
        color = vec3(0.0);
    }

    // Draw the active colormap in normalized device coordinates.
    vec2 legendMin = vec2(0.84, -0.72);
    vec2 legendMax = vec2(0.91,  0.72);
    vec2 innerMin = vec2(0.85, -0.70);
    vec2 innerMax = vec2(0.90,  0.70);
    bool inLegend = all(greaterThanEqual(ndc, legendMin)) &&
                    all(lessThanEqual(ndc, legendMax));
    bool inLegendInner = all(greaterThanEqual(ndc, innerMin)) &&
                         all(lessThanEqual(ndc, innerMax));
    if (inLegend) {
        color = vec3(0.55);
        if (inLegendInner) {
            float t = clamp(
                (ndc.y - innerMin.y) / (innerMax.y - innerMin.y),
                0.0, 1.0);
            color = legendColor(t, colorMode);
        }
    }

    outColor = vec4(color, 1.0);
}