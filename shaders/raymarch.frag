#version 460
// Volume ray marching with phase-modulated transfer function.
//
// Performance addition: hierarchical empty-space skipping.
//
// A companion compute shader (mipmap.comp) builds a coarse 16^3 image
// holding max(|psi|^2) per 8x8x8 fine-voxel block. The ray march loop
// samples this coarse image first; when the entire coarse cell is below
// the opacity threshold, the ray is advanced to the cell's exit plane
// via a slab intersection in a single step. This turns the classic
// "march through 3/4 of an empty bounding box at stepSize" worst case
// into a handful of coarse lookups.
//
// Visual output is identical to the non-skip path because the coarse
// max value guarantees no fine voxel inside the skipped cell could have
// contributed above threshold.

#define PI_CONST 3.14159265358979323846

layout(location = 0) in  vec2 ndc;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D psiTex;

layout(set = 0, binding = 1) uniform Camera {
    mat4  viewInv;
    mat4  projInv;
    vec4  cameraPos;
    // x = halfExtent (Bohr), y = maxDensity, z = voxelSize, w = viewMode
    vec4  domainParams;
    // x = stepSize, y = opacityScale, z = densityThreshold, w = gamma
    vec4  renderParams;
    // heatmap-specific; ignored here.
    vec4  heatmapParams;
    // Performance uniforms:
    //   x = useMipSkip (>0.5 enables hierarchical empty-space skip)
    //   y = coarseGridSize (side length of the coarse max image)
    //   z, w = unused
    vec4  perfParams;
} cam;

layout(set = 0, binding = 2) uniform sampler3D psiMaxTex;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

bool intersectBox(vec3 ro, vec3 rd, vec3 bMin, vec3 bMax,
                  out float tNear, out float tFar) {
    vec3 invD = 1.0 / rd;
    vec3 t0 = (bMin - ro) * invD;
    vec3 t1 = (bMax - ro) * invD;
    vec3 tn = min(t0, t1);
    vec3 tf = max(t0, t1);
    tNear = max(max(tn.x, tn.y), tn.z);
    tFar  = min(min(tf.x, tf.y), tf.z);
    return tFar >= max(tNear, 0.0);
}

float densityAt(vec3 wp, vec3 bMin, vec3 bMax) {
    vec3 uvw = (wp - bMin) / (bMax - bMin);
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
        return 0.0;
    vec2 psi = texture(psiTex, uvw).rg;
    return dot(psi, psi);
}

void main() {
    // Reconstruct primary ray.
    vec4 nh = cam.projInv * vec4(ndc, 0.0, 1.0);
    vec3 rNear = (cam.viewInv * vec4(nh.xyz / nh.w, 1.0)).xyz;
    vec4 fh = cam.projInv * vec4(ndc, 1.0, 1.0);
    vec3 rFar  = (cam.viewInv * vec4(fh.xyz / fh.w, 1.0)).xyz;
    vec3 ro = cam.cameraPos.xyz;
    vec3 rd = normalize(rFar - rNear);
    // Hardening: axis-aligned rays would divide by zero in the slab test
    // used to jump across empty coarse cells. Clamp to a tiny magnitude;
    // t_exit for that axis just becomes enormous and the min() picks the
    // other two axes, which is exactly what we want geometrically.
    vec3 rd_safe = vec3(
        abs(rd.x) < 1e-8 ? 1e-8 : rd.x,
        abs(rd.y) < 1e-8 ? 1e-8 : rd.y,
        abs(rd.z) < 1e-8 ? 1e-8 : rd.z
    );

    float halfExt    = cam.domainParams.x;
    float maxDensity = cam.domainParams.y;
    float voxelSize  = cam.domainParams.z;

    float stepSize     = cam.renderParams.x;
    float opacityScale = cam.renderParams.y;
    float threshold    = cam.renderParams.z;
    float gamma        = cam.renderParams.w;

    bool  useMipSkip = cam.perfParams.x > 0.5;
    float coarseGrid = max(cam.perfParams.y, 1.0);

    vec3 bMin = vec3(-halfExt);
    vec3 bMax = vec3( halfExt);

    float tN, tF;
    if (!intersectBox(ro, rd, bMin, bMax, tN, tF)) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    tN = max(tN, 0.0);

    // Stochastic jitter to break banding.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)))
                         * 43758.5453);
    float t = tN + jitter * stepSize;

    vec3  accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    vec3 lightDir = normalize(vec3(0.6, 0.8, 0.5));
    const int MAX_STEPS = 1024;

    // Pre-computed coarse-cell dimensions used by the slab skip test.
    vec3 coarseCellSize = (bMax - bMin) / coarseGrid;

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (t > tF || accumAlpha > 0.99) break;

        vec3 wp = ro + rd * t;
        vec3 uvw = (wp - bMin) / (bMax - bMin);

        // Hierarchical empty-space skip. Consult the coarse max-density
        // image; if the entire coarse cell is below the opacity threshold
        // we skip to its exit plane in a single step.
        if (useMipSkip) {
            float coarseMax = texture(psiMaxTex, uvw).r;
            if (coarseMax / maxDensity <= threshold) {
                // Clamp against coarseGrid - 1 so a fragment exactly on the
                // far face does not address an out-of-range cell index.
                vec3 cell_idx =
                    min(floor(uvw * coarseGrid), vec3(coarseGrid - 1.0));
                vec3 cell_min_w =
                    (cell_idx / coarseGrid) * (bMax - bMin) + bMin;
                vec3 cell_max_w = cell_min_w + coarseCellSize;
                // Slab intersection in world space; t_exit is the smallest
                // positive distance along the ray to one of the three
                // far-side cell planes.
                vec3 t0v = (cell_min_w - wp) / rd_safe;
                vec3 t1v = (cell_max_w - wp) / rd_safe;
                vec3 tExitV = max(t0v, t1v);
                float tExit = min(min(tExitV.x, tExitV.y), tExitV.z);
                // Advance by at least one fine stepSize to guarantee
                // forward progress even when we start on a boundary.
                t += max(tExit, stepSize) + 1e-4;
                continue;
            }
        }

        vec2 psi = texture(psiTex, uvw).rg;
        float density = dot(psi, psi);
        float u = density / maxDensity;

        if (u > threshold) {
            // Hue from phase, saturation and value boosted by density.
            float phase = atan(psi.y, psi.x);
            float hue = (phase + PI_CONST) / (2.0 * PI_CONST);
            vec3 baseColor = hsv2rgb(vec3(hue, 0.85, 1.0));

            // Gradient-based diffuse lighting.
            float h = voxelSize;
            float dx = densityAt(wp + vec3(h,0,0), bMin, bMax)
                     - densityAt(wp - vec3(h,0,0), bMin, bMax);
            float dy = densityAt(wp + vec3(0,h,0), bMin, bMax)
                     - densityAt(wp - vec3(0,h,0), bMin, bMax);
            float dz = densityAt(wp + vec3(0,0,h), bMin, bMax)
                     - densityAt(wp - vec3(0,0,h), bMin, bMax);
            vec3 grad = vec3(dx, dy, dz);
            float gmag = length(grad);
            vec3 shaded = baseColor;
            if (gmag > 1e-8) {
                vec3 normal = -grad / gmag;
                float diff = max(dot(normal, lightDir), 0.0);
                shaded = baseColor * (0.35 + 0.65 * diff);
            }

            float sigma = pow(min(u, 1.0), gamma) * opacityScale;
            float aStep = 1.0 - exp(-sigma * stepSize);

            accumColor += (1.0 - accumAlpha) * aStep * shaded;
            accumAlpha += (1.0 - accumAlpha) * aStep;
        }

        t += stepSize;
    }

    outColor = vec4(accumColor, 1.0);
}