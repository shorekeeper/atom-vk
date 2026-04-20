#version 460
// Volume ray marching with phase-modulated transfer function.
//
// At each sample along the ray we read the complex wavefunction, recover
// the probability density rho = |psi|^2 and the local phase arg(psi).
// The phase drives the hue (cyclic colormap), the density drives the
// opacity through a thresholded power law, and the gradient of the
// density provides the local surface normal for diffuse shading.
//
// Beer-Lambert front-to-back compositing is used. Empty space is fully
// transparent thanks to the hard threshold, so the orbital lobes stand
// out against a black background instead of merging into a colored cube.

// PI is reused above; declared after main only to keep the structure
// readable. GLSL allows top-level constants anywhere before linking.
#define PI_CONST 3.14159265358979323846

layout(location = 0) in  vec2 ndc;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D psiTex;

layout(set = 0, binding = 1) uniform Camera {
    mat4  viewInv;
    mat4  projInv;
    vec4  cameraPos;
    // x = halfExtent (Bohr), y = maxDensity, z = voxelSize, w = colorMode
    vec4  domainParams;
    // x = stepSize, y = opacityScale, z = densityThreshold, w = gamma
    vec4  renderParams;
} cam;

// Standard HSV to RGB. Used for cyclic phase coloring.
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

    float halfExt    = cam.domainParams.x;
    float maxDensity = cam.domainParams.y;
    float voxelSize  = cam.domainParams.z;

    float stepSize     = cam.renderParams.x;
    float opacityScale = cam.renderParams.y;
    float threshold    = cam.renderParams.z;
    float gamma        = cam.renderParams.w;

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

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (t > tF || accumAlpha > 0.99) break;

        vec3 wp = ro + rd * t;
        vec3 uvw = (wp - bMin) / (bMax - bMin);
        vec2 psi = texture(psiTex, uvw).rg;
        float density = dot(psi, psi);
        float u = density / maxDensity;

        // Hard threshold: empty space contributes nothing, eliminating
        // the foggy colored-cube look of the previous version.
        if (u > threshold) {
            // Hue from phase, saturation and value boosted by density.
            float phase = atan(psi.y, psi.x);
            float hue = (phase + PI_CONST) / (2.0 * PI_CONST);
            vec3 baseColor = hsv2rgb(vec3(hue, 0.85, 1.0));

            // Gradient-based diffuse lighting.
            // Six neighbour samples; cost is acceptable thanks to the
            // hard threshold gating.
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
                // Outward normal of an iso-density shell points opposite
                // to grad rho (rho decreases outward).
                vec3 normal = -grad / gmag;
                float diff = max(dot(normal, lightDir), 0.0);
                shaded = baseColor * (0.35 + 0.65 * diff);
            }

            // Beer-Lambert opacity with power law to keep weak regions
            // visible without flooding the image.
            float sigma = pow(min(u, 1.0), gamma) * opacityScale;
            float aStep = 1.0 - exp(-sigma * stepSize);

            accumColor += (1.0 - accumAlpha) * aStep * shaded;
            accumAlpha += (1.0 - accumAlpha) * aStep;
        }

        t += stepSize;
    }

    // Composite over pure black so the orbital is the only thing the eye
    // latches onto.
    outColor = vec4(accumColor, 1.0);
}
