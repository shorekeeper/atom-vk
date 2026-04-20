use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let vulkan_sdk = env::var("VULKAN_SDK")
        .expect("VULKAN_SDK is not set. Install the LunarG Vulkan SDK for Windows.");
    let glslc = PathBuf::from(&vulkan_sdk).join("Bin").join("glslc.exe");
    if !glslc.exists() {
        panic!("glslc.exe not found at {}", glslc.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from("shaders");

    let shaders = [
        ("wavefunction.comp", "wavefunction.comp.spv"),
        ("raymarch.vert",     "raymarch.vert.spv"),
        ("raymarch.frag",     "raymarch.frag.spv"),
        ("heatmap.vert",      "heatmap.vert.spv"),
        ("heatmap.frag",      "heatmap.frag.spv"),
        ("particles.comp",    "particles.comp.spv"),
        ("particles.vert",    "particles.vert.spv"),
        ("particles.frag",    "particles.frag.spv"),
        ("waves.vert",        "waves.vert.spv"),
        ("waves.frag",        "waves.frag.spv"),
    ];

    for (src, dst) in shaders.iter() {
        let src_path = shader_dir.join(src);
        let dst_path = out_dir.join(dst);
        let status = Command::new(&glslc)
            .arg(&src_path).arg("-o").arg(&dst_path)
            .arg("--target-env=vulkan1.2").arg("-O")
            .status().expect("Failed to launch glslc.exe");
        if !status.success() {
            panic!("glslc failed for {}", src_path.display());
        }
    }
}