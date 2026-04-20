mod win32;
mod vulkan;

use std::time::Instant;
use win32::Window;
use vulkan::{VulkanRenderer, OrbitalComponent, ViewMode};

fn energy(n: i32, z: f32) -> f32 { -(z*z) / (2.0 * (n*n) as f32) }

fn pure(n: i32, l: i32, m: i32) -> Vec<OrbitalComponent> {
    vec![OrbitalComponent {
        n, l, m, z: 1.0,
        c_real: 1.0, c_imag: 0.0,
        energy: energy(n, 1.0),
    }]
}

fn superposition(
    a: (i32,i32,i32), b: (i32,i32,i32),
) -> Vec<OrbitalComponent> {
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    vec![
        OrbitalComponent {
            n: a.0, l: a.1, m: a.2, z: 1.0,
            c_real: inv_sqrt2, c_imag: 0.0, energy: energy(a.0, 1.0),
        },
        OrbitalComponent {
            n: b.0, l: b.1, m: b.2, z: 1.0,
            c_real: inv_sqrt2, c_imag: 0.0, energy: energy(b.0, 1.0),
        },
    ]
}

struct Preset {
    name: &'static str,
    components: Vec<OrbitalComponent>,
    max_density: f32,
}

fn presets() -> Vec<Preset> {
    vec![
        Preset { name: "1s ground state",            components: pure(1,0,0), max_density: 0.30 },
        Preset { name: "2s (radial node)",           components: pure(2,0,0), max_density: 0.04 },
        Preset { name: "2p_z",                        components: pure(2,1,0), max_density: 0.015 },
        Preset { name: "3d_z^2",                      components: pure(3,2,0), max_density: 0.004 },
        Preset { name: "3d_xy",                       components: pure(3,2,-2),max_density: 0.004 },
        Preset { name: "Oscillating 1s+2p_z",        components: superposition((1,0,0),(2,1,0)), max_density: 0.10 },
        Preset { name: "Breathing 1s+2s",             components: superposition((1,0,0),(2,0,0)), max_density: 0.15 },
        Preset { name: "Rabi 2p_z+3d_z^2",            components: superposition((2,1,0),(3,2,0)), max_density: 0.010 },
        Preset { name: "4f (n=4,l=3,m=0)",           components: pure(4,3,0), max_density: 0.0010 },
    ]
}

// Virtual key codes.
const VK_1: usize = 0x31;
const VK_OEM_PLUS:  usize = 0xBB;
const VK_OEM_MINUS: usize = 0xBD;
const VK_SPACE: usize = 0x20;
const VK_V: usize = 0x56; // toggle View mode
const VK_S: usize = 0x53; // cycle Slice axis
const VK_C: usize = 0x43; // cycle Color mode
const VK_X: usize = 0x58; // toggle Contour overlay
const VK_P: usize = 0x50;  // toggle Particles
const VK_W: usize = 0x57;  // toggle Waves
const VK_B: usize = 0x42;  // toggle volume rendering (Background cloud)
const VK_E: usize = 0x45;  // Emit a wave from origin
const VK_LEFT_KEY:  usize = 0x25;
const VK_RIGHT_KEY: usize = 0x27;

fn main() {
    let window = Window::new("Quantum Atom Visualization (Vulkan)", 1280, 720);
    let mut renderer = VulkanRenderer::new(&window);

    let all = presets();
    let mut current = 5;
    apply_preset(&mut renderer, &all[current]);
    print_help();

    let start = Instant::now();
    while window.pump() {
        // Preset hotkeys.
        for i in 0..all.len().min(9) {
            if window.consume_key(VK_1 + i) {
                current = i;
                apply_preset(&mut renderer, &all[current]);
            }
        }
        if window.consume_key(VK_OEM_PLUS) {
            renderer.time_scale = (renderer.time_scale * 1.5).min(50.0);
            println!("time_scale = {:.2} a.u./s", renderer.time_scale);
        }
        if window.consume_key(VK_OEM_MINUS) {
            renderer.time_scale = (renderer.time_scale / 1.5).max(0.05);
            println!("time_scale = {:.2} a.u./s", renderer.time_scale);
        }

        // Camera and view-mode keys.
        if window.consume_key(VK_SPACE) {
            renderer.auto_orbit = !renderer.auto_orbit;
            println!("auto_orbit = {}", renderer.auto_orbit);
        }
        if window.consume_key(VK_V) {
            renderer.toggle_view_mode();
            println!("view = {}", match renderer.view_mode {
                ViewMode::Volume => "Volume", ViewMode::Heatmap => "Heatmap",
            });
        }
        if window.consume_key(VK_S) {
            renderer.cycle_slice_axis();
        }
        if window.consume_key(VK_C) {
            renderer.cycle_color_mode();
        }
        if window.consume_key(VK_X) {
            renderer.show_contour = !renderer.show_contour;
        }
        if window.consume_key(VK_P) {
            renderer.show_particles = !renderer.show_particles;
            if renderer.show_particles { renderer.request_particle_reseed(); }
            println!("particles = {}", renderer.show_particles);
        }
        if window.consume_key(VK_W) {
            renderer.show_waves = !renderer.show_waves;
        }
        if window.consume_key(VK_B) {
            renderer.show_volume = !renderer.show_volume;
        }
        if window.consume_key(VK_E) {
            // Emit a wave from origin: yellow shell of one Bohr per second, max 30 Bohr.
            renderer.emit_wave([0.0, 0.0, 0.0], [1.0, 1.0, 0.2], 30.0, 12.0);
        }
        if window.consume_key(VK_LEFT_KEY) {
            renderer.nudge_slice(-0.05);
        }
        if window.consume_key(VK_RIGHT_KEY) {
            renderer.nudge_slice(0.05);
        }

        // Mouse: LMB drag rotates, wheel zooms. Disables auto-orbit on drag
        // so the user can hold their viewpoint.
        let mouse = window.poll_mouse();
        if mouse.left && (mouse.dx != 0.0 || mouse.dy != 0.0) {
            renderer.auto_orbit = false;
            renderer.camera_drag(mouse.dx, mouse.dy);
        }
        if mouse.wheel != 0.0 {
            renderer.camera_zoom(mouse.wheel);
        }

        let t = start.elapsed().as_secs_f32();
        renderer.render_frame(t);
    }
    renderer.wait_idle();
}

fn apply_preset(r: &mut VulkanRenderer, p: &Preset) {
    r.components = p.components.clone();
    r.max_density = p.max_density;
    r.request_particle_reseed();
    println!("Preset: {}", p.name);
}

fn print_help() {
    println!(" Controls ");
    println!("1..9       : preset orbital / superposition");
    println!("+ / -      : faster / slower simulation time");
    println!("Space      : toggle auto-orbit camera");
    println!("LMB drag   : rotate camera");
    println!("Wheel      : zoom in/out");
    println!("V          : toggle Volume / Heatmap view");
    println!("S          : cycle slice axis (XY / XZ / YZ)  [heatmap]");
    println!("C          : cycle color mode (density / real / phase) [heatmap]");
    println!("X          : toggle zero-contour overlay [heatmap, real mode]");
    println!("Left/Right : scrub slice depth [heatmap]");
    println!("Esc        : quit");
}