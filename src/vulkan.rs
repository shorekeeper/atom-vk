#![allow(clippy::too_many_arguments)]

use ash::{vk, Entry, Instance, Device};
use ash::khr;
use std::ffi::{CString, CStr};
use std::mem::size_of;
use std::ptr;

use crate::win32::Window;

const COMP_SPV:           &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wavefunction.comp.spv"));
const MIPMAP_COMP_SPV:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mipmap.comp.spv"));
const VERT_SPV:           &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raymarch.vert.spv"));
const FRAG_SPV:           &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raymarch.frag.spv"));
const HEATMAP_VERT_SPV:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heatmap.vert.spv"));
const HEATMAP_FRAG_SPV:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heatmap.frag.spv"));
const PARTICLES_COMP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.comp.spv"));
const PARTICLES_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.vert.spv"));
const PARTICLES_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.frag.spv"));
const WAVES_VERT_SPV:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/waves.vert.spv"));
const WAVES_FRAG_SPV:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/waves.frag.spv"));
const TEXT_VERT_SPV:      &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text.vert.spv"));
const TEXT_FRAG_SPV:      &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text.frag.spv"));

// Raw Unifont-style hex font. Parsed once at startup into an ASCII atlas.
const FONT_HEX: &[u8] = include_bytes!("../shaders/font.hex");

pub const GRID_SIZE:        u32 = 128;
// Coarse max-density grid: one coarse voxel per 8^3 fine voxels.
pub const COARSE_GRID_SIZE: u32 = 16;
pub const COARSE_BLOCK:     u32 = GRID_SIZE / COARSE_GRID_SIZE; // 8
pub const HALF_EXTENT:      f32 = 20.0;
pub const FRAMES_IN_FLIGHT: usize = 2;
pub const MAX_COMPONENTS:   usize = 8;
pub const PARTICLE_COUNT:   u32 = 32_768;
pub const MAX_WAVES:        u32 = 64;

// Text overlay sizing. font.hex ASCII glyphs are 8x16 pixels; we pack the
// 128 ASCII code points into a 16 x 8 grid producing a 128 x 128 atlas.
pub const GLYPH_W:        u32 = 8;
pub const GLYPH_H:        u32 = 16;
pub const ATLAS_COLS:     u32 = 16;
pub const ATLAS_ROWS:     u32 = 8;
pub const ATLAS_W:        u32 = GLYPH_W * ATLAS_COLS; // 128
pub const ATLAS_H:        u32 = GLYPH_H * ATLAS_ROWS; // 128
pub const MAX_TEXT_QUADS: u32 = 1024;

// One eigenstate term in the superposition psi = sum_k c_k psi_nlm_k exp(-i E_k t).
#[derive(Clone, Copy, Default)]
pub struct OrbitalComponent {
    pub n: i32,
    pub l: i32,
    pub m: i32,
    pub z: f32,
    pub c_real: f32,
    pub c_imag: f32,
    pub energy: f32,
}

// std140 layout matches the GLSL "Components" uniform block.
#[repr(C, align(16))]
struct ComponentsUBO {
    num_components: i32,
    grid_size: i32,
    half_extent: f32,
    time: f32,
    nlm_z: [[f32; 4]; MAX_COMPONENTS],
    amp:   [[f32; 4]; MAX_COMPONENTS],
}

// Camera + global render params. perf_params is new; it controls the
// hierarchical empty-space skip inside raymarch.frag at runtime.
#[repr(C)]
struct CameraUBO {
    view_inv:       [[f32; 4]; 4],
    proj_inv:       [[f32; 4]; 4],
    camera_pos:     [f32; 4],
    domain_params:  [f32; 4], // halfExtent, maxDensity, voxelSize, viewMode
    render_params:  [f32; 4], // stepSize, opacityScale, threshold, gamma
    heatmap_params: [f32; 4], // sliceAxis, sliceOffset, colorMode, contourFlag
    perf_params:    [f32; 4], // useMipSkip, coarseGrid, unused, unused
}

// Push constants for the wavefunction evaluation compute shader: none.
// Push constants for the min-max mipmap build.
#[repr(C)]
struct MipmapPC {
    fine_grid_size:   i32,
    coarse_grid_size: i32,
    block_size:       i32,
    _pad:             i32,
}

// Push constants for the particle compute shader.
#[repr(C)]
struct ParticlesPC {
    half_extent:    f32,
    max_density:    f32,
    voxel_size:     f32,
    dt:             f32,
    mode:           u32,
    seed:           u32,
    num_particles:  u32,
    _pad:           u32,
}

// Push constants for particle and wave vertex shaders.
#[repr(C)]
struct GfxMatrixPC {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    point_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// CPU-side wave state, mirrored to GPU each frame as a small SSBO.
#[repr(C)]
#[derive(Clone, Copy)]
struct WaveGPU {
    origin_radius: [f32; 4],
    color_age:     [f32; 4],
}

#[derive(Clone, Copy)]
pub struct Wave {
    pub origin: [f32; 3],
    pub radius: f32,
    pub speed:  f32,
    pub max_radius: f32,
    pub color:  [f32; 3],
    pub age:    f32,
    pub alive:  bool,
}

// Per-glyph instance record mirrored to the text SSBO each frame.
#[repr(C)]
#[derive(Clone, Copy)]
struct TextQuadGPU {
    rect:    [f32; 4],
    uv_rect: [f32; 4],
    color:   [f32; 4],
}

#[repr(C)]
struct TextPC {
    screen_size: [f32; 2],
    _pad:        [f32; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode { Volume, Heatmap }

#[derive(Clone, Copy)]
pub enum SliceAxis { XY = 0, XZ = 1, YZ = 2 }

#[derive(Clone, Copy)]
pub enum ColorMode { Density = 0, Real = 1, Phase = 2 }

// Vendor name lookup for the overlay and startup log. Matches standard
// PCI vendor IDs reported by Vulkan device properties.
fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 => "AMD",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1010 => "ImgTec",
        0x106B => "Apple",
        _      => "Unknown",
    }
}

// Physical-device capabilities detected at startup. The renderer uses
// these flags to enable or bypass optional performance paths (async
// compute queue, subgroup-aware loops, etc.) without a second config pass.
pub struct DeviceCaps {
    pub vendor_id:           u32,
    pub device_id:           u32,
    pub device_name:         String,
    pub driver_version:      u32,
    pub api_version:         u32,
    pub graphics_family:     u32,
    pub async_compute_family:Option<u32>,
    pub subgroup_size:       u32,
    pub device_type:         vk::PhysicalDeviceType,
    pub device_memory_mb:    u64,
}

impl DeviceCaps {
    pub fn vendor_str(&self) -> &'static str { vendor_name(self.vendor_id) }
    pub fn has_async_compute(&self) -> bool { self.async_compute_family.is_some() }
}

// Optional async compute state. Present iff a separate compute queue
// family was found at startup. All wavefunction + mipmap work is then
// recorded into its own command buffer on a dedicated queue, overlapping
// on hardware with the graphics engine's raymarch/particles/waves/text.
struct AsyncCompute {
    queue:           vk::Queue,
    queue_family:    u32,
    // Timeline semaphore signaled by compute submissions, waited on by
    // graphics submissions. Monotonic counter; value N means compute
    // results for frame N-1 are complete.
    timeline:        vk::Semaphore,
    timeline_value:  u64,
    // Per-in-flight-frame command pool and buffer on the compute queue.
    cmd_pools:       Vec<vk::CommandPool>,
    cmd_buffers:     Vec<vk::CommandBuffer>,
}

struct FrameData {
    cmd_pool:        vk::CommandPool,
    cmd_buffer:      vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight_fence: vk::Fence,

    camera_buffer:   vk::Buffer,
    camera_memory:   vk::DeviceMemory,
    camera_mapped:   *mut CameraUBO,

    components_buffer: vk::Buffer,
    components_memory: vk::DeviceMemory,
    components_mapped: *mut ComponentsUBO,

    // Per-frame complex psi volume (written by wavefunction.comp, read
    // by raymarch, mipmap build, particles, heatmap).
    volume_image:  vk::Image,
    volume_memory: vk::DeviceMemory,
    volume_view:   vk::ImageView,

    // Per-frame coarse max-density mipmap (written by mipmap.comp, read
    // by raymarch.frag for hierarchical empty-space skipping).
    psi_max_image:  vk::Image,
    psi_max_memory: vk::DeviceMemory,
    psi_max_view:   vk::ImageView,

    compute_set:  vk::DescriptorSet,     // wavefunction.comp
    mipmap_set:   vk::DescriptorSet,     // mipmap.comp
    graphics_set: vk::DescriptorSet,     // raymarch / heatmap fragment

    // Becomes true after the first compute submission for this frame
    // slot so subsequent layout transitions can use the correct old
    // layout instead of UNDEFINED-with-discard.
    initialized: bool,
}

pub struct VulkanRenderer {
    _entry:           Entry,
    instance:         Instance,
    surface_loader:   khr::surface::Instance,
    surface:          vk::SurfaceKHR,
    physical_device:  vk::PhysicalDevice,
    device:           Device,
    queue_family:     u32,
    queue:            vk::Queue,

    pub caps: DeviceCaps,

    swapchain_loader: khr::swapchain::Device,
    swapchain:        vk::SwapchainKHR,
    swap_format:      vk::Format,
    swap_extent:      vk::Extent2D,
    swap_images:      Vec<vk::Image>,
    swap_views:       Vec<vk::ImageView>,

    volume_sampler:   vk::Sampler,

    descriptor_pool:        vk::DescriptorPool,
    compute_set_layout:     vk::DescriptorSetLayout,
    compute_pipeline_layout:vk::PipelineLayout,
    compute_pipeline:       vk::Pipeline,

    mipmap_set_layout:      vk::DescriptorSetLayout,
    mipmap_pipeline_layout: vk::PipelineLayout,
    mipmap_pipeline:        vk::Pipeline,

    graphics_set_layout:     vk::DescriptorSetLayout,
    graphics_pipeline_layout:vk::PipelineLayout,
    graphics_pipeline:       vk::Pipeline,
    heatmap_pipeline:        vk::Pipeline,

    frames:      Vec<FrameData>,
    frame_index: usize,

    async_compute: Option<AsyncCompute>,

    pub components:   Vec<OrbitalComponent>,
    pub time_scale:   f32,
    pub max_density:  f32,

    pub cam_yaw:    f32,
    pub cam_pitch:  f32,
    pub cam_radius: f32,
    pub auto_orbit: bool,

    pub view_mode:    ViewMode,
    pub slice_axis:   SliceAxis,
    pub slice_offset: f32,
    pub color_mode:   ColorMode,
    pub show_contour: bool,

    // Particle subsystem.
    particle_buffer:            vk::Buffer,
    particle_memory:            vk::DeviceMemory,
    particles_set_layout:       vk::DescriptorSetLayout,
    particles_set:              vk::DescriptorSet,
    particles_pipeline_layout:  vk::PipelineLayout,
    particles_compute_pipeline: vk::Pipeline,
    particles_gfx_set_layout:     vk::DescriptorSetLayout,
    particles_gfx_set:            vk::DescriptorSet,
    particles_gfx_pipeline_layout:vk::PipelineLayout,
    particles_gfx_pipeline:       vk::Pipeline,

    // Wave subsystem.
    wave_buffer:        vk::Buffer,
    wave_memory:        vk::DeviceMemory,
    wave_mapped:        *mut WaveGPU,
    wave_index_buffer:  vk::Buffer,
    wave_index_memory:  vk::DeviceMemory,
    wave_vertex_buffer: vk::Buffer,
    wave_vertex_memory: vk::DeviceMemory,
    wave_index_count:   u32,
    waves_set_layout:   vk::DescriptorSetLayout,
    waves_set:          vk::DescriptorSet,
    waves_pipeline_layout: vk::PipelineLayout,
    waves_pipeline:     vk::Pipeline,
    pub waves: Vec<Wave>,

    // Text overlay.
    font_image:           vk::Image,
    font_memory:          vk::DeviceMemory,
    font_view:            vk::ImageView,
    font_sampler:         vk::Sampler,
    text_set_layout:      vk::DescriptorSetLayout,
    text_set:             vk::DescriptorSet,
    text_pipeline_layout: vk::PipelineLayout,
    text_pipeline:        vk::Pipeline,
    text_ssbo:            vk::Buffer,
    text_ssbo_memory:     vk::DeviceMemory,
    text_ssbo_mapped:     *mut TextQuadGPU,

    // FPS is averaged over a ~0.25 s window. Independent of vsync and
    // simulation time scaling.
    fps_accum_frames: u32,
    fps_accum_start:  f32,
    fps_current:      f32,

    // Visibility flags.
    pub show_volume:    bool,
    pub show_particles: bool,
    pub show_waves:     bool,

    // Runtime perf toggles. perf_mipskip gates the hierarchical skip in
    // raymarch.frag via the uniform; perf_async gates whether wavefunction
    // and mipmap work is submitted to the async compute queue or inline
    // in the graphics command buffer.
    pub perf_mipskip:   bool,
    pub perf_async:     bool,

    needs_particle_init: bool,
    frame_counter:       u32,
    last_frame_time:     f32,
}

impl VulkanRenderer {

    pub fn new(window: &Window) -> Self {
        unsafe {
            let entry = Entry::load().expect("Failed to load Vulkan runtime (vulkan-1.dll)");

            
            // Instance
            
            let app_name    = CString::new("QuantumAtomViz").unwrap();
            let engine_name = CString::new("Custom").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_2);

            let instance_extensions = [
                khr::surface::NAME.as_ptr(),
                khr::win32_surface::NAME.as_ptr(),
            ];
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&instance_extensions),
                None,
            ).expect("vkCreateInstance");

            
            // Surface
            
            let win32_loader = khr::win32_surface::Instance::new(&entry, &instance);
            let surface = win32_loader.create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(window.hinstance as isize)
                    .hwnd(window.hwnd as isize),
                None,
            ).unwrap();
            let surface_loader = khr::surface::Instance::new(&entry, &instance);

            // Extended physical-device selection: picks primary G+C+present
            // family and opportunistically identifies an async compute
            // family for parallel wavefunction evaluation.
            let (physical_device, caps) =
                pick_physical_device(&instance, &surface_loader, surface);
            let queue_family = caps.graphics_family;

            // Startup log. Mirrors what the overlay shows on screen so
            // the user can verify from the console what capabilities the
            // selected adapter actually exposed.
            println!("GPU:     {} {}",
                     vendor_name(caps.vendor_id), caps.device_name);
            println!("Type:    {:?}, VRAM: {} MiB", caps.device_type, caps.device_memory_mb);
            println!("Driver:  0x{:08x}   API {}.{}.{}",
                     caps.driver_version,
                     vk::api_version_major(caps.api_version),
                     vk::api_version_minor(caps.api_version),
                     vk::api_version_patch(caps.api_version));
            println!("Queues:  graphics = {}, async compute = {}",
                     caps.graphics_family,
                     caps.async_compute_family
                         .map(|f| f.to_string())
                         .unwrap_or_else(|| "none".to_string()));
            println!("Subgroup size: {}", caps.subgroup_size);

            
            // Logical device (+ async compute queue if available)
            
            let features = vk::PhysicalDeviceFeatures::default()
                .fill_mode_non_solid(true);
            let mut dyn_render = vk::PhysicalDeviceDynamicRenderingFeatures::default()
                .dynamic_rendering(true);
            // Vulkan 1.2 core already includes timeline semaphores but
            // some drivers still want the explicit feature bit.
            let mut timeline_sem = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
                .timeline_semaphore(true);

            let queue_priorities = [1.0f32];
            let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(caps.graphics_family)
                    .queue_priorities(&queue_priorities)
            );
            if let Some(acf) = caps.async_compute_family {
                queue_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(acf)
                        .queue_priorities(&queue_priorities)
                );
            }

            let device_extensions = [
                khr::swapchain::NAME.as_ptr(),
                khr::dynamic_rendering::NAME.as_ptr(),
                khr::timeline_semaphore::NAME.as_ptr(),
            ];
            let device = instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&device_extensions)
                    .enabled_features(&features)
                    .push_next(&mut dyn_render)
                    .push_next(&mut timeline_sem),
                None,
            ).expect("vkCreateDevice");
            let queue = device.get_device_queue(caps.graphics_family, 0);

            // Optional async compute queue + timeline semaphore + per
            // frame command pool. All allocated upfront; runtime toggle
            // just chooses whether to submit on it or inline.
            let async_compute = if let Some(acf) = caps.async_compute_family {
                let q = device.get_device_queue(acf, 0);

                let mut ttype = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let timeline = device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut ttype),
                    None,
                ).unwrap();

                let mut pools = Vec::with_capacity(FRAMES_IN_FLIGHT);
                let mut bufs  = Vec::with_capacity(FRAMES_IN_FLIGHT);
                for _ in 0..FRAMES_IN_FLIGHT {
                    let pool = device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                            .queue_family_index(acf),
                        None,
                    ).unwrap();
                    let cb = device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    ).unwrap()[0];
                    pools.push(pool);
                    bufs.push(cb);
                }
                Some(AsyncCompute {
                    queue: q, queue_family: acf,
                    timeline, timeline_value: 0,
                    cmd_pools: pools, cmd_buffers: bufs,
                })
            } else { None };

            
            // Swapchain
            
            let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
            let (swapchain, swap_format, swap_extent, swap_images, swap_views) =
                create_swapchain(&surface_loader, &swapchain_loader, &device,
                                physical_device, surface,
                                window.width, window.height);

            let volume_sampler = create_sampler(&device);

            
            // Descriptor pool sized for everything we will allocate below
            
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: (FRAMES_IN_FLIGHT * 2 + 2) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: (FRAMES_IN_FLIGHT * 3 + 6) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (FRAMES_IN_FLIGHT * 3) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 6,
                },
            ];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets((FRAMES_IN_FLIGHT * 3 + 6) as u32),
                None,
            ).unwrap();

            
            // Wavefunction compute pipeline (writes complex psi to volume)
            
            let comp_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let compute_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&comp_bindings),
                None,
            ).unwrap();

            let comp_layouts = [compute_set_layout];
            let compute_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&comp_layouts),
                None,
            ).unwrap();

            let comp_module = create_shader_module(&device, COMP_SPV);
            let entry_name = CString::new("main").unwrap();
            let compute_pipeline = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(comp_module)
                        .name(&entry_name))
                    .layout(compute_pipeline_layout)],
                None,
            ).unwrap()[0];
            device.destroy_shader_module(comp_module, None);

            
            // Mipmap compute pipeline (coarse max density for empty-space skip)
            
            let mip_bindings = [
                vk::DescriptorSetLayoutBinding::default().binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default().binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let mipmap_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&mip_bindings),
                None,
            ).unwrap();
            let mip_set_layouts = [mipmap_set_layout];
            let mip_pc_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: size_of::<MipmapPC>() as u32,
            }];
            let mipmap_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&mip_set_layouts)
                    .push_constant_ranges(&mip_pc_range),
                None,
            ).unwrap();
            let mip_module = create_shader_module(&device, MIPMAP_COMP_SPV);
            let mipmap_pipeline = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(mip_module).name(&entry_name))
                    .layout(mipmap_pipeline_layout)],
                None,
            ).unwrap()[0];
            device.destroy_shader_module(mip_module, None);

            
            // Volume + heatmap graphics pipelines.
            // Binding 2 (psi_max sampler) is declared in the shared set
            // layout so raymarch.frag can use it; heatmap.frag just
            // ignores the binding (unused in its shader source).
            
            let gfx_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            let graphics_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&gfx_bindings),
                None,
            ).unwrap();

            let gfx_layouts = [graphics_set_layout];
            let graphics_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&gfx_layouts),
                None,
            ).unwrap();

            let graphics_pipeline = create_graphics_pipeline(
                &device, graphics_pipeline_layout, swap_format,
                VERT_SPV, FRAG_SPV);
            let heatmap_pipeline = create_graphics_pipeline(
                &device, graphics_pipeline_layout, swap_format,
                HEATMAP_VERT_SPV, HEATMAP_FRAG_SPV);

            
            // Per-frame resources (cmd buffer, fence/sems, UBOs, volume,
            // psi_max image, descriptor sets for compute/mipmap/graphics)
            
            let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let cmd_pool = device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(caps.graphics_family),
                    None,
                ).unwrap();
                let cmd_buffer = device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                ).unwrap()[0];

                let image_available = device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(), None).unwrap();
                let render_finished = device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default(), None).unwrap();
                let in_flight_fence = device.create_fence(
                    &vk::FenceCreateInfo::default()
                        .flags(vk::FenceCreateFlags::SIGNALED), None).unwrap();

                let (camera_buffer, camera_memory, camera_mapped) =
                    create_host_ubo::<CameraUBO>(&instance, &device, physical_device);
                let (components_buffer, components_memory, components_mapped) =
                    create_host_ubo::<ComponentsUBO>(&instance, &device, physical_device);

                let (volume_image, volume_memory, volume_view) =
                    create_volume_3d(&instance, &device, physical_device,
                                     GRID_SIZE, vk::Format::R32G32_SFLOAT);
                let (psi_max_image, psi_max_memory, psi_max_view) =
                    create_volume_3d(&instance, &device, physical_device,
                                     COARSE_GRID_SIZE, vk::Format::R32_SFLOAT);

                let set_layouts_compute = [compute_set_layout];
                let compute_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts_compute),
                ).unwrap()[0];

                let set_layouts_mip = [mipmap_set_layout];
                let mipmap_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts_mip),
                ).unwrap()[0];

                let set_layouts_gfx = [graphics_set_layout];
                let graphics_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts_gfx),
                ).unwrap()[0];

                let comp_image = [vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: volume_view,
                    image_layout: vk::ImageLayout::GENERAL,
                }];
                let comp_buf = [vk::DescriptorBufferInfo {
                    buffer: components_buffer,
                    offset: 0,
                    range: size_of::<ComponentsUBO>() as u64,
                }];
                let mip_src = [vk::DescriptorImageInfo {
                    sampler: volume_sampler,
                    image_view: volume_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let mip_dst = [vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: psi_max_view,
                    image_layout: vk::ImageLayout::GENERAL,
                }];
                let gfx_vol = [vk::DescriptorImageInfo {
                    sampler: volume_sampler,
                    image_view: volume_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let gfx_buf = [vk::DescriptorBufferInfo {
                    buffer: camera_buffer,
                    offset: 0,
                    range: size_of::<CameraUBO>() as u64,
                }];
                let gfx_max = [vk::DescriptorImageInfo {
                    sampler: volume_sampler,
                    image_view: psi_max_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                device.update_descriptor_sets(&[
                    vk::WriteDescriptorSet::default()
                        .dst_set(compute_set).dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&comp_image),
                    vk::WriteDescriptorSet::default()
                        .dst_set(compute_set).dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&comp_buf),
                    vk::WriteDescriptorSet::default()
                        .dst_set(mipmap_set).dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&mip_src),
                    vk::WriteDescriptorSet::default()
                        .dst_set(mipmap_set).dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&mip_dst),
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_set).dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&gfx_vol),
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_set).dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&gfx_buf),
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_set).dst_binding(2)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&gfx_max),
                ], &[]);

                frames.push(FrameData {
                    cmd_pool, cmd_buffer,
                    image_available, render_finished, in_flight_fence,
                    camera_buffer, camera_memory, camera_mapped,
                    components_buffer, components_memory, components_mapped,
                    volume_image, volume_memory, volume_view,
                    psi_max_image, psi_max_memory, psi_max_view,
                    compute_set, mipmap_set, graphics_set,
                    initialized: false,
                });
            }

            
            // Particle subsystem
            
            let particle_size = (PARTICLE_COUNT as usize * 16) as u64;
            let particle_buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(particle_size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            ).unwrap();
            let req = device.get_buffer_memory_requirements(particle_buffer);
            let mt = find_memory_type(&instance, physical_device, req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL);
            let particle_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size).memory_type_index(mt),
                None,
            ).unwrap();
            device.bind_buffer_memory(particle_buffer, particle_memory, 0).unwrap();

            let p_bindings = [
                vk::DescriptorSetLayoutBinding::default().binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default().binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let particles_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&p_bindings),
                None,
            ).unwrap();
            let p_set_layouts = [particles_set_layout];
            let p_pc_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: size_of::<ParticlesPC>() as u32,
            }];
            let particles_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&p_set_layouts)
                    .push_constant_ranges(&p_pc_range),
                None,
            ).unwrap();
            let p_module = create_shader_module(&device, PARTICLES_COMP_SPV);
            let particles_compute_pipeline = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(p_module).name(&entry_name))
                    .layout(particles_pipeline_layout)],
                None,
            ).unwrap()[0];
            device.destroy_shader_module(p_module, None);
            let particles_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&p_set_layouts),
            ).unwrap()[0];

            let pg_bindings = [
                vk::DescriptorSetLayoutBinding::default().binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX),
                vk::DescriptorSetLayoutBinding::default().binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default().binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            let particles_gfx_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&pg_bindings),
                None,
            ).unwrap();
            let pg_set_layouts = [particles_gfx_set_layout];
            let pg_pc_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<GfxMatrixPC>() as u32,
            }];
            let particles_gfx_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&pg_set_layouts)
                    .push_constant_ranges(&pg_pc_range),
                None,
            ).unwrap();
            let particles_gfx_pipeline = create_particle_pipeline(
                &device, particles_gfx_pipeline_layout, swap_format,
                PARTICLES_VERT_SPV, PARTICLES_FRAG_SPV,
            );
            let particles_gfx_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&pg_set_layouts),
            ).unwrap()[0];

            let p_img = [vk::DescriptorImageInfo {
                sampler: volume_sampler,
                image_view: frames[0].volume_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let p_buf = [vk::DescriptorBufferInfo {
                buffer: particle_buffer, offset: 0, range: vk::WHOLE_SIZE,
            }];
            let p_cam = [vk::DescriptorBufferInfo {
                buffer: frames[0].camera_buffer, offset: 0,
                range: size_of::<CameraUBO>() as u64,
            }];
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default()
                    .dst_set(particles_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&p_buf),
                vk::WriteDescriptorSet::default()
                    .dst_set(particles_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&p_img),
                vk::WriteDescriptorSet::default()
                    .dst_set(particles_gfx_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&p_buf),
                vk::WriteDescriptorSet::default()
                    .dst_set(particles_gfx_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&p_cam),
                vk::WriteDescriptorSet::default()
                    .dst_set(particles_gfx_set).dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&p_img),
            ], &[]);

            
            // Wave subsystem (instanced wireframe icospheres)
            
            let (wave_vertices, wave_indices) = build_icosphere(2);
            let (wave_vertex_buffer, wave_vertex_memory) = create_device_buffer_with_data(
                &instance, &device, physical_device, queue, caps.graphics_family,
                &wave_vertices, vk::BufferUsageFlags::VERTEX_BUFFER,
            );
            let (wave_index_buffer, wave_index_memory) = create_device_buffer_with_data(
                &instance, &device, physical_device, queue, caps.graphics_family,
                &wave_indices, vk::BufferUsageFlags::INDEX_BUFFER,
            );
            let wave_index_count = wave_indices.len() as u32;

            let wave_buf_size = (MAX_WAVES as usize * size_of::<WaveGPU>()) as u64;
            let (wave_buffer, wave_memory, wave_mapped_raw) = create_host_storage_buffer(
                &instance, &device, physical_device, wave_buf_size,
            );
            let wave_mapped = wave_mapped_raw as *mut WaveGPU;

            let w_bindings = [vk::DescriptorSetLayoutBinding::default().binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)];
            let waves_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&w_bindings),
                None,
            ).unwrap();
            let w_set_layouts = [waves_set_layout];
            let w_pc_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: 128, // mat4 view + mat4 proj
            }];
            let waves_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&w_set_layouts)
                    .push_constant_ranges(&w_pc_range),
                None,
            ).unwrap();
            let waves_pipeline = create_wave_pipeline(
                &device, waves_pipeline_layout, swap_format,
                WAVES_VERT_SPV, WAVES_FRAG_SPV,
            );
            let waves_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&w_set_layouts),
            ).unwrap()[0];
            let w_buf = [vk::DescriptorBufferInfo {
                buffer: wave_buffer, offset: 0, range: vk::WHOLE_SIZE,
            }];
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default()
                    .dst_set(waves_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&w_buf),
            ], &[]);

            
            // Text overlay subsystem (font.hex -> R8 atlas + instanced pipeline)
            
            let atlas_pixels = build_font_atlas(&parse_font_hex_ascii(FONT_HEX));
            let (font_image, font_memory, font_view) =
                create_device_image_2d_with_data(
                    &instance, &device, physical_device, queue, caps.graphics_family,
                    ATLAS_W, ATLAS_H, vk::Format::R8_UNORM, &atlas_pixels,
                );

            let font_sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
                    .unnormalized_coordinates(false),
                None,
            ).unwrap();

            let text_ssbo_size = (MAX_TEXT_QUADS as usize * size_of::<TextQuadGPU>()) as u64;
            let (text_ssbo, text_ssbo_memory, text_ssbo_raw) =
                create_host_storage_buffer(
                    &instance, &device, physical_device, text_ssbo_size);
            let text_ssbo_mapped = text_ssbo_raw as *mut TextQuadGPU;

            let t_bindings = [
                vk::DescriptorSetLayoutBinding::default().binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX),
                vk::DescriptorSetLayoutBinding::default().binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            let text_set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&t_bindings),
                None,
            ).unwrap();
            let t_set_layouts = [text_set_layout];
            let t_pc_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<TextPC>() as u32,
            }];
            let text_pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&t_set_layouts)
                    .push_constant_ranges(&t_pc_range),
                None,
            ).unwrap();
            let text_pipeline = create_text_pipeline(
                &device, text_pipeline_layout, swap_format,
                TEXT_VERT_SPV, TEXT_FRAG_SPV,
            );
            let text_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&t_set_layouts),
            ).unwrap()[0];
            let t_buf = [vk::DescriptorBufferInfo {
                buffer: text_ssbo, offset: 0, range: vk::WHOLE_SIZE,
            }];
            let t_img = [vk::DescriptorImageInfo {
                sampler: font_sampler,
                image_view: font_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default()
                    .dst_set(text_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&t_buf),
                vk::WriteDescriptorSet::default()
                    .dst_set(text_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&t_img),
            ], &[]);

            VulkanRenderer {
                _entry: entry, instance, surface_loader, surface,
                physical_device, device, queue_family: caps.graphics_family, queue,
                caps,
                swapchain_loader, swapchain, swap_format, swap_extent,
                swap_images, swap_views,
                volume_sampler,
                descriptor_pool,
                compute_set_layout, compute_pipeline_layout, compute_pipeline,
                mipmap_set_layout, mipmap_pipeline_layout, mipmap_pipeline,
                graphics_set_layout, graphics_pipeline_layout,
                graphics_pipeline, heatmap_pipeline,
                frames, frame_index: 0,
                async_compute,

                components: Vec::new(),
                time_scale: 3.0,
                max_density: 0.05,

                cam_yaw:    0.6,
                cam_pitch:  0.45,
                cam_radius: 55.0,
                auto_orbit: true,

                view_mode:    ViewMode::Volume,
                slice_axis:   SliceAxis::XZ,
                slice_offset: 0.0,
                color_mode:   ColorMode::Real,
                show_contour: true,

                particle_buffer, particle_memory,
                particles_set_layout, particles_set,
                particles_pipeline_layout, particles_compute_pipeline,
                particles_gfx_set_layout, particles_gfx_set,
                particles_gfx_pipeline_layout, particles_gfx_pipeline,

                wave_buffer, wave_memory, wave_mapped,
                wave_index_buffer, wave_index_memory,
                wave_vertex_buffer, wave_vertex_memory,
                wave_index_count,
                waves_set_layout, waves_set,
                waves_pipeline_layout, waves_pipeline,
                waves: Vec::new(),

                font_image, font_memory, font_view, font_sampler,
                text_set_layout, text_set,
                text_pipeline_layout, text_pipeline,
                text_ssbo, text_ssbo_memory, text_ssbo_mapped,

                fps_accum_frames: 0,
                fps_accum_start:  0.0,
                fps_current:      0.0,

                show_volume: true,
                show_particles: false,
                show_waves: true,

                perf_mipskip: true,
                perf_async:   true,

                needs_particle_init: true,
                frame_counter: 0,
                last_frame_time: 0.0,
            }
        }
    }

    pub fn render_frame(&mut self, time_seconds: f32) {
        unsafe {
            // FPS accumulator: count frames in a ~0.25 s window.
            if self.fps_accum_frames == 0 {
                self.fps_accum_start = time_seconds;
            }
            self.fps_accum_frames += 1;
            let fps_elapsed = time_seconds - self.fps_accum_start;
            if fps_elapsed >= 0.25 {
                self.fps_current = self.fps_accum_frames as f32 / fps_elapsed;
                self.fps_accum_frames = 0;
                self.fps_accum_start  = time_seconds;
            }

            let frame_idx = self.frame_index;

            // Snapshot per-frame handles.
            let cmd_buffer       = self.frames[frame_idx].cmd_buffer;
            let in_flight_fence  = self.frames[frame_idx].in_flight_fence;
            let image_available  = self.frames[frame_idx].image_available;
            let render_finished  = self.frames[frame_idx].render_finished;
            let volume_image     = self.frames[frame_idx].volume_image;
            let psi_max_image    = self.frames[frame_idx].psi_max_image;
            let compute_set      = self.frames[frame_idx].compute_set;
            let mipmap_set       = self.frames[frame_idx].mipmap_set;
            let graphics_set     = self.frames[frame_idx].graphics_set;
            let was_init         = self.frames[frame_idx].initialized;

            
            // Frame setup: wait, acquire, reset
            
            self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX).unwrap();
            let (image_index, _) = match self.swapchain_loader.acquire_next_image(
                self.swapchain, u64::MAX, image_available, vk::Fence::null(),
            ) {
                Ok(v) => v,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
                Err(e) => panic!("acquire: {:?}", e),
            };
            self.device.reset_fences(&[in_flight_fence]).unwrap();

            
            // Host-mapped uniform buffer updates
            
            let atomic_time = time_seconds * self.time_scale;
            self.write_components_ubo(&self.frames[frame_idx], atomic_time);
            self.write_camera_ubo(&self.frames[frame_idx], time_seconds);

            let dt = (time_seconds - self.last_frame_time).clamp(0.0, 0.05);
            self.last_frame_time = time_seconds;
            self.update_waves(dt);
            let alive_waves = self.waves.len() as u32;

            
            // Decide compute submission strategy
            //
            //  - If perf_async is on and an async compute queue exists,
            //    record wavefunction + mipmap to the async command buffer
            //    and submit it to the compute queue, signaling a timeline
            //    value the graphics submission waits on.
            //
            //  - Otherwise, record the same work inline at the head of
            //    the graphics command buffer (old behaviour).
            
            let use_async = self.perf_async && self.async_compute.is_some();

            let compute_timeline_value = if use_async {
                let ac = self.async_compute.as_mut().unwrap();
                ac.timeline_value += 1;
                let tv = ac.timeline_value;

                let acb = ac.cmd_buffers[frame_idx];
                let acf = ac.queue_family;
                let gf  = self.queue_family;

                self.device.reset_command_buffer(
                    acb, vk::CommandBufferResetFlags::empty()).unwrap();
                self.device.begin_command_buffer(
                    acb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ).unwrap();

                record_compute_work(
                    &self.device, acb,
                    volume_image, psi_max_image,
                    self.compute_pipeline, self.compute_pipeline_layout, compute_set,
                    self.mipmap_pipeline, self.mipmap_pipeline_layout, mipmap_set,
                    was_init,
                    // Release to graphics family if different.
                    if gf != acf { Some((acf, gf)) } else { None },
                );

                self.device.end_command_buffer(acb).unwrap();

                // Submit on async compute queue: no binary waits, signal
                // timeline at value tv. Wait on image_available is NOT
                // needed here; the compute work does not touch the
                // swapchain image.
                let mut tinfo = vk::TimelineSemaphoreSubmitInfo::default()
                    .signal_semaphore_values(std::slice::from_ref(&tv));
                let signal_sems = [ac.timeline];
                let cmd_arr = [acb];
                self.device.queue_submit(
                    ac.queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&cmd_arr)
                        .signal_semaphores(&signal_sems)
                        .push_next(&mut tinfo)],
                    vk::Fence::null(),
                ).unwrap();

                Some(tv)
            } else { None };

            
            // Graphics command buffer recording
            
            self.device.reset_command_buffer(
                cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();

            if !use_async {
                // Compute work inline on the graphics queue. No ownership
                // transfer required because the queue family does not
                // change between pipeline bind points.
                record_compute_work(
                    &self.device, cmd_buffer,
                    volume_image, psi_max_image,
                    self.compute_pipeline, self.compute_pipeline_layout, compute_set,
                    self.mipmap_pipeline, self.mipmap_pipeline_layout, mipmap_set,
                    was_init,
                    None,
                );
            } else {
                // Async path: acquire the two images from the compute
                // queue family if they differ from graphics.
                let ac = self.async_compute.as_ref().unwrap();
                if ac.queue_family != self.queue_family {
                    queue_family_acquire(
                        &self.device, cmd_buffer, volume_image,
                        ac.queue_family, self.queue_family,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                    queue_family_acquire(
                        &self.device, cmd_buffer, psi_max_image,
                        ac.queue_family, self.queue_family,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }
            }

            
            // Particles compute pass (still inline on graphics queue to
            // avoid doubling the SSBO ownership-transfer dance).
            
            let do_particles = self.show_particles && frame_idx == 0;
            if do_particles {
                self.frame_counter = self.frame_counter.wrapping_add(1);
                let mode = if self.needs_particle_init { 0u32 } else { 1u32 };
                let voxel_size = 2.0 * HALF_EXTENT / GRID_SIZE as f32;
                let pc = ParticlesPC {
                    half_extent:   HALF_EXTENT,
                    max_density:   self.max_density,
                    voxel_size,
                    dt:            dt * self.time_scale,
                    mode,
                    seed:          self.frame_counter,
                    num_particles: PARTICLE_COUNT,
                    _pad: 0,
                };
                let pc_bytes = std::slice::from_raw_parts(
                    &pc as *const _ as *const u8, size_of::<ParticlesPC>()
                );
                self.device.cmd_bind_pipeline(
                    cmd_buffer, vk::PipelineBindPoint::COMPUTE,
                    self.particles_compute_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer, vk::PipelineBindPoint::COMPUTE,
                    self.particles_pipeline_layout, 0, &[self.particles_set], &[]);
                self.device.cmd_push_constants(
                    cmd_buffer, self.particles_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
                let p_groups = (PARTICLE_COUNT + 63) / 64;
                self.device.cmd_dispatch(cmd_buffer, p_groups, 1, 1);

                let bbarr = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.particle_buffer)
                    .offset(0).size(vk::WHOLE_SIZE);
                self.device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[bbarr], &[]);

                self.needs_particle_init = false;
            }

            
            // Begin dynamic rendering into the swapchain image
            
            let swap_image = self.swap_images[image_index as usize];
            let swap_view  = self.swap_views[image_index as usize];

            image_barrier(
                &self.device, cmd_buffer, swap_image,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::empty(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(swap_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                });
            let attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: self.swap_extent,
                })
                .layer_count(1)
                .color_attachments(&attachments);
            self.device.cmd_begin_rendering(cmd_buffer, &rendering_info);

            self.device.cmd_set_viewport(cmd_buffer, 0, &[vk::Viewport {
                x: 0.0, y: 0.0,
                width:  self.swap_extent.width  as f32,
                height: self.swap_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            }]);
            self.device.cmd_set_scissor(cmd_buffer, 0, &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_extent,
            }]);

            
            // Draw 1: fullscreen volume ray marcher OR heatmap
            
            if self.show_volume {
                let pipeline = match self.view_mode {
                    ViewMode::Volume  => self.graphics_pipeline,
                    ViewMode::Heatmap => self.heatmap_pipeline,
                };
                self.device.cmd_bind_pipeline(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline_layout, 0, &[graphics_set], &[]);
                self.device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
            }

            let aspect = self.swap_extent.width as f32 / self.swap_extent.height as f32;
            let proj = perspective(45.0f32.to_radians(), aspect, 0.1, 500.0);
            let cam_pos = self.current_cam_pos(time_seconds);
            let view = look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

            
            // Draw 2: particle billboards
            
            if self.show_particles {
                let pc = GfxMatrixPC {
                    view, proj,
                    point_size: 0.18,
                    _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
                };
                let pc_bytes = std::slice::from_raw_parts(
                    &pc as *const _ as *const u8, size_of::<GfxMatrixPC>());
                self.device.cmd_bind_pipeline(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.particles_gfx_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.particles_gfx_pipeline_layout, 0, &[self.particles_gfx_set], &[]);
                self.device.cmd_push_constants(
                    cmd_buffer, self.particles_gfx_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX, 0, pc_bytes);
                self.device.cmd_draw(cmd_buffer, 6, PARTICLE_COUNT, 0, 0);
            }

            
            // Draw 3: wave shells
            
            if self.show_waves && alive_waves > 0 {
                let mut buf = [0u8; 128];
                let view_bytes = std::slice::from_raw_parts(view.as_ptr() as *const u8, 64);
                let proj_bytes = std::slice::from_raw_parts(proj.as_ptr() as *const u8, 64);
                buf[..64].copy_from_slice(view_bytes);
                buf[64..128].copy_from_slice(proj_bytes);

                self.device.cmd_bind_pipeline(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.waves_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.waves_pipeline_layout, 0, &[self.waves_set], &[]);
                self.device.cmd_push_constants(
                    cmd_buffer, self.waves_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX, 0, &buf);
                self.device.cmd_bind_vertex_buffers(
                    cmd_buffer, 0, &[self.wave_vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(
                    cmd_buffer, self.wave_index_buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed(
                    cmd_buffer, self.wave_index_count,
                    alive_waves.min(MAX_WAVES), 0, 0, 0);
            }

            
            // Draw 4: FPS / diagnostic text overlay
            
            let num_text_quads = self.build_overlay_text_quads();
            if num_text_quads > 0 {
                let pc = TextPC {
                    screen_size: [
                        self.swap_extent.width  as f32,
                        self.swap_extent.height as f32,
                    ],
                    _pad: [0.0; 2],
                };
                let pc_bytes = std::slice::from_raw_parts(
                    &pc as *const _ as *const u8, size_of::<TextPC>());
                self.device.cmd_bind_pipeline(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.text_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
                    self.text_pipeline_layout, 0, &[self.text_set], &[]);
                self.device.cmd_push_constants(
                    cmd_buffer, self.text_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX, 0, pc_bytes);
                self.device.cmd_draw(cmd_buffer, 6, num_text_quads, 0, 0);
            }

            self.device.cmd_end_rendering(cmd_buffer);

            image_barrier(
                &self.device, cmd_buffer, swap_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            );

            self.device.end_command_buffer(cmd_buffer).unwrap();

            
            // Graphics submission (waits on image_available + optional
            // compute timeline, signals render_finished + the fence).
            
            let mut wait_sems  = vec![image_available];
            let mut wait_stage = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let mut wait_vals  = vec![0u64];
            if let Some(tv) = compute_timeline_value {
                let ac = self.async_compute.as_ref().unwrap();
                wait_sems.push(ac.timeline);
                // Wait at the earliest stage that reads the volume/psi_max.
                wait_stage.push(vk::PipelineStageFlags::FRAGMENT_SHADER);
                wait_vals.push(tv);
            }
            let signal_sems = [render_finished];
            let signal_vals = [0u64];
            let cmd_bufs    = [cmd_buffer];
            let mut tinfo = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_vals)
                .signal_semaphore_values(&signal_vals);

            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stage)
                    .command_buffers(&cmd_bufs)
                    .signal_semaphores(&signal_sems)
                    .push_next(&mut tinfo)],
                in_flight_fence,
            ).unwrap();

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let _ = self.swapchain_loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_sems)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices),
            );

            self.frames[frame_idx].initialized = true;
            self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        }
    }

    fn update_waves(&mut self, dt: f32) {
        if self.waves.is_empty() { return; }
        for w in self.waves.iter_mut() {
            if !w.alive { continue; }
            w.radius += w.speed * dt;
            w.age    += dt;
            if w.radius > w.max_radius { w.alive = false; }
        }
        self.waves.retain(|w| w.alive);
        let n = self.waves.len().min(MAX_WAVES as usize);
        for i in 0..n {
            let w = &self.waves[i];
            let g = WaveGPU {
                origin_radius: [w.origin[0], w.origin[1], w.origin[2], w.radius],
                color_age:     [w.color[0],  w.color[1],  w.color[2],  w.age],
            };
            unsafe { ptr::write(self.wave_mapped.add(i), g); }
        }
    }

    // Lays out three short diagnostic lines at the top-left of the screen:
    // the FPS counter, the GPU name, and the active optimization flags.
    // Returns the glyph-instance count for vkCmdDraw.
    fn build_overlay_text_quads(&mut self) -> u32 {
        let fps_text   = format!("FPS: {:5.1}", self.fps_current);
        let gpu_text   = format!("GPU: {} {}",
                                  self.caps.vendor_str(),
                                  truncate_str(&self.caps.device_name, 40));
        let async_str  = if self.caps.has_async_compute() {
            if self.perf_async { "ASYNC:ON " } else { "ASYNC:OFF" }
        } else { "ASYNC:N/A" };
        let mip_str    = if self.perf_mipskip { "MIP:ON " } else { "MIP:OFF" };
        let perf_text  = format!("{}  {}", async_str, mip_str);

        let lines = [
            (fps_text.as_str(),  [1.0f32, 1.0, 0.4, 1.0]),
            (gpu_text.as_str(),  [0.7f32, 0.9, 1.0, 1.0]),
            (perf_text.as_str(), [0.7f32, 1.0, 0.7, 1.0]),
        ];

        let scale: f32 = 2.0;
        let gw = GLYPH_W as f32 * scale;
        let gh = GLYPH_H as f32 * scale;
        let line_gap: f32 = 2.0;
        let origin_x: f32 = 10.0;
        let mut y: f32 = 10.0;

        let mut count: u32 = 0;
        for (s, color) in lines.iter() {
            let mut pen_x = origin_x;
            for ch in s.bytes() {
                if count >= MAX_TEXT_QUADS { return count; }
                if ch == b' ' { pen_x += gw; continue; }
                if ch >= 0x80 { pen_x += gw; continue; }

                let cx = (ch as u32) % ATLAS_COLS;
                let cy = (ch as u32) / ATLAS_COLS;
                let u0 = (cx * GLYPH_W) as f32 / ATLAS_W as f32;
                let v0 = (cy * GLYPH_H) as f32 / ATLAS_H as f32;
                let u1 = ((cx + 1) * GLYPH_W) as f32 / ATLAS_W as f32;
                let v1 = ((cy + 1) * GLYPH_H) as f32 / ATLAS_H as f32;

                let q = TextQuadGPU {
                    rect:    [pen_x, y, gw, gh],
                    uv_rect: [u0, v0, u1, v1],
                    color:   *color,
                };
                unsafe { ptr::write(self.text_ssbo_mapped.add(count as usize), q); }
                count  += 1;
                pen_x  += gw;
            }
            y += gh + line_gap;
        }
        count
    }

    fn write_components_ubo(&self, frame: &FrameData, atomic_time: f32) {
        let mut ubo = ComponentsUBO {
            num_components: self.components.len() as i32,
            grid_size: GRID_SIZE as i32,
            half_extent: HALF_EXTENT,
            time: atomic_time,
            nlm_z: [[0.0; 4]; MAX_COMPONENTS],
            amp:   [[0.0; 4]; MAX_COMPONENTS],
        };
        for (i, c) in self.components.iter().take(MAX_COMPONENTS).enumerate() {
            ubo.nlm_z[i] = [c.n as f32, c.l as f32, c.m as f32, c.z];
            ubo.amp[i]   = [c.c_real, c.c_imag, c.energy, 0.0];
        }
        unsafe { ptr::write(frame.components_mapped, ubo); }
    }

    fn write_camera_ubo(&self, frame: &FrameData, t: f32) {
        let cam_pos = self.current_cam_pos(t);
        let view = look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let view_inv = mat4_invert(view);
        let aspect = self.swap_extent.width as f32 / self.swap_extent.height as f32;
        let proj = perspective(45.0f32.to_radians(), aspect, 0.1, 500.0);
        let proj_inv = mat4_invert(proj);

        let voxel_size = 2.0 * HALF_EXTENT / GRID_SIZE as f32;
        let view_mode_f = match self.view_mode {
            ViewMode::Volume  => 0.0,
            ViewMode::Heatmap => 1.0,
        };
        let slice_axis_f = self.slice_axis as i32 as f32;
        let color_mode_f = self.color_mode as i32 as f32;
        let contour_f    = if self.show_contour { 1.0 } else { 0.0 };
        let mipskip_f    = if self.perf_mipskip { 1.0 } else { 0.0 };

        let ubo = CameraUBO {
            view_inv, proj_inv,
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 1.0],
            domain_params:  [HALF_EXTENT, self.max_density, voxel_size, view_mode_f],
            render_params:  [0.10, 18.0, 0.02, 0.65],
            heatmap_params: [slice_axis_f, self.slice_offset, color_mode_f, contour_f],
            perf_params:    [mipskip_f, COARSE_GRID_SIZE as f32, 0.0, 0.0],
        };
        unsafe { ptr::write(frame.camera_mapped, ubo); }
    }

    pub fn camera_drag(&mut self, dx: f32, dy: f32) {
        let sens = 0.005;
        self.cam_yaw   -= dx * sens;
        self.cam_pitch += dy * sens;
        let limit = std::f32::consts::FRAC_PI_2 - 0.05;
        if self.cam_pitch >  limit { self.cam_pitch =  limit; }
        if self.cam_pitch < -limit { self.cam_pitch = -limit; }
    }

    pub fn camera_zoom(&mut self, notches: f32) {
        let factor = (-notches * 0.10).exp();
        self.cam_radius = (self.cam_radius * factor).clamp(8.0, 250.0);
    }

    pub fn current_cam_pos(&self, t: f32) -> [f32; 3] {
        let (yaw, pitch) = if self.auto_orbit {
            (self.cam_yaw + t * 0.20, self.cam_pitch)
        } else {
            (self.cam_yaw, self.cam_pitch)
        };
        let r = self.cam_radius;
        [
            r * pitch.cos() * yaw.cos(),
            r * pitch.sin(),
            r * pitch.cos() * yaw.sin(),
        ]
    }

    pub fn emit_wave(&mut self, origin: [f32; 3], color: [f32; 3], max_radius: f32, speed: f32) {
        if (self.waves.len() as u32) >= MAX_WAVES { self.waves.remove(0); }
        self.waves.push(Wave {
            origin, radius: 0.0, speed, max_radius,
            color, age: 0.0, alive: true,
        });
    }

    pub fn request_particle_reseed(&mut self) {
        self.needs_particle_init = true;
    }

    pub fn toggle_view_mode(&mut self) {
        self.view_mode = match self.view_mode {
            ViewMode::Volume  => ViewMode::Heatmap,
            ViewMode::Heatmap => ViewMode::Volume,
        };
    }

    pub fn cycle_slice_axis(&mut self) {
        self.slice_axis = match self.slice_axis {
            SliceAxis::XY => SliceAxis::XZ,
            SliceAxis::XZ => SliceAxis::YZ,
            SliceAxis::YZ => SliceAxis::XY,
        };
    }

    pub fn cycle_color_mode(&mut self) {
        self.color_mode = match self.color_mode {
            ColorMode::Density => ColorMode::Real,
            ColorMode::Real    => ColorMode::Phase,
            ColorMode::Phase   => ColorMode::Density,
        };
    }

    pub fn nudge_slice(&mut self, delta: f32) {
        self.slice_offset = (self.slice_offset + delta).clamp(-1.0, 1.0);
    }

    // Runtime perf toggles. Toggling async compute forces a device idle
    // so the change takes effect on a clean frame boundary.
    pub fn toggle_perf_mipskip(&mut self) {
        self.perf_mipskip = !self.perf_mipskip;
    }
    pub fn toggle_perf_async(&mut self) {
        if !self.caps.has_async_compute() { return; }
        unsafe { self.device.device_wait_idle().ok(); }
        self.perf_async = !self.perf_async;
        // Force recomputation next frame so images are in a known state.
        for f in self.frames.iter_mut() { f.initialized = false; }
    }

    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().unwrap(); }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Text overlay teardown.
            self.device.destroy_pipeline(self.text_pipeline, None);
            self.device.destroy_pipeline_layout(self.text_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.text_set_layout, None);
            self.device.unmap_memory(self.text_ssbo_memory);
            self.device.destroy_buffer(self.text_ssbo, None);
            self.device.free_memory(self.text_ssbo_memory, None);
            self.device.destroy_sampler(self.font_sampler, None);
            self.device.destroy_image_view(self.font_view, None);
            self.device.destroy_image(self.font_image, None);
            self.device.free_memory(self.font_memory, None);

            for f in &self.frames {
                self.device.unmap_memory(f.camera_memory);
                self.device.unmap_memory(f.components_memory);
                self.device.destroy_buffer(f.camera_buffer, None);
                self.device.free_memory(f.camera_memory, None);
                self.device.destroy_buffer(f.components_buffer, None);
                self.device.free_memory(f.components_memory, None);
                self.device.destroy_image_view(f.volume_view, None);
                self.device.destroy_image(f.volume_image, None);
                self.device.free_memory(f.volume_memory, None);
                self.device.destroy_image_view(f.psi_max_view, None);
                self.device.destroy_image(f.psi_max_image, None);
                self.device.free_memory(f.psi_max_memory, None);
                self.device.destroy_fence(f.in_flight_fence, None);
                self.device.destroy_semaphore(f.image_available, None);
                self.device.destroy_semaphore(f.render_finished, None);
                self.device.destroy_command_pool(f.cmd_pool, None);
            }

            if let Some(ac) = self.async_compute.take() {
                for p in ac.cmd_pools { self.device.destroy_command_pool(p, None); }
                self.device.destroy_semaphore(ac.timeline, None);
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.heatmap_pipeline, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.graphics_set_layout, None);
            self.device.destroy_pipeline(self.mipmap_pipeline, None);
            self.device.destroy_pipeline_layout(self.mipmap_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.mipmap_set_layout, None);
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device.destroy_pipeline_layout(self.compute_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.compute_set_layout, None);
            self.device.destroy_sampler(self.volume_sampler, None);
            for v in &self.swap_views {
                self.device.destroy_image_view(*v, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.device.destroy_pipeline(self.particles_compute_pipeline, None);
            self.device.destroy_pipeline_layout(self.particles_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.particles_set_layout, None);
            self.device.destroy_pipeline(self.particles_gfx_pipeline, None);
            self.device.destroy_pipeline_layout(self.particles_gfx_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.particles_gfx_set_layout, None);
            self.device.destroy_buffer(self.particle_buffer, None);
            self.device.free_memory(self.particle_memory, None);

            self.device.destroy_pipeline(self.waves_pipeline, None);
            self.device.destroy_pipeline_layout(self.waves_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.waves_set_layout, None);
            self.device.unmap_memory(self.wave_memory);
            self.device.destroy_buffer(self.wave_buffer, None);
            self.device.free_memory(self.wave_memory, None);
            self.device.destroy_buffer(self.wave_vertex_buffer, None);
            self.device.free_memory(self.wave_vertex_memory, None);
            self.device.destroy_buffer(self.wave_index_buffer, None);
            self.device.free_memory(self.wave_index_memory, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// ==================  helpers  ===================================

// Records the wavefunction + mipmap compute chain into an already-begun
// command buffer. Optionally emits release barriers that transfer queue
// family ownership to `dst_family` (for async compute on a separate queue
// family). When `release_to` is None, the resources stay on the current
// queue family.
unsafe fn record_compute_work(
    device: &Device, cmd: vk::CommandBuffer,
    volume_image: vk::Image, psi_max_image: vk::Image,
    wf_pipe: vk::Pipeline, wf_layout: vk::PipelineLayout, wf_set: vk::DescriptorSet,
    mip_pipe: vk::Pipeline, mip_layout: vk::PipelineLayout, mip_set: vk::DescriptorSet,
    was_init: bool,
    release_to: Option<(u32, u32)>,  // (src_family, dst_family)
) {
    // Transition both images to GENERAL for compute write. We intentionally
    // start from UNDEFINED to discard previous contents even when they
    // existed in SHADER_READ_ONLY layout - the new pass overwrites every
    // voxel so there is nothing worth preserving.
    let old = if was_init { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }
              else        { vk::ImageLayout::UNDEFINED };
    image_barrier(
        device, cmd, volume_image,
        old, vk::ImageLayout::GENERAL,
        vk::AccessFlags::SHADER_READ, vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    image_barrier(
        device, cmd, psi_max_image,
        old, vk::ImageLayout::GENERAL,
        vk::AccessFlags::SHADER_READ, vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );

    // Wavefunction evaluation.
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, wf_pipe);
    device.cmd_bind_descriptor_sets(
        cmd, vk::PipelineBindPoint::COMPUTE,
        wf_layout, 0, &[wf_set], &[]);
    let g = (GRID_SIZE + 7) / 8;
    device.cmd_dispatch(cmd, g, g, g);

    // Volume: GENERAL -> SHADER_READ_ONLY so mipmap.comp can sample it.
    image_barrier(
        device, cmd, volume_image,
        vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER,
    );

    // Min-max mipmap build.
    let mpc = MipmapPC {
        fine_grid_size:   GRID_SIZE as i32,
        coarse_grid_size: COARSE_GRID_SIZE as i32,
        block_size:       COARSE_BLOCK as i32,
        _pad: 0,
    };
    let mpc_bytes = std::slice::from_raw_parts(
        &mpc as *const _ as *const u8, size_of::<MipmapPC>());
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mip_pipe);
    device.cmd_bind_descriptor_sets(
        cmd, vk::PipelineBindPoint::COMPUTE,
        mip_layout, 0, &[mip_set], &[]);
    device.cmd_push_constants(
        cmd, mip_layout, vk::ShaderStageFlags::COMPUTE, 0, mpc_bytes);
    let cg = (COARSE_GRID_SIZE + 3) / 4;
    device.cmd_dispatch(cmd, cg, cg, cg);

    // psi_max: GENERAL -> SHADER_READ_ONLY for the fragment shader.
    image_barrier(
        device, cmd, psi_max_image,
        vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
    );

    // If we are on a different queue family than the subsequent graphics
    // submission, release ownership here. The matching acquire will be
    // recorded at the head of the graphics command buffer.
    if let Some((src_fam, dst_fam)) = release_to {
        queue_family_release(device, cmd, volume_image, src_fam, dst_fam,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        queue_family_release(device, cmd, psi_max_image, src_fam, dst_fam,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}

unsafe fn queue_family_release(
    device: &Device, cmd: vk::CommandBuffer, image: vk::Image,
    src_family: u32, dst_family: u32,
    layout: vk::ImageLayout,
) {
    let b = vk::ImageMemoryBarrier::default()
        .old_layout(layout).new_layout(layout)
        .src_queue_family_index(src_family)
        .dst_queue_family_index(dst_family)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        vk::DependencyFlags::empty(),
        &[], &[], &[b]);
}

unsafe fn queue_family_acquire(
    device: &Device, cmd: vk::CommandBuffer, image: vk::Image,
    src_family: u32, dst_family: u32,
    layout: vk::ImageLayout,
) {
    let b = vk::ImageMemoryBarrier::default()
        .old_layout(layout).new_layout(layout)
        .src_queue_family_index(src_family)
        .dst_queue_family_index(dst_family)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[], &[], &[b]);
}

unsafe fn create_host_ubo<T>(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice,
) -> (vk::Buffer, vk::DeviceMemory, *mut T) {
    let buffer = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size_of::<T>() as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    ).unwrap();
    let req = device.get_buffer_memory_requirements(buffer);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
    let memory = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt),
        None,
    ).unwrap();
    device.bind_buffer_memory(buffer, memory, 0).unwrap();
    let ptr = device.map_memory(memory, 0, vk::WHOLE_SIZE,
        vk::MemoryMapFlags::empty()).unwrap() as *mut T;
    (buffer, memory, ptr)
}

unsafe fn pick_physical_device(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, DeviceCaps) {
    let devices = instance.enumerate_physical_devices().unwrap();
    let mut best: Option<(vk::PhysicalDevice, DeviceCaps, i32)> = None;
    for pd in devices {
        let props = instance.get_physical_device_properties(pd);
        let mem_props = instance.get_physical_device_memory_properties(pd);
        let qprops = instance.get_physical_device_queue_family_properties(pd);

        // Subgroup size via Vulkan 1.1 properties2 chain.
        let mut sub_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut sub_props);
        instance.get_physical_device_properties2(pd, &mut props2);

        // Find the primary family: must support graphics + compute + present.
        let mut primary: Option<u32> = None;
        for (i, q) in qprops.iter().enumerate() {
            let pres = surface_loader
                .get_physical_device_surface_support(pd, i as u32, surface)
                .unwrap_or(false);
            if pres
                && q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && q.queue_flags.contains(vk::QueueFlags::COMPUTE)
            {
                primary = Some(i as u32);
                break;
            }
        }
        let primary = match primary { Some(p) => p, None => continue };

        // Opportunistically pick an async compute family: prefer a
        // compute-only family (no graphics bit) distinct from primary;
        // fall back to any other compute-capable family.
        let mut async_fam: Option<u32> = None;
        for (i, q) in qprops.iter().enumerate() {
            if i as u32 == primary { continue; }
            if q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                async_fam = Some(i as u32);
                break;
            }
        }
        if async_fam.is_none() {
            for (i, q) in qprops.iter().enumerate() {
                if i as u32 == primary { continue; }
                if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    async_fam = Some(i as u32);
                    break;
                }
            }
        }

        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU   => 100,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
            _ => 10,
        };

        // Approximate VRAM from the largest DEVICE_LOCAL heap.
        let mut vram_bytes: u64 = 0;
        for i in 0..mem_props.memory_heap_count {
            let h = mem_props.memory_heaps[i as usize];
            if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) && h.size > vram_bytes {
                vram_bytes = h.size;
            }
        }

        let name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy().into_owned();

        let caps = DeviceCaps {
            vendor_id:           props.vendor_id,
            device_id:           props.device_id,
            device_name:         name,
            driver_version:      props.driver_version,
            api_version:         props.api_version,
            graphics_family:     primary,
            async_compute_family:async_fam,
            subgroup_size:       sub_props.subgroup_size,
            device_type:         props.device_type,
            device_memory_mb:    vram_bytes / (1024 * 1024),
        };

        if best.as_ref().map(|b| b.2).unwrap_or(-1) < score {
            best = Some((pd, caps, score));
        }
    }
    let (pd, caps, _) = best.expect("no suitable Vulkan device");
    (pd, caps)
}

unsafe fn create_swapchain(
    surface_loader:   &khr::surface::Instance,
    swapchain_loader: &khr::swapchain::Device,
    device:           &Device,
    pd:               vk::PhysicalDevice,
    surface:          vk::SurfaceKHR,
    width: u32, height: u32,
) -> (vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>, Vec<vk::ImageView>) {
    let caps = surface_loader.get_physical_device_surface_capabilities(pd, surface).unwrap();
    let formats = surface_loader.get_physical_device_surface_formats(pd, surface).unwrap();
    let modes = surface_loader.get_physical_device_surface_present_modes(pd, surface).unwrap();
    let format = formats.iter().copied().find(|f|
        f.format == vk::Format::B8G8R8A8_SRGB
        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    ).unwrap_or(formats[0]);
    let mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else { vk::PresentModeKHR::FIFO };
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width:  width.clamp(caps.min_image_extent.width,  caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    let mut count = caps.min_image_count + 1;
    if caps.max_image_count > 0 { count = count.min(caps.max_image_count); }
    let swapchain = swapchain_loader.create_swapchain(
        &vk::SwapchainCreateInfoKHR::default()
            .surface(surface).min_image_count(count)
            .image_format(format.format).image_color_space(format.color_space)
            .image_extent(extent).image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode).clipped(true),
        None,
    ).unwrap();
    let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
    let views: Vec<_> = images.iter().map(|&img|
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(img).view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }),
            None,
        ).unwrap()
    ).collect();
    (swapchain, format.format, extent, images, views)
}

// Generalized 3D storage/sampler image creator. format must be usable with
// both STORAGE and SAMPLED (all variants of SFLOAT and UNORM qualify on
// current desktop drivers).
unsafe fn create_volume_3d(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice,
    size: u32, format: vk::Format,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(format)
            .extent(vk::Extent3D { width: size, height: size, depth: size })
            .mip_levels(1).array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None,
    ).unwrap();
    let req = device.get_image_memory_requirements(image);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let memory = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size).memory_type_index(mt),
        None,
    ).unwrap();
    device.bind_image_memory(image, memory, 0).unwrap();
    let view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image).view_type(vk::ImageViewType::TYPE_3D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            }),
        None,
    ).unwrap();
    (image, memory, view)
}

unsafe fn create_sampler(device: &Device) -> vk::Sampler {
    device.create_sampler(
        &vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false),
        None,
    ).unwrap()
}

unsafe fn create_shader_module(device: &Device, bytes: &[u8]) -> vk::ShaderModule {
    assert!(bytes.len() % 4 == 0);
    let mut code = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        code.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&code), None).unwrap()
}

unsafe fn create_graphics_pipeline(
    device: &Device, layout: vk::PipelineLayout, color_format: vk::Format,
    vert_spv: &[u8], frag_spv: &[u8],
) -> vk::Pipeline {
    let vert = create_shader_module(device, vert_spv);
    let frag = create_shader_module(device, frag_spv);
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&entry),
    ];
    let vi = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let cba = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cba);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
        .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
        .color_blend_state(&cb).dynamic_state(&dyn_state).layout(layout)
        .push_next(&mut rendering);
    let pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(), &[info], None).unwrap()[0];
    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    pipeline
}

unsafe fn find_memory_type(
    instance: &Instance, pd: vk::PhysicalDevice,
    type_bits: u32, props: vk::MemoryPropertyFlags,
) -> u32 {
    let mp = instance.get_physical_device_memory_properties(pd);
    for i in 0..mp.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mp.memory_types[i as usize].property_flags.contains(props) {
            return i;
        }
    }
    panic!("no suitable memory type")
}

unsafe fn image_barrier(
    device: &Device, cmd: vk::CommandBuffer, image: vk::Image,
    old_layout: vk::ImageLayout, new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags, dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout).new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        })
        .src_access_mask(src_access).dst_access_mask(dst_access);
    device.cmd_pipeline_barrier(
        cmd, src_stage, dst_stage, vk::DependencyFlags::empty(),
        &[], &[], &[barrier]);
}

unsafe fn create_host_storage_buffer(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice, size: u64,
) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
    let buf = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    ).unwrap();
    let req = device.get_buffer_memory_requirements(buf);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
    let mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
        None,
    ).unwrap();
    device.bind_buffer_memory(buf, mem, 0).unwrap();
    let ptr = device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
    (buf, mem, ptr)
}

unsafe fn create_device_buffer_with_data<T: Copy>(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice,
    queue: vk::Queue, queue_family: u32,
    data: &[T], usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let bytes = (data.len() * size_of::<T>()) as u64;

    let staging = device.create_buffer(
        &vk::BufferCreateInfo::default().size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE), None).unwrap();
    let req = device.get_buffer_memory_requirements(staging);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
    let staging_mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
        None).unwrap();
    device.bind_buffer_memory(staging, staging_mem, 0).unwrap();
    let p = device.map_memory(staging_mem, 0, bytes, vk::MemoryMapFlags::empty()).unwrap();
    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p as *mut u8, bytes as usize);
    device.unmap_memory(staging_mem);

    let buf = device.create_buffer(
        &vk::BufferCreateInfo::default().size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST | usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE), None).unwrap();
    let req = device.get_buffer_memory_requirements(buf);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
        None).unwrap();
    device.bind_buffer_memory(buf, mem, 0).unwrap();

    let pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family), None).unwrap();
    let cb = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)).unwrap()[0];
    device.begin_command_buffer(cb,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
    let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes };
    device.cmd_copy_buffer(cb, staging, buf, &[region]);
    device.end_command_buffer(cb).unwrap();
    let cb_arr = [cb];
    device.queue_submit(queue,
        &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
        vk::Fence::null()).unwrap();
    device.queue_wait_idle(queue).unwrap();
    device.destroy_command_pool(pool, None);
    device.destroy_buffer(staging, None);
    device.free_memory(staging_mem, None);

    (buf, mem)
}

fn build_icosphere(subdiv: u32) -> (Vec<f32>, Vec<u32>) {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut verts: Vec<[f32; 3]> = vec![
        [-1.0,  t,  0.0], [ 1.0,  t,  0.0], [-1.0, -t,  0.0], [ 1.0, -t,  0.0],
        [ 0.0, -1.0,  t], [ 0.0,  1.0,  t], [ 0.0, -1.0, -t], [ 0.0,  1.0, -t],
        [ t,  0.0, -1.0], [ t,  0.0,  1.0], [-t,  0.0, -1.0], [-t,  0.0,  1.0],
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [ 0,11, 5],[ 0, 5, 1],[ 0, 1, 7],[ 0, 7,10],[ 0,10,11],
        [ 1, 5, 9],[ 5,11, 4],[11,10, 2],[10, 7, 6],[ 7, 1, 8],
        [ 3, 9, 4],[ 3, 4, 2],[ 3, 2, 6],[ 3, 6, 8],[ 3, 8, 9],
        [ 4, 9, 5],[ 2, 4,11],[ 6, 2,10],[ 8, 6, 7],[ 9, 8, 1],
    ];
    for v in verts.iter_mut() {
        let l = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
        v[0]/=l; v[1]/=l; v[2]/=l;
    }
    for _ in 0..subdiv {
        use std::collections::HashMap;
        let mut cache: HashMap<(u32,u32), u32> = HashMap::new();
        let mut new_tris = Vec::new();
        for tri in &tris {
            let mut mid = |a: u32, b: u32| -> u32 {
                let key = if a<b { (a,b) } else { (b,a) };
                if let Some(&i) = cache.get(&key) { return i; }
                let va = verts[a as usize];
                let vb = verts[b as usize];
                let mut m = [(va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5];
                let l = (m[0]*m[0]+m[1]*m[1]+m[2]*m[2]).sqrt();
                m[0]/=l; m[1]/=l; m[2]/=l;
                let i = verts.len() as u32;
                verts.push(m);
                cache.insert(key, i);
                i
            };
            let a = mid(tri[0], tri[1]);
            let b = mid(tri[1], tri[2]);
            let c = mid(tri[2], tri[0]);
            new_tris.push([tri[0], a, c]);
            new_tris.push([tri[1], b, a]);
            new_tris.push([tri[2], c, b]);
            new_tris.push([a, b, c]);
        }
        tris = new_tris;
    }
    let mut vbuf = Vec::with_capacity(verts.len() * 3);
    for v in verts { vbuf.extend_from_slice(&v); }
    let mut ibuf = Vec::with_capacity(tris.len() * 3);
    for t in tris { ibuf.extend_from_slice(&t); }
    (vbuf, ibuf)
}

unsafe fn create_particle_pipeline(
    device: &Device, layout: vk::PipelineLayout, color_format: vk::Format,
    vert_spv: &[u8], frag_spv: &[u8],
) -> vk::Pipeline {
    let vert = create_shader_module(device, vert_spv);
    let frag = create_shader_module(device, frag_spv);
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&entry),
    ];
    let vi = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let cba = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cba);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
        .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
        .color_blend_state(&cb).dynamic_state(&dyn_state).layout(layout)
        .push_next(&mut rendering);
    let pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(), &[info], None).unwrap()[0];
    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    pipeline
}

unsafe fn create_wave_pipeline(
    device: &Device, layout: vk::PipelineLayout, color_format: vk::Format,
    vert_spv: &[u8], frag_spv: &[u8],
) -> vk::Pipeline {
    let vert = create_shader_module(device, vert_spv);
    let frag = create_shader_module(device, frag_spv);
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&entry),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0, stride: 12, input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attrs = [vk::VertexInputAttributeDescription {
        location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0,
    }];
    let vi = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::LINE)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let cba = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cba);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
        .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
        .color_blend_state(&cb).dynamic_state(&dyn_state).layout(layout)
        .push_next(&mut rendering);
    let pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(), &[info], None).unwrap()[0];
    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    pipeline
}

// Text overlay helpers (shared with the previous revision).
fn parse_font_hex_ascii(data: &[u8]) -> [[u8; 16]; 128] {
    let mut glyphs = [[0u8; 16]; 128];
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return glyphs,
    };
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let (cp_str, hex) = match line.split_once(':') {
            Some(v) => v, None => continue,
        };
        let cp = match u32::from_str_radix(cp_str.trim(), 16) {
            Ok(v) => v, Err(_) => continue,
        };
        if cp >= 128 { continue; }
        let hex = hex.trim();
        if hex.len() == 32 {
            for i in 0..16 {
                let s = &hex[i*2..i*2+2];
                glyphs[cp as usize][i] = u8::from_str_radix(s, 16).unwrap_or(0);
            }
        } else if hex.len() == 64 {
            for i in 0..16 {
                let s = &hex[i*4..i*4+2];
                glyphs[cp as usize][i] = u8::from_str_radix(s, 16).unwrap_or(0);
            }
        }
    }
    glyphs
}

fn build_font_atlas(glyphs: &[[u8; 16]; 128]) -> Vec<u8> {
    let w = ATLAS_W as usize;
    let h = ATLAS_H as usize;
    let mut px = vec![0u8; w * h];
    for c in 0..128u32 {
        let cx = (c % ATLAS_COLS) * GLYPH_W;
        let cy = (c / ATLAS_COLS) * GLYPH_H;
        for row in 0..GLYPH_H {
            let bits = glyphs[c as usize][row as usize];
            for bit in 0..GLYPH_W {
                let on = (bits >> (7 - bit)) & 1;
                if on != 0 {
                    let x = (cx + bit) as usize;
                    let y = (cy + row) as usize;
                    px[y * w + x] = 0xFF;
                }
            }
        }
    }
    px
}

unsafe fn create_device_image_2d_with_data(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice,
    queue: vk::Queue, queue_family: u32,
    width: u32, height: u32, format: vk::Format, data: &[u8],
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    let staging = device.create_buffer(
        &vk::BufferCreateInfo::default().size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE), None).unwrap();
    let req = device.get_buffer_memory_requirements(staging);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
    let staging_mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
        None).unwrap();
    device.bind_buffer_memory(staging, staging_mem, 0).unwrap();
    let p = device.map_memory(staging_mem, 0, data.len() as u64,
        vk::MemoryMapFlags::empty()).unwrap();
    std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, data.len());
    device.unmap_memory(staging_mem);

    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1).array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None).unwrap();
    let req = device.get_image_memory_requirements(image);
    let mt = find_memory_type(instance, pd, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let memory = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
        None).unwrap();
    device.bind_image_memory(image, memory, 0).unwrap();

    let pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family), None).unwrap();
    let cb = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)).unwrap()[0];
    device.begin_command_buffer(cb,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

    image_barrier(device, cb, image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER);

    let region = vk::BufferImageCopy {
        buffer_offset: 0, buffer_row_length: 0, buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0, base_array_layer: 0, layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D { width, height, depth: 1 },
    };
    device.cmd_copy_buffer_to_image(
        cb, staging, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);

    image_barrier(device, cb, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER);

    device.end_command_buffer(cb).unwrap();
    let cb_arr = [cb];
    device.queue_submit(queue,
        &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
        vk::Fence::null()).unwrap();
    device.queue_wait_idle(queue).unwrap();
    device.destroy_command_pool(pool, None);
    device.destroy_buffer(staging, None);
    device.free_memory(staging_mem, None);

    let view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image).view_type(vk::ImageViewType::TYPE_2D).format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            }),
        None).unwrap();

    (image, memory, view)
}

unsafe fn create_text_pipeline(
    device: &Device, layout: vk::PipelineLayout, color_format: vk::Format,
    vert_spv: &[u8], frag_spv: &[u8],
) -> vk::Pipeline {
    let vert = create_shader_module(device, vert_spv);
    let frag = create_shader_module(device, frag_spv);
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&entry),
    ];
    let vi = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let cba = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cba);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dyn_states);
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
        .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
        .color_blend_state(&cb).dynamic_state(&dyn_state).layout(layout)
        .push_next(&mut rendering);
    let pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(), &[info], None).unwrap()[0];
    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    pipeline
}

fn truncate_str(s: &str, n: usize) -> &str {
    if s.len() <= n { s } else { &s[..n] }
}

// ---- minimal linear algebra ----------------------------------------
fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy * 0.5).tan();
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = f / aspect;
    m[1][1] = -f;
    m[2][2] = far / (near - far);
    m[2][3] = -1.0;
    m[3][2] = (near * far) / (near - far);
    m
}
fn look_at(eye: [f32;3], center: [f32;3], up: [f32;3]) -> [[f32;4];4] {
    let f = nrm(sub(center, eye));
    let s = nrm(crs(f, up));
    let u = crs(s, f);
    let mut m = [[0.0f32;4];4];
    m[0][0]=s[0]; m[1][0]=s[1]; m[2][0]=s[2];
    m[0][1]=u[0]; m[1][1]=u[1]; m[2][1]=u[2];
    m[0][2]=-f[0]; m[1][2]=-f[1]; m[2][2]=-f[2];
    m[3][0]=-dt(s,eye); m[3][1]=-dt(u,eye); m[3][2]=dt(f,eye); m[3][3]=1.0;
    m
}
fn sub(a:[f32;3], b:[f32;3])->[f32;3]{[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
fn crs(a:[f32;3], b:[f32;3])->[f32;3]{
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn dt(a:[f32;3], b:[f32;3])->f32{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
fn nrm(a:[f32;3])->[f32;3]{
    let l=(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt().max(1e-8);
    [a[0]/l,a[1]/l,a[2]/l]
}
fn mat4_invert(m: [[f32;4];4]) -> [[f32;4];4] {
    let a = m;
    let mut inv = [[0.0f32;4];4];
    inv[0][0] =  a[1][1]*a[2][2]*a[3][3] - a[1][1]*a[2][3]*a[3][2] - a[2][1]*a[1][2]*a[3][3]
              +  a[2][1]*a[1][3]*a[3][2] + a[3][1]*a[1][2]*a[2][3] - a[3][1]*a[1][3]*a[2][2];
    inv[1][0] = -a[1][0]*a[2][2]*a[3][3] + a[1][0]*a[2][3]*a[3][2] + a[2][0]*a[1][2]*a[3][3]
              -  a[2][0]*a[1][3]*a[3][2] - a[3][0]*a[1][2]*a[2][3] + a[3][0]*a[1][3]*a[2][2];
    inv[2][0] =  a[1][0]*a[2][1]*a[3][3] - a[1][0]*a[2][3]*a[3][1] - a[2][0]*a[1][1]*a[3][3]
              +  a[2][0]*a[1][3]*a[3][1] + a[3][0]*a[1][1]*a[2][3] - a[3][0]*a[1][3]*a[2][1];
    inv[3][0] = -a[1][0]*a[2][1]*a[3][2] + a[1][0]*a[2][2]*a[3][1] + a[2][0]*a[1][1]*a[3][2]
              -  a[2][0]*a[1][2]*a[3][1] - a[3][0]*a[1][1]*a[2][2] + a[3][0]*a[1][2]*a[2][1];
    inv[0][1] = -a[0][1]*a[2][2]*a[3][3] + a[0][1]*a[2][3]*a[3][2] + a[2][1]*a[0][2]*a[3][3]
              -  a[2][1]*a[0][3]*a[3][2] - a[3][1]*a[0][2]*a[2][3] + a[3][1]*a[0][3]*a[2][2];
    inv[1][1] =  a[0][0]*a[2][2]*a[3][3] - a[0][0]*a[2][3]*a[3][2] - a[2][0]*a[0][2]*a[3][3]
              +  a[2][0]*a[0][3]*a[3][2] + a[3][0]*a[0][2]*a[2][3] - a[3][0]*a[0][3]*a[2][2];
    inv[2][1] = -a[0][0]*a[2][1]*a[3][3] + a[0][0]*a[2][3]*a[3][1] + a[2][0]*a[0][1]*a[3][3]
              -  a[2][0]*a[0][3]*a[3][1] - a[3][0]*a[0][1]*a[2][3] + a[3][0]*a[0][3]*a[2][1];
    inv[3][1] =  a[0][0]*a[2][1]*a[3][2] - a[0][0]*a[2][2]*a[3][1] - a[2][0]*a[0][1]*a[3][2]
              +  a[2][0]*a[0][2]*a[3][1] + a[3][0]*a[0][1]*a[2][2] - a[3][0]*a[0][2]*a[2][1];
    inv[0][2] =  a[0][1]*a[1][2]*a[3][3] - a[0][1]*a[1][3]*a[3][2] - a[1][1]*a[0][2]*a[3][3]
              +  a[1][1]*a[0][3]*a[3][2] + a[3][1]*a[0][2]*a[1][3] - a[3][1]*a[0][3]*a[1][2];
    inv[1][2] = -a[0][0]*a[1][2]*a[3][3] + a[0][0]*a[1][3]*a[3][2] + a[1][0]*a[0][2]*a[3][3]
              -  a[1][0]*a[0][3]*a[3][2] - a[3][0]*a[0][2]*a[1][3] + a[3][0]*a[0][3]*a[1][2];
    inv[2][2] =  a[0][0]*a[1][1]*a[3][3] - a[0][0]*a[1][3]*a[3][1] - a[1][0]*a[0][1]*a[3][3]
              +  a[1][0]*a[0][3]*a[3][1] + a[3][0]*a[0][1]*a[1][3] - a[3][0]*a[0][3]*a[1][1];
    inv[3][2] = -a[0][0]*a[1][1]*a[3][2] + a[0][0]*a[1][2]*a[3][1] + a[1][0]*a[0][1]*a[3][2]
              -  a[1][0]*a[0][2]*a[3][1] - a[3][0]*a[0][1]*a[1][2] + a[3][0]*a[0][2]*a[1][1];
    inv[0][3] = -a[0][1]*a[1][2]*a[2][3] + a[0][1]*a[1][3]*a[2][2] + a[1][1]*a[0][2]*a[2][3]
              -  a[1][1]*a[0][3]*a[2][2] - a[2][1]*a[0][2]*a[1][3] + a[2][1]*a[0][3]*a[1][2];
    inv[1][3] =  a[0][0]*a[1][2]*a[2][3] - a[0][0]*a[1][3]*a[2][2] - a[1][0]*a[0][2]*a[2][3]
              +  a[1][0]*a[0][3]*a[2][2] + a[2][0]*a[0][2]*a[1][3] - a[2][0]*a[0][3]*a[1][2];
    inv[2][3] = -a[0][0]*a[1][1]*a[2][3] + a[0][0]*a[1][3]*a[2][1] + a[1][0]*a[0][1]*a[2][3]
              -  a[1][0]*a[0][3]*a[2][1] - a[2][0]*a[0][1]*a[1][3] + a[2][0]*a[0][3]*a[1][1];
    inv[3][3] =  a[0][0]*a[1][1]*a[2][2] - a[0][0]*a[1][2]*a[2][1] - a[1][0]*a[0][1]*a[2][2]
              +  a[1][0]*a[0][2]*a[2][1] + a[2][0]*a[0][1]*a[1][2] - a[2][0]*a[0][2]*a[1][1];
    let det = a[0][0]*inv[0][0] + a[0][1]*inv[1][0] + a[0][2]*inv[2][0] + a[0][3]*inv[3][0];
    let inv_det = 1.0 / det;
    for i in 0..4 { for j in 0..4 { inv[i][j] *= inv_det; } }
    inv
}