#![allow(clippy::too_many_arguments)]

use ash::{vk, Entry, Instance, Device};
use ash::khr;
use std::ffi::CString;
use std::mem::size_of;
use std::ptr;

use crate::win32::Window;

const COMP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wavefunction.comp.spv"));
const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raymarch.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raymarch.frag.spv"));
const HEATMAP_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heatmap.vert.spv"));
const HEATMAP_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heatmap.frag.spv"));
const PARTICLES_COMP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.comp.spv"));
const PARTICLES_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.vert.spv"));
const PARTICLES_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particles.frag.spv"));
const WAVES_VERT_SPV:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/waves.vert.spv"));
const WAVES_FRAG_SPV:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/waves.frag.spv"));

pub const GRID_SIZE: u32 = 128;
pub const HALF_EXTENT: f32 = 20.0;
pub const FRAMES_IN_FLIGHT: usize = 2;
pub const MAX_COMPONENTS: usize = 8;
pub const PARTICLE_COUNT: u32 = 32_768;
pub const MAX_WAVES:      u32 = 64;

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

#[repr(C)]
struct CameraUBO {
    view_inv:      [[f32; 4]; 4],
    proj_inv:      [[f32; 4]; 4],
    camera_pos:    [f32; 4],
    domain_params:  [f32; 4], // halfExtent, maxDensity, voxelSize, viewMode
    render_params:  [f32; 4], // stepSize, opacityScale, threshold, gamma
    heatmap_params: [f32; 4], // sliceAxis, sliceOffset, colorMode, contourFlag
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

    // Per-frame volume image so frame N+1 cannot stomp on what frame N
    // is still sampling. Avoids the cross-submission WAR hazard without
    // requiring timeline semaphores.
    volume_image:  vk::Image,
    volume_memory: vk::DeviceMemory,
    volume_view:   vk::ImageView,

    compute_set:  vk::DescriptorSet,
    graphics_set: vk::DescriptorSet,

    volume_initialized: bool,
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
    origin_radius: [f32; 4],   // xyz origin, w current radius
    color_age:     [f32; 4],   // rgb color, a age (seconds)
}

#[derive(Clone, Copy)]
pub struct Wave {
    pub origin: [f32; 3],
    pub radius: f32,
    pub speed:  f32,           // Bohr per second
    pub max_radius: f32,
    pub color:  [f32; 3],
    pub age:    f32,
    pub alive:  bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Volume,
    Heatmap,
}

#[derive(Clone, Copy)]
pub enum SliceAxis { XY = 0, XZ = 1, YZ = 2 }

#[derive(Clone, Copy)]
pub enum ColorMode { Density = 0, Real = 1, Phase = 2 }

pub struct VulkanRenderer {
    _entry:           Entry,
    instance:         Instance,
    surface_loader:   khr::surface::Instance,
    surface:          vk::SurfaceKHR,
    physical_device:  vk::PhysicalDevice,
    device:           Device,
    queue_family:     u32,
    queue:            vk::Queue,

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
    graphics_set_layout:     vk::DescriptorSetLayout,
    graphics_pipeline_layout:vk::PipelineLayout,
    graphics_pipeline:       vk::Pipeline,

    frames:      Vec<FrameData>,
    frame_index: usize,

    pub components:   Vec<OrbitalComponent>,
    pub time_scale:   f32,
    pub max_density:  f32,

    // Camera state.
    pub cam_yaw:    f32,   // radians
    pub cam_pitch:  f32,   // radians, clamped to (-pi/2 + eps, pi/2 - eps)
    pub cam_radius: f32,   // Bohr radii
    pub auto_orbit: bool,

    // Heatmap state.
    pub view_mode:    ViewMode,
    pub slice_axis:   SliceAxis,
    pub slice_offset: f32,  // [-1, 1]
    pub color_mode:   ColorMode,
    pub show_contour: bool,

    // Second graphics pipeline for the heatmap path. Shares everything else
    // with the volume pipeline (same layout, same descriptor sets).
    heatmap_pipeline: vk::Pipeline,
    // Particle subsystem (shared across all in-flight frames; the volume
    // barrier between compute and graphics also covers SSBO read after write).
    particle_buffer:        vk::Buffer,
    particle_memory:        vk::DeviceMemory,
    particles_set_layout:   vk::DescriptorSetLayout,
    particles_set:          vk::DescriptorSet,
    particles_pipeline_layout: vk::PipelineLayout,
    particles_compute_pipeline: vk::Pipeline,

    // Particle graphics path uses a different layout (SSBO + camera UBO + sampler).
    particles_gfx_set_layout:    vk::DescriptorSetLayout,
    particles_gfx_set:           vk::DescriptorSet,
    particles_gfx_pipeline_layout: vk::PipelineLayout,
    particles_gfx_pipeline:      vk::Pipeline,

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

    // Visibility flags.
    pub show_volume:    bool,
    pub show_particles: bool,
    pub show_waves:     bool,

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

            let (physical_device, queue_family) =
                pick_physical_device(&instance, &surface_loader, surface);

            
            // Logical device
            
            // fillModeNonSolid is required because the wave shells render as
            // wireframe icospheres via vk::PolygonMode::LINE.
            let features = vk::PhysicalDeviceFeatures::default()
                .fill_mode_non_solid(true);
            let mut dyn_render = vk::PhysicalDeviceDynamicRenderingFeatures::default()
                .dynamic_rendering(true);
            let queue_priorities = [1.0f32];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities);
            let queue_infos = [queue_info];
            let device_extensions = [
                khr::swapchain::NAME.as_ptr(),
                khr::dynamic_rendering::NAME.as_ptr(),
            ];
            let device = instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&device_extensions)
                    .enabled_features(&features)
                    .push_next(&mut dyn_render),
                None,
            ).expect("vkCreateDevice");
            let queue = device.get_device_queue(queue_family, 0);

            
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
                    descriptor_count: (FRAMES_IN_FLIGHT + 1) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: (FRAMES_IN_FLIGHT + 4) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (FRAMES_IN_FLIGHT * 3) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 4,
                },
            ];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets((FRAMES_IN_FLIGHT * 2 + 4) as u32),
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

            
            // Volume + heatmap graphics pipelines (fullscreen ray marcher
            // and fullscreen heatmap; both consume sampler3D + camera UBO)
            
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

            
            // Per-frame resources (cmd buffer, fence/sems, UBOs, volume)
            
            let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let cmd_pool = device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family),
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
                    create_volume_3d(&instance, &device, physical_device, GRID_SIZE);

                let set_layouts_compute = [compute_set_layout];
                let compute_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts_compute),
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
                let gfx_image = [vk::DescriptorImageInfo {
                    sampler: volume_sampler,
                    image_view: volume_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let gfx_buf = [vk::DescriptorBufferInfo {
                    buffer: camera_buffer,
                    offset: 0,
                    range: size_of::<CameraUBO>() as u64,
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
                        .dst_set(graphics_set).dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&gfx_image),
                    vk::WriteDescriptorSet::default()
                        .dst_set(graphics_set).dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&gfx_buf),
                ], &[]);

                frames.push(FrameData {
                    cmd_pool, cmd_buffer,
                    image_available, render_finished, in_flight_fence,
                    camera_buffer, camera_memory, camera_mapped,
                    components_buffer, components_memory, components_mapped,
                    volume_image, volume_memory, volume_view,
                    compute_set, graphics_set,
                    volume_initialized: false,
                });
            }

            
            // Particle subsystem
            
            // Storage buffer holding PARTICLE_COUNT vec4 (xyz pos, w age).
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

            // Compute pipeline that initializes and advances the particles.
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

            // Graphics pipeline that renders particles as instanced billboards.
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

            // Bind the particle compute and graphics descriptor sets to frame 0's
            // volume and camera UBO. The simulation evaluates the same psi(r,t)
            // for every in-flight frame so reading from frame 0 is consistent.
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
                &instance, &device, physical_device, queue, queue_family,
                &wave_vertices, vk::BufferUsageFlags::VERTEX_BUFFER,
            );
            let (wave_index_buffer, wave_index_memory) = create_device_buffer_with_data(
                &instance, &device, physical_device, queue, queue_family,
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

            VulkanRenderer {
                _entry: entry, instance, surface_loader, surface,
                physical_device, device, queue_family, queue,
                swapchain_loader, swapchain, swap_format, swap_extent,
                swap_images, swap_views,
                volume_sampler,
                descriptor_pool,
                compute_set_layout, compute_pipeline_layout, compute_pipeline,
                graphics_set_layout, graphics_pipeline_layout,
                graphics_pipeline, heatmap_pipeline,
                frames, frame_index: 0,

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

                show_volume: true,
                show_particles: false,
                show_waves: true,

                needs_particle_init: true,
                frame_counter: 0,
                last_frame_time: 0.0,
            }
        }
    }

    pub fn render_frame(&mut self, time_seconds: f32) {
        unsafe {
            let frame_idx = self.frame_index;

            // Snapshot the per-frame Vulkan handles. They are all Copy, so
            // this avoids holding a long mutable borrow on self.frames and
            // sidesteps borrow-checker conflicts with self.write_* calls.
            let cmd_buffer       = self.frames[frame_idx].cmd_buffer;
            let in_flight_fence  = self.frames[frame_idx].in_flight_fence;
            let image_available  = self.frames[frame_idx].image_available;
            let render_finished  = self.frames[frame_idx].render_finished;
            let volume_image     = self.frames[frame_idx].volume_image;
            let compute_set      = self.frames[frame_idx].compute_set;
            let graphics_set     = self.frames[frame_idx].graphics_set;
            let volume_was_init  = self.frames[frame_idx].volume_initialized;

            
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

            
            // Update host-mapped uniform buffers for this frame
            
            let atomic_time = time_seconds * self.time_scale;
            self.write_components_ubo(&self.frames[frame_idx], atomic_time);
            self.write_camera_ubo(&self.frames[frame_idx], time_seconds);

            
            // Update wave CPU state and upload to mapped storage buffer
            
            let dt = (time_seconds - self.last_frame_time).clamp(0.0, 0.05);
            self.last_frame_time = time_seconds;
            self.update_waves(dt);
            let alive_waves = self.waves.len() as u32;

            
            // Begin command buffer recording
            
            self.device.reset_command_buffer(
                cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();

            
            // Compute pass 1: evaluate psi(r, t) into this frame's volume
            
            let old_layout = if volume_was_init {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            };
            image_barrier(
                &self.device, cmd_buffer, volume_image,
                old_layout, vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ, vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            self.device.cmd_bind_pipeline(
                cmd_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer, vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_layout, 0, &[compute_set], &[]);
            let groups = (GRID_SIZE + 7) / 8;
            self.device.cmd_dispatch(cmd_buffer, groups, groups, groups);

            image_barrier(
                &self.device, cmd_buffer, volume_image,
                vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            
            // Compute pass 2: particles (init or update)
            //
            // The particle compute shader is bound to frame[0]'s volume; the
            // wavefunction kernel above wrote into frame_idx's volume. To keep
            // the particles consistent we run them only when frame_idx == 0,
            // skipping particle work on the other in-flight frame. This costs
            // half the particle update rate but avoids cross-frame hazards.
            
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

                // SSBO read-after-write barrier: compute write -> vertex read.
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

            
            // Build view and projection matrices used by overlay passes
            
            let aspect = self.swap_extent.width as f32 / self.swap_extent.height as f32;
            let proj = perspective(45.0f32.to_radians(), aspect, 0.1, 500.0);
            let cam_pos = self.current_cam_pos(time_seconds);
            let view = look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

            
            // Draw 2: particle billboards (additive blend)
            
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

            
            // Draw 3: wave shells (instanced wireframe icospheres)
            
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

            self.device.cmd_end_rendering(cmd_buffer);

            
            // Transition swapchain image to PRESENT and submit
            
            image_barrier(
                &self.device, cmd_buffer, swap_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            );

            self.device.end_command_buffer(cmd_buffer).unwrap();

            let wait_sems   = [image_available];
            let wait_stage  = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let cmd_bufs    = [cmd_buffer];
            let signal_sems = [render_finished];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stage)
                    .command_buffers(&cmd_bufs)
                    .signal_semaphores(&signal_sems)],
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

            self.frames[frame_idx].volume_initialized = true;
            self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        }
    }

    // Helper called by render_frame: advance every alive wave by dt and copy
    // the surviving subset into the host-mapped wave SSBO.
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
        // Resolve camera position from spherical (yaw, pitch, radius).
        let (yaw, pitch) = if self.auto_orbit {
            (self.cam_yaw + t * 0.20, self.cam_pitch)
        } else {
            (self.cam_yaw, self.cam_pitch)
        };
        let r = self.cam_radius;
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

        let ubo = CameraUBO {
            view_inv, proj_inv,
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 1.0],
            domain_params:  [HALF_EXTENT, self.max_density, voxel_size, view_mode_f],
            render_params:  [0.10, 18.0, 0.02, 0.65],
            heatmap_params: [slice_axis_f, self.slice_offset, color_mode_f, contour_f],
        };
        unsafe { ptr::write(frame.camera_mapped, ubo); }
    }

    // Apply a mouse delta to the orbit camera. Positive dx rotates yaw
    // clockwise (looking down +Y), positive dy lifts the pitch.
    pub fn camera_drag(&mut self, dx: f32, dy: f32) {
        let sens = 0.005;
        self.cam_yaw   -= dx * sens;
        self.cam_pitch += dy * sens;
        let limit = std::f32::consts::FRAC_PI_2 - 0.05;
        if self.cam_pitch >  limit { self.cam_pitch =  limit; }
        if self.cam_pitch < -limit { self.cam_pitch = -limit; }
    }

    // Apply a wheel delta. Positive zooms in, negative zooms out.
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

    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().unwrap(); }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
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
                self.device.destroy_fence(f.in_flight_fence, None);
                self.device.destroy_semaphore(f.image_available, None);
                self.device.destroy_semaphore(f.render_finished, None);
                self.device.destroy_command_pool(f.cmd_pool, None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.heatmap_pipeline, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.graphics_set_layout, None);
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device.destroy_pipeline_layout(self.compute_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.compute_set_layout, None);
            self.device.destroy_sampler(self.volume_sampler, None);
            for v in &self.swap_views {
                self.device.destroy_image_view(*v, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
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
        }
    }
}

// helpers 

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
) -> (vk::PhysicalDevice, u32) {
    let devices = instance.enumerate_physical_devices().unwrap();
    let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
    for pd in devices {
        let props = instance.get_physical_device_properties(pd);
        let qprops = instance.get_physical_device_queue_family_properties(pd);
        for (i, q) in qprops.iter().enumerate() {
            let pres = surface_loader
                .get_physical_device_surface_support(pd, i as u32, surface)
                .unwrap_or(false);
            if pres
                && q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU   => 100,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
                    _ => 10,
                };
                if best.map(|b| b.2).unwrap_or(-1) < score {
                    best = Some((pd, i as u32, score));
                }
            }
        }
    }
    let (pd, qf, _) = best.expect("no suitable Vulkan device");
    (pd, qf)
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

unsafe fn create_volume_3d(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice, size: u32,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R32G32_SFLOAT)   // complex psi storage
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
            .format(vk::Format::R32G32_SFLOAT)
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

// Create a device-local buffer pre-filled via a one-shot staging copy.
unsafe fn create_device_buffer_with_data<T: Copy>(
    instance: &Instance, device: &Device, pd: vk::PhysicalDevice,
    queue: vk::Queue, queue_family: u32,
    data: &[T], usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let bytes = (data.len() * size_of::<T>()) as u64;

    // Staging
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

    // Device-local
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

    // One-shot copy via a transient command buffer.
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

// Returns vertices (vec3 each) and triangle indices for an icosphere of
// the given subdivision level. Vertices lie on the unit sphere.
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
    // Additive blending for glow.
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
    // Vertex input: vec3 position.
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
        .polygon_mode(vk::PolygonMode::LINE)        // wireframe shells
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

// minimal linear algebra 
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
