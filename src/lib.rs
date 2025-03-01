use std::{iter, time::SystemTime};

use wgpu::util::DeviceExt;
use winit::{
    dpi::{LogicalPosition, PhysicalSize},
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

mod compute;
mod storage_texture;
mod texture;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    elapsed_seconds: f32,
}

// Implement the desc() function for the vertex struct
// This returns a descriptor of the buffer layout
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // Set the array stride to the size of a vertex array
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            // step forward every vertex
            step_mode: wgpu::VertexStepMode::Vertex,
            // Set up an array of 2 vertex attributes
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    // Starts at an offset the size of the first attribute
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, 1.0, 0.0],
        tex_coords: [0.0, 1.0],
    }, // A
    Vertex {
        position: [-1.0, -1.0, 0.0],
        tex_coords: [0.0, 0.0],
    }, // B
    Vertex {
        position: [1.0, -1.0, 0.0],
        tex_coords: [1.0, 0.0],
    }, // C
    Vertex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 1.0],
    }, // D
];

// The index array.  Every 3 indices makes a triangle
const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0, /* padding */ 0];

// A struct to store basically everything to do with our program
struct Context<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    window: &'a Window,
    start_time: SystemTime,
    uniforms: Uniforms,
    uniforms_buffer: wgpu::Buffer,
    uniforms_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    storage_texture: storage_texture::StorageTexture,
    f32_write_texture: storage_texture::StorageTexture,
    f32_read_texture: storage_texture::StorageTexture,
    compute: compute::Compute,
}

// Implement the functions as members of State, so everything can just be referenced via "self"
// New() creates everything, puts it in a "Self" (State), then returns it.  Basically a factory pattern.
// I guess this makes it pretty tidy
impl<'a> Context<'a> {
    //    _   _
    //   | \ | |
    //   |  \| | _____      __
    //   | . ` |/ _ \ \ /\ / /
    //   | |\  |  __/\ V  V /
    //   |_| \_|\___| \_/\_/
    //
    //

    async fn new(window: &'a Window) -> Context<'a> {
        let window_size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = instance.create_surface(window).unwrap();

        // Request an adapter from the wgpu instance
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Request a device from the adapter
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Create the shader module from text file
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        //    _    _       _  __
        //   | |  | |     (_)/ _|
        //   | |  | |_ __  _| |_ ___  _ __ _ __ ___  ___
        //   | |  | | '_ \| |  _/ _ \| '__| '_ ` _ \/ __|
        //   | |__| | | | | | || (_) | |  | | | | | \__ \
        //    \____/|_| |_|_|_| \___/|_|  |_| |_| |_|___/
        //

        let start_time = SystemTime::now();
        let elapsed_seconds = 0.0;
        let uniforms = Uniforms { elapsed_seconds };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniforms_bind_group_layout"),
            });

        let uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
            label: Some("uniforms_bind_group"),
        });

        ///////////////////////////////////// UNIFORMS END /////////////////////////////////////

        //    _______        _
        //   |__   __|      | |
        //      | | _____  _| |_ _   _ _ __ ___  ___
        //      | |/ _ \ \/ / __| | | | '__/ _ \/ __|
        //      | |  __/>  <| |_| |_| | | |  __/\__ \
        //      |_|\___/_/\_\\__|\__,_|_|  \___||___/
        //
        //

        surface.configure(&device, &config);
        // let diffuse_bytes = include_bytes!("../assets/happy-tree.png");
        // let diffuse_bytes = include_bytes!("../assets/single_white_pixel.png");
        // let diffuse_bytes = include_bytes!("../assets/single_white_pixel_small.png");
        let diffuse_bytes = include_bytes!("../assets/single_white_pixel_very_small.png");
        // let diffuse_bytes = include_bytes!("../assets/white_circle.png");
        // let diffuse_bytes = include_bytes!("../assets/density_start_state_128_128.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "diffuse texture")
                .unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        ///////////////////////////////////// TEXTURES END /////////////////////////////////////

        //  _____ _            _ _
        // |  __ (_)          | (_)
        // | |__) | _ __   ___| |_ _ __   ___
        // |  ___/ | '_ \ / _ \ | | '_ \ / _ \
        // | |   | | |_) |  __/ | | | | |  __/
        // |_|   |_| .__/ \___|_|_|_| |_|\___|
        //         | |
        //         |_|

        // Establish the render pipeline layout with standard settings
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniforms_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the actual render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            // Use the layout created above
            layout: Some(&render_pipeline_layout),

            // Vertex state
            // Describe how to deal with vertices
            vertex: wgpu::VertexState {
                // use the shader module created above
                module: &shader_module,
                // Tell it to use the "vs_main" entry point (function)
                entry_point: Some("vs_main"),
                // use the vertex buffer layout returned by the desc() function, wrapped in an array ref
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },

            // Fragment state
            // Describe how to deal with fragments
            fragment: Some(wgpu::FragmentState {
                // use the shader module created above
                module: &shader_module,
                // Tell it to use the "fs_main" entry point (function)
                entry_point: Some("fs_main"),
                // Define the targets (only one in our case)
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    // How to do blending
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),

            // Primitive state (shapes)
            // Describe how to do primitive assembly and rasterization
            primitive: wgpu::PrimitiveState {
                // A list of vertices, every 3 vertices makes a triangle
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Front faces are any faces wound counter-clockwise
                front_face: wgpu::FrontFace::Ccw,
                // Don't show back faces
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },

            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });

        ///////////////////////////////////// PIPELINE END /////////////////////////////////////

        //     _____                            _
        //    / ____|                          | |
        //   | |     ___  _ __ ___  _ __  _   _| |_ ___
        //   | |    / _ \| '_ ` _ \| '_ \| | | | __/ _ \
        //   | |___| (_) | | | | | | |_) | |_| | ||  __/
        //    \_____\___/|_| |_| |_| .__/ \__,_|\__\___|
        //                         | |
        //                         |_|

        let storage_texture = storage_texture::StorageTexture::from_texture(
            &diffuse_texture.texture,
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )
        .unwrap();

        let f32_write_texture = storage_texture::StorageTexture::from_texture(
            &diffuse_texture.texture,
            &device,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )
        .unwrap();

        let f32_read_texture = storage_texture::StorageTexture::from_texture(
            &diffuse_texture.texture,
            &device,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
        .unwrap();

        let compute = compute::Compute::new(
            &storage_texture,
            &f32_write_texture,
            &f32_read_texture,
            &diffuse_texture,
            &device,
            shader_module,
        )
        .unwrap();

        ///////////////////////////////////// COMPUTE END //////////////////////////////////////

        // The vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // The index buffer
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        // Return the struct
        Self {
            surface,
            device,
            queue,
            config,
            window_size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            window,
            start_time,
            uniforms,
            uniforms_buffer,
            uniforms_bind_group,
            diffuse_bind_group,
            diffuse_texture,
            storage_texture,
            f32_write_texture,
            f32_read_texture,
            compute,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        let time_now = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap()
            .as_secs_f32();

        self.uniforms.elapsed_seconds = time_now;
        self.compute.uniforms.delta_time = time_now - self.compute.uniforms.elapsed_seconds;
        self.compute.uniforms.elapsed_seconds = time_now;

        self.queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        self.queue.write_buffer(
            &self.compute.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.compute.uniforms]),
        );

        // WIP_ONLY
        if self.uniforms.elapsed_seconds % 1.0 > 0.02 {
            // return;
        }

        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute_pass.set_bind_group(0, &self.compute.bind_group, &[]);
            compute_pass.set_pipeline(&self.compute.pipeline);
            compute_pass.dispatch_workgroups(
                self.storage_texture.dimensions.width as u32,
                self.storage_texture.dimensions.height as u32,
                1,
            );
        }

        command_encoder.copy_texture_to_texture(
            self.storage_texture.texture.as_image_copy(),
            self.diffuse_texture.texture.as_image_copy(),
            self.diffuse_texture.texture.size(),
        );

        command_encoder.copy_texture_to_texture(
            self.f32_write_texture.texture.as_image_copy(),
            self.f32_read_texture.texture.as_image_copy(),
            self.f32_read_texture.texture.size(),
        );

        self.queue.submit(Some(command_encoder.finish()));

        self.compute.uniforms.texel_group += 1;
        if self.compute.uniforms.texel_group == 4 {
            self.compute.uniforms.texel_group = 0;
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            // Create the render pass
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            // 2 args: buffer slot number to use, and a slice from the buffer
            render_pass.set_bind_group(0, &self.uniforms_bind_group, &[]);
            render_pass.set_bind_group(1, &self.diffuse_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// The "main" function
pub async fn run() {
    // Start the logger
    env_logger::init();

    // Create a new event loop
    let event_loop = EventLoop::new().unwrap();

    // Create a new window, using the above event loop
    let window = WindowBuilder::new()
        .with_title("Graphical Summation")
        .with_position(LogicalPosition::new(1300, 800))
        .with_inner_size(PhysicalSize {
            width: 1024,
            height: 1024,
        })
        .build(&event_loop)
        .unwrap();

    // Create a new state struct with the window we just created
    // State::new uses async code, so we're going to wait for it to finish
    let mut state = Context::new(&window).await;
    let mut surface_configured = false;

    // Run a closure using the event loop
    event_loop
        .run(move |event, control_flow| {
            match event {
                // Window stuff (close, resize etc)
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => control_flow.exit(),
                            WindowEvent::Resized(physical_size) => {
                                surface_configured = true;
                                state.resize(*physical_size);
                            }

                            // What to do on redraw
                            WindowEvent::RedrawRequested => {
                                // This tells winit that we want another frame after this one
                                state.window().request_redraw();

                                if !surface_configured {
                                    return;
                                }

                                state.update();
                                match state.render() {
                                    Ok(_) => {}
                                    // Reconfigure the surface if it's lost or outdated
                                    Err(
                                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                                    ) => state.resize(state.window_size),
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => {
                                        log::error!("OutOfMemory");
                                        control_flow.exit();
                                    }

                                    // This happens when the a frame takes too long to present
                                    Err(wgpu::SurfaceError::Timeout) => {
                                        log::warn!("Surface timeout")
                                    }

                                    Err(wgpu::SurfaceError::Other) => {
                                        log::warn!("Surface timeout")
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
