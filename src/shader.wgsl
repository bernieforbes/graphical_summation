// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

struct Uniforms {
    elapsed_seconds: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var color = vec4<f32>(0.0);
    var image = textureSample(t_diffuse, s_diffuse, vec2(in.tex_coords.x, 1.0 - in.tex_coords.y));
    color = image.rgba;
    
    return color;
}

struct ComputeUniforms {
    texel_group: u32,
    elapsed_seconds: f32,
    time_step: f32,
    delta_time: f32,
};

@group(0) @binding(0) var storage_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var diffuse_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> compute_uniforms: ComputeUniforms;
@group(0) @binding(3) var f32_write_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var f16_read_texture: texture_2d<f32>;

@compute
@workgroup_size(16,16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {

    let coord = id.xy;
    var this_texel = textureLoad(diffuse_texture, coord, 0);

    textureStore(storage_texture, coord, this_texel);
}
