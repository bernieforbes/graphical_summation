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

    var color = vec3<f32>(0.0);
    var image = textureSample(t_diffuse, s_diffuse, vec2<f32>(in.tex_coords.x, 1.0 - in.tex_coords.y));
    color = image.rgb;
    
    return vec4<f32>(color, 1.0);
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
@group(0) @binding(3) var f32_write_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var f32_read_texture: texture_2d<f32>;

fn add_to_texel(coord: vec2<i32>, amount: f32) {

    let loaded_texel = textureLoad(diffuse_texture, coord, 0);
    var value:f32 = loaded_texel.r + amount;

    textureStore(storage_texture, coord, vec4(vec3(value), 1.0));
}

fn equalize_with_neighbours(transfer_rate: f32, id: vec3<u32>) {

    let coord = vec2(i32(id.x), i32(id.y));

    var this_texel = textureLoad(diffuse_texture, coord, 0);
    var this_val = this_texel.r;

    var neighbours = array(
        vec2(i32(id.x - 1),   i32(id.y - 1)),
        vec2(i32(id.x),       i32(id.y - 1)),
        vec2(i32(id.x + 1),   i32(id.y - 1)),
        vec2(i32(id.x + 1),   i32(id.y)),
        vec2(i32(id.x + 1),   i32(id.y + 1)),
        vec2(i32(id.x),       i32(id.y + 1)),
        vec2(i32(id.x - 1),   i32(id.y + 1)),
        vec2(i32(id.x - 1),   i32(id.y))
    );

    let num_neighbors: i32 = 8;

    var total_transfer:f32 = 0.0;
    var i: i32 = 0;
    loop {
        if (i == num_neighbors) {
            break;
        }

        var neighbour = neighbours[i];
        var neighbour_val:f32 = textureLoad(diffuse_texture, neighbour, 0).r;
        let diff:f32 = neighbour_val - this_val;

        var transfer:f32 = diff * transfer_rate;
        total_transfer += transfer;

        i += 1;
    }

    add_to_texel(coord, total_transfer);
}


fn apply_gravity(velocity: vec2<f32>)-> vec2<f32> {
    let delta_time:f32 = compute_uniforms.delta_time;
    let time_step:f32 = compute_uniforms.time_step;
    let gravity:f32 = -9.81;

    let gravity_vec = vec2<f32>(0.0, gravity * (delta_time / time_step));

    return velocity + gravity_vec;
}


@compute
@workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {

    let coord = vec2(i32(id.x), i32(id.y));
    var this_texel = textureLoad(diffuse_texture, coord, 0);

    var f32_read_texel = textureLoad(f32_read_texture, coord, 0);
    var velocity = this_texel.rg;
    var pressure = this_texel.b;

    if (pressure > 0.0) {
        velocity = apply_gravity(velocity);
    }

    // let color = vec4(velocity, pressure, 1.0);

    // textureStore(storage_texture, coord, color);

    // if (velocity.y < 0.0) {
    //     textureStore(storage_texture, coord, vec4(1.0));
    // }

    if (f32_read_texel.r > 0.0) {
        // let color = vec4(vec3(this_texel.r), 1.0);
        // textureStore(storage_texture, coord, color);
        textureStore(storage_texture, coord, (vec4(1.0)));
    }

    if (id.x == 5 && id.y == 5) {
        textureStore(f32_write_texture, coord, (vec4(1.0)));
    }
}
