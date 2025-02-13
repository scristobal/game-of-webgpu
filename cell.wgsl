@group(0) @binding(0) var<uniform> grid_size: vec2<f32>;
@group(1) @binding(0) var<storage> cell_state: array<u32>;

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32
}

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) color : vec4<f32>
}

@vertex
fn vertex_main(input: VertexIn) -> VertexOut
{
    var output : VertexOut;

    let state = f32(cell_state[input.instance]);

    let i = f32(input.instance);
    let cell = vec2<f32>( i % grid_size.x, floor(i / grid_size.x));

    let cell_offset = cell / ( grid_size) * 2 ;
    let grid_position = (input.position*state + 1) / grid_size - 1 + cell_offset;

    output.position = vec4<f32>(grid_position, 0, 1);
    output.color = input.color;
    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    return fragData.color;
}

