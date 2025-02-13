@group(0) @binding(0) var<uniform> grid_size: vec2<f32>;

@group(1) @binding(0) var<storage> cell_state_in: array<u32>;
@group(1) @binding(1) var<storage, read_write> cell_state_out: array<u32>;

fn cell_index(cell: vec2u) -> u32 {
    return (cell.y % u32(grid_size.y)) * u32(grid_size.x) + (cell.x % u32(grid_size.x));
}

fn cell_active(x: u32, y: u32) -> u32 {
    return cell_state_in[cell_index(vec2(x, y))];
}

fn active_neighbors(cell: vec3u) -> u32 {
    return cell_active(cell.x+1, cell.y+1) +
        cell_active(cell.x+1, cell.y) +
        cell_active(cell.x+1, cell.y-1) +
        cell_active(cell.x, cell.y-1) +
        cell_active(cell.x-1, cell.y-1) +
        cell_active(cell.x-1, cell.y) +
        cell_active(cell.x-1, cell.y+1) +
        cell_active(cell.x, cell.y+1);
}

@compute
// in this case arbitrary, in general same workgroup can share memory and synchronize
// rule of thumb is size of 64, in this case 8*8
@workgroup_size(16,16)
fn compute_main(@builtin(global_invocation_id) cell: vec3u) {

    let num_active = active_neighbors(cell);

    let i = cell_index(cell.xy);

    // Conway's game of life rules:
    switch num_active {
        case 2u: { // Active cells with 2 neighbors stay the same.
            cell_state_out[i] = cell_state_in[i];
        }
        case 3u: { // Cells with 3 neighbors become or stay active.
            cell_state_out[i] = 1u;
        }
        default: { // Cells with < 2 or > 3 neighbors become inactive.
            cell_state_out[i] = 0u;
        }
    }
}

