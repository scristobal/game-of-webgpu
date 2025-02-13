// Adapted from https://codelabs.developers.google.com/your-first-webgpu-app
import cellShader from './cell.wgsl?raw'
import simulationShader from './simulation.wgsl?raw'

async function gameOfWebGPU() {
    // initialization and checks
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw 'WebGPU not supported.';

    // use adapter to get device
    const device = await adapter.requestDevice();

    // attach device to html canvas
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    const canvas = document.querySelector('canvas');
    if (!canvas) throw new Error('No canvas found, be sure to include a <canvas> element in your HTML');

    const scale = 1;

    canvas.width = window.innerWidth - (window.innerWidth % scale);
    canvas.height = window.innerHeight - (window.innerHeight % scale);

    const context = canvas.getContext('webgpu');
    if (!context) throw 'WebGPU not supported.';

    context.configure({
        device: device,
        format: canvasFormat
    });

    // Data preparation, grid size on both axis
    const GRID_SIZE_X = canvas.width / scale;
    const GRID_SIZE_Y = canvas.height / scale;

    // this represents the size of the board, since
    // it is constant for each iteration it should be a uniform
    const gridSizeArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]);

    const gridSizeBuffer: GPUBuffer = device.createBuffer({
        label: 'Grid Uniforms',
        size: gridSizeArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(gridSizeBuffer, 0, gridSizeArray);

    // This represent the cell state using two buffers
    // in each iteration one buffer will be used for
    // drawing and the other for computing the next state
    const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

    const cellStateStorage: [GPUBuffer, GPUBuffer] = [
        device.createBuffer({
            label: 'Cell State A',
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }),
        device.createBuffer({
            label: 'Cell State B',
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        })
    ];

    // initialization, no need to initialize cellStateStorage[1] as it will overwritten on first iteration
    for (let i = 0; i < cellStateArray.length; i++) {
        cellStateArray[i] = Math.random() > 0.5 ? 1 : 0;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    // each row contains a vertex data in the form (x, y, r, g, b, a), eg. position, color
    const vertices = new Float32Array([/* v_1 */ 1, 1, 0, 0, 0, 1, /* v_2 */ 1, -1, 0, 0, 0, 1, /* v_3 */ -1, -1, 0, 0, 0, 1, /* v_4 */ -1, 1, 0, 0, 0, 1]);

    // copy data into the GPU
    const vertexBuffer: GPUBuffer = device.createBuffer({
        label: 'Vertices',
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, vertices);

    // tells the GPU how is the data organized
    const vertexBufferLayout: GPUVertexBufferLayout = {
        arrayStride: 2 * 4 + 4 * 4, // <- this needs to match the sum of sizes of each attribute's format
        attributes: [
            {
                shaderLocation: 0, // Position, used inside `Cell shader` as `@location(0) position: vec2<f32>`
                format: 'float32x2' as const, // size is 2*4 bytes
                offset: 0
            },
            {
                shaderLocation: 1, // Color, used inside `Cell shader` as `@location(1) color: vec4<f32>`
                format: 'float32x4' as const, // size is 4*4 bytes
                offset: 2 * 4 // <- this should mach the size of the previous attribute(s)
            }
        ]
    };

    // a square is composed of 2 triangles arranged as triplets of points grouped in two triangles
    const indexes = new Uint32Array([2, 1, 0, 2, 0, 3]);

    const indexBuffer: GPUBuffer = device.createBuffer({
        label: 'Cell Vertex indexes',
        size: indexes.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(indexBuffer, 0, indexes);

    const indexFormat: GPUIndexFormat = 'uint32';

    // Cell drawing shaders this shaders are used to render the board
    const cellRenderShaderModule = device.createShaderModule({
        label: 'Cell shader',
        code: cellShader
    });

    // this shader is used to evolve the board state
    const cellSimulationShaderModule = device.createShaderModule({
        label: 'Game of Life simulation shader',
        code: simulationShader
    });

    // Glueing all together in a pipeline, first define the memory layout
    //
    // creates a bind group for our uniforms, binds will reflect in the `@bindings` inside a `@group`
    // because GPUBindGroupLayout is defined without attaching to a BindGroup yet
    const gridBindGroupLayout: GPUBindGroupLayout = device.createBindGroupLayout({
        label: 'Grid Bind Group Layout',
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: {}
            }
        ]
    });

    const cellBindGroupLayout: GPUBindGroupLayout = device.createBindGroupLayout({
        label: 'Cell Bind Group Layout',
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' } // Cell state input buffer
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' } // Cell state output buffer
            }
        ]
    });

    // this actually attaches a GPUBindGroupLayout to a GPUBindGroup creating a `@group` with
    // the `@binding` layout as defined in the previous step s
    const gridBindGroup: GPUBindGroup = device.createBindGroup({
        label: 'Grid Bind Group',
        layout: gridBindGroupLayout,
        entries: [
            {
                binding: 0, // <- this will be the binding for whatever group is assigned later, eg. `@group(0) @binding(0) var<uniform> grid_size: vec2<f32>;`
                resource: { buffer: gridSizeBuffer }
            }
        ]
    });

    let cellComputeBindGroup: GPUBindGroup = device.createBindGroup({
        label: 'Cell renderer bind group A',
        layout: cellBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: cellStateStorage[0] }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[1] }
            }
        ]
    });

    let cellRenderBindGroup: GPUBindGroup = device.createBindGroup({
        label: 'Cell renderer bind group B',
        layout: cellBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: cellStateStorage[1] }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[0] }
            }
        ]
    });

    // combine the `GPUBindGroups` into a `GPUPipelineLayout`
    const pipelineLayout: GPUPipelineLayout = device.createPipelineLayout({
        label: 'Cell Pipeline Layout',
        bindGroupLayouts: [gridBindGroupLayout, cellBindGroupLayout] // <- group 0 is grid, group 1 is cells, eg. ` @group(1) @binding(0) var<storage> cell_state: array<u32>;`
    });

    // use the sale pipeline layout for both pipelines
    const cellRenderPipeline: GPURenderPipeline = device.createRenderPipeline({
        label: 'Cell pipeline',
        layout: pipelineLayout,
        primitive: {
            topology: 'triangle-list'
        },
        vertex: {
            module: cellRenderShaderModule,
            entryPoint: 'vertex_main',
            buffers: [vertexBufferLayout]
        },
        fragment: {
            module: cellRenderShaderModule,
            entryPoint: 'fragment_main',
            targets: [
                {
                    format: canvasFormat
                }
            ]
        }
    });

    const cellSimulationPipeline: GPUComputePipeline = device.createComputePipeline({
        label: 'Simulation pipeline',
        layout: pipelineLayout,
        compute: {
            module: cellSimulationShaderModule,
            entryPoint: 'compute_main'
        }
    });

    // Render loop
    return function main() {
        // a new encoder is required every update
        const encoder = device.createCommandEncoder();

        // set up the simulation pass
        const computePass = encoder.beginComputePass();

        computePass.setPipeline(cellSimulationPipeline);

        computePass.setBindGroup(0, gridBindGroup);
        computePass.setBindGroup(1, cellComputeBindGroup);

        computePass.dispatchWorkgroups(128, 128); // <- equivalent of draw for render passes

        computePass.end();

        // set up the rendering pass requires a new view on the current texture
        const view = context.getCurrentTexture().createView();

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view,
                    loadOp: 'clear', // defaults to black
                    clearValue: { r: 1, g: 1, b: 1, a: 1 },
                    storeOp: 'store'
                }
            ]
        });

        renderPass.setPipeline(cellRenderPipeline);

        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, indexFormat);

        renderPass.setBindGroup(0, gridBindGroup);
        renderPass.setBindGroup(1, cellRenderBindGroup);

        renderPass.drawIndexed(indexes.length, GRID_SIZE_X * GRID_SIZE_Y, 0, 0, 0);
        // renderPass.draw(vertices.length / (2 + 4), GRID_SIZE_X * GRID_SIZE_Y);

        renderPass.end();

        // finish and submit
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        // each update `cellComputeBindGroup` and `cellRenderBindGroup` must be swapped,
        // the former is used to compute the new state while the latter is used to render
        [cellComputeBindGroup, cellRenderBindGroup] = [cellRenderBindGroup, cellComputeBindGroup];

        // Schedule next frame
        requestAnimationFrame(main);
    };
}

const version = import.meta.env.VITE_APP_VERSION;
console.log(`Using version ${version}`);

gameOfWebGPU()
    .then(requestAnimationFrame)
    .catch(errorHelper)
    .finally(() => console.log('done', new Date()));

function errorHelper(errorMessage: string) {
    const notSupportedError = new CustomEvent('initerror', {
        detail: errorMessage
    });

    document.dispatchEvent(notSupportedError);
}

// follow up https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
