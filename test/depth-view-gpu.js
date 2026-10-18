import { shaderSource } from '../src/blur.js';

// Exercise the exported production display shader with known inverse depths.
export async function depthViewGPU(device, check) {
  const module = device.createShaderModule({ code: shaderSource('display', false) });
  const pipeline = device.createRenderPipeline({ layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] } });
  const depths = [0, 0.25, 0.5, 1, 2, 10];
  const source = new Float32Array(depths.flatMap(d => [0, 0, 0, d]));
  const make = (size, usage) => device.createBuffer({ size, usage });
  const data = make(source.byteLength * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const uniform = make(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readback = make(256, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const texture = device.createTexture({ size: [depths.length, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  device.queue.writeBuffer(data, 0, source);
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniform } },
    ...[1, 2, 3].map(binding => ({ binding, resource: { buffer: data } })),
  ] });
  try {
    for (const layered of [0, 1]) for (const showSource of [0, 1]) for (const exposure of [0.25, 4]) {
      const params = new ArrayBuffer(32);
      new Uint32Array(params, 0, 2).set([depths.length, 1]);
      new Float32Array(params, 8, 1)[0] = exposure;
      new Uint32Array(params, 12, 4).set([1, showSource, layered, 6]);
      device.queue.writeBuffer(uniform, 0, params);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
      encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: 256 }, [depths.length, 1]);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(readback.getMappedRange());
      let error = 0;
      depths.forEach((d, i) => {
        const expected = showSource ? 0 : Math.round(255 * d / (1 + d));
        for (let c = 0; c < 3; c++) error = Math.max(error, Math.abs(bytes[4 * i + c] - expected));
        error = Math.max(error, Math.abs(bytes[4 * i + 3] - 255));
      });
      readback.unmap();
      check(`depth map display: layers ${layered}, source ${showSource}, exposure ${exposure}`, error, 1);
    }
  } finally {
    for (const b of [data, uniform, readback]) b.destroy();
    texture.destroy();
  }
}