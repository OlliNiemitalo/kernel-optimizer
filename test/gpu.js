// Open /test/gpu.html on the Vite server. Tests read linear GPU buffers, not tone-mapped screenshots.
import { BlurRenderer } from '../src/blur.js';
import { presetParams } from '../src/problem.js';
import { Shapes } from '../src/shapes.js';

async function read(device, buffer) {
  const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

export async function run() {
  const renderer = await BlurRenderer.create(document.createElement('canvas'));
  const device = renderer.device;
  const errors = [];
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  const tests = [];
  const check = (name, value, limit) => tests.push({ name, value, limit, pass: Number.isFinite(value) && value <= limit });
  // Odd, non-tile-aligned dimensions also exercise partial byte-map words and workgroups.
  const width = 97, height = 97;
  let transposed = false;
  const index = (x, y) => transposed ? x * width + y : y * width + x;
  const rgba = new Float32Array(width * height * 4);
  const radii = new Float32Array(width * height);
  const pixel = (data, x, y, lane) => data[4 * index(x, y) + lane] / data[4 * index(x, y) + 3];
  // Equal |CoC| gives equal support on either side of focus. A much farther light should
  // be hidden beyond that support for BOTH signs, rather than appearing at negative CoC.
  async function edge(nearRadius, { occlusion = 1, layers = 1, sizeBlur = null, precision = 'f32', farRadius = 12 } = {}) {
    rgba.fill(0);
    radii.fill(farRadius);
    for (let y = 16; y < 48; y++) for (let x = 32; x < 64; x++) {
      rgba[4 * index(x, y)] = 1;
      radii[index(x, y)] = nearRadius;
    }
    rgba[4 * index(28, 32) + 2] = 50;
    // Reuse allocations to catch stale per-frame occupancy/reach state after populated layers.
    if (renderer.width !== width) renderer.setImage(rgba, width, height);
    else device.queue.writeBuffer(renderer.source, 0, rgba);
    renderer.setRadiusMap(radii);
    // Exercise the depth path with an exact synthetic CoC field, without a scene shader.
    renderer.varying.kind = 'depth';
    renderer.setKernel(presetParams(3), 12, 0.2);
    await renderer.render({ occlusion, layers, sizeBlur, precision });
    if (layers > 1) {
      const tiles = await read(device, renderer.tiles);
      const occupied = new Uint32Array(tiles.buffer).at(-1) !== 0;
      check(`near layer occupancy for CoC ${nearRadius}/${farRadius}`, Number(occupied !== (nearRadius === -12)), 0);
    }
    const data = await read(device, layers > 1 ? renderer.layerBuffer : renderer.acc);
    if (layers > 1) for (let i = 3; i < width * height * 4; i += 4) data[i] = 1;
    return data;
  }
  try {
    for (const transpose of [false, true]) for (const precision of ['f32', 'f16']) for (const sizeBlur of [null, 1]) {
      transposed = transpose;
      const label = `${precision}, size blur ${sizeBlur}, transpose ${transpose}`;
      const near = await edge(-1, { precision, sizeBlur });
      const far = await edge(1, { precision, sizeBlur });
      const off = await edge(-1, { precision, sizeBlur, occlusion: 0 });
      check(`foreground blocks far light (${label})`, Math.abs(pixel(near, 36, 32, 2)), 1e-5);
      check(`background blocks farther light (${label})`, Math.abs(pixel(far, 36, 32, 2)), 1e-5);
      check(`test light reaches surface with occlusion off (${label})`, pixel(off, 36, 32, 2) > 0.01 ? 0 : 1, 0);
      const partial = await edge(-4, { precision, sizeBlur });
      check(`far light survives within destination support (${label})`, pixel(partial, 33, 32, 2) > 0.01 ? 0 : 1, 0);
      const layered = await edge(-1, { precision, sizeBlur, layers: 2 });
      let difference = 0;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) {
        difference = Math.max(difference, Math.abs(pixel(layered, x, y, c) - pixel(near, x, y, c)));
      }
      check(`an empty near layer is an identity (${label})`, difference, 1e-5);
      for (const layers of [1, 2]) {
        const spread = await edge(-12, { precision, sizeBlur, layers, farRadius: 0 });
        check(`foreground spreads over sharp background (${label}, ${layers} layers)`, pixel(spread, 28, 30, 0) > 0.05 ? 0 : 1, 0);
        if (layers === 2) {
          const noOcclusion = await edge(-12, { precision, sizeBlur, layers, farRadius: 0, occlusion: 0 });
          let difference = 0;
          for (let i = 0; i < width * height * 4; i++) difference = Math.max(difference, Math.abs(spread[i] - noOcclusion[i]));
          check(`separate constant-depth layers use compositing for visibility (${label})`, difference, 1e-5);
        }
      }
    }
    // Actual tetra geometry: with no near-layer members, enabling layers must not change
    // any linear RGB pixel, including antialiased silhouettes and bright stars.
    const shapes = new Shapes(device);
    shapes.yaw = 1.2;
    shapes.pitch = 0.1;
    renderer.setAnimated(shapes, 320, 200);
    renderer.setDepthOfField();
    renderer.setKernel(presetParams(3), 30, 0.2);
    for (const scale of [1, 1.5]) {
      shapes.scale = scale;
      await renderer.render({ invFocus: 0.5, layers: 1 });
      const single = await read(device, renderer.acc);
      await renderer.render({ invFocus: 0.5, layers: 2 });
      const layers = await read(device, renderer.layerBuffer);
      const tiles = await read(device, renderer.tiles);
      check(`tetra test has no near layer (scale ${scale})`, new Uint32Array(tiles.buffer).at(-1), 0);
      let difference = 0;
      for (let i = 0; i < single.length; i += 4) for (let c = 0; c < 3; c++) {
        difference = Math.max(difference, Math.abs(single[i + c] / single[i + 3] - layers[i + c]));
      }
      check(`tetra empty-layer identity (scale ${scale})`, difference, 1e-5);
    }
    check('WebGPU validation errors', errors.length, 0);
    return { passed: tests.every(t => t.pass), tests, errors };
  } finally {
    device.destroy();
  }
}