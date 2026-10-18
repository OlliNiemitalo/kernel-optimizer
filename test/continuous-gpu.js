import { radiusTables } from '../src/radius-tables.js';
import { presetParams, separableChannels } from '../src/problem.js';

export async function continuousGPU(renderer, check, read) {
  const width = 49, height = 49, center = 24;
  const device = renderer.device;
  const params = presetParams(3), channels = separableChannels(params);
  const reference = radiusTables(channels, 0.2, 12);
  const source = new Float32Array(width * height * 4);
  source[4 * (center * width + center)] = 1;
  const radii = new Float32Array(width * height).fill(12);
  renderer.setImage(source, width, height);
  renderer.setRadiusMap(radii);
  renderer.setKernel(params, 12, 0.2);
  const images = new Map();
  for (const precision of ['f32', 'f16']) for (const sizeBlur of [null, 1]) {
    for (const r of [0, 0.2, 0.9438, 0.9439, 1, 2.123, 3.1, 3.11, 4, 8.05]) {
      radii.fill(r);
      device.queue.writeBuffer(renderer.radiusBuffer, 0, radii);
      await renderer.render({ precision, sizeBlur, occlusion: 0 });
      const data = await read(device, renderer.acc);
      const effective = radii[0]; // actual f32 radius uploaded to the GPU
      const [node] = reference.coordinate(effective);
      const half = reference.halves[node + 1];
      const taps = [0, 1].map(dir => channels.map((_, ch) =>
        Array.from({ length: 2 * half + 1 }, (_, i) => reference.tap(effective, dir, ch, i - half))));
      const sums = taps.map(dir => dir.map(t => t.reduce((s, z) => [s[0] + z[0], s[1] + z[1]], [0, 0])));
      const dc = sums[0].reduce((s, h, ch) => s + h[0] * sums[1][ch][0] - h[1] * sums[1][ch][1], 0);
      let error = 0, coverageError = 0;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const dx = x - center, dy = y - center;
        let expected = 0;
        if (Math.abs(dx) <= half && Math.abs(dy) <= half) {
          for (let ch = 0; ch < channels.length; ch++) {
            const h = taps[0][ch][dx + half], v = taps[1][ch][dy + half];
            expected += h[0] * v[0] - h[1] * v[1];
          }
        }
        const i = 4 * (y * width + x);
        error = Math.max(error, Math.abs(data[i] / data[i + 3] - expected / dc));
        coverageError = Math.max(coverageError, Math.abs(data[i + 3] - 1));
      }
      const label = `r=${r}, ${precision}, size blur ${sizeBlur}`;
      check(`continuous impulse agrees with CPU interpolant (${label})`, error, precision === 'f32' ? 2e-5 : 0.003);
      check(`continuous unit coverage (${label})`, coverageError, precision === 'f32' ? 0.005 : 0.008);
      if (precision === 'f32' && sizeBlur === null) images.set(r, data);
      // The vertical averaging path must retain fractional radii too, not just interpolate
      // horizontally while silently rounding back to the old ladder before the vertical pass.
      const maps = await read(device, renderer.levelMaps);
      const offset = 5 * height * Math.ceil(width / 4) + 2 * width * height;
      let radiusError = 0;
      for (let i = 0; i < width * height; i++) radiusError = Math.max(radiusError, Math.abs(maps[offset + i] - effective));
      check(`fractional vertical radius retained (${label})`, radiusError, 1e-5);
    }
  }
  const difference = (a, b) => {
    const x = images.get(a), y = images.get(b);
    let e = 0;
    for (let i = 0; i < x.length; i += 4) e = Math.max(e, Math.abs(x[i] / x[i + 3] - y[i] / y[i + 3]));
    return e;
  };
  check('no jump at the former identity/blur cutoff', difference(0.9438, 0.9439), 0.0002);
  check('different radii within one old quantization bin produce different images', difference(3.1, 3.11) > 1e-5 ? 0 : 1, 0);

  // A column-wise ramp forces the horizontal source radius to differ from the vertical radius
  // at each destination. For an isolated impulse, the unnormalized RGB has an explicit reference.
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) radii[y * width + x] = 0.2 + 0.16 * x;
  device.queue.writeBuffer(renderer.radiusBuffer, 0, radii);
  for (const precision of ['f32', 'f16']) {
    await renderer.render({ precision, sizeBlur: null, occlusion: 0 });
    const data = await read(device, renderer.acc);
    const hr = radii[center * width + center];
    let error = 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const vr = radii[y * width + x];
      let expected = 0;
      for (let ch = 0; ch < channels.length; ch++) {
        const h = reference.tap(hr, 0, ch, x - center), v = reference.tap(vr, 1, ch, y - center);
        expected += h[0] * v[0] - h[1] * v[1];
      }
      expected *= reference.correction(hr, vr, true);
      error = Math.max(error, Math.abs(data[4 * (y * width + x)] - expected));
    }
    check(`mixed-radius ramp agrees with CPU (${precision})`, error, precision === 'f32' ? 2e-5 : 0.003);
  }
}