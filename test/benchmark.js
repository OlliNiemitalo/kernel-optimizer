// Capture raw sampler sources before changing revisions; the legacy storage key also accepts b6c1470.
// The baseline executes unchanged shaders on the SAME device as the candidate, with alternating
// order, identical frozen geometry, warmup, and per-pass GPU timestamp medians. No network video.
import { BlurRenderer } from '../src/blur.js';
import { Shapes } from '../src/shapes.js';
import { presetParams } from '../src/problem.js';

const median = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
export async function captureBaseline() {
  // Vite ?raw responds with a JS module; import its default to capture the original source.
  const raw = async path => (await import(/* @vite-ignore */ `${path}?raw&capture=${Date.now()}`)).default;
  const saved = { blur: await raw('/src/blur.js'), kernel: await raw('/src/continuous-kernel.js') };
  if (saved.blur.includes('radius-tables.js')) saved.tables = await raw('/src/radius-tables.js');
  sessionStorage.setItem('blur-baseline-b6c1470', JSON.stringify(saved));
}
export async function run(samples = 7) {
  const saved = JSON.parse(sessionStorage.getItem('blur-baseline-b6c1470') ?? 'null');
  if (!saved) throw new Error('Capture the baseline source in this tab before running the benchmark.');
  const kernelURL = URL.createObjectURL(new Blob([saved.kernel], { type: 'text/javascript' }));
  const tableURL = saved.tables ? URL.createObjectURL(new Blob([
    saved.tables.replace("'./continuous-kernel.js'", JSON.stringify(kernelURL)),
  ], { type: 'text/javascript' })) : null;
  const source = saved.blur.replace('"./problem.js"', JSON.stringify(new URL('/src/problem.js', location.href).href))
    .replace('"./continuous-kernel.js"', JSON.stringify(kernelURL))
    .replace('"./radius-tables.js"', JSON.stringify(tableURL));
  const baselineURL = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  const { BlurRenderer: Baseline } = await import(/* @vite-ignore */ baselineURL);
  const current = await BlurRenderer.create(document.createElement('canvas'));
  const device = current.device;
  if (!device.features.has('timestamp-query')) { device.destroy(); throw new Error('GPU timestamp queries required.'); }
  const baseline = new Baseline(document.createElement('canvas'), device, true);
  const shapes = new Shapes(device);
  Object.assign(shapes, { yaw: 1.342, pitch: 0.008, scale: 1.6487212659831005 });
  const renderers = [baseline, current];
  const results = [];
  try {
    for (const renderer of renderers) {
      renderer.setAnimated(shapes, 960, 600);
      renderer.setDepthOfField();
    }
    for (const radius of [15, 69]) for (const precision of ['f32', 'f16']) {
      for (const renderer of renderers) renderer.setKernel(presetParams(3), radius, 0.2);
      const options = { invFocus: 0.697019047240157 ** 2, precision, layers: 1, sizeBlur: 1, occlusion: 1 };
      for (let warmup = 0; warmup < 2; warmup++) for (const renderer of renderers) await renderer.render(options);
      const timings = [[], []];
      for (let i = 0; i < samples; i++) for (const index of (i % 2 ? [1, 0] : [0, 1])) {
        const result = await renderers[index].render(options);
        const gpu = result.gpuMs;
        timings[index].push({ ...gpu, total: Object.values(gpu).reduce((s, t) => s + t, 0) });
      }
      const medians = timings.map(rows => Object.fromEntries(Object.keys(rows[0]).map(k => [k, median(rows.map(r => r[k]))])));
      results.push({ radius, precision, baseline: medians[0], current: medians[1], speedup: medians[0].total / medians[1].total,
        tableBytes: renderers.map(r => r.kernel.size), samples: timings });
    }
    return results;
  } finally {
    device.destroy();
    URL.revokeObjectURL(baselineURL);
    URL.revokeObjectURL(kernelURL);
    if (tableURL) URL.revokeObjectURL(tableURL);
  }
}