// Regression tests for depth-guided layer smoothing. Read the shared source-radius plane,
// not the radius-table key or tone-mapped output: normalization can hide a corrupted kernel.
import { presetParams } from '../src/problem.js';
import { Shapes } from '../src/shapes.js';

async function sourceMaps(renderer, read) {
  const data = await read(renderer.device, renderer.levelMaps);
  const bytesPerMap = renderer.rowWords * renderer.height * 4;
  return {
    smooth: new Float32Array(data.buffer, 5 * bytesPerMap, renderer.width * renderer.height),
    mask: new Uint8Array(data.buffer, 4 * bytesPerMap, bytesPerMap),
  };
}

export async function layersGPU(renderer, check, read) {
  const width = 97, height = 99;
  // A sharp, narrow triangle against a defocused background. A separate blurry foreground
  // activates layers: it must not cause the triangle's tips to adopt the background radius.
  for (const transpose of [false, true]) {
    const w = transpose ? height : width, h = transpose ? width : height;
    const index = (x, y) => transpose ? x * w + y : y * w + x;
    const image = new Float32Array(w * h * 4).fill(1);
    const radii = new Float32Array(w * h);
    for (const focusRadius of [0, 0.25, -0.5]) {
      const focused = [], noisy = [];
      radii.fill(12);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = index(x, y);
        if (x >= 4 && x <= 34 && y >= 10 && y <= 88) {
          radii[i] = -24 + ((x + y) % 2 ? 0.75 : -0.75);
          if (x >= 12 && x <= 26 && y >= 24 && y <= 74) noisy.push(i);
        }
        if (y >= 20 && y <= 72 && Math.abs(x - 64) <= (y - 20) / 3) {
          radii[i] = focusRadius;
          focused.push(i);
        }
      }
      renderer.setImage(image, w, h);
      renderer.setRadiusMap(radii);
      renderer.varying.kind = 'depth';
      renderer.setKernel(presetParams(3), 32, 0.2);
      for (const interpolate of [false, true]) for (const precision of ['f32', 'f16']) {
        await renderer.render({ layers: 2, interpolate, precision, sizeBlur: 1, occlusion: 1 });
        const { smooth } = await sourceMaps(renderer, read);
        const tiles = await read(renderer.device, renderer.tiles);
        const label = `transpose=${transpose}, focus=${focusRadius}, ${precision}, interpolate=${interpolate}`;
        check(`layer fixture has foreground (${label})`, Number(new Uint32Array(tiles.buffer).at(-1) === 0), 0);
        let focusError = 0, noise = 0;
        for (const i of focused) focusError = Math.max(focusError, Math.abs(smooth[i] - radii[i]));
        for (const i of noisy) noise += (smooth[i] + 24) ** 2;
        check(`sharp triangle retains its source radii (${label})`, focusError, 1e-6);
        check(`depth-guided filtering still denoises foreground (${label})`, Math.sqrt(noise / noisy.length), 0.25);
        const composite = await read(renderer.device, renderer.layerBuffer);
        check(`layer composite remains finite (${label})`, Number(!composite.every(Number.isFinite)), 0);
      }
    }
  }

  // The reported tetra view: its blue silhouette used to acquire 67px radii, and its bottom
  // edge had an 8.47px radius jump exactly where membership crossed 128 despite smooth depth.
  const shapes = new Shapes(renderer.device);
  Object.assign(shapes, { yaw: 1.342, pitch: 0.008, scale: 1.6487212659831005 });
  renderer.setAnimated(shapes, 960, 600);
  renderer.setDepthOfField();
  renderer.setKernel(presetParams(3), 69, 0.2);
  for (const interpolate of [false, true]) {
    await renderer.render({ layers: 2, invFocus: 0.697019047240157 ** 2, interpolate });
    const raw = await read(renderer.device, renderer.radiusBuffer);
    const { smooth, mask } = await sourceMaps(renderer, read);
    let inflation = 0, jump = 0, pairs = 0, subpixelError = 0;
    for (let y = 0; y < 600; y++) for (let x = 0; x < 960; x++) {
      const i = y * 960 + x;
      if (Math.abs(raw[i]) < 2) inflation = Math.max(inflation, Math.abs(smooth[i]) - Math.abs(raw[i]));
      if (Math.abs(raw[i]) <= 0.5) subpixelError = Math.max(subpixelError, Math.abs(smooth[i] - raw[i]));
      if (x === 959) continue;
      const j = i + 1;
      if ((mask[i] >= 128) !== (mask[j] >= 128) && Math.abs(raw[i] - raw[j]) < 0.2 && Math.abs(raw[i]) < 20) {
        pairs++;
        jump = Math.max(jump, Math.abs(smooth[i] - smooth[j]));
      }
    }
    check(`tetra regression crosses soft layer boundary (interpolate=${interpolate})`, Number(pairs === 0), 0);
    check(`tetra near-focus radius inflation (interpolate=${interpolate})`, inflation, 0.5);
    check(`tetra subpixel radii unchanged (interpolate=${interpolate})`, subpixelError, 1e-6);
    check(`tetra no membership-128 radius notch (interpolate=${interpolate})`, jump, 0.3);
  }
  shapes.uniform.destroy();
  shapes.noCoc.destroy();
}