import { integratedKernel } from './continuous-kernel.js';

// Dense small-radius nodes, then a scale pyramid, for nearest lookup or optional interpolation.
// Each node contains pixel-footprint integrals with radius-independent complex channel sums;
// the runtime needs no analytic kernel evaluation, reciprocal radius, or profile resampling.
export const RADIUS_LINEAR_LIMIT = 4;
export const RADIUS_LINEAR_DENSITY = 64;
export const RADIUS_LEVELS_PER_OCTAVE = 64;
const LINEAR_NODES = RADIUS_LINEAR_LIMIT * RADIUS_LINEAR_DENSITY;
export const radiusKey = r => r <= RADIUS_LINEAR_LIMIT ? r * RADIUS_LINEAR_DENSITY
  : LINEAR_NODES + Math.log2(r / RADIUS_LINEAR_LIMIT) * RADIUS_LEVELS_PER_OCTAVE;
export const radiusAtKey = q => q <= LINEAR_NODES ? q / RADIUS_LINEAR_DENSITY
  : RADIUS_LINEAR_LIMIT * 2 ** ((q - LINEAR_NODES) / RADIUS_LEVELS_PER_OCTAVE);
const mix = (a, b, t) => a + (b - a) * t;

export function radiusTables(channels, transition, maxRadius, interpolate = true) {
  const profile = integratedKernel(channels, transition);
  const count = Math.max(2, Math.ceil(radiusKey(Math.max(1, maxRadius))) + 1);
  const radii = Array.from({ length: count }, (_, i) => radiusAtKey(i));
  // Pixel t contributes iff its interval overlaps the profile: |t| - 1/2 < r * support.
  // The touching endpoint has zero area. Avoid gathering an extra all-zero outer pixel.
  const halves = radii.map(r => Math.max(0, Math.ceil(r * profile.support - 0.5)));
  const half = halves.at(-1), stride = half + 8; // even taps, plus zero padding for eight outputs
  const tables = radii.map((r, i) => [0, 1].map(dir => channels.map((_, ch) => {
    const table = new Float64Array(2 * stride);
    for (let t = 0; t <= halves[i]; t++) {
      const z = profile.tap(r, dir, ch, t);
      table[2 * t] = z[0];
      table[2 * t + 1] = z[1];
    }
    return table;
  })));
  function coordinate(r) {
    const q = Math.min(Math.max(radiusKey(r), 0), count - 1);
    if (!interpolate) return [Math.floor(q + 0.5), 0];
    const i = Math.min(Math.floor(q), count - 2);
    return [i, q - i];
  }
  function tap(r, dir, ch, offset) {
    const [i, f] = coordinate(r), t = Math.abs(offset);
    if (t >= stride) return [0, 0];
    if (!interpolate) return [tables[i][dir][ch][2 * t], tables[i][dir][ch][2 * t + 1]];
    return [0, 1].map(k => mix(tables[i][dir][ch][2 * t + k], tables[i + 1][dir][ch][2 * t + k], f));
  }
  return { tables, radii, halves, half, stride, count, support: profile.support, tap, coordinate };
}