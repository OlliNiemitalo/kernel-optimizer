import { kernelProfile } from './continuous-kernel.js';

// Dense small-radius nodes, then a scale pyramid. These are interpolation nodes, NOT rounded
// radius choices. Each node may eventually contain a pixel-footprint-filtered kernel (mip-like);
// the runtime needs no analytic kernel evaluation, reciprocal radius, or profile resampling.
export const RADIUS_LINEAR_LIMIT = 4;
export const RADIUS_LINEAR_DENSITY = 32;
export const RADIUS_LEVELS_PER_OCTAVE = 32;
const LINEAR_NODES = RADIUS_LINEAR_LIMIT * RADIUS_LINEAR_DENSITY;
export const radiusKey = r => r <= RADIUS_LINEAR_LIMIT ? r * RADIUS_LINEAR_DENSITY
  : LINEAR_NODES + Math.log2(r / RADIUS_LINEAR_LIMIT) * RADIUS_LEVELS_PER_OCTAVE;
export const radiusAtKey = q => q <= LINEAR_NODES ? q / RADIUS_LINEAR_DENSITY
  : RADIUS_LINEAR_LIMIT * 2 ** ((q - LINEAR_NODES) / RADIUS_LEVELS_PER_OCTAVE);
const mix = (a, b, t) => a + (b - a) * t;
const dot = (h, v) => h.reduce((sum, z, ch) => sum + z[0] * v[ch][0] - z[1] * v[ch][1], 0);

export function radiusTables(channels, transition, maxRadius) {
  const profile = kernelProfile(channels, transition);
  const count = Math.max(2, Math.ceil(radiusKey(Math.max(1, maxRadius))) + 1);
  const radii = Array.from({ length: count }, (_, i) => radiusAtKey(i));
  const halves = radii.map(r => r === 0 ? 0 : Math.ceil(Math.max(r, 1) * profile.support));
  const half = halves.at(-1), stride = half + 8; // even taps, plus zero padding for eight outputs
  const sums = radii.map(profile.tapSums);
  const factors = sums.map(([h, v]) => [dot(h, profile.identity[1]), dot(profile.identity[0], v)]);
  const dc = sums.map((directions, i) => directions.map((dir, d) => dir.map(z => z.map(x => x / factors[i][d]))));
  const tables = radii.map((r, i) => [0, 1].map(dir => channels.map((_, ch) => {
    const table = new Float64Array(2 * stride);
    for (let t = 0; t <= halves[i]; t++) {
      const z = profile.rawTap(r, dir, ch, t);
      table[2 * t] = z[0] / factors[i][dir];
      table[2 * t + 1] = z[1] / factors[i][dir];
    }
    return table;
  })));
  const cross = dc.map(([h]) => dc.map(([, v]) => dot(h, v)));
  if (!factors.flat().every(f => Number.isFinite(f) && f > 0) || !cross.flat().every(c => Number.isFinite(c) && c > 0)) {
    throw new Error('Kernel has non-positive mixed-radius normalization.');
  }
  let small = 0;
  for (const row of cross) for (let j = 0; j < count; j++) if (Math.abs(Math.log(row[j])) > 0.003) small = Math.max(small, j + 1);
  const startRadius = Math.max(1, radii[small] ?? (2 * maxRadius + 1));
  const fadeStart = radiusKey(startRadius), fadeEnd = radiusKey(2 * startRadius);
  function coordinate(r) {
    const q = Math.min(Math.max(radiusKey(r), 0), count - 1);
    const i = Math.min(Math.floor(q), count - 2);
    return [i, q - i];
  }
  function tap(r, dir, ch, offset) {
    const [i, f] = coordinate(r), t = Math.abs(offset);
    if (t >= stride) return [0, 0];
    return [0, 1].map(k => mix(tables[i][dir][ch][2 * t + k], tables[i + 1][dir][ch][2 * t + k], f));
  }
  function correction(h, v, fast = false) {
    const [i, a] = coordinate(h), [j, b] = coordinate(v);
    // Interpolate DC weights, THEN take the reciprocal. Interpolating reciprocals is not
    // the normalization of interpolated taps: their mixed DC is bilinear by linearity.
    const c = mix(mix(cross[i][j], cross[i + 1][j], a), mix(cross[i][j + 1], cross[i + 1][j + 1], a), b);
    const f = fast ? Math.max(0, Math.min(1, (j + b - fadeStart) / (fadeEnd - fadeStart))) : 0;
    return mix(1 / c, 1, f * f * (3 - 2 * f));
  }
  return { tables, cross, radii, halves, half, stride, count, fadeStart, fadeEnd, support: profile.support, tap, correction, coordinate };
}