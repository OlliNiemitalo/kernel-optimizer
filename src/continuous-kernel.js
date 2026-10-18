// A single, even complex profile per channel/direction, sampled at 256 points per disc radius.
// Radius changes rescale its coordinate, not a discrete choice between kernels. All interpolants
// below are also used by the WGSL sampler; keep the CPU reference for numerical regression tests.
export const PROFILE_DENSITY = 256;
export const NORMALIZATION_LINEAR_RADIUS = 4;
export const NORMALIZATION_DENSITY = 64;

const mix = (a, b, t) => a + t * (b - a);
const dot = (h, v) => h.reduce((s, z, c) => s + z[0] * v[c][0] - z[1] * v[c][1], 0);

export function kernelProfile(channels, transition) {
  const density = PROFILE_DENSITY;
  const last = Math.ceil(density * (1 + transition));
  // One zero endpoint closes the truncated profile continuously instead of dropping a nonzero
  // tap when ceil(radius * support) changes. Extra padding makes the endpoint lookup safe.
  const stride = last + 3;
  const support = (last + 1) / density;
  const profiles = [0, 1].map(dir => channels.map(({ decay, phase, h0, h2, v0, v2 }) => {
    const [k0, k2] = dir ? [v0, v2] : [h0, h2];
    const p = new Float64Array(stride * 2);
    for (let t = 0; t <= last; t++) {
      const s = (t / density) ** 2;
      const re = Math.exp(-decay * s) * Math.cos(phase * s);
      const im = Math.exp(-decay * s) * Math.sin(phase * s);
      const wr = k0[0] + k2[0] * s, wi = k0[1] + k2[1] * s;
      p[2 * t] = wr * re - wi * im;
      p[2 * t + 1] = wr * im + wi * re;
    }
    return p;
  }));
  const sums = profiles.map(dir => dir.map(p => [0, 1].map(k => {
    let s = p[k];
    for (let t = 1; t <= last; t++) s += 2 * p[2 * t + k];
    return s;
  })));
  const gain = 1 / Math.sqrt(dot(sums[0], sums[1]));
  for (const dir of profiles) for (const p of dir) for (let i = 0; i < p.length; i++) p[i] *= gain;
  const identity = sums.map(dir => dir.map(z => z.map(v => v * gain)));

  function rawTap(r, dir, ch, t) {
    const q = Math.min(Math.abs(t) * density / Math.max(r, 1), stride - 2);
    const i = Math.floor(q), a = q - i, p = profiles[dir][ch];
    const tap = [0, 1].map(k => mix(p[2 * i + k], p[2 * (i + 1) + k], a) * density / Math.max(r, 1));
    // Subpixel radii blend continuously with the exact identity. The channel representation of
    // that identity is the reference profile's DC vector, so it also mixes well with large radii.
    return tap.map((v, k) => mix(t === 0 ? identity[dir][ch][k] : 0, v, Math.min(r, 1)));
  }
  function tapSums(r) {
    const half = r === 0 ? 0 : Math.ceil(Math.max(r, 1) * support);
    return [0, 1].map(dir => channels.map((_, ch) => {
      const s = [0, 0];
      for (let t = -half; t <= half; t++) {
        const z = rawTap(r, dir, ch, t);
        s[0] += z[0]; s[1] += z[1];
      }
      return s;
    }));
  }
  return { profiles, identity, density, stride, support, rawTap, tapSums };
}

// High-resolution reference retained for quality comparisons; the renderer uses radius tables.
export function continuousKernel(channels, transition, maxRadius = 256) {
  const { profiles, identity, density, stride, support, rawTap, tapSums } = kernelProfile(channels, transition);
  const key = r => r <= 4 ? r * 64 : 256 + Math.log2(r / 4) * 64;
  const count = Math.ceil(key(Math.max(4, maxRadius))) + 1;
  const radii = Array.from({ length: count }, (_, i) => i <= 256 ? i / 64 : 4 * 2 ** ((i - 256) / 64));
  const dc = radii.map(tapSums);
  const factors = dc.map(([h, v]) => [dot(h, identity[1]), dot(identity[0], v)]);
  const fixes = dc.map(([h], i) => dc.map(([, v], j) => factors[i][0] * factors[j][1] / dot(h, v)));
  let smallKey = 0;
  for (const row of fixes) for (let j = 0; j < count; j++) {
    if (Math.abs(Math.log(row[j])) > 0.003) smallKey = Math.max(smallKey, j + 1);
  }
  const smallRadius = Math.max(1, radii[smallKey] ?? (2 * maxRadius + 1));
  if (!factors.flat().every(v => Number.isFinite(v) && v > 0) || !fixes.flat().every(v => Number.isFinite(v) && v > 0)) {
    throw new Error('Kernel has non-positive mixed-radius normalization.');
  }
  function coordinate(r) {
    const q = Math.min(key(Math.max(r, 0)), count - 1);
    const i = Math.min(Math.floor(q), count - 2);
    return [i, q - i];
  }
  function tap(r, dir, ch, t) {
    const [i, a] = coordinate(r);
    const f = mix(factors[i][dir], factors[i + 1][dir], a);
    return rawTap(r, dir, ch, t).map(v => v / f);
  }
  function correction(h, v, fast = false) {
    const [i, a] = coordinate(h), [j, b] = coordinate(v);
    const f = mix(mix(fixes[i][j], fixes[i + 1][j], a), mix(fixes[i][j + 1], fixes[i + 1][j + 1], a), b);
    const t = fast ? Math.max(0, Math.min(1, v / smallRadius - 1)) : 0;
    return mix(f, 1, t * t * (3 - 2 * t));
  }
  return { profiles, identity, factors, fixes, density, stride, support, count, radii, smallRadius, rawTap, tap, correction };
}