// Problem definition for circularly symmetric disc kernels built from phased Gaussians.
// Component u: ((cos(c x²) r + sin(c x²) i) + x² (cos(c x²) r₁ + sin(c x²) i₁)) exp(-b x²),
// params laid out as [b, c, r, i, r₁, i₁]. A single component has r₁ = i₁ = 0; a merged pair
// (the limit of two coinciding rates) uses the x² term and costs two complex channels per pass.
// The same 1-D profile, with x² → x² + y², is the 2-D kernel.

export const PARAMS_PER_COMPONENT = 6;

// Equiripple sets from the article's source (phaseGaussParallelOpti.cpp), transition bandwidth 0.2.
export const PRESETS = {
  1: [0.862325, 1.624835, 0.767583, 1.862321],
  2: [
    0.886528, 5.268909, 0.411259, -0.548794,
    1.960518, 1.558213, 0.513282, 4.56111,
  ],
  3: [
    2.17649, 5.043495, 1.621035, -2.105439,
    1.019306, 9.027613, -0.28086, -0.162882,
    2.81511, 1.597273, -0.366471, 10.300301,
  ],
  4: [
    4.338459, 1.553635, -5.767909, 46.164397,
    3.839993, 4.693183, 9.795391, -15.227561,
    2.79188, 8.178137, -3.048324, 0.302959,
    1.34219, 12.328289, 0.010001, 0.24465,
  ],
  5: [
    4.892608, 1.685979, -22.356787, 85.91246,
    4.71187, 4.998496, 35.918936, -28.875618,
    4.052795, 8.244168, -13.212253, -1.578428,
    2.929212, 11.900859, 0.507991, 1.816328,
    1.512961, 16.116382, 0.138051, -0.01,
  ],
  6: [
    5.143778, 2.079813, -82.326596, 111.231024,
    5.612426, 6.153387, 113.878661, 58.004879,
    5.982921, 9.802895, 39.479083, -162.028887,
    6.505167, 11.059237, -71.286026, 95.027069,
    3.869579, 14.81052, 1.405746, -3.704914,
    2.201904, 19.032909, -0.152784, -0.107988,
  ],
};

export function presetParams(n) {
  const p = PRESETS[n];
  return Array.from({ length: n }, (_, u) => [...p.slice(u * 4, u * 4 + 4), 0, 0]).flat();
}

// The kernel doesn't depend on component order; ascending frequency matches the search's
// per-slot frequency bands, so a seeded kernel isn't distorted by the bound reflection.
export function sortByFrequency(params) {
  const n = params.length / PARAMS_PER_COMPONENT;
  const comps = Array.from({ length: n }, (_, u) => params.slice(u * PARAMS_PER_COMPONENT, (u + 1) * PARAMS_PER_COMPONENT));
  return comps.sort((a, b) => a[1] - b[1]).flat();
}

export function defaultProblem() {
  return {
    components: 3,
    multiplicity: [1, 1, 1],
    transition: 0.2,
    passSamples: 200,
    stopSamples: 200,
    frequencyBands: true,
  };
}

// 2 for components whose x² term is in use, else 1.
export function multiplicities(params) {
  const n = params.length / PARAMS_PER_COMPONENT;
  return Array.from({ length: n }, (_, u) => (params[u * 6 + 4] || params[u * 6 + 5] ? 2 : 1));
}

export const channelCount = (multiplicity) => multiplicity.reduce((a, m) => a + m, 0);

// The article's heuristic: channel slot k oscillates near (1.14 k + 0.5) π, within ±0.4 π.
export function centerFrequency(k) {
  return (k * 1.14 + 0.5) * Math.PI;
}

// Frequency band of each component: a merged pair spans the bands of the two slots it occupies.
export function frequencyBands(multiplicity) {
  let slot = 0;
  return multiplicity.map((m) => {
    const band = [centerFrequency(slot) - 0.4 * Math.PI, centerFrequency(slot + m - 1) + 0.4 * Math.PI];
    slot += m;
    return band;
  });
}

// Pass band x ∈ [0, 1] uniform; stop band from 1 + transition outwards with log spacing.
export function sampleGrid(problem, passSamples = problem.passSamples, stopSamples = problem.stopSamples) {
  const n = passSamples + stopSamples;
  const x = new Float64Array(n);
  const target = new Float64Array(n);
  for (let t = 0; t < passSamples; t++) {
    x[t] = t / (passSamples - 1);
    target[t] = 1;
  }
  for (let t = 0; t < stopSamples; t++) {
    x[passSamples + t] = 1 + problem.transition - Math.log((stopSamples - t) / stopSamples);
  }
  return { x, target, passSamples };
}

// Reflection bounds keep the search in a sensible region; initial ranges seed the population.
// The x² weights of single components are pinned to zero.
export function bounds(problem) {
  const n = problem.components;
  const d = n * PARAMS_PER_COMPONENT;
  const foldLo = new Float64Array(d);
  const foldHi = new Float64Array(d);
  const initLo = new Float64Array(d);
  const initHi = new Float64Array(d);
  const channels = channelCount(problem.multiplicity);
  const weight = 2 * Math.pow(2.5, channels - 1);
  const topFrequency = centerFrequency(channels - 1) + 0.4 * Math.PI;
  const bands = frequencyBands(problem.multiplicity);
  for (let u = 0; u < n; u++) {
    const k = u * PARAMS_PER_COMPONENT;
    foldLo[k] = 0.2;
    foldHi[k] = 20;
    initLo[k] = 0.3;
    initHi[k] = 1 + channels;
    if (problem.frequencyBands) {
      [foldLo[k + 1], foldHi[k + 1]] = bands[u];
      [initLo[k + 1], initHi[k + 1]] = bands[u];
    } else {
      foldLo[k + 1] = 0.2;
      foldHi[k + 1] = 4 * topFrequency;
      initLo[k + 1] = 0.2;
      initHi[k + 1] = topFrequency;
    }
    const merged = problem.multiplicity[u] === 2;
    for (const j of [2, 3, 4, 5]) {
      const live = j < 4 || merged;
      foldLo[k + j] = live ? -1e9 : 0;
      foldHi[k + j] = live ? 1e9 : 0;
      initLo[k + j] = live ? -weight : 0;
      initHi[k + j] = live ? weight : 0;
    }
  }
  return { foldLo, foldHi, initLo, initHi };
}

export function componentProfile(params, u, x) {
  const [b, c, r, i, r1, i1] = params.slice(u * 6, u * 6 + 6);
  const s = x * x;
  return (Math.cos(c * s) * (r + r1 * s) + Math.sin(c * s) * (i + i1 * s)) * Math.exp(-b * s);
}

export function profile(params, x) {
  let sum = 0;
  for (let u = 0; u < params.length / PARAMS_PER_COMPONENT; u++) sum += componentProfile(params, u, x);
  return sum;
}

export function maxError(params, grid) {
  let worst = 0;
  for (let t = 0; t < grid.x.length; t++) {
    worst = Math.max(worst, Math.abs(profile(params, grid.x[t]) - grid.target[t]));
  }
  return worst;
}

function complexSqrt([re, im]) {
  const m = Math.hypot(re, im);
  return [Math.sqrt((m + re) / 2), Math.sign(im || 1) * Math.sqrt(Math.max(0, (m - re) / 2))];
}

const scaled = ([re, im], k) => [re * k, im * k];

// Separable channels: each has horizontal kernel h(x) = (h0 + h2 x²) e^{λ x²} and vertical kernel
// v(y) = (v0 + v2 y²) e^{λ y²}, λ = -b + i c (complex coefficients as [re, im]), such that the
// 2-D kernel is Re Σ h(x) v(y). With B = r - i·i and A = r₁ - i·i₁ a component is Re[(B + A s) e^{λ s}].
//   Single:      one channel, h = v = √B e^{λ x²} (the article's weight absorption).
//   Merged pair: two channels, σ × (B + A y²)/σ and σ x² × A/σ, summing to (B + A x² + A y²) e^{λ s};
//                σ only balances magnitudes between the passes.
export function separableChannels(params) {
  const zero = [0, 0];
  const out = [];
  for (let u = 0; u < params.length / PARAMS_PER_COMPONENT; u++) {
    const [b, c, r, i, r1, i1] = params.slice(u * 6, u * 6 + 6);
    const B = [r, -i];
    const A = [r1, -i1];
    const base = { component: u, decay: b, phase: c };
    if (!r1 && !i1) {
      const w = complexSqrt(B);
      out.push({ ...base, h0: w, h2: zero, v0: w, v2: zero });
    } else {
      const sigma = Math.sqrt(Math.max(Math.hypot(...A), Math.hypot(...B)));
      out.push({ ...base, h0: [sigma, 0], h2: zero, v0: scaled(B, 1 / sigma), v2: scaled(A, 1 / sigma) });
      out.push({ ...base, h0: zero, h2: [sigma, 0], v0: scaled(A, 1 / sigma), v2: zero });
    }
  }
  return out;
}

export function formatParams(params, problem, error) {
  const n = params.length / PARAMS_PER_COMPONENT;
  const mult = multiplicities(params);
  const fmt = (v) => v.toFixed(6);
  const lines = [
    `Number of components: ${n} (${mult.filter((m) => m === 2).length} merged pairs, ` +
      `${channelCount(mult)} complex channels per pass), transition bandwidth: ${problem.transition}, ` +
      `ripple: ±${error.toFixed(6)}`,
    "",
    "// 1-D profile components (article format; merged pairs add an x*x term):",
  ];
  for (let u = 0; u < n; u++) {
    const [b, c, r, i, r1, i1] = params.slice(u * 6, u * 6 + 6);
    const osc = (wr, wi) => `cos(x*x*${fmt(c)}) * ${fmt(wr)} + sin(x*x*${fmt(c)}) * ${fmt(wi)}`;
    lines.push(
      mult[u] === 1
        ? `(${osc(r, i)}) * exp(-${fmt(b)}*x*x)`
        : `((${osc(r, i)}) + x*x * (${osc(r1, i1)})) * exp(-${fmt(b)}*x*x)`,
    );
  }
  lines.push("", "// b, c, r, i (single) or b, c, r, i, r1, i1 (merged pair):");
  for (let u = 0; u < n; u++) lines.push(params.slice(u * 6, u * 6 + (mult[u] === 1 ? 4 : 6)).map(fmt).join(", "));
  const channels = separableChannels(params);
  if (mult.every((m) => m === 1)) {
    lines.push("", "// Separable form (article's update): a, b, c, d with 1-D kernel (c + d i) exp((a + b i) x²)");
    for (const ch of channels) lines.push([-ch.decay, ch.phase, ...ch.h0].map(fmt).join(", "));
  } else {
    lines.push(
      "",
      "// Separable channels: kernel = Re Σ h(x) v(y), h(x) = (h0 + h2 x²) exp((a + b i) x²), v(y) likewise",
      "// a, b, h0.re, h0.im, h2.re, h2.im, v0.re, v0.im, v2.re, v2.im",
    );
    for (const ch of channels) lines.push([-ch.decay, ch.phase, ...ch.h0, ...ch.h2, ...ch.v0, ...ch.v2].map(fmt).join(", "));
  }
  return lines.join("\n");
}

// One component per line: 4 numbers (single) or 6 (merged pair). Unstructured text: groups of 4.
export function parseParams(text) {
  const numbersIn = (s) => (s.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  const rows = text.split(/\r?\n/).map(numbersIn).filter((row) => row.length > 0);
  if (rows.length > 1 && rows.every((row) => row.length === 4 || row.length === 6)) {
    return rows.flatMap((row) => (row.length === 4 ? [...row, 0, 0] : row));
  }
  const numbers = rows.flat();
  if (numbers.length === 0 || numbers.length % 4 !== 0) {
    throw new Error(
      `Expected one component per line (b, c, r, i or b, c, r, i, r1, i1), or groups of 4 numbers; got ${numbers.length}.`,
    );
  }
  return Array.from({ length: numbers.length / 4 }, (_, u) => [...numbers.slice(u * 4, u * 4 + 4), 0, 0]).flat();
}
