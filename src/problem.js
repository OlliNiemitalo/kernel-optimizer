// Problem definition for circularly symmetric disc kernels built from phased Gaussians.
// Component u: Re[P(x²) e^{λ x²}], λ = −b + ic, P(s) = Σ_k a_k s^k, a_k = r_k − i·i_k, that is
// Σ_k x^{2k} (cos(c x²) r_k + sin(c x²) i_k) exp(-b x²), params laid out as [b, c, r₀, i₀, r₁, i₁, …].
// Multiplicity m (terms k < m in use) is the limit of m coinciding rates and costs m complex
// channels per pass; unused terms are zero. The same 1-D profile, with x² → x² + y², is the 2-D kernel.

export const MAX_MULTIPLICITY = 10;
export const PARAMS_PER_COMPONENT = 2 + 2 * MAX_MULTIPLICITY;

const padded = (component) => [...component, ...Array(PARAMS_PER_COMPONENT - component.length).fill(0)];

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
  return Array.from({ length: n }, (_, u) => padded(p.slice(u * 4, u * 4 + 4))).flat();
}

// The article's "almost no negative values" sets, in its separable form (a, b, c, d): 1-D kernel
// (c + d i) e^{(a + b i) x²}, i.e. 2-D component (c + d i)² e^{(a + b i) s}.
export const NONNEGATIVE_PRESETS = {
  1: [[-2.0126256644, 1.0462519386, 1.7772360464, -1.5705215513]],
  2: [
    [-1.875593446, 5.686828181, -1.134115425, -0.4727562391],
    [-3.0859247607, 0.0311853116, 14.9843211153, -14.9911604837],
  ],
};

// Re[(c + d i)² e^{(a + b i) s}] = (c² − d²) cos(bs) e^{as} − 2cd sin(bs) e^{as}.
export function nonnegativePresetParams(n) {
  return NONNEGATIVE_PRESETS[n].flatMap(([a, b, c, d]) => padded([-a, b, c * c - d * d, -2 * c * d]));
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
    negativeWeight: 1,
  };
}

// Per component, 1 + the highest power of s with a nonzero weight (at least 1).
export function multiplicities(params) {
  const n = params.length / PARAMS_PER_COMPONENT;
  return Array.from({ length: n }, (_, u) => {
    let m = MAX_MULTIPLICITY;
    const k = u * PARAMS_PER_COMPONENT;
    while (m > 1 && !params[k + 2 * m] && !params[k + 2 * m + 1]) m--;
    return m;
  });
}

// Params stored with `stride` numbers per component (6 in older view states), in this layout.
export function withStride(params, stride) {
  const n = params.length / stride;
  if (!Number.isInteger(n) || stride < 4 || stride % 2) throw new Error(`Bad kernel layout: ${params.length} numbers, ${stride} per component.`);
  return Array.from({ length: n }, (_, u) => {
    const component = params.slice(u * stride, (u + 1) * stride);
    if (component.slice(PARAMS_PER_COMPONENT).some((v) => v)) throw new Error(`Multiplicity above ${MAX_MULTIPLICITY} is not supported.`);
    return padded(component.slice(0, PARAMS_PER_COMPONENT));
  }).flat();
}

export const channelCount = (multiplicity) => multiplicity.reduce((a, m) => a + m, 0);

// The article's heuristic: channel slot k oscillates near (1.14 k + 0.5) π, within ±0.4 π.
export function centerFrequency(k) {
  return (k * 1.14 + 0.5) * Math.PI;
}

// Frequency band of each component: one of multiplicity m spans the bands of the m slots it occupies.
export function frequencyBands(multiplicity) {
  let slot = 0;
  return multiplicity.map((m) => {
    const band = [centerFrequency(slot) - 0.4 * Math.PI, centerFrequency(slot + m - 1) + 0.4 * Math.PI];
    slot += m;
    return band;
  });
}

// Pass band x ∈ [0, 1] uniform; stop band from 1 + transition outwards with log spacing.
// negativeWeight multiplies the error wherever the kernel is negative.
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
  return { x, target, passSamples, negativeWeight: problem.negativeWeight ?? 1 };
}

// Reflection bounds keep the search in a sensible region; initial ranges seed the population.
// The weights of unused powers of s are pinned to zero.
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
      foldLo[k + 1] = 0; // c = 0 is a plain Gaussian, as in the article's non-negative 2-component set
      foldHi[k + 1] = 4 * topFrequency;
      initLo[k + 1] = 0.2;
      initHi[k + 1] = topFrequency;
    }
    const live = 2 + 2 * problem.multiplicity[u];
    for (let j = 2; j < PARAMS_PER_COMPONENT; j++) {
      foldLo[k + j] = j < live ? -1e9 : 0;
      foldHi[k + j] = j < live ? 1e9 : 0;
      initLo[k + j] = j < live ? -weight : 0;
      initHi[k + j] = j < live ? weight : 0;
    }
  }
  return { foldLo, foldHi, initLo, initHi };
}

// Component at offset k of params, at s = x² (Horner in s).
function componentAt(params, k, s) {
  let re = 0;
  let im = 0;
  for (let j = k + PARAMS_PER_COMPONENT - 2; j > k; j -= 2) {
    re = re * s + params[j];
    im = im * s + params[j + 1];
  }
  return (Math.cos(params[k + 1] * s) * re + Math.sin(params[k + 1] * s) * im) * Math.exp(-params[k] * s);
}

export function componentProfile(params, u, x) {
  return componentAt(params, u * PARAMS_PER_COMPONENT, x * x);
}

// Isolate each nonzero power of s, retaining the component's rate and parameter layout.
export function componentTerms(params) {
  return multiplicities(params).flatMap((m, component) => {
    const offset = component * PARAMS_PER_COMPONENT;
    const powers = Array.from({ length: m }, (_, power) => power)
      .filter((power) => params[offset + 2 + 2 * power] || params[offset + 3 + 2 * power]);
    return (powers.length ? powers : [0]).map((power) => {
      const term = Array(PARAMS_PER_COMPONENT).fill(0);
      term[0] = params[offset];
      term[1] = params[offset + 1];
      term[2 + 2 * power] = params[offset + 2 + 2 * power];
      term[3 + 2 * power] = params[offset + 3 + 2 * power];
      return { component, power, params: term };
    });
  });
}

// Kernel profile at s = x².
export function profileAt(params, s) {
  let sum = 0;
  for (let k = 0; k < params.length; k += PARAMS_PER_COMPONENT) sum += componentAt(params, k, s);
  return sum;
}

export function profile(params, x) {
  return profileAt(params, x * x);
}

// Max |K(x) − target|, weighted by grid.negativeWeight where K(x) < 0.
export function maxError(params, grid) {
  const w = grid.negativeWeight ?? 1;
  let worst = 0;
  for (let t = 0; t < grid.x.length; t++) {
    const k = profile(params, grid.x[t]);
    worst = Math.max(worst, Math.abs(k - grid.target[t]) * (k < 0 ? w : 1));
  }
  return worst;
}

function complexSqrt([re, im]) {
  const m = Math.hypot(re, im);
  return [Math.sqrt((m + re) / 2), Math.sign(im || 1) * Math.sqrt(Math.max(0, (m - re) / 2))];
}

// Complex polynomial Σ_k w[k] s^k at real s (Horner), as [re, im].
export function channelWeight(w, s) {
  let re = 0;
  let im = 0;
  for (let k = w.length - 1; k >= 0; k--) {
    re = re * s + w[k][0];
    im = im * s + w[k][1];
  }
  return [re, im];
}

const binomial = (n, k) => {
  let c = 1;
  for (let j = 1; j <= k; j++) c = (c * (n - k + j)) / j;
  return c;
};

// R of the Householder QR of a row-major rows × n matrix (rows ≥ n), as a row-major n × n array.
function triangularFactor(A, rows, n) {
  const a = Float64Array.from(A);
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < rows; i++) norm += a[i * n + k] ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = a[k * n + k] > 0 ? -norm : norm;
    a[k * n + k] -= alpha;
    let vv = 0;
    for (let i = k; i < rows; i++) vv += a[i * n + k] ** 2;
    for (let j = k + 1; j < n; j++) {
      let dot = 0;
      for (let i = k; i < rows; i++) dot += a[i * n + k] * a[i * n + j];
      const f = (2 * dot) / vv;
      for (let i = k; i < rows; i++) a[i * n + j] -= f * a[i * n + k];
    }
    a[k * n + k] = alpha;
  }
  const R = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) R[i * n + j] = a[i * n + j];
  return R;
}

// Eigenvalues and eigenvectors (columns of V) of a symmetric row-major n × n matrix, by cyclic Jacobi.
function symmetricEigen(S, n) {
  const a = Float64Array.from(S);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    let diag = 0;
    for (let p = 0; p < n; p++) {
      diag += a[p * n + p] ** 2;
      for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    }
    if (!(off > 1e-32 * diag)) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (apq === 0) continue;
      const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < n; k++) {
        const x = a[k * n + p], y = a[k * n + q];
        a[k * n + p] = c * x - s * y;
        a[k * n + q] = s * x + c * y;
      }
      for (let k = 0; k < n; k++) {
        const x = a[p * n + k], y = a[q * n + k];
        a[p * n + k] = c * x - s * y;
        a[q * n + k] = s * x + c * y;
      }
      for (let k = 0; k < n; k++) {
        const x = V[k * n + p], y = V[k * n + q];
        V[k * n + p] = c * x - s * y;
        V[k * n + q] = s * x + c * y;
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => a[i * n + i]), V };
}

// One pass of a channel at s = t²: Σ over terms (C(s) cos(phase s) + S(s) sin(phase s)) e^{-decay s},
// with complex polynomials C and S in s (coefficients [re, im] of s^0, s^1, …). Returns [re, im].
export function channelSide(side, s) {
  let re = 0;
  let im = 0;
  for (const { decay, phase, cos, sin } of side) {
    const e = Math.exp(-decay * s);
    const [cr, ci] = channelWeight(cos, s);
    const [sr, si] = channelWeight(sin, s);
    const co = Math.cos(phase * s) * e;
    const si_ = Math.sin(phase * s) * e;
    re += cr * co + sr * si_;
    im += ci * co + si * si_;
  }
  return [re, im];
}

// Each component's own separable form, in channelSide format: component u is Re[P(s) e^{λ s}] with
// P(x² + y²) = Σ_{j<m} x^{2j} Q_j(y²), Q_j(t) = Σ_l C(j + l, j) a_{j+l} t^l, so channel j has
// h = σ x^{2j} e^{λ x²} and v = Q_j(y²) e^{λ y²} / σ (m = 1: h = v = √a₀ e^{λ x²}, the article's form).
export function componentChannels(params) {
  const mult = multiplicities(params);
  // P(s) e^{λ s} = (P cos + iP sin) e^{-b s}
  const side = (u, P) => [{ decay: params[u * PARAMS_PER_COMPONENT], phase: params[u * PARAMS_PER_COMPONENT + 1], cos: P, sin: P.map(([re, im]) => [-im, re]) }];
  return mult.flatMap((m, u) => {
    const k0 = u * PARAMS_PER_COMPONENT;
    const a = Array.from({ length: m }, (_, q) => [params[k0 + 2 + 2 * q], -params[k0 + 3 + 2 * q]]);
    if (m === 1) {
      const w = complexSqrt(a[0]);
      return [{ component: u, h: side(u, [w]), v: side(u, [w]) }];
    }
    return Array.from({ length: m }, (_, j) => {
      const Q = Array.from({ length: m - j }, (_, l) => a[j + l].map((z) => z * binomial(j + l, j)));
      const sigma = Math.sqrt(Math.max(...Q.map((z) => Math.hypot(...z)))) || 1;
      const h = Array.from({ length: j + 1 }, (_, q) => (q === j ? [sigma, 0] : [0, 0]));
      return { component: u, h: side(u, h), v: side(u, Q.map((z) => z.map((c) => c / sigma))) };
    });
  });
}
const QUADRATURE_NODES = 1024;
const NEGLIGIBLE_TERM = 1e-14;

// Separable channels for rendering: the 2-D kernel is Re Σ_n h_n(x) v_n(y), each pass evaluated by
// channelSide. The real kernel is fᵀ(x) C f(y) in the real basis f = t^k {cos, sin}(c x²) e^{−b x²}
// (t = 2b x², every component and power k < m). Orthonormalizing f over [0, support] (QR, R) and
// diagonalizing R C Rᵀ = Σ e_j u_j u_jᵀ gives Re K = Σ e_j g_j(x) g_j(y) with g_j = fᵀ C Rᵀ u_j / e_j,
// paired into complex channels: Re[(√|e_a| g_a + i√|e_b| g_b)(±√|e_a| g_a ∓ i√|e_b| g_b)]. Unlike
// per-component forms, the passes carry no large cancelling parts that f16 intermediates can't resolve.
// With pairs = false, each real term e_j g_j(x) g_j(y) is its own real channel: h = √|e_j| g_j and
// v = sign(e_j) h.
export function separableChannels(params, support = 2, { pairs = true } = {}) {
  const mult = multiplicities(params);
  const terms = [];
  mult.forEach((m, u) => {
    const k0 = u * PARAMS_PER_COMPONENT;
    for (const part of [0, 1]) for (let k = 0; k < m; k++) terms.push({ u, part, k, b: params[k0], c: params[k0 + 1] });
  });
  const n = terms.length;
  const C = new Float64Array(n * n);
  let start = 0;
  mult.forEach((m, u) => {
    const k0 = u * PARAMS_PER_COMPONENT;
    const scale = 2 * params[k0];
    for (let k = 0; k < m; k++) for (let l = 0; k + l < m; l++) {
      const f = binomial(k + l, k) / scale ** (k + l);
      const re = params[k0 + 2 + 2 * (k + l)] * f; // Re a_{k+l}
      const im = -params[k0 + 3 + 2 * (k + l)] * f; // Im a_{k+l}
      // Re[a (cos_x + i sin_x)(cos_y + i sin_y)] = Re a (cos cos − sin sin) − Im a (cos sin + sin cos)
      C[(start + k) * n + start + l] = re;
      C[(start + m + k) * n + start + m + l] = -re;
      C[(start + k) * n + start + m + l] = -im;
      C[(start + m + k) * n + start + l] = -im;
    }
    start += 2 * m;
  });
  const rows = QUADRATURE_NODES;
  const A = new Float64Array(rows * n);
  for (let i = 0; i < rows; i++) {
    const x = ((i + 0.5) * support) / rows;
    const s = x * x;
    const w = Math.sqrt(support / rows);
    terms.forEach(({ part, k, b, c }, p) => {
      A[i * n + p] = w * (2 * b * s) ** k * (part ? Math.sin(c * s) : Math.cos(c * s)) * Math.exp(-b * s);
    });
  }
  const R = triangularFactor(A, rows, n);
  const T = new Float64Array(n * n); // C Rᵀ
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let k = j; k < n; k++) sum += C[i * n + k] * R[j * n + k];
    T[i * n + j] = sum;
  }
  const S = new Float64Array(n * n); // R C Rᵀ
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let k = i; k < n; k++) sum += R[i * n + k] * T[k * n + j];
    S[i * n + j] = sum;
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) S[i * n + j] = S[j * n + i] = (S[i * n + j] + S[j * n + i]) / 2;
  const { values, V } = symmetricEigen(S, n);
  const largest = Math.max(0, ...values.map(Math.abs));
  const kept = values
    .map((value, j) => ({ value, j }))
    .filter(({ value }) => Math.abs(value) > NEGLIGIBLE_TERM * largest)
    .sort((p, q) => Math.abs(q.value) - Math.abs(p.value))
    .map(({ value, j }) => {
      const coef = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += T[i * n + k] * V[k * n + j];
        coef[i] = sum / value;
      }
      return { value, coef };
    });
  // A pass as Σ_q factor_q g_q, factors complex: per component, cos and sin polynomials in s.
  const side = (parts) => mult.map((m, u) => {
    const k0 = u * PARAMS_PER_COMPONENT;
    const scale = 2 * params[k0];
    const poly = (part) => Array.from({ length: m }, (_, k) => {
      const p = terms.findIndex((t) => t.u === u && t.part === part && t.k === k);
      return parts.reduce(([re, im], { factor: [fr, fi], coef }) => [re + fr * coef[p] * scale ** k, im + fi * coef[p] * scale ** k], [0, 0]);
    });
    return { decay: params[k0], phase: params[k0 + 1], cos: poly(0), sin: poly(1) };
  });
  const channels = [];
  if (!pairs) {
    return kept.map(({ value, coef }) => {
      const s = Math.sqrt(Math.abs(value));
      return { h: side([{ factor: [s, 0], coef }]), v: side([{ factor: [Math.sign(value) * s, 0], coef }]), terms: 1, sign: Math.sign(value) };
    });
  }
  for (let j = 0; j < kept.length; j += 2) {
    const [a, b] = [kept[j], kept[j + 1]];
    const sa = Math.sqrt(Math.abs(a.value));
    const h = [{ factor: [sa, 0], coef: a.coef }];
    const v = [{ factor: [Math.sign(a.value) * sa, 0], coef: a.coef }];
    if (b) {
      const sb = Math.sqrt(Math.abs(b.value));
      h.push({ factor: [0, sb], coef: b.coef });
      v.push({ factor: [0, -Math.sign(b.value) * sb], coef: b.coef });
    }
    channels.push({ h: side(h), v: side(v), terms: b ? 2 : 1 });
  }
  return channels;
}

const powerOfX2 = (k) => Array(k).fill("x*x").join("*");

export function formatParams(params, problem, error) {
  const n = params.length / PARAMS_PER_COMPONENT;
  const mult = multiplicities(params);
  const fmt = (v) => v.toFixed(10);
  const higher = mult.filter((m) => m > 1).length;
  const lines = [
    `Number of components: ${n} (${higher} with repeated rates, multiplicities ${mult.join(", ")}; ` +
      `${2 * channelCount(mult)} rendering terms), transition bandwidth: ${problem.transition}, ` +
      `ripple: ±${error.toFixed(6)}`,
    "",
    "// Profile k(x) as a sum of components (article format; multiplicity m adds terms up to (x*x)^(m-1)):",
  ];
  for (let u = 0; u < n; u++) {
    const k = u * PARAMS_PER_COMPONENT;
    const [b, c] = [params[k], params[k + 1]];
    const osc = (q) => `cos(x*x*${fmt(c)}) * ${fmt(params[k + 2 + 2 * q])} + sin(x*x*${fmt(c)}) * ${fmt(params[k + 3 + 2 * q])}`;
    const terms = Array.from({ length: mult[u] }, (_, q) => (q ? `${powerOfX2(q)} * (${osc(q)})` : `(${osc(q)})`));
    lines.push(mult[u] === 1 ? `${terms[0]} * exp(-${fmt(b)}*x*x)` : `(${terms.join(" + ")}) * exp(-${fmt(b)}*x*x)`);
  }
  lines.push("", "// b, c, r, i (single), followed by r1, i1, r2, i2, … (weights of x^2, x^4, …) for multiplicity > 1:");
  for (let u = 0; u < n; u++) {
    lines.push(params.slice(u * PARAMS_PER_COMPONENT, u * PARAMS_PER_COMPONENT + 2 + 2 * mult[u]).map(fmt).join(", "));
  }
  if (mult.every((m) => m === 1)) {
    lines.push("", "// Separable form (article's update): a, b, c, d with 1-D kernel (c + d i) exp((a + b i) x²)");
    for (let u = 0; u < n; u++) {
      const k = u * PARAMS_PER_COMPONENT;
      lines.push([-params[k], params[k + 1], ...complexSqrt([params[k + 2], -params[k + 3]])].map(fmt).join(", "));
    }
  }
  const channels = separableChannels(params, 1 + problem.transition);
  const basis = mult.flatMap((m, u) => Array.from({ length: m }, (_, k) => [`${u} cos x^${2 * k}`, `${u} sin x^${2 * k}`]).flat());
  const row = (side) => side.flatMap(({ cos, sin }) => cos.flatMap((z, k) => [...z, ...sin[k]])).map(fmt).join(", ");
  lines.push(
    "",
    `// Rendering terms in ${channels.length} pairs, as the blur uses them: kernel(x, y) = Re Σ_n h_n(x) v_n(y), with`,
    "// h = H f(x) and v = V f(y), where f are the real basis functions of the components above:",
    "//   'u cos x^2k' = x^2k cos(c_u x²) exp(-b_u x²), 'u sin x^2k' = x^2k sin(c_u x²) exp(-b_u x²).",
    "// The kernel is f(x)ᵀ Re(Hᵀ V) f(y), and Re(Hᵀ V) is the components' weight matrix (up to rounding",
    "// and, at high multiplicity, nearly dependent basis directions). Rows: pairs (re, im: one term each); columns:",
    `// ${basis.join(", ")}`,
    "// H:",
    ...channels.map((ch) => row(ch.h)),
    "// V:",
    ...channels.map((ch) => row(ch.v)),
  );
  return lines.join("\n");
}

// One component per line: b, c, r, i and, for multiplicity m > 1, r1, i1, … up to r(m-1), i(m-1).
// The whole "Current kernel" text also works: its parameter section is used. Unstructured text: groups of 4.
export function parseParams(text) {
  const numbersIn = (s) => (s.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  let lines = text.split(/\r?\n/);
  const section = lines.findIndex((line) => line.trim().startsWith("// b, c, r, i"));
  if (section >= 0) {
    const end = lines.findIndex((line, i) => i > section && (!line.trim() || line.trim().startsWith("//")));
    lines = lines.slice(section + 1, end < 0 ? undefined : end);
  }
  const rows = lines.map(numbersIn).filter((row) => row.length > 0);
  if (rows.length >= (section >= 0 ? 1 : 2) && rows.every((row) => row.length >= 4 && row.length % 2 === 0 && row.length <= PARAMS_PER_COMPONENT)) {
    return rows.flatMap(padded);
  }
  const numbers = rows.flat();
  if (numbers.length === 0 || numbers.length % 4 !== 0) {
    throw new Error(
      `Expected one component per line (b, c, r, i, then r1, i1, … for multiplicity up to ${MAX_MULTIPLICITY}), ` +
        `or groups of 4 numbers; got ${numbers.length}.`,
    );
  }
  return Array.from({ length: numbers.length / 4 }, (_, u) => padded(numbers.slice(u * 4, u * 4 + 4))).flat();
}
