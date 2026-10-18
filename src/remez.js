// Remez-style minimax solvers for the kernel parameters (CPU, float64).
//
// chebyshev(): discrete linear Chebyshev approximation min_z max_j W_j |t_j − a_j·z| by the
// single-point exchange (Stiefel) algorithm, i.e. the dual simplex method on m + 1 reference rows.
// The weight may depend on the sign of the error (W⁺ when t_j > a_j·z, else W⁻).
//
// The kernel is linear in its weights (r, i, r₁, i₁) and nonlinear in decays and frequencies:
// optimalWeights() solves the weights exactly by exchange; remezSteps() refines everything.

import { PARAMS_PER_COMPONENT, maxError, multiplicities } from "./problem.js";

// Householder QR of a p×m row-major matrix (p ≥ m). Returns explicit Q (p×p) and R (m×m).
function qr(A, p, m) {
  const a = Float64Array.from(A);
  const Q = new Float64Array(p * p);
  for (let i = 0; i < p; i++) Q[i * p + i] = 1;
  const v = new Float64Array(p);
  for (let k = 0; k < m; k++) {
    let norm = 0;
    for (let i = k; i < p; i++) norm += a[i * m + k] ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = a[k * m + k] > 0 ? -norm : norm;
    let vv = 0;
    for (let i = k; i < p; i++) {
      v[i] = a[i * m + k] - (i === k ? alpha : 0);
      vv += v[i] * v[i];
    }
    if (vv === 0) continue;
    for (let j = k; j < m; j++) {
      let dot = 0;
      for (let i = k; i < p; i++) dot += v[i] * a[i * m + j];
      const f = (2 * dot) / vv;
      for (let i = k; i < p; i++) a[i * m + j] -= f * v[i];
    }
    for (let r = 0; r < p; r++) {
      let dot = 0;
      for (let i = k; i < p; i++) dot += Q[r * p + i] * v[i];
      const f = (2 * dot) / vv;
      for (let i = k; i < p; i++) Q[r * p + i] -= f * v[i];
    }
  }
  const R = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = i; j < m; j++) R[i * m + j] = a[i * m + j];
  return { Q, R };
}

function wellConditioned(R, m) {
  let lo = Infinity, hi = 0;
  for (let i = 0; i < m; i++) {
    const d = Math.abs(R[i * m + i]);
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  return hi > 0 && lo > 1e-13 * hi;
}

// Solve R z = Qᵀ y (first m components), i.e. least squares on the p rows.
function solveLeast(Q, R, p, m, y) {
  const z = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    let c = 0;
    for (let r = 0; r < p; r++) c += Q[r * p + k] * y[r];
    z[k] = c;
  }
  for (let i = m - 1; i >= 0; i--) {
    let c = z[i];
    for (let j = i + 1; j < m; j++) c -= R[i * m + j] * z[j];
    z[i] = c / R[i * m + i];
  }
  return z;
}

const weighted = (e, wPlus, wMinus) => Math.abs(e) * (e > 0 ? wPlus : wMinus);

// Rows chosen greedily by largest residual norm (row-pivoted Gram–Schmidt), plus the worst row
// of their interpolant.
function initialReference(flat, n, m, target, wPlus, wMinus) {
  const res = Float64Array.from(flat);
  const used = new Uint8Array(n);
  const ref = [];
  const q = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    let best = -1, bestNorm = 0;
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      let s = 0;
      for (let c = 0, o = j * m; c < m; c++) s += res[o + c] * res[o + c];
      if (s > bestNorm) {
        best = j;
        bestNorm = s;
      }
    }
    if (best < 0) throw new Error("Chebyshev basis is rank deficient on the grid.");
    used[best] = 1;
    ref.push(best);
    for (let c = 0; c < m; c++) q[c] = res[best * m + c] / Math.sqrt(bestNorm);
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      const o = j * m;
      let d = 0;
      for (let c = 0; c < m; c++) d += res[o + c] * q[c];
      for (let c = 0; c < m; c++) res[o + c] -= d * q[c];
    }
  }
  const A = new Float64Array(m * m);
  ref.forEach((j, i) => A.set(flat.subarray(j * m, (j + 1) * m), i * m));
  const { Q, R } = qr(A, m, m);
  const z = solveLeast(Q, R, m, m, ref.map((j) => target[j]));
  let worst = -1, worstErr = -1;
  for (let j = 0; j < n; j++) {
    if (used[j]) continue;
    let k = 0;
    for (let c = 0; c < m; c++) k += flat[j * m + c] * z[c];
    const a = weighted(target[j] - k, wPlus[j], wMinus[j]);
    if (a > worstErr) [worst, worstErr] = [j, a];
  }
  ref.push(worst);
  return ref;
}

// rows: n arrays of length m; target, wPlus, wMinus: length n.
// Returns { z, delta (reference level), error (max weighted error), reference, iterations, converged }.
export function chebyshev(rows, target, wPlus, wMinus, { reference = null, maxIterations = 2000, tolerance = 1e-10 } = {}) {
  const n = rows.length;
  const m = rows[0].length;
  const p = m + 1;
  let ref = reference && reference.length === p && new Set(reference).size === p && reference.every((j) => j < n)
    ? [...reference] : null;
  let fresh = false;
  const A = new Float64Array(p * m);
  const e = new Float64Array(n);
  const flat = new Float64Array(n * m);
  rows.forEach((row, j) => flat.set(row, j * m));
  let result = null;
  for (let it = 0; it < maxIterations; it++) {
    if (!ref) {
      ref = initialReference(flat, n, m, target, wPlus, wMinus);
      fresh = true;
    }
    ref.forEach((j, i) => A.set(rows[j], i * m));
    const { Q, R } = qr(A, p, m);
    if (!wellConditioned(R, m)) {
      if (fresh) throw new Error("Chebyshev reference is singular.");
      ref = null;
      continue;
    }
    // Dual vector: the left null vector of the reference matrix, oriented so Σ v t ≥ 0.
    const v = Float64Array.from({ length: p }, (_, i) => Q[i * p + m]);
    let sigma = 0;
    for (let i = 0; i < p; i++) sigma += v[i] * target[ref[i]];
    if (sigma < 0) {
      for (let i = 0; i < p; i++) v[i] = -v[i];
      sigma = -sigma;
    }
    const sign = Float64Array.from(v, (x) => (x < 0 ? -1 : 1));
    let denom = 0;
    for (let i = 0; i < p; i++) denom += Math.abs(v[i]) / (sign[i] > 0 ? wPlus[ref[i]] : wMinus[ref[i]]);
    const delta = sigma / denom;
    // Levelled reference errors e_k = δ s_k / W_k, then z interpolates t − e on the reference.
    const y = ref.map((j, i) => target[j] - (delta * sign[i]) / (sign[i] > 0 ? wPlus[j] : wMinus[j]));
    const z = solveLeast(Q, R, p, m, y);
    let worst = 0, worstErr = -1;
    for (let j = 0; j < n; j++) {
      let k = 0;
      for (let c = 0, o = j * m; c < m; c++) k += flat[o + c] * z[c];
      const ej = target[j] - k;
      e[j] = ej;
      const a = ej > 0 ? ej * wPlus[j] : -ej * wMinus[j];
      if (a > worstErr) {
        worst = j;
        worstErr = a;
      }
    }
    result = { z, delta, error: worstErr, reference: [...ref], iterations: it + 1, converged: false };
    if (worstErr <= delta * (1 + tolerance) + 1e-300 || ref.includes(worst)) {
      result.converged = true;
      return result;
    }
    // Entering row j*: ν solves Σ_R ν_k a_k = −a_{j*} (minimum norm); the new dual vector is
    // s*·(ν, 1) + α v, with α lowered until one reference weight reaches zero (that row leaves).
    const sStar = e[worst] > 0 ? 1 : -1;
    const w = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let c = -rows[worst][i];
      for (let j = 0; j < i; j++) c -= R[j * m + i] * w[j];
      w[i] = c / R[i * m + i];
    }
    let leave = -1, alpha = -Infinity;
    for (let i = 0; i < p; i++) {
      let nu = 0;
      for (let k = 0; k < m; k++) nu += Q[i * p + k] * w[k];
      const g = sStar * nu * sign[i];
      const bound = Math.abs(v[i]) > 1e-300 ? -g / Math.abs(v[i]) : g < 0 ? Infinity : -Infinity;
      if (bound > alpha) [leave, alpha] = [i, bound];
    }
    ref[leave] = worst;
    fresh = false;
  }
  return result;
}

// Parameter index layout: linear weights (r, i and, for merged pairs, r₁, i₁) and rates (b, c).
function layout(params) {
  return layoutFor(multiplicities(params));
}

function layoutFor(mult) {
  const linear = [];
  const rates = [];
  mult.forEach((mu, u) => {
    const k = u * PARAMS_PER_COMPONENT;
    linear.push(k + 2, k + 3);
    if (mu === 2) linear.push(k + 4, k + 5);
    rates.push(k, k + 1);
  });
  return { linear, rates };
}

// Columns ∂K/∂p at the grid: basis functions for linear indices, Jacobian columns for rates.
function columns(params, s, indices) {
  const n = s.length;
  const cols = new Map(indices.map((j) => [j, new Float64Array(n)]));
  for (let u = 0; u < params.length / PARAMS_PER_COMPONENT; u++) {
    const k = u * PARAMS_PER_COMPONENT;
    const [b, c, r, i, r1, i1] = params.slice(k, k + 6);
    const need = [0, 1, 2, 3, 4, 5].map((o) => cols.get(k + o));
    if (!need.some(Boolean)) continue;
    for (let t = 0; t < n; t++) {
      const st = s[t];
      const E = Math.exp(-b * st);
      const C = Math.cos(c * st) * E;
      const S = Math.sin(c * st) * E;
      const re = r + r1 * st;
      const im = i + i1 * st;
      if (need[0]) need[0][t] = -st * (C * re + S * im);
      if (need[1]) need[1][t] = st * (C * im - S * re);
      if (need[2]) need[2][t] = C;
      if (need[3]) need[3][t] = S;
      if (need[4]) need[4][t] = st * C;
      if (need[5]) need[5][t] = st * S;
    }
  }
  return indices.map((j) => cols.get(j));
}

// Rows with each column divided by its max magnitude, plus zeroed trailing rows.
function scaledRows(cols, n, extra = 0) {
  const m = cols.length;
  const scale = cols.map((col) => col.reduce((a, x) => Math.max(a, Math.abs(x)), 0) || 1);
  const flat = new Float64Array((n + extra) * m);
  for (let t = 0; t < n; t++) for (let c = 0; c < m; c++) flat[t * m + c] = cols[c][t] / scale[c];
  const rows = Array.from({ length: n + extra }, (_, t) => flat.subarray(t * m, (t + 1) * m));
  return { rows, scale };
}

// The negative-value weight applies where K < 0, i.e. where e = t − K > t. With t ∈ {0, 1} and
// ripple below 1, that is e > 0 in the stop band and never in the pass band.
function errorWeights(grid) {
  const w = grid.negativeWeight ?? 1;
  return {
    wPlus: Float64Array.from(grid.target, (t) => (t > 0 ? 1 : w)),
    wMinus: new Float64Array(grid.target.length).fill(1),
  };
}

// Minimax-optimal linear weights (r, i, r₁, i₁) for the given decays and frequencies.
export function optimalWeights(params, grid) {
  const sol = new WeightSolver(grid, multiplicities(params)).solve(params);
  return { params: sol.params, error: sol.error, level: sol.level, exchanges: sol.exchanges };
}

// Reusable weight solver for one grid and component structure; a previous reference warm-starts it.
export class WeightSolver {
  constructor(grid, multiplicity) {
    this.grid = grid;
    this.s = Float64Array.from(grid.x, (x) => x * x);
    this.target = Float64Array.from(grid.target);
    Object.assign(this, errorWeights(grid));
    this.multiplicity = multiplicity;
    this.linear = layoutFor(multiplicity).linear;
    const n = this.s.length, m = this.linear.length;
    this.flat = new Float64Array(n * m);
    this.rows = Array.from({ length: n }, (_, t) => this.flat.subarray(t * m, (t + 1) * m));
    this.scale = new Float64Array(m);
  }

  solve(params, reference = null) {
    const { s, flat, scale, linear } = this;
    const n = s.length, m = linear.length;
    // Basis columns, each scaled to unit maximum magnitude (same order as `linear`).
    scale.fill(0);
    let col = 0;
    this.multiplicity.forEach((mu, u) => {
      const b = params[u * PARAMS_PER_COMPONENT], c = params[u * PARAMS_PER_COMPONENT + 1];
      for (let t = 0; t < n; t++) {
        const st = s[t], E = Math.exp(-b * st);
        const i = t * m + col;
        flat[i] = Math.cos(c * st) * E;
        flat[i + 1] = Math.sin(c * st) * E;
        if (mu === 2) {
          flat[i + 2] = st * flat[i];
          flat[i + 3] = st * flat[i + 1];
        }
      }
      col += mu === 2 ? 4 : 2;
    });
    for (let t = 0; t < n; t++) for (let k = 0; k < m; k++) scale[k] = Math.max(scale[k], Math.abs(flat[t * m + k]));
    for (let k = 0; k < m; k++) scale[k] ||= 1;
    for (let t = 0; t < n; t++) for (let k = 0; k < m; k++) flat[t * m + k] /= scale[k];
    const sol = chebyshev(this.rows, this.target, this.wPlus, this.wMinus, { reference: reference ?? this.last });
    this.last = sol.reference;
    const out = Array.from(params);
    linear.forEach((j, k) => (out[j] = sol.z[k] / scale[k]));
    // Below 1 the pass band can't be negative, so the exchange's error equals maxError.
    const error = sol.error < 1 ? sol.error : maxError(out, this.grid);
    return { params: out, error, level: sol.delta, exchanges: sol.iterations, reference: sol.reference };
  }
}

// A search candidate's re-solved weights, or infinite cost if its basis is singular.
export function solveCandidate(solver, { params, reference }) {
  try {
    const sol = solver.solve(params, reference);
    return { params: sol.params, cost: Number.isFinite(sol.error) ? sol.error : Infinity, reference: sol.reference };
  } catch {
    return { params, cost: Infinity, reference: null };
  }
}

// Profile on the grid, summed in the same order as problem.js profile().
function gridProfile(params, s) {
  const K = new Float64Array(s.length);
  for (let k = 0; k < params.length; k += PARAMS_PER_COMPONENT) {
    const [b, c, r, i, r1, i1] = params.slice(k, k + 6);
    for (let t = 0; t < s.length; t++) {
      const st = s[t];
      K[t] += (Math.cos(c * st) * (r + r1 * st) + Math.sin(c * st) * (i + i1 * st)) * Math.exp(-b * st);
    }
  }
  return K;
}

// For each parameter: which of (λ, B, A) it moves, and the complex direction. Order b, c, r, i, r₁, i₁.
const DIRECTIONS = [[0, -1, 0], [0, 0, 1], [1, 1, 0], [1, 0, -1], [2, 1, 0], [2, 0, -1]];

function derivativeSpec(live) {
  const comps = new Map();
  live.forEach((j, a) => {
    const u = Math.floor(j / PARAMS_PER_COMPONENT);
    const [kind, cr, ci] = DIRECTIONS[j % PARAMS_PER_COMPONENT];
    if (!comps.has(u)) comps.set(u, []);
    comps.get(u).push({ a, kind, cr, ci });
  });
  return { m: live.length, comps };
}

// K(s), ∂K/∂s, ∂²K/∂s² and, if full, over live parameters ∂K/∂p, ∂²K/∂p∂s, ∂²K/∂p∂q.
// Component: Re F, F = (B + A s) e^{λ s}, λ = −b + ic, B = r − i·i, A = r₁ − i·i₁. F is linear in
// B and A, so the nonzero second parameter derivatives are F_λλ = s² P e, F_Bλ = s e, F_Aλ = s² e.
function pointDerivatives(params, spec, s, full = true) {
  const { m, comps } = spec;
  let K = 0, Ks = 0, Kss = 0;
  const Kp = full && new Float64Array(m), Kps = full && new Float64Array(m), Kpp = full && new Float64Array(m * m);
  for (let k = 0, u = 0; k < params.length; k += PARAMS_PER_COMPONENT, u++) {
    const [b, c, r, i, r1, i1] = params.slice(k, k + 6);
    const lr = -b, li = c, Ar = r1, Ai = -i1;
    const E = Math.exp(-b * s), er = Math.cos(c * s) * E, ei = Math.sin(c * s) * E;
    const Pr = r + Ar * s, Pi = -i + Ai * s;
    const LPr = lr * Pr - li * Pi, LPi = lr * Pi + li * Pr;
    const PEr = Pr * er - Pi * ei, PEi = Pr * ei + Pi * er;
    const Qr = Ar + LPr, Qi = Ai + LPi; // A + λP
    const Tr = 2 * (lr * Ar - li * Ai) + lr * LPr - li * LPi, Ti = 2 * (lr * Ai + li * Ar) + lr * LPi + li * LPr;
    K += PEr;
    Ks += Qr * er - Qi * ei;
    Kss += Tr * er - Ti * ei;
    const dirs = full && comps.get(u);
    if (!dirs) continue;
    const lEr = lr * er - li * ei, lEi = lr * ei + li * er;
    const Ur = Pr + s * Qr, Ui = Pi + s * Qi;
    const F = [[s * PEr, s * PEi], [er, ei], [s * er, s * ei]];
    const G = [[Ur * er - Ui * ei, Ur * ei + Ui * er], [lEr, lEi], [er + s * lEr, ei + s * lEi]];
    const LL = [s * s * PEr, s * s * PEi], BL = [s * er, s * ei], AL = [s * s * er, s * s * ei];
    const second = [[LL, BL, AL], [BL, null, null], [AL, null, null]];
    for (const d of dirs) {
      Kp[d.a] = d.cr * F[d.kind][0] - d.ci * F[d.kind][1];
      Kps[d.a] = d.cr * G[d.kind][0] - d.ci * G[d.kind][1];
      for (const f of dirs) {
        const X = second[d.kind][f.kind];
        if (!X) continue;
        const pr = d.cr * f.cr - d.ci * f.ci, pi = d.cr * f.ci + d.ci * f.cr;
        Kpp[d.a * m + f.a] = pr * X[0] - pi * X[1];
      }
    }
  }
  return { K, Ks, Kss, Kp, Kps, Kpp };
}

// Moves an active grid point to the continuous local extremum of its band (in s = x²), unless it
// sits at a band end. Interior extrema carry the extremum-motion term of the reduced Hessian.
function locateExtremum(params, spec, s0, band, sigma) {
  let s = s0;
  if (s0 > band.lo && s0 < band.hi) {
    for (let k = 0; k < 6; k++) {
      const d = pointDerivatives(params, spec, s, false);
      if (!(sigma * d.Kss > 0)) break;
      const step = Math.max(-band.step, Math.min(band.step, -d.Ks / d.Kss));
      s = Math.max(band.lo, Math.min(band.hi, s + step));
      if (Math.abs(step) < 1e-14 * Math.max(1, s)) break;
    }
  }
  const d = pointDerivatives(params, spec, s);
  return { s, d, interior: sigma * d.Kss > 0 && s > band.lo && s < band.hi };
}

// Dense Gaussian elimination with partial pivoting; returns null if singular.
function solveDense(M, rhs, size) {
  const a = Float64Array.from(M), x = Float64Array.from(rhs);
  for (let k = 0; k < size; k++) {
    let piv = k;
    for (let i = k + 1; i < size; i++) if (Math.abs(a[i * size + k]) > Math.abs(a[piv * size + k])) piv = i;
    if (!(Math.abs(a[piv * size + k]) > 1e-300)) return null;
    if (piv !== k) {
      for (let j = 0; j < size; j++) [a[k * size + j], a[piv * size + j]] = [a[piv * size + j], a[k * size + j]];
      [x[k], x[piv]] = [x[piv], x[k]];
    }
    for (let i = k + 1; i < size; i++) {
      const f = a[i * size + k] / a[k * size + k];
      if (f === 0) continue;
      for (let j = k; j < size; j++) a[i * size + j] -= f * a[k * size + j];
      x[i] -= f * x[k];
    }
  }
  for (let i = size - 1; i >= 0; i--) {
    let c = x[i];
    for (let j = i + 1; j < size; j++) c -= a[i * size + j] * x[j];
    x[i] = c / a[i * size + i];
  }
  return x.every(Number.isFinite) ? x : null;
}

// Constraint gradients ∇c_k of c_k = σ_k W_k (t_k − K(s_k)), in column-scaled units.
const scaledGradients = (active, scale) =>
  active.map(({ d, sigma, weight }) => Float64Array.from(d.Kp, (x, j) => (-sigma * weight * x) / scale[j]));

// First-order multiplier estimate: min ‖Σ λ_k ∇c_k‖² subject to Σ λ_k = 1, clamped to λ ≥ 0.
function multipliers(active, scale) {
  const q = active.length, size = q + 1;
  const g = scaledGradients(active, scale);
  const M = new Float64Array(size * size), rhs = new Float64Array(size);
  for (let k = 0; k < q; k++) {
    for (let l = 0; l < q; l++) M[k * size + l] = g[k].reduce((acc, x, j) => acc + x * g[l][j], 0) + (k === l ? 1e-12 : 0);
    M[k * size + q] = M[q * size + k] = 1;
  }
  rhs[q] = 1;
  const x = solveDense(M, rhs, size);
  const lambda = Array.from({ length: q }, (_, k) => Math.max(0, x ? x[k] : 1));
  const sum = lambda.reduce((acc, v) => acc + v, 0);
  return sum > 0 ? lambda.map((v) => v / sum) : lambda.map(() => 1 / q);
}

// Newton step on the KKT system of min δ subject to c_k(p) = δ on the active set:
//   c_k(p) = δ,  Σ λ_k ∇c_k = 0,  Σ λ_k = 1,
// with the reduced Hessian ∇²c_k = −σW (K_pp − K_ps K_psᵀ / K_ss) at moving interior extrema.
// ν·scale² on the diagonal is Levenberg damping in column-scaled units.
function newtonStep(p, live, active, lambda, delta, nu, scale) {
  const m = live.length, q = active.length, size = m + 1 + q;
  const M = new Float64Array(size * size), rhs = new Float64Array(size);
  const g = active.map(({ d, sigma, weight }) => Float64Array.from(d.Kp, (x) => -sigma * weight * x));
  active.forEach(({ d, sigma, weight, target }, k) => {
    M.set(g[k], k * size);
    M[k * size + m] = -1;
    rhs[k] = delta - sigma * weight * (target - d.K);
  });
  for (let j = 0; j < m; j++) {
    const row = (q + j) * size;
    for (let l = 0; l < m; l++) {
      let h = j === l ? nu * scale[j] * scale[j] : 0;
      active.forEach(({ d, sigma, weight, interior }, k) => {
        const motion = interior ? (d.Kps[j] * d.Kps[l]) / d.Kss : 0;
        h -= lambda[k] * sigma * weight * (d.Kpp[j * m + l] - motion);
      });
      M[row + l] = h;
    }
    let r = 0;
    for (let k = 0; k < q; k++) {
      M[row + m + 1 + k] = g[k][j];
      r += lambda[k] * g[k][j];
    }
    rhs[q + j] = -r;
  }
  const last = (q + m) * size;
  for (let k = 0; k < q; k++) M[last + m + 1 + k] = 1;
  rhs[q + m] = 1 - lambda.reduce((a, x) => a + x, 0);
  const x = solveDense(M, rhs, size);
  if (!x) return null;
  const next = Array.from(p);
  live.forEach((j, a) => (next[j] += x[a]));
  return { params: next, lambda: lambda.map((l, k) => l + x[m + 1 + k]), delta: delta + x[m] };
}

// Nonlinear minimax refinement of all kernel parameters, as a generator yielding after each
// iteration. Start: exchange-optimal weights for the given rates. Each iteration then tries
//  1. a trust-region Gauss–Newton step (Osborne–Watson): the linearized minimax problem in new
//     weights and rate increments, solved by exchange, with rows |μ ẑ_i| ≤ δ bounding each
//     rate's scaled change (warm-started reference);
//  2. second-order Remez steps: Newton on the KKT system of the near-active extrema (degenerate
//     alternation allowed), points with negative multipliers dropped, Levenberg damping, and a
//     second-order correction back onto the equal-ripple manifold;
// and keeps the better if it lowers the grid maximum error.
export function* remezSteps(params, grid, { iterations = 50, minDecay = 0.05, newtonSteps = 8 } = {}) {
  const s = Float64Array.from(grid.x, (x) => x * x);
  const n = s.length;
  const target = Float64Array.from(grid.target);
  const negativeWeight = grid.negativeWeight ?? 1;
  const { wPlus, wMinus } = errorWeights(grid);
  const { linear, rates } = layout(params);
  const live = [...linear, ...rates];
  const spec = derivativeSpec(live);
  const ext = (a, fill) => Float64Array.from({ length: n + rates.length }, (_, j) => (j < n ? a[j] : fill));
  const tAll = ext(target, 0), wpAll = ext(wPlus, 1), wmAll = ext(wMinus, 1);
  const passSamples = grid.passSamples ?? n;
  const bandOf = (j) => {
    const [a, b] = j < passSamples ? [0, passSamples - 1] : [passSamples, n - 1];
    const step = 2 * Math.max(s[Math.min(j + 1, b)] - s[j], s[j] - s[Math.max(j - 1, a)]);
    return { lo: s[a], hi: s[b], step };
  };
  const valid = (q) => q.every(Number.isFinite) && rates.every((j, i) => i % 2 === 1 || q[j] >= minDecay);
  // Same weighting as problem.js maxError.
  const errorsOf = (K) => Float64Array.from(K, (k, j) => Math.abs(k - target[j]) * (k < 0 ? negativeWeight : 1));
  const cost = (q) => (valid(q) ? errorsOf(gridProfile(q, s)).reduce((a, x) => Math.max(a, x), 0) : Infinity);

  // Band-local maxima of the weighted error within a factor (1 − κ) of the maximum.
  const nearActive = (q, kappa = 0.05) => {
    const K = gridProfile(q, s);
    const a = errorsOf(K);
    const top = a.reduce((x, y) => Math.max(x, y), 0);
    const out = [];
    for (let j = 0; j < n; j++) {
      const first = j === 0 || j === passSamples, last = j === n - 1 || j === passSamples - 1;
      if (a[j] < (1 - kappa) * top || (!first && a[j - 1] > a[j]) || (!last && a[j + 1] >= a[j])) continue;
      const sigma = target[j] > K[j] ? 1 : -1;
      out.push({ j, sigma, weight: sigma > 0 ? wPlus[j] : wMinus[j], target: target[j] });
    }
    return out;
  };

  // Newton with active-set dropping, then up to three minimum-norm corrections c_k = δ.
  const activeSetNewton = (q, active, lambda, delta, nu, scale) => {
    let set = active, lam = lambda, t = null;
    for (let drop = 0; drop < active.length; drop++) {
      t = newtonStep(q, live, set, lam, delta, nu, scale);
      if (!t || set.length <= 1) break;
      let worst = 0;
      t.lambda.forEach((v, k) => v < t.lambda[worst] && (worst = k));
      if (t.lambda[worst] >= 0) break;
      set = set.filter((_, k) => k !== worst);
      lam = lam.filter((_, k) => k !== worst);
      const sum = lam.reduce((acc, v) => acc + v, 0);
      lam = sum > 0 ? lam.map((v) => v / sum) : lam.map(() => 1 / lam.length);
    }
    if (!t) return null;
    let next = t.params, level = t.delta;
    for (let pass = 0; pass < 3 && valid(next); pass++) {
      const at = set.map((a) => ({ ...a, ...locateExtremum(next, spec, a.s, bandOf(a.j), a.sigma) }));
      const r = at.map(({ d, sigma, weight, target: tk }) => level - sigma * weight * (tk - d.K));
      if (Math.max(...r.map(Math.abs)) <= 1e-15) break;
      const A = scaledGradients(at, scale).map((g) => [...g, -1]);
      const k = A.length;
      const AAt = new Float64Array(k * k);
      for (let i = 0; i < k; i++) for (let l = 0; l < k; l++) AAt[i * k + l] = A[i].reduce((acc, x, j) => acc + x * A[l][j], 0);
      const y = solveDense(AAt, r, k);
      if (!y) break;
      next = next.slice();
      live.forEach((j, a) => (next[j] += A.reduce((acc, row, i) => acc + row[a] * y[i], 0) / scale[a]));
      level += A.reduce((acc, row, i) => acc + row[live.length] * y[i], 0);
    }
    return { params: next, count: set.length };
  };

  let p = Array.from(params);
  let error = cost(p);
  let exchanges = 0;
  const first = optimalWeights(p, grid);
  exchanges += first.exchanges;
  if (first.error < error) [p, error] = [first.params, first.error];
  yield { iteration: 0, params: p, error, exchanges, method: "weights" };

  let mu = 1, nu = 1e-6, reference = null;
  for (let it = 1; it <= iterations; it++) {
    const { rows, scale } = scaledRows(columns(p, s, live), n, rates.length);
    rates.forEach((_, i) => (rows[n + i][linear.length + i] = mu));
    const sol = chebyshev(rows, tAll, wpAll, wmAll, { reference });
    exchanges += sol.iterations;
    const predicted = error - sol.delta;
    const trial = Array.from(p);
    linear.forEach((j, c) => (trial[j] = sol.z[c] / scale[c]));
    rates.forEach((j, i) => (trial[j] += sol.z[linear.length + i] / scale[linear.length + i]));
    const trialError = cost(trial);
    const rho = (error - trialError) / Math.max(predicted, 1e-300);
    let best = { params: trial, error: trialError, method: "gauss-newton" };

    let q = p, qError = error;
    for (let k = 0; k < newtonSteps; k++) {
      const located = nearActive(q).map((a) => ({ ...a, ...locateExtremum(q, spec, s[a.j], bandOf(a.j), a.sigma) }));
      const lambda = multipliers(located, scale);
      let step = null, trialNu = nu;
      for (let tries = 0; tries < 6 && !step; tries++) {
        const t = activeSetNewton(q, located, lambda, qError, trialNu, scale);
        const e = t ? cost(t.params) : Infinity;
        if (e < qError) step = { params: t.params, error: e };
        else trialNu *= 8;
      }
      if (!step) {
        nu = Math.min(nu * 4, 1e6);
        break;
      }
      nu = Math.min(Math.max(trialNu / 8, 1e-12), 1e6);
      [q, qError] = [step.params, step.error];
    }
    if (qError < best.error) best = { params: q, error: qError, method: "newton" };

    const accepted = best.error < error;
    if (!(trialError < error)) mu *= 4;
    else if (rho > 0.75) mu = Math.max(mu / 4, 1e-6);
    else if (rho < 0.25) mu *= 2;
    if (accepted) [p, error, reference] = [best.params, best.error, sol.reference];
    yield { iteration: it, params: p, error, exchanges, method: accepted ? best.method : null };
    if (!accepted && (predicted <= 1e-9 * error || mu > 1e8)) return;
  }
}

// Runs remezSteps to completion.
export function remezPolish(params, grid, options) {
  let last = null;
  const history = [];
  for (const step of remezSteps(params, grid, options)) {
    history.push({ iteration: step.iteration, error: step.error, method: step.method });
    last = step;
  }
  return { params: last.params, error: maxError(last.params, grid), history, exchanges: last.exchanges };
}

