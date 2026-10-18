// Node smoke test (Wasm backend): kernel math, separable channels, DE, fine-tuning.
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { DifferentialEvolution } from "../src/de.js";
import { tapTables } from "../src/blur.js";
import { continuousKernel } from "../src/continuous-kernel.js";
import { FineTuner } from "../src/finetune.js";
import {
  bounds, defaultProblem, maxError, multiplicities, nonnegativePresetParams, parseParams, presetParams, profile, sampleGrid,
  separableChannels,
} from "../src/problem.js";

const devices = await init();
defaultDevice(devices.includes("wasm") ? "wasm" : "cpu");

function check(name, value, tolerance) {
  console.log(`${value <= tolerance ? "PASS" : "FAIL"} ${name}: ${value.toExponential(2)} (≤ ${tolerance})`);
  if (value > tolerance) process.exitCode = 1;
}

const problemFor = (multiplicity) => ({ ...defaultProblem(), components: multiplicity.length, multiplicity });

// The new radius sampler is continuous, including its subpixel identity and the point at which
// the small-radius correction fades out. Check all preset channel sums, not merely normalized RGB:
// incorrect DC weight would show up as incorrect opacity when layers are enabled.
{
  let worst = 0, identity = 0, symmetry = 0, jump = 0;
  const sets = [1, 2, 3, 4, 5, 6].map(n => presetParams(n));
  sets.push([3, 0, 1, 0, 1, 0]); // polynomial/merged channel, including a zero constant coefficient
  for (const params of sets) {
    const channels = separableChannels(params);
    const c = continuousKernel(channels, 0.2, 128);
    const rs = [0, 0.01, 0.2, 0.51, 0.9438, 0.99, 1, 1.02, 1.17, 1.8, 2.13, 3.7, 4, 5.3, 7.99, 8, 9, 15, 32, 69, 120,
      ...Array.from({ length: 24 }, (_, i) => 2 ** (-4 + i * 11 / 23))];
    const sums = rs.map(r => [0, 1].map(dir => channels.map((_, ch) => {
      const sum = [0, 0];
      const half = r === 0 ? 0 : Math.ceil(Math.max(r, 1) * c.support);
      for (let t = -half; t <= half; t++) {
        const z = c.tap(r, dir, ch, t), opposite = c.tap(r, dir, ch, -t);
        for (let k = 0; k < 2; k++) {
          sum[k] += z[k];
          symmetry = Math.max(symmetry, Math.abs(z[k] - opposite[k]));
        }
      }
      return sum;
    })));
    for (let i = 0; i < rs.length; i++) for (let j = 0; j < rs.length; j++) {
      const dc = sums[i][0].reduce((s, h, ch) => s + h[0] * sums[j][1][ch][0] - h[1] * sums[j][1][ch][1], 0);
      worst = Math.max(worst, Math.abs(dc * c.correction(rs[i], rs[j], true) - 1));
      if (i === 0 && j === 0) identity = Math.max(identity, Math.abs(dc - 1));
    }
    for (const boundary of [0.943874, 1, 4, c.smallRadius, 2 * c.smallRadius]) {
      for (let t = -3; t <= 3; t++) for (let ch = 0; ch < channels.length; ch++) {
        const a = c.tap(boundary - 1e-7, 0, ch, t), b = c.tap(boundary + 1e-7, 0, ch, t);
        jump = Math.max(jump, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
      }
      jump = Math.max(jump, Math.abs(c.correction(1.17, boundary - 1e-7, true) - c.correction(1.17, boundary + 1e-7, true)));
    }
  }
  check("continuous mixed-radius unit coverage, including fast path", worst, 0.005);
  check("continuous zero radius is identity", identity, 1e-10);
  check("interpolated profile symmetry", symmetry, 0);
  check("continuous sampler has no threshold jumps", jump, 1e-5);
}

// Varying-radius tap tables: a source's horizontal taps at one level followed by vertical taps at
// another weigh C[l1][l2]; it must stay well away from 0. The tables are scaled so that every mix
// has unit weight within tolerance, except at a small vertical level, where the renderer applies
// fix[l1][l2]. The identity level must stay exact (a single center tap, 2-D weight 1).
{
  const radii = [null, ...Array.from({ length: 37 }, (_, l) => 2 ** (l / 6))];
  const tapSum = (taps) => {
    let re = 0, im = 0;
    for (let i = 0; i < taps.length; i += 2) { re += taps[i]; im += taps[i + 1]; }
    return [re, im];
  };
  let worstCross = 0;
  let worstMix = 0;
  let worstIdentity = 0;
  for (const n of [1, 2, 3, 4, 5, 6]) {
    const { tables, cross, fix, small, halves } = tapTables(separableChannels(presetParams(n)), radii, 0.2);
    for (const row of cross) for (const c of row) worstCross = Math.max(worstCross, Math.abs(Math.log(c)));
    const sums = tables.map((byDir) => byDir.map((byChannel) => byChannel.map(tapSum)));
    tables.forEach((_, ls) => tables.forEach((_, lv) => {
      let w = 0;
      sums[ls][0].forEach(([hr, hi], ch) => { const [vr, vi] = sums[lv][1][ch]; w += hr * vr - hi * vi; });
      if (lv >= 1 && lv < small) w *= fix[ls][lv];
      worstMix = Math.max(worstMix, Math.abs(Math.log(w)));
    }));
    worstIdentity = Math.max(worstIdentity, halves[0]);
    if (tables[0][0][0].length !== 2) worstIdentity = Infinity;
  }
  check("varying-radius levels mix with raw weights within a factor of 1.5 of 1 (|log|)", worstCross, Math.log(1.5));
  check("varying-radius mixes, scaled and fixed, have unit weight (|log|)", worstMix, 0.0051);
  check("identity is a single tap", worstIdentity, 0);
}

// A merged pair is the limit of two coinciding rates: (B + A s) e^{λs} = lim Re[B e^{λs} + (A/δ)(e^{(λ+δ)s} − e^{λs})].
{
  const [b, c, r, i, r1, i1] = [2.1, 6.3, 0.7, -1.4, 3.2, 0.9];
  const merged = [b, c, r, i, r1, i1];
  const d = 1e-5;
  let worst = 0;
  for (const x of [0, 0.3, 0.9, 1.4, 2]) {
    const s = x * x;
    const term = (bb) => [Math.exp(-bb * s) * Math.cos(c * s), Math.exp(-bb * s) * Math.sin(c * s)];
    const [e0r, e0i] = term(b);
    const [edr, edi] = term(b - d); // rate λ + δ with δ = d (real)
    const limit = r * e0r + i * e0i + (r1 * (edr - e0r) + i1 * (edi - e0i)) / d;
    worst = Math.max(worst, Math.abs(limit - profile(merged, x)));
  }
  check("merged pair equals the coinciding-rate limit", worst, 1e-3);
}

// Separable channels reproduce the 2-D kernel exactly: Re Σ h(x) v(y) = profile(√(x² + y²)).
{
  const params = [...presetParams(2), 3.1, 9.4, -0.8, 2.2, 1.7, -0.6];
  const channels = separableChannels(params);
  const evalK = ({ decay, phase }, [k0r, k0i], [k2r, k2i], t) => {
    const s = t * t;
    const e = Math.exp(-decay * s);
    const [er, ei] = [e * Math.cos(phase * s), e * Math.sin(phase * s)];
    const [wr, wi] = [k0r + k2r * s, k0i + k2i * s];
    return [wr * er - wi * ei, wr * ei + wi * er];
  };
  let worst = 0;
  for (const [x, y] of [[0, 0], [0.4, -0.7], [1.1, 0.2], [-0.9, 1.3]]) {
    let sum = 0;
    for (const ch of channels) {
      const [hr, hi] = evalK(ch, ch.h0, ch.h2, x);
      const [vr, vi] = evalK(ch, ch.v0, ch.v2, y);
      sum += hr * vr - hi * vi;
    }
    worst = Math.max(worst, Math.abs(sum - profile(params, Math.hypot(x, y))));
  }
  check(`separable channels (${channels.length} for multiplicities ${multiplicities(params)})`, worst, 1e-12);
  const reparsed = parseParams("0.886528, 5.268909, 0.411259, -0.548794\n3.1, 9.4, -0.8, 2.2, 1.7, -0.6");
  check("parse one component per line", Math.abs(profile(reparsed, 0.7) - profile([...params.slice(0, 6), ...params.slice(12)], 0.7)), 1e-12);
}

// jax-js batch cost agrees with the double-precision JS cost.
{
  const problem = problemFor([1, 1, 1, 1, 1]);
  const grid = sampleGrid(problem);
  const de = new DifferentialEvolution({
    components: 5, grid, bounds: bounds(problem), populationSize: 8, F: 0.76, CR: 0.999, seed: 1,
    startingPoint: Float32Array.from(presetParams(5)),
  });
  const snap = await de.snapshot();
  check("jax-js cost of preset 5 vs JS", Math.abs(snap.costs[0] - maxError(presetParams(5), grid)), 2e-5);
  de.dispose();
}

// The article's non-negative sets: converted from its (a, b, c, d) form, they match its stated
// stop-band maxima, and the weighted jax-js cost matches the JS one.
{
  const nnProblem = { ...problemFor([1, 1]), frequencyBands: false, negativeWeight: 100 };
  const grid = sampleGrid(nnProblem);
  for (const [n, stated] of [[1, 0.3079699786], [2, 0.1422940279]]) {
    const p = nonnegativePresetParams(n);
    let stop = 0;
    for (let x = 1.2; x < 4; x += 0.001) stop = Math.max(stop, Math.abs(profile(p, x)));
    check(`non-negative preset ${n}: stop-band maximum vs article`, Math.abs(stop - stated) / stated, 0.01);
  }
  const p = nonnegativePresetParams(2);
  const de = new DifferentialEvolution({
    components: 2, grid, bounds: bounds(nnProblem), populationSize: 8, F: 0.76, CR: 0.999, seed: 1,
    startingPoint: Float32Array.from(p),
  });
  const snap = await de.snapshot();
  check("jax-js negative-weighted cost vs JS", Math.abs(snap.costs[0] - maxError(p, grid)), 2e-5);
  de.dispose();
}

// DE with merged pairs: single components keep zero x² weights, costs match JS.
for (const multiplicity of [[1, 1], [2, 1]]) {
  const problem = problemFor(multiplicity);
  const grid = sampleGrid(problem);
  const de = new DifferentialEvolution({
    components: multiplicity.length, grid, bounds: bounds(problem), populationSize: 256, F: 0.76, CR: 0.999, seed: 7,
  });
  de.step(400);
  const snap = await de.snapshot();
  console.log(`DE multiplicities ${multiplicity}: generation ${de.generation}, best ±${snap.bestCost.toFixed(6)}`);
  check(`  JS cost of DE best (${multiplicity})`, Math.abs(snap.bestCost - maxError(snap.best, grid)), 2e-5);
  const pinned = multiplicity.flatMap((m, u) => (m === 1 ? [snap.best[u * 6 + 4], snap.best[u * 6 + 5]] : []));
  check(`  single components' x² weights stay zero (${multiplicity})`, Math.max(0, ...pinned.map(Math.abs)), 0);
  de.dispose();
}

// Fine-tuning improves a perturbed 3-component preset and keeps unused x² weights at zero.
{
  const problem = problemFor([1, 1, 1]);
  const dense = sampleGrid(problem, 1000, 1000);
  const start = presetParams(3).map((v, j) => v * (1 + 0.01 * Math.sin(j * 1.7)));
  const tuner = new FineTuner({ params: start, components: 3, grid: dense, learningRate: 3e-4, halfLife: 400 });
  let err = maxError(start, dense);
  const startErr = err;
  let params = start;
  for (let round = 0; round < 12; round++) {
    const r = await tuner.run(100, 100 * Math.pow(10, round / 11), err);
    if (r.error < err) [err, params] = [r.error, r.params];
  }
  tuner.dispose();
  console.log(`fine-tune: ±${startErr.toFixed(6)} → ±${err.toFixed(6)} (article preset ±${maxError(presetParams(3), dense).toFixed(6)})`);
  check("fine-tune improves", err < startErr ? 0 : 1, 0);
  check("fine-tune keeps x² weights of single components zero", Math.max(...[4, 5, 10, 11, 16, 17].map((j) => Math.abs(params[j]))), 0);
}
