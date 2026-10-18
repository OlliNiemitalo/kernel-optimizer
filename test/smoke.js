// Node smoke test (Wasm backend): kernel math, separable channels, DE, fine-tuning.
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { DifferentialEvolution, initialPopulation } from "../src/de.js";
import assert from "node:assert/strict";
import { tapTables } from "../src/blur.js";
import { continuousKernel, integratedKernel, kernelProfile } from "../src/continuous-kernel.js";
import { radiusTables } from "../src/radius-tables.js";
import { FineTuner } from "../src/finetune.js";
import { testViewControls } from './view-controls.js';
import {
  bounds, defaultProblem, maxError, multiplicities, nonnegativePresetParams, parseParams, presetParams, profile, sampleGrid,
  separableChannels,
} from "../src/problem.js";

const devices = await init();
defaultDevice(devices.includes("wasm") ? "wasm" : "cpu");

function check(name, value, tolerance) {
  const pass = Number.isFinite(value) && value <= tolerance;
  console.log(`${pass ? "PASS" : "FAIL"} ${name}: ${value.toExponential(2)} (≤ ${tolerance})`);
  if (!pass) process.exitCode = 1;
}

const problemFor = (multiplicity) => ({ ...defaultProblem(), components: multiplicity.length, multiplicity });

testViewControls();
console.log('PASS view-dependent control availability and diagnostic selection');

// Test whole-population local jitter, including centers outside the global initialization box.
for (const multiplicity of [[1], [2]]) {
  const problem = { ...problemFor(multiplicity), frequencyBands: false, passSamples: 32, stopSamples: 32 };
  const limits = bounds(problem);
  const center = [3, 2, 12, -8, multiplicity[0] === 2 ? 1 : 0, 0];
  const options = { bounds: limits, populationSize: 32, dimensions: 6, seed: 123, center };
  const population = initialPopulation(options);
  assert.deepEqual(population, initialPopulation(options));
  assert.notDeepEqual(population, initialPopulation({ ...options, seed: 124 }));
  assert.deepEqual(Array.from(population.slice(0, 6)), center);
  const withoutCenter = initialPopulation({ ...options, includeCenter: false });
  assert.notDeepEqual(Array.from(withoutCenter.slice(0, 6)), center);
  for (let k = 1; k < 32; k++) {
    assert.ok(population.slice(k * 6, (k + 1) * 6).some((v, j) => v !== center[j]));
    for (let j = 0; j < 6; j++) {
      const v = population[k * 6 + j];
      const radius = Math.min(0.05 * (limits.initHi[j] - limits.initLo[j]),
        center[j] - limits.foldLo[j], limits.foldHi[j] - center[j]);
      assert.ok(Math.abs(v - center[j]) <= radius + 1e-6);
      assert.ok(v >= limits.foldLo[j] && v <= limits.foldHi[j]);
    }
  }
  const edge = initialPopulation({ ...options, center: [3, 0, 12, -8, 0, 0] });
  assert.equal(edge[7], 0);
  assert.throws(() => initialPopulation({ ...options, center: [NaN, 2, 12, -8, 0, 0] }));
  assert.throws(() => initialPopulation({ ...options, center: [0, 2, 12, -8, 0, 0] }));
  assert.throws(() => initialPopulation({ ...options, center: [1] }));
  if (multiplicity[0] === 1) assert.throws(() => initialPopulation({ ...options, center: [3, 2, 12, -8, 1, 0] }));
  const de = new DifferentialEvolution({ components: 1, grid: sampleGrid(problem), bounds: limits,
    populationSize: 32, F: 0.5, CR: 0.9, seed: 123, populationCenter: center });
  try {
    const snap = await de.snapshot();
    assert.deepEqual(snap.population, population);
    assert.ok(Array.from(snap.costs).every(Number.isFinite));
    de.step();
    assert.ok((await de.snapshot()).bestCost <= snap.bestCost);
    for (const gain of [5, 30, 1]) {
      const before = await de.snapshot();
      const key = Array.from(await de.key.data());
      const generation = de.generation;
      const changedProblem = { ...problem, transition: 0.35, passSamples: 40, stopSamples: 50, negativeWeight: gain };
      const grid = sampleGrid(changedProblem);
      const newBounds = bounds({ ...changedProblem, frequencyBands: true });
      de.updateProblem(grid, newBounds);
      const after = await de.snapshot();
      assert.deepEqual(after.population, before.population);
      assert.equal(de.generation, generation);
      assert.deepEqual(Array.from(await de.key.data()), key);
      const verifyCosts = (snapshot) => {
        for (let k = 0; k < de.P; k++) {
          const expected = maxError(Array.from(snapshot.population.slice(k * de.D, (k + 1) * de.D)), grid);
          assert.ok(Math.abs(snapshot.costs[k] - expected) < 1e-4 * Math.max(1, expected));
        }
      };
      verifyCosts(after);
      de.inject(center);
      verifyCosts(await de.snapshot());
      for (let step = 0; step < 3; step++) {
        de.step();
        verifyCosts(await de.snapshot());
      }
    }
  } finally {
    de.dispose();
  }
}
console.log('PASS centered population, optional exact seed, objective re-scoring, retained population/RNG/generation and updated DE/injection costs');

{
  let dcError = 0, channelError = 0, floatChannelError = 0, quadratureError = 0, integratedError = 0, continuity = 0;
  const sets = [1, 2, 3, 4, 5, 6].map(presetParams);
  sets.push([3, 0, 1, 0, 1, 0]); // merged polynomial channel: no division by a small channel sum
  for (const params of sets) {
    const channels = separableChannels(params);
    const tables = radiusTables(channels, 0.2, 128);
    const reference = integratedKernel(channels, 0.2);
    const profile = kernelProfile(channels, 0.2);
    const rs = [0, 0.001, 0.17, 0.41, 0.43, 0.51, 0.94387, 1, 1.02, 1.17, 1.8, 2.13, 3.7, 4, 5.3, 8.05, 15, 32, 69, 120];
    // Sum every stored node independently. Interpolating vectors with these fixed sums must
    // preserve them too. Also check the actual f32 upload precision, not only CPU doubles.
    for (const node of tables.tables) for (const dir of [0, 1]) for (let ch = 0; ch < channels.length; ch++) {
      const taps = node[dir][ch];
      for (let k = 0; k < 2; k++) {
        let sum = taps[k], floatSum = Math.fround(taps[k]);
        for (let t = 1; t < tables.stride; t++) {
          sum += 2 * taps[2 * t + k];
          floatSum += 2 * Math.fround(taps[2 * t + k]);
        }
        channelError = Math.max(channelError, Math.abs(sum - reference.identity[dir][ch][k]));
        floatChannelError = Math.max(floatChannelError, Math.abs(floatSum - reference.identity[dir][ch][k]));
      }
    }
    // Independent cell integration: clip every linear segment against the pixel interval and
    // sum trapezoids directly, without using the production cumulative primitive.
    for (const r of rs.filter(r => r > 0)) for (const dir of [0, 1]) for (let ch = 0; ch < channels.length; ch++) {
      const p = profile.profiles[dir][ch], last = p.length / 2 - 1;
      for (const t of [0, 1, 2, Math.floor(r * profile.support)]) {
        const actual = reference.tap(r, dir, ch, t);
        for (let k = 0; k < 2; k++) {
          let expected = 0;
          const lo = (t - 0.5) * profile.density / r, hi = (t + 0.5) * profile.density / r;
          for (let i = -last; i < last; i++) {
            const a = Math.max(lo, i), b = Math.min(hi, i + 1);
            if (b <= a) continue;
            const u = p[2 * Math.abs(i) + k], v = p[2 * Math.abs(i + 1) + k];
            expected += (b - a) * (u + (v - u) * ((a + b) / 2 - i));
          }
          quadratureError = Math.max(quadratureError, Math.abs(actual[k] - expected));
        }
      }
    }
    const sums = rs.map(r => [0, 1].map(dir => channels.map((_, ch) => {
      const sum = [0, 0];
      for (let t = -tables.half; t <= tables.half; t++) {
        const z = tables.tap(r, dir, ch, t);
        sum[0] += z[0]; sum[1] += z[1];
      }
      return sum;
    })));
    for (let i = 0; i < rs.length; i++) for (let j = 0; j < rs.length; j++) {
      const w = sums[i][0].reduce((s, h, ch) => s + h[0] * sums[j][1][ch][0] - h[1] * sums[j][1][ch][1], 0);
      dcError = Math.max(dcError, Math.abs(w - 1));
    }
    // Radius interpolation error against direct pixel integration (not the old point samples).
    for (const r of rs) {
      const half = tables.halves[tables.coordinate(r)[0] + 1];
      const tap = (model, dir) => channels.map((_, ch) => Array.from({ length: 2 * half + 1 }, (_, t) => model.tap(r, dir, ch, t - half)));
      const h = tap(tables, 0), v = tap(tables, 1), rh = tap(reference, 0), rv = tap(reference, 1);
      let l1 = 0;
      for (let x = 0; x <= 2 * half; x++) for (let y = 0; y <= 2 * half; y++) {
        const value = (a, b) => channels.reduce((s, _, ch) => s + a[ch][x][0] * b[ch][y][0] - a[ch][x][1] * b[ch][y][1], 0);
        const a = value(h, v);
        const b = value(rh, rv);
        l1 += Math.abs(a - b);
      }
      integratedError = Math.max(integratedError, l1);
    }
    for (const r of tables.radii.slice(1)) for (let ch = 0; ch < channels.length; ch++) {
      const a = tables.tap(r - 1e-7, 0, ch, 1), b = tables.tap(r + 1e-7, 0, ch, 1);
      continuity = Math.max(continuity, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
    }
  }
  check('radius-table mixed DC without correction', dcError, 1e-10);
  check('every node preserves complex channel sums', channelError, 1e-11);
  check('uploaded f32 taps preserve complex channel sums', floatChannelError, 2e-6);
  check('pixel integrals versus independent segment quadrature', quadratureError, 1e-11);
  check('radius interpolation integrated absolute kernel error', integratedError, 0.01);
  check('radius-table continuity at every node', continuity, 1e-5);
}

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
