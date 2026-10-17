// Generation-synchronous differential evolution on jax-js, GPU-friendly:
// one jitted call per generation, population and costs stay on the device.
//
// Population management follows hamk-uas/logistics_sim ga.h: every member is a
// target once per generation; a fresh random permutation σ assigns donors so that
// target σ[k] uses σ[k+1], σ[k+2], σ[k+3] (mod P) — four distinct members, each used
// once per role per generation. Children replace their own target only if strictly
// better. Because the population is an unordered set, it is simply kept in σ order.

import { jit, numpy as np, random } from "@hamk-uas/jax-js-nonconsuming";
import { profileBatch } from "./model.js";
import { PARAMS_PER_COMPONENT } from "./problem.js";

// Max |profile − target| over the sample grid for each parameter vector in a batch.
export function batchCost(params, s2, target, components) {
  const err = np.abs(profileBatch(params, s2, components).sub(target.reshape([1, -1])));
  return err.max(1);
}

// Reflect once at each bound, then clamp (the article's constraint handling).
function fold(x, lo, hi) {
  const below = np.where(np.less(x, lo), lo.mul(2).sub(x), x);
  const above = np.where(np.greater(below, hi), hi.mul(2).sub(below), below);
  return np.clip(above, lo, hi);
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class DifferentialEvolution {
  // grid: { x, target } sample points; bounds: { foldLo, foldHi, initLo, initHi }.
  constructor({ components, grid, bounds, populationSize, F, CR, seed, startingPoint }) {
    const P = populationSize;
    const D = components * PARAMS_PER_COMPONENT;
    if (P < 4) throw new Error("Population size must be at least 4.");
    this.P = P;
    this.D = D;
    this.generation = 0;

    const s2 = Float32Array.from(grid.x, (x) => x * x);
    this.s2 = np.array(s2);
    this.target = np.array(Float32Array.from(grid.target));
    this.lo = np.array(Float32Array.from(bounds.foldLo)).reshape([1, D]);
    this.hi = np.array(Float32Array.from(bounds.foldHi)).reshape([1, D]);
    const roll = (k) => np.array(Int32Array.from({ length: P }, (_, i) => (i + k) % P), { dtype: np.int32 });
    this.roll = [roll(1), roll(2), roll(3)];
    this.positions = np.array(Float32Array.from({ length: D }, (_, j) => j)).reshape([1, D]);

    const rand = mulberry32(seed);
    const init = new Float32Array(P * D);
    for (let k = 0; k < P; k++) {
      for (let j = 0; j < D; j++) {
        init[k * D + j] = bounds.initLo[j] + (bounds.initHi[j] - bounds.initLo[j]) * rand();
      }
    }
    if (startingPoint && startingPoint.length === D) init.set(startingPoint, 0);

    this.costFn = jit((pop, s2v, tv) => batchCost(pop, s2v, tv, components));
    {
      using foldFn = jit(fold);
      using raw = np.array(init).reshape([P, D]);
      this.pop = foldFn(raw, this.lo, this.hi);
    }
    this.cost = this.costFn(this.pop, this.s2, this.target);
    this.key = random.key(seed >>> 0);

    const logCR = CR < 1 ? Math.log(CR) : 0;
    this.stepFn = jit((pop, cost, key, s2v, tv, lo, hi, r1, r2, r3, positions) => {
      const keys = random.split(key, 4);
      const sub = (i) => {
        using idx = np.array([i], { dtype: np.int32 });
        return np.take(keys, idx, 0).reshape([2]);
      };
      const perm = np.argsort(random.uniform(sub(0), [P]));
      const targets = np.take(pop, perm, 0);
      const targetCost = np.take(cost, perm, 0);
      const a = np.take(targets, r1, 0);
      const b = np.take(targets, r2, 0);
      const c = np.take(targets, r3, 0);
      const mutant = a.add(b.sub(c).mul(F));

      // Exponential crossover: a contiguous (cyclic) run of mutated genes starting at a
      // random position, continuing with probability CR per extra gene.
      const start = np.floor(random.uniform(sub(1), [P, 1]).mul(D));
      const length =
        CR < 1
          ? np.minimum(np.floor(np.log(random.uniform(sub(2), [P, 1]).neg().add(1)).div(logCR)).add(1), D)
          : np.fullLike(start, D);
      const offset = np.remainder(positions.sub(start).add(D), D);
      const trial = fold(np.where(np.less(offset, length), mutant, targets), lo, hi);

      const trialCost = batchCost(trial, s2v, tv, components);
      const better = np.less(trialCost, targetCost);
      const nextPop = np.where(better.reshape([P, 1]), trial, targets);
      const nextCost = np.where(better, trialCost, targetCost);
      return [nextPop, nextCost, sub(3)];
    });
  }

  // Enqueue generations without reading back.
  step(generations = 1) {
    for (let g = 0; g < generations; g++) {
      const [pop, cost, key] = this.stepFn(
        this.pop, this.cost, this.key, this.s2, this.target, this.lo, this.hi,
        this.roll[0], this.roll[1], this.roll[2], this.positions,
      );
      this.pop.dispose();
      this.cost.dispose();
      this.key.dispose();
      this.pop = pop;
      this.cost = cost;
      this.key = key;
      this.generation++;
    }
  }

  async snapshot() {
    const [population, costs] = await Promise.all([this.pop.data(), this.cost.data()]);
    let best = 0;
    let sum = 0;
    for (let k = 0; k < costs.length; k++) {
      sum += costs[k];
      if (costs[k] < costs[best]) best = k;
    }
    return {
      population,
      costs,
      best: Array.from(population.subarray(best * this.D, (best + 1) * this.D)),
      bestCost: costs[best],
      meanCost: sum / costs.length,
    };
  }

  dispose() {
    for (const a of [this.pop, this.cost, this.key, this.s2, this.target, this.lo, this.hi, this.positions, ...this.roll]) {
      a.dispose();
    }
    this.costFn.dispose();
    this.stepFn.dispose();
  }
}
