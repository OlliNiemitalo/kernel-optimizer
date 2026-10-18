// Differential evolution over decays and frequencies only (variable projection): every candidate's
// weights are the exact minimax optimum for its rates, solved by exchange in float64 on the CPU,
// in Web Workers when available. Same generation-synchronous ga.h-style donors, exponential
// crossover and fold as src/de.js. Operations are queued; snapshot() waits for them.

import { initialPopulation, mulberry32 } from "./population.js";
import { PARAMS_PER_COMPONENT, maxError } from "./problem.js";
import { WeightSolver, solveCandidate } from "./remez.js";

function fold(x, lo, hi) {
  if (x < lo) x = 2 * lo - x;
  if (x > hi) x = 2 * hi - x;
  return Math.min(hi, Math.max(lo, x));
}

// Same-thread solver backend (Node, tests).
export function localSolvers() {
  let solver = null;
  return {
    threads: 0,
    setProblem: (grid, multiplicity) => (solver = new WeightSolver(grid, multiplicity)),
    solve: async (items) => items.map((item) => solveCandidate(solver, item)),
    dispose() {},
  };
}

// Web Worker pool; each worker holds a solver for the current problem.
export function workerSolvers(count) {
  const workers = Array.from({ length: count }, () =>
    new Worker(new URL("./weight-worker.js", import.meta.url), { type: "module" }));
  const pending = new Map();
  let nextId = 0;
  for (const w of workers) {
    w.onmessage = ({ data }) => {
      pending.get(data.id)?.resolve(data.out);
      pending.delete(data.id);
    };
    w.onerror = (e) => {
      for (const p of pending.values()) p.reject(new Error(e.message || "Weight worker failed."));
      pending.clear();
    };
  }
  const call = (w, items) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    w.postMessage({ type: "solve", id, items });
  });
  return {
    threads: count,
    setProblem(grid, multiplicity) {
      const plain = { x: grid.x, target: grid.target, passSamples: grid.passSamples, negativeWeight: grid.negativeWeight };
      for (const w of workers) w.postMessage({ type: "problem", grid: plain, multiplicity });
    },
    async solve(items) {
      const size = Math.ceil(items.length / workers.length);
      const parts = await Promise.all(workers.map((w, i) => call(w, items.slice(i * size, (i + 1) * size))));
      return parts.flat();
    },
    dispose() {
      for (const w of workers) w.terminate();
    },
  };
}

export class ProjectedEvolution {
  constructor({ components, multiplicity, grid, bounds, populationSize, F, CR, seed, startingPoint,
    populationCenter = null, includeCenter = true, solvers = localSolvers() }) {
    const P = populationSize;
    const D = components * PARAMS_PER_COMPONENT;
    if (P < 4) throw new Error("Population size must be at least 4.");
    const init = initialPopulation({ bounds, populationSize: P, dimensions: D, seed, center: populationCenter, includeCenter });
    if (!populationCenter && startingPoint && startingPoint.length === D) init.set(startingPoint, 0);
    Object.assign(this, { P, D, F, CR, multiplicity, solvers, generation: 0, revision: 0, failure: null });
    this.rates = Array.from({ length: 2 * components }, (_, i) => Math.floor(i / 2) * PARAMS_PER_COMPONENT + (i % 2));
    this.rand = mulberry32((seed ^ 0x9e3779b9) >>> 0);
    this.setBounds(bounds);
    this.pop = Array.from({ length: P }, (_, k) => Array.from(init.subarray(k * D, (k + 1) * D), (v, j) => fold(v, this.lo[j], this.hi[j])));
    this.cost = new Float64Array(P).fill(Infinity);
    this.refs = new Array(P).fill(null);
    this.grid = grid;
    solvers.setProblem(grid, multiplicity);
    this.queue = Promise.resolve();
    this.enqueue(() => this.evaluate(this.pop.map((params, k) => ({ k, params })), true));
    const exact = populationCenter ? includeCenter && populationCenter : startingPoint?.length === D && startingPoint;
    if (exact) this.enqueue(() => this.place(0, exact));
  }

  setBounds(bounds) {
    this.lo = Float64Array.from(bounds.foldLo);
    this.hi = Float64Array.from(bounds.foldHi);
  }

  enqueue(task) {
    this.queue = this.queue.then(() => (this.failure ? undefined : task())).catch((e) => (this.failure = e));
  }

  // Solves the candidates' weights; each replaces member k if forced or better.
  async evaluate(candidates, force = false) {
    const out = await this.solvers.solve(candidates.map(({ k, params }) => ({ params, reference: force ? null : this.refs[k] })));
    candidates.forEach(({ k }, i) => {
      if (force || out[i].cost < this.cost[k]) {
        this.pop[k] = out[i].params;
        this.cost[k] = out[i].cost;
        this.refs[k] = out[i].reference;
      }
    });
  }

  // Re-solves every member's weights under the new objective; rates are kept.
  updateProblem(grid, bounds) {
    this.revision++;
    this.enqueue(() => {
      this.grid = grid;
      this.solvers.setProblem(grid, this.multiplicity);
      this.setBounds(bounds);
      return this.evaluate(this.pop.map((params, k) => ({ k, params })), true);
    });
  }

  // Sets member k to a given kernel. Its own weights are kept (they may be tuned on a denser grid
  // than the DE grid); they are re-solved only if folding into the search bounds moved its rates.
  place(k, params) {
    const folded = Array.from(params, (v, j) => fold(v, this.lo[j], this.hi[j]));
    if (folded.some((v, j) => v !== params[j])) return this.evaluate([{ k, params: folded }], true);
    this.pop[k] = folded;
    this.cost[k] = maxError(folded, this.grid);
    this.refs[k] = null;
  }

  // Replaces the worst member with the given kernel.
  inject(params) {
    this.revision++;
    this.enqueue(() => {
      let worst = 0;
      for (let k = 1; k < this.P; k++) if (!(this.cost[k] <= this.cost[worst])) worst = k;
      return this.place(worst, params);
    });
  }

  step(generations = 1) {
    this.generation += generations;
    this.revision++;
    for (let g = 0; g < generations; g++) this.enqueue(() => this.evaluate(this.trials()));
  }

  trials() {
    const { P, F, CR, rates, rand } = this;
    const R = rates.length;
    const logCR = CR < 1 ? Math.log(CR) : 0;
    const perm = Array.from({ length: P }, (_, k) => k);
    for (let k = P - 1; k > 0; k--) {
      const j = Math.floor(rand() * (k + 1));
      [perm[k], perm[j]] = [perm[j], perm[k]];
    }
    return perm.map((target, k) => {
      const [a, b, c] = [1, 2, 3].map((o) => this.pop[perm[(k + o) % P]]);
      const start = Math.floor(rand() * R);
      const length = CR < 1 ? Math.min(R, 1 + Math.floor(Math.log(1 - rand()) / logCR)) : R;
      const params = this.pop[target].slice();
      for (let i = 0; i < R; i++) {
        if ((i - start + R) % R >= length) continue;
        const j = rates[i];
        params[j] = fold(a[j] + F * (b[j] - c[j]), this.lo[j], this.hi[j]);
      }
      return { k: target, params };
    });
  }

  async snapshot() {
    await this.queue;
    if (this.failure) throw this.failure;
    const population = new Float32Array(this.P * this.D);
    let best = 0, sum = 0;
    for (let k = 0; k < this.P; k++) {
      population.set(this.pop[k], k * this.D);
      sum += this.cost[k];
      if (this.cost[k] < this.cost[best]) best = k;
    }
    return {
      population,
      costs: Float32Array.from(this.cost),
      best: this.pop[best].slice(),
      bestCost: this.cost[best],
      meanCost: sum / this.P,
    };
  }

  dispose() {
    this.failure ??= new Error("Optimizer disposed.");
    this.solvers.dispose();
  }
}
