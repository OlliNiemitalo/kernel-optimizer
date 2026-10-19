// Gradient-based fine-tuning of a parameter vector with jax-js autodiff and optax Adam.
// Minimizes a smooth maximum of |profile − target| (log-sum-exp with sharpness β),
// raising β over the run so the objective approaches the equiripple (minimax) cost.

import { jit, lax, numpy as np, tree, valueAndGrad } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates } from "@hamk-uas/jax-js-nonconsuming/optax";
import { profileBatch } from "./model.js";
import { PARAMS_PER_COMPONENT, maxError, multiplicities } from "./problem.js";

export class FineTuner {
  // Parameters are optimized in normalized units θ = p / scale so one learning rate fits all.
  // A zero scale freezes a parameter at zero (the x² weights of single components).
  constructor({ params, components, grid, learningRate, halfLife = 1000 }) {
    this.components = components;
    this.grid = grid;
    const D = params.length;
    const mult = multiplicities(params);
    const live = (j) => j % PARAMS_PER_COMPONENT < 2 + 2 * mult[Math.floor(j / PARAMS_PER_COMPONENT)];
    const scale = Float32Array.from(params, (v, j) => (live(j) ? Math.max(Math.abs(v), 0.05) : 0));
    const minTheta = Float32Array.from({ length: D }, (_, j) =>
      j % PARAMS_PER_COMPONENT === 0 ? 0.05 / scale[j] : -1e30,
    );
    this.scale = scale;
    this.s2 = np.array(Float32Array.from(grid.x, (x) => x * x));
    this.target = np.array(Float32Array.from(grid.target));
    this.scaleArr = np.array(scale);
    this.minTheta = np.array(minTheta);
    this.theta = np.array(Float32Array.from(params, (v, j) => (scale[j] ? v / scale[j] : 0)));

    const negativeWeight = grid.negativeWeight ?? 1;
    const loss = (theta, beta, s2, target, scaleArr) => {
      const prof = profileBatch(theta.mul(scaleArr), s2, components, Math.max(...mult)).reshape([-1]);
      const e = prof.sub(target);
      let a = np.sqrt(e.mul(e).add(1e-12));
      if (negativeWeight !== 1) a = np.where(np.less(prof, 0), a.mul(negativeWeight), a);
      const m = lax.stopGradient(a.max());
      return np.log(np.exp(a.sub(m).mul(beta)).sum()).div(beta).add(m);
    };
    this.optimizer = adam((count) => np.exp(count.mul(-Math.LN2 / halfLife)).mul(learningRate));
    this.state = this.optimizer.init(this.theta);
    this.stepFn = jit((theta, state, beta, s2, target, scaleArr, minTheta) => {
      const [value, grads] = valueAndGrad(loss)(theta, beta, s2, target, scaleArr);
      const [updates, nextState] = this.optimizer.update(grads, state);
      return [np.maximum(applyUpdates(theta, updates), minTheta), nextState, value];
    });
    this.step = 0;
  }

  // Run steps with β = sharpness / (current max error), then report the true max error.
  async run(steps, sharpness, currentMax) {
    using beta = np.array(sharpness / Math.max(currentMax, 1e-7));
    let value = null;
    for (let k = 0; k < steps; k++) {
      const [theta, state, v] = this.stepFn(
        this.theta, this.state, beta, this.s2, this.target, this.scaleArr, this.minTheta,
      );
      this.theta.dispose();
      tree.dispose(this.state);
      value?.dispose();
      this.theta = theta;
      this.state = state;
      value = v;
      this.step++;
    }
    const smoothMax = value ? await value.consumeData() : null;
    const theta = await this.theta.data();
    const params = Array.from(theta, (t, j) => t * this.scale[j]);
    return { params, smoothMax: smoothMax?.[0], error: maxError(params, this.grid) };
  }

  dispose() {
    this.theta.dispose();
    tree.dispose(this.state);
    for (const a of [this.s2, this.target, this.scaleArr, this.minTheta]) a.dispose();
    this.stepFn.dispose();
  }
}
