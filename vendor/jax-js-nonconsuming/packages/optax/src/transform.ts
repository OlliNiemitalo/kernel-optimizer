import { JsTree, numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";

import {
  GradientTransformation,
  identity,
  initEmptyState,
  ScalarOrSchedule,
  Schedule,
} from "./base";
import {
  treeBiasCorrection,
  treeNorm,
  treeUpdateMoment,
  treeZerosLike,
} from "./treeUtils";

function u32(x: number): np.Array {
  return np.array(x, { dtype: np.uint32 });
}

export type ScaleByAdamOptions = {
  b1?: number;
  b2?: number;
  eps?: number;
  epsRoot?: number;
  nesterov?: boolean;
};

export function scaleByAdam({
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
  epsRoot = 0.0,
  nesterov = false,
}: ScaleByAdamOptions = {}): GradientTransformation {
  return {
    init(params) {
      const mu = treeZerosLike(params); // first moment
      const nu = treeZerosLike(params); // second moment
      return { count: u32(0), mu, nu };
    },
    update(updates, state, _params) {
      const {
        count: oldCount,
        mu: oldMu,
        nu: oldNu,
      } = state as {
        count: np.Array;
        mu: JsTree<np.Array>;
        nu: JsTree<np.Array>;
      };
      const mu = treeUpdateMoment(updates, oldMu, b1, 1);
      const nu = treeUpdateMoment(updates, oldNu, b2, 2);
      const count = oldCount.add(1);
      tree.dispose(oldMu);
      tree.dispose(oldNu);
      oldCount.dispose();
      let muHat: typeof mu;
      if (nesterov) {
        const countP1 = count.add(1);
        const muBC = treeBiasCorrection(mu, b1, countP1);
        const updBC = treeBiasCorrection(updates, b1, count);
        countP1.dispose();
        muHat = tree.map(
          (m: np.Array, g: np.Array) => {
            const mScaled = m.mul(b1);
            const gScaled = g.mul(1 - b1);
            const result = mScaled.add(gScaled);
            mScaled.dispose();
            gScaled.dispose();
            return result;
          },
          muBC,
          updBC,
        );
        tree.dispose(muBC);
        tree.dispose(updBC);
      } else {
        muHat = treeBiasCorrection(mu, b1, count);
      }
      const nuHat = treeBiasCorrection(nu, b2, count);
      updates = tree.map(
        (m: np.Array, v: np.Array) => {
          const vEps = v.add(epsRoot);
          const sqrtV = np.sqrt(vEps);
          vEps.dispose();
          const denom = sqrtV.add(eps);
          sqrtV.dispose();
          const result = m.div(denom);
          denom.dispose();
          return result;
        },
        muHat,
        nuHat,
      ) as typeof updates;
      tree.dispose(muHat);
      tree.dispose(nuHat);
      return [updates, { count, mu, nu }];
    },
  };
}

/** Scale by a constant step size. */
export function scale(stepSize: number): GradientTransformation {
  return {
    init: initEmptyState,
    update(updates, state, _params) {
      updates = tree.map((g: np.Array) => g.mul(stepSize), updates);
      return [updates, state];
    },
  };
}

/**
 * Scale updates using a custom schedule for the step size.
 *
 * JIT-compatible when the schedule body uses `np.*` array operations.
 */
export function scaleBySchedule(stepSizeFn: Schedule): GradientTransformation {
  return {
    init(_params) {
      return { count: u32(0) }; // initial step
    },
    update(updates, state, _params) {
      const { count } = state as { count: np.Array };
      using countFloat = np.astype(count, np.float32);
      const stepSize = stepSizeFn(countFloat);
      updates = tree.map((g: np.Array) => {
        const result = g.mul(stepSize);
        return result;
      }, updates) as typeof updates;
      stepSize.dispose();
      const newCount = count.add(1);
      count.dispose();
      return [updates, { count: newCount }];
    },
  };
}

/** Scale by the (negative) learning rate (either as scalar or as schedule). */
export function scaleByLearningRate(
  learningRate: ScalarOrSchedule,
  flipSign = true,
): GradientTransformation {
  if (learningRate === undefined) return identity();
  const m = flipSign ? -1 : 1;
  if (typeof learningRate === "function") {
    return scaleBySchedule((count: np.Array) => {
      const lr = learningRate(count);
      const result = lr.mul(m);
      lr.dispose();
      return result;
    });
  }
  return scale(m * learningRate);
}

/** Clip gradients by global norm. */
export function clipByGlobalNorm(maxNorm: number): GradientTransformation {
  return {
    init: initEmptyState,
    update(updates, state, _params) {
      const gNorm = treeNorm(updates);
      const trigger = np.less(gNorm, maxNorm);

      const clippedUpdates = tree.map((t: np.Array) => {
        const scaled = t.div(gNorm);
        const clipped = scaled.mul(maxNorm);
        scaled.dispose();
        const result = np.where(trigger, t, clipped);
        clipped.dispose();
        return result;
      }, updates);

      trigger.dispose();
      gNorm.dispose();

      return [clippedUpdates, state];
    },
  };
}

export type MaskFn = (tree: JsTree<np.Array>) => JsTree<np.Array>;

export type AddDecayedWeightsOptions = {
  weightDecay?: ScalarOrSchedule;
  mask?: JsTree<np.Array> | MaskFn | null;
};

/** Add parameter scaled by weight decay.
 *
 * JIT-compatible with both scalar and schedule `weightDecay`.
 */
export function addDecayedWeights({
  weightDecay = 0.0,
  mask = null,
}: AddDecayedWeightsOptions = {}): GradientTransformation {
  const isSchedule = typeof weightDecay === "function";

  return {
    init(_params) {
      if (isSchedule) {
        return { count: u32(0) };
      } else {
        return [];
      }
    },
    update(updates, state, params) {
      if (!params) {
        throw new Error("addDecayedWeights requires params to be provided");
      }

      let newState: typeof state;
      let currentWeightDecay: number | np.Array;

      if (isSchedule) {
        const { count } = state as { count: np.Array };
        using countFloat = np.astype(count, np.float32);
        currentWeightDecay = (weightDecay as Schedule)(countFloat);
        // jax-js-lint: allow-non-using — transferred into returned newState
        const newCount = count.add(1);
        count.dispose();
        newState = { count: newCount };
      } else {
        currentWeightDecay = weightDecay as number;
        newState = state;
      }

      if (currentWeightDecay === 0.0) {
        return [updates, newState];
      }

      let decayedParams: JsTree<np.Array>;
      let maskTreeToDispose: JsTree<np.Array> | null = null;
      if (mask) {
        const maskTree = typeof mask === "function" ? mask(updates) : mask;
        if (typeof mask === "function") maskTreeToDispose = maskTree;

        decayedParams = tree.map(
          (p: np.Array, m: np.Array) => {
            const pm = p.mul(m);
            const result = pm.mul(currentWeightDecay);
            pm.dispose();
            return result;
          },
          params,
          maskTree,
        );
        if (maskTreeToDispose) tree.dispose(maskTreeToDispose);
      } else {
        decayedParams = tree.map(
          (p: np.Array) => p.mul(currentWeightDecay),
          params,
        );
      }

      if (typeof currentWeightDecay !== "number") currentWeightDecay.dispose();

      updates = tree.map(
        (g: np.Array, d: np.Array) => g.add(d),
        updates,
        decayedParams,
      ) as typeof updates;
      tree.dispose(decayedParams);

      return [updates, newState];
    },
  };
}

export type TraceOptions = {
  decay?: number;
  nesterov?: boolean;
};

/** Compute a trace of past updates. */
export function trace({
  decay = 0.9,
  nesterov = false,
}: TraceOptions = {}): GradientTransformation {
  return {
    init(params) {
      const trace = treeZerosLike(params);
      return { trace };
    },
    update(updates, state, _params) {
      const { trace: prevTrace } = state as { trace: JsTree<np.Array> };

      // new_trace = g + decay * t
      const newTrace = tree.map(
        (g: np.Array, t: np.Array) => {
          const scaled = t.mul(decay);
          const result = g.add(scaled);
          scaled.dispose();
          return result;
        },
        updates,
        prevTrace,
      );
      tree.dispose(prevTrace);

      let finalUpdates: typeof updates;
      if (nesterov) {
        // Nesterov: updates = g + decay * new_trace
        finalUpdates = tree.map(
          (g: np.Array, t: np.Array) => {
            const scaled = t.mul(decay);
            const result = g.add(scaled);
            scaled.dispose();
            return result;
          },
          updates,
          newTrace,
        ) as typeof updates;
      } else {
        // Standard momentum: updates = new_trace
        // Ref the tree so chain can safely dispose intermediate updates
        // without affecting the state's reference.
        // jax-js-lint: allow-ref
        finalUpdates = tree.ref(newTrace) as typeof updates;
      }

      return [finalUpdates, { trace: newTrace }];
    },
  };
}
