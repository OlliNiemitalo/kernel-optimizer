import {
  DType,
  JsTree,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";

export function treeZerosLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.zerosLike(x, { dtype }), tr);
}

export function treeOnesLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.onesLike(x, { dtype }), tr);
}

function ipow(a: np.Array, order: number) {
  if (!Number.isInteger(order) || order <= 0) {
    throw new Error("Order must be a positive integer");
  }
  if (order === 1) return a;
  let result = a.mul(a);
  for (let i = 2; i < order; i++) {
    const prev = result;
    result = result.mul(a);
    prev.dispose();
  }
  return result;
}

export function treeUpdateMoment(
  updates: JsTree<np.Array>,
  moments: JsTree<np.Array>,
  decay: number,
  order: number,
): JsTree<np.Array> {
  return tree.map(
    (g: np.Array, t: np.Array) => {
      const gPow = ipow(g, order);
      const scaledG = gPow.mul(1 - decay);
      if (order > 1) gPow.dispose();
      const scaledT = t.mul(decay);
      const result = scaledG.add(scaledT);
      scaledG.dispose();
      scaledT.dispose();
      return result;
    },
    updates,
    moments,
  );
}

/**
 * Performs bias correction, dividing by 1-decay^count.
 *
 * Uses array operations instead of `.item()` so this function is
 * compatible with JIT tracing (count can be a tracer).
 */
export function treeBiasCorrection(
  moments: JsTree<np.Array>,
  decay: number,
  count: np.Array,
): JsTree<np.Array> {
  using countFloat = np.astype(count, np.float32);
  using decayArr = np.array(decay);
  using decayPow = np.power(decayArr, countFloat);
  using one = np.array(1.0);
  using oneMinusDecayPow = one.sub(decayPow);
  using correction = one.div(oneMinusDecayPow);
  return tree.map((t: np.Array) => t.mul(correction), moments);
}

/** Sum all elements across all arrays in a pytree. */
export function treeSum(tr: JsTree<np.Array>): np.Array {
  const [leaves] = tree.flatten(tr);
  let total = np.array(0.0);
  for (const leaf of leaves) {
    const s = np.sum(leaf);
    const prevTotal = total;
    total = total.add(s);
    s.dispose();
    prevTotal.dispose();
  }
  return total;
}

/** Max of all elements across all arrays in a pytree. */
export function treeMax(tr: JsTree<np.Array>): np.Array {
  const [leaves] = tree.flatten(tr);
  let maxVal = np.array(-Infinity);
  for (const leaf of leaves) {
    const m = np.max(leaf);
    const prevMax = maxVal;
    maxVal = np.maximum(maxVal, m);
    m.dispose();
    prevMax.dispose();
  }
  return maxVal;
}

export type NormOrd = 1 | 2 | "inf" | "infinity" | number | null;

/** Compute the vector norm of the given ord of a pytree. */
export function treeNorm(
  tr: JsTree<np.Array>,
  ord: NormOrd = null,
  squared = false,
): np.Array {
  if (ord === null || ord === 2) {
    const squaredTree = tree.map(np.square, tr);
    const sqNorm = treeSum(squaredTree);
    tree.dispose(squaredTree);
    if (squared) return sqNorm;
    const result = np.sqrt(sqNorm);
    sqNorm.dispose();
    return result;
  } else if (ord === 1) {
    const absTree = tree.map(np.abs, tr);
    const result = treeSum(absTree);
    tree.dispose(absTree);
    if (squared) {
      const sq = np.square(result);
      result.dispose();
      return sq;
    }
    // eslint-disable-next-line jax-js/no-use-after-dispose -- dispose is conditional (squared branch only)
    return result;
  } else if (ord === "inf" || ord === "infinity" || ord === Infinity) {
    const absTree = tree.map(np.abs, tr);
    const result = treeMax(absTree);
    tree.dispose(absTree);
    if (squared) {
      const sq = np.square(result);
      result.dispose();
      return sq;
    }
    // eslint-disable-next-line jax-js/no-use-after-dispose -- dispose is conditional (squared branch only)
    return result;
  } else {
    throw new Error(`Unsupported ord: ${ord}`);
  }
}
