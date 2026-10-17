import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function checkSameShape(a: np.Array, b: np.Array) {
  if (
    a.shape.length !== b.shape.length ||
    a.shape.some((dim, i) => dim !== b.shape[i])
  ) {
    throw new Error(
      `Shape mismatch: ${JSON.stringify(a.shape)} vs ${JSON.stringify(b.shape)}`,
    );
  }
}

/**
 * Calculates squared error for a set of predictions.
 *
 * Mean squared error can be computed as `squaredError(a, b).mean()`.
 */
export function squaredError(
  predictions: np.Array,
  targets?: np.Array,
): np.Array {
  if (targets) {
    checkSameShape(predictions, targets);
    const delta = predictions.sub(targets);
    return delta.mul(delta);
  } else {
    return predictions.mul(predictions);
  }
}

/**
 * Calculates the L2 loss for a set of predictions.
 *
 * This is equivalent to 0.5 * squared error, where the constant is standard
 * from "Pattern Recognition and Machine Learning" by Bishop.
 */
export function l2Loss(predictions: np.Array, targets?: np.Array): np.Array {
  return squaredError(predictions, targets).mul(0.5);
}
