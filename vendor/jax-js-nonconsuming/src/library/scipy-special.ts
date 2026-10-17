// Mirrors the `jax.scipy.special` module in JAX.

import { Array, log, negative } from "./numpy";
import { jit } from "../frontend/jaxpr";

export { erf } from "./lax";
export { erfc } from "./lax";
export { logSoftmax } from "./nn";

/**
 * @function
 * The logit function, `logit(p) = log(p / (1-p))`.
 */
export const logit = jit(function logit(x: Array): Array {
  // logit(p) = log(p / (1-p)). Avoid subtract(1, x) which creates a
  // receiver-position concrete scalar; use negative + add for Lit inlining.
  using negX = negative(x);
  using oneMinusX = negX.add(1);
  using ratio = x.div(oneMinusX);
  return log(ratio);
});

export { logsumexp } from "./nn";
export { softmax } from "./nn";
