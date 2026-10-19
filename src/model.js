// Traced (jax-js) kernel profile, shared by the optimizer and the fine-tuner.

import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { MAX_MULTIPLICITY, PARAMS_PER_COMPONENT } from "./problem.js";
import { sincos } from "./trig.js";

// params: [batch, components·PARAMS_PER_COMPONENT] → profile values [batch, samples] at s = x².
// Only powers of s below `terms` are evaluated; the rest must be zero.
export function profileBatch(params, s2, components, terms = MAX_MULTIPLICITY) {
  const p = params.reshape([-1, components, PARAMS_PER_COMPONENT]);
  const [b, c, ...weights] = Array.from({ length: 2 + 2 * terms }, (_, k) => {
    using idx = np.array([k], { dtype: np.int32 });
    return np.take(p, idx, 2); // [batch, components, 1]
  });
  const s = s2.reshape([1, 1, -1]);
  const [sin, cos] = sincos(c.mul(s));
  // Σ_k (r_k, i_k) s^k by Horner, from the highest power.
  let re = weights.at(-2);
  let im = weights.at(-1);
  for (let k = weights.length - 4; k >= 0; k -= 2) {
    re = re.mul(s).add(weights[k]);
    im = im.mul(s).add(weights[k + 1]);
  }
  return cos.mul(re).add(sin.mul(im)).mul(np.exp(b.neg().mul(s))).sum(1);
}
