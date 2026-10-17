// Traced (jax-js) kernel profile, shared by the optimizer and the fine-tuner.

import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { PARAMS_PER_COMPONENT } from "./problem.js";
import { sincos } from "./trig.js";

// params: [batch, components·6] → profile values [batch, samples] at s = x².
export function profileBatch(params, s2, components) {
  const p = params.reshape([-1, components, PARAMS_PER_COMPONENT]);
  const [b, c, r, i, r1, i1] = Array.from({ length: PARAMS_PER_COMPONENT }, (_, k) => {
    using idx = np.array([k], { dtype: np.int32 });
    return np.take(p, idx, 2); // [batch, components, 1]
  });
  const s = s2.reshape([1, 1, -1]);
  const [sin, cos] = sincos(c.mul(s));
  const re = r.add(r1.mul(s));
  const im = i.add(i1.mul(s));
  return cos.mul(re).add(sin.mul(im)).mul(np.exp(b.neg().mul(s))).sum(1);
}
