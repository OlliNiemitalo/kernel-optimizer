// Accurate sin/cos for traced jax-js code. GPU built-ins can be off by ~1e-4 absolute,
// which the large, cancelling weights of many-component kernels amplify past the ripple.
// Cody–Waite reduction by π/2 (exact for |θ| < ~1e4 in float32), then Taylor polynomials
// on [−π/4, π/4] (error < 2e-9), then quadrant selection. Differentiable.

import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

const TWO_OVER_PI = 0.6366197723675814;
const P1 = 1.5703125;
const P2 = 4.837512969970703125e-4;
const P3 = 7.549789954891882e-8;

export function sincos(theta) {
  const q = np.floor(theta.mul(TWO_OVER_PI).add(0.5));
  const r = theta.sub(q.mul(P1)).sub(q.mul(P2)).sub(q.mul(P3));
  const r2 = r.mul(r);
  // Horner forms of the Taylor series of sin r and cos r.
  const s = r.mul(
    r2.mul(r2.mul(r2.mul(r2.mul(1 / 362880).sub(1 / 5040)).add(1 / 120)).sub(1 / 6)).add(1),
  );
  const c = r2
    .mul(r2.mul(r2.mul(r2.mul(r2.mul(-1 / 3628800).add(1 / 40320)).sub(1 / 720)).add(1 / 24)).sub(0.5))
    .add(1);
  const m = np.remainder(q, 4); // quadrant 0..3
  const is0 = np.equal(m, 0);
  const is1 = np.equal(m, 1);
  const is2 = np.equal(m, 2);
  const sin = np.where(is0, s, np.where(is1, c, np.where(is2, s.neg(), c.neg())));
  const cos = np.where(is0, c, np.where(is1, s.neg(), np.where(is2, c.neg(), s)));
  return [sin, cos];
}
