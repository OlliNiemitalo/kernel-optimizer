// Complex primitives that need to be implemented in software.

import { CodeGenerator } from "./wasmblr";

/** Given a local `x`, evaluate `sum[i](a_i * x^i)` and push to stack. */
function _poly(cg: CodeGenerator, x: number, as: number[]): void {
  if (as.length === 0) throw new Error("_poly needs at least one coefficient");
  cg.f32.const(as[as.length - 1]);
  for (let i = as.length - 2; i >= 0; i--) {
    cg.local.get(x);
    cg.f32.mul();
    if (as[i] !== 0) {
      cg.f32.const(as[i]);
      cg.f32.add();
    }
  }
}

/**
 * Approximate e^x.
 *
 * Method: range-reduce x = k*ln2 + r with k = round(x/ln2), |r|<=~0.3466
 *         then e^x = 2^k * P(r), where P is 5th-order poly (Taylor).
 */
export function wasm_exp(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const k_f = cg.local.declare(cg.f32);
    const k = cg.local.declare(cg.i32);
    const r = cg.local.declare(cg.f32);
    const p = cg.local.declare(cg.f32);
    const scale = cg.local.declare(cg.f32);

    // k = nearest(x / ln2)
    cg.local.get(0);
    cg.f32.const(1 / Math.LN2);
    cg.f32.mul();
    cg.f32.nearest();
    cg.local.tee(k_f);
    cg.i32.trunc_sat_f32_s();
    cg.local.set(k);

    // Handle overflow: if k > 127, return Infinity
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.gt_s();
    cg.if(cg.void);
    {
      cg.f32.const(Infinity);
      cg.return();
    }
    cg.end();

    // Handle underflow: if k < -126, return 0
    cg.local.get(k);
    cg.i32.const(-126);
    cg.i32.lt_s();
    cg.if(cg.void);
    {
      cg.f32.const(0.0);
      cg.return();
    }
    cg.end();

    // r = x - k*ln2
    cg.local.get(0);
    cg.local.get(k_f);
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(r);

    // Taylor series: exp(r) = sum[k] r^k / k!
    _poly(cg, r, [1, 1, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720]);
    cg.local.set(p);

    // scale = 2^k via exponent bits: ((k + 127) << 23)
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.add();
    cg.i32.const(23);
    cg.i32.shl();
    cg.f32.reinterpret_i32();
    cg.local.set(scale);

    // result = P(r) * 2^k
    cg.local.get(p);
    cg.local.get(scale);
    cg.f32.mul();
  });
}

/**
 * Approximate ln(x), x > 0.
 *
 * Method: decompose x = m * 2^e with m in [1,2), e integer (via bit ops)
 *         ln(x) = e*ln2 + ln(m);  use atanh-style series with t=(m-1)/(m+1)
 *         ln(m) ≈ 2*(t + t^3/3 + t^5/5 + t^7/7)
 */
export function wasm_log(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const bits = cg.local.declare(cg.i32);
    const e = cg.local.declare(cg.i32);
    const m = cg.local.declare(cg.f32);
    const t = cg.local.declare(cg.f32);
    const t2 = cg.local.declare(cg.f32);

    // Handle negative quickly: if x < 0 -> NaN
    cg.local.get(0);
    cg.f32.const(0.0);
    cg.f32.lt();
    cg.if(cg.void);
    {
      cg.f32.const(NaN);
      cg.return();
    }
    cg.end();

    // bits = reinterpret(x)
    cg.local.get(0);
    cg.i32.reinterpret_f32();
    cg.local.tee(bits);

    // e = ((bits >> 23) & 0xff) - 127
    cg.i32.const(23);
    cg.i32.shr_u();
    cg.i32.const(255);
    cg.i32.and();
    cg.i32.const(127);
    cg.i32.sub();
    cg.local.set(e);

    // If subnormal or zero, return -Infinity
    cg.local.get(e);
    cg.i32.const(-127);
    cg.i32.eq();
    cg.if(cg.void);
    {
      cg.f32.const(-Infinity);
      cg.return();
    }
    cg.end();

    // If Infinity or NaN, return x
    cg.local.get(e);
    cg.i32.const(128);
    cg.i32.eq();
    cg.if(cg.void);
    {
      cg.local.get(0);
      cg.return();
    }
    cg.end();

    // m_bits = (bits & 0x7fffff) | 0x3f800000  => m in [1,2)
    cg.local.get(bits);
    cg.i32.const(0x7fffff);
    cg.i32.and();
    cg.i32.const(0x3f800000);
    cg.i32.or();
    cg.f32.reinterpret_i32();
    cg.local.set(m);

    // t = (m - 1) / (m + 1)
    cg.local.get(m);
    cg.f32.const(1.0);
    cg.f32.sub();
    cg.local.get(m);
    cg.f32.const(1.0);
    cg.f32.add();
    cg.f32.div();
    cg.local.set(t);

    // powers of t
    cg.local.get(t);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.set(t2); // t^2

    // lnm ≈ 2 * ( t + t^3/3 + t^5/5 + t^7/7 )
    _poly(cg, t2, [2, 2 / 3, 2 / 5, 2 / 7]);
    cg.local.get(t);
    cg.f32.mul();

    // el2 = e * ln2
    cg.local.get(e);
    cg.f32.convert_i32_s();
    cg.f32.const(Math.LN2);
    cg.f32.mul();

    // ln(x) ≈ e*ln2 + ln(m)
    cg.f32.add();
  });
}

/**
 * Common helper to approximate sin(x) and cos(x).
 *
 * Method: reduce to y in [-π, π], then quadrant via q = round(y/(π/2))
 *         z = y - q*(π/2); use one of two polynomials on z:
 *         sin(z) ≈ z + z^3*(-1/6) + z^5*(1/120) + z^7*(-1/5040)
 *         cos(z) ≈ 1 + z^2*(-1/2) + z^4*(1/24) + z^6*(-1/720) + z^8*(1/40320)
 */
function _sincos(cg: CodeGenerator): { q: number; sz: number; cz: number } {
  const y = cg.local.declare(cg.f32);
  const qf = cg.local.declare(cg.f32);
  const q = cg.local.declare(cg.i32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const sz = cg.local.declare(cg.f32);
  const cz = cg.local.declare(cg.f32);

  // y = x - round(x / (2π)) * (2π)
  cg.local.get(0);
  cg.local.get(0);
  cg.f32.const(1 / (2 * Math.PI));
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.f32.const(2 * Math.PI);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.set(y);

  // q = round(y / (π/2)); z = y - q*(π/2)
  cg.local.get(y);
  cg.f32.const(2 / Math.PI);
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.i32.trunc_sat_f32_s();
  cg.local.set(q);

  cg.local.get(y);
  cg.local.get(qf);
  cg.f32.const(Math.PI / 2);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.tee(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);

  // sin poly
  _poly(cg, z2, [1, -1 / 6, 1 / 120, -1 / 5040]);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(sz);

  // cos poly
  _poly(cg, z2, [1, -1 / 2, 1 / 24, -1 / 720, 1 / 40320]);
  cg.local.set(cz);

  return { q, sz, cz };
}

/**
 * Approximate sin(x).
 *
 * Quadrant mapping: k=q mod 4: 0: +sz, 1: +cz, 2: -sz, 3: -cz
 */
export function wasm_sin(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);

    // select magnitude and apply sign: ((q & 2) ? -1 : 1) * ((q & 1) ? cz : sz)
    cg.local.get(cz);
    cg.local.get(sz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}

/**
 * Approximate cos(x).
 *
 * Quadrant mapping: k=q mod 4: 0: +cz, 1: -sz, 2: -cz, 3: +sz
 */
export function wasm_cos(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);

    // select magnitude and apply sign: ((q+1) & 2) ? -1 : 1) * ((q & 1) ? sz : cz)
    cg.local.get(sz);
    cg.local.get(cz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.add();
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}

/** Helper function for approximating arctan(x).  */
function _atan(cg: CodeGenerator) {
  const x = cg.local.declare(cg.f32);
  const abs_x = cg.local.declare(cg.f32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const p = cg.local.declare(cg.f32);

  cg.local.set(x);

  // abs_x = |x|
  cg.local.get(x);
  cg.f32.abs();
  cg.local.set(abs_x);

  // if |x| >= 1, use reciprocal: z = 1/|x|, else z = |x|
  cg.f32.const(1.0);
  cg.local.get(abs_x);
  cg.f32.div();
  cg.local.get(abs_x);
  cg.local.get(abs_x);
  cg.f32.const(1.0);
  cg.f32.ge();
  cg.select();
  cg.local.set(z);

  // z2 = z^2
  cg.local.get(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);

  // Rational approximation: atan(z) ≈ z * P(z^2) / Q(z^2)
  // P(u) = A0 + A1*u + A2*u^2, where u = z^2
  // Q(u) = 1 + B1*u + B2*u^2
  // Fitted coefficients (max error ~5e-7 on [0,1]):
  //   A0 = 0.999998614341, A1 = 0.661705427875, A2 = 0.0415796528637
  //   B1 = 0.994987933645, B2 = 0.173698870181

  // Compute P(z^2) = A0 + z^2*(A1 + z^2*A2)
  _poly(cg, z2, [0.999998614341, 0.661705427875, 0.0415796528637]);

  // Compute Q(z^2) = 1.0 + z^2*(B1 + z^2*B2)
  _poly(cg, z2, [1.0, 0.994987933645, 0.173698870181]);

  // result = z * (P / Q)
  cg.f32.div();
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(p);

  // if |x| >= 1, result = π/2 - result
  cg.f32.const(Math.PI / 2);
  cg.local.get(p);
  cg.f32.sub();
  cg.local.get(p);
  cg.local.get(abs_x);
  cg.f32.const(1.0);
  cg.f32.ge();
  cg.select();

  // apply sign of x
  cg.local.get(x);
  cg.f32.copysign();
}

/**
 * Approximate atan(x).
 *
 * Method: if |x| < 1, use rational approximation: atan(x) ≈ x * P(x^2) / Q(x^2)
 *         where P(u) = A0 + A1*u + A2*u^2 (degree 2)
 *               Q(u) = 1 + B1*u + B2*u^2 (degree 2)
 *         if |x| >= 1, use: atan(x) = sign(x)*π/2 - atan(1/x)
 *         (fitted coefficients, max error ~5e-7 on [0,1])
 */
export function wasm_atan(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0);
    _atan(cg);
  });
}

/**
 * Approximate asin(x).
 *
 * Method: asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
 */
export function wasm_asin(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0); // x
    cg.f32.const(1.0);
    cg.local.get(0);
    cg.local.get(0);
    cg.f32.mul(); // x^2
    cg.f32.sub(); // 1 - x^2
    cg.f32.sqrt(); // sqrt(1 - x^2)
    cg.f32.const(1.0);
    cg.f32.add(); // 1 + sqrt(1 - x^2)
    cg.f32.div(); // x / (1 + sqrt(1 - x^2))
    _atan(cg);
    cg.f32.const(2.0);
    cg.f32.mul();
  });
}

/**
 * Helper function for erf/erfc approximation.
 *
 * See `_erfapprox` in alu.ts for details on the algorithm used.
 */
function _erfapprox(cg: CodeGenerator, exp_func: number) {
  const x = cg.local.declare(cg.f32);
  const t = cg.local.declare(cg.f32);

  cg.local.set(x);

  // Coefficients
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;

  // t = 1 / (1 + p * x)
  cg.f32.const(1.0);
  cg.f32.const(1.0);
  cg.f32.const(p);
  cg.local.get(x);
  cg.f32.mul();
  cg.f32.add();
  cg.f32.div();
  cg.local.set(t);

  _poly(cg, t, [0, a1, a2, a3, a4, a5]);

  // Multiply by exp(-x^2)
  cg.local.get(x);
  cg.f32.neg();
  cg.local.get(x);
  cg.f32.mul();
  cg.call(exp_func);
  cg.f32.mul();
}

/** Approximate erf(x) (error function). */
export function wasm_erf(cg: CodeGenerator, exp: number): number {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.f32.const(1.0);
    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp);
    cg.f32.sub();

    // Multiply by sign of x
    cg.local.get(0);
    cg.f32.copysign();
  });
}

/** Approximate erfc(x) (complementary error function). */
export function wasm_erfc(cg: CodeGenerator, exp: number): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const e = cg.local.declare(cg.f32);

    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp);
    cg.local.set(e);

    // Compute E for x >= 0, or 2 - E for x < 0
    cg.f32.const(2.0);
    cg.local.get(e);
    cg.f32.sub();

    cg.local.get(e);

    cg.local.get(0);
    cg.f32.const(0.0);
    cg.f32.lt();
    cg.select();
  });
}

/**
 * Threefry2x32 pseudorandom number generator.
 *
 * Takes two 32-bit keys and two 32-bit counters as input,
 * returns two 32-bit pseudorandom values.
 */
export function wasm_threefry2x32(cg: CodeGenerator): number {
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [cg.i32, cg.i32], () => {
    const ks0 = cg.local.declare(cg.i32);
    const ks1 = cg.local.declare(cg.i32);
    const ks2 = cg.local.declare(cg.i32);
    const x0 = cg.local.declare(cg.i32);
    const x1 = cg.local.declare(cg.i32);

    // x0 += x1; x1 = rotl(x1, rot) ^ x0
    const mix = (rot: number) => {
      cg.local.get(x0);
      cg.local.get(x1);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.i32.const(rot);
      cg.i32.rotl();
      cg.local.get(x0);
      cg.i32.xor();
      cg.local.set(x1);
    };

    // x0 += k0; x1 += k1 + round
    const keySchedule = (k0: number, k1: number, round: number) => {
      cg.local.get(x0);
      cg.local.get(k0);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.local.get(k1);
      cg.i32.add();
      cg.i32.const(round);
      cg.i32.add();
      cg.local.set(x1);
    };

    // Key schedule: ks0 = key0; ks1 = key1; ks2 = key0 ^ key1 ^ 0x1BD11BDA
    cg.local.get(0);
    cg.local.set(ks0);
    cg.local.get(1);
    cg.local.set(ks1);
    cg.local.get(0);
    cg.local.get(1);
    cg.i32.xor();
    cg.i32.const(0x1bd11bda);
    cg.i32.xor();
    cg.local.set(ks2);

    // x0 = ctr0 + ks0; x1 = ctr1 + ks1
    cg.local.get(2); // ctr0
    cg.local.get(ks0);
    cg.i32.add();
    cg.local.set(x0);
    cg.local.get(3); // ctr1
    cg.local.get(ks1);
    cg.i32.add();
    cg.local.set(x1);

    // Round 1
    (mix(13), mix(15), mix(26), mix(6));
    keySchedule(ks1, ks2, 1);

    // Round 2
    (mix(17), mix(29), mix(16), mix(24));
    keySchedule(ks2, ks0, 2);

    // Round 3
    (mix(13), mix(15), mix(26), mix(6));
    keySchedule(ks0, ks1, 3);

    // Round 4
    (mix(17), mix(29), mix(16), mix(24));
    keySchedule(ks1, ks2, 4);

    // Round 5
    (mix(13), mix(15), mix(26), mix(6));
    keySchedule(ks2, ks0, 5);

    // Return x0 and x1
    cg.local.get(x0);
    cg.local.get(x1);
  });
}
