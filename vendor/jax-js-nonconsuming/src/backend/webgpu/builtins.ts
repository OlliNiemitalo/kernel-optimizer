// Port of the JavaScript `threefry2x32()` function, see alu.ts for details.
export const threefrySrc = `
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`;

// Port of the JavaScript `erf()` and `erfc()` functions, see alu.ts for details.
export const erfSrc = `
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`;
