// Mirrors the `jax.scipy.linalg` module in JAX.

import * as laxLinalg from "./lax-linalg";
import { type ArrayLike } from "../frontend/array";
import type { Array } from "../frontend/array";

/**
 * Solve the equation `a @ x = b` for `x`, assuming `a` is a triangular matrix.
 *
 * This wraps `lax.linalg.triangularSolve` with a SciPy-compatible API.
 *
 * @param a - A triangular matrix or batch of triangular matrices with shape
 *   `[..., n, n]`.
 * @param b - Right-hand side matrix or batch of matrices with shape
 *   `[..., n, k]` or `[..., n]`.
 * @param options - Optional parameters:
 *   - `lower` — If `true`, `a` is lower triangular (default `false`).
 *   - `trans` — `0`: no transpose, `1`: transpose `a`, `2`: conjugate
 *     transpose (same as `1` for real dtypes). Default `0`.
 *   - `unitDiagonal` — If `true`, the diagonal of `a` is assumed to be all
 *     ones (default `false`).
 *
 * @returns The solution `x` with the same shape as `b`.
 *
 * @example
 * ```ts
 * import { scipyLinalg, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * const L = np.array([[2., 0.], [1., 3.]]);
 * const b = np.array([[4.], [7.]]);
 * const x = scipyLinalg.solveTriangular(L, b, { lower: true });
 * // x ≈ [[2.], [1.6667]]
 *
 * L.dispose();
 * b.dispose();
 * x.dispose();
 * ```
 */
export function solveTriangular(
  a: ArrayLike,
  b: ArrayLike,
  {
    lower = false,
    trans = 0,
    unitDiagonal = false,
  }: {
    lower?: boolean;
    trans?: 0 | 1 | 2;
    unitDiagonal?: boolean;
  } = {},
): Array {
  // trans=2 (conjugate transpose) is the same as trans=1 for real dtypes.
  const transposeA = trans === 1 || trans === 2;
  return laxLinalg.triangularSolve(a, b, {
    leftSide: true,
    lower,
    transposeA,
    unitDiagonal,
  });
}

/**
 * Compute the Cholesky decomposition of a symmetric positive-definite matrix.
 *
 * @param a - A symmetric positive-definite matrix or batch with shape
 *   `[..., n, n]`.
 * @param options - Optional parameters:
 *   - `lower` — If `true`, return the lower-triangular Cholesky factor
 *     (default `false`, returns upper-triangular — SciPy convention).
 *
 * @returns The Cholesky factor with shape `[..., n, n]`.
 *
 * @example
 * ```ts
 * import { scipyLinalg, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * const A = np.array([[4., 2.], [2., 3.]]);
 * const L = scipyLinalg.cholesky(A, { lower: true });
 * // L ≈ [[2., 0.], [1., 1.4142]]
 *
 * A.dispose();
 * L.dispose();
 * ```
 */
export function cholesky(
  a: ArrayLike,
  { lower = false }: { lower?: boolean } = {},
): Array {
  // Note: SciPy default is upper=true (returns upper Cholesky factor),
  // while lax.linalg.cholesky default is lower (upper=false → returns lower).
  return laxLinalg.cholesky(a, { upper: !lower });
}
