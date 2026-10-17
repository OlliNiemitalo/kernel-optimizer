import * as lax from "./lax";
import { triangularSolve } from "./lax-linalg";
import * as np from "./numpy";
import { Array, ArrayLike, fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";
import { checkAxis, generalBroadcast } from "../utils";

/**
 * Apply a pre-computed LU factorization to solve A x = b.
 *
 * @param lu - Combined L/U matrix from `lax.linalg.lu(A)`.
 * @param P  - Permutation matrix derived from the LU permutation vector.
 * @param b  - Right-hand side to solve.
 * @param d  - DisposableStack that owns intermediate arrays.
 * @returns Solution x (not registered on `d`; caller decides ownership).
 */
function luSolveWithP(
  lu: Array,
  P: Array,
  b: Array,
  d: DisposableStack,
): Array {
  const Pb = d.use(np.matmul(P, b));
  const LPb = d.use(
    triangularSolve(lu, Pb, {
      leftSide: true,
      lower: true,
      unitDiagonal: true,
    }),
  );
  return triangularSolve(lu, LPb, { leftSide: true, lower: false });
}

function checkSquare(name: string, a: Array) {
  if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) {
    throw new Error(
      `${name}: input must be at least 2D square matrix, got ${a.aval}`,
    );
  }
  return a.shape[a.ndim - 1];
}

/**
 * Compute the Cholesky decomposition of a (batched) positive-definite matrix.
 *
 * This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
 * the input matrix, which is on by default.
 */
export function cholesky(
  a: ArrayLike,
  {
    upper = false,
    symmetrizeInput = true,
  }: {
    upper?: boolean;
    symmetrizeInput?: boolean;
  } = {},
): Array {
  a = fudgeArray(a);
  checkSquare("cholesky", a);
  if (symmetrizeInput) {
    using at = np.matrixTranspose(a);
    using sum = a.add(at);
    using sym = sum.mul(0.5);
    return lax.linalg.cholesky(sym, { upper });
  }
  return lax.linalg.cholesky(a, { upper });
}

/**
 * Compute the cross-product of two 3D vectors.
 *
 * This is a simpler and less flexible version of `jax.numpy.cross()`.
 * Both inputs must have size 3 along the specified axis.
 */
export function cross(x1: ArrayLike, x2: ArrayLike, axis: number = -1): Array {
  const a1 = checkAxis(axis, np.ndim(x1));
  const a2 = checkAxis(axis, np.ndim(x2));
  if (np.shape(x1)[a1] !== 3)
    throw new Error(
      `linalg.cross: x1 must have size 3 along axis ${axis}, got ${np.shape(x1)[a1]}`,
    );
  if (np.shape(x2)[a2] !== 3)
    throw new Error(
      `linalg.cross: x2 must have size 3 along axis ${axis}, got ${np.shape(x2)[a2]}`,
    );
  return np.cross(x1, x2, { axis });
}

/** Compute the determinant of a square matrix (batched). */
export function det(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("det", a);
  const luResult = lax.linalg.lu(a);
  using lu = luResult[0];
  using pivots = luResult[1];
  using _perm = luResult[2];
  using indices = np.arange(n);
  using neq = pivots.notEqual(indices);
  using neqInt = neq.astype(np.int32);
  using sumAxis = neqInt.sum(-1);
  using parity = sumAxis.mod(2);
  using neg2 = parity.mul(-2);
  using sign = neg2.add(1); // (-1)^parity
  using diag = lu.diagonal(0, -1, -2);
  using prod = np.prod(diag, -1);
  return prod.mul(sign);
}

export { diagonal } from "./numpy";

/**
 * Compute the inverse of a square matrix (batched).
 *
 * For small matrices (n ≤ 4), uses an analytical Cramer's rule formula
 * (adjugate / determinant) that compiles to a single fused kernel under JIT.
 * Larger matrices fall through to LU-based `solve(A, I)`.
 */
export function inv(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("inv", a);

  // Analytical fast path for small matrices.
  // Uses Cramer's rule: inv(A) = adj(A) / det(A).
  // All operations are elementwise and fuse into a single JIT kernel.
  // AD works natively through these standard ops — no custom rules needed.
  if (n === 1) {
    return np.reciprocal(a);
  }
  if (n === 2) {
    return inv2x2(a);
  }
  if (n === 3) {
    return inv3x3(a);
  }
  if (n === 4) {
    return inv4x4(a);
  }

  using eye = np.eye(n, { dtype: a.dtype });
  return solve(a, eye);
}

// ── Analytical small-matrix inverse helpers ──────────────────────────────
// All use [..., i, j] slicing for batch support.  Intermediates are tracked
// with `using` for ownership correctness in both eager and JIT modes.

/** Analytical 2×2 inverse: [[d,-b],[-c,a]] / (ad-bc) */
function inv2x2(a: Array): Array {
  // Extract elements: a[...,0,0], a[...,0,1], a[...,1,0], a[...,1,1]
  using a00 = idx2d(a, 0, 0);
  using a01 = idx2d(a, 0, 1);
  using a10 = idx2d(a, 1, 0);
  using a11 = idx2d(a, 1, 1);

  // det = a00*a11 - a01*a10
  using p1 = a00.mul(a11) as Array;
  using p2 = a01.mul(a10) as Array;
  using det = p1.sub(p2) as Array;
  using invDet = np.reciprocal(det);

  // adjugate: [[a11, -a01], [-a10, a00]]
  using neg_a01 = a01.neg() as Array;
  using neg_a10 = a10.neg() as Array;
  using row0 = np.stack([a11, neg_a01], -1);
  using row1 = np.stack([neg_a10, a00], -1);
  using adj = np.stack([row0, row1], -2);

  // Expand invDet to [..., 1, 1] for broadcasting
  using invDet1 = np.expandDims(invDet, -1);
  using invDet2 = np.expandDims(invDet1, -1);
  return adj.mul(invDet2) as Array;
}

/** Analytical 3×3 inverse via cofactor matrix / determinant. */
function inv3x3(a: Array): Array {
  using a00 = idx2d(a, 0, 0);
  using a01 = idx2d(a, 0, 1);
  using a02 = idx2d(a, 0, 2);
  using a10 = idx2d(a, 1, 0);
  using a11 = idx2d(a, 1, 1);
  using a12 = idx2d(a, 1, 2);
  using a20 = idx2d(a, 2, 0);
  using a21 = idx2d(a, 2, 1);
  using a22 = idx2d(a, 2, 2);

  // Adjugate matrix entries (cofactors, transposed)
  using c00 = det2(a11, a22, a12, a21);
  using c01 = det2(a02, a21, a01, a22);
  using c02 = det2(a01, a12, a02, a11);
  using c10 = det2(a12, a20, a10, a22);
  using c11 = det2(a00, a22, a02, a20);
  using c12 = det2(a02, a10, a00, a12);
  using c20 = det2(a10, a21, a11, a20);
  using c21 = det2(a01, a20, a00, a21);
  using c22 = det2(a00, a11, a01, a10);

  // det via Laplace expansion along row 0
  using cof01 = det2(a10, a22, a12, a20);
  using cof02 = det2(a10, a21, a11, a20);
  using t1 = a00.mul(c00) as Array;
  using t2 = a01.mul(cof01) as Array;
  using t3 = a02.mul(cof02) as Array;
  using t12 = t1.sub(t2) as Array;
  using det = t12.add(t3) as Array;
  using invDet = np.reciprocal(det);

  using row0 = np.stack([c00, c01, c02], -1);
  using row1 = np.stack([c10, c11, c12], -1);
  using row2 = np.stack([c20, c21, c22], -1);
  using adj = np.stack([row0, row1, row2], -2);

  using invDet1 = np.expandDims(invDet, -1);
  using invDet2 = np.expandDims(invDet1, -1);
  return adj.mul(invDet2) as Array;
}

/** Analytical 4×4 inverse via cofactor matrix / determinant. */
function inv4x4(a: Array): Array {
  // Extract all 16 elements
  using a00 = idx2d(a, 0, 0);
  using a01 = idx2d(a, 0, 1);
  using a02 = idx2d(a, 0, 2);
  using a03 = idx2d(a, 0, 3);
  using a10 = idx2d(a, 1, 0);
  using a11 = idx2d(a, 1, 1);
  using a12 = idx2d(a, 1, 2);
  using a13 = idx2d(a, 1, 3);
  using a20 = idx2d(a, 2, 0);
  using a21 = idx2d(a, 2, 1);
  using a22 = idx2d(a, 2, 2);
  using a23 = idx2d(a, 2, 3);
  using a30 = idx2d(a, 3, 0);
  using a31 = idx2d(a, 3, 1);
  using a32 = idx2d(a, 3, 2);
  using a33 = idx2d(a, 3, 3);

  // 2×2 minors from rows 2,3
  using s0 = det2(a20, a31, a30, a21);
  using s1 = det2(a20, a32, a30, a22);
  using s2 = det2(a20, a33, a30, a23);
  using s3 = det2(a21, a32, a31, a22);
  using s4 = det2(a21, a33, a31, a23);
  using s5 = det2(a22, a33, a32, a23);

  // 2×2 minors from rows 0,1
  using c0 = det2(a00, a11, a10, a01);
  using c1 = det2(a00, a12, a10, a02);
  using c2 = det2(a00, a13, a10, a03);
  using c3 = det2(a01, a12, a11, a02);
  using c4 = det2(a01, a13, a11, a03);
  using c5 = det2(a02, a13, a12, a03);

  // det = c0*s5 - c1*s4 + c2*s3 + c3*s2 - c4*s1 + c5*s0
  using d1 = c0.mul(s5) as Array;
  using d2 = c1.mul(s4) as Array;
  using d3 = c2.mul(s3) as Array;
  using d4 = c3.mul(s2) as Array;
  using d5 = c4.mul(s1) as Array;
  using d6 = c5.mul(s0) as Array;
  using dt1 = d1.sub(d2) as Array;
  using dt2 = dt1.add(d3) as Array;
  using dt3 = dt2.add(d4) as Array;
  using dt4 = dt3.sub(d5) as Array;
  using det = dt4.add(d6) as Array;
  using invDet = np.reciprocal(det);

  // Adjugate matrix (transposed cofactors)
  using adj00 = madd3(a11, s5, a13, s3, a12, s4); // a11*s5 - a12*s4 + a13*s3
  using adj01n = madd3(a01, s5, a03, s3, a02, s4);
  using adj01 = adj01n.neg() as Array;
  using adj02 = madd3(a31, c5, a33, c3, a32, c4);
  using adj03n = madd3(a21, c5, a23, c3, a22, c4);
  using adj03 = adj03n.neg() as Array;

  using adj10n = madd3(a10, s5, a13, s1, a12, s2);
  using adj10 = adj10n.neg() as Array;
  using adj11 = madd3(a00, s5, a03, s1, a02, s2);
  using adj12n = madd3(a30, c5, a33, c1, a32, c2);
  using adj12 = adj12n.neg() as Array;
  using adj13 = madd3(a20, c5, a23, c1, a22, c2);

  using adj20 = madd3(a10, s4, a13, s0, a11, s2);
  using adj21n = madd3(a00, s4, a03, s0, a01, s2);
  using adj21 = adj21n.neg() as Array;
  using adj22 = madd3(a30, c4, a33, c0, a31, c2);
  using adj23n = madd3(a20, c4, a23, c0, a21, c2);
  using adj23 = adj23n.neg() as Array;

  using adj30n = madd3(a10, s3, a12, s0, a11, s1);
  using adj30 = adj30n.neg() as Array;
  using adj31 = madd3(a00, s3, a02, s0, a01, s1);
  using adj32n = madd3(a30, c3, a32, c0, a31, c1);
  using adj32 = adj32n.neg() as Array;
  using adj33 = madd3(a20, c3, a22, c0, a21, c1);

  using row0 = np.stack([adj00, adj01, adj02, adj03], -1);
  using row1 = np.stack([adj10, adj11, adj12, adj13], -1);
  using row2 = np.stack([adj20, adj21, adj22, adj23], -1);
  using row3 = np.stack([adj30, adj31, adj32, adj33], -1);
  using adj = np.stack([row0, row1, row2, row3], -2);

  using invDet1 = np.expandDims(invDet, -1);
  using invDet2 = np.expandDims(invDet1, -1);
  return adj.mul(invDet2) as Array;
}

/** Compute a*b - c*d, disposing intermediates. */
function det2(a: Array, b: Array, c: Array, d: Array): Array {
  using ab = a.mul(b) as Array;
  using cd = c.mul(d) as Array;
  return ab.sub(cd) as Array;
}

/** Compute a*b - c*d + e*f, disposing intermediates. */
function madd3(
  a: Array,
  b: Array,
  e: Array,
  f: Array,
  c: Array,
  d: Array,
): Array {
  using ab = a.mul(b) as Array;
  using cd = c.mul(d) as Array;
  using ef = e.mul(f) as Array;
  using t1 = ab.sub(cd) as Array;
  return t1.add(ef) as Array;
}

/**
 * Extract element a[..., i, j] from a batched matrix, returning shape [...].
 * Uses core.shrink (ShapeTracker view) + reshape — O(1), no data copy.
 */
function idx2d(a: Array, i: number, j: number): Array {
  const nd = a.ndim;
  const slices: [number, number][] = [];
  for (let d = 0; d < nd - 2; d++) slices.push([0, a.shape[d]]);
  slices.push([i, i + 1], [j, j + 1]);
  using sliced = core.shrink(a, slices) as Array;
  return np.squeeze(sliced, [-1, -2]);
}

/**
 * Compute the QR decomposition of a (batched) matrix.
 *
 * Returns the thin QR factorization `A = Q @ R`, where `Q` has orthonormal
 * columns and `R` is upper triangular.
 *
 * @param a - A matrix or batch of matrices with shape `[..., m, n]`.
 * @returns A tuple `[Q, R]` where `Q` has shape `[..., m, k]` and `R` has
 *   shape `[..., k, n]`, with `k = min(m, n)`.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * const A = np.array([[1., 2.], [3., 4.], [5., 6.]]);
 * const [Q, R] = np.linalg.qr(A);
 * // Q has shape [3, 2], R has shape [2, 2]
 * A.dispose();
 * Q.dispose();
 * R.dispose();
 * ```
 */
export function qr(a: ArrayLike): [Array, Array] {
  return lax.linalg.qr(a);
}

/**
 * Return the least-squares solution to a linear equation.
 *
 * For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
 * For underdetermined systems, this finds the minimum-norm solution for `x`.
 *
 * This currently uses Cholesky decomposition to solve the normal equations,
 * under the hood. The method is not as robust as QR or SVD.
 *
 * @param a coefficient matrix of shape `(M, N)`
 * @param b right-hand side of shape `(M,)` or `(M, K)`
 * @return least-squares solution of shape `(N,)` or `(N, K)`
 */
export function lstsq(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim !== 2)
    throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
  const [m, n] = a.shape;
  if (b.shape[0] !== m)
    throw new Error(
      `lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`,
    );
  using at = np.matrixTranspose(a);
  if (m <= n) {
    // Underdetermined or square system: A.T @ (A @ A.T)^-1 @ B
    using aat = np.matmul(a, at); // A @ A.T, shape (M, M)
    using l = cholesky(aat, { symmetrizeInput: false }); // L @ L.T = A @ A.T
    using lb = triangularSolve(l, b, { leftSide: true, lower: true }); // L^-1 @ B
    using llb = triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A @ A.T)^-1 @ B
    return np.matmul(at, llb); // A.T @ (A @ A.T)^-1 @ B
  } else {
    // Overdetermined system: (A.T @ A)^-1 @ A.T @ B
    using ata = np.matmul(at, a); // A.T @ A, shape (N, N)
    using l = cholesky(ata, { symmetrizeInput: false }); // L @ L.T = A.T @ A
    using atb = np.matmul(at, b); // A.T @ B
    using lb = triangularSolve(l, atb, { leftSide: true, lower: true }); // L^-1 @ A.T @ B
    return triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A.T @ A)^-1 @ A.T @ B
  }
}

export { matmul } from "./numpy";

/** Raise a square matrix to an integer power, via repeated squarings. */
export function matrixPower(a: ArrayLike, n: number): Array {
  if (!Number.isInteger(n))
    throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
  a = fudgeArray(a);
  const m = checkSquare("matrixPower", a);
  if (n === 0) {
    using eye = np.eye(m, { dtype: a.dtype });
    return np.broadcastTo(eye, a.shape);
  }
  const isInputOwned = n < 0;
  if (n < 0) {
    a = inv(a);
    n = -n;
  }
  let result: Array | null = null;
  let a2k = a; // a^(2^k)
  for (let k = 0; n; k++) {
    if (k > 0) {
      const prev = a2k;
      a2k = np.matmul(a2k, a2k);
      // Dispose old a2k unless it's the original input (not owned by us)
      if (prev !== a || isInputOwned) prev[Symbol.dispose]();
    }
    if (n % 2 === 1) {
      const prev = result;
      result = result === null ? a2k : np.matmul(result, a2k);
      if (prev !== null && prev !== a2k) prev[Symbol.dispose]();
    }
    n = Math.floor(n / 2);
  }
  // Dispose a2k if it's not the result and it's not the input
  if (a2k !== result && (a2k !== a || isInputOwned)) a2k[Symbol.dispose]();
  return result!;
}

export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";

/** Return sign and natural logarithm of the determinant of `a`. */
export function slogdet(a: ArrayLike): [Array, Array] {
  a = fudgeArray(a);
  const n = checkSquare("slogdet", a);
  const luResult = lax.linalg.lu(a);
  using lu = luResult[0];
  using pivots = luResult[1];
  using _perm = luResult[2];
  using indices = np.arange(n);
  using neq = pivots.notEqual(indices);
  using neqInt = neq.astype(np.int32);
  using parityBase = neqInt.sum(-1);
  using diag = lu.diagonal(0, -1, -2);
  using diagNeg = diag.less(0);
  using diagNegInt = diagNeg.astype(np.int32);
  using diagNegSum = diagNegInt.sum(-1);
  using parityTotal = parityBase.add(diagNegSum);
  using parity = parityTotal.mod(2);
  using neg2 = parity.mul(-2);
  const sign = neg2.add(1); // (-1)^parity — returned, NOT using
  using absDiag = np.abs(diag);
  using logDiag = np.log(absDiag);
  const logabsdet = logDiag.sum(-1); // returned, NOT using
  return [sign, logabsdet];
}

/**
 * Solve a linear system of equations.
 *
 * This solves a (batched) linear system of equations `a @ x = b` for `x` given
 * `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
 *
 * Gradient flows through the LU decomposition via the TriangularSolve JVP rule
 * (which correctly masks dA with triu()). Both ∂L/∂A and ∂L/∂b are supported.
 *
 * @param a - Coefficient matrix of shape `(..., N, N)`.
 * @param b - Values of shape `(N,)` or `(..., N, M)`.
 * @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
 */
export function solve(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  const n = checkSquare("solve", a);
  if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
  const bIs1d = b.ndim === 1;
  using d = new DisposableStack();
  if (bIs1d) {
    b = d.use(b.reshape([...b.shape, 1]));
  }
  if (b.shape[b.ndim - 2] !== n) {
    throw new Error(
      `solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`,
    );
  }
  const m = b.shape[b.ndim - 1];
  const batchDims = generalBroadcast(
    a.shape.slice(0, -2),
    b.shape.slice(0, -2),
  ) as number[];
  const aTargetShape = [...batchDims, n, n];
  if (
    a.shape.length !== aTargetShape.length ||
    a.shape.some((dim, i) => dim !== aTargetShape[i])
  ) {
    a = d.use(np.broadcastTo(a, aTargetShape));
  }
  const bTargetShape = [...batchDims, n, m as number];
  if (
    b.shape.length !== bTargetShape.length ||
    b.shape.some((dim, i) => dim !== bTargetShape[i])
  ) {
    b = d.use(np.broadcastTo(b, bTargetShape));
  }

  // Factor A. Gradient flows freely through the LU JVP (TriSolve triu mask fixed).
  // Stop gradient only on permRaw — permutation is integer-valued, no gradient.
  // Do NOT stop gradient on 'a' itself — stopGradient(a) creates a fully-known
  // PETracer that PE disposal cascades through, which would free 'a' early.
  const [lu, pivotsRaw, permRaw] = lax.linalg.lu(a);
  d.use(lu);
  d.use(pivotsRaw);
  d.use(permRaw);
  const permutation = lax.stopGradient(permRaw);

  // Build permutation matrix P (derived from stopGradient'd factorization).
  const arangeN = d.use(np.arange(n));
  const permR = d.use(permutation.reshape([...permutation.shape, 1]));
  const eq = d.use(arangeN.equal(permR));
  const P = d.use(eq.astype(b.dtype));

  // Solve x = A^{-1} b via the LU factorization.
  // Gradient flows through both b (TriSolve transpose rule) and A (TriSolve JVP).
  let x = luSolveWithP(lu, P, b, d);
  if (bIs1d) {
    d.use(x);
    x = np.squeeze(x, -1);
  }
  return x;
}

export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";

/**
 * Compute the vector norm of an array.
 *
 * @param x - Input array.
 * @param ord - Order of the norm (default 2). Supports `Infinity`, `-Infinity`, `0`, or any real number.
 * @param axis - Axis/axes to reduce over (default: all axes).
 * @param keepdims - Whether to keep reduced dimensions as size 1.
 * @returns The norm of `x`, reduced over the given axes.
 */
export function vectorNorm(
  x: ArrayLike,
  {
    ord = 2,
    axis = null,
    keepdims = false,
  }: {
    ord?: number;
    axis?: number | number[] | null;
    keepdims?: boolean;
  } = {},
): Array {
  x = fudgeArray(x);
  const ax = axis ?? null;

  if (ord === Infinity) {
    using absX = np.abs(x);
    return np.max(absX, ax, { keepdims });
  } else if (ord === -Infinity) {
    using absX = np.abs(x);
    return np.min(absX, ax, { keepdims });
  } else if (ord === 0) {
    using neq = x.notEqual(0);
    using casted = neq.astype(x.dtype);
    return casted.sum(ax, { keepdims });
  } else {
    using absX = np.abs(x);
    using powered = np.power(absX, ord);
    using summed = powered.sum(ax, { keepdims });
    return np.power(summed, 1 / ord);
  }
}
