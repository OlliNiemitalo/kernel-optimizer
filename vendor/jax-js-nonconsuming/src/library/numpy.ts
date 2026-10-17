// Port of the `jax.numpy` module, typically imported as `np`.

import { AluOp, DType, isFloatDtype, promoteTypes } from "../alu";
import { defaultDevice } from "../backend";
import type { Dim } from "../dim";
import {
  arange,
  Array,
  array,
  type ArrayLike,
  DTypeShapeAndDevice,
  eye,
  fudgeArray,
  full,
  fullLike as fullLikeTracer,
  identity,
  linspace,
  logspace,
  // eslint-disable-next-line no-restricted-imports -- re-exported as np.ones/np.zeros public API, not called internally.
  ones,
  onesLike as onesLikeTracer,
  tri,
  tril,
  triu,
  // eslint-disable-next-line no-restricted-imports -- re-exported as np.ones/np.zeros public API, not called internally.
  zeros,
  zerosLike as zerosLikeTracer,
} from "../frontend/array";
import * as core from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { moveaxis as moveaxisTracer } from "../frontend/vmap";
import { Pair } from "../shape";
import {
  checkAxis,
  DEBUG,
  deepEqual,
  generalBroadcast,
  prod as iprod,
  normalizeAxis,
  range,
  rep,
} from "../utils";
import * as lax from "./lax";
import { iinfo } from "./numpy";
import { finfo } from "./numpy";
import {
  computeEinsumPath,
  EinsumInput,
  parseEinsumExpression,
} from "./numpy/einsum";

export {
  arange,
  Array,
  type ArrayLike,
  array,
  DType,
  eye,
  full,
  identity,
  linspace,
  logspace,
  ones,
  promoteTypes,
  tri,
  tril,
  triu,
  zeros,
};

export * as fft from "./numpy-fft";
export * as linalg from "./numpy-linalg";

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const uint32 = DType.Uint32;
export const bool = DType.Bool;
export const float16 = DType.Float16;
export const float64 = DType.Float64;

export { finfo, iinfo } from "./numpy/dtype-info";

// Constants section

/** Euler's constant, `e = 2.7182818284590...` */
export const e = Math.E;

/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
export const eulerGamma = 0.5772156649015329;

/** Positive infinity. */
export const inf = Number.POSITIVE_INFINITY;

/** Floating-point representation of NaN. */
export const nan = Number.NaN;

/** This is Pi, `π = 3.14159265358979...` */
export const pi = Math.PI;

// Note: These primitive wrappers have fudged types.
//
// They can take any `TracerValue` and return any `Tracer` subclass based on the
// current stack of interpreters. But we hide that away from users to mimic
// JAX's composable tracing transformations.

/** @function Element-wise addition, with broadcasting. */
export const add = core.add as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Element-wise multiplication, with broadcasting. */
export const multiply = core.mul as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Numerical negative of every element of an array. */
export const negative = core.neg as (x: ArrayLike) => Array;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
export const reciprocal = core.reciprocal as (x: ArrayLike) => Array;
/** @function Round input down to the nearest integer. */
export const floor = core.floor as (x: ArrayLike) => Array;
/** @function Round input up to the nearest integer. */
export const ceil = core.ceil as (x: ArrayLike) => Array;
/** @function Element-wise sine function (takes radians). */
export const sin = core.sin as (x: ArrayLike) => Array;
/** @function Element-wise cosine function (takes radians). */
export const cos = core.cos as (x: ArrayLike) => Array;
/** @function Element-wise inverse sine function (inverse of sin). */
export const asin = core.asin as (x: ArrayLike) => Array;
/** @function Element-wise inverse tangent function (inverse of tan). */
export const atan = core.atan as (x: ArrayLike) => Array;
/** @function Calculate the exponential of all elements in the input array. */
export const exp = core.exp as (x: ArrayLike) => Array;
/** @function Calculate the natural logarithm of all elements in the input array. */
export const log = core.log as (x: ArrayLike) => Array;
/** @function Calculate the square root of all elements in the input array. */
export const sqrt = core.sqrt as (x: ArrayLike) => Array;
/** @function Return element-wise minimum of the input arrays. */
export const minimum = core.min as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Return element-wise maximum of the input arrays. */
export const maximum = core.max as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const greater = core.greater as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const less = core.less as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const equal = core.equal as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const notEqual = core.notEqual as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const greaterEqual = core.greaterEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** @function Compare two arrays element-wise. */
export const lessEqual = core.lessEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;

/** Compute element-wise logical AND. */
export function logicalAnd(x: ArrayLike, y: ArrayLike): Array {
  using xBool = astype(x, DType.Bool);
  using yBool = astype(y, DType.Bool);
  return xBool.mul(yBool);
}

/** Compute element-wise logical OR. */
export function logicalOr(x: ArrayLike, y: ArrayLike): Array {
  using xBool = astype(x, DType.Bool);
  using yBool = astype(y, DType.Bool);
  return xBool.add(yBool);
}

/** Compute element-wise logical XOR. */
export function logicalXor(x: ArrayLike, y: ArrayLike): Array {
  using xBool = astype(x, DType.Bool);
  using yBool = astype(y, DType.Bool);
  return notEqual(xBool, yBool);
}

/** Compute element-wise logical NOT. */
export function logicalNot(x: ArrayLike): Array {
  using xBool = astype(x, DType.Bool);
  return notEqual(xBool, true);
}

/** Compute element-wise bitwise AND. */
export function bitwiseAnd(x: ArrayLike, y: ArrayLike): Array {
  return core.bitCombine(x, y, "and") as Array;
}

/** Compute element-wise bitwise OR. */
export function bitwiseOr(x: ArrayLike, y: ArrayLike): Array {
  return core.bitCombine(x, y, "or") as Array;
}

/** Compute element-wise bitwise XOR. */
export function bitwiseXor(x: ArrayLike, y: ArrayLike): Array {
  return core.bitCombine(x, y, "xor") as Array;
}

/** Compute element-wise bitwise NOT (inversion). */
export function invert(x: ArrayLike): Array {
  const arr = fudgeArray(x);
  let allOnes: ArrayLike;
  switch (arr.dtype) {
    case DType.Bool:
      allOnes = true;
      break;
    case DType.Uint32:
      allOnes = 0xffffffff;
      break;
    case DType.Int32:
      allOnes = -1;
      break;
    default:
      throw new TypeError(`invert: unsupported dtype ${arr.dtype}`);
  }
  return core.bitCombine(arr, allOnes, "xor") as Array;
}

export { invert as bitwiseInvert, invert as bitwiseNot };

/** Compute element-wise left bit shift. */
export function leftShift(x: ArrayLike, y: ArrayLike): Array {
  return core.bitShift(x, y, "shl") as Array;
}

/** Compute element-wise right bit shift. */
export function rightShift(x: ArrayLike, y: ArrayLike): Array {
  return core.bitShift(x, y, "shr") as Array;
}

export { leftShift as bitwiseLeftShift, rightShift as bitwiseRightShift };

/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
export const where = core.where as (
  cond: ArrayLike,
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/**
 * @function
 * Permute the dimensions of an array. Defaults to reversing the axis order.
 */
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[],
) => Array;
/**
 * @function
 * Give a new shape to an array without changing its data.
 *
 * One shape dimension can be -1. In this case, the value is inferred from the
 * length of the array and remaining dimensions.
 */
export const reshape = core.reshape as (x: ArrayLike, shape: number[]) => Array;
/**
 * @function
 * Move axes of an array to new positions. Other axes retain original order.
 */
export const moveaxis = moveaxisTracer as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;
/**
 * @function
 * Add padding (zeros) to an array.
 *
 * The `width` argument is either an integer or pair of integers, in which case
 * all axes are padded with the same width. Or if it is an array of pairs, each
 * pair specifies the padding for its corresponding axis.
 */
export const pad = core.pad as (
  x: ArrayLike,
  width: number | Pair | Pair[] | Record<number, Pair>,
) => Array;

/**
 * @function
 * Return the number of dimensions of an array.
 */
export const ndim = core.ndim as (x: ArrayLike) => number;

/** @function Return the shape of an array. */
export const shape = core.getShape as (x: ArrayLike) => number[];

/**
 * @function
 * Return an array of zeros with the same shape and type as a given array.
 */
export const zerosLike = zerosLikeTracer as (
  a: ArrayLike,
  opts?: DTypeShapeAndDevice,
) => Array;

/**
 * @function
 * Return an array of ones with the same shape and type as a given array.
 */
export const onesLike = onesLikeTracer as (
  a: ArrayLike,
  opts?: DTypeShapeAndDevice,
) => Array;

/**
 * @function
 * Return a full array with the same shape and type as a given array.
 */
export const fullLike = fullLikeTracer as (
  a: ArrayLike,
  fillValue: number | boolean | Array,
  opts?: DTypeShapeAndDevice,
) => Array;

/**
 * Return the number of elements in an array, optionally along an axis.
 */
export function size(a: ArrayLike, axis?: number): number {
  const s = shape(a);
  return axis === undefined ? iprod(s) : s[axis];
}

/** Convert an array to a specified dtype. */
export function astype(a: ArrayLike, dtype: DType): Array {
  return fudgeArray(a).astype(dtype);
}

/** Sum of the elements of the array over a given axis, or axes. */
export function sum(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Add, axis, opts) as Array;
}

/** Product of the array elements over a given axis. */
export function prod(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Mul, axis, opts) as Array;
}

/** Return the minimum of array elements along a given axis. */
export function min(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Min, axis, opts) as Array;
}

/** Return the maximum of array elements along a given axis. */
export function max(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Max, axis, opts) as Array;
}

/**
 * Test whether any array element along a given axis evaluates to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
export function any(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).any(axis, opts);
}

/**
 * Test whether all array elements along a given axis evaluate to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
export function all(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).all(axis, opts);
}

/** Return the peak-to-peak range along a given axis (`max - min`). */
export function ptp(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  using maxVal = max(a, axis, opts);
  using minVal = min(a, axis, opts);
  return maxVal.sub(minVal);
}

/** Compute the average of the array elements along the specified axis. */
export function mean(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).mean(axis, opts);
}

/**
 * Compute the weighted average along the specified axis.
 *
 * If no axis is specified, mean is computed along all the axes. The weights
 * should have shape matching that of `a`, or if an axis is specified, it should
 * match the shape along those axes.
 */
export function average(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: { weights?: ArrayLike } & core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  if (opts?.weights == null) {
    return mean(a, axis, opts);
  }
  const weights = fudgeArray(opts.weights);
  axis = normalizeAxis(axis, ndim(a));

  const wShape = weights.shape;
  const aShape = a.shape;

  if (deepEqual(wShape, aShape)) {
    // weights match shape of a exactly
    using scl = sum(weights, axis, opts);
    using prod = multiply(a, weights) as Array;
    using num = sum(prod, axis, opts);
    return num.div(scl);
  } else if (
    axis.length === 1 &&
    wShape.length === 1 &&
    wShape[0] === aShape[axis[0]]
  ) {
    // 1-D weights along a single axis: reshape to broadcast
    const broadcastShape = aShape.map((_, i) =>
      i === axis[0] ? wShape[0] : 1,
    );
    using wReshaped = reshape(weights, broadcastShape);
    using scl = sum(wReshaped, axis, opts);
    using prod = multiply(a, wReshaped) as Array;
    using num = sum(prod, axis, opts);
    return num.div(scl);
  } else {
    throw new Error(
      `average: weights shape ${JSON.stringify(wShape)} is not compatible with array shape ${JSON.stringify(aShape)} and axis ${JSON.stringify(axis)}`,
    );
  }
}

/**
 * Returns the indices of the minimum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
export function argmin(
  a: ArrayLike,
  axis?: number,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  let ravelled: Array | undefined;
  if (axis === undefined) {
    ravelled = a.ravel();
    a = ravelled;
    axis = 0; // Default to the first axis of the flattened array.
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  try {
    const shape = a.shape;
    using minVal = min(a, axis, { keepdims: true });
    using isMax = equal(a, minVal);
    using length = array(shape[axis], { dtype: int32, device: a.device });
    // Index by length-i instead of i, so we can take the max and get the first i.
    using range = arange(shape[axis], 0, -1, {
      dtype: int32,
      device: a.device,
    });
    using reshaped = range.reshape([
      shape[axis],
      ...rep(shape.length - axis - 1, 1),
    ]);
    using isMaxInt = isMax.astype(DType.Int32);
    using idx = isMaxInt.mul(reshaped);
    using maxIdx = max(idx, axis, opts);
    return length.sub(maxIdx);
  } finally {
    ravelled?.dispose();
  }
}

/**
 * Returns the indices of the maximum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
export function argmax(
  a: ArrayLike,
  axis?: number,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  let ravelled: Array | undefined;
  if (axis === undefined) {
    ravelled = a.ravel();
    a = ravelled;
    axis = 0; // Default to the first axis of the flattened array.
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  try {
    const shape = a.shape;
    using maxVal = max(a, axis, { keepdims: true });
    using isMax = equal(a, maxVal);
    using length = array(shape[axis], { dtype: int32, device: a.device });
    // Index by length-i instead of i, so we can take the max and get the first i.
    using range = arange(shape[axis], 0, -1, {
      dtype: int32,
      device: a.device,
    });
    using reshaped = range.reshape([
      shape[axis],
      ...rep(shape.length - axis - 1, 1),
    ]);
    using isMaxInt = isMax.astype(DType.Int32);
    using idx = isMaxInt.mul(reshaped);
    using maxIdx = max(idx, axis, opts);
    return length.sub(maxIdx);
  } finally {
    ravelled?.dispose();
  }
}

/**
 * Cumulative sum of elements along an axis.
 *
 * Uses `lax.associativeScan` for O(N log N) work with O(log N) parallel depth,
 * replacing the previous O(N²) tril+matmul approach.
 */
export function cumsum(a: ArrayLike, axis?: number): Array {
  a = fudgeArray(a);
  const inputA = a;
  if (axis === undefined) {
    a = a.ravel();
    axis = 0;
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  try {
    return lax.associativeScan((x: Array, y: Array) => x.add(y), a, { axis });
  } finally {
    if (a !== inputA) a.dispose();
  }
}

export { cumsum as cumulativeSum };

/** Reverse the elements in an array along the given axes. */
export function flip(x: ArrayLike, axis: core.Axis = null): Array {
  const nd = ndim(x);
  axis = normalizeAxis(axis, nd);
  return core.flip(x, axis) as Array;
}

/**
 * Split an array into multiple sub-arrays along an axis.
 *
 * @param a - The input array to split.
 * @param indicesOrSections - If an integer, it indicates the number of equal
 * sections to create along the specified axis. If a list of integers, it
 * specifies the indices at which to split the array.
 * @param axis - The axis along which to split the array. Default is 0.
 */
export function split(
  a: ArrayLike,
  indicesOrSections: number | number[],
  axis: number = 0,
): Array[] {
  a = fudgeArray(a);
  axis = checkAxis(axis, a.ndim);
  const size = a.shape[axis];
  let sizes: number[];
  if (typeof indicesOrSections === "number") {
    if (size % indicesOrSections !== 0) {
      throw new Error(
        `Array of size ${size} cannot be split into ${indicesOrSections} equal parts`,
      );
    }
    const partSize = size / indicesOrSections;
    sizes = rep(indicesOrSections, partSize);
  } else {
    const indices = indicesOrSections.map((i) => (i < 0 ? i + size : i));
    sizes = [indices[0]];
    for (let i = 1; i < indices.length; i++)
      sizes.push(indices[i] - indices[i - 1]);
    sizes.push(size - indices[indices.length - 1]);
  }
  // Split in groups of up to 8 outputs, as the transpose rule turns into a
  // Concatenate primitive that has limited input arguments.
  const results: Array[] = [];
  for (let i = 0; i < sizes.length; i += 7) {
    if (i === sizes.length) {
      results.push(a);
      break;
    } else if (i + 8 >= sizes.length) {
      results.push(...(core.split(a, axis, sizes.slice(i)) as Array[]));
      break;
    } else {
      const groupSizes = [
        ...sizes.slice(i, i + 7),
        sizes.slice(i + 7).reduce((x, y) => x + y, 0),
      ];
      const outs = core.split(a, axis, groupSizes) as Array[];
      results.push(...outs.slice(0, -1));
      a = outs[outs.length - 1];
    }
  }
  return results;
}

/**
 * Join a sequence of arrays along an existing axis.
 *
 * The arrays must have the same shape, except in the dimension corresponding to
 * `axis` (the first, by default).
 *
 * No scalars can be passed to this function, as the axis is then ambiguous.
 */
export function concatenate(xs: Array[], axis: number = 0): Array {
  if (xs.length === 0)
    throw new Error("Need at least one array to concatenate");
  const shapes = xs.map(shape);
  axis = checkAxis(axis, shapes[0].length);
  for (let i = 1; i < shapes.length; i++) {
    if (
      shapes[i].length !== shapes[0].length ||
      !shapes[i].every((d, j) => j === axis || d === shapes[0][j])
    ) {
      throw new Error(
        `Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`,
      );
    }
  }
  // Concatenate the arrays in groups of 8 to avoid possibly exceeding the
  // `maxArgs` of the backend.
  let result = xs[0];
  for (let i = 1; i < xs.length; i += 7) {
    const group = xs.slice(i, i + 7);
    const prev = result;
    result = core.concatenate([result, ...group], axis) as Array;
    if (prev !== xs[0]) prev.dispose();
  }
  return result;
}

/**
 * Join a sequence of arrays along a new axis.
 *
 * The `axis` parameter specifies the index of the new axis in the dimensions of
 * the result. For example, if `axis=0` it will be the first dimension and if
 * `axis=-1` it will be the last dimension.
 *
 * All shapes must have the same shape.
 */
export function stack(xs: ArrayLike[], axis: number = 0): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to stack");
  }
  const shapes = xs.map((x) => shape(x));
  if (!shapes.every((s) => deepEqual(s, shapes[0]))) {
    throw new Error(
      `Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`,
    );
  }
  axis = checkAxis(axis, shapes[0].length + 1); // +1 for the new axis
  const newShape = shapes[0].toSpliced(axis, 0, 1);
  // Track fudgeArray intermediates separately from reshape results.
  // When xs contains non-Array elements (numbers, TypedArrays), fudgeArray
  // creates a new Array, then .reshape() creates a view. Both must be
  // disposed, but they are different objects.
  const fudged = xs.map((x) => fudgeArray(x));
  const newlyCreated = fudged.filter((f, i) => f !== xs[i]);
  const newArrays = fudged.map((x) => x.reshape(newShape));
  try {
    return concatenate(newArrays, axis) as Array;
  } finally {
    for (const a of newArrays) a[Symbol.dispose]();
    for (const a of newlyCreated) a[Symbol.dispose]();
  }
}

/**
 * Horizontally stack arrays. Inputs are promoted to rank at least 1, then
 * concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
 */
export function hstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to hstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    return stack(xs); // Rank-0 arrays become rank-1
  } else if (nds[0] === 1) {
    return concatenate(xs as Array[]); // Rank-1 arrays become rank-1
  } else {
    // Rank-2 or higher arrays are concatenated along axis 1
    return concatenate(xs as Array[], 1);
  }
}

/**
 * Vertically stack arrays. Inputs are promoted to rank at least 2, then
 * concatenated along axis 0.
 */
export function vstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to vstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    using stacked = stack(xs);
    return stacked.reshape([-1, 1]); // Rank-0 arrays become rank-2
  } else if (nds[0] === 1) {
    return stack(xs); // Rank-1 arrays become rank-2
  } else {
    // Rank-2 or higher arrays are concatenated along axis 0
    return concatenate(xs as Array[]);
  }
}

/**
 * Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
 * concatenated along axis 2.
 */
export function dstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to dstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    using stacked = stack(xs);
    return stacked.reshape([1, 1, -1]); // Rank-0 arrays become rank-3
  } else if (nds[0] === 1) {
    using ret = stack(xs, -1); // Tricky!
    return ret.reshape([1, ...ret.shape]);
  } else if (nds[0] === 2) {
    return stack(xs, -1);
  } else {
    return concatenate(xs as Array[], 2);
  }
}

/**
 * Stack arrays column-wise. Inputs are promoted to rank at least 2, then
 * concatenated along axis 1.
 */
export function columnStack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to columnStack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    using stacked = stack(xs);
    return stacked.reshape([1, -1]); // Rank-0 arrays become rank-2
  } else if (nds[0] === 1) {
    return stack(xs, -1); // Rank-1 arrays become rank-2
  } else {
    // Rank-2 or higher arrays are concatenated along axis 1
    return concatenate(xs as Array[], 1);
  }
}

/**
 * Assemble an nd-array from nested lists of blocks.
 *
 * `block` is a generalization of `vstack` and `hstack`. Blocks in the
 * innermost lists are concatenated along the last axis, then those results
 * are concatenated along the second-to-last axis, and so on.
 *
 * For the common 2-D case, `block([[A, B], [C, D]])` is equivalent to
 * `vstack([hstack([A, B]), hstack([C, D])])`.
 *
 * @param arrays - A nested list of arrays (or scalars). Nesting depth
 *   determines which axes are concatenated. All blocks in a row must agree on
 *   dimensions perpendicular to concatenation.
 *
 * @returns The assembled array.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * const A = np.ones([2, 2]);
 * const B = np.zeros([2, 3]);
 * const C = np.zeros([3, 2]);
 * const D = np.ones([3, 3]);
 * const M = np.block([[A, B], [C, D]]);
 * // M has shape [5, 5]
 *
 * A.dispose(); B.dispose(); C.dispose(); D.dispose(); M.dispose();
 * ```
 */
export function block(arrays: any): Array {
  return _blockImpl(arrays, 0);
}

function _blockImpl(arrays: any, depth: number): Array {
  // Base case: a single array or scalar
  if (!globalThis.Array.isArray(arrays)) {
    return fudgeArray(arrays);
  }
  if (arrays.length === 0) {
    throw new Error("Need at least one array to block");
  }
  if (arrays.length === 1) {
    // Single-element nests transfer ownership directly to the caller.
    return _blockImpl(arrays[0], depth + 1);
  }

  // Recursively process each element, then concatenate along the appropriate
  // axis. Innermost lists (highest depth) concatenate along the last axis,
  // so we track the depth and compute the axis at the end.
  const processed: Array[] = [];
  using created = new DisposableStack();
  for (const item of arrays) {
    const arr = _blockImpl(item, depth + 1);
    processed.push(arr);
    // Track only arrays we created recursively, not direct user inputs.
    if (globalThis.Array.isArray(item)) created.use(arr);
  }

  // Determine the axis: depth 0 → axis 0, depth 1 → axis 1, etc.
  // But we need to ensure arrays have enough dimensions.
  // NumPy/JAX's block infers the total nesting depth first, then pads
  // arrays to that rank.
  const maxNdim = Math.max(...processed.map((a) => a.ndim));
  const axis = depth;

  // Promote all arrays to at least max rank if needed.
  const promoted: Array[] = [];
  using promotedTemps = new DisposableStack();
  for (const arr of processed) {
    if (arr.ndim < maxNdim) {
      // Prepend dimensions to match max rank.
      const newShape = [
        ...new globalThis.Array(maxNdim - arr.ndim).fill(1),
        ...arr.shape,
      ];
      promoted.push(promotedTemps.use(arr.reshape(newShape)));
    } else {
      promoted.push(arr);
    }
  }

  return concatenate(promoted, axis);
}

/** Flip an array vertically (axis=0). */
export function flipud(x: ArrayLike): Array {
  return flip(x, 0);
}

/** Flip an array horizontally (axis=1). */
export function fliplr(x: ArrayLike): Array {
  return flip(x, 1);
}

export { transpose as permuteDims };

/** Interchange two axes of an array. */
export function swapaxes(a: ArrayLike, axis1: number, axis2: number): Array {
  a = fudgeArray(a);
  axis1 = checkAxis(axis1, a.ndim);
  axis2 = checkAxis(axis2, a.ndim);
  if (axis1 === axis2) return a;
  const perm = range(a.ndim);
  perm[axis1] = axis2;
  perm[axis2] = axis1;
  return transpose(a, perm);
}

/** Transpose the last two dimensions of an array. */
export function matrixTranspose(a: ArrayLike): Array {
  if (ndim(a) < 2)
    throw new Error(`matrixTranspose: input array must be at least 2D`);
  return moveaxis(a, -1, -2);
}

/** Return a 1-D flattened array containing the elements of the input. */
export function ravel(a: ArrayLike): Array {
  return fudgeArray(a).ravel();
}

/** Remove one or more length-1 axes from an array. */
export function squeeze(a: ArrayLike, axis: core.Axis = null): Array {
  const as = shape(a);
  if (axis === null) {
    axis = range(as.length).filter((i) => as[i] === 1);
  } else if (typeof axis === "number") {
    axis = [axis];
  }
  axis = axis.map((a) => checkAxis(a, as.length));
  for (const a of axis) {
    if (as[a] !== 1) throw new Error("Cannot squeeze axis with size != 1");
  }
  const newShape = as.filter((_, i) => !axis.includes(i));
  return reshape(a, newShape);
}

/**
 * Expand the shape of an array by inserting new axes of length 1.
 *
 * @param a - Input array.
 * @param axis - Position(s) in the expanded axes where the new axis (or axes)
 *   is placed. Can be a single integer or an array of integers.
 * @returns Array with the number of dimensions increased.
 *
 * @example
 * ```ts
 * using x = np.array([1, 2]);
 * using x0 = np.expandDims(x, 0); // Shape [1, 2]
 * using x1 = np.expandDims(x, 1); // Shape [2, 1]
 * using x2 = np.expandDims(x, [0, 2]); // Shape [1, 2, 1]
 * ```
 */
export function expandDims(a: ArrayLike, axis: number | number[]): Array {
  const as = shape(a);
  axis = typeof axis === "number" ? [axis] : axis;
  axis = normalizeAxis(axis, as.length + axis.length);

  const newShape: number[] = [];
  let srcIdx = 0;
  for (let i = 0; i < as.length + axis.length; i++) {
    if (axis.includes(i)) {
      newShape.push(1);
    } else {
      newShape.push(as[srcIdx++]);
    }
  }
  return reshape(a, newShape);
}

/**
 * Repeat each element of an array after themselves.
 *
 * If no axis is provided, use the flattened input array, and return a flat
 * output array.
 */
export function repeat(a: ArrayLike, repeats: number, axis?: number): Array {
  if (!Number.isInteger(repeats) || repeats < 0) {
    throw new Error(
      `repeat: repeats must be a non-negative integer, got ${repeats}`,
    );
  }
  a = fudgeArray(a);
  let ravelled: Array | undefined;
  if (axis === undefined) {
    ravelled = ravel(a);
    a = ravelled;
    axis = 0;
  }
  axis = checkAxis(axis, a.ndim);
  if (repeats === 1) {
    ravelled?.dispose();
    return a.reshape(a.shape);
  }
  const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
  const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
  using broadcasted = core.broadcast(a, broadcastedShape, [axis + 1]) as Array;
  const result = broadcasted.reshape(finalShape);
  ravelled?.dispose();
  return result;
}

/**
 * Construct an array by repeating A the number of times given by reps.
 *
 * If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
 * integers, the resulting array will have a shape of `(reps[0] * d1,
 * reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
 */
export function tile(a: ArrayLike, reps: number | number[]): Array {
  a = fudgeArray(a);
  if (typeof reps === "number") reps = [reps];
  if (!reps.every((r) => Number.isInteger(r) && r >= 0)) {
    throw new Error(
      `tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`,
    );
  }
  // Prepend 1s to match dimensions
  const ndiff = reps.length - a.ndim;
  let reshaped: Array | undefined;
  if (ndiff > 0) {
    reshaped = a.reshape([...rep(ndiff, 1), ...a.shape]);
    a = reshaped;
  }
  if (ndiff < 0) reps = [...rep(-ndiff, 1), ...reps];
  // Build broadcasted shape by interleaving reps where > 1: [r1, d1, r2, d2, ...]
  const broadcastedShape: number[] = [];
  const broadcastAxes: number[] = [];
  for (let i = 0; i < a.ndim; i++) {
    if (reps[i] > 1) {
      broadcastedShape.push(reps[i]);
      broadcastAxes.push(broadcastedShape.length - 1);
    }
    broadcastedShape.push(a.shape[i]);
  }
  const finalShape = a.shape.map((d, i) => reps[i] * d);
  using broadcasted = core.broadcast(
    a,
    broadcastedShape,
    broadcastAxes,
  ) as Array;
  const result = broadcasted.reshape(finalShape);
  reshaped?.dispose();
  return result;
}

/**
 * Broadcast an array to a shape, with NumPy-style broadcasing rules.
 *
 * In other words, this lets you append axes to the left, and/or expand
 * dimensions where the shape is 1.
 */
export function broadcastTo(a: ArrayLike, shape: number[]) {
  const nd = ndim(a);
  if (shape.length < nd) {
    throw new Error(
      `broadcastTo: target shape ${JSON.stringify(shape)} has fewer dimensions than input array: ${nd}`,
    );
  }
  return core.broadcast(a, shape, range(shape.length - nd)) as Array;
}

/** Broadcast input shapes to a common output shape. */
export function broadcastShapes(...shapes: number[][]): number[] {
  if (shapes.length === 0) return [];
  return (shapes as Dim[][]).reduce(generalBroadcast) as number[];
}

/** Broadcast arrays to a common shape. */
export function broadcastArrays(...arrays: ArrayLike[]): Array[] {
  const shapes = arrays.map((a) => shape(a));
  const outShape = broadcastShapes(...shapes);
  return arrays.map((a) => broadcastTo(a, outShape));
}

/**
 * Return specified diagonals.
 *
 * If a is 2D, return the diagonal of the array with the given offset. If a is
 * 3D or higher, compute diagonals along the two given axes (default: 0, 1).
 *
 * This returns a view over the existing array. The shape of the resulting array
 * is determined by removing the two axes along which the diagonal is taken,
 * then appending a new axis to the right with holding the diagonals.
 */
export function diagonal(
  a: ArrayLike,
  offset?: number,
  axis1?: number,
  axis2?: number,
): Array {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}

/**
 * Extract a diagonal or construct a diagonal array.
 *
 * If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
 * array, return a 2D array with v on the k-th diagonal.
 */
export function diag(v: ArrayLike, k = 0): Array {
  const a = fudgeArray(v);
  if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
  if (a.ndim === 1) {
    const n = a.shape[0];
    using eyeN = eye(n, { dtype: a.dtype });
    using mask = eyeN.equal(1);
    using zeros = zerosLike(a);
    if (k === 0) {
      return where(mask, a, zeros);
    }
    using ret = where(mask, a, zeros);
    if (k > 0) {
      return pad(ret, [
        [0, k],
        [k, 0],
      ]);
    } else {
      return pad(ret, [
        [-k, 0],
        [0, -k],
      ]);
    }
  } else if (a.ndim === 2) {
    return diagonal(a, k);
  } else {
    throw new Error("numpy.diag only supports 1D and 2D arrays");
  }
}

/** Calculate the sum of the diagonal of an array along the given axes. */
export function trace(a: ArrayLike, offset = 0, axis1 = 0, axis2 = 1): Array {
  using diag = diagonal(a, offset, axis1, axis2);
  return diag.sum(-1);
}

/**
 * Return a sorted copy of an array.
 *
 * The array is sorted along a specified axis (the last by default). This may be
 * an unstable sort, and it dispatches to device-specific implementation.
 */
export function sort(a: ArrayLike, axis: number = -1): Array {
  return fudgeArray(a).sort(axis);
}

/**
 * Return indices that would sort an array. Unlike `sort`, this is guaranteed to
 * be a stable sorting algorithm; it always returns the smaller index first in
 * event of ties.
 *
 * Returns an array of `int32` indices.
 *
 * The array is sorted along a specified axis (the last by default).
 */
export function argsort(a: ArrayLike, axis: number = -1): Array {
  return fudgeArray(a).argsort(axis);
}

/**
 * Take elements from an array along an axis.
 *
 * This is equivalent to advanced indexing with integer indices over that
 * numbered axis. By default, the flattened array is used.
 */
export function take(
  a: ArrayLike,
  indices: ArrayLike,
  axis: number | null = null,
): Array {
  let ravelled: Array | undefined;
  if (axis === null) {
    ravelled = ravel(a);
    a = ravelled;
    axis = 0;
  }
  axis = checkAxis(axis, ndim(a));
  const result = core.gather(a, [indices], [axis], axis) as Array;
  ravelled?.dispose();
  return result;
}

/**
 * Return if two arrays are element-wise equal within a tolerance.
 *
 * The formula used is `|actual - expected| <= atol + rtol * |expected|`, with
 * NaN values comparing equal if `equalNaN` is true.
 */
export function allclose(
  actual: Parameters<typeof array>[0],
  expected: Parameters<typeof array>[0],
  options?: { rtol?: number; atol?: number; equalNaN?: boolean },
): boolean {
  const { rtol = 1e-5, atol = 1e-7, equalNaN = false } = options ?? {};

  const x = array(actual);
  const y = array(expected);
  // Only dispose arrays we created (not caller's existing arrays).
  const xOwned = x !== actual;
  const yOwned = y !== expected;
  try {
    if (!deepEqual(x.shape, y.shape)) {
      return false;
    }
    const xData = x.dataSync();
    const yData = y.dataSync();
    for (let i = 0; i < xData.length; i++) {
      if (
        equalNaN
          ? isNaN(xData[i]) !== isNaN(yData[i])
          : isNaN(xData[i]) || isNaN(yData[i])
      ) {
        return false;
      }
      if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) {
        return false;
      }
    }
    return true;
  } finally {
    if (xOwned) x.dispose();
    if (yOwned) y.dispose();
  }
}

/**
 * Check if two arrays are element-wise equal.
 *
 * Returns False if the arrays have different shapes. If `equalNaN` is True,
 * NaNs in the same position are considered equal.
 */
export function arrayEqual(
  a1: ArrayLike,
  a2: ArrayLike,
  opts?: { equalNaN?: boolean },
): Array {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  if (!deepEqual(a1.shape, a2.shape)) {
    return array(false);
  }
  if (opts?.equalNaN) {
    // Where both are NaN, treat as equal; otherwise use element-wise equal
    using nanA1 = isnan(a1);
    using nanA2 = isnan(a2);
    using nanMask = nanA1.mul(nanA2);
    using eq = a1.equal(a2);
    using merged = where(nanMask, true, eq) as Array;
    return merged.all();
  }
  using eq = a1.equal(a2);
  return eq.all();
}

/**
 * Check if two arrays are element-wise equal after broadcasting.
 *
 * Unlike `arrayEqual`, this allows inputs with different but
 * broadcast-compatible shapes.
 */
export function arrayEquiv(a1: ArrayLike, a2: ArrayLike): Array {
  a1 = fudgeArray(a1);
  a2 = fudgeArray(a2);
  try {
    const [b1, b2] = broadcastArrays(a1, a2);
    using _b1 = b1;
    using _b2 = b2;
    using eq = b1.equal(b2);
    return eq.all();
  } catch {
    return array(false);
  }
}

/** Matrix product of two arrays with NumPy-style broadcasting of batch dims. */
export function matmul(x: ArrayLike, y: ArrayLike): Array {
  if (ndim(x) === 0 || ndim(y) === 0) {
    throw new Error("matmul: x and y must be at least 1D");
  }
  ((x = x as Array), (y = y as Array));
  if (x.ndim === 1 && y.ndim === 1) {
    return core.dot(x, y) as Array;
  }
  if (x.ndim === 1) {
    // [K] @ [..., K, M] → [..., M]: prepend 1, matmul, squeeze
    using x2 = reshape(x, [1, x.shape[0]]);
    using result = matmul(x2, y);
    return squeeze(result, -2);
  }
  if (y.ndim === 1) {
    // [..., N, K] @ [K] → [..., N]: append 1, matmul, squeeze
    using y2 = reshape(y, [y.shape[0], 1]);
    using result = matmul(x, y2);
    return squeeze(result, -1);
  }
  // Both >= 2D: x [...batchX, N, K], y [...batchY, K, M]
  const xBatch = x.shape.slice(0, -2);
  const yBatch = y.shape.slice(0, -2);
  const batchShape = broadcastShapes(xBatch, yBatch);
  const numBatchDims = batchShape.length;

  // Route unbatched 2D float matmul through tiledMatmul on WebGPU inside JIT.
  // In eager mode, blockMap iterates block-by-block in JS which is very slow
  // for large matrices; only the fused shader path (JIT) is fast.
  if (
    core.insideTrace() &&
    numBatchDims === 0 &&
    x.dtype === y.dtype &&
    isFloatDtype(x.dtype) &&
    defaultDevice() === "webgpu" &&
    typeof x.shape[0] === "number" &&
    typeof x.shape[1] === "number" &&
    typeof y.shape[1] === "number" &&
    Math.min(
      x.shape[0] as number,
      x.shape[1] as number,
      y.shape[1] as number,
    ) >= 512
  ) {
    return lax.tiledMatmul(x, y);
  }

  // Broadcast batch dims if needed (O(1) ShapeTracker views, no allocation)
  const xTarget = [...batchShape, x.shape.at(-2)!, x.shape.at(-1)!];
  const yTarget = [...batchShape, y.shape.at(-2)!, y.shape.at(-1)!];
  const xNeedsBroadcast = !deepEqual(x.shape, xTarget);
  const yNeedsBroadcast = !deepEqual(y.shape, yTarget);
  const xB = xNeedsBroadcast ? broadcastTo(x, xTarget) : x;
  const yB = yNeedsBroadcast ? broadcastTo(y, yTarget) : y;
  try {
    return lax.dot(xB, yB, {
      lhsContractingDims: [-1],
      rhsContractingDims: [-2],
      lhsBatchDims: range(numBatchDims),
      rhsBatchDims: range(numBatchDims),
    });
  } finally {
    if (xNeedsBroadcast) (xB as Array).dispose();
    if (yNeedsBroadcast) (yB as Array).dispose();
  }
}

/** Matrix-vector product. x1 is [..., M, N], x2 is [..., N] → [..., M]. */
export function matvec(x1: ArrayLike, x2: ArrayLike): Array {
  if (ndim(x1) < 2 || ndim(x2) < 1)
    throw new Error("matvec: x1 must be at least 2D and x2 at least 1D");
  return einsum("...mn,...n->...m", x1, x2);
}

/** Vector-matrix product. x1 is [..., N], x2 is [..., N, M] → [..., M]. */
export function vecmat(x1: ArrayLike, x2: ArrayLike): Array {
  if (ndim(x1) < 1 || ndim(x2) < 2)
    throw new Error("vecmat: x1 must be at least 1D and x2 at least 2D");
  return einsum("...n,...nm->...m", x1, x2);
}

/** Dot product of two arrays. */
export function dot(x: ArrayLike, y: ArrayLike): Array {
  if (ndim(x) === 0 || ndim(y) === 0) {
    // Standard, scalar multiplication
    return multiply(x, y);
  }
  ((x = x as Array), (y = y as Array));
  if (y.ndim === 1) {
    // Matrix-vector product
    return core.dot(x, y) as Array;
  }
  // Otherwise, this is the "sum product" between the last axis of x, and the
  // second-to-last axis of y. (y.ndim >= 2)
  //
  // dot(x, y)[i,j,k,m] = sum(x[i,j,:] * y[k,:,m])
  return lax.dot(x, y, {
    lhsContractingDims: [-1],
    rhsContractingDims: [-2],
  });
}

/**
 * Compute the tensor dot product of two N-dimensional arrays.
 *
 * The behavior is determined by `axes`. If an integer `k`, sum over the last
 * `k` axes of x and the first `k` axes of y. If a tuple, then the first array
 * corresponds to the axes of x and the second to the axes of y.
 */
export function tensordot(
  x: ArrayLike,
  y: ArrayLike,
  axes: number | [number[], number[]] = 2,
): Array {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (typeof axes === "number") axes = [range(-axes, 0), range(axes)];
  return lax.dot(x, y, {
    lhsContractingDims: axes[0],
    rhsContractingDims: axes[1],
  });
}

// ── Einsum fast-path for common batch-matmul patterns ────────────────────
// Skips parseEinsumExpression, computeEinsumPath, and processSingleTensor
// for patterns that lower directly to matmul / swapaxes. Returns null when
// the pattern is not recognised — caller falls through to general einsum.

function einsumFastPath(subscripts: string, args: any[]): Array | null {
  const nOps = args.length - 1; // first arg is the subscripts string

  // 1-operand: batch transpose
  if (nOps === 1) {
    if (subscripts === "nij->nji" || subscripts === "ij->ji") {
      return swapaxes(fudgeArray(args[1]), -2, -1);
    }
    return null;
  }

  // 2-operand: batch matmul variants
  if (nOps === 2) {
    const a = fudgeArray(args[1]);
    const b = fudgeArray(args[2]);
    switch (subscripts) {
      // Standard matmul (batch or unbatched)
      case "nij,njk->nik":
      case "ij,jk->ik":
      case "ij,njk->nik":
      case "nij,jk->nik":
        return matmul(a, b);
      // Matmul with B transposed
      case "nij,nkj->nik":
      case "ij,kj->ik":
      case "ij,nkj->nik": {
        using bt = swapaxes(b, -2, -1);
        return matmul(a, bt);
      }
      // Matmul with A transposed
      case "nji,njk->nik":
      case "ji,jk->ik": {
        using at = swapaxes(a, -2, -1);
        return matmul(at, b);
      }
      // Batched matrix-vector product
      case "nij,nj->ni":
      case "ij,j->i": {
        using bCol = expandDims(b, -1);
        using result = matmul(a, bCol);
        return squeeze(result, [-1]);
      }
      // Attention score: Q[B,L,N,H] @ K[B,S,N,H]^T -> [B,N,L,S]
      case "BLNH,BSNH->BNLS": {
        using at = transpose(a, [0, 2, 1, 3]); // [B,N,L,H]
        using bt = transpose(b, [0, 2, 3, 1]); // [B,N,H,S]
        return matmul(at, bt);
      }
      // Attention weighted values: attn[B,N,L,S] @ V[B,S,N,H] -> [B,L,N,H]
      case "BNLS,BSNH->BLNH": {
        using bt = transpose(b, [0, 2, 1, 3]); // [B,N,S,H]
        using result = matmul(a, bt); // [B,N,L,H]
        return transpose(result, [0, 2, 1, 3]); // [B,L,N,H]
      }
    }
    return null;
  }

  // 3-operand: matmul chains (A×B then ×C or ×C^T)
  if (nOps === 3) {
    const a = fudgeArray(args[1]);
    const b = fudgeArray(args[2]);
    const c = fudgeArray(args[3]);
    switch (subscripts) {
      // AB × C^T
      case "nij,njk,nlk->nil":
      case "ij,jk,lk->il":
      case "nij,jk,nlk->nil":
      case "ij,njk,lk->nil":
      case "ij,njk,nlk->nil": {
        using ab = matmul(a, b);
        using ct = swapaxes(c, -2, -1);
        return matmul(ab, ct);
      }
      // AB × C
      case "nij,njk,nkl->nil":
      case "ij,jk,kl->il": {
        using ab = matmul(a, b);
        return matmul(ab, c);
      }
      // A^T × B × C
      case "ji,jk,kl->il": {
        using at = swapaxes(a, -2, -1);
        using atb = matmul(at, b);
        return matmul(atb, c);
      }
    }
    return null;
  }

  return null;
}

/**
 * Einstein summation with string subscripts.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * using a = np.ones([2, 3]);
 * using b = np.ones([3]);
 * using out = np.einsum("ij,j", a, b); // Shape [2]
 * ```
 */
export function einsum(subscripts: string, ...operands: ArrayLike[]): Array;

/**
 * Einstein summation alternating between arrays and numeric indices.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * using a = np.ones([2, 3]);
 * using b = np.ones([3]);
 * using out = np.einsum(a, [0, 1], b, [1]); // Shape [2]
 * ```
 */
export function einsum(...args: (ArrayLike | number[])[]): Array;

/**
 * Einstein summation.
 *
 * This is a general API for performing tensor reductions, products,
 * transpositions, and traces using Einstein notation for referring to named
 * axes. See the docs for `numpy.einsum()` for more information.
 *
 * <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
 *
 * The full einsum API is implemented, including implicit and explicit output
 * indices, ellipsis broadcasting, Unicode subscripts, and an optimal path
 * ordering algorithm. It lowers to one or more calls to:
 *
 * - `jax.lax.dot()`
 * - `jax.numpy.diagonal()`
 * - `jax.numpy.sum()`
 * - `jax.numpy.transpose()`
 */
export function einsum(...args: any[]): Array {
  if (args.length === 0)
    throw new Error("einsum: must provide at least one argument");

  // Fast path: detect common batch-matmul / transpose patterns and lower
  // directly to matmul, skipping subscript parsing and path computation.
  if (typeof args[0] === "string" && args.length >= 2) {
    const fast = einsumFastPath(args[0] as string, args);
    if (fast !== null) return fast;
  }

  let input: EinsumInput;
  let operands: Array[] = [];
  if (typeof args[0] === "string") {
    // Subscripts mode with remaining arguments as arrays.
    operands = args.slice(1).map(fudgeArray);
    input = parseEinsumExpression(
      args[0],
      operands.map((x) => x.shape),
    );
  } else {
    // Alternating arrays and index arrays.
    const n = args.length >> 1;
    const shapes: number[][] = [];
    const lhsIndices: number[][] = [];
    for (let i = 0; i < n; i++) {
      operands.push(fudgeArray(args[2 * i]));
      shapes.push(operands[i].shape);
      lhsIndices.push(args[2 * i + 1] as number[]);
    }
    let rhsIndex: number[];
    if (args.length % 2 === 1) {
      rhsIndex = args[2 * n] as number[];
    } else {
      // Implicit output indices: all indices that appear only once in the
      // inputs, in the order they appear.
      const indexCount: number[] = [];
      for (const i of lhsIndices.flat()) {
        indexCount[i] = (indexCount[i] ?? 0) + 1;
      }
      rhsIndex = [...indexCount.entries()]
        .filter(([_, count]) => count === 1)
        .map(([i, _]) => i);
    }
    input = { lhsIndices, rhsIndex, shapes };
  }

  const path = computeEinsumPath(input);
  if (DEBUG >= 3)
    console.info(`einsum: computed path: ${path.approximateFlops} flops`);

  // These will be mutated as we do each step of the einsum.
  const indexUsageCounts: number[] = [];
  for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) {
    indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
  }
  const indices = [...input.lhsIndices];

  // Track all intermediate arrays created by einsum for disposal.
  const numUserOperands = operands.length;
  const intermediates: Array[] = [];

  // Process a tensor by reducing all indices not used in any other expressions,
  // and also taking diagonals for duplicate indices.
  const processSingleTensor = (
    ar: Array,
    index: number[],
    doNotReduce: number[] = [],
  ): [Array, number[]] => {
    index = index.slice();
    // 1. Take diagonal to remove all duplicated indices.
    diag: while (true) {
      for (let i = 0; i < index.length; i++) {
        const idx = index[i];
        const j = index.indexOf(idx, i + 1);
        if (j !== -1) {
          ar = diagonal(ar, 0, i, j);
          intermediates.push(ar);
          index.splice(j, 1);
          index.splice(i, 1);
          index.push(idx);
          continue diag;
        }
      }
      break;
    }
    // 2. Reduce all indices that are not in `doNotReduce` and have zero usage.
    for (let i = index.length - 1; i >= 0; i--) {
      const idx = index[i];
      if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
        ar = sum(ar, i);
        intermediates.push(ar);
        index.splice(i, 1);
      }
    }
    return [ar, index];
  };

  for (const [i, j] of path.path) {
    let indexReduced: number[] = [];
    const indexGroup: number[] = [];
    for (const idx of [...indices[i], ...indices[j]]) {
      if (!indexGroup.includes(idx)) indexGroup.push(idx);
      // If the index is not in the output and isn't in any other inputs,
      // we can consider it reduced here.
      if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
    }
    const [a, aidx] = processSingleTensor(operands[i], indices[i], indices[j]);
    const [b, bidx] = processSingleTensor(operands[j], indices[j], indices[i]);
    // At this point, aidx and bidx are both unique index sets that need to be
    // in the output. We can use dot to combine them along reduction dims.
    indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
    const indexBatch = aidx.filter(
      (idx) => bidx.includes(idx) && !indexReduced.includes(idx),
    );
    const result = lax.dot(a, b, {
      lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
      rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
      lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
      rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx)),
    });
    intermediates.push(result);
    operands.push(result);
    indices.push([
      ...indexBatch,
      ...aidx.filter((idx) => !bidx.includes(idx)),
      ...bidx.filter((idx) => !aidx.includes(idx)),
    ]);
    for (const idx of indices[indices.length - 1]) ++indexUsageCounts[idx];
  }

  // Special case: Einsum with just one operand produces an empty path, but we
  // may still need to do any reductions or traces. Process one more time.
  for (const idx of indices[operands.length - 1]) --indexUsageCounts[idx];
  const [ar, index] = processSingleTensor(
    operands[operands.length - 1],
    indices[operands.length - 1],
  );

  // At this point, `index` _must_ be some permutation of `rhsIndex`.
  const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
  const finalResult = ar.transpose(finalPerm);

  // Dispose all intermediates except the final result and user operands.
  const keep = new Set<Array>(operands.slice(0, numUserOperands));
  keep.add(finalResult);
  for (const a of intermediates) {
    if (!keep.has(a)) a.dispose();
  }

  return finalResult;
}

/**
 * Compute the inner product of two arrays.
 *
 * Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
 * contraction on the last axis.
 *
 * Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
 */
export function inner(x: ArrayLike, y: ArrayLike): Array {
  return lax.dot(fudgeArray(x), fudgeArray(y), {
    lhsContractingDims: [-1],
    rhsContractingDims: [-1],
  });
}

/**
 * Compute the outer product of two arrays.
 *
 * If the input arrays are not 1D, they will be flattened. Returned array will
 * be of shape `[x.size, y.size]`.
 */
export function outer(x: ArrayLike, y: ArrayLike): Array {
  using rx = ravel(x);
  using ry = ravel(y);
  using rxR = rx.reshape([rx.shape[0], 1]);
  return multiply(rxR, ry);
}

/**
 * @function Compute the cross product of two arrays.
 *
 * Supports 2D (scalar result) and 3D cross products, with optional axis
 * arguments. If `axis` is given, it overrides `axisa`, `axisb`, and `axisc`.
 */
export const cross = jit(
  function cross(
    a: ArrayLike,
    b: ArrayLike,
    {
      axisa = -1,
      axisb = -1,
      axisc = -1,
      axis,
    }: { axisa?: number; axisb?: number; axisc?: number; axis?: number } = {},
  ): Array {
    if (axis !== undefined) {
      axisa = axis;
      axisb = axis;
      axisc = axis;
    }
    axisa = checkAxis(axisa, ndim(a));
    axisb = checkAxis(axisb, ndim(b));
    using d = new DisposableStack();
    const aM = d.use(moveaxis(a, axisa, -1));
    const bM = d.use(moveaxis(b, axisb, -1));
    const da = aM.shape.at(-1)!;
    const db = bM.shape.at(-1)!;
    if ((da !== 2 && da !== 3) || (db !== 2 && db !== 3)) {
      throw new Error(
        `cross: incompatible dimensions for cross product (got ${da} and ${db})`,
      );
    }

    if (da === 2 && db === 2) {
      const [a0, a1] = split(aM, 2, -1);
      d.use(a0);
      d.use(a1);
      const [b0, b1] = split(bM, 2, -1);
      d.use(b0);
      d.use(b1);
      const p1 = d.use(a0.mul(b1));
      const p2 = d.use(a1.mul(b0));
      const diff = d.use(p1.sub(p2));
      return squeeze(diff, -1);
    }

    // Pad 2D inputs to 3D with a zero third component
    let aPad = aM;
    let bPad = bM;
    if (da === 2) {
      const zeroShape = [...aM.shape.slice(0, -1), 1];
      const z = d.use(zeros(zeroShape));
      aPad = d.use(concatenate([aM, z], -1));
    }
    if (db === 2) {
      const zeroShape = [...bM.shape.slice(0, -1), 1];
      const z = d.use(zeros(zeroShape));
      bPad = d.use(concatenate([bM, z], -1));
    }

    const [a0, a1, a2] = split(aPad, 3, -1);
    d.use(a0);
    d.use(a1);
    d.use(a2);
    const [b0, b1, b2] = split(bPad, 3, -1);
    d.use(b0);
    d.use(b1);
    d.use(b2);
    const p1 = d.use(a1.mul(b2));
    const p2 = d.use(a2.mul(b1));
    const c0 = d.use(p1.sub(p2));
    const p3 = d.use(a2.mul(b0));
    const p4 = d.use(a0.mul(b2));
    const c1 = d.use(p3.sub(p4));
    const p5 = d.use(a0.mul(b1));
    const p6 = d.use(a1.mul(b0));
    const c2 = d.use(p5.sub(p6));
    const cat = d.use(concatenate([c0, c1, c2], -1));
    return moveaxis(cat, -1, axisc);
  },
  { staticArgnums: [2] },
);

/** Vector dot product of two arrays along a given axis. */
export function vecdot(
  x: ArrayLike,
  y: ArrayLike,
  { axis }: { axis?: number } = {},
): Array {
  const xaxis = checkAxis(axis ?? -1, ndim(x));
  const yaxis = checkAxis(axis ?? -1, ndim(y));
  if (shape(x)[xaxis] !== shape(y)[yaxis]) {
    throw new Error(
      "vecdot: shapes " +
        `${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} ` +
        `not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`,
    );
  }
  using xm = moveaxis(x, xaxis, -1);
  using ym = moveaxis(y, yaxis, -1);
  return core.dot(xm, ym) as Array;
}

/**
 * Return the dot product of two vectors.
 *
 * Like vecdot() but flattens the arguments first into vectors.
 */
export function vdot(x: ArrayLike, y: ArrayLike): Array {
  using rx = ravel(x);
  using ry = ravel(y);
  return core.dot(rx, ry) as Array;
}

function _convImpl(name: string, x: Array, y: Array, mode: string): Array {
  if (x.ndim !== 1 || y.ndim !== 1) {
    throw new Error(
      `${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`,
    );
  }
  let flipOutput = false; // for correlate: output[k] = sum(x[i + k] * y[i]) = sum(y[i + (-k)] * x[i])
  if (x.shape[0] < y.shape[0]) {
    [x, y] = [y, x];
    if (name === "correlate") flipOutput = true;
  }
  using d = new DisposableStack();
  if (name === "convolve") {
    y = d.use(flip(y));
  }

  let padding: lax.PaddingType;
  if (mode === "valid") padding = "VALID";
  else if (mode === "same") padding = "SAME_LOWER";
  else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
  else {
    throw new Error(
      `${name}: invalid mode ${mode}, expected "full", "same", or "valid"`,
    );
  }

  const xs = d.use(x.slice(null, null));
  const ys = d.use(y.slice(null, null));
  const convResult = d.use(lax.conv(xs, ys, [1], padding));
  const z = convResult.slice(0, 0);
  if (flipOutput) {
    d.use(z);
    return flip(z);
  }
  return z;
}

/** Convolution of two one-dimensional arrays. */
export function convolve(
  x: Array,
  y: Array,
  mode: "full" | "same" | "valid" = "full",
): Array {
  return _convImpl("convolve", x, y, mode);
}

/** Correlation of two one dimensional arrays. */
export function correlate(
  x: Array,
  y: Array,
  mode: "full" | "same" | "valid" = "valid",
): Array {
  return _convImpl("correlate", x, y, mode);
}

/**
 * Return a tuple of coordinate matrices from coordinate vectors.
 *
 * Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
 * fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
 */
export function meshgrid(
  xs: Array[],
  { indexing }: { indexing?: "xy" | "ij" } = {},
): Array[] {
  indexing ??= "xy"; // Default for numpy is "xy"

  for (const x of xs) {
    if (x.ndim !== 1) {
      throw new Error(
        `meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`,
      );
    }
  }
  if (xs.length <= 1) return xs;
  if (indexing === "xy") {
    // For "xy" indexing, we just have to reverse the first two values.
    const [a, b, ...rest] = xs;
    const [rb, ra, ...rrest] = meshgrid([b, a, ...rest], { indexing: "ij" });
    return [ra, rb, ...rrest];
  }

  // Now do the actual meshgrid construction, using movement operators.
  const shape = xs.map((x) => x.shape[0]);
  return xs.map(
    (x, i) =>
      core.broadcast(x, shape, [
        ...range(i),
        ...range(i + 1, xs.length),
      ]) as Array,
  );
}

/**
 * Clip (limit) the values in an array.
 *
 * Given an interval, values outside the interval are clipped to the interval
 * edges. For example, if an interval of [0, 1] is specified, values smaller
 * than 0 become 0, and values larger than 1 become 1.
 *
 * If either bound is undefined, it is ignored.
 */
export function clip(a: ArrayLike, min?: ArrayLike, max?: ArrayLike): Array {
  a = fudgeArray(a);
  if (max !== undefined && min !== undefined) {
    using clipped = minimum(a, max);
    return maximum(clipped, min);
  }
  if (max !== undefined) return minimum(a, max);
  if (min !== undefined) return maximum(a, min);
  return a; // No clipping, just return the original array.
}

/**
 * Calculate the absolute value element-wise.
 *
 * This is the same function as `jax.numpy.abs()`.
 */
export function absolute(x: ArrayLike): Array {
  x = fudgeArray(x);
  using cond = less(x, 0) as Array;
  using negX = (x as Array).mul(-1) as Array;
  return where(cond, negX, x);
}

export { absolute as abs };

/** Return an element-wise indication of sign of the input. */
export function sign(x: ArrayLike): Array {
  x = fudgeArray(x);
  using neq = notEqual(x, 0);
  using lt = less(x, 0);
  using inner = where(lt, -1, 1);
  using result = where(neq, inner, 0);
  // Propagate NaN: NaN != NaN is true per IEEE 754, so notEqual(x, x)
  // detects NaN values. Return x (NaN) for those positions.
  using isnan = notEqual(x, x);
  return where(isnan, x, result);
}

/** @function Return the value with the magnitude of x and the sign of y, element-wise. */
export const copysign = jit(function copysign(x: Array, y: Array): Array {
  using absX = absolute(x);
  using signY = sign(y);
  return absX.mul(signY);
});

/** @function Return element-wise positive values of the input (no-op). */
export const positive = fudgeArray;

/**
 * Return the Hamming window of size M, a taper with a weighted cosine bell.
 *
 * `w(n) = 0.54 - 0.46 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
 */
export function hamming(M: number): Array {
  using ls = linspace(0, 2 * Math.PI, M);
  using c = cos(ls);
  using scaled = c.mul(-0.46);
  return scaled.add(0.54);
}

/**
 * Return the Hann window of size M, a taper with a weighted cosine bell.
 *
 * `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
 */
export function hann(M: number): Array {
  using ls = linspace(0, 2 * Math.PI, M);
  using c = cos(ls);
  using scaled = c.mul(-0.5);
  return scaled.add(0.5);
}

/**
 * @function
 * Compute the Heaviside step function. It is defined piecewise:
 * - `heaviside(x1, x2) = 0` for `x1 < 0`,
 * - `heaviside(x1, x2) = x2` for `x1 == 0`,
 * - `heaviside(x1, x2) = 1` for `x1 > 0`.
 */
export const heaviside = jit(function heaviside(x1: Array, x2: Array) {
  using lt = less(x1, 0);
  using eq = equal(x1, 0);
  using inner = where(eq, x2, 1);
  return where(lt, 0, inner);
});

/** Calculate element-wise square of the input array. */
export function square(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.mul(x);
}

/** Element-wise tangent function (takes radians). */
export function tan(x: ArrayLike): Array {
  x = fudgeArray(x);
  using sinX = sin(x);
  using cosX = cos(x);
  return sinX.div(cosX);
}

/**
 * @function
 * Return the normalized sinc function.
 *
 * The sinc function is defined as `sin(πx) / (πx)` for `x != 0`, and `1` for `x = 0`.
 * This is the normalized sinc function commonly used in signal processing.
 *
 * **Note:** JVP is not supported at x=0 due to discontinuous derivative. This
 * requires a custom JVP rule to handle properly (see JAX implementation).
 */
export const sinc = jit(function sinc(x: Array): Array {
  using pix = x.mul(Math.PI);
  using eq = equal(x, 0);
  using sinPix = sin(pix);
  using ratio = sinPix.div(pix);
  return where(eq, 1, ratio);
});

/** Element-wise inverse cosine function (inverse of cos). */
export function acos(x: ArrayLike): Array {
  using asinX = asin(x);
  return subtract(pi / 2, asinX);
}

/**
 * @function
 * Return element-wise hypotenuse for the given legs of a right triangle.
 *
 * In the original NumPy/JAX implementation, this function is more numerically
 * stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
 * stability improvements.
 */
export const hypot = jit(function hypot(x1: Array, x2: Array) {
  using sq1 = square(x1);
  using sq2 = square(x2);
  using sum = sq1.add(sq2);
  return sqrt(sum);
});

/**
 * @function
 * Element-wise arc tangent of y/x with correct quadrant.
 *
 * Returns the angle in radians between the positive x-axis and the point (x, y).
 * The result is in the range [-π, π].
 *
 * Uses numerically stable formulas:
 * - When x >= 0: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
 * - When x < 0:  atan2(y, x) = 2 * atan((sqrt(x^2 + y^2) - x) / y)
 *
 * The output is ill-defined when both x and y are zero.
 */
export const atan2 = jit(function atan2(y: Array, x: Array) {
  using r = sqrt(square(x).add(square(y)));
  using xNeg = less(x, 0);

  // Select numerator and denominator based on sign of x
  // When x >= 0: numer = y,     denom = r + x
  // When x < 0:  numer = r - x, denom = y
  using numer = where(xNeg, r.sub(x), y) as Array;
  using denom = where(xNeg, y, r.add(x)) as Array;

  using atanResult = atan(numer.div(denom)).mul(2) as Array;
  // atan2(0, 0) = 0 by NumPy convention (the formula above produces NaN)
  using xZero = equal(x, 0);
  using yZero = equal(y, 0);
  using bothZero = xZero.mul(yZero);
  return where(bothZero, 0, atanResult);
});

export { asin as arcsin, acos as arccos, atan as arctan, atan2 as arctan2 };

/** Element-wise subtraction, with broadcasting. */
export function subtract(x: ArrayLike, y: ArrayLike): Array {
  // Use neg+add instead of fudgeArray(x).sub(y) so scalar x is inlined
  // as a Lit in the jaxpr rather than creating a receiver-position const.
  using negY = negative(y);
  return add(x, negY);
}

/** Calculates the floating-point division of x by y element-wise. */
export function trueDivide(x: ArrayLike, y: ArrayLike): Array {
  const xFloatDtype =
    x instanceof Array && isFloatDtype(x.dtype) ? x.dtype : null;
  const yFloatDtype =
    y instanceof Array && isFloatDtype(y.dtype) ? y.dtype : null;

  if (xFloatDtype && yFloatDtype) {
    using recipY = reciprocal(y);
    return multiply(x, recipY);
  }
  if (xFloatDtype) {
    using yf = astype(y, xFloatDtype);
    using recipY = reciprocal(yf);
    return multiply(x, recipY);
  }
  if (yFloatDtype) {
    using xf = astype(x, yFloatDtype);
    using recipY = reciprocal(y);
    return multiply(xf, recipY);
  }
  using xf = astype(x, DType.Float32);
  using yf = astype(y, DType.Float32);
  using recipY = reciprocal(yf);
  return multiply(xf, recipY);
}

export { trueDivide as divide };

/**
 * Return the largest integer smaller or equal to the division of the inputs.
 *
 * The result is always rounded towards negative infinity.
 *
 * For floating-point inputs, this is equivalent to `floor(x / y)`.
 * For integer inputs, we use `(x - remainder(x, y)) / y` to handle
 * negative values correctly (note: may overflow near int32 boundaries).
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Element-wise floor division of x by y.
 */
export function floorDivide(x: ArrayLike, y: ArrayLike): Array {
  const xArr = fudgeArray(x);
  const yArr = fudgeArray(y);
  try {
    if (isFloatDtype(xArr.dtype) || isFloatDtype(yArr.dtype)) {
      // For floats, floor(x / y) works correctly
      using div = trueDivide(xArr, yArr);
      return floor(div);
    }
    // For integers, use (x - remainder(x, y)) / y to round toward -infinity
    // This avoids the truncation behavior of idiv which rounds toward zero
    using rem = remainder(xArr, yArr);
    using diff = subtract(xArr, rem);
    return diff.div(yArr) as Array;
  } finally {
    if (xArr !== (x as any)) xArr.dispose();
    if (yArr !== (y as any)) yArr.dispose();
  }
}

/**
 * @function
 * Calculate element-wise floating-point modulo operation.
 */
export const fmod = jit(function fmod(x: Array, y: Array): Array {
  using q = core.idiv(x, y) as Array;
  using yq = y.mul(q);
  return x.sub(yq);
});

/**
 * @function
 * Calculate element-wise remainder of the division (matches sign of y).
 */
export const remainder = jit(function remainder(x: Array, y: Array): Array {
  // The `Mod` primitive matches the sign of x (JS truncation rounding).
  // This function must match the sign of y instead (Python/NumPy floor modulo).
  // Single-mod + conditional correction: add y when signs of r and y disagree.
  using r = core.mod(x, y) as Array;
  // Need correction when r != 0 and sign(r) != sign(y).
  // sign(r) != sign(y) iff r*y < 0.
  using ry = r.mul(y);
  using needsCorrection = ry.less(0);
  using corrected = r.add(y);
  return where(needsCorrection, corrected, r);
});

/**
 * Return element-wise quotient and remainder simultaneously.
 *
 * Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Tuple of [quotient, remainder].
 */
export function divmod(x: ArrayLike, y: ArrayLike): [Array, Array] {
  const xArr = fudgeArray(x);
  const yArr = fudgeArray(y);
  // floorDivide and remainder both use their inputs non-destructively
  const result: [Array, Array] = [
    floorDivide(xArr, yArr),
    remainder(xArr, yArr),
  ];
  if (xArr !== (x as any)) xArr.dispose();
  if (yArr !== (y as any)) yArr.dispose();
  return result;
}

/** Round input to the nearest integer towards zero. */
export function trunc(x: ArrayLike): Array {
  return core.idiv(x, 1) as Array; // Integer division truncates the decimal part.
}

/**
 * @function
 * Round to the given number of decimals.
 *
 * Uses banker's rounding (round half to even) to match NumPy/JAX behavior.
 */
export const round = jit(
  function round(a: Array, decimals: number = 0): Array {
    if (decimals === 0) {
      return rint(a);
    }
    const factor = 10 ** decimals;
    using scaled = a.mul(factor);
    using rounded = rint(scaled);
    return rounded.mul(1 / factor);
  },
  { staticArgnums: [1] },
);

export { round as around };

/**
 * @function
 * Round to the nearest integer, with ties going to the nearest even integer.
 */
export const rint = jit(function rint(x: Array): Array {
  // Banker's rounding: round half to even.
  using added = x.add(0.5);
  using rounded = floor(added);
  // When the fractional part is exactly 0.5, round to even.
  using floored = floor(x);
  using diff = x.sub(floored);
  using half = diff.equal(0.5);
  using rem = remainder(rounded, 2);
  using odd = rem.notEqual(0);
  using cond = half.mul(odd);
  using alt = rounded.sub(1);
  return where(cond, alt, rounded);
});

/**
 * Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
 *
 * This is the inverse of `frexp()`.
 */
export function ldexp(x1: ArrayLike, x2: ArrayLike): Array {
  using e = exp2(x2);
  return multiply(x1, e);
}

/**
 * Decompose floating-point values into mantissa and two's exponent.
 *
 * The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
 * `x != 0`, and the exponent is an integer such that
 * `x = mantissa * 2**exponent`.
 */
export function frexp(x: ArrayLike): [Array, Array] {
  x = fudgeArray(x);
  using absx = absolute(x);
  using log2abs = log2(absx);
  using floorLog = floor(log2abs);
  using shiftedExp = floorLog.add(1);
  using exponentI32 = shiftedExp.astype(DType.Int32);
  using eq = equal(x, 0);
  const exponent = where(eq, 0, exponentI32);
  using exponentAsX = exponent.astype(x.dtype);
  using scale = exp2(exponentAsX);
  const mantissa = x.div(scale);
  return [mantissa, exponent];
}

/** Calculate `2**p` for all p in the input array. */
export function exp2(p: ArrayLike): Array {
  using prod = multiply(p, Math.LN2);
  return exp(prod);
}

/** Return the base-2 logarithm of x, element-wise. */
export function log2(x: ArrayLike): Array {
  using logX = log(x);
  return logX.mul(Math.LOG2E);
}

/** Return the base-10 logarithm of x, element-wise. */
export function log10(x: ArrayLike): Array {
  using logX = log(x);
  return logX.mul(Math.LOG10E);
}

/** Calculate `exp(x) - 1` element-wise with improved precision for small x. */
export function expm1(x: ArrayLike): Array {
  // Kahan's formula: u = exp(x); if u == 1 then x, else (u-1)*x/log(u).
  // This avoids catastrophic cancellation when x is near zero.
  using u = exp(x);
  using um1 = u.sub(1);
  using logU = log(u);
  // When u == 1 (x ≈ 0), log(u) is also 0. Guard the denominator to avoid
  // 0/0 = NaN, which would leak through autodiff (where differentiates both branches).
  using uEq1 = u.equal(1);
  using safeLogU = where(uEq1, 1, logU) as Array;
  using um1x = um1.mul(fudgeArray(x));
  using ratio = um1x.div(safeLogU);
  return where(uEq1, x, ratio);
}

/** Calculate the natural logarithm of `1 + x` element-wise with improved precision for small x. */
export function log1p(x: ArrayLike): Array {
  // Kahan's formula: u = 1 + x; if u == 1 then x, else log(u)*x/(u-1).
  // This avoids catastrophic cancellation when x is near zero.
  using u = add(1, x);
  using logU = log(u);
  using um1 = u.sub(1);
  // When u == 1 (x ≈ 0), u-1 is 0. Guard the denominator to avoid
  // 0/0 = NaN, which would leak through autodiff (where differentiates both branches).
  using uEq1 = u.equal(1);
  using safeUm1 = where(uEq1, 1, um1) as Array;
  using logUx = logU.mul(fudgeArray(x));
  using ratio = logUx.div(safeUm1);
  return where(uEq1, x, ratio);
}

/** Convert angles from degrees to radians. */
export function deg2rad(x: ArrayLike): Array {
  return multiply(x, pi / 180);
}

/** @function Alias of `jax.numpy.deg2rad()`. */
export const radians = deg2rad;

/** Convert angles from radians to degrees. */
export function rad2deg(x: ArrayLike): Array {
  return multiply(x, 180 / pi);
}

/** @function Alias of `jax.numpy.rad2deg()`. */
export const degrees = rad2deg;

/**
 * @function
 * Computes first array raised to power of second array, element-wise.
 */
export const power = jit(function power(x1: Array, x2: Array) {
  // TODO: This is a little bit inefficient since we need to handle negative
  // numbers to integer exponents, should eventually move it into the backend.
  using x2i = trunc(x2);
  // Should be NaN if x1 < 0 and x2 is non-integer.
  using shouldBeNaN = multiply(x2.notEqual(x2i), x1.less(0));
  // If x2 is odd integer, result sign matches x1, else it's positive.
  using parityRaw = core.mod(x2i, 2) as Array;
  using parityOdd = parityRaw.notEqual(0);
  using resultSign = where(parityOdd, where(x1.less(0), -1, 1), 1) as Array;
  using absX1 = absolute(x1);
  using logAbsX1 = log(absX1);
  using scaled = logAbsX1.mul(x2);
  using magnitude = exp(scaled);
  using signedMagnitude = magnitude.mul(resultSign);
  return where(shouldBeNaN, nan, signedMagnitude);
});

export { power as pow };

/** @function Calculate the element-wise cube root of the input array. */
export const cbrt = jit(function cbrt(x: Array) {
  // This isn't just power(x, 1/3) since we need to handle negative numbers.
  using sgn = where(less(x, 0), -1, 1) as Array;
  using signedX = x.mul(sgn);
  using logSignedX = log(signedX);
  using scaled = logSignedX.mul(1 / 3);
  using magnitude = exp(scaled);
  return sgn.mul(magnitude);
});

/**
 * @function
 * Calculate element-wise hyperbolic sine of input.
 *
 * `sinh(x) = (exp(x) - exp(-x)) / 2`
 */
export const sinh = jit(function sinh(x: Array) {
  using ex = exp(x);
  using emx = reciprocal(ex);
  using diff = ex.sub(emx);
  return diff.mul(0.5);
});

/**
 * @function
 * Calculate element-wise hyperbolic cosine of input.
 *
 * `cosh(x) = (exp(x) + exp(-x)) / 2`
 */
export const cosh = jit(function cosh(x: Array) {
  using ex = exp(x);
  using emx = reciprocal(ex);
  using sum = ex.add(emx);
  return sum.mul(0.5);
});

/**
 * @function
 * Calculate element-wise hyperbolic tangent of input.
 *
 * `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 */
export const tanh = jit(function tanh(x: Array) {
  // Avoid overflow for large x by taking advantage of alternate representations:
  // tanh(x) = -tanh(-x) = (1 - e^{-2x}) / (1 + e^{-2x})
  using ltZero = less(x, 0);
  using negsgn = where(ltZero, 1, -1);
  using signed = x.mul(negsgn);
  using scaled = signed.mul(2);
  using en2x = exp(scaled);
  using numer = en2x.sub(1);
  using denom = en2x.add(1);
  using ratio = numer.div(denom);
  return ratio.mul(negsgn);
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic sine of input.
 *
 * `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
 */
export const arcsinh = jit(function arcsinh(x: Array) {
  using sq = square(x);
  using shifted = sq.add(1);
  using root = sqrt(shifted);
  using arg = x.add(root);
  return log(arg);
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic cosine of input.
 *
 * `arccosh(x) = ln(x + sqrt(x^2 - 1))`
 */
export const arccosh = jit(function arccosh(x: Array) {
  using sq = square(x);
  using shifted = sq.sub(1);
  using root = sqrt(shifted);
  using arg = x.add(root);
  return log(arg);
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic tangent of input.
 *
 * `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
 */
export const arctanh = jit(function arctanh(x: Array) {
  using onePlusX = add(1, x);
  using oneMinusX = subtract(1, x);
  using ratio = onePlusX.div(oneMinusX);
  using logRatio = log(ratio);
  return logRatio.mul(0.5);
});

export { arcsinh as asinh, arccosh as acosh, arctanh as atanh };

/**
 * Compute the variance of an array.
 *
 * The variance is computed for the flattened array by default, otherwise over
 * the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
export function var_(
  x: ArrayLike,
  axis: core.Axis = null,
  opts?: { mean?: ArrayLike; correction?: number } & core.ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  if (n === 0) {
    throw new Error("var: cannot compute variance over zero-length axis");
  }
  using d = new DisposableStack();
  const mu =
    opts?.mean !== undefined
      ? opts.mean
      : (() => {
          const m = mean(x, axis, { keepdims: true });
          d.use(m);
          return m;
        })();
  const centered = d.use(x.sub(mu));
  const sq = d.use(square(centered));
  const summed = d.use(sq.sum(axis, { keepdims: opts?.keepdims }));
  return summed.mul(1 / (n - (opts?.correction ?? 0)));
}

/**
 * Compute the standard deviation of an array.
 *
 * The standard deviation is computed for the flattened array by default,
 * otherwise over the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
export function std(
  x: ArrayLike,
  axis: core.Axis = null,
  opts?: { mean?: ArrayLike; correction?: number } & core.ReduceOpts,
): Array {
  using v = var_(x, axis, opts);
  return sqrt(v);
}

/** Estimate the sample covariance of a set of variables. */
export function cov(
  x: ArrayLike,
  y: ArrayLike | null = null,
  { rowvar = true }: { rowvar?: boolean } = {},
): Array {
  // x should shape (M, N) or (N,), representing N observations of M variables.
  using d = new DisposableStack();
  let a = fudgeArray(x);
  if (a.ndim === 1) {
    a = d.use(a.reshape([1, a.shape[0]]));
  }
  // optional set of additional observations, concatenated to m
  if (y !== null) {
    let b = fudgeArray(y);
    if (b.ndim === 1) {
      b = d.use(b.reshape([1, b.shape[0]]));
    }
    a = d.use(vstack([a, b]));
  }
  if (!rowvar) {
    a = d.use(a.transpose());
  }
  const [_M, N] = a.shape;
  using mean = a.mean(1, { keepdims: true });
  const centered = d.use(a.sub(mean)); // Center variables
  using xt = centered.transpose();
  using dotResult = dot(centered, xt);
  return dotResult.div(N - 1); // [M, M]
}

/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
export function corrcoef(x: ArrayLike, y?: ArrayLike): Array {
  using c = cov(x, y);
  using variances = diag(c);
  using norm = sqrt(outer(variances, variances));
  return c.div(norm);
}

/** Test element-wise for positive or negative infinity, return bool array. */
export function isinf(x: ArrayLike): Array {
  x = fudgeArray(x);
  if (!isFloatDtype(x.dtype)) return fullLike(x, false);
  using posInf = x.equal(Infinity);
  using negInf = x.equal(-Infinity);
  return posInf.add(negInf);
}

/** Test element-wise for NaN (Not a Number). */
export function isnan(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.notEqual(x) : fullLike(x, false);
}

/** Test element-wise for negative infinity, return bool array. */
export function isneginf(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike(x, false);
}

/** Test element-wise for positive infinity, return bool array. */
export function isposinf(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike(x, false);
}

/**
 * Replace NaN and infinite entries in an array.
 *
 * By default, NaNs are replaced with `0.0`, and infinities are are substituted
 * with the corresponding maximum or minimum finite values.
 */
export function nanToNum(
  x: ArrayLike,
  {
    nan = 0.0,
    posinf = null,
    neginf = null,
  }: {
    nan?: ArrayLike;
    posinf?: ArrayLike | null;
    neginf?: ArrayLike | null;
  } = {},
): Array {
  x = fudgeArray(x);
  using nanMask = isnan(x);
  using afterNan = where(nanMask, nan, x);
  posinf ??= isFloatDtype(afterNan.dtype)
    ? finfo(afterNan.dtype).max
    : iinfo(afterNan.dtype).max;
  neginf ??= isFloatDtype(afterNan.dtype)
    ? finfo(afterNan.dtype).min
    : iinfo(afterNan.dtype).min;
  using posInfMask = isposinf(afterNan);
  using afterPosInf = where(posInfMask, posinf, afterNan);
  using negInfMask = isneginf(afterPosInf);
  return where(negInfMask, neginf, afterPosInf);
}

/**
 * @function
 * Test element-wise for finite values (not infinity or NaN).
 */
export const isfinite = jit(function isfinite(x: Array): Array {
  if (!isFloatDtype(x.dtype)) return fullLike(x, true);
  // A value is finite iff it equals itself (not NaN) and its absolute value
  // is not infinity. Combine into a single boolean expression.
  using absX = absolute(x);
  using notInf = absX.notEqual(Infinity);
  using notNan = x.equal(x);
  return notInf.mul(notNan);
});
