// Mirrors the `jax.lax` module in JAX.
//
// Unlike in JAX, this does not actually underpin `jax.numpy` as a more "core"
// set of operations, as they both build open the same foundations.

const JsArray = globalThis.Array;

import { DType } from "../alu";
import { type BackendCapabilities, getBackend } from "../backend";
import {
  Array,
  array as arrayFn,
  ArrayLike,
  fudgeArray,
  fullInternal,
  zerosLike,
} from "../frontend/array";
import * as core from "../frontend/core";
import { bind1, Primitive, ShapedArray } from "../frontend/core";
import * as jaxpr from "../frontend/jaxpr";
import { moveaxis, vmap } from "../frontend/vmap";
import { type Dim, Pair } from "../shape";
import * as tree from "../tree";
import { checkAxis, deepEqual, prod, range, rep, zipn } from "../utils";
import { blockIndex, blockMap } from "./lax-block-map";
import { scan } from "./lax-scan";

export * as linalg from "./lax-linalg";
export { scan };
export type { ScanOptions } from "./lax-scan";
export { associativeScan } from "./lax-associative-scan";
export type { AssociativeScanOptions } from "./lax-associative-scan";
export { blockIndex, blockMap };
export type { BlockMapOptions } from "./lax-block-map";

/** Elementwise bitcast an array into a new dtype. */
export function bitcastConvertType(x: ArrayLike, newDtype: DType): Array {
  return fudgeArray(x).view(newDtype);
}

/**
 * Dimension numbers for general `dot()` primitive.
 *
 * Contracting dimensions act as a tensor contraction (reduction) along the
 * given axis. They must be the same size in both operands. Batch dimensions
 * are treated as vectorized, leading batch dimensions.
 *
 * The return value has a shape where the first dimensions are shared batch
 * dimensions, followed by `lhs` non-contracting dimensions, followed by
 * `rhs` non-contracting dimensions.
 */
export type DotDimensionNumbers = {
  lhsContractingDims?: number[];
  rhsContractingDims?: number[];
  lhsBatchDims?: number[];
  rhsBatchDims?: number[];
};

/**
 * General dot product/contraction operator.
 *
 * Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
 * `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
 */
export function dot(
  lhs: Array,
  rhs: Array,
  {
    lhsContractingDims: lc = [],
    rhsContractingDims: rc = [],
    lhsBatchDims: lb = [],
    rhsBatchDims: rb = [],
  }: DotDimensionNumbers = {},
): Array {
  // First do input validation, helps with debugging.
  if (lc.length !== rc.length) {
    throw new Error(
      `dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`,
    );
  } else if (lb.length !== rb.length) {
    throw new Error(
      `dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`,
    );
  }
  lc = lc.map((a) => checkAxis(a, lhs.ndim));
  rc = rc.map((a) => checkAxis(a, rhs.ndim));
  lb = lb.map((a) => checkAxis(a, lhs.ndim));
  rb = rb.map((a) => checkAxis(a, rhs.ndim));
  if (lc.some((a) => lb.includes(a))) {
    throw new Error(
      `dot: lhs contracting dims ${JSON.stringify(lc)} ` +
        `overlap with batch dims ${JSON.stringify(lb)}`,
    );
  } else if (rc.some((a) => rb.includes(a))) {
    throw new Error(
      `dot: rhs contracting dims ${JSON.stringify(rc)} ` +
        `overlap with batch dims ${JSON.stringify(rb)}`,
    );
  }

  // Compute "free" dimensions: output shape is [...{lb/rb}, ...lf, ...rf].
  const lf = range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
  const rf = range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
  using lhs2 = lhs.transpose([...lb, ...lf, ...lc]);
  using rhs2 = rhs.transpose([...rb, ...rf, ...rc]);

  if (lc.length === 0) {
    // There is no contraction to perform, just do a product (not `dot`).
    using lhsR = lhs2.reshape([
      ...lb.map((a) => lhs.shape[a]),
      ...lf.map((a) => lhs.shape[a]),
      ...rep(rf.length, 1),
    ]);
    using rhsR = rhs2.reshape([
      ...rb.map((a) => rhs.shape[a]),
      ...rep(lf.length, 1),
      ...rf.map((a) => rhs.shape[a]),
    ]);
    return core.mul(lhsR, rhsR) as Array;
  }

  // Otherwise, we need to do a `dot` contraction.
  const dotShapeX = lc.map((a) => lhs.shape[a]);
  const dotShapeY = rc.map((a) => rhs.shape[a]);
  if (!deepEqual(dotShapeX, dotShapeY)) {
    throw new Error(
      `dot: shapes not aligned along contracting dims:` +
        ` ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`,
    );
  }
  using lhsR = lhs2.reshape([
    ...lb.map((a) => lhs.shape[a]),
    ...lf.map((a) => lhs.shape[a]),
    ...rep(rf.length, 1),
    prod(dotShapeX),
  ]);
  using rhsR = rhs2.reshape([
    ...rb.map((a) => rhs.shape[a]),
    ...rep(lf.length, 1),
    ...rf.map((a) => rhs.shape[a]),
    prod(dotShapeY),
  ]);
  return core.dot(lhsR, rhsR) as Array;
}

export type PaddingType = "VALID" | "SAME" | "SAME_LOWER" | Pair[];

function padtypeToPads(
  inShape: number[],
  filterShape: number[],
  strides: number[],
  dilation: number[],
  padding: string,
): [number, number][] {
  const padType = padding.toUpperCase();
  switch (padType) {
    case "VALID":
      return rep<[number, number]>(inShape.length, [0, 0]);
    case "SAME":
    case "SAME_LOWER": {
      const outShape = inShape.map((size, i) => Math.ceil(size / strides[i]));
      const padSizes = zipn(
        outShape,
        strides,
        filterShape,
        dilation,
        inShape,
      ).map(([o, s, k, d, i]) =>
        Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i),
      );
      if (padType === "SAME") {
        return padSizes.map((size) => [size >> 1, size - (size >> 1)]);
      } else {
        return padSizes.map((size) => [size - (size >> 1), size >> 1]);
      }
    }
    default:
      throw new Error(`Unknown padding type: ${padType}`);
  }
}

/**
 * General n-dimensional convolution operator, with optional dilation.
 *
 * The semantics of this operation mimic the `jax.lax.conv_general_dilated`
 * function in JAX, which wraps XLA's general convolution operator.
 *
 * @param lhs - Input tensor; shape `[N, C_in, ...xs]`
 * @param rhs - Convolution kernel; shape `[C_out, C_in / G, ...ks]`
 * @param windowStrides - Strides for each spatial dimension
 * @param padding - Padding for each spatial dimension, or a string
 *   (`"VALID"`, `"SAME"`, or `"SAME_LOWER"`)
 */
export function convGeneralDilated(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  {
    lhsDilation,
    rhsDilation,
    featureGroupCount = 1,
  }: {
    lhsDilation?: number[];
    rhsDilation?: number[];
    featureGroupCount?: number;
  } = {},
): Array {
  if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
  if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
  if (typeof padding === "string") {
    if (lhsDilation?.some((d) => d !== 1)) {
      throw new Error(
        "String padding is not supported for transposed convolutions",
      );
    }
    padding = padtypeToPads(
      lhs.shape.slice(2),
      rhs.shape.slice(2),
      windowStrides,
      rhsDilation ?? rep(rhs.ndim - 2, 1),
      padding,
    );
  }
  if (featureGroupCount !== 1) {
    // We implement grouped convolutions by using leading vmapDims in the
    // convolution operator, then concatenating at the end.
    //
    // lhs: [N, C_in, ...xs]         -> [G, N, C_in / G, ...xs]
    // rhs: [C_out, C_in / G, ...ks] -> [G, C_out / G, C_in / G, ...ks]
    //
    // Convolve normally to get [G, N, C_out / G, ...ys], then move the G axis
    // back and reshape to [N, C_out, ...ys].
    const G = featureGroupCount;
    const [N, C_in, ...xs] = lhs.shape;
    const [C_out, C_in_per_group, ...ks] = rhs.shape;
    if (C_in % G !== 0) {
      throw new Error(
        `featureGroupCount=${G} must divide input channels=${C_in}`,
      );
    }
    if (C_out % G !== 0) {
      throw new Error(
        `featureGroupCount=${G} must divide output channels=${C_out}`,
      );
    }
    if (C_in / G !== C_in_per_group) {
      throw new Error(
        `rhs input channels=${C_in_per_group} must equal lhs input channels / groups=${C_in / G}`,
      );
    }
    using lhsReshaped = lhs.reshape([N, G, C_in / G, ...xs]);
    using lhsGrouped = moveaxis(lhsReshaped, 1, 0);
    using rhsGrouped = rhs.reshape([G, C_out / G, C_in_per_group, ...ks]);
    using result = core.conv(lhsGrouped, rhsGrouped, {
      vmapDims: 1, // Batch over G
      strides: windowStrides,
      padding,
      lhsDilation,
      rhsDilation,
    }) as Array;
    const ys = result.shape.slice(3);
    using moved = moveaxis(result, 0, 1) as Array;
    return moved.reshape([N, C_out, ...ys]);
  }
  return core.conv(lhs, rhs, {
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation,
  }) as Array;
}

/** Convenience wrapper around `convGeneralDilated`. */
export function convWithGeneralPadding(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  lhsDilation?: number[],
  rhsDilation?: number[],
): Array {
  return convGeneralDilated(lhs, rhs, windowStrides, padding, {
    lhsDilation,
    rhsDilation,
  });
}

/** Convenience wrapper around `convGeneralDilated`. */
export function conv(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
): Array {
  return convGeneralDilated(lhs, rhs, windowStrides, padding);
}

/**
 * Convenience wrapper for calculating the N-d convolution "transpose".
 *
 * This function directly calculates a fractionally strided conv rather than
 * indirectly calculating the gradient (transpose) of a forward convolution.
 * It is equivalent to the JAX version, except:
 *
 * - The `use_consistent_padding` option is not available. We only have the
 *   consistent padding case (JAX version >0.8.4).
 * - The order of dimensions matches `lax.conv_general_dilated`.
 *
 * Unlike PyTorch/TensorFlow, by default we don't reverse the kernel's spatial
 * dimensions or the `(C_out, C_in)` axis order. To get this behavior, set
 * `transposeKernel` to true.
 *
 * @param lhs - Input tensor; shape `[N, C_in, ...xs]`
 * @param rhs - Convolution kernel; shape `[C_out, C_in, ...ks]`
 * @param strides - Sequence of n integers, sets fractional stride
 * @param padding - Apply padding of `dilation * (kernel_size - 1) - padding` to
 *   each side of the input, so it acts like gradient of `conv()`
 * @param rhsDilation - Atrous dilation for the kernel
 * @param transposeKernel - Flip spatial axes and swap the input/output channels
 *   of the kernel; its shape should be `[C_in, C_out, ...ks]`
 */
export function convTranspose(
  lhs: Array,
  rhs: Array,
  strides: number[],
  padding: PaddingType,
  {
    rhsDilation,
    transposeKernel = false,
  }: {
    rhsDilation?: number[];
    transposeKernel?: boolean;
  } = {},
): Array {
  // Reference: https://github.com/jax-ml/jax/blob/c656803/jax/_src/lax/convolution.py#L296
  const kernelShape = rhs.shape.slice(2);
  // Calculate correct output shape from padding and strides.
  rhsDilation = rhsDilation ?? rep(kernelShape.length, 1);
  const effectiveKernel = kernelShape.map((k, i) =>
    Math.max(0, (k - 1) * rhsDilation[i] + 1),
  );
  const pads = effectiveKernel.map((k, i) =>
    convTransposePadding(
      k,
      strides[i],
      typeof padding === "string" ? padding : padding[i],
    ),
  );
  if (transposeKernel) {
    // Flip spatial axes and swap C_out/C_in.
    using flipped = core.flip(rhs, range(2, rhs.ndim)) as Array;
    rhs = moveaxis(flipped, 0, 1) as Array;
  }
  return convGeneralDilated(lhs, rhs, rep(lhs.ndim - 2, 1), pads, {
    lhsDilation: strides,
    rhsDilation,
  });
}

// Reference: https://github.com/jax-ml/jax/pull/32268
function convTransposePadding(
  k: number,
  s: number,
  padding: string | Pair,
): Pair {
  let padLen: number;
  let pad1: number;
  if (padding === "SAME") {
    padLen = k + s - 2;
    pad1 = s > k - 1 ? k - 1 : Math.ceil(padLen / 2);
  } else if (padding === "VALID") {
    padLen = k + s - 2 + Math.max(k - s, 0);
    pad1 = k - 1;
  } else if (JsArray.isArray(padding)) {
    const pads = [k - 1 - padding[0], k - 1 - padding[1]];
    pad1 = pads[0];
    padLen = pads[0] + pads[1];
  } else {
    throw new Error(`convTranspose: Invalid padding type ${padding}`);
  }
  return [pad1, padLen - pad1];
}

/** Reduce a computation over padded windows. */
export function reduceWindow(
  operand: Array,
  computation: (x: Array) => Array,
  windowDimensions: number[],
  windowStrides?: number[],
): Array {
  if (operand.ndim < windowDimensions.length) {
    throw new Error(
      `Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`,
    );
  }
  if (!windowStrides) windowStrides = rep(windowDimensions.length, 1);

  for (let i = 0; i < operand.ndim; i++) {
    // Vmap the computation over any pre-pooled dimensions.
    computation = vmap(computation, 0) as any;
  }
  const pooled = bind1(Primitive.Pool, [operand], {
    window: windowDimensions,
    strides: windowStrides,
  }) as Array;
  const result = computation(pooled);
  if (pooled !== result) pooled[Symbol.dispose]?.();
  return result;
}

/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
export function erf(x: ArrayLike): Array {
  return core.erf(x) as Array;
}

/**
 * The complementary error function: `erfc(x) = 1 - erf(x)`.
 *
 * This function is more accurate than `1 - erf(x)` for large values of `x`,
 * where `erf(x)` is very close to 1.
 */
export function erfc(x: ArrayLike): Array {
  return core.erfc(x) as Array;
}

/**
 * Stops gradient computation.
 *
 * Behaves as the identity function but prevents the flow of gradients during
 * forward or reverse-mode automatic differentiation.
 */
export function stopGradient(x: ArrayLike): Array {
  return core.stopGradient(x) as Array;
}

/**
 * Replace a contiguous slice of `dst` with `src`, returning a new array.
 *
 * **Single-axis form** (existing):
 * ```ts
 * lax.dynamicUpdateSlice(dst, src, offset, axis?)
 * ```
 * Equivalent to `dst.at[axis, offset:offset+src.shape[axis]].set(src)`.
 *
 * **ND form** (new):
 * ```ts
 * lax.dynamicUpdateSlice(dst, src, [startRow, startCol, ...])
 * ```
 * Equivalent to `dst.at[s0:s0+h0, s1:s1+h1, ...].set(src)` where
 * `si = startIndices[i]` and `hi = src.shape[i]`.
 *
 * @param dst  - Target array.
 * @param src  - Source array (same rank as dst for ND form, or matching non-axis dims for single-axis).
 * @param offsetOrStartIndices - Single axis offset (number) or ND start indices (number[]).
 * @param axis - Axis to update (only for single-axis form, default 0).
 */
export function dynamicUpdateSlice(
  dst: ArrayLike,
  src: ArrayLike,
  offset: number,
  axis?: number,
): Array;
export function dynamicUpdateSlice(
  dst: ArrayLike,
  src: ArrayLike,
  startIndices: number[],
): Array;
export function dynamicUpdateSlice(
  dst: ArrayLike,
  src: ArrayLike,
  offsetOrStartIndices: number | number[],
  axis?: number,
): Array {
  if (typeof offsetOrStartIndices === "number") {
    // Single-axis form: dynamicUpdateSlice(dst, src, offset, axis?)
    return core.dynamicUpdateSlice(
      fudgeArray(dst),
      fudgeArray(src),
      offsetOrStartIndices,
      axis ?? 0,
    ) as Array;
  }
  // ND form: dynamicUpdateSlice(dst, src, startIndices)
  if (axis !== undefined) {
    throw new Error(
      "lax.dynamicUpdateSlice: axis parameter is not supported with ND startIndices",
    );
  }
  return core.dynamicUpdateSliceGeneral(
    fudgeArray(dst),
    fudgeArray(src),
    offsetOrStartIndices,
  ) as Array;
}

/**
 * Extracts a contiguous slice of size `limit - start` along `axis`.
 *
 * Equivalent to `np.take(x, range(start, limit), axis)`, but implemented
 * as a zero-copy ShapeTracker view via `Primitive.Shrink`.
 *
 * Negative indices are supported (counted from end of axis).
 * If `limit` is omitted it defaults to the axis size.
 *
 * @param x     - Input array.
 * @param start - Start index (inclusive). Negative counts from end.
 * @param limit - End index (exclusive). Negative counts from end. Defaults to axis size.
 * @param axis  - Axis to slice along (default 0).
 */
export function sliceInDim(
  x: ArrayLike,
  start: number,
  limit?: number,
  axis: number = 0,
): Array {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const axisSize = x.shape[axis];

  // Normalise negative indices
  if (start < 0) start += axisSize;
  if (limit === undefined) limit = axisSize;
  else if (limit < 0) limit += axisSize;

  // Clamp to valid range
  start = Math.max(0, Math.min(start, axisSize));
  limit = Math.max(start, Math.min(limit, axisSize));

  const slices: Pair[] = x.shape.map((s, i) =>
    i === axis ? [start, limit!] : [0, s],
  );
  return core.shrink(x, slices) as Array;
}

/**
 * Extracts a single element (or size-1 slice) along `axis`.
 *
 * With `keepdims = false` (default), the indexed axis is removed from the
 * result shape, mimicking scalar indexing (`arr[i]`). With `keepdims = true`,
 * the axis is preserved with size 1.
 *
 * AD-safe: differentiable with respect to `operand` (gradient is a one-hot
 * scatter back into a zeros array at `index`). Not differentiable with
 * respect to `index`.
 *
 * @param operand  - Input array.
 * @param index    - Index along `axis`. Negative counts from end.
 * @param axis     - Axis to index along (default 0).
 * @param keepdims - If true, keep the indexed axis with size 1 (default false).
 */
export function dynamicIndexInDim(
  operand: ArrayLike,
  index: number,
  axis: number = 0,
  keepdims: boolean = false,
): Array {
  operand = fudgeArray(operand);
  axis = checkAxis(axis, operand.ndim);
  const axisSize = operand.shape[axis];
  if (index < 0) index += axisSize;
  const result = sliceInDim(operand, index, index + 1, axis);
  if (keepdims) return result;
  // Squeeze the axis — reshape to remove the size-1 dimension
  const newShape = result.shape.filter((_, i) => i !== axis);
  const squeezed = core.reshape(result, newShape) as Array;
  result.dispose();
  return squeezed;
}

/**
 * Returns top `k` values and their indices along the specified axis of operand.
 *
 * This is a _stable_ algorithm: If two elements are equal, the lower-index
 * element appears first.
 *
 * @returns A tuple of `(values, indices)`, where `values` and `indices` have
 * the same shape as `x`, except along `axis` where they have size `k`.
 */
export function topK(
  x: ArrayLike,
  k: number,
  axis: number = -1,
): [Array, Array] {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const size = x.shape[axis];

  if (k < 0 || k > size)
    throw new Error(`topK: k must be in the range [0, ${size}], got ${k}`);
  if (k === 0) {
    const outShape = x.shape.slice();
    outShape[axis] = 0;
    const y = zerosLike(x, { shape: outShape });
    const yi = zerosLike(x, { dtype: DType.Int32, shape: outShape });
    return [y, yi];
  }

  // We want to sort it in descending order, therefore we reverse before and
  // after argsort. This ensures that ties are resolved by smaller index.
  const flipped = core.flip(x, [axis]) as Array;
  const moved = moveaxis(flipped, axis, -1) as Array;
  const argsortResult = core.argsort(moved);
  const y = argsortResult[0];
  const yi = argsortResult[1];
  const extract = (a: core.Tracer) => {
    const sliced = a.slice(...rep(a.ndim - 1, [] as []), [-k]);
    const movedBack = moveaxis(sliced, -1, axis);
    const result = core.flip(movedBack, [axis]) as Array;
    if (movedBack !== sliced) (movedBack as Array).dispose();
    (sliced as Array).dispose();
    return result;
  };
  const neg = yi.neg() as Array;
  const adjusted = neg.add(size - 1) as Array;
  const result: [Array, Array] = [extract(y), extract(adjusted)];
  adjusted.dispose();
  neg.dispose();
  yi.dispose();
  y.dispose();
  if (moved !== flipped) (moved as Array).dispose();
  flipped.dispose();
  return result;
}

/**
 * Extract a slice of `operand` at runtime-computed `startIndices` with
 * compile-time-constant `sliceSizes`. Out-of-bounds start indices are
 * clamped to `[0, dimSize - sliceSize]`.
 */
export function dynamicSlice(
  operand: ArrayLike,
  startIndices: ArrayLike[],
  sliceSizes: number[],
): Array {
  const x = fudgeArray(operand);
  const starts = startIndices.map(fudgeArray);
  if (starts.length !== x.ndim) {
    throw new Error(
      `lax.dynamicSlice: expected ${x.ndim} start indices, got ${starts.length}`,
    );
  }
  if (sliceSizes.length !== x.ndim) {
    throw new Error(
      `lax.dynamicSlice: expected ${x.ndim} slice sizes, got ${sliceSizes.length}`,
    );
  }
  for (const start of starts) {
    if (start.ndim !== 0) {
      throw new Error(
        `lax.dynamicSlice: start indices must be scalars, got shape ${start.shape}`,
      );
    }
  }
  const result = bind1(Primitive.DynamicSlice, [x, ...starts], {
    sliceSizes,
  }) as Array;
  return result;
}

/**
 * Like {@link dynamicSlice} but without min/max clamping on start indices.
 * The caller MUST guarantee that start indices are in-bounds:
 * `0 <= start[k] && start[k] + sliceSizes[k] <= operandShape[k]` for all k.
 *
 * @internal Used by `tiledMatmul` where padding guarantees alignment.
 */
export function uncheckedDynamicSlice(
  operand: ArrayLike,
  startIndices: ArrayLike[],
  sliceSizes: number[],
): Array {
  const x = fudgeArray(operand);
  const starts = startIndices.map(fudgeArray);
  const result = bind1(Primitive.UncheckedDynamicSlice, [x, ...starts], {
    sliceSizes,
  }) as Array;
  return result;
}

/**
 * Sequential loop with a carried state.
 *
 * **Performance Note (WebGPU):** Under `jit()`, the compiler may rewrite this into a
 * fused `BlockMap` that executes the entire loop in a single GPU dispatch. The rewrite
 * applies when **all** of the following hold:
 * - Backend is WebGPU (WASM already compiles foriLoop natively via mega-module)
 * - Loop bounds are concrete integers with ≥ 2 iterations
 * - Body consists entirely of pointwise elementwise operations (no matmul, reduce, scan, etc.)
 * - All carries have the same concrete shape, rank 1–2, with each dimension ≥ 4
 * - Constants are either same-shape-as-carry or scalar
 * - Total shared memory for tiles fits within `maxComputeWorkgroupStorageSize`
 * - The foriLoop is at the top level of the jit-compiled function (not nested in scan/blockMap)
 *
 * When the rewrite fires, the loop runs in GPU register threads (`var<private>`) without
 * VRAM roundtrips. Use `_lastForiRewritten()` (test-only) or `setDebug(1)` to verify.
 *
 * **Autodiff limitations:**
 * - `grad(jit(f))` is **not supported** when `f` contains `foriLoop`. Use `jit(grad(f))` instead.
 * - Reverse-mode with symbolic (Dim) bounds requires `lower = 0`; general symbolic lower bounds
 *   are not yet supported in the transpose rule.
 *
 * **Dynamic bounds:** The type signature accepts `Dim` for both bounds, but eager-mode execution
 * requires concrete numbers. Symbolic `Dim` bounds are resolved at runtime only under `jit()`.
 */
export function foriLoop<C extends tree.JsTree<Array>>(
  lower: number | Dim,
  upper: number | Dim,
  body: (i: Array, carry: C) => C,
  init: C,
): C {
  if (typeof lower === "number") lower = Math.floor(lower);
  if (typeof upper === "number") upper = Math.floor(upper);
  if (
    typeof lower === "number" &&
    typeof upper === "number" &&
    upper - lower <= 0
  )
    return init;

  const [initFlat, initTree] = tree.flatten(init);

  const dummyIArr = arrayFn(0, { dtype: DType.Int32 });
  const dummyI = core.getAval(dummyIArr);
  dummyIArr.dispose();

  const traceAvals = [dummyI, ...initFlat.map((a) => core.getAval(a as Array))];

  const traceFn = (i: Array, ...args: Array[]) => {
    const initTracers = tree.unflatten(initTree, args);
    const out = body(i, initTracers as C);
    const [outFlat] = tree.flatten(out);
    return outFlat;
  };

  const { jaxpr: closedJaxpr } = jaxpr.makeJaxpr(traceFn)(...traceAvals);

  const outFlat = core.bind(
    Primitive.ForiLoop,
    [...closedJaxpr.consts, ...(initFlat as Array[])],
    {
      jaxpr: closedJaxpr.jaxpr,
      numConsts: closedJaxpr.consts.length,
      lower,
      upper,
    },
  ) as Array[];

  closedJaxpr.dispose();

  return tree.unflatten(initTree, outFlat) as C;
}

/**
 * Workgroup-local associative prefix scan (Kogge-Stone).
 *
 * Computes an inclusive prefix scan using an associative binary operator.
 * Designed for use inside {@link blockMap} bodies where each workgroup runs
 * the scan entirely in shared memory.
 *
 * In eager mode / inside `block_map` fallback, delegates to the
 * `associativeScan` Kogge-Stone implementation (correct result, just not
 * shared-memory-accelerated).
 *
 * In JIT mode inside a `block_map` body, the fused shader compiler emits
 * Kogge-Stone rounds in WGSL with `workgroupBarrier()` between rounds and
 * ping-pong shared-memory arrays.
 *
 * @param fn - Associative binary operator: `(a, b) => c`. Must not close
 *   over mutable state. Must dispose any internally-created Arrays.
 * @param elems - 1D input array.
 * @returns Prefix scan result with the same shape as `elems`.
 *
 * @example Cumulative sum inside a block_map body
 * ```ts
 * const result = lax.blockMap(
 *   (block) => lax.workgroupAssociativeScan((a, b) => np.add(a, b), block),
 *   xs,
 *   { blockShape: [256] },
 * );
 * ```
 */
export function workgroupAssociativeScan(
  fn: (a: Array, b: Array) => Array,
  elems: Array,
): Array {
  const elemAval = core.getAval(elems);

  // Trace the binary operator with scalar avals (scan-axis-removed shapes)
  const scalarAval = new ShapedArray(
    elemAval.shape.slice(1),
    elemAval.dtype,
    elemAval.weakType,
  );
  const traceFn = (a: Array, b: Array): Array[] => {
    const result = fn(a, b);
    return [result];
  };
  const { jaxpr: closedJaxpr } = jaxpr.makeJaxpr(traceFn)(
    scalarAval,
    scalarAval,
  );

  const results = core.bind(
    Primitive.WorkgroupAssociativeScan,
    [...closedJaxpr.consts, elems],
    {
      jaxpr: closedJaxpr.jaxpr,
      numConsts: closedJaxpr.consts.length,
    },
  ) as Array[];

  closedJaxpr.dispose();
  return results[0];
}

/**
 * Concrete padding: allocate a zero-filled buffer and copy the original
 * data into it via dynamicUpdateSlice.  The result has no ShapeTracker
 * mask, so downstream codegen produces raw reads instead of
 * `select(0, read, valid_mask)`.
 *
 * Falls back to the lightweight mask-based `core.pad` when the overhead
 * would be disproportionate (tiny matrices or large expansion ratios).
 */
function padConcrete(
  x: Array,
  padWidths: Record<number, [number, number]>,
): { result: Array; concrete: boolean } {
  const shape = x.shape as number[];
  const paddedShape = shape.slice();
  for (const [axisStr, [lo, hi]] of Object.entries(padWidths)) {
    const axis = parseInt(axisStr);
    paddedShape[axis] += lo + hi;
  }
  const origElements = prod(shape);
  const paddedElements = prod(paddedShape);

  // Heuristic gate: only materialize when copy cost is negligible
  if (origElements <= 1024 || paddedElements / origElements >= 1.25) {
    return { result: core.pad(x, padWidths) as Array, concrete: false };
  }

  // Pad one axis at a time via DUS. Each iteration expands one dimension
  // so the src/dst shapes match on all non-axis dimensions.
  const axes = Object.entries(padWidths);
  let current: Array = x;
  let ownsCurrentFromPadding = false;
  for (const [axisStr, [lo]] of axes) {
    const axis = parseInt(axisStr);
    // Build the target shape: current's shape but with this axis expanded
    const targetShape = (current.shape as number[]).slice();
    targetShape[axis] = paddedShape[axis];
    const zeros = fullInternal(new ShapedArray(targetShape, x.dtype, false), 0);
    const next = core.dynamicUpdateSlice(zeros, current, lo, axis) as Array;
    zeros.dispose();
    if (ownsCurrentFromPadding) current.dispose();
    current = next;
    ownsCurrentFromPadding = true;
  }
  return { result: current, concrete: true };
}

/**
 * Options for {@link tiledMatmul}.
 */
export interface TiledMatmulOptions {
  /** Block size for the output tile rows (default: auto-selected). */
  Br?: number;
  /** Block size for the output tile columns (default: auto-selected). */
  Bc?: number;
  /** Block size for the contraction dimension (default: auto-selected). */
  Bk?: number;
  /** Register tiling: each thread handles threadTile[g] elements per axis. */
  threadTile?: number[];
}

/**
 * Choose tile configuration based on device capabilities.
 *
 * The candidates are tried from most to least aggressive. Each must satisfy:
 * - numThreads ≤ maxComputeInvocationsPerWorkgroup
 * - shmem usage ≤ maxComputeWorkgroupStorageSize
 *
 * The candidate list was benchmarked on NVIDIA RTX 4070 Ti Super and Intel
 * xe-lpg (Meteor Lake) with outer-product loop ordering (O12) and carry-direct
 * accumulation (O12a). Validated against WebGPU device limit survey:
 * - 100% of devices: maxInvocations ≥ 256, shmem ≥ 16384
 * - 81% of devices: maxInvocations ≥ 1024, shmem ≥ 32768
 *
 * 64×64 tt44 is the top choice: 16 outputs per thread maximize register reuse
 * in the outer-product reduction loop. At N=2048 it achieves 3536 GFLOP/s on
 * NVIDIA (vs 2439 for 32×32 tt22) and 533 on Intel iGPU (vs 305).
 *
 * On older Intel iGPUs (gen-9, gen-11, gen-12lp) the limited register file
 * causes catastrophic spilling with threadTile=[4,4], making execution 100×+
 * slower and triggering TDR at N≥1024. These architectures fall through to
 * 32×32 tt22 which benchmarked at 32.4 GFLOP/s (gen-9, N=1024).
 */
function chooseTileConfig(
  caps: BackendCapabilities,
  dtype: DType,
): Required<Pick<TiledMatmulOptions, "Br" | "Bc" | "Bk">> &
  Pick<TiledMatmulOptions, "threadTile"> {
  const maxInvocations = caps.maxComputeInvocationsPerWorkgroup ?? 256;
  const maxShmem = caps.maxComputeWorkgroupStorageSize ?? 16384;
  const bytesPerElem = dtype === DType.Float16 ? 2 : 4;
  const arch = caps.adapterArchitecture?.toLowerCase() ?? "";
  const vendor = caps.adapterVendor?.toLowerCase() ?? "";

  // Gate: 64×64 tt44 (256 threads × 16 registers) causes catastrophic register
  // spill on older Intel iGPUs (gen-9, gen-11, gen-12lp — ≤32 EU), triggering
  // TDR. 32×32 tt44 (64 threads × 16 registers) works fine thanks to O12b
  // carry-accumulation fusion eliminating the separate accumulator array.
  const isOldIntelIGPU =
    arch.startsWith("gen-") || arch === "gen_12lp" || arch === "gen_11";

  // Only allow 64×64 tt44 (256 threads × 16 regs = 4096 total registers) on
  // GPUs positively identified as having large register files. Unknown GPUs
  // and mobile GPUs (Adreno, Mali) may suffer the same register spill as
  // old Intel iGPUs. 32×32 tt44 (64 threads × 16 regs) is the safe default.
  const isKnownPowerful =
    vendor === "nvidia" ||
    vendor === "apple" ||
    arch.startsWith("xe-") || // Intel Arc / Meteor Lake
    arch.startsWith("rdna"); // AMD RDNA

  // Candidates ordered by expected performance (best first).
  // numThreads = (Br/tt[0]) * (Bc/tt[1])
  // shmem ≈ (Br*Bk + Bk*Bc) * bytesPerElem + bank padding overhead
  const candidates: {
    Br: number;
    Bc: number;
    Bk: number;
    threadTile?: [number, number];
  }[] = [
    { Br: 64, Bc: 64, Bk: 16, threadTile: [4, 4] }, // 256 threads, 16 out/thread — best on powerful GPUs
    { Br: 32, Bc: 32, Bk: 16, threadTile: [4, 4] }, // 64 threads, 16 out/thread — good for small register files
    { Br: 32, Bc: 32, Bk: 16, threadTile: [2, 2] }, // 256 threads, 4 out/thread
    { Br: 16, Bc: 16, Bk: 16 }, // 256 threads, 1 out/thread (safe fallback)
  ];

  for (const c of candidates) {
    const tt = c.threadTile;
    const numThreads = tt ? (c.Br / tt[0]) * (c.Bc / tt[1]) : c.Br * c.Bc;
    if (numThreads > maxInvocations) continue;

    // Skip large tt44 (256 threads) on GPUs with insufficient register files.
    // Small tt44 (64 threads at 32×32) is safe and 1.5× faster than tt22.
    // On unknown GPUs, also skip 64×64 to avoid catastrophic register spill.
    if (tt && tt[0] >= 4 && tt[1] >= 4 && numThreads >= 256) {
      if (isOldIntelIGPU || !isKnownPowerful) continue;
    }

    // Estimate shmem: A tile + B tile + bank padding (~6% overhead)
    const tileA = c.Br * c.Bk * bytesPerElem;
    const tileB = c.Bk * c.Bc * bytesPerElem;
    const padOverhead = Math.ceil((tileA + tileB) * 0.07);
    const shmemBytes = tileA + tileB + padOverhead;
    if (shmemBytes > maxShmem) continue;

    return c;
  }

  // Ultimate fallback — always fits (256 threads, 2KB shmem)
  return { Br: 16, Bc: 16, Bk: 16 };
}

/**
 * Tiled matrix multiplication via {@link blockMap} + {@link foriLoop}.
 *
 * Computes `A @ B` for 2D matrices by decomposing the multiplication into
 * block tiles processed in shared memory on WebGPU. Falls back to the standard
 * matmul kernel on WASM/CPU.
 *
 * M, N, and K are padded with zeros if not divisible by the tile sizes.
 *
 * When no explicit tile sizes are provided, the configuration is automatically
 * selected based on the GPU's compute limits (workgroup size, shared memory).
 *
 * @param A - Left matrix of shape `[M, K]`.
 * @param B - Right matrix of shape `[K, N]`.
 * @param options - Optional tile size configuration.
 * @returns Result matrix of shape `[M, N]`.
 */
export function tiledMatmul(
  A: Array,
  B: Array,
  options?: TiledMatmulOptions,
): Array {
  // tiledMatmul is a WebGPU-specific optimization. On non-WebGPU backends,
  // fall back to a simple dot product — blockMap + foriLoop traces hundreds
  // of IR nodes whose intermediates leak through grad on WASM/CPU.
  if (getBackend().type !== "webgpu") {
    return dot(A, B, {
      lhsContractingDims: [1],
      rhsContractingDims: [0],
    });
  }

  // --- Strategy 3: Zero-shmem Micro-panel Tiling ---
  // Gen-9 and Gen-11 have very few Execution Units and suffer profoundly from
  // workgroupBarrier() synchronizations. Furthermore, their emulated f16
  // suffers from register bloat during shared memory reads.
  // We explicitly route f16 on these architectures to the zero-shmem flat
  // shader (core.dot), which uses the Tuner's applyUpcast (vectorized
  // loads) and applyUnroll to act as a micro-panel tiled flat shader.
  // This restores the ~75—111 GFLOP/s performance natively.
  const arch =
    getBackend().capabilities.adapterArchitecture?.toLowerCase() ?? "";
  const isOldIntelIGPU =
    arch.startsWith("gen-") || arch === "gen_12lp" || arch === "gen_11";

  if (isOldIntelIGPU && A.dtype === DType.Float16) {
    return dot(A, B, {
      lhsContractingDims: [1],
      rhsContractingDims: [0],
    });
  }

  // When caller doesn't specify tile sizes, auto-select based on device caps.
  let resolved: Required<Pick<TiledMatmulOptions, "Br" | "Bc" | "Bk">> &
    Pick<TiledMatmulOptions, "threadTile">;
  if (
    options?.Br !== undefined ||
    options?.Bc !== undefined ||
    options?.Bk !== undefined
  ) {
    resolved = {
      Br: options.Br ?? 16,
      Bc: options.Bc ?? 16,
      Bk: options.Bk ?? 16,
      threadTile: options.threadTile,
    };
  } else {
    const caps = getBackend().capabilities;
    const auto = chooseTileConfig(caps, A.dtype);
    resolved = {
      ...auto,
      // Explicit threadTile overrides auto-selected one
      threadTile: options?.threadTile ?? auto.threadTile,
    };
  }
  const { Br, Bc, Bk, threadTile } = resolved;

  if (A.ndim !== 2 || B.ndim !== 2) {
    throw new Error(
      `tiledMatmul: expected 2D inputs, got ${A.ndim}D and ${B.ndim}D`,
    );
  }
  const [M, K] = A.shape as [number, number];
  const [K2, N] = B.shape as [number, number];
  if (K !== K2) {
    throw new Error(
      `tiledMatmul: contraction dims must match, got ${K} and ${K2}`,
    );
  }

  // Pad M, N, K to multiples of tile sizes if needed
  const Mpad = Math.ceil(M / Br) * Br;
  const Npad = Math.ceil(N / Bc) * Bc;
  const Kpad = Math.ceil(K / Bk) * Bk;
  let aInput = A;
  let bInput = B;
  const needsPad = Mpad !== M || Npad !== N || Kpad !== K;
  if (needsPad) {
    const padM = Mpad - M;
    const padN = Npad - N;
    const padK = Kpad - K;
    if (padM > 0 || padK > 0) {
      const aPad: Record<number, [number, number]> = {};
      if (padM > 0) aPad[0] = [0, padM];
      if (padK > 0) aPad[1] = [0, padK];
      const { result } = padConcrete(A, aPad);
      aInput = result;
    }
    if (padN > 0 || padK > 0) {
      const bPad: Record<number, [number, number]> = {};
      if (padK > 0) bPad[0] = [0, padK];
      if (padN > 0) bPad[1] = [0, padN];
      const { result } = padConcrete(B, bPad);
      bInput = result;
    }
  }

  try {
    const numKTiles = Kpad / Bk;
    const fullResult = blockMap(
      ({ A: aTile, B: bTile }: { A: Array; B: Array }) =>
        foriLoop(
          0,
          numKTiles,
          (k: Array, acc: Array) => {
            using kIdx = core.mul(
              k,
              arrayFn(Bk, { dtype: DType.Int32 }),
            ) as Array;
            using z0 = arrayFn(0, { dtype: DType.Int32 });
            // Padding guarantees aTile and bTile shapes are exact tile multiples,
            // so kIdx is always in-bounds — use unchecked for no min/max clamping.
            using a = uncheckedDynamicSlice(aTile, [z0, kIdx], [Br, Bk]);
            using b = uncheckedDynamicSlice(bTile, [kIdx, z0], [Bk, Bc]);
            using prod = dot(a, b, {
              lhsContractingDims: [1],
              rhsContractingDims: [0],
            });
            return core.add(acc, prod) as Array;
          },
          fullInternal(new ShapedArray([Br, Bc], A.dtype, false), 0),
        ),
      { A: aInput, B: bInput },
      {
        blockShape: [Br, Bc],
        inAxes: [
          [0, null],
          [null, 1],
        ],
        outAxes: [[0, 1]],
        threadTile,
      },
    );
    // Slice off padding if M or N were padded (dynamicSlice produces contiguous output, unlike shrink)
    if (Mpad !== M || Npad !== N) {
      using fullPadded = fullResult;
      using z0 = arrayFn(0, { dtype: DType.Int32 });
      return dynamicSlice(fullPadded, [z0, z0], [M, N]);
    }
    return fullResult;
  } finally {
    if (needsPad) {
      aInput.dispose();
      bInput.dispose();
    }
  }
}
