/** @file Core library internals and interpreter stack, based on Autodidax. */

import { AluGroup, AluOp, DType, isFloatDtype, promoteTypes } from "../alu";
import { Routines } from "../routine";
import {
  type Dim,
  dimCompatible,
  dimEquals,
  dimProduct,
  hasSymbolicDims,
  isSymbolicDim,
  type Pair,
  type SizeExpr,
} from "../shape";
import {
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import {
  checkAxis,
  DEBUG,
  deepEqual,
  generalBroadcast,
  invertPermutation,
  isNumberPair,
  isPermutation,
  normalizeAxis,
  prod,
  range,
  rep,
} from "../utils";
import type { ConvParams } from "./convolution";
import type { Jaxpr } from "./jaxpr";

/**
 * Frontend primitive operations, which are lowered into Kernel objects before
 * being dispatched to the backend.
 *
 * Any operation between arrays can be described in these parts. This is also
 * the set of primitives that can occur in Jaxpr programs, and the level at
 * which transformations like vmap, grad, and jvp occur. They are loosely based
 * on [XLA](https://openxla.org/xla/operation_semantics).
 *
 * All n-ary operations support broadcasting, with NumPy semantics.
 */
export enum Primitive {
  // Arithmetic (binary operations)
  Add = "add",
  Mul = "mul",
  Idiv = "idiv",
  Mod = "mod", // uses sign of numerator, C-style, matches JS but not Python
  Min = "min",
  Max = "max",
  BitCombine = "bit_combine",
  BitShift = "bit_shift",

  // Unary functions and type casting
  Neg = "neg",
  Reciprocal = "reciprocal",
  Floor = "floor",
  Ceil = "ceil",
  StopGradient = "stop_gradient",
  Cast = "cast",
  Bitcast = "bitcast",
  Sin = "sin",
  Cos = "cos",
  Asin = "asin",
  Atan = "atan",
  Exp = "exp",
  Log = "log",
  Erf = "erf",
  Erfc = "erfc",
  Sqrt = "sqrt",

  // Reductions, convolution and pooling
  Reduce = "reduce",
  Dot = "dot", // sum(x*y, axis=-1)
  Conv = "conv", // see lax.conv_general_dilated
  Pool = "pool",
  PoolTranspose = "pool_transpose",

  // Utility
  Compare = "compare",
  Where = "where",
  Concatenate = "concatenate",
  Split = "split",
  RandomBits = "random_bits",
  Gather = "gather",

  // Movement
  Transpose = "transpose",
  Broadcast = "broadcast",
  Reshape = "reshape",
  Flip = "flip",
  Shrink = "shrink",
  Pad = "pad",
  // Update a contiguous slice along a single axis: dst[axis=offset:offset+len] = src
  DynamicUpdateSlice = "dynamic_update_slice",
  // Update a contiguous slice with ND start indices: dst[starts[0]:..., starts[1]:..., ...] = src
  DynamicUpdateSliceGeneral = "dynamic_update_slice_general",
  // Extract a contiguous slice using dynamic start indices
  DynamicSlice = "dynamic_slice",
  // Like DynamicSlice but without min/max clamping (caller guarantees in-bounds)
  UncheckedDynamicSlice = "unchecked_dynamic_slice",
  // Reverse elements along an axis (materializing, not a view)
  Reverse = "reverse",
  // Atomic scatter-add: target[indices[i]] += updates[i]
  ScatterAdd = "scatter_add",

  // Routines (custom lowering)
  Sort = "sort", // sort(x, axis=-1), unstable
  Argsort = "argsort", // argsort(x, axis=-1), stable
  TriangularSolve = "triangular_solve", // A is upper triangular, A @ X.T = B.T
  Cholesky = "cholesky", // A is positive-definite, A = L @ L^T
  LU = "lu", // LU decomposition with partial pivoting
  QR = "qr", // QR decomposition via Householder reflections

  // JIT compilation
  Jit = "jit",

  // Control flow
  Scan = "scan",
  AssociativeScan = "associative_scan",
  BlockMap = "block_map",
  BlockIndex = "block_index",
  ForiLoop = "fori_loop",
  WorkgroupAssociativeScan = "workgroup_associative_scan",
}

interface PrimitiveParamsImpl extends Record<Primitive, Record<string, any>> {
  [Primitive.BitCombine]: { op: "and" | "or" | "xor" };
  [Primitive.BitShift]: { op: "shl" | "shr" };
  [Primitive.Cast]: { dtype: DType };
  [Primitive.Bitcast]: { dtype: DType };
  [Primitive.Reduce]: { op: AluOp; axis: number[] };
  [Primitive.Conv]: ConvParams;
  [Primitive.Pool]: { window: number[]; strides: number[] };
  [Primitive.PoolTranspose]: {
    inShape: number[];
    window: number[];
    strides: number[];
  };
  [Primitive.Compare]: { op: CompareOp };
  [Primitive.Concatenate]: { axis: number };
  [Primitive.Split]: { axis: number; sizes: number[] };
  [Primitive.RandomBits]: { shape: number[]; mode: "xor" | 0 | 1 };
  [Primitive.Gather]: { axis: number[]; outDim: number };
  [Primitive.Transpose]: { perm: number[] };
  [Primitive.Broadcast]: { shape: Dim[]; axis: number[] };
  [Primitive.Reshape]: { shape: number[] };
  [Primitive.Flip]: { axis: number[] };
  [Primitive.Reverse]: { axis: number };
  [Primitive.Shrink]: { slice: Pair[] };
  [Primitive.Pad]: { width: Pair[] };
  // Update a contiguous slice along a single axis: dst[axis=offset:offset+src.shape[axis]] = src
  [Primitive.DynamicUpdateSlice]: { offset: number; axis: number };
  // Update a contiguous ND slice: dst[starts[0]:..., starts[1]:..., ...] = src
  [Primitive.DynamicUpdateSliceGeneral]: { startIndices: number[] };
  [Primitive.DynamicSlice]: { sliceSizes: number[] };
  [Primitive.UncheckedDynamicSlice]: { sliceSizes: number[] };
  // Scatter-add: target[indices[i]] += updates[i] along axis
  [Primitive.ScatterAdd]: { axis: number };
  [Primitive.TriangularSolve]: { unitDiagonal: boolean };
  [Primitive.Jit]: {
    name: string;
    jaxpr: Jaxpr;
    numConsts: number;
    dynamicAxes?: Record<number, string>;
  };
  [Primitive.Scan]: {
    jaxpr: Jaxpr;
    numCarry: number;
    numConsts: number;
    length: number | Dim;
    reverse: boolean;
    /** Accepted scan path(s). Throws if actual path is not in this list. */
    acceptPath?: string | string[];
    /**
     * Control gradient checkpointing for the backward pass.
     * - `undefined` or `true` (default): use √N checkpointing with segment size ceil(√N)
     * - A positive integer: use that as the segment size
     * - `false`: store all intermediate carries (O(N) memory, no recomputation)
     */
    checkpoint?: boolean | number;
    /** Set by JVP rule when this scan was produced by differentiating an outer scan. */
    isJvpTransformed?: boolean;
  };
  [Primitive.AssociativeScan]: {
    jaxpr: Jaxpr;
    numLeaves: number;
    axis: number;
    reverse: boolean;
  };
  [Primitive.BlockMap]: {
    jaxpr: Jaxpr;
    blockShape: number[];
    inAxes: (number | null)[][];
    outAxes: (number | null)[][];
    numConsts: number;
    numInputs: number;
    /** Explicit grid shape. When provided, overrides grid inference from mapped inputs. */
    gridShape?: number[];
    /** Register tiling: each thread handles threadTile[g] outputs per axis. */
    threadTile?: number[];
    /** Per-input per-grid-axis overlap: halo[i][g] = [lo, hi]. Expands input tiles beyond output block range. */
    halo?: [number, number][][];
    /** Set by JVP rule when this block_map was produced by differentiating an outer block_map. */
    isJvpTransformed?: boolean;
    /** Original forward body jaxpr, set by JVP rule for stencil analysis in transpose. */
    originalJaxpr?: Jaxpr;
    /** Original body's numConsts, set alongside originalJaxpr. */
    originalNumConsts?: number;
  };
  [Primitive.ForiLoop]: {
    jaxpr: Jaxpr;
    numConsts: number;
    lower: number | Dim;
    upper: number | Dim;
    /** Set by JVP rule when this fori_loop was produced by differentiating. */
    isJvpTransformed?: boolean;
  };
  [Primitive.WorkgroupAssociativeScan]: {
    jaxpr: Jaxpr;
    numConsts: number;
  };
}

/** Type of parameters taken by each primitive. */
export type PrimitiveParams<T extends Primitive> =
  T extends keyof PrimitiveParamsImpl
    ? PrimitiveParamsImpl[T]
    : Record<string, never>;

export enum CompareOp {
  Less = "less",
  Equal = "equal",
  NotEqual = "not_equal",
  LessEqual = "less_equal",
}

// These primitives are handled via `Routine` instead of `Kernel` and are not
// compatible with operator fusion.
export const routinePrimitives = new Map([
  [Primitive.Sort, Routines.Sort],
  [Primitive.Argsort, Routines.Argsort],
  [Primitive.TriangularSolve, Routines.TriangularSolve],
  [Primitive.Cholesky, Routines.Cholesky],
  [Primitive.LU, Routines.LU],
  [Primitive.QR, Routines.QR],
]);

export function add(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Add, [x, y]);
}

export function mul(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mul, [x, y]);
}

export function idiv(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Idiv, [x, y]);
}

export function mod(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mod, [x, y]);
}

export function min(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Min, [x, y]);
}

export function max(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Max, [x, y]);
}

export function bitCombine(
  x: TracerValue,
  y: TracerValue,
  op: "and" | "or" | "xor",
) {
  return bind1(Primitive.BitCombine, [x, y], { op });
}

export function bitShift(x: TracerValue, y: TracerValue, op: "shl" | "shr") {
  return bind1(Primitive.BitShift, [x, y], { op });
}

export function neg(x: TracerValue) {
  return bind1(Primitive.Neg, [x]);
}

export function reciprocal(x: TracerValue) {
  return bind1(Primitive.Reciprocal, [x]);
}

export function floor(x: TracerValue) {
  return bind1(Primitive.Floor, [x]);
}

export function ceil(x: TracerValue) {
  return bind1(Primitive.Ceil, [x]);
}

export function stopGradient(x: TracerValue) {
  return bind1(Primitive.StopGradient, [x]);
}

export function cast(x: TracerValue, dtype: DType) {
  return bind1(Primitive.Cast, [x], { dtype });
}

export function bitcast(x: TracerValue, dtype: DType) {
  return bind1(Primitive.Bitcast, [x], { dtype });
}

export function sin(x: TracerValue) {
  return bind1(Primitive.Sin, [x]);
}

export function cos(x: TracerValue) {
  return bind1(Primitive.Cos, [x]);
}

export function asin(x: TracerValue) {
  return bind1(Primitive.Asin, [x]);
}

export function atan(x: TracerValue) {
  return bind1(Primitive.Atan, [x]);
}

export function exp(x: TracerValue) {
  return bind1(Primitive.Exp, [x]);
}

export function log(x: TracerValue) {
  return bind1(Primitive.Log, [x]);
}

export function erf(x: TracerValue) {
  return bind1(Primitive.Erf, [x]);
}

export function erfc(x: TracerValue) {
  return bind1(Primitive.Erfc, [x]);
}

export function sqrt(x: TracerValue) {
  return bind1(Primitive.Sqrt, [x]);
}

/** @inline */
export type Axis = number | number[] | null;

/** @inline */
export type ReduceOpts = { keepdims?: boolean };

export function reduce(
  x: TracerValue,
  op: AluOp,
  axis: Axis = null,
  opts?: ReduceOpts,
) {
  if (!AluGroup.Reduce.has(op)) {
    throw new TypeError(`Invalid reduce operation: ${op}`);
  }
  axis = normalizeAxis(axis, ndim(x));
  const originalShape = getShape(x);
  let result = bind1(Primitive.Reduce, [x], { op, axis });
  if (opts?.keepdims) {
    const preReshape = result;
    result = result.reshape(
      originalShape.map((dim, i) => (axis.includes(i) ? 1 : dim)),
    );
    if (preReshape !== result) preReshape.dispose();
  }
  return result;
}

export function dot(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Dot, [x, y]);
}

export function conv(x: Tracer, y: Tracer, params: Partial<ConvParams> = {}) {
  if (x.ndim !== y.ndim) {
    throw new Error(
      `conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`,
    );
  }
  const vmapDims = params.vmapDims ?? 0;
  const n = x.ndim - 2 - vmapDims;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  // conv shape check is delayed until interpretation.
  return bind1(Primitive.Conv, [x, y], {
    vmapDims,
    strides: params.strides ?? rep(n, 1),
    padding: params.padding ?? rep(n, [0, 0]),
    lhsDilation: params.lhsDilation ?? rep(n, 1),
    rhsDilation: params.rhsDilation ?? rep(n, 1),
  });
}

export function compare(x: TracerValue, y: TracerValue, op: CompareOp) {
  return bind1(Primitive.Compare, [x, y], { op });
}
export function greater(x: TracerValue, y: TracerValue) {
  return compare(y, x, CompareOp.Less);
}
export function less(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.Less);
}
export function equal(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.Equal);
}
export function notEqual(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.NotEqual);
}
export function greaterEqual(x: TracerValue, y: TracerValue) {
  return compare(y, x, CompareOp.LessEqual);
}
export function lessEqual(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.LessEqual);
}

export function where(cond: TracerValue, x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Where, [cond, x, y]);
}

export function concatenate(xs: TracerValue[], axis: number): Tracer {
  if (xs.length === 0)
    throw new Error("concatenate requires at least one input");
  const avals = xs.map((x) => ShapedArray.fromAval(getAval(x)));
  axis = checkAxis(axis, avals[0].ndim);
  for (const x of avals) {
    if (
      x.ndim !== avals[0].ndim ||
      !x.shape.every((s, i) => i === axis || s === avals[0].shape[i])
    )
      throw new Error(
        `Concatenate: inputs ${avals[0]} and ${x} must match shapes except on axis ${axis}`,
      );
  }
  // The eager Concatenate impl creates a single kernel with all inputs.
  // On WebGPU, this exceeds maxStorageBuffersPerShaderStage when xs.length > ~7.
  // Tree-reduce in groups of 7 to keep each kernel within buffer limits.
  // (In JIT, splitGraphDataflow handles splitting, so the extra equations
  //  from tree-reduction are harmless — they'll be fused or kept as-is.)
  const MAX_CONCAT_CHUNK = 7;
  if (xs.length > MAX_CONCAT_CHUNK) {
    const chunks: TracerValue[] = [];
    for (let i = 0; i < xs.length; i += MAX_CONCAT_CHUNK) {
      const group = xs.slice(i, Math.min(i + MAX_CONCAT_CHUNK, xs.length));
      chunks.push(
        group.length === 1
          ? group[0]
          : bind1(Primitive.Concatenate, group, { axis }),
      );
    }
    const result = concatenate(chunks, axis);
    // In eager mode, dispose intermediate chunk arrays (but not original inputs
    // or the result). During tracing, intermediates are graph nodes — don't touch.
    if (!insideAbstractTrace()) {
      const xsSet = new Set(xs);
      for (const c of chunks) {
        if (
          c !== result &&
          !xsSet.has(c) &&
          typeof (c as any).dispose === "function"
        ) {
          (c as any).dispose();
        }
      }
    }
    return result;
  }
  return bind1(Primitive.Concatenate, xs, { axis });
}

export function split(x: TracerValue, axis: number, sizes: number[]) {
  axis = checkAxis(axis, ndim(x));
  if (sizes.some((s) => s < 0 || !Number.isInteger(s))) {
    throw new Error(
      `split: sizes must be nonnegative integers, got ${JSON.stringify(sizes)}`,
    );
  }
  const totalSize = sizes.reduce((a, b) => a + b, 0);
  if (totalSize !== getShape(x)[axis]) {
    throw new Error(
      `split: sizes must sum to the size of the axis ${axis}, got ${totalSize}`,
    );
  }
  return bind(Primitive.Split, [x], { axis, sizes });
}

export function randomBits(
  k0: Tracer,
  k1: Tracer,
  shape: number[],
  mode: "xor" | 0 | 1 = "xor",
) {
  if (
    !deepEqual(k0.shape, k1.shape) ||
    k0.dtype !== DType.Uint32 ||
    k1.dtype !== DType.Uint32
  )
    throw new Error(
      `randomBits: key parts must be uint32 with the same shape, got` +
        ` ${ShapedArray.fromAval(k0.aval)} and ${ShapedArray.fromAval(k1.aval)}`,
    );
  return bind1(Primitive.RandomBits, [k0, k1], { shape, mode });
}

export function gather(
  x: TracerValue,
  indices: TracerValue[],
  axis: number[], // one for each index
  outDim: number,
) {
  // Evaluate advanced indexing x[:, ...indices, :], with the index dimensions
  // starting at axis `outDim` in the output.
  if (indices.length === 0) {
    throw new Error("gather() requires at least one index");
  }
  if (!Array.isArray(axis) || axis.length !== indices.length) {
    throw new Error(
      `Invalid gather() axis: expected ${indices.length} axes, got ${JSON.stringify(axis)}`,
    );
  }
  axis = axis.map((a) => checkAxis(a, ndim(x)));
  if (new Set(axis).size !== axis.length) {
    throw new Error(
      `Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`,
    );
  }
  outDim = checkAxis(outDim, ndim(x) - axis.length + 1);
  return bind1(Primitive.Gather, [x, ...indices], { axis, outDim });
}

export function transpose(x: TracerValue, perm?: number[]) {
  perm = perm
    ? perm.map((a) => checkAxis(a, ndim(x)))
    : range(ndim(x)).reverse();
  if (!isPermutation(perm, ndim(x))) {
    throw new Error(
      `Invalid transpose permutation for ${ndim(x)} axes: ${JSON.stringify(perm)}`,
    );
  }
  return bind1(Primitive.Transpose, [x], { perm });
}

export function broadcast(x: TracerValue, shape: Dim[], axis: number[]) {
  axis = normalizeAxis(axis, shape.length);
  return bind1(Primitive.Broadcast, [x], { shape, axis });
}

export function reshape(x: TracerValue, shape: number | number[]) {
  if (typeof shape === "number") shape = [shape];
  const originalShape = getShape(x);
  const autoIdx = shape.indexOf(-1);
  if (autoIdx !== -1) {
    const remaining = prod(originalShape) / -prod(shape);
    if (!Number.isInteger(remaining) || remaining < 0) {
      throw new Error(
        `Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`,
      );
    }
    shape = shape.toSpliced(autoIdx, 1, remaining);
  }
  if (prod(originalShape) !== prod(shape)) {
    throw new Error(
      `Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`,
    );
  }
  return bind1(Primitive.Reshape, [x], { shape });
}

export function flip(x: TracerValue, axis: number[]) {
  axis = normalizeAxis(axis, ndim(x));
  return bind1(Primitive.Flip, [x], { axis });
}

export function reverse(x: TracerValue, axis: number) {
  axis = normalizeAxis([axis], ndim(x))[0];
  return bind1(Primitive.Reverse, [x], { axis });
}

export function shrink(x: TracerValue, slice: Pair[]) {
  const shape = getShape(x);
  if (!Array.isArray(slice) || !slice.every(isNumberPair)) {
    throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
  }
  if (slice.length !== shape.length) {
    throw new Error(
      `Invalid shrink(): expected ${shape.length} axes, got ${slice.length}`,
    );
  }
  for (let i = 0; i < shape.length; i++) {
    const [start, end] = slice[i];
    if (start > end || start < 0 || end > shape[i]) {
      throw new Error(
        `Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape[i]}`,
      );
    }
  }
  return bind1(Primitive.Shrink, [x], { slice });
}

export function pad(
  x: TracerValue,
  width: number | Pair | Pair[] | Record<number, Pair>,
) {
  const nd = ndim(x);
  let w: Pair[];
  if (typeof width === "number") {
    w = [[width, width]];
  } else if (isNumberPair(width)) {
    w = [width as Pair];
  } else if (!Array.isArray(width)) {
    const indicesAndPairs = Object.entries(width);
    w = rep<Pair>(nd, [0, 0]);
    for (const [k, v] of indicesAndPairs) {
      w[checkAxis(parseInt(k), nd)] = v;
    }
  } else if (!width.every(isNumberPair)) {
    throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
  } else {
    w = width;
  }
  if (w.length === 1) {
    const [w0, w1] = w[0]; // A single pair should be repeated for all axes.
    w = rep(nd, () => [w0, w1] as Pair);
  } else if (w.length !== nd) {
    throw new Error(`Invalid pad(): expected ${nd} axes, got ${w.length}`);
  }
  return bind1(Primitive.Pad, [x], { width: w });
}

export function dynamicUpdateSlice(
  dst: TracerValue,
  src: TracerValue,
  offset: number,
  axis: number = 0,
) {
  offset = Math.floor(offset);
  if (!Number.isInteger(offset) || offset < 0) {
    throw new Error(
      `dynamicUpdateSlice: offset must be a nonnegative integer, got ${offset}`,
    );
  }
  return bind1(Primitive.DynamicUpdateSlice, [dst, src], { offset, axis });
}

/**
 * ND dynamic update slice: replace dst[starts[0]:..., starts[1]:..., ...] with src.
 * src.shape must satisfy src.shape[i] + startIndices[i] <= dst.shape[i] for all i,
 * and src.ndim must equal dst.ndim.
 */
export function dynamicUpdateSliceGeneral(
  dst: TracerValue,
  src: TracerValue,
  startIndices: number[],
) {
  const dstNdim = ndim(dst);
  const srcNdim = ndim(src);
  if (srcNdim !== dstNdim) {
    throw new Error(
      `dynamicUpdateSliceGeneral: src and dst must have same rank, got src=${srcNdim} dst=${dstNdim}`,
    );
  }
  if (startIndices.length !== dstNdim) {
    throw new Error(
      `dynamicUpdateSliceGeneral: expected ${dstNdim} start indices, got ${startIndices.length}`,
    );
  }
  // Clone before coercion to avoid mutating the caller's array
  const indices = startIndices.slice();
  for (let i = 0; i < indices.length; i++) {
    const s = Math.floor(indices[i]);
    if (!Number.isInteger(s) || s < 0) {
      throw new Error(
        `dynamicUpdateSliceGeneral: start index ${i} must be a nonnegative integer, got ${startIndices[i]}`,
      );
    }
    indices[i] = s;
  }
  return bind1(Primitive.DynamicUpdateSliceGeneral, [dst, src], {
    startIndices: indices,
  });
}

/**
 * Scatter-add: `target[indices[i]] += updates[i]` along the given axis.
 *
 * Returns a new array equal to target but with updates[j] added at
 * position indices[j] along `axis`. Multiple indices mapping to the
 * same position are summed (order-independent). The original target is
 * NOT consumed.
 */
export function scatterAdd(
  target: TracerValue,
  indices: TracerValue,
  updates: TracerValue,
  axis: number,
) {
  axis = checkAxis(axis, ndim(target));
  return bind1(Primitive.ScatterAdd, [target, indices, updates], { axis });
}

/**
 * Internal triangular solve. Dispatches `Primitive.TriangularSolve` which
 * always operates on an **upper-triangular** A using the convention
 * `A @ X^T = B^T` (returns X = B @ A^{-T}).
 *
 * When `lower` is true, A is flipped to upper form before dispatching.
 *
 * This is NOT the public API — see `lax.linalg.triangularSolve` for the
 * user-facing function with `leftSide`/`transposeA` parameters.
 */
export function triSolve(
  a: TracerValue,
  b: TracerValue,
  {
    lower = false,
    unitDiagonal = false,
  }: { lower?: boolean; unitDiagonal?: boolean } = {},
) {
  const as = getShape(a);
  const bs = getShape(b);
  if (as.length < 2 || bs.length < 2)
    throw new Error(`triangular_solve: must be >=2D, got a=${as}, b=${bs}`);
  const n = as[as.length - 2];
  if (n !== as[as.length - 1] || n !== bs[bs.length - 1])
    throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
  if (!lower) {
    return bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
  }
  // Lower-triangular: convert to upper by flipping, solve, flip back.
  // Dispose intermediates (no-op during tracing, frees buffers in eager).
  using aFlipped = flip(a, [-2, -1]) as Tracer;
  using bFlipped = flip(b, [-1]) as Tracer;
  using x = bind1(Primitive.TriangularSolve, [aFlipped, bFlipped], {
    unitDiagonal,
  }) as Tracer;
  return flip(x, [-1]);
}

export function cholesky(x: TracerValue) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2])
    throw new Error(`cholesky: expected batch of square matrices, got ${aval}`);
  return bind1(Primitive.Cholesky, [x]);
}

export function lu(x: TracerValue) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2)
    throw new Error(`lu: expected batch of matrices, got ${aval}`);
  return bind(Primitive.LU, [x]);
}

export function qr(x: TracerValue) {
  const aval = ShapedArray.fromAval(getAval(x));
  if (aval.ndim < 2)
    throw new Error(`qr: expected batch of matrices, got ${aval}`);
  return bind(Primitive.QR, [x]);
}

export function sort(x: TracerValue) {
  const nd = ndim(x);
  if (nd === 0) throw new Error("sort: requires at least 1D input");
  return bind1(Primitive.Sort, [x]);
}

export function argsort(x: TracerValue) {
  const nd = ndim(x);
  if (nd === 0) throw new Error("argsort: requires at least 1D input");
  return bind(Primitive.Argsort, [x]);
}

export function bind1<P extends Primitive>(
  prim: P,
  args: TracerValue[],
  params: PrimitiveParams<P> = {} as any,
) {
  const [results] = bind(prim, args, params);
  return results;
}

type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace; // Concrete Trace subclass.
  globalData: any | null;
  /**
   * True for graph-building traces where values are abstract or shared with
   * the outer trace (JaxprTrace, PartialEvalTrace, JVPTrace). These traces
   * hold values that must not be disposed by inner cleanup code.
   *
   * False/unset for concrete-value traces (EvalTrace, BatchTrace) where
   * operations produce fresh arrays that can be safely disposed.
   */
  isAbstract?: boolean;
};

const traceStack: MainTrace[] = []; // Global trace stack, mutable
let dynamicTrace: MainTrace | null = null;

/**
 * Push an interpreter onto the trace stack. Use this like:
 * `using main = newMain(...);`
 */
export function newMain(
  traceType: any,
  globalData: any | null = null,
): Disposable & MainTrace {
  const level = traceStack.length;
  const main = { level, traceType, globalData };
  traceStack.push(main);
  return Object.assign(main, {
    [Symbol.dispose]() {
      traceStack.pop();
    },
  });
}

/**
 * Set the current dynamic trace, which stashes the current interpreter stack
 * and acts temporarily as the bottom of the stack. Use this like:
 * `using _dynamic = newDynamic(main);`
 */
export function newDynamic(main: MainTrace): Disposable {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main;
  return {
    [Symbol.dispose]() {
      dynamicTrace = prevDynamicTrace;
    },
  };
}

export function currentTraceLevel(): number {
  return traceStack[traceStack.length - 1].level;
}

/**
 * Returns true if code is currently executing inside a trace context (e.g.
 * makeJaxpr, jit, vmap). Disposal of ClosedJaxpr constants is unsafe inside
 * traces because the consts may be JaxprTracers or BatchTracers from an outer
 * trace that must remain alive.
 *
 * Note: traceStack always has at least 1 entry (the base EvalTrace at level 0),
 * so we check for length > 1 to detect non-eval traces.
 */
export function insideTrace(): boolean {
  return traceStack.length > 1;
}

/**
 * Returns true if any abstract (graph-building) trace is on the stack.
 *
 * Unlike `insideTrace()`, this returns false when only concrete-value traces
 * like BatchTrace are active. Cleanup of intermediate arrays is safe when
 * no abstract traces are present — BatchTrace wraps concrete arrays that are
 * independently owned, so disposing them doesn't affect outer trace state.
 *
 * Use this instead of `insideTrace()` at cleanup sites where the values being
 * disposed are concrete arrays (not abstract tracers from a graph-building trace).
 */
export function insideAbstractTrace(): boolean {
  for (let i = 1; i < traceStack.length; i++) {
    if (traceStack[i].isAbstract) return true;
  }
  return false;
}

/**
 * True while user body code is executing inside a makeJaxpr call.
 * Set around the `fn(tracers)` call, NOT during JaxprBuilder.build().
 */
let _inMakeJaxprBody = false;
export function inMakeJaxprBody(): boolean {
  return _inMakeJaxprBody;
}
export function _setInMakeJaxprBody(v: boolean): void {
  _inMakeJaxprBody = v;
}

/**
 * True while user code is executing inside the forward partial-eval scope.
 * Arrays born here are PE-owned until they are either lifted into the PE trace
 * or disposed lexically before escaping the scope.
 */
let _inPartialEvalScope = false;
export function inPartialEvalScope(): boolean {
  return _inPartialEvalScope;
}

let _partialEvalCreatedArrays: Tracer[] | null = null;
export function registerPartialEvalCreatedArray(value: Tracer): void {
  _partialEvalCreatedArrays?.push(value);
}
export function withPartialEvalScope<T>(
  createdArrays: Tracer[],
  fn: () => T,
): T {
  const previousPeScope = _inPartialEvalScope;
  const previousPeCreatedArrays = _partialEvalCreatedArrays;
  _inPartialEvalScope = true;
  _partialEvalCreatedArrays = createdArrays;
  try {
    return fn();
  } finally {
    _inPartialEvalScope = previousPeScope;
    _partialEvalCreatedArrays = previousPeCreatedArrays;
  }
}

/**
 * Returns true if any abstract trace exists below the given level on the stack.
 *
 * Used by jvpFlat to decide whether its own intermediate cleanup is safe.
 * When only BatchTraces are below, cleanup is safe. When PE or JaxprTrace
 * is below, the intermediates are managed by that outer trace.
 */
export function hasAbstractTraceBelow(level: number): boolean {
  for (let i = 1; i < level && i < traceStack.length; i++) {
    if (traceStack[i].isAbstract) return true;
  }
  return false;
}

export type TracerValue = Tracer | number | boolean;

export abstract class Trace {
  constructor(readonly main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: Tracer[],
    params: PrimitiveParams<P>,
  ): Tracer[];
}

/** Internal representation of an array value. */
export interface AbstractValue {
  /**
   * Shape of the array. Dimensions are concrete numbers in most contexts.
   * During polymorphic tracing (jit with dynamic_axes), some dimensions may
   * be SymDim instances representing unknown sizes.
   */
  shape: Dim[];

  /** Concrete data type of array elements. */
  dtype: DType;

  /**
   * Arrays created from JavaScript numbers (e.g., `np.array(3)`) are created as
   * _weakly typed_ unless a dtype is explicitly specified.
   *
   * Weakly typed values will automatically cast to the data type of other
   * arrays when used as an operand as an expression. This property only affects
   * how they promote in type casting; their memory layout is still determined
   * by the actual `dtype` field.
   *
   * ```ts
   * using x = np.array(3); // weakType = true, dtype = float32
   * using y = np.array([1, 2], { dtype: np.int32 }); // weakType = false, dtype = int32
   * using z = x.add(y); // z has dtype int32 because x is weakly typed
   * ```
   *
   * Weak types are present in JIT programs in their spec (e.g., Jaxpr inputs
   * and outputs can be weakly typed) form. But they're solely a frontend
   * concept. Backends are not aware of weak types.
   */
  weakType: boolean;
}

/**
 * Broadcast shapes and promote types with casting for two avals.
 *
 * This implements the weak type behavior described in `promoteTypes()`, but not
 * implemented in that function as `weakType` is not passed.
 */
export function promoteAvals(a: AbstractValue, b: AbstractValue): ShapedArray {
  const shape = generalBroadcast(a.shape, b.shape);
  const weakType = a.weakType && b.weakType;
  let dtype: DType;
  if (a.weakType === b.weakType) {
    // Both weak or both strong: use normal promotion rules.
    dtype = promoteTypes(a.dtype, b.dtype);
  } else if (a.weakType) {
    // a is weak, b is strong: use b's dtype.
    dtype = promoteTypes(b.dtype, DType.Uint32);
  } else {
    // b is weak, a is strong: use a's dtype.
    dtype = promoteTypes(a.dtype, DType.Uint32);
  }
  return new ShapedArray(shape, dtype, weakType);
}

export abstract class Tracer {
  /** @ignore */
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;
  abstract toString(): string;

  /**
   * Access an array by reference, incrementing the reference count.
   *
   * Increment the reference count of this array.
   *
   * In most code, you don't need `.ref` because operations do NOT consume
   * their inputs. The main use case is when you need an array to survive
   * an explicit `.dispose()` call or `using` block.
   *
   * ```
   * using x = np.array([1, 2, 3]);
   * using y = np.add(x, x); // works — x is not consumed
   * ```
   */
  abstract get ref(): this;

  /**
   * Manually decrement the reference count of the array.
   *
   * Arrays are created with reference count 1. When the reference count
   * reaches 0, the underlying memory is freed.
   *
   * You can use the `using` keyword for automatic disposal:
   *
   * ```
   * {
   *   using x = np.array([1, 2, 3]);
   *   // x is automatically disposed at the end of this block
   * }
   * ```
   */
  abstract dispose(): void;

  /**
   * Enables `using` declarations for automatic disposal at block exit.
   *
   * Default is a no-op on the abstract Tracer base. Only concrete `Array`
   * overrides this to call `dispose()`. Calling `dispose()` here would be
   * unsafe for JVPTracer / BatchTracer / PartialEvalTracer because it would
   * cascade prematurely during `grad()` / `jvp()` / `vmap()` tracing.
   */
  [Symbol.dispose]() {}

  /** Current reference count (only meaningful on concrete Arrays). */
  get refCount(): number {
    return 0;
  }

  /**
   * Whether this tracer still owns a live cleanup obligation.
   *
   * Cleanup code should use this instead of probing ad hoc fields like
   * `refCount` or `isAlive` directly.
   */
  isAliveForCleanup(): boolean {
    return true;
  }

  /** The shape of the array. */
  get shape(): number[] {
    return this.aval.shape as number[];
  }
  /** The total number of elements in the array. */
  get size(): number {
    return prod(this.shape);
  }
  /** The dtype of elements stored in the array. */
  get dtype(): DType {
    return this.aval.dtype;
  }
  /**
   * Whether the array is weakly typed.
   *
   * Weakly typed arrays will cast to the dtype of the other operand. See
   * `promoteTypes()` for details.
   */
  get weakType(): boolean {
    return this.aval.weakType;
  }

  /** The number of dimensions of the array. */
  get ndim(): number {
    return this.shape.length;
  }

  /** @ignore */
  fullLower(): Tracer {
    return this; // default implementation
  }

  // These types aren't technically correct since they don't account for the
  // fact that tracers can be lifted to different levels. But they simplify the
  // API visible to users.

  neg() {
    return neg(this) as this;
  }
  add(other: this | TracerValue) {
    return add(this, other) as this;
  }
  mul(other: this | TracerValue) {
    return mul(this, other) as this;
  }
  mod(other: this | TracerValue) {
    // Note: Unlike `jax.numpy.remainder()`, this has JS rounding behavior where
    // the result matches the sign of the numerator. `1 % -2 === 1`.
    return mod(this, other) as this;
  }
  greater(other: this | TracerValue) {
    return greater(this, other) as this;
  }
  less(other: this | TracerValue) {
    return less(this, other) as this;
  }
  equal(other: this | TracerValue) {
    return equal(this, other) as this;
  }
  notEqual(other: this | TracerValue) {
    return notEqual(this, other) as this;
  }
  greaterEqual(other: this | TracerValue) {
    return greaterEqual(this, other) as this;
  }
  lessEqual(other: this | TracerValue) {
    return lessEqual(this, other) as this;
  }

  /** Sum of the elements of the array over a given axis, or axes. */
  sum(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Add, axis, opts) as this;
  }

  /** Product of the array elements over a given axis. */
  prod(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Mul, axis, opts) as this;
  }

  /** Compute the average of the array elements along the specified axis. */
  mean(axis: Axis = null, opts?: ReduceOpts) {
    axis = normalizeAxis(axis, this.ndim);
    const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
    if (n === 0) {
      throw new Error("mean: cannot compute mean over zero-length axis");
    }
    const originalDtype = this.dtype;
    const castDtype = promoteTypes(originalDtype, DType.Float32);
    const casted = this.astype(castDtype);
    let summed: Tracer | null = null;
    let scaled: Tracer | null = null;
    let final_: Tracer | null = null;
    try {
      summed = reduce(casted, AluOp.Add, axis, opts);
      scaled = summed.mul(1 / n);
      final_ = scaled.astype(originalDtype);
      return final_ as this;
    } finally {
      if (scaled && scaled !== final_) scaled.dispose();
      if (summed && summed !== scaled && summed !== final_) summed.dispose();
      if (
        casted !== (this as Tracer) &&
        casted !== summed &&
        casted !== scaled &&
        casted !== final_
      ) {
        casted.dispose();
      }
    }
  }

  /** Minimum of the elements of the array along a given axis. */
  min(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Min, axis, opts) as this;
  }

  /** Maximum of the elements of the array along a given axis. */
  max(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Max, axis, opts) as this;
  }

  /** Test whether all array elements along a given axis evaluate to true. */
  all(axis: Axis = null, opts?: ReduceOpts) {
    const boolArr = this.astype(DType.Bool);
    try {
      return boolArr.min(axis, opts);
    } finally {
      if (boolArr !== (this as Tracer)) boolArr.dispose();
    }
  }

  /** Test whether any array element along a given axis evaluates to true. */
  any(axis: Axis = null, opts?: ReduceOpts) {
    const boolArr = this.astype(DType.Bool);
    try {
      return boolArr.max(axis, opts);
    } finally {
      if (boolArr !== (this as Tracer)) boolArr.dispose();
    }
  }

  /** Permute the dimensions of an array. Defaults to reversing the axis order. */
  transpose(perm?: number[]): this {
    return transpose(this, perm) as this;
  }

  /**
   * Give a new shape to an array without changing its data.
   *
   * One shape dimension can be -1. In this case, the value is inferred from the
   * length of the array and remaining dimensions.
   */
  reshape(shape: number | number[]): this {
    return reshape(this, shape) as this;
  }

  /** Copy the array and cast to a specified dtype. */
  astype(dtype: DType): this {
    if (this.dtype === dtype) return this; // No-op.
    return cast(this, dtype) as this;
  }

  /** Return a bitwise cast of the array, viewed as a new dtype. */
  view(dtype?: DType): this {
    if (!dtype || dtype === this.dtype) return this; // No-op.
    return bitcast(this, dtype) as this;
  }

  // Below this line are composite operations built from primitives.

  /** Subtract an array from this one. */
  sub(other: this | TracerValue): this {
    using n = neg(other);
    return this.add(n);
  }

  /** Divide an array by this one. */
  div(other: this | TracerValue): this {
    if (isFloatDtype(this.dtype)) {
      using r = reciprocal(other);
      return this.mul(r);
    }
    return idiv(this, other) as this;
  }

  /** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
  diagonal(offset = 0, axis1 = 0, axis2 = 1): this {
    if (!Number.isInteger(offset))
      throw new TypeError(`offset must be an integer, got ${offset}`);
    if (offset < 0) return this.diagonal(-offset, axis2, axis1);
    axis1 = checkAxis(axis1, this.ndim);
    axis2 = checkAxis(axis2, this.ndim);
    if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
    if (offset >= this.shape[axis2])
      throw new Error("offset exceeds axis size");

    // First, make sure that the last two axes are being taken.
    //
    // We can just move them to the end, since the behavior of diagonal() is to
    // append the new diagonal axis to the right side / end of the shape.
    const toDispose: Tracer[] = [];
    let ar: Tracer = this;
    let result: Tracer | null = null;
    try {
      if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
        const perm = range(ar.ndim)
          .filter((i) => i !== axis1 && i !== axis2)
          .concat(axis1, axis2);
        ar = ar.transpose(perm);
        toDispose.push(ar);
      }

      const [n, m] = ar.shape.slice(-2);
      const diagSize = Math.min(n, m - offset);

      // Pad and reshape ar into a skewed array of shape [..., diagSize, m+1].
      ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
      toDispose.push(ar);
      const npad = diagSize * (m + 1) - n * m;
      if (npad > 0) {
        ar = pad(ar, [...rep<Pair>(ar.ndim - 1, [0, 0]), [0, npad]]);
        toDispose.push(ar);
      } else if (npad < 0) {
        ar = shrink(
          ar,
          [...ar.shape.slice(0, -1), n * m + npad].map<Pair>((x) => [0, x]),
        );
        toDispose.push(ar);
      }
      ar = ar.reshape([...ar.shape.slice(0, -1), diagSize, m + 1]);
      toDispose.push(ar);

      // Now slice the #offset element of the last axis, and this gives a diagonal.
      const sliced = shrink(ar, [
        ...ar.shape.slice(0, -1).map<Pair>((x) => [0, x]),
        [offset, offset + 1],
      ]);
      toDispose.push(sliced);
      result = sliced.reshape(ar.shape.slice(0, -1));
      return result as this;
    } finally {
      for (const v of toDispose) if (v !== result) v[Symbol.dispose]();
    }
  }

  /** Flatten the array without changing its data. */
  flatten(): this {
    return this.reshape(-1);
  }
  /** Flatten the array without changing its data. */
  ravel(): this {
    return this.reshape(-1);
  }

  /**
   * Iterate over the first dimension of this array, returning slices.
   *
   * This can be used to destructure arrays. For example:
   *
   * ```js
   * using x = np.array([[1, 2], [3, 4]]);
   * const [a, b] = x;
   * console.log(a.js()); // [1, 2]
   * console.log(b.js()); // [3, 4]
   * a.dispose();
   * b.dispose();
   * ```
   */
  *[Symbol.iterator](): IterableIterator<this> {
    if (this.ndim === 0) throw new Error("Cannot iterate over a scalar array");
    let residual: Tracer = this;
    const n = this.shape[0];
    const subarrayShape = this.shape.slice(1);
    for (let i = 0; i < n; i++) {
      if (i < n - 1) {
        const lr = split(residual, 0, [1, residual.shape[0] - 1]);
        // Dispose previous residual (but not `this`)
        if (residual !== (this as Tracer)) residual.dispose();
        const first = lr[0];
        const reshaped = first.reshape(subarrayShape);
        if (first !== reshaped) first.dispose();
        yield reshaped as this;
        residual = lr[1];
      } else {
        // Last element: reshape directly, no split needed.
        // Avoids creating an empty [0, ...] residual that leaks if the
        // generator is abandoned (e.g., by array destructuring).
        const reshaped = residual.reshape(subarrayShape);
        if (residual !== reshaped && residual !== (this as Tracer))
          residual.dispose();
        yield reshaped as this;
      }
    }
  }

  /**
   * Return a sorted copy of an array in ascending order.
   *
   * See `jax.numpy.sort` for full docs.
   */
  sort(axis: number = -1): this {
    axis = checkAxis(axis, this.ndim);
    if (this.shape[axis] <= 1) return this.reshape(this.shape) as this;
    if (axis === this.ndim - 1) return sort(this) as this;
    const perm = range(this.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    const transposed = this.transpose(perm);
    const sorted = sort(transposed);
    const result = sorted.transpose(invertPermutation(perm));
    if (transposed !== (this as Tracer)) transposed.dispose();
    if (sorted !== result) sorted.dispose();
    return result as this;
  }

  /**
   * Return the indices that would sort an array. Unlike `sort`, this is
   * guaranteed to be a stable sorting algorithm; it always returns the smaller
   * index first in event of ties.
   *
   * See `jax.numpy.argsort` for full docs.
   */
  argsort(axis: number = -1): this {
    axis = checkAxis(axis, this.ndim);
    if (axis === this.ndim - 1) {
      const [y, yi] = argsort(this);
      y.dispose();
      return yi as this;
    }
    const perm = range(this.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    const transposed = this.transpose(perm);
    const [y, yi] = argsort(transposed);
    y.dispose();
    if (transposed !== (this as Tracer)) transposed.dispose();
    const result = yi.transpose(invertPermutation(perm));
    if (yi !== result) yi.dispose();
    return result as this;
  }

  /**
   * Slice an array along one or more axes.
   *
   * This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
   * mimic this in JavaScript, we would write:
   *
   * ```js
   * x.slice([1, 3], 2, [], null);
   * ```
   *
   * The `slice` method accepts a variable number of arguments, each of which
   * can be a number, an empty array, a single-element array, a two-element
   * array, or `null`. The arguments are interpreted as follows:
   *
   * - A number `n` means to access the `n`-th element along that axis, removing
   *   that axis from the resulting shape.
   * - An empty array `[]` means to keep that axis as-is, like `:` in Python.
   * - A single-element array `[i]` means to start slicing from index `i`
   *   (inclusive) to the end of the axis, like `x[i:]`.
   * - A two-element array `[i, j]` means to slice from index `i` (inclusive)
   *   to index `j` (exclusive), like `x[i:j]`.
   * - `null` means to add a new axis at that position, like `np.newaxis`.
   *
   * Like in Python, negative indices are supported, which count from the end of
   * the axis. For example, `-1` means the last element.
   *
   * Strided slices are not yet implemented, so you cannot write `x[::2]` or
   * similar.
   *
   * Advanced indexing by integer arrays is also supported. This translates to
   * the "gather" primitive, and it allows you to access specific elements of
   * the array by integer indices stored in another array.
   */
  slice(...index: (number | [] | [number] | Pair | null | Tracer)[]): this {
    const checkBounds = (n: number, i: number): number => {
      if (i > n || i < -n)
        throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
      return i < 0 ? n + i : i;
    };

    const hasAdvancedIdx = index.some((value) => value instanceof Tracer);
    const axesForGather: number[] = [];
    let outDim = -1; // Will be set if we have advanced indexing.
    if (hasAdvancedIdx) {
      // Find out which axes have scalar indices or advanced indices used. If
      // those axes are contiguous, we'll output advanced indexes there,
      // otherwise they get transposed to the beginning of the shape.
      //
      // Example:
      //
      // In [3]: np.zeros((10,)*4).shape
      // Out[3]: (10, 10, 10, 10)
      //
      // In [4]: np.zeros((10,)*4)[:, [1,2], [1,2], :].shape
      // Out[4]: (10, 2, 10)
      //
      // In [5]: np.zeros((10,)*4)[:, [1,2], :, [1,2]].shape
      // Out[5]: (2, 10, 10)
      //
      // In [6]: np.zeros((10,)*4)[:, [1,2], 5, [1,2]].shape
      // Out[6]: (10, 2)
      //
      // In [7]: np.zeros((10,)*4)[:, [1,2], :, 5].shape
      // Out[7]: (2, 10, 10)
      //
      // For more, see https://stackoverflow.com/questions/55829631
      const advancedAxes: number[] = [];
      let currentAxisForGather = 0;
      for (let i = 0; i < index.length; i++) {
        const value = index[i];
        if (value instanceof Tracer) {
          advancedAxes.push(i);
          axesForGather.push(currentAxisForGather++);
        } else if (typeof value === "number") {
          advancedAxes.push(i);
          // Do not increment currentAxisForGather, this is squeezed out.
        } else {
          currentAxisForGather++;
        }
      }
      if (
        advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !==
        advancedAxes.length - 1
      ) {
        // If the advanced axes are not contiguous, we need to transpose them
        // to the beginning of the shape.
        outDim = 0;
      } else {
        outDim = axesForGather[0]; // OK, keep them in the same place.
      }
    }

    // 1. Calculate shape / operations for "basic" slicing.
    // 2. Squeeze out, i.e., reshape (1,) axes from scalar indices.
    // 3. Do gather if needed (hasAdvancedIdx).
    const slice: Pair[] = [];
    const basicShape: number[] = [];
    let needsReshape = false;
    let axis = 0;
    for (const value of index) {
      if (value === null) {
        // Add a new axis at this position.
        basicShape.push(1);
        needsReshape = true;
      } else if (typeof value === "number") {
        // Access the i-th element along this axis.
        if (axis >= this.shape.length) throw new RangeError("Too many indices");
        const i = checkBounds(this.shape[axis++], value);
        slice.push([i, i + 1]);
        needsReshape = true;
      } else if (Array.isArray(value)) {
        if (axis >= this.shape.length) throw new RangeError("Too many indices");
        const n = this.shape[axis++];
        if (value.length === 0) {
          // Keep this axis as-is, like `:`.
          basicShape.push(n);
          slice.push([0, n]);
        } else if (value.length === 1) {
          // Slice from index i to the end, like `x[i:]`.
          const i = checkBounds(n, value[0]);
          basicShape.push(n - i);
          slice.push([i, n]);
        } else if (value.length === 2) {
          // Slice from index i to index j, like `x[i:j]`.
          const [i, j] = value.map((v) => checkBounds(n, v));
          if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
          basicShape.push(j - i);
          slice.push([i, j]);
        }
      } else if (value instanceof Tracer) {
        // Keep the full axis as-is like `:`, until the gather step.
        const n = this.shape[axis++];
        basicShape.push(n);
        slice.push([0, n]);
      } else {
        throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
      }
    }
    while (axis < this.shape.length) {
      // If we didn't specify an index for this axis, keep it as-is.
      slice.push([0, this.shape[axis]]);
      basicShape.push(this.shape[axis++]);
    }
    let result = shrink(this, slice);
    if (needsReshape) {
      const prev = result;
      result = reshape(result, basicShape);
      if (prev !== result) prev[Symbol.dispose]?.();
    }

    if (hasAdvancedIdx) {
      const prev = result;
      result = gather(
        result,
        index.filter((a) => a instanceof Tracer),
        axesForGather,
        outDim,
      );
      if (prev !== result) prev[Symbol.dispose]?.();
    }

    return result as this;
  }
}

export function ndim(x: TracerValue) {
  if (x instanceof Tracer) {
    return x.shape.length;
  } else {
    return 0;
  }
}

export function getShape(x: TracerValue): number[] {
  return x instanceof Tracer ? x.shape : [];
}

// Note: Autodidax has a `ConcreteArray` type with "arrayAbstractionLevel" set
// to a higher value. I didn't see how this would be useful yet, so currently
// the only `AbstractValue` is a `ShapedArray` instance.
export class ShapedArray implements AbstractValue {
  constructor(
    readonly shape: Dim[],
    readonly dtype: DType,
    readonly weakType: boolean,
  ) {}

  static fromAval(aval: AbstractValue) {
    return new ShapedArray(aval.shape, aval.dtype, aval.weakType);
  }

  get ndim() {
    return this.shape.length;
  }

  get size() {
    // Size is only defined for fully concrete shapes.
    let p = 1;
    for (const d of this.shape) {
      if (isSymbolicDim(d))
        throw new Error(
          `Cannot compute size of symbolic shape [${this.shape}]`,
        );
      p *= d;
    }
    return p;
  }

  /**
   * Compute element count as a SizeExpr. Supports symbolic dimensions.
   * Returns a concrete number when all dims are concrete, or a SymbolicSize
   * otherwise.
   */
  get sizeExpr(): SizeExpr {
    return dimProduct(this.shape);
  }

  scalar() {
    return new ShapedArray([], this.dtype, this.weakType);
  }

  toString() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }

  /** Check for abstract equality (shapes and dtype match). */
  equals(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.ndim === other.ndim &&
        this.shape.every((d, i) => dimEquals(d, other.shape[i])))
    );
  }

  /**
   * Check if this ShapedArray is compatible with `other`, allowing
   * concrete dims to match symbolic dims.
   */
  compatible(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.ndim === other.ndim &&
        this.shape.every((d, i) => dimCompatible(d, other.shape[i])))
    );
  }

  /** True when all dimensions are concrete numbers. */
  get isConcrete() {
    return !hasSymbolicDims(this.shape);
  }
}

export function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ShapedArray(
      [],
      typeof x === "boolean" ? DType.Bool : DType.Float32,
      typeof x === "boolean" ? false : true,
    );
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

export function bind<P extends Primitive>(
  prim: P,
  args: TracerValue[],
  params: PrimitiveParams<P> = {} as any,
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  if (DEBUG >= 5) {
    console.info(
      `processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`,
    );
  }
  const lowered = outs.map((out) => out.fullLower());
  // Dispose freshly-created pureArray intermediates from fullRaise.
  // Only disposes concrete Arrays (via [Symbol.dispose] which is a no-op on
  // abstract tracers). Safe because processPrimitive has already consumed them.
  for (let i = 0; i < args.length; i++) {
    if (!(args[i] instanceof Tracer)) {
      tracers[i][Symbol.dispose]();
    }
  }
  return lowered;
}

function findTopTrace(xs: TracerValue[]): Trace {
  let topMain: MainTrace = traceStack[0];
  for (const x of xs) {
    if (x instanceof Tracer && x._trace.main.level > topMain.level) {
      topMain = x._trace.main;
    }
  }
  if (dynamicTrace && dynamicTrace.level > topMain.level) {
    topMain = dynamicTrace;
  }
  return new topMain.traceType(topMain);
}

export function fullRaise(trace: Trace, val: TracerValue): Tracer {
  if (!(val instanceof Tracer)) {
    // remember to assert type(val) in jax_types
    return trace.pure(val);
  }
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main)) {
    return val;
  } else if (val._trace.main.level < level) {
    return trace.lift(val);
  } else if (val._trace.main.level > level) {
    throw new Error(
      `Can't lift Tracer level ${val._trace.main.level} to level ${level}`,
    );
  } else {
    throw new Error(
      `Different traces at same level: ${val._trace.constructor}, ${trace.constructor}.`,
    );
  }
}

export class TreeMismatchError extends TypeError {
  constructor(where: string, left: JsTreeDef, right: JsTreeDef) {
    super(`Mismatched tree structures in ${where}: ${left} != ${right}`);
  }
}

type Store<T> = { value: T | undefined };

const customJvpDefSymbol = Symbol("customJvpDef");
const customVjpDefSymbol = Symbol("customVjpDef");
const customJvpFlatInTreeSymbol = Symbol("customJvpFlatInTree");
const customJvpOutTreeStoreSymbol = Symbol("customJvpOutTreeStore");

export function getCustomJvpDef(f: any): any {
  return f?.[customJvpDefSymbol];
}

export function setCustomJvpDef(f: any, def: any): void {
  f[customJvpDefSymbol] = def;
}

export function getCustomVjpDef(f: any): any {
  return f?.[customVjpDefSymbol];
}

export function setCustomVjpDef(f: any, def: any): void {
  f[customVjpDefSymbol] = def;
}

export function getCustomJvpFlatInTree(f: any): JsTreeDef | undefined {
  return f?.[customJvpFlatInTreeSymbol];
}

export function getCustomJvpOutTreeStore(f: any): Store<JsTreeDef> | undefined {
  return f?.[customJvpOutTreeStoreSymbol];
}

export function propagateCustomDerivativeMetadata(
  source: any,
  target: any,
  options?: { flatInTree?: JsTreeDef; outTreeStore?: Store<JsTreeDef> },
): void {
  const customJvpDef = getCustomJvpDef(source);
  if (customJvpDef !== undefined) {
    target[customJvpDefSymbol] = customJvpDef;
    if (options?.flatInTree !== undefined) {
      target[customJvpFlatInTreeSymbol] = options.flatInTree;
    }
    if (options?.outTreeStore !== undefined) {
      target[customJvpOutTreeStoreSymbol] = options.outTreeStore;
    }
  }
  const customVjpDef = getCustomVjpDef(source);
  if (customVjpDef !== undefined) {
    target[customVjpDefSymbol] = customVjpDef;
  }
}

/** Flatten a function of `JsTree` input/output for use in tracing. */
export function flattenFun(f: any, inTree: JsTreeDef): [any, Store<JsTreeDef>] {
  const store: Store<JsTreeDef> = { value: undefined };
  const flatFun = (...argsFlat: any[]) => {
    const pytreeArgs = treeUnflatten(inTree, argsFlat);
    const out = f(...pytreeArgs);
    const [outFlat, outTree] = treeFlatten(out);
    store.value = outTree;
    return outFlat;
  };
  propagateCustomDerivativeMetadata(f, flatFun, {
    flatInTree: inTree,
    outTreeStore: store,
  });
  return [flatFun, store];
}

/** Like flattenFun, but expects f to return [main, aux] tuple. */
export function flattenFunWithAux(
  f: any,
  inTree: JsTreeDef,
): [any, Store<JsTreeDef>, Store<any>] {
  const store: Store<JsTreeDef> = { value: undefined };
  const auxStore: Store<any> = { value: undefined };
  const flatFun = (...argsFlat: any[]) => {
    const pytreeArgs = treeUnflatten(inTree, argsFlat);
    const result = f(...pytreeArgs);
    if (!Array.isArray(result) || result.length !== 2) {
      throw new Error(
        "Function with `hasAux: true` must return [output, aux] tuple",
      );
    }
    const [out, aux] = result;
    const [outFlat, outTree] = treeFlatten(out);
    store.value = outTree;
    auxStore.value = aux;
    return outFlat;
  };
  propagateCustomDerivativeMetadata(f, flatFun, {
    flatInTree: inTree,
    outTreeStore: store,
  });
  return [flatFun, store, auxStore];
}

export class UseAfterFreeError extends ReferenceError {
  constructor(tracer: Tracer) {
    super(`Referenced tracer ${tracer.toString()} has been disposed`);
  }
}

// Late-bound hook for the associativeScan Kogge-Stone core algorithm.
// Registered by `lax-associative-scan.ts` at module load to avoid circular imports
// between `array.ts` and the library module.
//
// Accepts the body jaxpr (traced with element shapes, scan axis removed) + consts,
// and uses vmap internally to vectorize the body over the batch dimension.
/** @internal */
export let _associativeScanCoreImpl:
  | ((
      bodyJaxpr: any,
      consts: any[],
      elems: any[],
      numLeaves: number,
      axis: number,
      reverse: boolean,
    ) => any[])
  | null = null;

/** @internal */
export function _registerAssociativeScanCoreImpl(
  impl: NonNullable<typeof _associativeScanCoreImpl>,
): void {
  _associativeScanCoreImpl = impl;
}
