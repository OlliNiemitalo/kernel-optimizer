import { DType } from "../alu";
import { defaultDevice } from "../backend";
import { assertNonNull, checkAxis, range, rep, unzip2, zip } from "../utils";
import { arange, eye, fullInternal, pureArray } from "./array";
import {
  AbstractValue,
  bind,
  bind1,
  broadcast,
  concatenate,
  conv,
  dot,
  flattenFun,
  flip,
  fullRaise,
  gather,
  hasAbstractTraceBelow,
  insideAbstractTrace,
  ndim,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  randomBits,
  reduce,
  reshape,
  reverse,
  scatterAdd,
  ShapedArray,
  shrink,
  split,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
} from "./core";
import {
  JsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import {
  _registerCacheSizeGetter,
  _registerJitCacheDisposer,
} from "./check-leaks";
import {
  buildCachedJaxprArtifact,
  CachedJaxprArtifact,
  Jaxpr,
  jaxprAsFun,
  makeJaxpr,
} from "./jaxpr";
import { jvp } from "./jvp";

function mappedAval(batchDim: number, aval: AbstractValue) {
  const shape = [...aval.shape];
  shape.splice(batchDim, 1); // Remove the batch dimension.
  return new ShapedArray(shape, aval.dtype, aval.weakType);
}

/** Move one axis to a different index. */
export function moveaxis(x: TracerValue, src: number, dst: number) {
  const t = pureArray(x);
  src = checkAxis(src, t.ndim);
  dst = checkAxis(dst, t.ndim);
  if (src === dst) return t;
  const perm = range(t.ndim);
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose(t, perm);
}

function moveBatchAxis(
  axisSize: number,
  src: number | null,
  dst: number,
  x: Tracer,
) {
  if (src === null) {
    // not_mapped
    const targetShape = [...x.shape];
    targetShape.splice(dst, 0, axisSize);
    return broadcast(x, targetShape, [dst]);
  } else if (src === dst) {
    return x;
  } else {
    return moveaxis(x, src, dst);
  }
}

function padRankAfterBatch(
  x: Tracer,
  targetNdim: number,
  origX?: Tracer,
): Tracer {
  if (x.ndim >= targetNdim) return x;
  const reshaped = reshape(x, [
    x.shape[0],
    ...rep(targetNdim - x.ndim, 1),
    ...x.shape.slice(1),
  ]);
  if (origX !== undefined && x !== origX) {
    x[Symbol.dispose]();
  }
  return reshaped;
}

function alignAndPadBatchAxes(
  axisSize: number,
  args: Tracer[],
  dims: (number | null)[],
  targetNdim: number,
): Tracer[] {
  return args.map((x, i) => {
    if (dims[i] === null) return x;
    const moved = moveBatchAxis(axisSize, dims[i], 0, x);
    return padRankAfterBatch(moved, targetNdim, x);
  });
}

class BatchTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly val: Tracer,
    readonly batchDim: number | null,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    if (this.batchDim === null) {
      return this.val.aval;
    } else {
      return mappedAval(this.batchDim, this.val.aval);
    }
  }

  toString(): string {
    return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
  }

  get ref() {
    // Must propagate to inner value so dispose() remains balanced.
    // jax-js-lint: allow-ref
    this.val.ref;
    return this;
  }
  dispose() {
    this.val.dispose();
  }

  override isAliveForCleanup(): boolean {
    return this.val.isAliveForCleanup();
  }

  fullLower(): Tracer {
    if (this.batchDim === null) {
      return this.val.fullLower();
    } else {
      return this;
    }
  }
}

class BatchTrace extends Trace {
  pure(val: TracerValue) {
    const arr = pureArray(val);
    // Track arrays created from raw (non-Tracer) values while batching.
    // These are anonymous temporaries (e.g., scalar literals) and should be
    // released after vmapFlat completes unless they escape via outputs.
    if (!(val instanceof Tracer)) {
      this.pureIntermediates.push(arr);
    }
    return this.lift(arr);
  }

  lift(val: Tracer): Tracer {
    return new BatchTracer(this, val, null);
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: BatchTracer[],
    params: PrimitiveParams<P>,
  ): BatchTracer[] {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === undefined) {
      throw new Error(`No vmap rule for: ${primitive}`);
    }
    if (bdimsIn.every((d) => d === null)) {
      // This should not usually happen because `fullLower()` would unwrap the
      // BatchTracer before getting here. However, I'm not sure about this in
      // edge cases, so it's better to just be safe.
      const valOuts = bind(primitive, valsIn, params);
      const outs = valOuts.map((x) => new BatchTracer(this, x, null));
      this.intermediates.push(...outs);
      return outs;
    }
    const [valOuts, bdimOuts] = vmapRule(
      this.axisSize,
      valsIn,
      bdimsIn,
      params,
    );
    if (valOuts.length !== bdimOuts.length) {
      throw new Error(
        `vmap rule for ${primitive} returned mismatched lengths: ` +
          `${valOuts.length} vs ${bdimOuts.length}`,
      );
    }
    const outs = zip(valOuts, bdimOuts).map(
      ([x, bd]) => new BatchTracer(this, x, bd),
    );
    this.intermediates.push(...outs);
    return outs;
  }

  get axisSize(): number {
    return this.main.globalData.axisSize;
  }

  get pureIntermediates(): Tracer[] {
    return this.main.globalData.pureIntermediates;
  }

  get intermediates(): BatchTracer[] {
    return this.main.globalData.intermediates;
  }
}

// Apply a primitive to batched arguments with built-in broadcasting rules.
//
// This defines "how" a primitive should be vectorized over batch dimensions.
// The caller is guaranteed to pass at least one of `dims` as non-null.
type VmapRule<P extends Primitive> = (
  axisSize: number,
  args: Tracer[],
  dims: (number | null)[],
  params: PrimitiveParams<P>,
) => [Tracer[], (number | null)[]];

/**
 * Process a primitive with built-in broadcasting.
 *
 * Reference: https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/interpreters/batching.py#L1029
 */
function broadcastBatcher<P extends Primitive>(prim: P): VmapRule<P> {
  return (axisSize, args, dims, params) => {
    if (args.length === 0) {
      throw new Error("Empty list in broadcastBatcher");
    }
    // Determine the output ndim after broadcasting, including batch.
    const nd = Math.max(
      ...args.map((x, i) => ndim(x) + (dims[i] === null ? 1 : 0)),
    );

    const firstIdx = dims.findIndex((d) => d !== null);
    const firstBdim = dims[firstIdx]! - args[firstIdx].ndim; // e.g., -1 if last dim
    if (
      // If only agreeing batch dims, or scalars, just call the primitive.
      zip(args, dims).every(
        ([x, d]) =>
          (d === null && ndim(x) < -firstBdim) ||
          (d !== null && d - x.ndim === firstBdim),
      )
    ) {
      return [[bind1(prim, args, params)], [nd + firstBdim]];
    }

    // Move the batch axes to the front. If needed, expand arrays so that all
    // inputs have the same number of dimensions.
    const origArgs = args;
    args = alignAndPadBatchAxes(axisSize, args, dims, nd);
    const result = bind1(prim, args, params);
    // Dispose moveBatchAxis/reshape intermediates after they've been consumed
    for (let i = 0; i < args.length; i++) {
      if (args[i] !== origArgs[i]) args[i][Symbol.dispose]();
    }
    return [[result], [0]];
  };
}

function unopBatcher<P extends Primitive>(prim: P): VmapRule<P> {
  return (axisSize, [x], [xBdim], params) => {
    return [[bind1(prim, [x], params)], [xBdim]];
  };
}

function lastDimsBatcher<P extends Primitive>(
  prim: P,
  inputDims: number,
  numOutputs: number = 1,
): VmapRule<P> {
  return (axisSize, [x], [xBdim], params) => {
    assertNonNull(xBdim);
    if (xBdim < x.ndim - inputDims) {
      return [bind(prim, [x], params), rep(numOutputs, xBdim)];
    }
    const origX = x;
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    const result = bind(prim, [x], params);
    if (x !== origX) x[Symbol.dispose]();
    return [result, rep(numOutputs, 0)];
  };
}

const vmapRules: Partial<{ [P in Primitive]: VmapRule<P> }> = {
  [Primitive.BlockMap](axisSize, args, dims, params) {
    const { jaxpr, numConsts, blockShape, inAxes, outAxes, numInputs } = params;

    const newInAxes = inAxes.map((axes, i) => {
      const bdim = dims[numConsts + i];
      return axes.map((axis) =>
        axis === null ? null : bdim !== null && bdim <= axis ? axis + 1 : axis,
      );
    });

    const newOutAxes = outAxes.map((axes) =>
      axes.map((axis) => (axis === null ? null : axis >= 0 ? axis + 1 : axis)),
    );

    const outBdims = [];
    const bdim = dims.find((d) => d !== null) ?? 0;
    for (let i = 0; i < outAxes.length; i++) outBdims.push(bdim);

    const { jaxpr: batchedJaxpr } = vmapJaxpr(jaxpr, axisSize, dims);

    const res = bind(Primitive.BlockMap, args, {
      jaxpr: batchedJaxpr,
      numConsts,
      numInputs,
      blockShape,
      inAxes: newInAxes,
      outAxes: newOutAxes,
      threadTile: params.threadTile,
      halo: params.halo,
    });

    return [res, outBdims];
  },
  [Primitive.Add]: broadcastBatcher(Primitive.Add),
  [Primitive.Mul]: broadcastBatcher(Primitive.Mul),
  [Primitive.Idiv]: broadcastBatcher(Primitive.Idiv),
  [Primitive.Mod]: broadcastBatcher(Primitive.Mod),
  [Primitive.Min]: broadcastBatcher(Primitive.Min),
  [Primitive.Max]: broadcastBatcher(Primitive.Max),
  [Primitive.BitCombine]: broadcastBatcher(Primitive.BitCombine),
  [Primitive.BitShift]: broadcastBatcher(Primitive.BitShift),
  [Primitive.Neg]: unopBatcher(Primitive.Neg),
  [Primitive.Reciprocal]: unopBatcher(Primitive.Reciprocal),
  [Primitive.Floor]: unopBatcher(Primitive.Floor),
  [Primitive.Ceil]: unopBatcher(Primitive.Ceil),
  [Primitive.StopGradient]: unopBatcher(Primitive.StopGradient),
  [Primitive.BlockIndex]() {
    // BlockIndex is body-local (inside block_map); not batched by vmap.
    return [[bind1(Primitive.BlockIndex, [], {})], [null]];
  },
  [Primitive.Cast]: unopBatcher(Primitive.Cast),
  [Primitive.Bitcast]: unopBatcher(Primitive.Bitcast),
  [Primitive.Sin]: unopBatcher(Primitive.Sin),
  [Primitive.Cos]: unopBatcher(Primitive.Cos),
  [Primitive.Asin]: unopBatcher(Primitive.Asin),
  [Primitive.Atan]: unopBatcher(Primitive.Atan),
  [Primitive.Exp]: unopBatcher(Primitive.Exp),
  [Primitive.Log]: unopBatcher(Primitive.Log),
  [Primitive.Erf]: unopBatcher(Primitive.Erf),
  [Primitive.Erfc]: unopBatcher(Primitive.Erfc),
  [Primitive.Sqrt]: unopBatcher(Primitive.Sqrt),
  [Primitive.Reduce](axisSize, [x], [xBdim], { op, axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduce(x, op, newAxis)], [outBdim]];
  },
  [Primitive.Dot](axisSize, [origX, origY], [xBdim, yBdim]) {
    // Move batch axes to the front so block_map's per-thread validity check
    // works correctly: with batch outermost, each thread handles exactly one
    // batch element's worth of data.  The old strategy (batch at ndim-2)
    // scattered batch-0 elements across multiple threads, causing boundary
    // blocks to zero out valid data assigned to "invalid" threads.

    // core.dot executes a standard right-to-left broadcast over a multiplication
    // and then reduces the last axis. Under vmap, x and y may have different ranks
    // despite sharing a batch size at axis 0. Standard broadcast will misalign the
    // dimensions (e.g. [B, M, K] and [B, K] -> tries to align M and B).
    // alignAndPadBatchAxes pads the lower rank tensor with size-1 axes right
    // after the batch axis to preserve mathematical alignment.
    const maxNdim = Math.max(
      origX.ndim + (xBdim === null ? 1 : 0),
      origY.ndim + (yBdim === null ? 1 : 0),
    );
    const [x, y] = alignAndPadBatchAxes(
      axisSize,
      [origX, origY],
      [xBdim, yBdim],
      maxNdim,
    );

    const z = dot(x, y);
    if (x !== origX) x[Symbol.dispose]();
    if (y !== origY) y[Symbol.dispose]();
    return [[z], [0]];
  },
  [Primitive.Conv](axisSize, [x, y], [xBdim, yBdim], params) {
    // Move batch axes to the front, then increment params.vmapDims.
    const origX = x,
      origY = y;
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    y = moveBatchAxis(axisSize, yBdim, 0, y);
    const z = conv(x, y, { ...params, vmapDims: params.vmapDims + 1 });
    if (x !== origX) x[Symbol.dispose]();
    if (y !== origY) y[Symbol.dispose]();
    return [[z], [0]];
  },
  [Primitive.Pool](axisSize, [x], [xBdim], { window, strides }) {
    assertNonNull(xBdim);
    // Pool operates on the trailing window.length dimensions.
    // Leading dimensions pass through unchanged.
    if (xBdim < x.ndim - window.length) {
      return [bind(Primitive.Pool, [x], { window, strides }), [xBdim]];
    }
    // Batch dim overlaps with pooled spatial dims — move to front
    const origX = x;
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    const result = bind(Primitive.Pool, [x], { window, strides });
    if (x !== origX) x[Symbol.dispose]();
    return [result, [0]];
  },
  [Primitive.PoolTranspose](
    axisSize,
    [x],
    [xBdim],
    { inShape, window, strides },
  ) {
    assertNonNull(xBdim);
    // Move batch to front, prepend batch size to inShape
    const origX = x;
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    const result = bind(Primitive.PoolTranspose, [x], {
      inShape: [axisSize, ...inShape],
      window,
      strides,
    });
    if (x !== origX) x[Symbol.dispose]();
    return [result, [0]];
  },
  [Primitive.Compare]: broadcastBatcher(Primitive.Compare),
  [Primitive.Where]: broadcastBatcher(Primitive.Where),
  [Primitive.Concatenate](axisSize, xs, xBdims, { axis }) {
    const minBdim = Math.min(...xBdims.filter((d) => d !== null));
    const origXs = xs;
    xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
    const newAxis = axis + (minBdim <= axis ? 1 : 0);
    const result = concatenate(xs, newAxis);
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] !== origXs[i]) xs[i][Symbol.dispose]();
    }
    return [[result], [minBdim]];
  },
  [Primitive.Split](axisSize, [x], [xBdim], { axis, sizes }) {
    assertNonNull(xBdim);
    const newAxis = axis + (xBdim <= axis ? 1 : 0);
    const outs = split(x, newAxis, sizes);
    return [outs, rep(outs.length, xBdim)];
  },
  [Primitive.RandomBits](axisSize, [k0, k1], [bdim0, bdim1], { shape, mode }) {
    const origK0 = k0,
      origK1 = k1;
    k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
    k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
    const result = randomBits(k0, k1, [axisSize, ...shape], mode);
    if (k0 !== origK0) k0[Symbol.dispose]();
    if (k1 !== origK1) k1[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.Gather](
    axisSize,
    [x, ...indices],
    [xBdim, ...indicesBdim],
    { axis, outDim },
  ) {
    if (indicesBdim.every((d) => d === null)) {
      // If none of the indices are mapped, this is an ordinary Gather on larger
      // x array. Just recalculate axis numbers.
      assertNonNull(xBdim);
      const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
      let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
      let newOutDim = outDim;
      if (newOutDim < newBdim) newBdim += axis.length;
      else newOutDim += 1;
      return [[gather(x, indices, newAxis, newOutDim)], [newBdim]];
    }
    // If indices are mapped, move those mapped axes to front.
    const nd = Math.max(
      ...indices.map((m, i) => ndim(m) + (indicesBdim[i] === null ? 1 : 0)),
    );
    const origIndices = [...indices];
    indices = indices.map((m, i) => {
      if (indicesBdim[i] === null) return m;
      const moved = moveBatchAxis(axisSize, indicesBdim[i], 0, m);
      if (moved.ndim < nd) {
        const reshaped = moved.reshape([
          moved.shape[0],
          ...rep(nd - moved.ndim, 1),
          ...moved.shape.slice(1),
        ]);
        if (moved !== m) moved[Symbol.dispose]();
        return reshaped;
      }
      return moved;
    });
    // Now there are two cases. If x is not mapped, dispatch directly.
    if (xBdim === null) {
      const result = gather(x, indices, axis, outDim);
      for (let i = 0; i < indices.length; i++) {
        if (indices[i] !== origIndices[i]) indices[i][Symbol.dispose]();
      }
      return [[result], [outDim]];
    } else {
      // Otherwise, we need a new `arange(axisSize)` index.
      // For simplicity, let's also move x's batch axis to the front.
      const origX = x;
      x = moveBatchAxis(axisSize, xBdim, 0, x);
      const newAxis = [0, ...axis.map((ax) => ax + 1)];
      const arangeArr = arange(axisSize);
      const extraBatchIndex = arangeArr.reshape([-1, ...rep(nd - 1, 1)]);
      arangeArr.dispose();
      indices.splice(0, 0, extraBatchIndex);
      const result = gather(x, indices, newAxis, outDim);
      if (x !== origX) x.dispose();
      extraBatchIndex.dispose();
      for (let i = 1; i < indices.length; i++) {
        // skip extraBatchIndex at 0
        if (indices[i] !== origIndices[i - 1]) indices[i].dispose();
      }
      return [[result], [outDim]];
    }
  },
  [Primitive.Transpose](axisSize, [x], [xBdim], { perm }) {
    assertNonNull(xBdim);
    const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
    newPerm.splice(xBdim, 0, xBdim); // Keep the batch dim in place.
    return [[transpose(x, newPerm)], [xBdim]];
  },
  [Primitive.Broadcast](axisSize, [x], [xBdim], { shape, axis }) {
    assertNonNull(xBdim);
    const newShape = shape.toSpliced(xBdim, 0, axisSize);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[broadcast(x, newShape, newAxis)], [xBdim]];
  },
  [Primitive.Reshape](axisSize, [x], [xBdim], { shape }) {
    // Move xBdim to the front, so reshape can have contiguous axes.
    const origX = x;
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    const result = reshape(x, [axisSize, ...shape]);
    if (x !== origX) x[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.Flip](axisSize, [x], [xBdim], { axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[flip(x, newAxis)], [xBdim]];
  },
  [Primitive.Reverse](axisSize, [x], [xBdim], { axis }) {
    assertNonNull(xBdim);
    const newAxis = axis + (xBdim <= axis ? 1 : 0);
    return [[reverse(x, newAxis)], [xBdim]];
  },
  [Primitive.Shrink](axisSize, [x], [xBdim], { slice }) {
    assertNonNull(xBdim);
    const newSlice = slice.toSpliced(xBdim, 0, [0, axisSize]);
    return [[shrink(x, newSlice)], [xBdim]];
  },
  [Primitive.Pad](axisSize, [x], [xBdim], { width }) {
    assertNonNull(xBdim);
    const newWidth = width.toSpliced(xBdim, 0, [0, 0]);
    return [[pad(x, newWidth)], [xBdim]];
  },
  [Primitive.Sort]: lastDimsBatcher(Primitive.Sort, 1),
  [Primitive.Argsort]: lastDimsBatcher(Primitive.Argsort, 1, 2),
  [Primitive.TriangularSolve](
    axisSize,
    [a, b],
    [aBdim, bBdim],
    { unitDiagonal },
  ) {
    if (aBdim === null) {
      // If only vmapping over b, we can just call TriangularSolve directly.
      const origB = b;
      b = moveBatchAxis(axisSize, bBdim, -3, b);
      const bMoved = b;
      const [s, m, n] = b.shape.slice(-3);
      b = b.reshape([...b.shape.slice(0, -3), s * m, n]);
      if (bMoved !== origB) bMoved[Symbol.dispose]();
      let x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
      const xOrig = x;
      const savedShape = [...b.shape.slice(0, -2)];
      x = x.reshape([...savedShape, s, m, n]);
      if (b !== origB) b[Symbol.dispose]();
      xOrig[Symbol.dispose]();
      return [[x], [x.ndim - 3]];
    }
    const origA = a,
      origB = b;
    a = moveBatchAxis(axisSize, aBdim, 0, a);
    b = moveBatchAxis(axisSize, bBdim, 0, b);
    const x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
    if (a !== origA) a[Symbol.dispose]();
    if (b !== origB) b[Symbol.dispose]();
    return [[x], [0]];
  },
  [Primitive.Cholesky]: lastDimsBatcher(Primitive.Cholesky, 2),
  [Primitive.LU]: lastDimsBatcher(Primitive.LU, 2, 3),
  [Primitive.QR]: lastDimsBatcher(Primitive.QR, 2, 2),
  [Primitive.Jit](axisSize, args, dims, { name, jaxpr }) {
    const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind(Primitive.Jit, [...newJaxpr.constEnv.values, ...args], {
      name: `${name}_vmap`,
      jaxpr: newJaxpr.jaxpr,
      numConsts: newJaxpr.constEnv.length,
    });
    return [outs, rep(outs.length, 0)];
  },
  [Primitive.DynamicUpdateSlice](
    axisSize,
    [dst, src],
    [dstBdim, srcBdim],
    { offset, axis },
  ) {
    // Move both batch dims to front, shift DUS axis by 1
    const origDst = dst,
      origSrc = src;
    dst = moveBatchAxis(axisSize, dstBdim, 0, dst);
    src = moveBatchAxis(axisSize, srcBdim, 0, src);
    const newAxis = axis + 1;
    // JIT DUS only supports axis=0 (contiguous memory copy), so decompose
    // axis≠0 into shrink+concatenate which the JIT handles natively.
    let result;
    if (newAxis === 0) {
      result = bind1(Primitive.DynamicUpdateSlice, [dst, src], {
        offset,
        axis: 0,
      });
    } else {
      const dstShape = dst.shape;
      const srcLen = src.shape[newAxis];
      const leftSlice = dstShape.map((s, i) =>
        i === newAxis
          ? ([0, offset] as [number, number])
          : ([0, s] as [number, number]),
      );
      const rightSlice = dstShape.map((s, i) =>
        i === newAxis
          ? ([offset + srcLen, s] as [number, number])
          : ([0, s] as [number, number]),
      );
      using left = shrink(dst, leftSlice);
      using right = shrink(dst, rightSlice);
      result = concatenate([left, src, right], newAxis);
    }
    if (dst !== origDst) dst[Symbol.dispose]();
    if (src !== origSrc) src[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.DynamicUpdateSliceGeneral](
    axisSize,
    [dst, src],
    [dstBdim, srcBdim],
    { startIndices },
  ) {
    // Move both batch dims to front, shift all axes by 1
    const origDst = dst,
      origSrc = src;
    dst = moveBatchAxis(axisSize, dstBdim, 0, dst);
    src = moveBatchAxis(axisSize, srcBdim, 0, src);
    // Batch axis has offset 0 (src and dst share batch dim at position 0)
    const result = bind1(Primitive.DynamicUpdateSliceGeneral, [dst, src], {
      startIndices: [0, ...startIndices],
    });
    if (dst !== origDst) dst[Symbol.dispose]();
    if (src !== origSrc) src[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.ScatterAdd](
    axisSize,
    [target, indices, updates],
    [tBdim, iBdim, uBdim],
    { axis },
  ) {
    // scatterAdd indices are 1-D [updatesLen] — the same positions apply to
    // all outer/inner slices.  When indices are not batched we can simply
    // shift the scatter axis and leave indices as-is.
    if (iBdim !== null) {
      throw new Error(
        "vmap(scatterAdd): batched indices not yet supported — " +
          "use shared (non-batched) indices",
      );
    }
    const origTarget = target,
      origUpdates = updates;
    target = moveBatchAxis(axisSize, tBdim, 0, target);
    updates = moveBatchAxis(axisSize, uBdim, 0, updates);
    const result = scatterAdd(target, indices, updates, axis + 1);
    if (target !== origTarget) target[Symbol.dispose]();
    if (updates !== origUpdates) updates[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.Scan](
    axisSize,
    args,
    dims,
    { jaxpr, numCarry, numConsts, length, reverse },
  ) {
    // vmap of scan: batch over independent scans
    //
    // Scan args layout: [...consts, ...initCarry, ...xs]
    // Body takes: [...consts, ...carry, ...x_slice] -> [...new_carry, ...y]
    //
    // Move all batch dimensions to consistent positions:
    // - consts: batch at axis 0
    // - carry: batch at axis 0
    // - xs: batch at axis 1 (axis 0 is scan length)
    // Then vmap the body to handle the batch dimension.

    const numX = args.length - numConsts - numCarry;
    const numY = jaxpr.outs.length - numCarry;

    // Split args
    const consts = args.slice(0, numConsts);
    const initCarry = args.slice(numConsts, numConsts + numCarry);
    const xs = args.slice(numConsts + numCarry);

    const constDims = dims.slice(0, numConsts);
    const carryDims = dims.slice(numConsts, numConsts + numCarry);
    const xsDims = dims.slice(numConsts + numCarry);

    // Move batch dims to consistent positions
    const movedConsts = consts.map((c, i) =>
      moveBatchAxis(axisSize, constDims[i], 0, c),
    );
    const movedCarry = initCarry.map((c, i) =>
      moveBatchAxis(axisSize, carryDims[i], 0, c),
    );
    // For xs, move batch to axis 1 (after the length axis)
    const movedXs = xs.map((x, i) => {
      if (xsDims[i] === null) {
        // Not mapped - broadcast batch dim at axis 1
        const newShape = [x.shape[0], axisSize, ...x.shape.slice(1)];
        return broadcast(x, newShape, [1]);
      } else if (xsDims[i] === 0) {
        // Batch at axis 0, need it at axis 1 (after length)
        return moveaxis(x, 0, 1);
      } else {
        // Batch at some other axis - move to axis 1
        return moveBatchAxis(axisSize, xsDims[i], 1, x);
      }
    });

    // Body dims: all at axis 0 (consts, carry, x_slice all have batch at axis 0)
    const bodyDims: (number | null)[] = [
      ...rep(numConsts, 0),
      ...rep(numCarry, 0),
      ...rep(numX, 0),
    ];

    // Create vmapped body jaxpr
    const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);

    // Build scan args with moved arrays
    const scanArgs = [
      ...vmappedBody.constEnv.values,
      ...movedConsts,
      ...movedCarry,
      ...movedXs,
    ];

    // Run the scan
    // numConsts must include BOTH the vmapJaxpr's own consts AND the original
    // scan consts (movedConsts). The vmapped body jaxpr still expects the
    // original consts as its first N inputs — these must remain const (not
    // evolve as carry) in the scan.
    const results = bind(Primitive.Scan, scanArgs, {
      jaxpr: vmappedBody.jaxpr,
      numCarry,
      numConsts: vmappedBody.constEnv.length + numConsts,
      length,
      reverse,
    });

    // Dispose moveBatchAxis/moveaxis/broadcast intermediates
    for (let i = 0; i < movedConsts.length; i++) {
      if (movedConsts[i] !== consts[i]) movedConsts[i][Symbol.dispose]();
    }
    for (let i = 0; i < movedCarry.length; i++) {
      if (movedCarry[i] !== initCarry[i]) movedCarry[i][Symbol.dispose]();
    }
    for (let i = 0; i < movedXs.length; i++) {
      if (movedXs[i] !== xs[i]) movedXs[i][Symbol.dispose]();
    }

    // Results: carry has batch at axis 0, ys has batch at axis 1
    // Move ys batch from axis 1 to axis 0
    const carryOut = results.slice(0, numCarry);
    const ysOut = results.slice(numCarry);

    const movedYs = ysOut.map((y) => {
      const moved = moveaxis(y, 1, 0);
      if (moved !== y) y[Symbol.dispose]();
      return moved;
    });

    return [[...carryOut, ...movedYs], rep(numCarry + numY, 0)];
  },
  [Primitive.AssociativeScan](
    axisSize,
    args,
    dims,
    { jaxpr, numLeaves, axis, reverse },
  ) {
    // vmap of associativeScan: batch over independent scans
    //
    // AssociativeScan args: [...consts, ...elems_leaves]
    // Body takes: [...consts, ...a_leaves, ...b_leaves] -> [...result_leaves]
    //
    // Move all batch dims to axis 0, then run batched scan with scan axis
    // shifted to axis+1 (since batch is at axis 0).

    const numConsts = args.length - numLeaves;

    const consts = args.slice(0, numConsts);
    const elems = args.slice(numConsts);

    const constDims = dims.slice(0, numConsts);
    const elemDims = dims.slice(numConsts);

    // Move batch to axis 0 for all args
    const movedConsts = consts.map((c, i) =>
      moveBatchAxis(axisSize, constDims[i], 0, c),
    );
    const movedElems = elems.map((e, i) =>
      moveBatchAxis(axisSize, elemDims[i], 0, e),
    );

    // Body dims: all batch at axis 0
    // (consts get batch at 0, a_leaves get batch at 0, b_leaves get batch at 0)
    const bodyDims: (number | null)[] = [
      ...rep(numConsts, 0),
      ...rep(numLeaves * 2, 0),
    ];

    const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);

    const newArgs = [
      ...vmappedBody.constEnv.values,
      ...movedConsts,
      ...movedElems,
    ];

    const results = bind(Primitive.AssociativeScan, newArgs, {
      jaxpr: vmappedBody.jaxpr,
      numLeaves,
      axis: axis + 1, // scan axis shifts since batch is at axis 0
      reverse,
    });

    // Dispose moved intermediates
    for (let i = 0; i < movedConsts.length; i++) {
      if (movedConsts[i] !== consts[i]) movedConsts[i][Symbol.dispose]();
    }
    for (let i = 0; i < movedElems.length; i++) {
      if (movedElems[i] !== elems[i]) movedElems[i][Symbol.dispose]();
    }

    // Results have batch at axis 0
    return [results, rep(numLeaves, 0)];
  },
  [Primitive.DynamicSlice](axisSize, args, dims, { sliceSizes }) {
    const [xBdim, ...idxBdims] = dims;
    if (idxBdims.some((d) => d !== null)) {
      throw new Error(
        "vmap(dynamicSlice): batched start indices not supported",
      );
    }
    assertNonNull(xBdim);
    const origX = args[0];
    const x = moveBatchAxis(axisSize, xBdim, 0, origX);
    // Zero index for the inserted batch dimension — always in-bounds
    using zero = fullInternal(new ShapedArray([], DType.Int32, false), 0);
    const result = bind1(Primitive.DynamicSlice, [x, zero, ...args.slice(1)], {
      sliceSizes: [axisSize, ...sliceSizes],
    });
    if (x !== origX) x[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.UncheckedDynamicSlice](axisSize, args, dims, { sliceSizes }) {
    const [xBdim, ...idxBdims] = dims;
    if (idxBdims.some((d) => d !== null)) {
      throw new Error(
        "vmap(uncheckedDynamicSlice): batched start indices not supported",
      );
    }
    assertNonNull(xBdim);
    const origX = args[0];
    const x = moveBatchAxis(axisSize, xBdim, 0, origX);
    // Zero index for the batch dimension — in-bounds by construction (0 + B == B)
    using zero = fullInternal(new ShapedArray([], DType.Int32, false), 0);
    const result = bind1(
      Primitive.UncheckedDynamicSlice,
      [x, zero, ...args.slice(1)],
      { sliceSizes: [axisSize, ...sliceSizes] },
    );
    if (x !== origX) x[Symbol.dispose]();
    return [[result], [0]];
  },
  [Primitive.ForiLoop](
    axisSize,
    args,
    dims,
    { jaxpr, numConsts, lower, upper, isJvpTransformed },
  ) {
    // vmap of fori_loop: batch over independent loops
    //
    // ForiLoop args layout: [...consts, ...carry]
    // Body takes: [i, ...carry] -> [...newCarry]
    // The loop index `i` is scalar int32, NOT batched.
    const numCarry = args.length - numConsts;

    const consts = args.slice(0, numConsts);
    const initCarry = args.slice(numConsts);

    const constDims = dims.slice(0, numConsts);
    const carryDims = dims.slice(numConsts);

    // Move batch dims to axis 0
    const movedConsts = consts.map((c, i) =>
      moveBatchAxis(axisSize, constDims[i], 0, c),
    );
    const movedCarry = initCarry.map((c, i) =>
      moveBatchAxis(axisSize, carryDims[i], 0, c),
    );

    // Body dims: [0, ..., 0 (consts batched), null (i not batched), 0, ..., 0 (carry batched)]
    // Body jaxpr inBinders: [const_0, ..., const_k, i, carry_0, ..., carry_n]
    const bodyDims: (number | null)[] = [
      ...rep(numConsts, 0),
      null,
      ...rep(numCarry, 0),
    ];

    const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);

    const foriArgs = [
      ...vmappedBody.constEnv.values,
      ...movedConsts,
      ...movedCarry,
    ];

    const results = bind(Primitive.ForiLoop, foriArgs, {
      jaxpr: vmappedBody.jaxpr,
      numConsts: vmappedBody.constEnv.length + numConsts,
      lower,
      upper,
      ...(isJvpTransformed ? { isJvpTransformed } : {}),
    });

    // Dispose intermediates from moveBatchAxis
    for (let i = 0; i < movedConsts.length; i++) {
      if (movedConsts[i] !== consts[i]) movedConsts[i][Symbol.dispose]();
    }
    for (let i = 0; i < movedCarry.length; i++) {
      if (movedCarry[i] !== initCarry[i]) movedCarry[i][Symbol.dispose]();
    }

    return [results, rep(numCarry, 0)];
  },
  [Primitive.WorkgroupAssociativeScan]() {
    throw new Error("vmap for WorkgroupAssociativeScan not implemented");
  },
};

const vmapJaxprCache = new Map<Jaxpr, Map<string, CachedJaxprArtifact>>();

_registerJitCacheDisposer(() => {
  for (const inner of vmapJaxprCache.values()) {
    for (const artifact of inner.values()) artifact.constEnv.dispose();
    inner.clear();
  }
  vmapJaxprCache.clear();
});
_registerCacheSizeGetter("vmapJaxpr", () => {
  let total = 0;
  for (const inner of vmapJaxprCache.values()) total += inner.size;
  return total;
});

export function vmapJaxpr(
  jaxpr: Jaxpr,
  axisSize: number,
  dims: (number | null)[],
): CachedJaxprArtifact {
  // Include backend type: consts in the cached const environment are concrete arrays
  // living on whichever device was active at first vmap. Cross-device reuse
  // would read stale data from the wrong backend.
  const cacheKey = defaultDevice() + "," + JSON.stringify([axisSize, dims]);
  const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;

  // Consts in the Jaxpr become real inputs after vmap transformation, which is
  // why we ignore numConsts.
  //
  // See the comment in jvpJaxpr() to explain more about what's going on here.
  // This is handling vmap-of-jit, which is a bit tricky. We need to turn the
  // Jaxpr back into a function and retrace it.
  const inAvals = jaxpr.inBinders.map((v, i) => {
    if (dims[i] === null) return v.aval;
    const shape = [...v.aval.shape];
    shape.splice(dims[i], 0, axisSize); // Insert the mapped axis into the shape.
    return new ShapedArray(shape, v.aval.dtype, v.aval.weakType);
  });
  const artifact = buildCachedJaxprArtifact(
    makeJaxpr((args: Tracer[]) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(
      inAvals,
    ),
  );

  if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, new Map());
  vmapJaxprCache.get(jaxpr)!.set(cacheKey, artifact);
  return artifact;
}

function vmapFlat(
  f: (...x: Tracer[]) => TracerValue[],
  inAxes: (number | null)[],
  args: TracerValue[],
): Tracer[] {
  let axisSize: number | undefined = undefined;
  for (let i = 0; i < args.length; i++) {
    if (inAxes[i] !== null) {
      const arg = args[i];
      if (!(arg instanceof Tracer)) {
        throw new TypeError("vmap requires Tracer argument for mapped axes");
      }
      const size = arg.shape[inAxes[i]!];
      if (axisSize === undefined) {
        axisSize = size;
      } else if (axisSize !== size) {
        throw new TypeError(
          "vmap requires all mapped axes to have the same size",
        );
      }
    }
  }
  if (axisSize === undefined) {
    throw new TypeError("vmap requires at least one mapped axis");
  }

  let valsOut: Tracer[], bdimsOut: (number | null)[];
  let shouldDisposeValOut = false;
  {
    using main = newMain(BatchTrace, {
      axisSize,
      pureIntermediates: [] as Tracer[],
      intermediates: [] as BatchTracer[],
    });
    const trace = new BatchTrace(main);
    const tracersIn = args.map((x, i) =>
      inAxes[i] === null
        ? pureArray(x)
        : new BatchTracer(trace, pureArray(x), inAxes[i]),
    );
    const outs = f(...tracersIn);
    const tracersOut = outs.map((out) => fullRaise(trace, out) as BatchTracer);
    [valsOut, bdimsOut] = unzip2(tracersOut.map((t) => [t.val, t.batchDim]));

    // Dispose anonymous arrays created via BatchTrace.pure() that do not
    // escape through outputs. Skip cleanup when a lower abstract trace exists:
    // in that case ownership belongs to the lower trace machinery.
    shouldDisposeValOut = !hasAbstractTraceBelow(main.level);

    if (shouldDisposeValOut) {
      const outputTracers = new Set<BatchTracer>(tracersOut);
      const outputVals = new Set<Tracer>(valsOut);
      for (const t of trace.intermediates) {
        if (outputTracers.has(t)) continue;
        if (outputVals.has(t.val)) continue;
        try {
          t.dispose();
        } catch {
          // Already disposed.
        }
      }

      const outputSet = new Set(valsOut);
      const disposed = new Set<Tracer>();
      for (const arr of trace.pureIntermediates) {
        if (outputSet.has(arr) || disposed.has(arr)) continue;
        disposed.add(arr);
        if (arr.refCount > 0) arr[Symbol.dispose]?.();
      }
    }
  }
  return zip(valsOut, bdimsOut).map(([valOut, bdim]) => {
    const result = moveBatchAxis(axisSize, bdim, 0, valOut);
    if (result !== valOut) {
      if (shouldDisposeValOut) {
        try {
          valOut.dispose();
        } catch {
          // Already disposed.
        }
      } else {
        valOut[Symbol.dispose]();
      }
    }
    return result;
  }); // outs_transposed
}

export function vmap(
  f: (...x: any[]) => JsTree<TracerValue>,
  inAxes: number | JsTree<number | null>[] = 0,
): (...x: JsTree<TracerValue>[]) => JsTree<Tracer> {
  return (...args: any[]) => {
    const [argsFlat, inTree] = treeFlatten(args);
    let inAxesFlat: (number | null)[] = [];
    if (typeof inAxes === "number") {
      // If mapping over a single axis, just use it for all inputs.
      inAxesFlat = rep(argsFlat.length, inAxes);
    } else {
      // Allow either `null | number` (or undefined), or a tree structure
      // matching each input.
      for (let i = 0; i < args.length; i++) {
        if (inAxes[i] == null) {
          inAxesFlat.push(...rep(inTree.childTreedefs[i].size, null));
        } else if (typeof inAxes[i] === "number") {
          inAxesFlat.push(
            ...rep(inTree.childTreedefs[i].size, inAxes[i] as number),
          );
        } else {
          // Must be a tree structure.
          const [axesFlat, axesTreeDef] = treeFlatten(inAxes[i]);
          if (!inTree.childTreedefs[i].equals(axesTreeDef)) {
            throw new TreeMismatchError(
              "vmap",
              inTree.childTreedefs[i],
              axesTreeDef,
            );
          }
          inAxesFlat.push(...axesFlat);
        }
      }
    }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === undefined) {
      throw new Error("outTree was not set in vmap");
    }
    return treeUnflatten(outTree.value, outsFlat);
  };
}

// See also: jacrev()
export function jacfwd(f: any) {
  return function jacobianForward(x: Tracer) {
    if (x.shape.length !== 1) {
      throw new TypeError("jacfwd only supports 1D inputs");
    }
    const [size] = x.shape;
    const pushfwd = (v: Tracer) => {
      const [primals, tangents] = jvp(f, [x], [v]);
      // In eager vmap contexts, primals can be BatchTracers whose
      // [Symbol.dispose] is a no-op (Tracer base). Dispose explicitly to
      // release their underlying concrete arrays.
      // In abstract tracing contexts (e.g., jit(jacfwd(...))), keep the
      // previous no-op behavior to avoid mutating tracer bookkeeping.
      if (!insideAbstractTrace()) {
        const [primalsFlat] = treeFlatten(primals);
        for (const p of primalsFlat) {
          if (p instanceof Tracer) {
            try {
              p.dispose();
            } catch {
              // Already disposed.
            }
          }
        }
      } else {
        const [primalsFlat] = treeFlatten(primals);
        for (const p of primalsFlat) {
          if (p instanceof Tracer) p[Symbol.dispose]();
        }
      }
      return tangents;
    };
    const eyeMatrix = eye(size, undefined, { dtype: x.dtype });
    // Claim the creation ref: this function takes explicit ownership and
    // will dispose eyeMatrix below. Without this, if tracing captures
    // eyeMatrix as a const, makeJaxpr would also try to balance the
    // creation ref, causing a double-balance.
    eyeMatrix.claimCreationRef("jacfwd-eye-matrix");
    const result = vmap(pushfwd, [0])(eyeMatrix);
    eyeMatrix.dispose();
    return result;
  };
}
