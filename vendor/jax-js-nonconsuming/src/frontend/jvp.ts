import { DType } from "../alu";
import { AluOp, isFloatDtype } from "../alu";
import { defaultDevice } from "../backend";
import { Pair } from "../shape";
import {
  JsTree,
  flatten as treeFlatten,
  map as treeMap,
  unflatten as treeUnflatten,
} from "../tree";
import { checkAxis, unzip2, zip } from "../utils";
import { arange, Array, eye, pureArray, tril, triu, zerosLike } from "./array";
import {
  _registerCacheSizeGetter,
  _registerJitCacheDisposer,
} from "./check-leaks";
import {
  AbstractValue,
  argsort,
  asin,
  atan,
  bind,
  bind1,
  bitcast,
  broadcast,
  cast,
  cholesky,
  cos,
  currentTraceLevel,
  dot,
  erf,
  erfc,
  exp,
  flattenFun,
  flattenFunWithAux,
  fullRaise,
  gather,
  getCustomJvpDef,
  getCustomJvpFlatInTree,
  getCustomJvpOutTreeStore,
  getCustomVjpDef,
  hasAbstractTraceBelow,
  idiv,
  less,
  log,
  lu,
  max,
  min,
  mod,
  neg,
  newMain,
  notEqual,
  pad,
  Primitive,
  PrimitiveParams,
  qr,
  reciprocal,
  reduce,
  scatterAdd,
  ShapedArray,
  sin,
  sqrt,
  Trace,
  Tracer,
  TracerValue,
  TreeMismatchError,
  triSolve,
  where,
} from "./core";
import {
  buildCachedJaxprArtifact,
  CachedJaxprArtifact,
  evalJaxpr,
  Jaxpr,
  jaxprAsFun,
  makeJaxpr,
} from "./jaxpr";
import { moveaxis } from "./vmap";

class JVPTracer extends Tracer {
  #rc = 1;

  constructor(
    trace: Trace,
    readonly primal: Tracer,
    readonly tangent: Tracer,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.primal.aval;
  }

  toString(): string {
    return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
  }

  get ref() {
    this.#rc++;
    return this;
  }
  dispose() {
    if (--this.#rc === 0) {
      this.primal.dispose();
      this.tangent.dispose();
    }
  }

  /**
   * Override Tracer's no-op to cascade disposal to primal and tangent.
   *
   * Also skip when a lower abstract trace exists. In nested abstract
   * compositions (e.g., makeJaxpr(jvp(...))), core.bind may call
   * Symbol.dispose on raised raw-literal arguments. Cascading here would free
   * primals/tangents that have already been captured as Jaxpr consts.
   *
   * During PE tracing, lexical disposal is now allowed. Known PE values are
   * either independently retained by instantiateConst/.ref or later swept by
   * the PE cleanup path, which already tolerates already-disposed wrappers.
   */
  [Symbol.dispose]() {
    if (!hasAbstractTraceBelow(this._trace.main.level)) {
      this.dispose();
    }
  }
}

class JVPTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    const zero = zerosLike(val);
    // Claim ownership of the zero tangent's creation ref. The creation ref
    // is balanced by liftedTangents cleanup in jvpFlat — makeJaxpr must NOT
    // double-balance it.
    zero.claimCreationRef("jvp-lifted-tangents");
    const data = this.main.globalData as JvpGlobalData | null;
    if (data) data.liftedTangents.push(zero);
    return new JVPTracer(this, val, zero);
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: JVPTracer[],
    params: PrimitiveParams<P>,
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent]),
    );
    const jvpRule: JvpRule<P> | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    const result = zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t),
    );
    // Track intermediates for cleanup in jvpFlat. globalData is the shared
    // JvpGlobalData — safe because all JVPTrace instances from the same
    // main share the same globalData reference.
    const data = this.main.globalData as JvpGlobalData | null;
    if (data) data.intermediates.push(...result);
    return result;
  }
}

/** Data shared between JVPTrace instances from the same main for cleanup. */
type JvpGlobalData = {
  intermediates: JVPTracer[];
  liftedTangents: Tracer[];
};

type JvpRule<P extends Primitive> = (
  primals: Tracer[],
  tangents: Tracer[],
  params: PrimitiveParams<P>,
) => [Tracer[], Tracer[]];

type JvpRetainedHandoffKind = "sort-idx" | "argsort-idx";

let _jvpRetainedHandoffObserver:
  | ((kind: JvpRetainedHandoffKind, value: Tracer) => void)
  | null = null;

export function _setJvpRetainedHandoffObserver(
  observer: ((kind: JvpRetainedHandoffKind, value: Tracer) => void) | null,
): void {
  _jvpRetainedHandoffObserver = observer;
}

/**
 * JVP-local helper for handing a retained handle to downstream tracing/capture.
 *
 * This makes ownership transfer local and explicit at the handoff site
 * instead of relying on downstream users to retain the original local owner
 * before it is disposed.
 */
function withLocalJvpRetainedHandoff<T extends Tracer, R>(
  kind: JvpRetainedHandoffKind,
  value: T,
  use: (retained: T) => R,
): R {
  // jax-js-lint: allow-ref
  const retained = value.ref;
  try {
    _jvpRetainedHandoffObserver?.(kind, retained);
    return use(retained);
  } finally {
    retained.dispose();
  }
}

/** Rule that applies the same operation to primals and tangents. */
function linearTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return (primals, tangents, params) => {
    const ys = bind(primitive, primals, params);
    const dys = bind(primitive, tangents, params);
    return [ys, dys];
  };
}

/** Rule for product of gradients in bilinear operations. */
function bilinearTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return ([x, y], [dx, dy], params) => {
    const primal = bind1(primitive, [x, y], params);
    using xdy = bind1(primitive, [x, dy], params);
    using dxy = bind1(primitive, [dx, y], params);
    const tangent = xdy.add(dxy); // (xy)' = xy' + x'y
    return [[primal], [tangent]];
  };
}

/** Rule that zeros out any tangents. */
function zeroTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return (primals, tangents, params) => {
    const ys = bind(primitive, primals, params);
    return [ys, ys.map((y) => zerosLike(y))];
  };
}

/** Compute `a @ b.T`, batched to last two axes. */
function batchMatmulT(a: Tracer, b: Tracer): Tracer {
  using aReshaped = a.reshape(a.shape.toSpliced(-1, 0, 1));
  using bReshaped = b.reshape(b.shape.toSpliced(-2, 0, 1));
  return dot(aReshaped, bReshaped);
}
/** Batch matrix transpose. */
function mT(a: Tracer): Tracer {
  return moveaxis(a, -2, -1);
}
function sliceAxis(a: Tracer, axis: number, p: Pair): Tracer {
  const slices = globalThis.Array(a.shape.length).fill([]);
  slices[checkAxis(axis, a.ndim)] = p;
  return a.slice(...slices);
}
function padAxis(a: Tracer, axis: number, p: Pair): Tracer {
  const pads = globalThis.Array(a.shape.length).fill([0, 0]);
  pads[checkAxis(axis, a.ndim)] = p;
  return pad(a, pads);
}

const jvpRules: { [P in Primitive]: JvpRule<P> } = {
  [Primitive.Add]: linearTangentsJvp(Primitive.Add),
  [Primitive.Mul]: bilinearTangentsJvp(Primitive.Mul),
  [Primitive.Idiv]: zeroTangentsJvp(Primitive.Idiv),
  [Primitive.Mod]([x, y], [dx, dy]) {
    // x % y = x - y * trunc(x / y)
    // d(x % y) = dx - dy * trunc(x / y)
    if (!isFloatDtype(x.dtype) && !isFloatDtype(y.dtype)) {
      const result = mod(x, y);
      return [[result], [zerosLike(result)]];
    }
    using q = idiv(x, y);
    using dyq = dy.mul(q);
    const tangent = dx.sub(dyq);
    return [[mod(x, y)], [tangent]];
  },
  [Primitive.Min]([x, y], [dx, dy]) {
    using cond = less(y, x);
    return [[min(x, y)], [where(cond, dy, dx)]];
  },
  [Primitive.Max]([x, y], [dx, dy]) {
    using cond = less(x, y);
    return [[max(x, y)], [where(cond, dy, dx)]];
  },
  [Primitive.BitCombine]: zeroTangentsJvp(Primitive.BitCombine),
  [Primitive.BitShift]: zeroTangentsJvp(Primitive.BitShift),
  [Primitive.Neg]: linearTangentsJvp(Primitive.Neg),
  [Primitive.Reciprocal]([x], [dx]) {
    // d(1/x) = -x^-2 * dx
    const xRecip = reciprocal(x);
    using xRecipSq = xRecip.mul(xRecip);
    using negXRecipSq = neg(xRecipSq);
    return [[xRecip], [negXRecipSq.mul(dx)]];
  },
  [Primitive.Floor]: zeroTangentsJvp(Primitive.Floor),
  [Primitive.Ceil]: zeroTangentsJvp(Primitive.Ceil),
  [Primitive.StopGradient]: zeroTangentsJvp(Primitive.StopGradient),
  [Primitive.Cast]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]]; // No-op if dtype is the same.
    // If floating-point, cast to the new dtype. Otherwise discard the tangent.
    if (isFloatDtype(dtype) && isFloatDtype(x.dtype)) {
      return [[cast(x, dtype)], [cast(dx, dtype)]];
    } else {
      return [[cast(x, dtype)], [zerosLike(x)]];
    }
  },
  [Primitive.Bitcast]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]]; // No-op if dtype is the same.
    return [[bitcast(x, dtype)], [zerosLike(x)]];
  },
  [Primitive.Sin]([x], [dx]) {
    using cosX = cos(x);
    return [[sin(x)], [cosX.mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    using sinX = sin(x);
    using negSinX = neg(sinX);
    return [[cos(x)], [negSinX.mul(dx)]];
  },
  [Primitive.Asin]([x], [dx]) {
    // d(asin(x)) = 1/sqrt(1 - x^2) * dx
    using one = cast(1, x.dtype);
    using xSq = x.mul(x);
    using oneMinusXSq = one.sub(xSq);
    using recip = reciprocal(oneMinusXSq);
    using denom = sqrt(recip);
    return [[asin(x)], [denom.mul(dx)]];
  },
  [Primitive.Atan]([x], [dx]) {
    // d(atan(x)) = 1/(1 + x^2) * dx
    using one = cast(1, x.dtype);
    using xSq = x.mul(x);
    using denom = one.add(xSq);
    return [[atan(x)], [dx.div(denom)]];
  },
  [Primitive.Exp]([x], [dx]) {
    // d(exp(x)) = exp(x) * dx
    const z = exp(x);
    return [[z], [z.mul(dx)]];
  },
  [Primitive.Log]([x], [dx]) {
    // d(log(x)) = 1/x * dx
    using recipX = reciprocal(x);
    return [[log(x)], [recipX.mul(dx)]];
  },
  [Primitive.Erf]([x], [dx]) {
    // d(erf(x)) = 2/sqrt(pi) * exp(-x^2) * dx
    const coeff = 2 / Math.sqrt(Math.PI);
    using xSq = x.mul(x);
    using negXSq = neg(xSq);
    using expTerm = exp(negXSq);
    using scaled = expTerm.mul(coeff);
    return [[erf(x)], [scaled.mul(dx)]];
  },
  [Primitive.Erfc]([x], [dx]) {
    // d(erfc(x)) = -2/sqrt(pi) * exp(-x^2) * dx
    const coeff = -2 / Math.sqrt(Math.PI);
    using xSq = x.mul(x);
    using negXSq = neg(xSq);
    using expTerm = exp(negXSq);
    using scaled = expTerm.mul(coeff);
    return [[erfc(x)], [scaled.mul(dx)]];
  },
  [Primitive.Sqrt]([x], [dx]) {
    // d(sqrt(x)) = 1/(2*sqrt(x)) * dx
    const z = sqrt(x);
    using z2 = z.mul(2);
    using recipZ2 = reciprocal(z2);
    return [[z], [recipZ2.mul(dx)]];
  },
  [Primitive.Reduce]([x], [dx], { op, axis }) {
    if (op === AluOp.Add) {
      return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
    } else if (op === AluOp.Mul) {
      // Multivariate product rule: (abc)'/abc = a'/a + b'/b + c'/c
      const primal = reduce(x, op, axis);
      using bcast = broadcast(primal, x.shape, axis);
      using recip = reciprocal(x);
      using bcastTimesRecip = bcast.mul(recip);
      using bcastTimesRecipTimesDx = bcastTimesRecip.mul(dx);
      const tangent = bcastTimesRecipTimesDx.sum(axis);
      return [[primal], [tangent]];
    } else if (op === AluOp.Min || op === AluOp.Max) {
      const primal = reduce(x, op, axis);
      using bcastPrimal = broadcast(primal, x.shape, axis);
      using notMin = notEqual(x, bcastPrimal);
      using whereNotMin0 = where(notMin, 0.0, 1.0);
      using minCount = whereNotMin0.sum(axis);
      using whereNotMin0Dx = where(notMin, 0.0, dx);
      using sumDx = whereNotMin0Dx.sum(axis);
      const tangent = sumDx.div(minCount);
      return [[primal], [tangent]];
    } else {
      throw new Error(`JVP rule not implemented for reduce op: ${op}`);
    }
  },
  [Primitive.Pool]: linearTangentsJvp(Primitive.Pool),
  [Primitive.PoolTranspose]: linearTangentsJvp(Primitive.PoolTranspose),
  [Primitive.Dot]: bilinearTangentsJvp(Primitive.Dot),
  [Primitive.Conv]: bilinearTangentsJvp(Primitive.Conv),
  [Primitive.Compare]: zeroTangentsJvp(Primitive.Compare),
  [Primitive.BlockIndex]: zeroTangentsJvp(Primitive.BlockIndex),
  [Primitive.Where]([cond, x, y], [_dcond, dx, dy]) {
    return [[where(cond, x, y)], [where(cond, dx, dy)]];
  },
  [Primitive.Concatenate]: linearTangentsJvp(Primitive.Concatenate),
  [Primitive.Split]: linearTangentsJvp(Primitive.Split),
  [Primitive.RandomBits]: zeroTangentsJvp(Primitive.RandomBits),
  [Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
    // d(gather(x, indices)) = gather(dx, indices).
    // Note: We ignore the tangents for indices, since they are not differentiable.
    const indicesRef = indices;
    return [
      [gather(x, indices, axis, outDim)],
      [gather(dx, indicesRef, axis, outDim)],
    ];
  },
  [Primitive.ScatterAdd](
    [target, indices, updates],
    [dTarget, _dIndices, dUpdates],
    { axis },
  ) {
    // d(scatter_add(target, indices, updates, axis))
    //   = scatter_add(dTarget, indices, dUpdates, axis)
    // Indices are not differentiable.
    const primal = scatterAdd(target, indices, updates, axis);
    const tangent = scatterAdd(dTarget, indices, dUpdates, axis);
    return [[primal], [tangent]];
  },
  [Primitive.Transpose]: linearTangentsJvp(Primitive.Transpose),
  [Primitive.Broadcast]: linearTangentsJvp(Primitive.Broadcast),
  [Primitive.Reshape]: linearTangentsJvp(Primitive.Reshape),
  [Primitive.Flip]: linearTangentsJvp(Primitive.Flip),
  [Primitive.Reverse]: linearTangentsJvp(Primitive.Reverse),
  [Primitive.Shrink]: linearTangentsJvp(Primitive.Shrink),
  [Primitive.Pad]: linearTangentsJvp(Primitive.Pad),
  [Primitive.Sort]([x], [dx]) {
    // Propagate both primals and derivatives along the sorted order.
    const [y, idx] = argsort(x);
    try {
      const gatherResult = withLocalJvpRetainedHandoff(
        "sort-idx",
        idx,
        (retainedIdx) => gather(dx, [retainedIdx], [-1], -1),
      );
      return [[y], [gatherResult]];
    } finally {
      idx.dispose();
    }
  },
  [Primitive.Argsort]([x], [dx]) {
    const [y, idx] = argsort(x);
    const gatherResult = withLocalJvpRetainedHandoff(
      "argsort-idx",
      idx,
      (retainedIdx) => gather(dx, [retainedIdx], [-1], -1),
    );
    const zerosIdx = zerosLike(idx);
    return [
      [y, idx],
      [gatherResult, zerosIdx],
    ];
  },
  [Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
    // The primitive solves A @ X.T = B.T, where A is upper triangular.
    // Only the upper triangle of A affects X, so we mask dA accordingly.
    // JVP: dA @ X.T + A @ dX.T = dB.T
    //   => A @ dX.T = dB.T - triu(dA) @ X.T
    //   => dX.T = A^-1 @ (dB.T - triu(dA) @ X.T)
    const x = triSolve(a, b, { unitDiagonal }); // (A^-1 @ B.T).T
    // Mask dA to the triangle actually read by the solver.
    // unitDiagonal means the diagonal is forced to 1 (not read from A).
    using maskedDa = (
      unitDiagonal ? triu(da as any, 1) : triu(da as any)
    ) as Tracer;
    using dax = batchMatmulT(maskedDa, x); // triu(dA) @ X.T
    using mTdax = mT(dax);
    using rhsT = db.sub(mTdax); // (dB.T - triu(dA) @ X.T).T
    const dx = triSolve(a, rhsT, { unitDiagonal });
    return [[x], [dx]];
  },
  [Primitive.QR]([a], [da]) {
    // QR decomposition JVP (thin QR, m >= n only).
    // Reference: Papanicolopulos (2024), arXiv:2409.13374, §4.1, eqs. (9)–(12).
    // B = dA R⁻¹,  E = QᵀB,  Ψ = triu(E) + tril(E,−1)ᵀ,
    // dR = Ψ R,  dQ = B − QΨ.
    const [Q, R] = qr(a);
    const m = a.shape[a.ndim - 2] as number;
    const n = a.shape[a.ndim - 1] as number;
    if (m < n)
      throw new Error(
        "qr jvp: m < n (wide matrices) not yet supported, got " + `${m}x${n}`,
      );
    // B = dA R⁻¹  [m × n]
    // core.triSolve(Rᵀ, x, {lower:true}) = x R⁻¹
    using Rt = mT(R);
    using B = triSolve(Rt, da, { lower: true }) as Tracer;
    // E = Qᵀ B  [n × n]
    using Qt = mT(Q);
    using Bt = mT(B);
    using E = batchMatmulT(Qt, Bt) as Tracer;
    // Ψ = triu(E) + tril(E,−1)ᵀ  (upper triangular)
    using trilE = tril(E as any, -1) as Tracer;
    using trilEt = mT(trilE);
    using triuE = triu(E as any) as Tracer;
    using Psi = triuE.add(trilEt) as Tracer;
    // dR = Ψ R  [n × n]
    const dR = batchMatmulT(Psi, Rt);
    // dQ = B − QΨ  [m × n]
    using Psit = mT(Psi);
    using QPsi = batchMatmulT(Q, Psit) as Tracer;
    const dQ = B.sub(QPsi);
    return [
      [Q, R],
      [dQ, dR],
    ];
  },
  [Primitive.Cholesky]([a], [da]) {
    // If L = cholesky(A), so that A = L @ L^T, then
    // dL = L @ tril(S - 0.5 * diag(S)),
    //   where S = L^{-1} @ dA @ L^{-T}
    const L = cholesky(a);
    using mTda = mT(da);
    using daSymm = da.add(mTda);
    using da2 = daSymm.mul(0.5); // Symmetrize dA for grad
    using W = triSolve(L, da2, { lower: true }); // (L^-1 @ dA.T).T = dA @ L^-T
    using mTW = mT(W);
    using ST = triSolve(L, mTW, { lower: true });
    using triuST1 = triu(ST as any, 1);
    using triuST0 = triu(ST as any);
    using triuSum = triuST1.add(triuST0);
    using triuHalf = triuSum.mul(0.5);
    const dL = batchMatmulT(L, triuHalf);
    return [[L], [dL]];
  },
  [Primitive.LU]([a], [da]) {
    // https://github.com/jax-ml/jax/blob/jax-v0.8.2/jax/_src/lax/linalg.py#L1484
    const [luMatrix, pivots, permutation] = lu(a);
    const [m, n] = a.shape.slice(-2);
    const k = Math.min(m, n);
    // Extract full L: lower triangular with unit diagonal, shape [..., m, m]
    using luSliceL = sliceAxis(luMatrix, -1, [0, k]);
    // Note: lLower/uUpper are NOT declared with `using` when m<=k / n<=k
    // because in that case they alias lPadded/uPadded directly. During PE
    // tracing, explicit disposal here would fight the PE cleanup path, so the
    // PE intermediate cleanup handles those aliases instead.
    const lLower = tril(luSliceL as any, -1);
    const lPaddedNeedsDispose = m > k;
    const lPadded = lPaddedNeedsDispose
      ? padAxis(lLower, -1, [0, m - k])
      : lLower;
    using eyeM = eye(m, { dtype: a.dtype });
    using L = lPadded.add(eyeM);
    if (lPaddedNeedsDispose) lPadded[Symbol.dispose]();
    lLower[Symbol.dispose]();
    // Extract full U: upper triangular, shape [..., n, n]
    // U = triu(lu[:k, :]) padded to [..., n, n] + eye for remaining rows
    using luSliceU = sliceAxis(luMatrix, -2, [0, k]);
    const uUpper = triu(luSliceU as any);
    const uPaddedNeedsDispose = n > k;
    const uPadded = uPaddedNeedsDispose
      ? padAxis(uUpper, -2, [0, n - k])
      : uUpper;
    using uEye =
      n > k
        ? (() => {
            using innerEye = eye(n - k, { dtype: a.dtype });
            using padded1 = padAxis(innerEye, -1, [k, 0]);
            return padAxis(padded1, -2, [k, 0]);
          })()
        : zerosLike(uPadded);
    using U = uPadded.add(uEye);
    if (uPaddedNeedsDispose) uPadded[Symbol.dispose]();
    uUpper[Symbol.dispose]();
    // Apply permutation to da: P @ da (reorder rows)
    using permReshaped = permutation.reshape([...permutation.shape, 1]);
    using arangeM = arange(m);
    using permEq = permReshaped.equal(arangeM);
    using P = permEq.astype(da.dtype);
    using mTda = mT(da);
    using pda = batchMatmulT(P, mTda);
    // Solve L @ la = P @ da for la (la = L^{-1} @ P @ da)
    using mTpda = mT(pda);
    using solvedPda = triSolve(L, mTpda, {
      lower: true,
      unitDiagonal: true,
    });
    using la = mT(solvedPda);
    // Solve lau @ U = la for lau (lau = la @ U^{-1})
    using mTU = mT(U);
    using lau = triSolve(mTU, la, { lower: true });
    using trilLau = tril(lau as any, -1);
    using mTtrilLau = mT(trilLau);
    using lDot = batchMatmulT(L, mTtrilLau); // L' = L @ tril(lau)
    using triuLau = triu(lau as any);
    using mTU2 = mT(U);
    using uDot = batchMatmulT(triuLau, mTU2); // U' = triu(lau) @ U
    // Return values must NOT use `using` — they're returned to the caller
    const luDot = lDot.add(uDot);
    const zerosPivots = zerosLike(pivots);
    const zerosPerm = zerosLike(permutation);
    return [
      [luMatrix, pivots, permutation],
      [luDot, zerosPivots, zerosPerm],
    ];
  },
  [Primitive.Jit](primals, tangents, { name, jaxpr }) {
    const newJaxpr = jvpJaxpr(jaxpr);
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.constEnv.values, ...primals, ...tangents],
      {
        name: `${name}_jvp`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.constEnv.length,
      },
    );
    const n = outs.length / 2;
    if (!Number.isInteger(n))
      throw new Error("internal: JVP Jaxpr output length is not even");
    const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
    return [primalsOut, tangentsOut];
  },
  [Primitive.DynamicUpdateSlice]: linearTangentsJvp(
    Primitive.DynamicUpdateSlice,
  ),
  [Primitive.DynamicUpdateSliceGeneral]: linearTangentsJvp(
    Primitive.DynamicUpdateSliceGeneral,
  ),
  [Primitive.Scan](
    primals,
    tangents,
    { jaxpr, numCarry, numConsts, length, reverse, checkpoint },
  ) {
    // JVP of scan: run a combined scan that processes both primals and tangents.
    //
    // Original scan:
    //   body: (consts, carry, x) -> (new_carry, y)
    //   scan: (consts, init_carry, xs) -> (final_carry, ys)
    //
    // JVP body from jvpJaxpr expects inputs as: [all primals..., all tangents...]
    //   i.e., [consts, carry, x, consts_dot, carry_dot, x_dot]
    // And outputs: [primal_outs..., tangent_outs...]
    //   i.e., [new_carry, y, new_carry_dot, y_dot]
    //
    // But scan feeds body as: [consts..., carry..., x...]
    // So for JVP scan with doubled carry/xs, body receives:
    //   [constsP, constsT, carryP, carryT, xP, xT]  (scan order)
    //
    // We need to reorder to match jvpJaxpr expectations:
    //   [constsP, carryP, xP, constsT, carryT, xT]  (jvp order)
    //
    // Similarly for outputs, jvpJaxpr produces:
    //   [new_carryP, yP, new_carryT, yT]  (jvp order)
    // But scan expects:
    //   [new_carryP, new_carryT, yP, yT]  (scan order, carry then ys)

    const numX = primals.length - numConsts - numCarry;
    const numY = jaxpr.outs.length - numCarry;

    // Transform the body jaxpr to compute JVP
    const jvpBody = jvpJaxpr(jaxpr);

    // jvpBody.jaxpr.inBinders = [jvpConsts..., primals..., tangents...]
    //   where primals = [constsP, carryP, xP] and tangents = [constsT, carryT, xT]
    // jvpBody.constEnv.values = the actual values for jvpConsts
    const numJvpConsts = jvpBody.constEnv.length;
    const numBodyInputs = numConsts + numCarry + numX;

    // Get the body input avals in JVP order (primals then tangents)
    const jvpOrderAvals = jvpBody.jaxpr.inBinders
      .slice(numJvpConsts)
      .map((v) => v.aval);

    // Reorder to scan order: [constsP, constsT, carryP, carryT, xP, xT]
    const constsP_avals = jvpOrderAvals.slice(0, numConsts);
    const carryP_avals = jvpOrderAvals.slice(numConsts, numConsts + numCarry);
    const xP_avals = jvpOrderAvals.slice(numConsts + numCarry, numBodyInputs);
    const constsT_avals = jvpOrderAvals.slice(
      numBodyInputs,
      numBodyInputs + numConsts,
    );
    const carryT_avals = jvpOrderAvals.slice(
      numBodyInputs + numConsts,
      numBodyInputs + numConsts + numCarry,
    );
    const xT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numCarry);

    const wrapperInAvals = [
      ...constsP_avals,
      ...constsT_avals,
      ...carryP_avals,
      ...carryT_avals,
      ...xP_avals,
      ...xT_avals,
    ];

    const { jaxpr: wrapperJaxpr } = makeJaxpr(
      (...scanOrderArgs: Tracer[]): Tracer[] => {
        // scanOrderArgs layout: [constsP, constsT, carryP, carryT, xP, xT]
        const constsP_in = scanOrderArgs.slice(0, numConsts);
        const constsT_in = scanOrderArgs.slice(numConsts, numConsts * 2);
        const carryP_in = scanOrderArgs.slice(
          numConsts * 2,
          numConsts * 2 + numCarry,
        );
        const carryT_in = scanOrderArgs.slice(
          numConsts * 2 + numCarry,
          numConsts * 2 + numCarry * 2,
        );
        const xP_in = scanOrderArgs.slice(
          numConsts * 2 + numCarry * 2,
          numConsts * 2 + numCarry * 2 + numX,
        );
        const xT_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2 + numX);

        // Reorder to jvp order: [constsP, carryP, xP, constsT, carryT, xT]
        const jvpOrderArgs = [
          ...constsP_in,
          ...carryP_in,
          ...xP_in,
          ...constsT_in,
          ...carryT_in,
          ...xT_in,
        ];

        // Call the jvpBody jaxpr with jvpConsts (captured) first, then reordered body args
        const jvpOutputs = bind(
          Primitive.Jit,
          [...jvpBody.constEnv.values, ...jvpOrderArgs],
          {
            jaxpr: jvpBody.jaxpr,
            numConsts: numJvpConsts,
            name: "jvp_body",
          },
        );

        // jvpOutputs layout: [carryP..., yP..., carryT..., yT...]
        // Reorder to scan output order: [carryP..., carryT..., yP..., yT...]
        const carryP_out = jvpOutputs.slice(0, numCarry);
        const yP_out = jvpOutputs.slice(numCarry, numCarry + numY);
        const carryT_out = jvpOutputs.slice(
          numCarry + numY,
          numCarry * 2 + numY,
        );
        const yT_out = jvpOutputs.slice(numCarry * 2 + numY);

        return [...carryP_out, ...carryT_out, ...yP_out, ...yT_out];
      },
    )(...wrapperInAvals);

    // Original args: consts (numConsts), carry (numCarry), xs (numX)
    const constsP = primals.slice(0, numConsts);
    const carryP = primals.slice(numConsts, numConsts + numCarry);
    const xsP = primals.slice(numConsts + numCarry);

    const constsT = tangents.slice(0, numConsts);
    const carryT = tangents.slice(numConsts, numConsts + numCarry);
    const xsT = tangents.slice(numConsts + numCarry);

    // Build scan args in scan order:
    // [wrapperConsts..., constsP, constsT, carryP, carryT, xsP, xsT]
    const scanArgsJvp = [
      ...wrapperJaxpr.consts,
      ...constsP,
      ...constsT,
      ...carryP,
      ...carryT,
      ...xsP,
      ...xsT,
    ];

    const results = bind(Primitive.Scan, scanArgsJvp, {
      jaxpr: wrapperJaxpr.jaxpr,
      numCarry: numCarry * 2,
      numConsts: wrapperJaxpr.consts.length + numConsts * 2,
      length,
      reverse,
      checkpoint,
      isJvpTransformed: true,
    });

    // Dispose the wrapper jaxpr (not cached)
    // Note: jvpBody is cached via jvpJaxprCache, so we don't dispose it
    wrapperJaxpr.dispose();

    // Results layout from wrapper: [carryP..., carryT..., yP..., yT...]
    const carryOutP = results.slice(0, numCarry);
    const carryOutT = results.slice(numCarry, numCarry * 2);
    const ysP = results.slice(numCarry * 2, numCarry * 2 + numY);
    const ysT = results.slice(numCarry * 2 + numY);

    const primalsOut = [...carryOutP, ...ysP];
    const tangentsOut = [...carryOutT, ...ysT];

    return [primalsOut, tangentsOut];
  },

  [Primitive.AssociativeScan](
    primals,
    tangents,
    { jaxpr, numLeaves, axis, reverse },
  ) {
    // JVP of associativeScan: double the body to compute both primals and tangents
    // in a single associativeScan with 2× numLeaves.
    //
    // Original body: (consts, a, b) -> result   (each group has numLeaves arrays)
    // JVP body from jvpJaxpr expects: [primals..., tangents...] order
    //   i.e., [consts, a, b, consts_dot, a_dot, b_dot] -> [result, result_dot]
    //
    // But doubled associativeScan feeds body in scan order:
    //   [constsP, constsT, aP, aT, bP, bT]
    //
    // Wrapper reorders: scan order -> jvp order for inputs.
    // Outputs [resultP, resultT] are already in correct scan order.

    const numConsts = primals.length - numLeaves;

    // Transform body jaxpr for JVP
    const jvpBody = jvpJaxpr(jaxpr);
    const numJvpConsts = jvpBody.constEnv.length;
    const numBodyInputs = numConsts + numLeaves * 2; // consts + a + b

    // Get avals in JVP order (primals then tangents)
    const jvpOrderAvals = jvpBody.jaxpr.inBinders
      .slice(numJvpConsts)
      .map((v) => v.aval);

    // Split JVP-order avals into groups
    const constsP_avals = jvpOrderAvals.slice(0, numConsts);
    const aP_avals = jvpOrderAvals.slice(numConsts, numConsts + numLeaves);
    const bP_avals = jvpOrderAvals.slice(numConsts + numLeaves, numBodyInputs);
    const constsT_avals = jvpOrderAvals.slice(
      numBodyInputs,
      numBodyInputs + numConsts,
    );
    const aT_avals = jvpOrderAvals.slice(
      numBodyInputs + numConsts,
      numBodyInputs + numConsts + numLeaves,
    );
    const bT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numLeaves);

    // Wrapper in-avals in scan order: [constsP, constsT, aP, aT, bP, bT]
    const wrapperInAvals = [
      ...constsP_avals,
      ...constsT_avals,
      ...aP_avals,
      ...aT_avals,
      ...bP_avals,
      ...bT_avals,
    ];

    const { jaxpr: wrapperJaxpr } = makeJaxpr(
      (...scanOrderArgs: Tracer[]): Tracer[] => {
        // scanOrderArgs layout: [constsP, constsT, aP, aT, bP, bT]
        const cP = scanOrderArgs.slice(0, numConsts);
        const cT = scanOrderArgs.slice(numConsts, numConsts * 2);
        const aP = scanOrderArgs.slice(
          numConsts * 2,
          numConsts * 2 + numLeaves,
        );
        const aT = scanOrderArgs.slice(
          numConsts * 2 + numLeaves,
          numConsts * 2 + numLeaves * 2,
        );
        const bP = scanOrderArgs.slice(
          numConsts * 2 + numLeaves * 2,
          numConsts * 2 + numLeaves * 2 + numLeaves,
        );
        const bT = scanOrderArgs.slice(
          numConsts * 2 + numLeaves * 2 + numLeaves,
        );

        // Reorder to jvp order: [constsP, aP, bP, constsT, aT, bT]
        const jvpOrderArgs = [...cP, ...aP, ...bP, ...cT, ...aT, ...bT];

        // Inline the jvpBody jaxpr via evalJaxpr instead of bind(Primitive.Jit).
        // Using Primitive.Jit here would JIT-compile the body with the traced
        // element shapes (e.g., scalar []), producing shape-specialized kernels
        // that return scalar outputs even when the impl rule feeds batched
        // inputs ([N-stride, ...]). evalJaxpr inlines the equations into the
        // wrapper jaxpr, preserving shape-polymorphic evaluation.
        //
        // evalJaxpr is non-consuming — consts stay alive (cache-owned via
        // jvpJaxprCache), no .ref needed.
        const jvpOutputs = evalJaxpr(jvpBody.jaxpr, [
          ...jvpBody.constEnv.values,
          ...jvpOrderArgs,
        ]);

        // Outputs: [resultP..., resultT...] — already in correct scan order
        return jvpOutputs;
      },
    )(...wrapperInAvals);

    // Build doubled primitive args
    const constsP = primals.slice(0, numConsts);
    const elemsP = primals.slice(numConsts);
    const constsT = tangents.slice(0, numConsts);
    const elemsT = tangents.slice(numConsts);

    const results = bind(
      Primitive.AssociativeScan,
      [...wrapperJaxpr.consts, ...constsP, ...constsT, ...elemsP, ...elemsT],
      {
        jaxpr: wrapperJaxpr.jaxpr,
        numLeaves: numLeaves * 2,
        axis,
        reverse,
      },
    );

    // Dispose the wrapper jaxpr (not cached)
    // Note: jvpBody is cached via jvpJaxprCache, so we don't dispose it
    wrapperJaxpr.dispose();

    // Results: [resultP_0..nL-1, resultT_0..nL-1]
    const primalsOut = results.slice(0, numLeaves);
    const tangentsOut = results.slice(numLeaves);

    return [primalsOut, tangentsOut];
  },
  [Primitive.BlockMap](primals, tangents, params) {
    const { jaxpr, numConsts, blockShape, inAxes, outAxes, numInputs } = params;
    const jvpBody = jvpJaxpr(jaxpr);

    const constsP = primals.slice(0, numConsts);
    const constsT = tangents.slice(0, numConsts);
    const inputsP = primals.slice(numConsts);
    const inputsT = tangents.slice(numConsts);

    const doubleAxes = (axes: (number | null)[][]) => [...axes, ...axes];
    const jvpInAxes = doubleAxes(inAxes);
    const jvpOutAxes = doubleAxes(outAxes);

    // jvpBody expects [jvpConsts, allPrimals, allTangents] where
    // allPrimals = [constsP, inputsP] and allTangents = [constsT, inputsT].
    // BlockMap groups as [consts (not tiled), inputs (tiled)], so we must
    // wrap the jvpBody to remap [constsP, constsT, inputsP, inputsT] →
    // [constsP, inputsP, constsT, inputsT] before calling jvpBody.
    const nJvp = jvpBody.constEnv.length;
    const nC = numConsts;
    const nI = numInputs;
    const wrapperInAvals = [
      ...jvpBody.jaxpr.inBinders.slice(0, nJvp).map((v) => v.aval),
      ...constsP.map((_, i) => jaxpr.inBinders[i].aval),
      ...constsT.map((_, i) => jaxpr.inBinders[i].aval),
      ...inputsP.map((_, i) => jaxpr.inBinders[numConsts + i].aval),
      ...inputsT.map((_, i) => jaxpr.inBinders[numConsts + i].aval),
    ];
    const { jaxpr: wrappedBody } = makeJaxpr(
      (...args: Tracer[]): Tracer[] => {
        // args order: [jvpConsts, constsP, constsT, inputsP, inputsT]
        const jc = args.slice(0, nJvp);
        const cP = args.slice(nJvp, nJvp + nC);
        const cT = args.slice(nJvp + nC, nJvp + nC * 2);
        const iP = args.slice(nJvp + nC * 2, nJvp + nC * 2 + nI);
        const iT = args.slice(nJvp + nC * 2 + nI);
        // jvpBody expects: [jvpConsts, constsP, inputsP, constsT, inputsT]
        return evalJaxpr(jvpBody.jaxpr, [...jc, ...cP, ...iP, ...cT, ...iT]);
      },
      { validateRefs: false },
    )(...wrapperInAvals);

    const doubledConsts = [...wrappedBody.consts, ...constsP, ...constsT];

    const doubledOut = bind(
      Primitive.BlockMap,
      [...doubledConsts, ...inputsP, ...inputsT],
      {
        jaxpr: wrappedBody.jaxpr,
        numConsts: doubledConsts.length,
        numInputs: numInputs * 2,
        blockShape,
        inAxes: jvpInAxes,
        outAxes: jvpOutAxes,
        threadTile: params.threadTile,
        halo: params.halo ? [...params.halo, ...params.halo] : undefined,
        isJvpTransformed: true,
        originalJaxpr: jaxpr,
        originalNumConsts: numConsts,
      },
    );

    // Ownership boundary: the wrapper ClosedJaxpr is local/transferred state
    // used only to build the Primitive.BlockMap params. Once bind() returns,
    // any tracing/capture that needed wrappedBody.consts has taken its own
    // retained ownership. Only wrappedBody.jaxpr participates in later
    // transpose-cache lookups; the wrapper's builder-owned const handles must
    // be balanced locally instead of being treated as cache-owned state.
    wrappedBody.dispose();

    const numOutP = doubledOut.length / 2;
    return [doubledOut.slice(0, numOutP), doubledOut.slice(numOutP)];
  },
  [Primitive.WorkgroupAssociativeScan](
    primals,
    tangents,
    { jaxpr, numConsts },
  ) {
    // JVP of WorkgroupAssociativeScan: double the body like AssociativeScan.
    // Original body: (consts, a, b) -> result
    // JVP body expects: [constsP, aP, bP, constsT, aT, bT] -> [resultP, resultT]
    // Doubled scan feeds: [constsP, constsT, aP, aT, bP, bT]
    // Wrapper reorders scan order -> jvp order.

    const numElems = primals.length - numConsts;
    const jvpBody = jvpJaxpr(jaxpr);
    const numJvpConsts = jvpBody.constEnv.length;
    const numBodyInputs = numConsts + numElems * 2; // consts + a + b

    const jvpOrderAvals = jvpBody.jaxpr.inBinders
      .slice(numJvpConsts)
      .map((v) => v.aval);

    const constsP_avals = jvpOrderAvals.slice(0, numConsts);
    const aP_avals = jvpOrderAvals.slice(numConsts, numConsts + numElems);
    const bP_avals = jvpOrderAvals.slice(numConsts + numElems, numBodyInputs);
    const constsT_avals = jvpOrderAvals.slice(
      numBodyInputs,
      numBodyInputs + numConsts,
    );
    const aT_avals = jvpOrderAvals.slice(
      numBodyInputs + numConsts,
      numBodyInputs + numConsts + numElems,
    );
    const bT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numElems);

    const wrapperInAvals = [
      ...constsP_avals,
      ...constsT_avals,
      ...aP_avals,
      ...aT_avals,
      ...bP_avals,
      ...bT_avals,
    ];

    const { jaxpr: wrapperJaxpr } = makeJaxpr(
      (...scanOrderArgs: Tracer[]): Tracer[] => {
        // scanOrderArgs: [constsP, constsT, aP, aT, bP, bT]
        const cP = scanOrderArgs.slice(0, numConsts);
        const cT = scanOrderArgs.slice(numConsts, numConsts * 2);
        const aP = scanOrderArgs.slice(numConsts * 2, numConsts * 2 + numElems);
        const aT = scanOrderArgs.slice(
          numConsts * 2 + numElems,
          numConsts * 2 + numElems * 2,
        );
        const bP = scanOrderArgs.slice(
          numConsts * 2 + numElems * 2,
          numConsts * 2 + numElems * 2 + numElems,
        );
        const bT = scanOrderArgs.slice(numConsts * 2 + numElems * 2 + numElems);

        const jvpOrderArgs = [...cP, ...aP, ...bP, ...cT, ...aT, ...bT];
        return evalJaxpr(jvpBody.jaxpr, [
          ...jvpBody.constEnv.values,
          ...jvpOrderArgs,
        ]);
      },
    )(...wrapperInAvals);

    const constsP = primals.slice(0, numConsts);
    const elemsP = primals.slice(numConsts);
    const constsT = tangents.slice(0, numConsts);
    const elemsT = tangents.slice(numConsts);

    const results = bind(
      Primitive.WorkgroupAssociativeScan,
      [...wrapperJaxpr.consts, ...constsP, ...constsT, ...elemsP, ...elemsT],
      {
        jaxpr: wrapperJaxpr.jaxpr,
        numConsts: wrapperJaxpr.consts.length + numConsts * 2,
      },
    );

    wrapperJaxpr.dispose();

    const primalsOut = results.slice(0, numElems);
    const tangentsOut = results.slice(numElems);
    return [primalsOut, tangentsOut];
  },
  [Primitive.DynamicSlice](primals, tangents, { sliceSizes }) {
    const operandT = tangents[0];
    const startsP = primals.slice(1);
    const outP = bind1(Primitive.DynamicSlice, primals, { sliceSizes });
    const outT = bind1(Primitive.DynamicSlice, [operandT, ...startsP], {
      sliceSizes,
    });
    return [[outP], [outT]];
  },
  [Primitive.UncheckedDynamicSlice](primals, tangents, { sliceSizes }) {
    const operandT = tangents[0];
    const startsP = primals.slice(1);
    const outP = bind1(Primitive.UncheckedDynamicSlice, primals, {
      sliceSizes,
    });
    const outT = bind1(
      Primitive.UncheckedDynamicSlice,
      [operandT, ...startsP],
      {
        sliceSizes,
      },
    );
    return [[outP], [outT]];
  },
  [Primitive.ForiLoop](primals, tangents, { jaxpr, numConsts, lower, upper }) {
    const numCarry = primals.length - numConsts;
    const jvpBody = jvpJaxpr(jaxpr);

    // Original jaxpr expects [consts, i, carry]
    const dummyI = new ShapedArray([], DType.Int32, false);

    const { jaxpr: wrapperClosedJaxpr } = makeJaxpr((...args: Tracer[]) => {
      // args: [constsP, constsT, i, carryP, carryT]
      const constsP = args.slice(0, numConsts);
      const constsT = args.slice(numConsts, numConsts * 2);
      const i = args[numConsts * 2];
      const carryP = args.slice(
        numConsts * 2 + 1,
        numConsts * 2 + 1 + numCarry,
      );
      const carryT = args.slice(numConsts * 2 + 1 + numCarry);

      // jvpBody expects: [...jvpConsts, constsP, iP, carryP, constsT, iT, carryT]
      const jvpConsts = jvpBody.constEnv.values;
      const iP = i;
      const iT = zerosLike(i);

      const callArgs = [
        ...jvpConsts,
        ...constsP,
        iP,
        ...carryP,
        ...constsT,
        iT,
        ...carryT,
      ];
      const result = evalJaxpr(jvpBody.jaxpr, callArgs);

      // iT is the zero tangent for the loop counter — always dead in the
      // JVP'd body (loop counters don't participate in differentiation).
      // If the JaxprBuilder never captured it, its creation ref remains.
      // Dispose explicitly to prevent leak.
      if (iT instanceof Array) {
        iT.dispose();
      }

      return result;
    })(
      ...primals.slice(0, numConsts),
      ...tangents.slice(0, numConsts),
      dummyI,
      ...primals.slice(numConsts),
      ...tangents.slice(numConsts),
    );

    const scanArgs = [
      ...wrapperClosedJaxpr.consts,
      ...primals.slice(0, numConsts),
      ...tangents.slice(0, numConsts),
      ...primals.slice(numConsts),
      ...tangents.slice(numConsts),
    ];

    const results = bind(Primitive.ForiLoop, scanArgs, {
      jaxpr: wrapperClosedJaxpr.jaxpr,
      numConsts: wrapperClosedJaxpr.consts.length + numConsts * 2,
      lower,
      upper,
      isJvpTransformed: true,
    });

    // We created an anonymous jaxpr, its consts are bound as primals to the outer trace/eager. We must dispose the wrapper to balance refcounts.
    wrapperClosedJaxpr.dispose();

    return [results.slice(0, numCarry), results.slice(numCarry)];
  },
};

const jvpJaxprCacheByBackend = new Map<
  Jaxpr,
  Map<string, CachedJaxprArtifact>
>();

// Register for cleanup during checkLeaks.stop() to avoid leaking
// ClosedJaxpr consts (e.g., zerosLike tangents) across test boundaries.
_registerJitCacheDisposer(() => {
  for (const inner of jvpJaxprCacheByBackend.values()) {
    for (const artifact of inner.values()) artifact.constEnv.dispose();
  }
  jvpJaxprCacheByBackend.clear();
});
_registerCacheSizeGetter("jvpJaxpr", () => jvpJaxprCacheByBackend.size);

function jvpJaxpr(jaxpr: Jaxpr): CachedJaxprArtifact {
  // Include backend type: consts in the cached const environment (e.g. zero tangents)
  // are concrete arrays on whichever device was active at first JVP. Cross-device
  // reuse would read stale data from the wrong backend.
  const backendKey = defaultDevice();
  const cached = jvpJaxprCacheByBackend.get(jaxpr)?.get(backendKey);
  if (cached) return cached;

  // Note: Following the implementation in Autodidax, consts in the Jaxpr become
  // real inputs after JVP transformation, since they are part of the primals
  // and the JVP rule takes in [primals, tangents] as a pair.
  //
  // This is also why we can ignore `numConsts` in the JVP rule. Anyway, this
  // only happens in jvp-of-jit cases, where you understandably have to
  // sacrifice some performance versus wrapping jit() outside.
  const inAvals = jaxpr.inBinders.map((v) => v.aval);
  const artifact = buildCachedJaxprArtifact(
    makeJaxpr((primals: Tracer[], tangents: Tracer[]) =>
      jvpFlat(jaxprAsFun(jaxpr), primals, tangents),
    )(inAvals, inAvals),
  );

  let inner = jvpJaxprCacheByBackend.get(jaxpr);
  if (!inner) {
    inner = new Map();
    jvpJaxprCacheByBackend.set(jaxpr, inner);
  }
  inner.set(backendKey, artifact);
  return artifact;
}

function jvpFlat(
  f: (...x: Tracer[]) => TracerValue[],
  primals: TracerValue[],
  tangents: TracerValue[],
): [Tracer[], Tracer[]] {
  const jvpData: JvpGlobalData = {
    intermediates: [],
    liftedTangents: [],
  };
  using main = newMain(JVPTrace, jvpData);
  main.isAbstract = true;
  const trace = new JVPTrace(main);
  // Track arrays newly created by pureArray from raw values (e.g., scalar 3 → Array).
  // These are not in intermediateTracers (only processPrimitive adds there)
  // and must be disposed separately to avoid leaks.
  const newlyCreatedInputs: Tracer[] = [];
  const tracersIn = zip(primals, tangents).map(([x, t]) => {
    const px = pureArray(x);
    const pt = pureArray(t);
    if (px !== x) newlyCreatedInputs.push(px);
    if (pt !== t) newlyCreatedInputs.push(pt);
    return new JVPTracer(trace, px, pt);
  });
  const outs = f(...tracersIn);
  const tracersOut = outs.map((out) => fullRaise(trace, out) as JVPTracer);
  const result: [Tracer[], Tracer[]] = unzip2(
    tracersOut.map((t) => [t.primal, t.tangent]),
  );
  // Dispose intermediate JVPTracers' primals and tangents that were created
  // during function body execution but are not in the output. Uses
  // [Symbol.dispose] which is a no-op on abstract tracers (safe for all contexts).
  // The refCount > 0 guard prevents double-free when nested jvp calls
  // (e.g., deriv(deriv(f))) dispose intermediates via user code before
  // the outer jvpFlat's cleanup runs.
  // Skip cleanup when an abstract trace (PE, JaxprTrace, outer JVP) is below
  // this JVP on the stack — those traces manage lifetimes and own the values.
  // Cleanup IS safe when only BatchTrace is below (vmap(jvp(...))).
  if (!hasAbstractTraceBelow(main.level)) {
    const outputSet = new Set<JVPTracer>(tracersOut);
    for (const t of jvpData.intermediates) {
      if (!outputSet.has(t)) {
        if (t.primal.refCount > 0) t.primal[Symbol.dispose]();
        if (t.tangent.refCount > 0) t.tangent[Symbol.dispose]();
      }
    }
    // Dispose arrays created from raw values for input primals/tangents.
    // These are anonymous (created by pureArray, not user-owned) and are
    // not tracked by jvpData.intermediates. Skip arrays that appear in the
    // output (identity function case: output IS the input primal/tangent).
    const outputArrays = new Set<Tracer>([...result[0], ...result[1]]);
    for (const a of newlyCreatedInputs) {
      if (!outputArrays.has(a) && a.refCount > 0) {
        a[Symbol.dispose]?.();
      }
    }
  }
  // Dispose zero tangents created by JVPTrace.lift() for lifted inputs.
  // These are freshly created by the JVP trace (not owned by PE or other
  // abstract traces), so they must be cleaned up unconditionally.
  // Use .dispose() directly so this cleanup does not depend on scope-exit
  // semantics or wrapper-level disposal hooks.
  {
    const outputTangents = new Set<Tracer>(result[1]);
    for (const z of jvpData.liftedTangents) {
      if (!outputTangents.has(z) && z.refCount > 0) {
        z.dispose();
      }
    }
  }
  return result;
}

export function jvp<F extends (...x: any[]) => any>(
  f: F,
  primals: JsTree<TracerValue>[],
  tangents: JsTree<TracerValue>[],
  { hasAux = false } = {},
): [any, any, any?] {
  // --- custom_jvp fast path ---
  // If `f` was created by `customJvp(fn, jvpRule)`, call the user-supplied
  // JVP rule directly instead of tracing through `fn`.
  const cjd = getCustomJvpDef(f) as { fn: any; jvpRule: any } | undefined;
  if (cjd) {
    if (hasAux) {
      throw new Error("customJvp functions do not support hasAux in jvp()");
    }
    const flatInTree = getCustomJvpFlatInTree(f);
    const flatOutTreeStore = getCustomJvpOutTreeStore(f);
    if (flatInTree) {
      if (flatOutTreeStore === undefined) {
        throw new Error(
          "internal: flattened customJvp function missing outTree store",
        );
      }
      // Flat calling convention: use the custom rule once, then populate the
      // flattened wrapper's outTree store from the primal result.
      const primalsUser = treeUnflatten(flatInTree, primals);
      const tangentsUser = treeUnflatten(flatInTree, tangents);
      const [primalOutUser, tangentOutUser] = cjd.jvpRule(
        primalsUser,
        tangentsUser,
      );
      const [primalOutFlat, outTree] = treeFlatten(primalOutUser);
      flatOutTreeStore.value = outTree;
      const [tangentOutFlat, tangentTree] = treeFlatten(tangentOutUser);
      if (!outTree.equals(tangentTree)) {
        throw new TreeMismatchError("customJvp", outTree, tangentTree);
      }
      return [primalOutFlat, tangentOutFlat];
    }
    // Direct call: jvp(customJvpFn, primals, tangents)
    const [primalOut, tangentOut] = cjd.jvpRule(primals, tangents);
    return [primalOut, tangentOut];
  }

  if (getCustomVjpDef(f)) {
    throw new Error(
      "Function has a customVjp but was differentiated in forward-mode (jvp/jacfwd). " +
        "customVjp only defines reverse-mode rules. Implement customJvp for this path.",
    );
  }

  const [primalsFlat, inTree] = treeFlatten(primals);
  const [tangentsFlat, inTree2] = treeFlatten(tangents);
  if (!inTree.equals(inTree2)) {
    throw new TreeMismatchError("jvp", inTree, inTree2);
  }

  let flatFun, outTree, aux;
  if (hasAux) {
    [flatFun, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [flatFun, outTree] = flattenFun(f, inTree);
  }

  const [primalsOutFlat, tangentsOutFlat] = jvpFlat(
    flatFun,
    primalsFlat,
    tangentsFlat,
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in jvp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const tangentsOut = treeUnflatten(outTree.value, tangentsOutFlat);

  if (hasAux) {
    return [primalsOut, tangentsOut, lowerAux(aux!.value)];
  }
  return [primalsOut, tangentsOut];
}

/** Lowering for auxiliary data returned in `hasAux: true` methods. */
export function lowerAux(aux: any): any {
  const level = currentTraceLevel();

  return treeMap((x: Tracer) => {
    if (x instanceof Tracer) {
      while (x._trace.main.level > level) {
        if (x instanceof JVPTracer) {
          x = x.primal;
        } else {
          const y = x.fullLower();
          if (y._trace.main.level >= x._trace.main.level)
            throw new Error("internal: lowerAux did not reduce trace level");
          x = y;
        }
      }
    }
    return x;
  }, aux);
}
