/** @file Implementations of vjp() and partial evaluation. */
// jax-js-lint: allow-ref — .ref is the core ownership mechanism in autodiff internals

import { AluOp, type DType, isFloatDtype } from "../alu";
import { defaultDevice } from "../backend";
import { type Dim, dimSub, hasSymbolicDims } from "../dim";
import { jvp, lowerAux } from "./jvp";
import {
  dispose as treeDispose,
  flatten as treeFlatten,
  map as treeMap,
  unflatten as treeUnflatten,
} from "../tree";
import {
  checkInts,
  DEBUG,
  deepEqual,
  generalBroadcast,
  invertPermutation,
  partitionList,
  range,
  toposort,
  unzip2,
} from "../utils";
import {
  array,
  eye,
  fullInternal,
  Array as JaxArray,
  onesLike,
  pureArray,
  zerosLike,
} from "./array";
import { aotLinearize } from "./artifacts";
import {
  _registerCacheSizeGetter,
  _registerJitCacheDisposer,
} from "./check-leaks";
import {
  AbstractValue,
  add,
  bind,
  broadcast,
  cast,
  concatenate,
  conv,
  currentTraceLevel,
  dynamicUpdateSlice,
  dynamicUpdateSliceGeneral,
  flattenFun,
  flattenFunWithAux,
  flip,
  fullRaise,
  gather,
  getCustomVjpDef,
  inMakeJaxprBody,
  insideAbstractTrace,
  max,
  min,
  mul,
  ndim,
  neg,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  reduce,
  reshape,
  reverse,
  scatterAdd,
  setCustomJvpDef,
  setCustomVjpDef,
  ShapedArray,
  shrink,
  split,
  stopGradient,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  triSolve,
  UseAfterFreeError,
  where,
  withPartialEvalScope,
} from "./core";
import {
  abstractEvalRules,
  buildCachedJaxprArtifact,
  CachedJaxprArtifact,
  ClosedJaxpr,
  evalJaxpr,
  Jaxpr,
  JaxprEqn,
  Lit,
  makeJaxpr,
  OwnedFunction,
  typecheckJaxpr,
  Var,
} from "./jaxpr";
import { analyzeLinearStencil, reverseStencilSlice } from "./stencil-analysis";
import { jacfwd, moveaxis, vmap } from "./vmap";

/** Internal zeros allocation (marking handled by fullInternal).
 * For symbolic shapes during tracing, broadcasts a scalar zero. */
const zerosInternal = (shape: Dim[] | number[], dtype: DType) => {
  if (hasSymbolicDims(shape as Dim[])) {
    const scalar = fullInternal({ shape: [], dtype, weakType: false }, 0);
    const axis = range(shape.length);
    return broadcast(scalar, shape as Dim[], axis);
  }
  return fullInternal({ shape, dtype, weakType: false }, 0);
};

/** Array value that can either be known or unknown. */
class PartialVal {
  constructor(
    readonly val: Tracer | null,
    readonly aval: ShapedArray,
  ) {}

  static known(val: Tracer): PartialVal {
    return new PartialVal(val, ShapedArray.fromAval(val.aval));
  }

  static unknown(aval: AbstractValue): PartialVal {
    return new PartialVal(null, ShapedArray.fromAval(aval));
  }

  get isKnown(): boolean {
    return this.val !== null;
  }

  toString(): string {
    return this.val ? this.val.toString() : this.aval.toString();
  }
}

/**
 * Phase 1 of partial evaluation: trace `f` under a PartialEvalTrace and
 * produce a forward jaxpr together with PE intermediates for ownership disposal.
 *
 * This is the core of partial evaluation. The returned intermediates must be
 * disposed by the caller via `disposePeIntermediates(...)` to balance rc=1
 * from creation.
 *
 * Extracted as the reusable forward PE builder in M1.1 to enable composition with
 * `buildBackwardJaxpr` and future artifact types.
 */
function buildForwardJaxpr(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): {
  jaxpr: ClosedJaxpr;
  pvalsOut: PartialVal[];
  peIntermediates: Tracer[];
} {
  let jaxpr: ClosedJaxpr;
  let pvalsOut: PartialVal[];
  let knownIntermediates: Tracer[];
  {
    using main = newMain(PartialEvalTrace);
    main.isAbstract = true;
    const trace = new PartialEvalTrace(main);
    const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
    const unknownTracersIn = tracersIn
      .filter((t) => !t.pval.isKnown)
      .map((t) => t.ref);

    const peCreatedArrays: Tracer[] = [];
    let outs: any;
    try {
      outs = withPartialEvalScope(peCreatedArrays, () => f(...tracersIn));
    } catch (e) {
      for (const t of peCreatedArrays) {
        try {
          t.dispose();
        } catch {
          /* already disposed */
        }
      }
      throw e;
    }

    const tracersOut: PartialEvalTracer[] = outs.map((out: TracerValue) =>
      fullRaise(trace, out),
    );

    pvalsOut = tracersOut.map((t) => t.pval);
    const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
    jaxpr = partialEvalGraphToJaxpr(unknownTracersIn, unknownTracersOut);

    // Dispose unreachable Const PETracers: instantiateConst calls .ref on the
    // underlying value for each Const PETracer it creates. The toposort in
    // partialEvalGraphToJaxpr disposes reachable Const PETracers (balancing
    // the .ref). But Const PETracers from unreachable computations (e.g. aux
    // branches in hasAux) are never processed, leaving dangling .ref calls.
    // Dispose any Const PETracer not already disposed by the toposort.
    //
    // In deeply nested transform stacks, a Const PETracer can occasionally
    // resolve to an already-disposed underlying value (owned and released by
    // an outer cleanup path). Keep draining the PETracer wrappers while
    // tolerating those stale-value cases.
    for (const ct of trace.allConstPETracers) {
      if (!ct.isAlive) continue;
      try {
        ct.dispose();
      } catch {
        // Already-disposed underlying value in nested transform cleanup.
      }
    }

    knownIntermediates = trace.knownIntermediates;
    knownIntermediates.push(...peCreatedArrays);
  }

  return {
    jaxpr,
    pvalsOut,
    peIntermediates: knownIntermediates,
  };
}

/**
 * Unwrap tracer wrappers (BatchTracer, etc.) to find the underlying concrete
 * Array. Chases the `.val` chain until it reaches a non-wrapper value.
 * Returns the input unchanged if it's already a concrete Array.
 */
function unwrapToConcreteArray(t: Tracer): Tracer {
  let current: any = t;
  while (current && typeof current === "object" && "val" in current) {
    const inner = current.val;
    if (inner === current || !(inner instanceof Tracer)) break;
    current = inner;
  }
  return current;
}

/**
 * Dispose PE intermediates that aren't protected (outputs, aux, retained consts).
 *
 * During PE, all-known evaluations create concrete arrays. Output values
 * (in protectedVals) are returned to the caller. ClosedJaxpr consts have
 * independent ownership via .ref in partialEvalGraphToJaxpr. Everything
 * else (pure intermediates AND computed consts) needs one dispose here
 * to balance the rc=1 from creation.
 */
export function disposePeIntermediates(
  peIntermediates: Tracer[],
  protectedVals: Set<Tracer>,
): void {
  const allProtected = new Set<Tracer>(protectedVals);
  for (const v of protectedVals) {
    const concrete = unwrapToConcreteArray(v);
    if (concrete !== v) allProtected.add(concrete);
  }
  const disposed = new Set<Tracer>();
  const diagSeen = DEBUG >= 1 ? new Set<Tracer>() : undefined;
  const disposedForDiag: string[] = [];
  const hasOuterAbstractTrace = insideAbstractTrace();
  const allowNestedBodyCleanup = hasOuterAbstractTrace && inMakeJaxprBody();
  for (const t of peIntermediates) {
    if (allProtected.has(t)) continue;
    if (disposed.has(t)) continue;
    if (!t.isAliveForCleanup()) continue;
    const concrete = unwrapToConcreteArray(t);
    if (concrete !== t && allProtected.has(concrete)) continue;
    if (concrete !== t && !concrete.isAliveForCleanup()) continue;

    // Nested grad/hessian inside traced control flow runs PE cleanup while an
    // outer abstract trace is still active. In that case, disposing tracer
    // wrappers can cascade into outer-owned values. Restrict cleanup to the
    // concrete arrays that were actually born in this PE scope.
    if (allowNestedBodyCleanup) {
      if (!(concrete instanceof JaxArray)) continue;
      if (!concrete.hasPartialEvalCreationRef) continue;
      disposed.add(t);
      disposed.add(concrete);
      if (diagSeen && !diagSeen.has(concrete)) {
        diagSeen.add(concrete);
        disposedForDiag.push(concrete.toString());
      }
      try {
        concrete.dispose();
      } catch {
        // Nested cleanup may race with an already-balanced path.
      }
      continue;
    }

    if (hasOuterAbstractTrace) continue;

    disposed.add(t);
    if (concrete !== t) disposed.add(concrete); // eslint-disable-line jax-js/no-use-after-dispose -- false positive: dispose is behind `continue`
    /* eslint-disable jax-js/no-use-after-dispose -- false positive: earlier concrete.dispose() is in a `continue` branch that never reaches here */
    if (diagSeen && !diagSeen.has(concrete)) {
      diagSeen.add(concrete);
      const desc =
        concrete instanceof JaxArray ? concrete.toString() : "tracer";
      disposedForDiag.push(desc);
    }
    /* eslint-enable jax-js/no-use-after-dispose */
    try {
      t.dispose();
    } catch {
      // Hardening for stale nested-transform wrappers. Normal cleanup should
      // have been filtered by explicit protection and liveness checks above.
    }
  }
  if (DEBUG >= 1 && disposedForDiag.length > 0) {
    console.warn(
      `[jax-js] grad/vjp rescued ${disposedForDiag.length} PE intermediate(s) ` +
        `that would leak in eager mode: ${disposedForDiag.join(", ")}. ` +
        `Extract to \`using\` bindings for eager/jit parity.`,
    );
  }
}

/**
 * Extract concrete Arrays from a value tree that may contain JVPTracers,
 * PartialEvalTracers, or other wrappers. Uses lowerAux to walk through
 * the tracer chain (JVPTracer→PETracer→concrete Array).
 */
export function collectConcreteArrays(value: any): Tracer[] {
  const result: Tracer[] = [];
  const lowered = lowerAux(value);
  treeMap((x: any) => {
    if (x instanceof Tracer) result.push(x);
    return x;
  }, lowered);
  return result;
}

export function flushPendingConcreteArrays(arrays: Tracer[]): void {
  if (insideAbstractTrace()) return;
  for (const arr of arrays) {
    if (arr instanceof JaxArray) {
      arr._flushPendingSync();
    }
  }
}

/** Result of linearizeFlatUtil: forward jaxpr + primals + PE cleanup state. */
export interface ForwardResult {
  primalsOut: Tracer[];
  jaxpr: ClosedJaxpr;
  peIntermediates: Tracer[];
}

/**
 * Helper function with shared Jaxpr logic between linearize and vjp.
 *
 * Internally, vjp() looks very similar to linearize() but returns a function
 * evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
 * of pushing forward tangents.
 *
 * Uses `buildForwardJaxpr` (M1.1) and returns the PE intermediate list for
 * deterministic cleanup.
 */
export function linearizeFlatUtil(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): ForwardResult {
  const pvalsIn = [
    ...primalsIn.map(PartialVal.known),
    ...primalsIn.map((t) => PartialVal.unknown(t.aval)),
  ];
  const fJvp = (...x: Tracer[]) => {
    // Args contain both primals and tangents, concatenated.
    const k = x.length / 2;
    const [primalsOut, tangentsOut] = jvp(f, x.slice(0, k), x.slice(k, 2 * k));
    return [...primalsOut, ...tangentsOut];
  };
  const { jaxpr, pvalsOut, peIntermediates } = buildForwardJaxpr(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new Error("Not all primal values are known after partial evaluation");
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  return { primalsOut, jaxpr, peIntermediates };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
  auxStore?: { value: any },
): [Tracer[], (...args: Tracer[]) => Tracer[], () => void] {
  // AOT linearize: trace f, build forward jaxpr, dispose PE intermediates.
  // skipBackward: linearize only needs the forward jaxpr, not the backward.
  // Pass auxStore ref (not pre-computed arrays) — auxStore.value is only
  // populated during linearizeFlatUtil, which runs inside aotLinearize.
  const { primal, pullback } = aotLinearize(f, primalsIn, {
    auxStore,
    skipBackward: true,
  });

  // Pullback is unused for linearize — dispose immediately (no-op stub).
  pullback[Symbol.dispose]();

  // Get primal outputs and residuals; residuals are only needed to keep the
  // forward jaxpr consts alive until fLin is done — dispose them in the
  // outer dispose function.
  const { primalsOut, residuals } = primal.run(primalsIn);

  // fLin evaluates the forward jaxpr with tangent inputs (same as before).
  // evalJaxpr is non-consuming — consts stay alive, owned by forwardJaxpr.
  const forwardJaxpr = primal.forwardJaxpr;
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(forwardJaxpr.jaxpr, [...forwardJaxpr.consts, ...tangents]);

  // Dispose residuals + primal artifact (which owns the forward jaxpr).
  const dispose = () => {
    residuals[Symbol.dispose]();
    primal[Symbol.dispose]();
  };

  return [primalsOut, fLin, dispose];
}

export function linearize(
  f: (...primals: any[]) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...tangents: any[]) => any>, any?] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  // Wrap scalar primals to Arrays; track which are newly created for disposal.
  const wrappedPrimals = primalsInFlat.map(pureArray);
  const [primalsOutFlat, fLinFlat, dispose] = linearizeFlat(
    fFlat,
    wrappedPrimals,
    hasAux ? aux : undefined,
  );
  // Dispose newly-created pureArray wrappers. After linearizeFlat returns, the
  // wrappers are only used for .aval (shape/dtype metadata), which is safe
  // to read after disposal. Skip wrappers that appear in primalsOutFlat
  // (identity function case: output IS the input primal).
  if (!insideAbstractTrace()) {
    const primalsOutSet = new Set(primalsOutFlat);
    for (let i = 0; i < wrappedPrimals.length; i++) {
      if (
        wrappedPrimals[i] !== primalsInFlat[i] &&
        !primalsOutSet.has(wrappedPrimals[i])
      ) {
        wrappedPrimals[i].dispose();
      }
    }
  }
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = ((...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    // Wrap tangents as Arrays. evalJaxpr is non-consuming, so we must
    // dispose the wrappers afterward. Pass-through outputs (where the
    // wrapper IS the output) are protected by evalJaxpr's .ref on
    // pass-through results.
    const tangentWrappers = tangentsInFlat.map(pureArray);
    const tangentsOutFlat = fLinFlat(...tangentWrappers);
    for (const w of tangentWrappers) w.dispose();
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  }) as OwnedFunction<(...tangents: any[]) => any>;
  fLin.dispose = dispose;
  fLin[Symbol.dispose] = dispose;
  if (hasAux) {
    return [primalsOut, fLin, lowerAux(aux!.value)];
  }
  return [primalsOut, fLin];
}

// Used in PartialEvalTracer to track recipes for "unknown" partial vals.
type JaxprRecipe =
  | {
      type: "LambdaBinding";
    }
  | {
      // Note: Not really a constant, actually just a "known" value translated
      // into unknown for abstract evaluation rules.
      type: "Const";
      val: Tracer; // holds reference
    }
  | {
      type: "JaxprEqn";
      prim: Primitive;
      tracersIn: PartialEvalTracer[]; // holds reference
      params: Record<string, any>;
      avalsOut: ShapedArray[];
      tracerRefsOut: WeakRef<PartialEvalTracer>[];
    };

class PartialEvalTracer extends Tracer {
  #rc: number; // PartialEvalTracer reference count, used to free references.

  // Note: Either pval is known and recipe is null, or pval is unknown and
  // recipe describes how to compute the value.
  constructor(
    trace: Trace,
    readonly pval: PartialVal,
    readonly recipe: JaxprRecipe | null,
  ) {
    super(trace);
    this.#rc = 1;
  }

  get aval(): AbstractValue {
    return this.pval.aval;
  }

  toString(): string {
    if (!this.recipe) {
      return `PartialEvalTracer(${this.pval.toString()})`;
    } else {
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
    }
  }

  get ref() {
    if (this.#rc <= 0) {
      throw new UseAfterFreeError(this);
    }
    this.#rc++;
    return this;
  }
  /** Whether this PETracer hasn't been fully disposed yet. */
  get isAlive(): boolean {
    return this.#rc > 0;
  }

  override isAliveForCleanup(): boolean {
    return this.#rc > 0;
  }

  dispose() {
    if (this.#rc <= 0) {
      throw new UseAfterFreeError(this);
    }
    if (--this.#rc === 0) {
      // Cascade dispose to owned values. Known pval values and Const recipe
      // values are ref'd by partialEvalGraphToJaxpr before cleanup, so the
      // cascade here just releases the PETracer's share of ownership.
      // JaxprEqn tracersIn are NOT cascaded — they are handled by the
      // graph-wide toposort cleanup in partialEvalGraphToJaxpr.
      if (this.pval.isKnown) {
        this.pval.val!.dispose();
      } else if (this.recipe?.type === "Const") {
        this.recipe.val.dispose();
      }
    }
  }

  fullLower(): Tracer {
    if (this.pval.isKnown) return this.pval.val!;
    return this;
  }
}

class PartialEvalTrace extends Trace {
  newArg(pval: PartialVal) {
    if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }

  pure(val: TracerValue): Tracer {
    const arr = pureArray(val);
    // Track literal-created Arrays so the PE cleanup path can release them.
    // pureArray() returns existing Tracers as-is, so non-Tracer inputs (numbers,
    // TypedArrays) produce genuinely new allocations that need disposal.
    // Without this, literals used as args in JVP rules (0.0, 1.0, etc.) leak
    // a backend slot each.
    if (!(val instanceof Tracer)) {
      this.knownIntermediates.push(arr);
    } else if (val instanceof JaxArray && val.hasPartialEvalCreationRef) {
      this.knownIntermediates.push(arr);
    }
    return new PartialEvalTracer(this, PartialVal.known(arr), null);
  }
  lift = this.pure;

  // Track concrete Arrays created during all-known evaluation.
  // Stored on main.globalData so all PETrace instances created by findTopTrace
  // (which creates new trace instances for the same MainTrace) share the list.
  get knownIntermediates(): Tracer[] {
    let arr = this.main.globalData?._knownIntermediates;
    if (!arr) {
      arr = [];
      if (!this.main.globalData) (this.main as any).globalData = {};
      this.main.globalData._knownIntermediates = arr;
    }
    return arr;
  }

  // All Const PETracers created by instantiateConst. Stored on main.globalData
  // so all PETrace instances from the same MainTrace share the list. Used to
  // dispose Const PETracers unreachable from tracersOut (e.g. hasAux aux
  // computations that reference input arrays but aren't in the jaxpr outputs).
  get allConstPETracers(): PartialEvalTracer[] {
    let arr = this.main.globalData?._allConstPETracers;
    if (!arr) {
      arr = [];
      if (!this.main.globalData) (this.main as any).globalData = {};
      this.main.globalData._allConstPETracers = arr;
    }
    return arr;
  }

  instantiateConst(tracer: PartialEvalTracer) {
    if (!tracer.pval.isKnown) {
      return tracer;
    } else {
      // Translate known value into unknown "Const" recipe for abstract eval.
      // .ref gives ClosedJaxpr independent ownership of the const value.
      // The matching dispose happens in partialEvalGraphToJaxpr cleanup
      // (for reachable consts) or the forward-PE cleanup path (for unreachable
      // ones).
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      const val = tracer.pval.val!.ref;
      const constTracer = new PartialEvalTracer(this, pval, {
        type: "Const",
        val,
      });
      this.allConstPETracers.push(constTracer);
      return constTracer;
    }
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: PartialEvalTracer[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      const results = bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
      // Track concrete results for disposal of intermediates after PE.
      for (const r of results) this.knownIntermediates.push(r);
      return results;
    }
    if (primitive === Primitive.Jit) {
      // Special case, needs its own PartialEvalTrace handling because unlike
      // other primtiives, Jit can have subexpressions that are known while
      // other outputs are unknown.
      const { name, jaxpr, numConsts } =
        params as PrimitiveParams<Primitive.Jit>;
      return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
    }
    if (primitive === Primitive.ForiLoop) {
      return this.#partialEvalForiLoop(params as PrimitiveParams<any>, tracers);
    }
    if (primitive === Primitive.Scan) {
      // Special case for JVP'd scan: primal outputs depend only on primal inputs
      return this.#partialEvalScan(
        params as PrimitiveParams<Primitive.Scan>,
        tracers,
      );
    }
    if (primitive === Primitive.AssociativeScan) {
      return this.#partialEvalAssociativeScan(
        params as PrimitiveParams<Primitive.AssociativeScan>,
        tracers,
      );
    }
    if (primitive === Primitive.BlockMap) {
      return this.#partialEvalBlockMap(
        params as PrimitiveParams<Primitive.BlockMap>,
        tracers,
      );
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [], // Populated later on
    };
    const tracersOut = avalsOut.map((aval, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
    });
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }

  /**
   * Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
   * values as possible (with JIT) and forwarding the unknown ones.
   *
   * Used when encountering a Jit rule during the trace.
   */
  #partialEvalJaxpr(
    name: string,
    jaxpr: Jaxpr,
    numConsts: number,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    void numConsts; // Unused
    jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

    const inUnknowns = tracers.map((t) => !t.pval.isKnown);
    const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(
      jaxpr,
      inUnknowns,
    );

    const [knownTracers, unknownTracers] = partitionList(inUnknowns, tracers);

    const outs1Res = bind(
      Primitive.Jit,
      knownTracers.map((t) => t.ref.fullLower()),
      { name: `${name}_peval`, jaxpr: jaxpr1, numConsts: 0 },
    );
    const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
    const res = outs1Res.slice(jaxpr1.outs.length - numRes);

    const resTracers = res.map((x) =>
      this.instantiateConst(fullRaise(this, x) as PartialEvalTracer),
    );
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Jit,
      tracersIn: resTracers.concat(unknownTracers),
      params: { name: `${name}_resid`, jaxpr: jaxpr2, numConsts: 0 },
      avalsOut: jaxpr2.outs.map((x) => x.aval),
      tracerRefsOut: [], // populated later
    };
    const outs2 = jaxpr2.outs.map((x, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        recipe.tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
    });
    recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));

    // Stitch the known and unknown output tracers together, both with Jit.
    let i = 0;
    let j = 0;
    return outUnknowns.map((unk) => (unk ? outs2[j++] : outs1[i++]));
  }

  /**
   * Partial eval for Scan primitive.
   *
   * When scan is encountered during partial evaluation (e.g., inside JVP for VJP):
   * - If all inputs are known, just run the scan
   * - If this is a JVP'd scan (doubled carry/xs), we can split primal (known)
   *   from tangent (unknown) outputs
   * - Otherwise, mark all outputs as unknown
   */
  #partialEvalForiLoop(params: any, tracers: PartialEvalTracer[]): Tracer[] {
    const isKnown = tracers.map((t) => t.pval.isKnown);
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      const inputs = tracers.map((t) => t.fullLower());
      return bind(Primitive.ForiLoop, inputs, params);
    }

    const { isJvpTransformed, numConsts: _numConsts } = params;
    if (!isJvpTransformed) {
      throw new Error(
        "Backward pass or partial eval of ForiLoop with unknown inputs requires isJvpTransformed flag",
      );
    }

    // Evaluate full loop with zeros for tangents
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        return zerosInternal(t.pval.aval.shape, t.pval.aval.dtype);
      }
    });

    const fullOuts = bind(Primitive.ForiLoop, fullInputs, params);

    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[Primitive.ForiLoop](avalsIn, params);

    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.ForiLoop,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    const tracersOut: PartialEvalTracer[] = [];
    const numCarry = avalsOut.length / 2; // isJvpTransformed means numCarry is doubled

    // Primals out (first half) are known from full execution
    for (let i = 0; i < numCarry; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), recipe),
      );
    }

    // Tangents out (second half) are unknown
    for (let i = numCarry; i < 2 * numCarry; i++) {
      fullOuts[i].dispose(); // Tangents evaluated with zeros are garbage - dispose them
      tracersIn.forEach((t) => t.ref);
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));

    const retainedKnownOutputs = new Set<Tracer>();
    for (let i = 0; i < numCarry; i++) {
      retainedKnownOutputs.add(fullOuts[i]);
    }
    for (const out of retainedKnownOutputs) {
      this.knownIntermediates.push(out);
    }

    // Clean up fullInputs: balance the .ref taken at creation (known inputs)
    // and the creation ref of synthesized zeros (unknown inputs).
    // If an outer JaxprTrace captured a value via getOrMakeConstTracer,
    // that builder holds an independent .ref — disposing here is safe.
    for (const inp of fullInputs) {
      if (!retainedKnownOutputs.has(inp)) {
        inp.dispose();
      }
    }

    return tracersOut;
  }

  #partialEvalScan(
    params: PrimitiveParams<Primitive.Scan>,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    const { numConsts: _numConsts, numCarry } = params;

    // Determine which tracers are known/unknown
    const isKnown = tracers.map((t) => t.pval.isKnown);

    // Check if any inputs are unknown
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      // All inputs known, just run the scan
      const inputs = tracers.map((t) => t.fullLower());
      return bind(Primitive.Scan, inputs, params);
    }

    // Get abstract values for all outputs
    const avalsIn = tracers.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[Primitive.Scan](avalsIn, params);
    const numY = avalsOut.length - numCarry;

    const isJvpScan = params.isJvpTransformed ?? false;

    if (!isJvpScan) {
      // Not a JVP scan, mark all outputs as unknown
      const tracersIn = tracers.map((t) => this.instantiateConst(t));
      const recipe: JaxprRecipe = {
        type: "JaxprEqn",
        prim: Primitive.Scan,
        tracersIn,
        params,
        avalsOut,
        tracerRefsOut: [],
      };
      const tracersOut = avalsOut.map((aval, i) => {
        if (i > 0) tracersIn.forEach((t) => t.ref);
        return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
      });
      recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
      return tracersOut;
    }

    // This is a JVP'd scan. We need to:
    // 1. Run primal-only computation to get known outputs
    // 2. Create a residual jaxpr for tangent computation

    const numPrimalCarry = numCarry / 2;
    const numPrimalY = numY / 2;

    // Run primal-only computation using known inputs + zeros for tangent
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        return zerosInternal(t.pval.aval.shape, t.pval.aval.dtype);
      }
    });

    const fullOuts = bind(Primitive.Scan, fullInputs, params);

    // Create tracersIn for the residual jaxpr
    const tracersIn = tracers.map((t) => this.instantiateConst(t));

    // Build recipe for the full scan
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Scan,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    // Build output tracers
    const tracersOut: PartialEvalTracer[] = [];

    // Primal carry outputs (first numPrimalCarry) are known
    for (let i = 0; i < numPrimalCarry; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null),
      );
    }

    // Tangent carry outputs are unknown
    let isFirstUnknown = true;
    for (let i = numPrimalCarry; i < numCarry; i++) {
      fullOuts[i].dispose();
      if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
      isFirstUnknown = false;
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    // Primal Y outputs are known
    for (let i = 0; i < numPrimalY; i++) {
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.known(fullOuts[numCarry + i]),
          null,
        ),
      );
    }

    // Tangent Y outputs are unknown
    for (let i = numPrimalY; i < numY; i++) {
      fullOuts[numCarry + i].dispose();
      tracersIn.forEach((t) => t.ref);
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.unknown(avalsOut[numCarry + i]),
          recipe,
        ),
      );
    }

    const retainedKnownOutputs = new Set<Tracer>();
    for (let i = 0; i < numPrimalCarry; i++) {
      retainedKnownOutputs.add(fullOuts[i]);
    }
    for (let i = 0; i < numPrimalY; i++) {
      retainedKnownOutputs.add(fullOuts[numCarry + i]);
    }
    for (const out of retainedKnownOutputs) {
      this.knownIntermediates.push(out);
    }

    // Clean up fullInputs: balance the .ref taken at line 920 (known inputs)
    // and the creation ref of synthesized zeros (unknown inputs).
    // For both: if an outer JaxprTrace captured the value via
    // getOrMakeConstTracer, that builder holds an independent .ref, so
    // disposing here is safe (rc stays ≥ 1 from the builder's ownership).
    for (const inp of fullInputs) {
      if (!retainedKnownOutputs.has(inp) && inp.refCount > 0) {
        inp.dispose();
      }
    }

    // tracerRefsOut: known positions get null ref
    recipe.tracerRefsOut = tracersOut.map((t) =>
      t.pval.isKnown ? (null as any) : new WeakRef(t),
    );

    return tracersOut;
  }

  /**
   * Partial eval for AssociativeScan primitive.
   *
   * Same approach as scan PE: if JVP-doubled (even numLeaves), split
   * primal (known) from tangent (unknown) outputs.
   */
  #partialEvalAssociativeScan(
    params: PrimitiveParams<Primitive.AssociativeScan>,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    const { numLeaves } = params;
    const isKnown = tracers.map((t) => t.pval.isKnown);
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      const inputs = tracers.map((t) => t.fullLower());
      return bind(Primitive.AssociativeScan, inputs, params);
    }

    const avalsIn = tracers.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[Primitive.AssociativeScan](
      avalsIn,
      params,
    );

    const isJvpScan = numLeaves % 2 === 0;

    if (!isJvpScan) {
      const tracersIn = tracers.map((t) => this.instantiateConst(t));
      const recipe: JaxprRecipe = {
        type: "JaxprEqn",
        prim: Primitive.AssociativeScan,
        tracersIn,
        params,
        avalsOut,
        tracerRefsOut: [],
      };
      const tracersOut = avalsOut.map((aval, i) => {
        if (i > 0) tracersIn.forEach((t) => t.ref);
        return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
      });
      recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
      return tracersOut;
    }

    const numOrigLeaves = numLeaves / 2;

    // Run full scan with zeros for unknown inputs
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        return zerosInternal(t.pval.aval.shape, t.pval.aval.dtype);
      }
    });

    const fullOuts = bind(Primitive.AssociativeScan, fullInputs, params);

    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.AssociativeScan,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    const tracersOut: PartialEvalTracer[] = [];

    // First numOrigLeaves outputs are known (primal results)
    for (let i = 0; i < numOrigLeaves; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null),
      );
    }

    // Last numOrigLeaves outputs are unknown (tangent results)
    for (let i = numOrigLeaves; i < numLeaves; i++) {
      fullOuts[i].dispose();
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    const retainedKnownOutputs = new Set<Tracer>(
      fullOuts.slice(0, numOrigLeaves),
    );
    for (const out of retainedKnownOutputs) {
      this.knownIntermediates.push(out);
    }
    // Clean up fullInputs: balance the .ref taken at creation (known inputs)
    // and the creation ref of synthesized zeros (unknown inputs).
    // If an outer JaxprTrace captured a value via getOrMakeConstTracer,
    // that builder holds an independent .ref — disposing here is safe.
    for (const inp of fullInputs) {
      if (!retainedKnownOutputs.has(inp) && inp.refCount > 0) {
        inp.dispose();
      }
    }

    recipe.tracerRefsOut = tracersOut.map((t) =>
      t.pval.isKnown ? (null as any) : new WeakRef(t),
    );

    return tracersOut;
  }

  /**
   * Partial eval for BlockMap primitive.
   *
   * Same approach as scan/fori PE: if JVP-doubled, run the full block_map
   * concretely (zeros for unknowns), mark primal outputs as known and
   * tangent outputs as unknown for the residual jaxpr.
   */
  #partialEvalBlockMap(
    params: PrimitiveParams<Primitive.BlockMap>,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    const isKnown = tracers.map((t) => t.pval.isKnown);
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      const inputs = tracers.map((t) => t.fullLower());
      const results = bind(Primitive.BlockMap, inputs, params);
      for (const r of results) this.knownIntermediates.push(r);
      return results;
    }

    const isJvp = params.isJvpTransformed ?? false;
    if (!isJvp) {
      // Non-JVP BlockMap with some unknown inputs: all outputs unknown.
      const tracersIn = tracers.map((t) => this.instantiateConst(t));
      const avalsIn = tracersIn.map((t) => t.pval.aval);
      const avalsOut = abstractEvalRules[Primitive.BlockMap](avalsIn, params);
      const recipe: JaxprRecipe = {
        type: "JaxprEqn",
        prim: Primitive.BlockMap,
        tracersIn,
        params,
        avalsOut,
        tracerRefsOut: [],
      };
      const tracersOut = avalsOut.map((aval, i) => {
        if (i > 0) tracersIn.forEach((t) => t.ref);
        return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
      });
      recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
      return tracersOut;
    }

    // JVP-doubled BlockMap: outputs = [primalOuts..., tangentOuts...]
    // Primal outputs are known, tangent outputs are unknown.
    const numOutputs = abstractEvalRules[Primitive.BlockMap](
      tracers.map((t) => t.pval.aval),
      params,
    ).length;
    const numPrimalOuts = numOutputs / 2;

    // Run full block_map with zeros substituted for unknown inputs.
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        return zerosInternal(t.pval.aval.shape, t.pval.aval.dtype);
      }
    });

    const fullOuts = bind(Primitive.BlockMap, fullInputs, params);

    // Build residual jaxpr recipe for tangent computation.
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsOut = abstractEvalRules[Primitive.BlockMap](
      tracersIn.map((t) => t.pval.aval),
      params,
    );

    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.BlockMap,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    const tracersOut: PartialEvalTracer[] = [];

    // Primal outputs are known from the concrete execution.
    for (let i = 0; i < numPrimalOuts; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null),
      );
    }

    // Tangent outputs are unknown — part of the residual jaxpr.
    let isFirstUnknown = true;
    for (let i = numPrimalOuts; i < numOutputs; i++) {
      fullOuts[i].dispose();
      if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
      isFirstUnknown = false;
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    // Clean up: dispose .ref'd inputs that aren't retained as known outputs.
    const retainedKnownOutputs = new Set<Tracer>(
      fullOuts.slice(0, numPrimalOuts),
    );
    for (const out of retainedKnownOutputs) {
      this.knownIntermediates.push(out);
    }
    // Clean up fullInputs: balance the .ref taken at creation (known inputs)
    // and the creation ref of synthesized zeros (unknown inputs).
    // If an outer JaxprTrace captured a value via getOrMakeConstTracer,
    // that builder holds an independent .ref — disposing here is safe.
    for (const inp of fullInputs) {
      if (!retainedKnownOutputs.has(inp) && inp.refCount > 0) {
        inp.dispose();
      }
    }

    recipe.tracerRefsOut = tracersOut.map((t) =>
      t.pval.isKnown ? (null as any) : new WeakRef(t),
    );

    return tracersOut;
  }
}

/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(
  jaxpr: Jaxpr,
  inUnknowns: boolean[],
  instantiate?: boolean[],
): { jaxpr1: Jaxpr; jaxpr2: Jaxpr; outUnknowns: boolean[]; numRes: number } {
  jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

  const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
  const knownVars = new Set(knownIns); // Var that we can evaluate immediately.
  const residuals = new Set<Var>(); // Vars to evaluate in eqns1, and pass to eqns2 (subset of knownVars).

  const eqns1: JaxprEqn[] = [];
  const eqns2: JaxprEqn[] = [];
  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive === Primitive.Jit) {
      throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
    }
    const hasUnknowns = eqn.inputs.some(
      (x) => x instanceof Var && !knownVars.has(x),
    );
    if (hasUnknowns) {
      for (const x of eqn.inputs) {
        if (x instanceof Var && knownVars.has(x)) {
          residuals.add(x);
        }
      }
      eqns2.push(eqn);
    } else {
      eqns1.push(eqn);
      for (const v of eqn.outBinders) {
        knownVars.add(v);
      }
    }
  }
  const outUnknowns = jaxpr.outs.map(
    (x) => x instanceof Var && !knownVars.has(x),
  );
  // If instantiate is provided, move selected outputs into residuals.
  if (instantiate !== undefined) {
    for (let i = 0; i < jaxpr.outs.length; i++) {
      const x = jaxpr.outs[i];
      if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
        residuals.add(x);
        outUnknowns[i] = true; // Mark as unknown.
      }
    }
  }

  const residualsL = Array.from(residuals);
  const [ins1, ins2] = partitionList(inUnknowns, jaxpr.inBinders);
  const [outs1, outs2] = partitionList(outUnknowns, jaxpr.outs);
  const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
  const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
  return { jaxpr1, jaxpr2, outUnknowns, numRes: residualsL.length };
}

/**
 * Convert the graph representation of a partial eval to a standard Jaxpr.
 * Also called `tracers_to_jaxpr()` in JAX.
 */
function partialEvalGraphToJaxpr(
  tracersIn: PartialEvalTracer[],
  tracersOut: PartialEvalTracer[],
): ClosedJaxpr {
  const tracerToVar = new Map<PartialEvalTracer, Var>();
  const constToVar = new Map<Tracer, Var>();
  const processedEqns = new Set<JaxprRecipe>(); // Avoid translating the same equation multiple times.
  const constPETracers: PartialEvalTracer[] = []; // Intermediate Const PETracers to dispose.
  const eqns: JaxprEqn[] = [];

  for (const t of tracersIn) {
    tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
  }

  for (const t of toposort(tracersOut, (t) =>
    t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [],
  )) {
    if (!t.recipe) {
      throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
    }
    if (t.recipe.type === "LambdaBinding") {
      // Check that the binding is in the input list.
      if (!tracersIn.includes(t)) {
        throw new TypeError("LambdaBinding tracer not in input list");
      }
    } else if (t.recipe.type === "Const") {
      const val = t.recipe.val;
      let binder = constToVar.get(val);
      if (!binder) {
        binder = new Var(ShapedArray.fromAval(val.aval));
        constToVar.set(val, binder);
      }
      tracerToVar.set(t, binder);
      constPETracers.push(t); // Track for disposal
    } else if (t.recipe.type === "JaxprEqn") {
      if (!processedEqns.has(t.recipe)) {
        processedEqns.add(t.recipe);
        const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t)!);
        const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
        for (let i = 0; i < outBinders.length; i++) {
          const ref = t.recipe.tracerRefsOut[i];
          // ref can be null for known outputs in partial-eval of JVP'd scan
          const tracerOut = ref?.deref?.();
          if (tracerOut) {
            tracerToVar.set(tracerOut, outBinders[i]);
          }
        }
        eqns.push(
          new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders),
        );
      }
    }
  }

  const [consts, constvars] = unzip2(constToVar.entries());
  const inBinders = [
    ...constvars,
    ...tracersIn.map((t) => tracerToVar.get(t)!),
  ];
  const outVars = tracersOut.map((t) => tracerToVar.get(t)!);
  let jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check

  // Give ClosedJaxpr independent ownership of its consts before PETracer
  // cleanup. PE consts always get .ref because the PETracer disposal cascade
  // will consume the creation ref.
  for (const c of consts) {
    // jax-js-lint: allow-ref
    (c as Tracer).ref;
  }

  // Cleanup PETracer wrappers:
  // 1) Const PETracers: their recipe.val was .ref'd by instantiateConst,
  //    so cascade to recipe.val.dispose() just balances the .ref (safe).
  //    The extra .ref above ensures ClosedJaxpr retains its own ownership.
  //    Multi-output equations (Scan, Jit) .ref input tracers once per
  //    additional output in processPrimitive, inflating PETracer #rc above 1.
  //    A single .dispose() won't reach #rc=0 to trigger the cascade.
  //    Loop until fully disposed so the cascade fires exactly once.
  // 2) Unknown (non-Const) PETracers in tracersIn/tracersOut: no val to
  //    cascade to, so dispose() is effectively a no-op.
  // SKIP known PETracers — they hold borrowed references to caller-owned
  // Arrays (e.g., user inputs, forward pass results). Disposing would free
  // the caller's arrays prematurely.
  for (const t of constPETracers) {
    while (t.isAlive) t.dispose();
  }
  for (const t of tracersIn) {
    if (!t.pval.isKnown) t.dispose();
  }
  for (const t of tracersOut) {
    if (!t.pval.isKnown) t.dispose();
  }

  jaxpr = jaxpr.simplify();
  if (DEBUG >= 5) {
    console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
  }

  return ClosedJaxpr.borrowed(jaxpr, consts);
}

// implementation of vjp and grad

/** Marker type for pullback, used by transpose rules. */
class UndefPrimal {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }
}

/**
 * Helper to get or compute a primal (known) variable's value during transpose.
 * For intermediate variables that are known (computed from only known inputs),
 * we need to evaluate the equations that produce them.
 */
function getOrComputePrimal(
  jaxpr: Jaxpr,
  knownVars: Set<Var>,
  knownPrimals: Map<Var, Tracer>,
  v: Var,
  internalArrays?: Set<Tracer>,
): Tracer {
  // Return .ref so the caller (transpose rule) gets an independent copy.
  // Transpose rules like Add and Where explicitly dispose known primals;
  // without .ref the shared value in knownPrimals would be freed too.
  if (knownPrimals.has(v)) {
    const r = knownPrimals.get(v)!.ref;
    if (internalArrays) internalArrays.add(r);
    return r;
  }

  // Find the equation that produces this variable
  const eqn = jaxpr.eqns.find((eq) => eq.outBinders.some((out) => out === v));
  if (!eqn) {
    throw new Error(
      `Internal error: could not find equation producing variable`,
    );
  }

  // Recursively get values for inputs
  const inputVals = eqn.inputs.map((inp) =>
    inp instanceof Lit
      ? array(inp.value, { dtype: inp.dtype })
      : getOrComputePrimal(jaxpr, knownVars, knownPrimals, inp, internalArrays),
  );

  // Evaluate this equation
  const results = bind(eqn.primitive, inputVals, eqn.params as any);

  // Store all output values
  for (let i = 0; i < eqn.outBinders.length; i++) {
    knownPrimals.set(eqn.outBinders[i], results[i]);
  }

  // Return .ref so the caller gets an independent copy.
  const result = knownPrimals.get(v);
  if (!result) {
    throw new Error(`Internal error: variable not produced by equation`);
  }
  const r = result.ref;
  if (internalArrays) internalArrays.add(r);
  return r;
}

/**
 * Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
 *
 * Will raise a TypeError if the provided Jaxpr is not a linear function of its,
 * inputs, as general expressions cannot be transposed.
 */
function evalJaxprTransposed(
  jaxpr: Jaxpr,
  args: (Tracer | UndefPrimal)[],
  cotangents: Tracer[],
): Tracer[] {
  // Track which variables are known (primal) vs unknown (tangent).
  // A variable is known if ALL its inputs are known (primal values propagate).
  // A variable is unknown if ANY of its inputs are unknown (tangent dependency).
  const knownVars = new Set<Var>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownVars.add(jaxpr.inBinders[i]);
    }
  }

  // Forward pass: propagate "known" status through equations
  for (const eqn of jaxpr.eqns) {
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );
    if (allInputsKnown) {
      // All inputs are known → all outputs are known (primal computation)
      for (const outVar of eqn.outBinders) {
        knownVars.add(outVar);
      }
    }
  }

  // Now collect actual Tracer values for known input variables
  const knownPrimals = new Map<Var, Tracer>();
  const argPrimals = new Set<Var>(); // Track which primals are from args (owned by caller)
  const argPrimalInitRc = new Map<Var, number>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      const arg = args[i] as Tracer;
      knownPrimals.set(jaxpr.inBinders[i], arg);
      argPrimals.add(jaxpr.inBinders[i]);
      const concrete = unwrapToConcreteArray(arg);
      argPrimalInitRc.set(
        jaxpr.inBinders[i],
        concrete instanceof JaxArray ? concrete.refCount : arg.refCount,
      );
    }
  }

  const ctStore = new Map<Var, Tracer>();

  // Track arrays created internally for batch disposal at the end.
  // Includes: zeros from readCotangent, accumulated sums from writeCotangent,
  // literal arrays from primalsIn, and non-external values entering ctStore.
  const internalArrays = new Set<Tracer>();
  // Track externally-owned cotangents (seeds from caller) — never dispose these.
  const externalCts = new Set<Tracer>();
  for (const ct of cotangents) {
    if (ct instanceof Tracer) externalCts.add(ct);
  }

  const readCotangent = (v: Var) => {
    const ct = ctStore.get(v);
    if (ct) {
      // We should read a cotangent at most once, as an out binder.
      ctStore.delete(v);
      return ct;
    } else {
      const z = zerosInternal(v.aval.shape, v.aval.dtype);
      // Tracked in internalArrays for cleanup. If an outer makeJaxpr builder
      // captures this via getOrMakeConstTracer, it takes an independent .ref.
      // The internalArrays cleanup balances the creation ref; the builder's
      // ClosedJaxpr.dispose() balances the .ref.
      internalArrays.add(z);
      return z;
    }
  };

  const writeCotangent = (v: Var, ct: Tracer | null) => {
    if (ct !== null) {
      // Track non-external cotangent values for batch disposal.
      // Don't dispose eagerly — ct may be aliased (e.g. Add transpose returns [ct, ct]).
      if (!externalCts.has(ct)) internalArrays.add(ct);
      const oldCt = ctStore.get(v);
      // May need to accumulate cotangents if used in multiple JaxprEqns.
      if (oldCt) {
        const sum = add(oldCt, ct);
        internalArrays.add(sum);
        ctStore.set(v, sum);
      } else {
        ctStore.set(v, ct);
      }
    }
  };

  for (let i = 0; i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var) writeCotangent(v, cotangents[i]);
  }

  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    // Inputs are primalsIn and cotangentsOut, outputs are cotangentsIn. We're
    // using the known primal values to _pull back_ cotangents for unknown
    // values. Tricky!

    // Check if all inputs are known (using our forward-propagated knownVars)
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );

    if (allInputsKnown) {
      // Skip equations where all inputs are known (residual equations).
      // These don't depend on unknowns and don't contribute to the linear function.
      continue;
    }

    // For equations with mixed inputs, we need to get residual values for known inputs
    // and mark unknown inputs as UndefPrimal
    const primalsIn = eqn.inputs.map((v) => {
      if (v instanceof Lit) {
        const lit = array(v.value, { dtype: v.dtype });
        // Tracked in internalArrays for cleanup. If an outer makeJaxpr builder
        // captures this, it takes an independent .ref via getOrMakeConstTracer.
        internalArrays.add(lit);
        return lit;
      }
      return knownVars.has(v)
        ? getOrComputePrimal(jaxpr, knownVars, knownPrimals, v, internalArrays)
        : new UndefPrimal(v.aval);
    });

    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params as any);
    for (let j = 0; j < eqn.inputs.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownVars.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      } else if (cotangentsIn[j] !== null) {
        throw new Error("internal: cotangent should be null");
      }
    }
  }

  const results: Tracer[] = [];
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (args[i] instanceof UndefPrimal) {
      results.push(readCotangent(jaxpr.inBinders[i]));
    }
  }

  // Flush result arrays' pending backend dispatches before disposing
  // intermediates. Pending operations hold incRef on their I/O slots; if
  // results carry unsubmitted PEs while intermediates are disposed, shared
  // PE refcounts never reach zero and the slots they reference leak.
  // Submitting first releases those cross-references cleanly.
  flushPendingConcreteArrays(results);

  // Always restore input-known primals to their initial refcount.
  // This balances temporary .ref borrows from getOrComputePrimal even when
  // running inside a trace (e.g. grad(scan) checkpoint path).
  for (const v of argPrimals) {
    const val = knownPrimals.get(v);
    const initialRc = argPrimalInitRc.get(v);
    if (val && initialRc !== undefined) {
      const concrete = unwrapToConcreteArray(val);
      const currentRc =
        concrete instanceof JaxArray ? concrete.refCount : val.refCount;
      const excess = currentRc - initialRc;
      for (let i = 0; i < excess; i++) {
        try {
          val.dispose();
        } catch {
          break;
        }
      }
    }
  }

  // Dispose internally-created arrays and computed primals.
  // When inside an abstract trace (e.g., inner grad running during outer grad's
  // tracing), computed primals and internal arrays may be tracers from outer
  // traces. Disposing them would cascade (JVPTracer.dispose → primal.dispose)
  // and free values still needed by outer evaluations. Skip disposal when
  // abstract traces are on the stack. BatchTrace is safe — values are concrete.
  if (!insideAbstractTrace()) {
    const returnedSet = new Set(results);

    // 1. For computed primals (created by getOrComputePrimal forward recomputation):
    //    Fully dispose — nobody else owns them.
    for (const [v, t] of knownPrimals.entries()) {
      if (!argPrimals.has(v) && !returnedSet.has(t)) {
        try {
          while (t.refCount > 0) t.dispose();
        } catch {
          // Already disposed.
        }
      }
    }

    // 2. Dispose internal arrays (zeros, accumulated sums, literal arrays).
    //    Skip arg primals (same objects due to .ref returning `this`).
    //    Skip returned results and externally-owned cotangents.
    for (const arr of internalArrays) {
      if (!returnedSet.has(arr) && !externalCts.has(arr)) {
        // Check if this is an arg primal (same object via .ref)
        let isArgPrimal = false;
        for (const v of argPrimals) {
          if (knownPrimals.get(v) === arr) {
            isArgPrimal = true;
            break;
          }
        }
        if (!isArgPrimal) {
          try {
            arr.dispose();
          } catch {
            // Already disposed by a transpose rule or nested operation.
          }
        }
      }
    }
  }

  return results;
}

/**
 * Inverse operation of `generalBroadcast()` for backpropagation.
 *
 * `x` has the shape of the result of an operation that was broadcasted with
 * `target` (it's a cotangent during backprop). Returns a tracer with rank and
 * shape equal to `target`.
 */
function unbroadcast(x: Tracer, target: UndefPrimal): Tracer {
  const shape = target.aval.shape;

  // 1. Remove extra dimensions from x, if any.
  //    x can either have rank == target.ndim (fine!), or rank > target.ndim.
  //    In the latter case, we need to trim off extra dimensions on the left.
  const extraDims = x.ndim > shape.length ? range(x.ndim - shape.length) : [];
  if (x.ndim < shape.length) {
    throw new Error(
      `unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape})`,
    );
  }

  // 2. Reduce (but keep) dimensions of x that are 1 in target.
  const unsqueeze: number[] = [];
  const keptReduceDims: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    // i is indexed according to target.
    const indexFromEnd = shape.length - i; // >= 1
    const indexInX = x.ndim - indexFromEnd;
    const xLen = x.shape[indexInX];
    if (xLen > 1 && shape[i] === 1) {
      unsqueeze.push(i);
      keptReduceDims.push(indexInX);
    } else if (shape[i] !== xLen) {
      throw new Error("internal: unbroadcast shape mismatch");
    }
  }

  const reductionDims = [...extraDims, ...keptReduceDims];
  if (reductionDims.length === 0) return x;
  let result = x.sum(reductionDims);
  if (!deepEqual(result.shape, shape)) {
    const sumResult = result;
    result = broadcast(sumResult, shape, unsqueeze); // keep dims selectively
    sumResult.dispose();
  }
  return result;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule<P extends Primitive> = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: PrimitiveParams<P>,
) => (Tracer | null)[];

/**
 * Slice a traced array along mapped axes for a given block index.
 * No halo — just a clean [j*B, (j+1)*B] slice (clamped for partial last block).
 */
function sliceBlock(
  arr: Tracer,
  axes: (number | null)[],
  blockIdx: number[],
  blockShape: number[],
  gridRank: number,
): Tracer {
  const shape = arr.aval.shape;
  const slicePairs: [number, number][] = (shape as number[]).map((s) => [0, s]);
  let anySlice = false;
  for (let g = 0; g < gridRank; g++) {
    if (axes[g] === null) continue;
    const ax = axes[g]!;
    const dim = shape[ax] as number;
    const start = blockIdx[g] * blockShape[g];
    const end = Math.min(start + blockShape[g], dim);
    if (start !== 0 || end !== dim) anySlice = true;
    slicePairs[ax] = [start, end];
  }
  if (!anySlice) return arr.ref;
  const sliced = shrink(arr, slicePairs) as Tracer;
  // Pad to full blockShape if the last block is partial.
  const padWidth: [number, number][] = (shape as number[]).map(() => [0, 0]);
  let anyPad = false;
  for (let g = 0; g < gridRank; g++) {
    if (axes[g] === null) continue;
    const ax = axes[g]!;
    const got = slicePairs[ax][1] - slicePairs[ax][0];
    if (got < blockShape[g]) {
      padWidth[ax] = [0, blockShape[g] - got];
      anyPad = true;
    }
  }
  if (!anyPad) return sliced;
  const padded = pad(sliced, padWidth) as Tracer;
  sliced.dispose();
  return padded;
}

/**
 * Slice a traced array along mapped axes with halo expansion.
 * Replicates the eager block_map boundary clamping + zero-padding logic.
 */
function sliceBlockWithHalo(
  arr: Tracer,
  axes: (number | null)[],
  haloTable: [number, number][],
  blockIdx: number[],
  blockShape: number[],
  gridRank: number,
): Tracer {
  const shape = arr.aval.shape;
  const slicePairs: [number, number][] = (shape as number[]).map((s) => [0, s]);
  const padWidth: [number, number][] = (shape as number[]).map(() => [0, 0]);
  let anySlice = false;
  let anyPad = false;
  for (let g = 0; g < gridRank; g++) {
    if (axes[g] === null) continue;
    const ax = axes[g]!;
    const dim = shape[ax] as number;
    const [lo, hi] = haloTable[g];
    const start = blockIdx[g] * blockShape[g] - lo;
    const end = blockIdx[g] * blockShape[g] + blockShape[g] + hi;
    const clampedStart = Math.max(0, start);
    const clampedEnd = Math.min(end, dim);
    slicePairs[ax] = [clampedStart, clampedEnd];
    anySlice = anySlice || clampedStart !== 0 || clampedEnd !== dim;

    const leftPad = start < 0 ? -start : 0;
    const tileSize = blockShape[g] + lo + hi;
    const actual = clampedEnd - clampedStart;
    const rightPad = tileSize - actual - leftPad;
    if (leftPad > 0 || rightPad > 0) {
      padWidth[ax] = [leftPad, rightPad];
      anyPad = true;
    }
  }
  let result: Tracer;
  if (anySlice) {
    result = shrink(arr, slicePairs) as Tracer;
  } else {
    result = arr.ref;
  }
  if (!anyPad) return result;
  const padded = pad(result, padWidth) as Tracer;
  result.dispose();
  return padded;
}

// You need a transpose rule for a primitive p if:
//  - p is used in jvpRules, while computing a tangent (not primal)
//  - in this use, at least one argument to p is a tangent
//
// This computes a backward pass, so it pulls back cotangents to the inputs of p
// that are UndefPrimal (i.e., tangents that weren't sent forward).
const transposeRules: Partial<{ [P in Primitive]: TransposeRule<P> }> = {
  [Primitive.BlockMap](cts, args, params) {
    const {
      jaxpr,
      numConsts,
      blockShape,
      inAxes,
      outAxes,
      numInputs: _numInputs,
    } = params;
    const gridRank = blockShape.length;
    const undefMask = args.map((x) => x instanceof UndefPrimal);

    // Check if any differentiated input has non-zero halo. When true,
    // overlapping gradient regions require the scatter-add fallback path.
    let needsHaloVjp = false;
    if (params.halo) {
      for (let i = numConsts; i < args.length; i++) {
        if (!undefMask[i]) continue;
        const inputHalo = params.halo[i - numConsts];
        for (const [lo, hi] of inputHalo) {
          if (lo !== 0 || hi !== 0) {
            needsHaloVjp = true;
            break;
          }
        }
        if (needsHaloVjp) break;
      }
    }

    const bodyUndefPrimals = jaxpr.inBinders.map((_, i) => undefMask[i]);
    const transposedBody = transposeJaxpr(jaxpr, bodyUndefPrimals);

    const constResiduals = args
      .slice(0, numConsts)
      .filter((_, i) => !undefMask[i]) as Tracer[];
    const argResiduals = args
      .slice(numConsts)
      .filter((_, i) => !undefMask[numConsts + i]) as Tracer[];

    if (!needsHaloVjp) {
      // ---- Standard path: no halo overlap → delegate to block_map ----
      const residualAxes = inAxes.filter((_, i) => !undefMask[numConsts + i]);
      const transposeInAxes = [...residualAxes, ...outAxes];

      const nullOutAxis = Array(gridRank).fill(null);
      const constOutAxes = args
        .slice(0, numConsts)
        .filter((_, i) => undefMask[i])
        .map(() => nullOutAxis);
      const argOutAxes = inAxes.filter((_, i) => undefMask[numConsts + i]);
      const transposeOutAxes = [...constOutAxes, ...argOutAxes];

      const jaxprConsts = transposedBody.constEnv.values;
      const allConsts = [...jaxprConsts, ...constResiduals];

      // Forward halo for residual inputs; cotangent inputs (from outAxes)
      // have no halo — they match blockShape, not blockShape+halo.
      let transposeHalo: [number, number][][] | undefined;
      if (params.halo) {
        const residualHalo = params.halo.filter(
          (_, i) => !undefMask[numConsts + i],
        );
        const ctHalo = outAxes.map(() =>
          blockShape.map(() => [0, 0] as [number, number]),
        );
        transposeHalo = [...residualHalo, ...ctHalo];
      }

      const transposedOut = bind(
        Primitive.BlockMap,
        [...allConsts, ...argResiduals, ...(cts as Tracer[])],
        {
          jaxpr: transposedBody.jaxpr,
          blockShape,
          inAxes: transposeInAxes,
          outAxes: transposeOutAxes,
          numConsts: allConsts.length,
          numInputs: argResiduals.length + cts.length,
          threadTile: params.threadTile,
          halo: transposeHalo,
        },
      );

      // transposedBody comes from transposeJaxprCache. The BlockMap bind
      // consumes only its Jaxpr/const-environment contents; the cached const
      // environment remains cache-owned state and must not be disposed here.

      const numUndefConsts = constOutAxes.length;
      const constCts = transposedOut.slice(0, numUndefConsts);
      const argCts = transposedOut.slice(numUndefConsts);

      const outVals = [];
      let constIdx = 0,
        argIdx = 0;
      for (let i = 0; i < args.length; i++) {
        if (!undefMask[i]) outVals.push(null);
        else if (i < numConsts) outVals.push(constCts[constIdx++]);
        else outVals.push(argCts[argIdx++]);
      }
      return outVals;
    }

    // ---- Halo VJP: try gather-based stencil synthesis (Step 2) ----
    // If the forward body is a linear stencil (only Shrink/Add/Mul(Lit)),
    // synthesize a reversed-stencil backward body and run it as a single
    // block_map with reversed halo. O(1) dispatches instead of O(numBlocks).

    const halo = params.halo!;
    const gatherResult = (() => {
      // Need the original forward body for analysis. Only available after JVP.
      const origJaxpr = params.originalJaxpr;
      if (!origJaxpr) return null;
      const origNC = params.originalNumConsts ?? 0;

      // Original number of inputs (doubled by JVP → halve).
      const origNI = _numInputs / 2;

      // Original halo and inAxes are the first half of the doubled arrays.
      const origHalo = halo.slice(0, origNI);
      const origInAxes = inAxes.slice(0, origNI);

      // Eligible: single halo-ed input in the original body, single output.
      if (origJaxpr.outs.length !== 1) return null;

      let origHaloIdx = -1;
      let origHaloCount = 0;
      for (let i = 0; i < origNI; i++) {
        if (origHalo[i].some(([lo, hi]) => lo !== 0 || hi !== 0)) {
          origHaloCount++;
          origHaloIdx = i;
        }
      }
      if (origHaloCount !== 1) return null;

      // Guard: gather path only produces gradient for the haloed input.
      // If other tangent inputs are also differentiated, fall through to
      // pad+add which handles all inputs.
      for (let i = 0; i < origNI; i++) {
        if (i === origHaloIdx) continue;
        if (undefMask[numConsts + origNI + i]) return null;
      }

      // Analyze original forward body for linear stencil.
      const bodyInputIdx = origNC + origHaloIdx;
      const desc = analyzeLinearStencil(origJaxpr, bodyInputIdx);
      if (desc === null) return null;

      if (DEBUG >= 1)
        console.log(
          "halo-VJP: using gather-based stencil synthesis (" +
            desc.terms.length +
            " terms)",
        );

      // Compute backward halo: [hi, lo] per grid axis.
      const fwdHalo = origHalo[origHaloIdx];
      const bwdHalo: [number, number][] = fwdHalo.map(
        ([lo, hi]) => [hi, lo] as [number, number],
      );
      const inputInAxes = origInAxes[origHaloIdx];

      // Reverse each stencil term's Shrink slice.
      const reversedTerms = desc.terms.map((term) => ({
        slice: reverseStencilSlice(
          term.slice,
          inputInAxes,
          fwdHalo,
          blockShape,
        ),
        weight: term.weight,
      }));

      // Backward tile aval: same total size as forward tile (B+lo+hi = B+hi+lo).
      const fwdTileAval = origJaxpr.inBinders[bodyInputIdx].aval;
      const bwdTileAval = new ShapedArray(
        fwdTileAval.shape as number[],
        fwdTileAval.dtype,
        false,
      );

      // Synthesize backward body via makeJaxpr.
      const { jaxpr: bwdBodyJaxpr } = makeJaxpr(
        (ctTile: Tracer): Tracer => {
          let result: TracerValue | null = null;
          for (const term of reversedTerms) {
            let val: TracerValue = shrink(ctTile, term.slice);
            if (term.weight !== 1.0) val = mul(val, term.weight);
            result = result === null ? val : add(result, val);
          }
          return result as Tracer;
        },
        { validateRefs: false },
      )(bwdTileAval);

      // The ct for the tangent output: in the doubled BlockMap outputs
      // [outP..., outT...], the tangent outputs are the second half.
      const numOrigOutputs = origJaxpr.outs.length; // 1
      const ctTangent = cts[numOrigOutputs];
      if (ctTangent == null) return null; // no gradient signal → skip

      // Build backward block_map. outAxes[0] == outAxes[numOrigOutputs]
      // since JVP duplicates them.
      const bwdConsts = bwdBodyJaxpr.consts;
      const bwdOut = bind(
        Primitive.BlockMap,
        [...bwdConsts, ctTangent as Tracer],
        {
          jaxpr: bwdBodyJaxpr.jaxpr,
          blockShape,
          inAxes: [outAxes[0]], // ct follows forward output tiling
          outAxes: [inputInAxes], // gradient follows forward input tiling
          numConsts: bwdConsts.length,
          numInputs: 1,
          halo: [[...bwdHalo]],
        },
      );

      bwdBodyJaxpr.dispose();

      // Assemble outVals. In the doubled args:
      //   [doubledConsts..., primalInputs..., tangentInputs...]
      // The tangent input with halo is at: numConsts + origNI + origHaloIdx.
      const tangentArgIdx = numConsts + origNI + origHaloIdx;

      const outVals: (Tracer | null)[] = [];
      for (let i = 0; i < args.length; i++) {
        if (i === tangentArgIdx) outVals.push(bwdOut[0] as Tracer);
        else outVals.push(null);
      }
      return outVals;
    })();

    if (gatherResult !== null) return gatherResult;

    // ---- Halo VJP: pad+add fallback (Phase C.4 Step 1) ----
    // Overlapping halo regions mean adjacent blocks' gradient patches share
    // elements. Standard block_map concat/trim would lose overlap
    // contributions. Strategy: unroll the block loop at trace time, pad each
    // gradient patch to the padded-accumulator size, and add.
    if (DEBUG >= 1) console.log("halo-VJP: using pad+add accumulation");

    // Compute grid shape and original dims from ALL args (including UndefPrimal).
    const gridShape: number[] = new Array(gridRank).fill(0);
    const origDims: number[] = new Array(gridRank).fill(0);
    for (let i = 0; i < args.length - numConsts; i++) {
      const axes = inAxes[i];
      const shape = args[numConsts + i].aval.shape;
      for (let g = 0; g < gridRank; g++) {
        if (axes[g] !== null && gridShape[g] === 0) {
          const dim = shape[axes[g]!] as number;
          gridShape[g] = Math.ceil(dim / blockShape[g]);
          origDims[g] = dim;
        }
      }
    }
    const numBlocks = gridShape.reduce((a, b) => a * b, 1);

    // Build input tables for slicing.
    // residualAxes[i] / residualHaloTable[i]: per-residual-arg axes and halo.
    const residualAxes = inAxes.filter((_, i) => !undefMask[numConsts + i]);
    const residualHaloTable: [number, number][][] = [];
    for (let i = 0; i < args.length - numConsts; i++) {
      if (!undefMask[numConsts + i]) residualHaloTable.push(halo[i]);
    }

    // Transposed body outputs: [grad_consts..., grad_args...].
    // constOutAxes: null for all grid axes (summed across blocks).
    // argOutAxes: inAxes for each differentiated arg.
    // argHaloTable[i]: halo for each differentiated arg (from original halo).
    const nullOutAxis = Array(gridRank).fill(null) as (number | null)[];
    const constOutAxes: (number | null)[][] = [];
    for (let i = 0; i < numConsts; i++) {
      if (undefMask[i]) constOutAxes.push(nullOutAxis);
    }
    const argOutAxes: (number | null)[][] = [];
    const argHaloTable: [number, number][][] = [];
    for (let i = 0; i < args.length - numConsts; i++) {
      if (undefMask[numConsts + i]) {
        argOutAxes.push(inAxes[i]);
        argHaloTable.push(halo[i]);
      }
    }
    const allOutAxes = [...constOutAxes, ...argOutAxes];
    const allOutHalo = [
      ...constOutAxes.map(() =>
        blockShape.map(() => [0, 0] as [number, number]),
      ),
      ...argHaloTable,
    ];

    // Compute per-output accumulator shapes and body output shapes.
    const outTypes = transposedBody.jaxpr.outs.map((a) => a.aval);
    const numOutputs = outTypes.length;
    const accumShapes: number[][] = [];
    const bodyOutShapes: number[][] = [];
    for (let o = 0; o < numOutputs; o++) {
      const bodyShape = [...outTypes[o].shape] as number[];
      bodyOutShapes.push(bodyShape);
      const accumShape = [...bodyShape];
      const axes = allOutAxes[o];
      const oHalo = allOutHalo[o];
      for (let g = 0; g < gridRank; g++) {
        if (axes[g] !== null) {
          const [lo, hi] = oHalo[g];
          accumShape[axes[g]!] = origDims[g] + lo + hi;
        }
      }
      accumShapes.push(accumShape);
    }

    // Initialize accumulators to zero.
    let accumulators: Tracer[] = accumShapes.map((shape, o) =>
      zerosInternal(shape, outTypes[o].dtype),
    );

    // ---- Unrolled block loop ----
    for (let flatIdx = 0; flatIdx < numBlocks; flatIdx++) {
      // Convert flat index to grid coordinates.
      const blockIdx: number[] = new Array(gridRank);
      let remaining = flatIdx;
      for (let g = gridRank - 1; g >= 0; g--) {
        blockIdx[g] = remaining % gridShape[g];
        remaining = Math.floor(remaining / gridShape[g]);
      }

      // Slice each residual input with halo expansion + boundary padding.
      const blockResiduals: Tracer[] = [];
      for (let ri = 0; ri < argResiduals.length; ri++) {
        blockResiduals.push(
          sliceBlockWithHalo(
            argResiduals[ri],
            residualAxes[ri],
            residualHaloTable[ri],
            blockIdx,
            blockShape,
            gridRank,
          ),
        );
      }

      // Slice each cotangent (no halo, block-aligned).
      const blockCts: Tracer[] = [];
      for (let ci = 0; ci < cts.length; ci++) {
        blockCts.push(
          sliceBlock(
            cts[ci] as Tracer,
            outAxes[ci],
            blockIdx,
            blockShape,
            gridRank,
          ),
        );
      }

      // Evaluate transposed body for this block.
      const bodyOuts = evalJaxpr(transposedBody.jaxpr, [
        ...transposedBody.constEnv.values,
        ...constResiduals,
        ...blockResiduals,
        ...blockCts,
      ]);

      // Accumulate each output via pad + add.
      const newAccumulators: Tracer[] = [];
      for (let o = 0; o < numOutputs; o++) {
        const axes = allOutAxes[o];
        const bodyShape = bodyOutShapes[o];
        const accumShape = accumShapes[o];

        // Compute pad widths: place the gradient patch at the correct
        // position in the padded accumulator.
        let grad: Tracer = bodyOuts[o] as Tracer;
        const padWidth: [number, number][] = bodyShape.map(() => [0, 0]);
        let anyPad = false;
        let needsTrim = false;

        for (let g = 0; g < gridRank; g++) {
          if (axes[g] === null) continue;
          const ax = axes[g]!;
          const bDim = bodyShape[ax]; // B+lo+hi for halo, B for non-halo
          // In padded space, block j writes at offset j*B:
          // padded_accum[0] = original[-lo], padded_accum[lo] = original[0].
          // Block j covers original [j*B-lo, j*B+B+hi), which in padded
          // space is [j*B-lo+lo, j*B+B+hi+lo) = [j*B, j*B+bDim).
          // But account for the lo offset already baked into accumShape.
          const offset = blockIdx[g] * blockShape[g];
          const available = accumShape[ax] - offset;
          if (bDim > available) {
            // Last block extends past accumulator — trim gradient.
            needsTrim = true;
          }
          padWidth[ax] = [offset, Math.max(0, available - bDim)];
          anyPad = anyPad || padWidth[ax][0] !== 0 || padWidth[ax][1] !== 0;
        }

        // Trim gradient if it overflows the accumulator (partial last block).
        if (needsTrim) {
          const trimSlice: [number, number][] = bodyShape.map((s) => [0, s]);
          for (let g = 0; g < gridRank; g++) {
            if (axes[g] === null) continue;
            const ax = axes[g]!;
            const offset = blockIdx[g] * blockShape[g];
            const available = accumShape[ax] - offset;
            if (bodyShape[ax] > available) {
              trimSlice[ax] = [0, available];
              // Update padWidth: after trimming, rightPad = 0.
              padWidth[ax] = [offset, 0];
            }
          }
          const trimmed = shrink(grad, trimSlice) as Tracer;
          grad.dispose();
          grad = trimmed;
        }

        // Pad gradient to accumulator shape and add.
        if (anyPad || needsTrim) {
          const padded = pad(grad, padWidth) as Tracer;
          grad.dispose();
          grad = padded;
        }

        const updated = add(accumulators[o], grad) as Tracer;
        accumulators[o].dispose();
        grad.dispose();
        newAccumulators.push(updated);
      }
      accumulators = newAccumulators;

      // Dispose block slices and body outputs.
      for (const r of blockResiduals) r.dispose();
      for (const c of blockCts) c.dispose();
    }

    // Slice padded accumulators to remove halo padding.
    const finalOutputs: Tracer[] = [];
    for (let o = 0; o < numOutputs; o++) {
      const axes = allOutAxes[o];
      const oHalo = allOutHalo[o];
      let needsSlice = false;
      const slicePairs: [number, number][] = accumShapes[o].map((s) => [0, s]);
      for (let g = 0; g < gridRank; g++) {
        if (axes[g] === null) continue;
        const ax = axes[g]!;
        const [lo, hi] = oHalo[g];
        if (lo !== 0 || hi !== 0) {
          slicePairs[ax] = [lo, lo + origDims[g]];
          needsSlice = true;
        }
      }
      if (needsSlice) {
        const sliced = shrink(accumulators[o], slicePairs) as Tracer;
        accumulators[o].dispose();
        finalOutputs.push(sliced);
      } else {
        finalOutputs.push(accumulators[o]);
      }
    }

    // transposedBody comes from transposeJaxprCache. Its retained wrapper
    // stays cache-owned across calls; only the local padded accumulators above
    // are caller-owned and disposed in this path.

    // Assemble outputs in the correct order.
    const numUndefConsts = constOutAxes.length;
    const constCts = finalOutputs.slice(0, numUndefConsts);
    const argCts = finalOutputs.slice(numUndefConsts);

    const outVals = [];
    let constIdx = 0,
      argIdx = 0;
    for (let i = 0; i < args.length; i++) {
      if (!undefMask[i]) outVals.push(null);
      else if (i < numConsts) outVals.push(constCts[constIdx++]);
      else outVals.push(argCts[argIdx++]);
    }
    return outVals;
  },
  [Primitive.Mul]([ct], [x, y]) {
    // TODO: For transpose rules on operations that have type promotion rules,
    // make sure the gradient is cast back to the correct dtype.
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    if (x instanceof UndefPrimal) {
      const prod = mul(ct, y as Tracer);
      const result = unbroadcast(prod, x);
      if (result !== prod) prod.dispose();
      return [result, null];
    } else {
      const prod = mul(x as Tracer, ct);
      const result = unbroadcast(prod, y as UndefPrimal);
      if (result !== prod) prod.dispose();
      return [null, result];
    }
  },
  [Primitive.Cast]([ct], [x], { dtype: _dtype }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Cast);
    return [cast(ct, x.aval.dtype)];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    if (x instanceof UndefPrimal && y instanceof UndefPrimal)
      // Non-consuming: ct survives both unbroadcast calls, no .ref needed.
      return [unbroadcast(ct, x), unbroadcast(ct, y)];
    return x instanceof UndefPrimal
      ? ((y as Tracer).dispose(), [unbroadcast(ct, x), null])
      : ((x as Tracer).dispose(), [null, unbroadcast(ct, y as UndefPrimal)]);
  },
  [Primitive.Reduce]([ct], [x], { op, axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reduce);
    if (op === AluOp.Add) {
      return [broadcast(ct, x.aval.shape as number[], axis)];
    } else {
      // Forward-mode jvp of product does not involve any products.
      // The same applies to min/max as non-additive reductions.
      throw new NonlinearError(Primitive.Reduce);
    }
  },
  [Primitive.Pool]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pool);
    return bind(Primitive.PoolTranspose, [ct], {
      inShape: x.aval.shape as number[],
      window,
      strides,
    });
  },
  [Primitive.PoolTranspose]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.PoolTranspose);
    return bind(Primitive.Pool, [ct], { window, strides });
  },
  [Primitive.Dot]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Dot);
    const axisSize = generalBroadcast(x.aval.shape, y.aval.shape).slice(
      -1,
    )[0] as number;
    const ctBroad = broadcast(ct, ct.shape.concat(axisSize), [-1]); // Undo the contraction.
    if (x instanceof UndefPrimal) {
      const prod = mul(ctBroad, y as Tracer);
      const result = unbroadcast(prod, x);
      if (result !== prod) prod.dispose();
      ctBroad.dispose();
      return [result, null];
    } else {
      const prod = mul(x as Tracer, ctBroad);
      const result = unbroadcast(prod, y as UndefPrimal);
      if (result !== prod) prod.dispose();
      ctBroad.dispose();
      return [null, result];
    }
  },
  [Primitive.Conv]([ct], [lhs, rhs], params) {
    if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Conv);
    // See rules for transposing a convolution in `convolution.ts`.
    const v = params.vmapDims;
    // Permutation to swap batch/channel dims (axes v and v+1), keeping vmapDims first.
    const rev01 = [...range(v), v + 1, v, ...range(v + 2, ct.ndim)];
    if (lhs instanceof UndefPrimal) {
      // Transpose to LHS (activations).
      using kernelTransposed = transpose(rhs as Tracer, rev01) as Tracer; // Reverse in <-> out channels.
      using kernel = flip(
        kernelTransposed,
        range(v + 2, kernelTransposed.ndim),
      ) as Tracer; // Flip spatial dimensions.
      const result = conv(ct, kernel, {
        vmapDims: v,
        strides: params.lhsDilation,
        // Reference: _conv_general_vjp_lhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          // dilated kernel_size in this dimension
          const dilatedKernel =
            (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padBefore = dilatedKernel - 1 - pl;
          // Cannot calculate `padAfter = dilatedKernel - 1 - pr` because strides
          // may not produce an equal dilated kernel, for instance, 6 / stride 2
          // produces [X_X_X_], but dilating the cotangents recovers [Y_Y_Y].
          //
          // Instead, we set it to make the output shape (before strides) match
          // with dilatedLhs, currently it's less than dilatedLhs.
          const dilatedLhs =
            ((lhs.aval.shape[i + v + 2] as number) - 1) *
              params.lhsDilation[i] +
            1;
          const padAfter =
            dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
          return [padBefore, padAfter];
        }),
        lhsDilation: params.strides,
        rhsDilation: params.rhsDilation,
      });
      return [result, null];
    } else {
      // Transpose to RHS (filter).
      using newLhs = transpose(lhs as Tracer, rev01) as Tracer; // Reverse batch <-> in channels.
      using newRhs = transpose(ct, rev01) as Tracer; // Reverse batch <-> out channels.
      using convResult = conv(newLhs, newRhs, {
        vmapDims: v,
        strides: params.rhsDilation,
        // Reference: _conv_general_vjp_rhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          const dilatedLhs =
            ((lhs.aval.shape[i + v + 2] as number) - 1) *
              params.lhsDilation[i] +
            1;
          const dilatedKernel =
            ((rhs.aval.shape[i + v + 2] as number) - 1) *
              params.rhsDilation[i] +
            1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padFromLhs = dilatedCt - dilatedLhs;
          const padFromRhs = dilatedKernel - pl - 1;
          return [pl, padFromLhs + padFromRhs];
        }),
        lhsDilation: params.lhsDilation,
        rhsDilation: params.strides,
      }) as Tracer;
      const result = transpose(convResult, rev01); // Reverse in <-> out channels.
      return [null, result];
    }
  },
  [Primitive.Where]([ct], [cond, x, y]) {
    // Cotangent should be zero where cond doesn't apply.
    const cts: (Tracer | null)[] = [null, null, null];
    if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal) {
      const masked = where(cond, ct, 0);
      cts[1] = unbroadcast(masked, x);
      if (cts[1] !== masked) masked.dispose();
    } else {
      x.dispose();
    }
    if (y instanceof UndefPrimal) {
      const masked = where(cond, 0, ct);
      cts[2] = unbroadcast(masked, y);
      if (cts[2] !== masked) masked.dispose();
    } else {
      y.dispose();
    }
    // ct and cond are in internalArrays — batch cleanup handles them.
    return cts;
  },
  [Primitive.Concatenate]([ct], inputs, { axis }) {
    // The backprop of concatenate is split along `axis`.
    //
    // Inputs that are `UndefPrimal` are tangent variables (unknown in the
    // residual sense). Inputs that are concrete arrays arise from
    // `linearTangentsJvp` substituting `zeros_like(primal)` for the zero
    // tangent of a constant-inside-the-function (e.g. `np.ones(...)` called
    // inside a `grad`-traced body). These are semantically zero tangents, so
    // no gradient flows back to them — they are NOT primal values leaking into
    // the tangent computation.
    //
    // If ALL inputs are concrete (no tangent variables at all), the
    // concatenate is truly nonlinear (a primal leaked in) — throw.
    if (inputs.every((x) => !(x instanceof UndefPrimal)))
      throw new NonlinearError(Primitive.Concatenate);

    const sizes = inputs.map((x) => x.aval.shape[axis] as number);
    const splits = split(ct, axis, sizes);

    // Fast path: all inputs are tangent variables.
    if (inputs.every((x) => x instanceof UndefPrimal)) return splits;

    // Mixed case: some inputs are concrete zero-tangent constants.
    // Return the corresponding split for tangent-variable inputs, and
    // dispose the unused split + return null for known-constant inputs
    // (null signals "no cotangent needed for this known input").
    return inputs.map((inp, i) => {
      if (inp instanceof UndefPrimal) return splits[i];
      splits[i].dispose();
      return null;
    });
  },
  [Primitive.Split](cts, [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Split);
    return [concatenate(cts, axis)];
  },
  [Primitive.Gather]([ct], [x, ...indices], { axis, outDim: _outDim }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
    if (indices.some((i) => i instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Gather);
    // Transpose of gather is scatter_add: accumulate cotangent back to
    // source positions.  Handles both permutation and duplicate indices.
    const idx = indices[0] as Tracer;
    using z = zerosInternal(x.aval.shape, ct.dtype) as Tracer;
    const result = scatterAdd(z, idx, ct, axis[0]);
    return [result, null];
  },
  [Primitive.ScatterAdd]([ct], [target, indices, updates], { axis }) {
    // Transpose of scatter_add(target, indices, updates, axis):
    //   d/d(target)  = ct  (scatter_add is additive in target → identity)
    //   d/d(indices) = null (integer, non-differentiable)
    //   d/d(updates) = gather(ct, indices, axis) — reverse the scatter
    const ctTarget = target instanceof UndefPrimal ? ct : null;
    const ctUpdates =
      updates instanceof UndefPrimal
        ? (gather(ct, [indices as Tracer], [axis], axis) as Tracer)
        : null;
    return [ctTarget, null, ctUpdates];
  },
  [Primitive.Transpose]([ct], [x], { perm }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    return [transpose(ct, invertPermutation(perm))];
  },
  [Primitive.Broadcast]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduce(ct, AluOp.Add, axis)];
  },
  [Primitive.Reshape]([ct], [x], _) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reshape);
    return [reshape(ct, x.aval.shape as number[])];
  },
  [Primitive.Flip]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
    return [flip(ct, axis)];
  },
  [Primitive.Reverse]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reverse);
    return [reverse(ct, axis)];
  },
  [Primitive.Shrink]([ct], [x], { slice }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Shrink);
    const width = slice.map(
      ([s, e], i) => [s, (x.aval.shape[i] as number) - e] as [number, number],
    );
    return [pad(ct, width)];
  },
  [Primitive.Pad]([ct], [x], { width }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pad);
    const slice = width.map(
      ([s, _e], i) => [s, s + (x.aval.shape[i] as number)] as [number, number],
    );
    return [shrink(ct, slice)];
  },
  [Primitive.TriangularSolve]([ct], [a, b], { unitDiagonal }) {
    if (a instanceof UndefPrimal || !(b instanceof UndefPrimal))
      throw new NonlinearError(Primitive.TriangularSolve);
    // The adjoint of solving a @ x.T = b.T for x, when differentiating w.r.t. b:
    //   If forward is: x.T = a^{-1} @ b.T
    //   Then adjoint is: ct_b.T = a^{-T} @ ct_x.T, so we just transpose A
    // Note: The primitive always operates on upper triangular a, so a^T is lower.
    using aT = moveaxis(a, -2, -1) as Tracer;
    const ctB = triSolve(aT, ct, {
      lower: true,
      unitDiagonal,
    });
    return [null, ctB];
  },
  [Primitive.Jit](cts, args, { name, jaxpr }) {
    // We need this one because the jvp() rule for Jit generates a Jit
    // with the transformed Jaxpr. So grad-of-jit will result in a transposed
    // Jit, which we need to handle.
    const undefPrimals = args.map((x) => x instanceof UndefPrimal);
    const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
    const residuals = args.filter((x, i) => !undefPrimals[i]) as Tracer[];
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.constEnv.values.map((c) => c.ref), ...residuals, ...cts],
      {
        name: `${name}_t`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.constEnv.length,
      },
    );
    // Now pull cotangents back to the corresponding UndefPrimal inputs.
    let i = 0;
    return undefPrimals.map((isUndef) => (isUndef ? outs[i++] : null));
  },
  [Primitive.DynamicUpdateSlice]([ct], [dst, src], { offset, axis }) {
    // DUS is linear in both dst and src:
    //   output = dst everywhere except slice at [offset..offset+len] on axis, where output = src
    // Transpose:
    //   ct_dst = DUS(ct, zeros_like_src, offset, axis)  — zero out the slice region
    //   ct_src = shrink(ct, slice_ranges)               — extract the slice region
    let ctDst: Tracer | null = null;
    let ctSrc: Tracer | null = null;

    const srcShape = (
      src instanceof UndefPrimal ? src.aval.shape : (src as Tracer).shape
    ) as number[];
    const dstShape = (
      dst instanceof UndefPrimal ? dst.aval.shape : (dst as Tracer).shape
    ) as number[];

    if (dst instanceof UndefPrimal) {
      using z = zerosInternal(srcShape, ct.dtype) as Tracer;
      ctDst = dynamicUpdateSlice(ct, z, offset, axis) as Tracer;
    }

    if (src instanceof UndefPrimal) {
      if (srcShape.length === dstShape.length) {
        // Same-rank case: extract the slice from ct
        const slices = dstShape.map(
          (s, i) =>
            (i === axis ? [offset, offset + srcShape[i]] : [0, s]) as [
              number,
              number,
            ],
        );
        ctSrc = shrink(ct, slices) as Tracer;
      } else {
        // Stacked case (axis=0, src.ndim = dst.ndim - 1): extract and reshape
        const slices = dstShape.map(
          (s, i) =>
            (i === 0 ? [offset, offset + 1] : [0, s]) as [number, number],
        );
        using sliced = shrink(ct, slices) as Tracer;
        ctSrc = reshape(sliced, srcShape) as Tracer;
      }
    }

    return [ctDst, ctSrc];
  },
  [Primitive.DynamicUpdateSliceGeneral]([ct], [dst, src], { startIndices }) {
    // ND DUS is linear in both dst and src (same-rank only):
    //   ct_dst = DUSGeneral(ct, zeros_like_src, startIndices)
    //   ct_src = shrink(ct, per-axis ranges)
    let ctDst: Tracer | null = null;
    let ctSrc: Tracer | null = null;

    const srcShape = (
      src instanceof UndefPrimal ? src.aval.shape : (src as Tracer).shape
    ) as number[];
    const dstShape = (
      dst instanceof UndefPrimal ? dst.aval.shape : (dst as Tracer).shape
    ) as number[];

    if (dst instanceof UndefPrimal) {
      using z = zerosInternal(srcShape, ct.dtype) as Tracer;
      ctDst = dynamicUpdateSliceGeneral(ct, z, startIndices) as Tracer;
    }

    if (src instanceof UndefPrimal) {
      const slices = dstShape.map(
        (s, i) =>
          [startIndices[i], startIndices[i] + srcShape[i]] as [number, number],
      );
      ctSrc = shrink(ct, slices) as Tracer;
    }

    return [ctDst, ctSrc];
  },
  [Primitive.Scan](
    cts,
    args,
    {
      jaxpr,
      numCarry,
      numConsts,
      length: lengthDim,
      reverse,
      checkpoint: _checkpoint,
      isJvpTransformed,
    },
  ) {
    // Scan transpose rule for backward pass through scan.
    //
    // Emits two in-graph Primitive.Scan calls (forward + backward) rather
    // than imperative JS loop unrolling, so the backend compiler handles
    // iteration.  This supports dynamic/symbolic scan lengths.

    const numX = args.length - numConsts - numCarry;
    const numY = cts.length - numCarry;

    const isJvpScan = isJvpTransformed ?? false;

    const numPrimalCarry = isJvpScan ? numCarry / 2 : 0;
    const numPrimalX = isJvpScan ? numX / 2 : 0;
    const numPrimalY = isJvpScan ? numY / 2 : 0;

    // Identify which args are effectively UndefPrimal (need cotangents)
    const undefMask = args.map((x, i) => {
      if (x instanceof UndefPrimal) return true;
      if (!isJvpScan) return false;

      // JVP structure: second half of each group is tangent
      if (i < numConsts) {
        return false;
      } else if (i < numConsts + numCarry) {
        const carryIdx = i - numConsts;
        return carryIdx >= numPrimalCarry;
      } else {
        const xIdx = i - numConsts - numCarry;
        return xIdx >= numPrimalX;
      }
    });

    const bodyNumConsts = numConsts;
    const bodyNumCarry = numCarry;

    // Body input layout: [consts..., carry..., x...]
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < jaxpr.inBinders.length; i++) {
      if (i < bodyNumConsts) {
        bodyUndefPrimals.push(undefMask[i]);
      } else if (i < bodyNumConsts + bodyNumCarry) {
        bodyUndefPrimals.push(undefMask[numConsts + (i - bodyNumConsts)]);
      } else {
        bodyUndefPrimals.push(
          undefMask[numConsts + numCarry + (i - bodyNumConsts - bodyNumCarry)],
        );
      }
    }

    // Get residual (primal) values from scan args
    const constArgs = args.slice(0, numConsts);
    const carryArgs = args.slice(numConsts, numConsts + numCarry);
    const xsArgs = args.slice(numConsts + numCarry);

    // Split into primal and tangent parts
    const constResiduals = constArgs.filter(
      (_, i) => !undefMask[i],
    ) as Tracer[];
    const carryResiduals = carryArgs.filter(
      (_, i) => !undefMask[numConsts + i],
    ) as Tracer[];
    const xsResiduals = xsArgs.filter(
      (_, i) => !undefMask[numConsts + numCarry + i],
    ) as Tracer[];

    const actualNumPrimalCarry = isJvpScan
      ? numPrimalCarry
      : carryArgs.map((_, i) => !undefMask[numConsts + i]).filter((x) => x)
          .length;

    if (actualNumPrimalCarry === 0 || carryResiduals.length === 0) {
      throw new Error(
        "Scan transpose: no carry residuals available. grad() through scan " +
          "requires primal carry values to be available as residuals.",
      );
    }

    // ---- Build compile-time spec ----

    // Forward-only body jaxpr
    const forwardInTypes = jaxpr.inBinders
      .filter((_, i) => !bodyUndefPrimals[i])
      .map((v) => v.aval);

    const { jaxpr: primalForwardJaxpr } = makeJaxpr(
      (...primalInputs: Tracer[]): Tracer[] => {
        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            const aval = jaxpr.inBinders[i].aval;
            fullInputs.push(zerosInternal(aval.shape, aval.dtype));
          } else {
            fullInputs.push(primalInputs[primalIdx++]);
          }
        }
        const outs = evalJaxpr(jaxpr, fullInputs);
        const primalCarryOuts = outs.slice(0, numPrimalCarry);
        const primalYOuts = outs.slice(
          numCarry,
          numCarry + Math.floor(numY / 2),
        );
        for (let i = numPrimalCarry; i < numCarry; i++) outs[i].dispose();
        for (let i = numCarry + Math.floor(numY / 2); i < outs.length; i++)
          outs[i].dispose();
        return [...primalCarryOuts, ...primalYOuts];
      },
      { validateRefs: false },
    )(...forwardInTypes);

    // Tangent-only body jaxpr
    const numTangentConsts = numConsts - constResiduals.length;
    const numTangentCarry = numCarry - numPrimalCarry;
    const numTangentX = numX - numPrimalX;

    const tangentBodyInAvals = [
      ...jaxpr.inBinders
        .filter((_, i) => !bodyUndefPrimals[i])
        .map((v) => v.aval),
      ...jaxpr.inBinders
        .filter((_, i) => bodyUndefPrimals[i])
        .map((v) => v.aval),
    ];

    const { jaxpr: tangentBody } = makeJaxpr(
      (...tangentBodyArgs: Tracer[]): Tracer[] => {
        const numPrimalInputs = jaxpr.inBinders.filter(
          (_, i) => !bodyUndefPrimals[i],
        ).length;
        const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
        const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);

        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        let tangentIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            fullInputs.push(tangentInputs[tangentIdx++]);
          } else {
            fullInputs.push(primalResiduals[primalIdx++]);
          }
        }

        const fullOuts = evalJaxpr(jaxpr, fullInputs);

        const tangentOuts: Tracer[] = [];
        for (let i = numPrimalCarry; i < numCarry; i++) {
          tangentOuts.push(fullOuts[i]);
        }
        for (let i = numCarry + numPrimalY; i < fullOuts.length; i++) {
          tangentOuts.push(fullOuts[i]);
        }

        for (let i = 0; i < numPrimalCarry; i++) fullOuts[i].dispose();
        for (let i = numCarry; i < numCarry + numPrimalY; i++)
          fullOuts[i].dispose();

        return tangentOuts;
      },
      { validateRefs: false },
    )(...tangentBodyInAvals);

    // Transpose the tangent body (cache-owned result)
    const tangentBodyUndefPrimals = [
      ...Array(
        tangentBody.jaxpr.inBinders.length -
          (numTangentConsts + numTangentCarry + numTangentX),
      ).fill(false),
      ...Array(numTangentConsts + numTangentCarry + numTangentX).fill(true),
    ];

    const transposedBody = transposeJaxpr(
      tangentBody.jaxpr,
      tangentBodyUndefPrimals,
    );

    const actualUndefMask = args.map((x) => x instanceof UndefPrimal);

    // ---- Forward Scan: replay primal forward, stacking carries as ys ----
    //
    // The backward step at iteration k needs primalCarry[k] (carry BEFORE
    // step k).  We emit a forward Scan whose ys capture those snapshots.
    //
    // Body signature: (carry=primalCarry, x=xSlice) → (newCarry, oldCarry)
    //   consts = [primalForwardJaxpr.consts, constResiduals]   (captured)
    //   carry  = primalCarry  (size numPrimalCarry)
    //   xs     = xsResiduals  (sliced per iteration)
    //   ys     = oldCarry snapshot  (stacked, for backward)

    const fwdBodyInAvals = [
      ...carryResiduals.map((t) => t.aval),
      ...xsResiduals.map(
        (t) => new ShapedArray(t.shape.slice(1), t.dtype, false),
      ),
    ];

    const { jaxpr: fwdScanBody } = makeJaxpr(
      (...fwdArgs: Tracer[]): Tracer[] => {
        const carryIn = fwdArgs.slice(0, carryResiduals.length);
        const xIn = fwdArgs.slice(carryResiduals.length);

        // Build full primal input for primalForwardJaxpr
        const fwdFullInputs: Tracer[] = [];
        using fwdDisposables = new DisposableStack();
        let constIdx = 0;
        let carryIdx = 0;
        let xIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            const aval = jaxpr.inBinders[i].aval;
            const z = fwdDisposables.use(zerosInternal(aval.shape, aval.dtype));
            fwdFullInputs.push(z);
          } else if (i < numConsts) {
            fwdFullInputs.push(constResiduals[constIdx++]);
          } else if (i < numConsts + numCarry) {
            fwdFullInputs.push(carryIn[carryIdx++]);
          } else {
            fwdFullInputs.push(xIn[xIdx++]);
          }
        }

        const fwdOuts = evalJaxpr(jaxpr, fwdFullInputs);
        const newCarry = fwdOuts.slice(0, numPrimalCarry);
        // Dispose tangent carry and all ys (not needed)
        for (let i = numPrimalCarry; i < fwdOuts.length; i++)
          fwdOuts[i].dispose();

        // carry output = newCarry, ys = oldCarry (snapshot before this step)
        return [...newCarry, ...carryIn];
      },
      { validateRefs: false },
    )(...fwdBodyInAvals);

    const fwdScanOuts = bind(
      Primitive.Scan,
      [...fwdScanBody.consts, ...carryResiduals, ...xsResiduals],
      {
        jaxpr: fwdScanBody.jaxpr,
        numConsts: fwdScanBody.consts.length,
        numCarry: carryResiduals.length,
        length: lengthDim,
        reverse,
      },
    ) as Tracer[];

    // ys = stacked old carries (one per iteration)
    const stackedPrimalCarries = fwdScanOuts.slice(carryResiduals.length);
    // Dispose final carry (not needed)
    for (let i = 0; i < carryResiduals.length; i++) fwdScanOuts[i].dispose();

    // ---- Backward Scan: accumulate cotangents in reverse ----
    //
    // Body signature:
    //   carry = [ctConstsAccum..., ctCarryRunning...]
    //   xs    = [primalCarry_k, xSlice_k, ctY_k]       (per-iteration)
    //   ys    = [ctXStep_k]                              (stacked)
    //
    // Runs transposedBody per step, accumulates const cotangents.

    const ctCarryAll = cts.slice(0, numCarry);
    const ctYsAll = cts.slice(numCarry);
    // In JVP-doubled scan, first half of carry cts are primal
    // Do NOT dispose cts-derived values here — they may be external
    // cotangents owned by evalJaxprTransposed (the caller). Internal ones
    // are tracked in internalArrays and cleaned up by the caller.
    const ctCarryInit = ctCarryAll.slice(numPrimalCarry);
    // Tangent ys are the second half
    const numYPrimal = Math.floor(numY / 2);
    const tangentCtYs = ctYsAll.slice(numYPrimal);

    // Tangent const avals (for zero-init of accumulator)
    const numPrimalInputs = jaxpr.inBinders.filter(
      (_, i) => !bodyUndefPrimals[i],
    ).length;
    const tangentConstsAvals = tangentBodyInAvals.slice(
      numPrimalInputs,
      numPrimalInputs + numTangentConsts,
    );
    const ctConstsZeroInit = Array.from({ length: numTangentConsts }, (_, i) =>
      zerosInternal(tangentConstsAvals[i].shape, tangentConstsAvals[i].dtype),
    );
    const ctCarrySafeInit = ctCarryInit.map((c, i) => {
      if (c) {
        return c;
      } else {
        return zerosInternal(
          carryResiduals[carryResiduals.length - numTangentCarry + i].shape,
          carryResiduals[carryResiduals.length - numTangentCarry + i].dtype,
        );
      }
    });

    // Build backward scan body avals
    const bwdCarryAvals = [
      ...ctConstsZeroInit.map((t) => t.aval),
      ...ctCarrySafeInit.map((t) => t.aval),
    ];
    const bwdXsAvals = [
      ...stackedPrimalCarries.map(
        (t) => new ShapedArray(t.shape.slice(1), t.dtype, false),
      ),
      ...xsResiduals.map(
        (t) => new ShapedArray(t.shape.slice(1), t.dtype, false),
      ),
      ...tangentCtYs.map(
        (t) => new ShapedArray(t.shape.slice(1), t.dtype, false),
      ),
    ];
    const bwdBodyInAvals = [...bwdCarryAvals, ...bwdXsAvals];

    const { jaxpr: bwdScanBody } = makeJaxpr(
      (...bwdArgs: Tracer[]): Tracer[] => {
        // Unpack carry
        const ctConstsIn = bwdArgs.slice(0, numTangentConsts);
        const ctCarryIn = bwdArgs.slice(
          numTangentConsts,
          numTangentConsts + numTangentCarry,
        );

        // Unpack xs
        let off = numTangentConsts + numTangentCarry;
        const primalCarry_k = bwdArgs.slice(off, off + carryResiduals.length);
        off += carryResiduals.length;
        const xSlice_k = bwdArgs.slice(off, off + xsResiduals.length);
        off += xsResiduals.length;
        const ctY_k = bwdArgs.slice(off);

        // Build transposedBody input: [tb.consts, constResiduals, primalCarry_k, xSlice_k, ctCarryIn, ctY_k]
        const transposedInputs = [
          ...transposedBody.constEnv.values,
          ...constResiduals,
          ...primalCarry_k,
          ...xSlice_k,
          ...ctCarryIn,
          ...ctY_k,
        ];
        const transposedOuts = evalJaxpr(
          transposedBody.jaxpr,
          transposedInputs,
        );

        // Extract cotangents from transposed outputs
        const ctConstsStep = transposedOuts.slice(0, numTangentConsts);
        const ctCarryNext = transposedOuts.slice(
          numTangentConsts,
          numTangentConsts + numTangentCarry,
        );
        const ctXStep = transposedOuts.slice(
          numTangentConsts + numTangentCarry,
        );

        // Accumulate const cotangents
        const ctConstsOut = ctConstsIn.map((c, i) => {
          const sum = bind(Primitive.Add, [c, ctConstsStep[i]])[0];
          ctConstsStep[i].dispose();
          return sum;
        });

        // carry = [ctConstsOut, ctCarryNext], ys = [ctXStep]
        return [...ctConstsOut, ...ctCarryNext, ...ctXStep];
      },
      { validateRefs: false },
    )(...bwdBodyInAvals);

    const bwdScanOuts = bind(
      Primitive.Scan,
      [
        ...bwdScanBody.consts,
        ...ctConstsZeroInit,
        ...ctCarrySafeInit,
        ...stackedPrimalCarries,
        ...xsResiduals,
        ...tangentCtYs,
      ],
      {
        jaxpr: bwdScanBody.jaxpr,
        numConsts: bwdScanBody.consts.length,
        numCarry: numTangentConsts + numTangentCarry,
        length: lengthDim,
        reverse: !reverse,
      },
    ) as Tracer[];

    // Extract results from backward scan
    const finalCtConsts = bwdScanOuts.slice(0, numTangentConsts);
    const finalCtCarry = bwdScanOuts.slice(
      numTangentConsts,
      numTangentConsts + numTangentCarry,
    );
    const ctXsStacked = bwdScanOuts.slice(numTangentConsts + numTangentCarry);

    // ---- Assemble output cotangents ----
    const result: (Tracer | null)[] = [];
    let ctConstIdx = 0;
    let ctCarryIdx = 0;
    let ctXIdx = 0;

    for (let i = 0; i < actualUndefMask.length; i++) {
      const isJvpTangent = undefMask[i];

      if (!actualUndefMask[i]) {
        if (isJvpTangent) {
          if (i < numConsts) {
            finalCtConsts[ctConstIdx++].dispose();
          } else if (i < numConsts + numCarry) {
            finalCtCarry[ctCarryIdx++].dispose();
          } else {
            ctXsStacked[ctXIdx++].dispose();
          }
        }
        result.push(null);
      } else if (i < numConsts) {
        result.push(finalCtConsts[ctConstIdx++]);
      } else if (i < numConsts + numCarry) {
        result.push(finalCtCarry[ctCarryIdx++]);
      } else {
        result.push(ctXsStacked[ctXIdx++]);
      }
    }

    // Dispose remaining (unused indices)
    for (let i = ctConstIdx; i < finalCtConsts.length; i++)
      finalCtConsts[i].dispose();
    for (let i = ctCarryIdx; i < finalCtCarry.length; i++)
      finalCtCarry[i].dispose();
    for (let i = ctXIdx; i < ctXsStacked.length; i++) ctXsStacked[i].dispose();

    // Dispose intermediate arrays owned by this rule.
    // stackedPrimalCarries: created by the forward scan bind (owned).
    // ctConstsZeroInit: freshly-created zeros (owned).
    // ctCarrySafeInit: only dispose entries we created (not borrowed cts).
    // tangentCtYs: borrowed from cts — NOT disposed (caller owns them).
    for (const t of stackedPrimalCarries) t.dispose();
    for (const t of ctConstsZeroInit) t.dispose();
    for (let i = 0; i < ctCarrySafeInit.length; i++) {
      if (ctCarrySafeInit[i] !== ctCarryInit[i]) ctCarrySafeInit[i].dispose();
    }

    // Clean up owned jaxprs
    tangentBody.dispose();
    fwdScanBody.dispose();
    bwdScanBody.dispose();
    primalForwardJaxpr.dispose();

    return result;
  },
  [Primitive.AssociativeScan](cts, args, params) {
    // AssociativeScan transpose: reverse sequential recurrence via Scan.
    //
    // Instead of O(N) loop unrolling (creating huge jaxprs and O(N) dispatch
    // steps), this uses two Primitive.Scan calls:
    //   1. Forward scan to recompute prefix results (stacked as ys)
    //   2. Backward scan (reverse=true) to compute cotangents
    //
    // This produces a compact jaxpr with two Scan equations, enabling
    // compiled-loop execution (constant-size program, O(1) JIT steps).

    const {
      jaxpr: bodyJaxpr,
      numLeaves,
      axis: _axis,
      reverse: assocReverse,
    } = params as PrimitiveParams<Primitive.AssociativeScan>;
    const numConsts = args.length - numLeaves;
    const numOrigLeaves = numLeaves / 2;

    // Build body undef mask: which body inputs are tangent?
    const undefMask = args.map((x) => x instanceof UndefPrimal);
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < bodyJaxpr.inBinders.length; i++) {
      if (i < numConsts) {
        bodyUndefPrimals.push(undefMask[i]);
      } else if (i < numConsts + numLeaves) {
        bodyUndefPrimals.push(i - numConsts >= numOrigLeaves);
      } else {
        bodyUndefPrimals.push(i - numConsts - numLeaves >= numOrigLeaves);
      }
    }

    const numTangentConsts = bodyUndefPrimals
      .slice(0, numConsts)
      .filter((x) => x).length;

    // Transpose the full body jaxpr (cache-owned)
    const transposedBody = transposeJaxpr(bodyJaxpr, bodyUndefPrimals);

    // Get primal residuals
    const constResiduals = args
      .slice(0, numConsts)
      .filter((_, i) => !undefMask[i]) as Tracer[];
    let primalElems = args.slice(
      numConsts,
      numConsts + numOrigLeaves,
    ) as Tracer[];

    const N = primalElems[0].shape[0] as number; // axis is always 0

    // For reverse=true, flip primal elems so forward recompute and backward
    // loop work in "forward order".
    if (assocReverse) {
      primalElems = primalElems.map((e) => flip(e, [0]));
    }

    // Helper: slice arr[idx] removing leading dim
    const sliceAt = (arr: Tracer, idx: number): Tracer => {
      const ranges: [number, number][] = [[idx, idx + 1]];
      for (let d = 1; d < arr.ndim; d++)
        ranges.push([0, arr.shape[d] as number]);
      const sl = shrink(arr, ranges);
      const r = reshape(sl, arr.shape.slice(1) as number[]);
      sl.dispose();
      return r;
    };

    // Helper: slice arr[from:] along axis 0
    const sliceFrom = (arr: Tracer, from: number): Tracer => {
      const ranges: [number, number][] = [[from, arr.shape[0] as number]];
      for (let d = 1; d < arr.ndim; d++)
        ranges.push([0, arr.shape[d] as number]);
      return shrink(arr, ranges);
    };

    // ----- Forward Scan: recompute prefix results as stacked ys -----
    //
    // Carry = prev_y (numOrigLeaves)
    // Xs = primalElems[1:] per step (N-1 iterations)
    // Body: (prev_y, x_i) => [new_y=body(prev_y,x_i), ys=prev_y_snapshot]
    //   ys capture the carry BEFORE each step (needed by backward)

    const fwdInitCarry = primalElems.map((e) => sliceAt(e, 0));
    const fwdXs = primalElems.map((e) => sliceFrom(e, 1)); // [N-1, ...]

    const fwdBodyInAvals = [
      ...fwdInitCarry.map((t) => t.aval), // carry avals
      ...fwdXs.map(
        (t) => new ShapedArray(t.shape.slice(1) as number[], t.dtype, false),
      ), // x per step avals
    ];

    const { jaxpr: fwdBody } = makeJaxpr(
      (...fwdArgs: Tracer[]): Tracer[] => {
        const carry = fwdArgs.slice(0, numOrigLeaves);
        const xSlice = fwdArgs.slice(numOrigLeaves);

        // Evaluate primal body: newY = body(carry, xSlice) with zero tangents
        const bodyArgs: Tracer[] = [];
        using disposables = new DisposableStack();
        let cri = 0;
        for (let i = 0; i < numConsts; i++) {
          if (bodyUndefPrimals[i]) {
            const z = disposables.use(
              zerosInternal(
                bodyJaxpr.inBinders[i].aval.shape,
                bodyJaxpr.inBinders[i].aval.dtype,
              ),
            );
            bodyArgs.push(z);
          } else {
            bodyArgs.push(constResiduals[cri++]);
          }
        }
        for (let i = 0; i < numOrigLeaves; i++) bodyArgs.push(carry[i]);
        for (let i = numOrigLeaves; i < numLeaves; i++) {
          const z = disposables.use(
            zerosInternal(
              bodyJaxpr.inBinders[numConsts + i].aval.shape,
              bodyJaxpr.inBinders[numConsts + i].aval.dtype,
            ),
          );
          bodyArgs.push(z);
        }
        for (let i = 0; i < numOrigLeaves; i++) bodyArgs.push(xSlice[i]);
        for (let i = numOrigLeaves; i < numLeaves; i++) {
          const z = disposables.use(
            zerosInternal(
              bodyJaxpr.inBinders[numConsts + numLeaves + i].aval.shape,
              bodyJaxpr.inBinders[numConsts + numLeaves + i].aval.dtype,
            ),
          );
          bodyArgs.push(z);
        }

        const outs = evalJaxpr(bodyJaxpr, bodyArgs);
        const primalOuts = outs.slice(0, numOrigLeaves);
        for (let i = numOrigLeaves; i < outs.length; i++) outs[i].dispose();

        // carry = newY, ys = newY passthrough (enables compiled-loop)
        return [...primalOuts, ...primalOuts];
      },
      { validateRefs: false },
    )(...fwdBodyInAvals);

    const fwdScanOuts = bind(
      Primitive.Scan,
      [...fwdBody.consts, ...fwdInitCarry, ...fwdXs],
      {
        jaxpr: fwdBody.jaxpr,
        numConsts: fwdBody.consts.length,
        numCarry: numOrigLeaves,
        length: N - 1,
        reverse: false,
      },
    ) as Tracer[];

    // Stacked Ys are [y[1], ..., y[N-1]] (carry output at each iteration).
    // We need stackedPrevYs = [y[0], ..., y[N-2]] (carry input at each step).
    // Reconstruct by prepending initCarry and removing the last element.
    const stackedAfterYs = fwdScanOuts.slice(numOrigLeaves);
    for (let i = 0; i < numOrigLeaves; i++) fwdScanOuts[i].dispose();

    const stackedPrevYs: Tracer[] = [];
    for (let j = 0; j < numOrigLeaves; j++) {
      const leafShape = fwdInitCarry[j].shape as number[];
      const y0 = broadcast(fwdInitCarry[j], [1, ...leafShape], [0]);
      const rest = shrink(stackedAfterYs[j], [
        [0, N - 2],
        ...leafShape.map((s) => [0, s] as [number, number]),
      ]);
      const prev = concatenate([y0, rest], 0);
      y0.dispose();
      rest.dispose();
      stackedAfterYs[j].dispose();
      stackedPrevYs.push(prev);
    }

    // ----- Split cotangents -----
    const ctPrimal = cts.slice(0, numOrigLeaves);
    let ctTangent = cts.slice(numOrigLeaves);
    const disposedCts = new Set<Tracer>();
    for (const c of ctPrimal) {
      if (!disposedCts.has(c)) {
        disposedCts.add(c);
        c.dispose();
      }
    }
    if (assocReverse) {
      ctTangent = ctTangent.map((c) => flip(c, [0]));
    }

    // ----- Backward Scan: accumulate cotangents in reverse -----
    //
    // Carry = [ctConstsAccum (numTangentConsts), ctCarry (numOrigLeaves)]
    // Xs = [stackedPrevYs[k], primalElems[1:][k], ctTangent[1:][k]]
    //   k=0..N-2 → original positions 1..N-1
    // reverse=true: processes from position N-1 down to 1
    // Ys = [ctBIter (numOrigLeaves)] per step

    const bwdPrimalSliced = primalElems.map((e) => sliceFrom(e, 1));
    const bwdCtSliced = ctTangent.map((c) => sliceFrom(c, 1));

    // Leaf element shapes (carry shapes) — from forward init carry
    const leafElemAvals = fwdBodyInAvals.slice(0, numOrigLeaves);

    // Tangent const shapes from transposed body output layout
    const ctConstsAvals: ShapedArray[] = [];
    for (let j = 0; j < numTangentConsts; j++) {
      ctConstsAvals.push(transposedBody.jaxpr.outs[j].aval);
    }

    const ctConstsZeroInit = ctConstsAvals.map((a) =>
      zerosInternal(a.shape, a.dtype),
    );
    const ctCarryInit = leafElemAvals.map((a) =>
      zerosInternal(a.shape, a.dtype),
    );

    const bwdCarryAvals = [
      ...ctConstsZeroInit.map((t) => t.aval),
      ...ctCarryInit.map((t) => t.aval),
    ];
    const bwdXsAvals = [
      ...stackedPrevYs.map(
        (t) => new ShapedArray(t.shape.slice(1) as number[], t.dtype, false),
      ),
      ...bwdPrimalSliced.map(
        (t) => new ShapedArray(t.shape.slice(1) as number[], t.dtype, false),
      ),
      ...bwdCtSliced.map(
        (t) => new ShapedArray(t.shape.slice(1) as number[], t.dtype, false),
      ),
    ];

    const { jaxpr: bwdBody } = makeJaxpr(
      (...bwdArgs: Tracer[]): Tracer[] => {
        // Unpack carry
        const ctConstsIn = bwdArgs.slice(0, numTangentConsts);
        const ctCarryIn = bwdArgs.slice(
          numTangentConsts,
          numTangentConsts + numOrigLeaves,
        );

        // Unpack xs
        let off = numTangentConsts + numOrigLeaves;
        const yPrev = bwdArgs.slice(off, off + numOrigLeaves);
        off += numOrigLeaves;
        const xPi = bwdArgs.slice(off, off + numOrigLeaves);
        off += numOrigLeaves;
        const ctYi = bwdArgs.slice(off, off + numOrigLeaves);

        // Effective cotangent = ctCarry + ctYi
        const ctEff = ctCarryIn.map((c, j) => add(c, ctYi[j]));

        // Build transposed body inputs:
        // [tb.consts, constResiduals, yPrev, xPi, zeros_primal_ct, ctEff]
        const tbInputs: Tracer[] = [
          ...transposedBody.constEnv.values,
          ...constResiduals,
        ];
        using tbDisposables = new DisposableStack();
        for (const y of yPrev) tbInputs.push(y);
        for (const x of xPi) tbInputs.push(x);
        for (let j = 0; j < numOrigLeaves; j++) {
          const z = tbDisposables.use(
            zerosInternal(leafElemAvals[j].shape, leafElemAvals[j].dtype),
          );
          tbInputs.push(z);
        }
        for (const c of ctEff) tbInputs.push(c);

        const tbOuts = evalJaxpr(transposedBody.jaxpr, tbInputs);
        for (const c of ctEff) c.dispose();

        // Extract: [ctConstsI, ctANew, ctBIter]
        let oi = 0;
        const ctConstsStep: Tracer[] = [];
        for (let j = 0; j < numTangentConsts; j++)
          ctConstsStep.push(tbOuts[oi++]);
        const ctANew: Tracer[] = [];
        for (let j = 0; j < numOrigLeaves; j++) ctANew.push(tbOuts[oi++]);
        const ctBIter: Tracer[] = [];
        for (let j = 0; j < numOrigLeaves; j++) ctBIter.push(tbOuts[oi++]);

        // Accumulate const cotangents
        const ctConstsOut = ctConstsIn.map((c, i) => {
          const s = add(c, ctConstsStep[i]);
          ctConstsStep[i].dispose();
          return s;
        });

        // carry = [ctConstsOut, ctANew], ys = [ctBIter]
        return [...ctConstsOut, ...ctANew, ...ctBIter];
      },
      { validateRefs: false },
    )(...[...bwdCarryAvals, ...bwdXsAvals]);

    const bwdScanOuts = bind(
      Primitive.Scan,
      [
        ...bwdBody.consts,
        ...ctConstsZeroInit,
        ...ctCarryInit,
        ...stackedPrevYs,
        ...bwdPrimalSliced,
        ...bwdCtSliced,
      ],
      {
        jaxpr: bwdBody.jaxpr,
        numConsts: bwdBody.consts.length,
        numCarry: numTangentConsts + numOrigLeaves,
        length: N - 1,
        reverse: true,
      },
    ) as Tracer[];

    // Extract backward scan results
    const finalCtConsts = bwdScanOuts.slice(0, numTangentConsts);
    const finalCtCarry = bwdScanOuts.slice(
      numTangentConsts,
      numTangentConsts + numOrigLeaves,
    );
    // bwdYsStacked[j] shape [N-1, ...leaf_j] — cts for positions 1..N-1
    const bwdYsStacked = bwdScanOuts.slice(numTangentConsts + numOrigLeaves);

    // Position 0: ctX0 = finalCtCarry + ctY0
    const ctY0 = ctTangent.map((c) => sliceAt(c, 0));
    const ctX0 = finalCtCarry.map((c, j) => {
      const s = add(c, ctY0[j]);
      c.dispose();
      ctY0[j].dispose();
      return s;
    });

    // Stack: prepend ctX0 to backward ys for full [N, ...] result
    const ctXsStacked: Tracer[] = [];
    for (let j = 0; j < numOrigLeaves; j++) {
      const x0Exp = broadcast(ctX0[j], [1, ...ctX0[j].shape], [0]);
      let stacked = concatenate([x0Exp, bwdYsStacked[j]], 0);
      if (assocReverse) {
        const flipped = flip(stacked, [0]);
        stacked.dispose();
        stacked = flipped;
      }
      x0Exp.dispose();
      ctX0[j].dispose();
      bwdYsStacked[j].dispose();
      ctXsStacked.push(stacked);
    }

    // Dispose intermediates
    for (const t of stackedPrevYs) t.dispose();
    for (const t of bwdPrimalSliced) t.dispose();
    for (const t of bwdCtSliced) t.dispose();
    for (const t of ctConstsZeroInit) t.dispose();
    for (const t of ctCarryInit) t.dispose();
    for (const x of fwdXs) x.dispose();
    for (const c of fwdInitCarry) c.dispose();
    for (const ct of ctTangent) {
      if (!disposedCts.has(ct)) {
        disposedCts.add(ct);
        ct.dispose();
      }
    }
    if (assocReverse) for (const e of primalElems) e.dispose();

    // Clean up owned jaxprs
    fwdBody.dispose();
    bwdBody.dispose();

    // Assemble output: null for known primals, cotangent for unknowns.
    const result: (Tracer | null)[] = [];
    let ctCI = 0;
    for (let i = 0; i < args.length; i++) {
      if (!(args[i] instanceof UndefPrimal)) {
        result.push(null);
      } else if (i < numConsts) {
        result.push(finalCtConsts[ctCI++]);
      } else {
        // Tangent leaf at structural index within the elems
        const tangentLeafIdx = i - numConsts - numOrigLeaves;
        result.push(ctXsStacked[tangentLeafIdx]);
      }
    }

    // Dispose unused ctXsStacked entries
    const consumedIdxs = new Set<number>();
    for (let i = 0; i < args.length; i++) {
      if (args[i] instanceof UndefPrimal && i >= numConsts) {
        consumedIdxs.add(i - numConsts - numOrigLeaves);
      }
    }
    for (let j = 0; j < ctXsStacked.length; j++) {
      if (!consumedIdxs.has(j)) ctXsStacked[j].dispose();
    }
    // Dispose unused finalCtConsts
    for (let j = ctCI; j < finalCtConsts.length; j++)
      finalCtConsts[j].dispose();

    return result;
  },
  [Primitive.WorkgroupAssociativeScan](cts, args, params) {
    // WorkgroupAssociativeScan transpose: same reverse sequential recurrence
    // as AssociativeScan but simpler (no axis/reverse, single-leaf groups).
    // This runs inside a BlockMap body during grad.
    //
    // Forward:  y[0] = x[0],  y[i] = body(y[i-1], x[i])  for i=1..N-1
    // Backward: iterate N-1..1, transposing body at each position.

    const { jaxpr: bodyJaxpr, numConsts } =
      params as PrimitiveParams<Primitive.WorkgroupAssociativeScan>;
    const numElems = (args.length - numConsts) / 2;
    const numOrigElems = numElems; // numElems is already halved after JVP doubling

    const undefMask = args.map((x) => x instanceof UndefPrimal);
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < bodyJaxpr.inBinders.length; i++) {
      if (i < numConsts) {
        bodyUndefPrimals.push(undefMask[i]);
      } else if (i < numConsts + numElems) {
        bodyUndefPrimals.push(undefMask[numConsts + (i - numConsts)]);
      } else if (i < numConsts + numElems * 2) {
        bodyUndefPrimals.push(
          undefMask[numConsts + numElems + (i - numConsts - numElems)],
        );
      } else if (i < numConsts + numElems * 3) {
        bodyUndefPrimals.push(
          undefMask[numConsts + (i - numConsts - numElems * 2)],
        );
      } else {
        bodyUndefPrimals.push(
          undefMask[numConsts + numElems + (i - numConsts - numElems * 3)],
        );
      }
    }

    const numTangentConsts = bodyUndefPrimals
      .slice(0, numConsts)
      .filter((x) => x).length;

    const transposedBody = transposeJaxpr(bodyJaxpr, bodyUndefPrimals);

    const constResiduals = args
      .slice(0, numConsts)
      .filter((_, i) => !undefMask[i]) as Tracer[];
    const primalElems = args.slice(
      numConsts,
      numConsts + numOrigElems,
    ) as Tracer[];

    const N = primalElems[0].shape[0]; // axis always 0

    const sliceAt = (arr: Tracer, idx: number): Tracer => {
      const ranges: [number, number][] = [[idx, idx + 1]];
      for (let d = 1; d < arr.ndim; d++) ranges.push([0, arr.shape[d]]);
      const sl = shrink(arr, ranges);
      const r = reshape(sl, arr.shape.slice(1));
      sl.dispose();
      return r;
    };

    // Primal-only evaluation of body
    const evalPrimalBody = (aLeaves: Tracer[], bLeaves: Tracer[]): Tracer[] => {
      const bodyArgs: Tracer[] = [];
      using disposables = new DisposableStack();
      let cri = 0;
      for (let i = 0; i < numConsts; i++) {
        if (bodyUndefPrimals[i]) {
          const z = disposables.use(
            zerosInternal(
              bodyJaxpr.inBinders[i].aval.shape,
              bodyJaxpr.inBinders[i].aval.dtype,
            ),
          );
          bodyArgs.push(z);
        } else {
          bodyArgs.push(constResiduals[cri++]);
        }
      }
      for (const a of aLeaves) bodyArgs.push(a);
      // Zero tangent "a" slots
      for (let i = numOrigElems; i < numElems * 2; i++) {
        const z = disposables.use(
          zerosInternal(
            bodyJaxpr.inBinders[numConsts + i].aval.shape,
            bodyJaxpr.inBinders[numConsts + i].aval.dtype,
          ),
        );
        bodyArgs.push(z);
      }
      for (const b of bLeaves) bodyArgs.push(b);
      // Zero tangent "b" slots
      for (let i = numOrigElems; i < numElems * 2; i++) {
        const z = disposables.use(
          zerosInternal(
            bodyJaxpr.inBinders[numConsts + numElems * 2 + i].aval.shape,
            bodyJaxpr.inBinders[numConsts + numElems * 2 + i].aval.dtype,
          ),
        );
        bodyArgs.push(z);
      }
      const outs = evalJaxpr(bodyJaxpr, bodyArgs);
      const pOuts = outs.slice(0, numOrigElems);
      for (let i = numOrigElems; i < outs.length; i++) outs[i].dispose();
      return pOuts;
    };

    // Recompute forward primals
    const allYP: Tracer[][] = [];
    allYP.push(primalElems.map((e) => sliceAt(e, 0)));
    for (let i = 1; i < N; i++) {
      const xi = primalElems.map((e) => sliceAt(e, i));
      const newY = evalPrimalBody(allYP[i - 1], xi);
      for (const x of xi) x.dispose();
      allYP.push(newY);
    }

    // Backward loop
    const ctPrimal = cts.slice(0, numOrigElems);
    const ctTangent = cts.slice(numOrigElems);
    const disposedCts = new Set<Tracer>();
    for (const c of ctPrimal) {
      if (!disposedCts.has(c)) {
        disposedCts.add(c);
        c.dispose();
      }
    }

    let ctCarry: Tracer[] = allYP[0].map((y) =>
      zerosInternal(y.shape, y.dtype),
    );
    let ctConstsAccum: Tracer[] | null = null;
    const ctXsPerLeaf: Tracer[][] = Array.from(
      { length: numOrigElems },
      () => [],
    );

    for (let i = N - 1; i >= 1; i--) {
      const ctYi = ctTangent.map((c) => sliceAt(c, i));
      const ctEff = ctCarry.map((c, j) => add(c, ctYi[j]));
      for (const c of ctCarry) c.dispose();
      for (const c of ctYi) c.dispose();

      const xPi = primalElems.map((e) => sliceAt(e, i));
      const yPrev = allYP[i - 1];

      const tbInputs: Tracer[] = [
        ...transposedBody.constEnv.values,
        ...constResiduals,
      ];
      using tbDisposables = new DisposableStack();
      for (const y of yPrev) tbInputs.push(y);
      for (const x of xPi) tbInputs.push(x);
      // Zero cotangents for primal outputs
      for (let j = 0; j < numOrigElems; j++) {
        const z = tbDisposables.use(
          zerosInternal(allYP[0][j].shape, allYP[0][j].dtype),
        );
        tbInputs.push(z);
      }
      for (const c of ctEff) tbInputs.push(c);

      const tbOuts = evalJaxpr(transposedBody.jaxpr, tbInputs);

      let oi = 0;
      const ctConstsI: Tracer[] = [];
      for (let j = 0; j < numTangentConsts; j++) ctConstsI.push(tbOuts[oi++]);
      const ctANew: Tracer[] = [];
      for (let j = 0; j < numOrigElems; j++) ctANew.push(tbOuts[oi++]);
      const ctBIter: Tracer[] = [];
      for (let j = 0; j < numOrigElems; j++) {
        const ct = tbOuts[oi++];
        if (ctANew.includes(ct)) ct.ref; // jax-js-lint: allow-ref
        ctBIter.push(ct);
      }

      ctCarry = ctANew;

      if (ctConstsAccum === null) {
        ctConstsAccum = ctConstsI;
      } else {
        for (let j = 0; j < ctConstsAccum.length; j++) {
          const s = add(ctConstsAccum[j], ctConstsI[j]);
          ctConstsAccum[j].dispose();
          ctConstsI[j].dispose();
          ctConstsAccum[j] = s;
        }
      }

      for (let j = 0; j < numOrigElems; j++) ctXsPerLeaf[j].push(ctBIter[j]);
      for (const x of xPi) x.dispose();
      for (const c of ctEff) c.dispose();
    }

    // Position 0
    const ctY0 = ctTangent.map((c) => sliceAt(c, 0));
    const ctX0 = ctCarry.map((c, j) => {
      const s = add(c, ctY0[j]);
      c.dispose();
      ctY0[j].dispose();
      return s;
    });
    for (let j = 0; j < numOrigElems; j++) ctXsPerLeaf[j].push(ctX0[j]);

    // Cleanup
    for (const yp of allYP) for (const y of yp) y.dispose();
    for (const ct of ctTangent) {
      if (!disposedCts.has(ct)) {
        disposedCts.add(ct);
        ct.dispose();
      }
    }

    // Stack per-position cotangents
    const ctXsStacked: Tracer[] = [];
    for (let j = 0; j < numOrigElems; j++) {
      const perPos = ctXsPerLeaf[j].reverse();
      const expanded = perPos.map((ct) => broadcast(ct, [1, ...ct.shape], [0]));
      const stacked = concatenate(expanded, 0);
      const disposed = new Set<Tracer>();
      for (const ct of expanded) {
        if (!disposed.has(ct)) {
          disposed.add(ct);
          ct.dispose();
        }
      }
      for (const ct of perPos) {
        if (!disposed.has(ct)) {
          disposed.add(ct);
          ct.dispose();
        }
      }
      ctXsStacked.push(stacked);
    }

    // Assemble output
    const result: (Tracer | null)[] = [];
    let ctCI = 0;
    for (let i = 0; i < args.length; i++) {
      if (!(args[i] instanceof UndefPrimal)) {
        result.push(null);
      } else if (i < numConsts) {
        result.push(ctConstsAccum![ctCI++]);
      } else {
        const tangentLeafIdx = i - numConsts - numOrigElems;
        result.push(ctXsStacked[tangentLeafIdx]);
      }
    }

    const consumedIdxs = new Set<number>();
    for (let i = 0; i < args.length; i++) {
      if (args[i] instanceof UndefPrimal && i >= numConsts) {
        consumedIdxs.add(i - numConsts - numOrigElems);
      }
    }
    for (let j = 0; j < ctXsStacked.length; j++) {
      if (!consumedIdxs.has(j)) ctXsStacked[j].dispose();
    }

    return result;
  },
  [Primitive.DynamicSlice]([ct], primals, { sliceSizes }) {
    const origShape = (
      primals[0] instanceof UndefPrimal
        ? primals[0].aval.shape
        : (primals[0] as Tracer).shape
    ) as number[];
    const starts = primals.slice(1) as Tracer[];

    let curr = ct;
    const currShape = [...sliceSizes];
    for (let ax = starts.length - 1; ax >= 0; ax--) {
      if (sliceSizes[ax] === origShape[ax]) continue;

      currShape[ax] = origShape[ax];
      // Note: fullInternal from array.ts takes (shape, fillValue, dtype)
      using target = fullInternal(
        new ShapedArray(currShape, (ct as Tracer).dtype, false),
        0,
      );

      const maxStart = (origShape[ax] as number) - sliceSizes[ax];
      using zeroArr = array(0, { dtype: starts[ax].dtype });
      using maxArr = array(maxStart, { dtype: starts[ax].dtype });

      const clampedStart1 = min(starts[ax], maxArr) as Tracer;
      const start = max(zeroArr, clampedStart1) as Tracer;

      using indicesBase = array(new Int32Array(range(sliceSizes[ax])), {
        dtype: starts[ax].dtype,
      });
      const indices = add(indicesBase, start);

      const nextCurr = scatterAdd(target, indices, curr, ax) as Tracer;

      clampedStart1.dispose();
      start.dispose();
      indices.dispose();
      if (curr !== ct) curr.dispose();
      curr = nextCurr;
    }

    return [curr, ...starts.map(() => null)];
  },
  // UncheckedDynamicSlice adjoint: same as DynamicSlice (indices in-bounds
  // by construction, clamping is a harmless no-op in the backward pass)
  [Primitive.UncheckedDynamicSlice]([ct], primals, { sliceSizes }) {
    const origShape = (
      primals[0] instanceof UndefPrimal
        ? primals[0].aval.shape
        : (primals[0] as Tracer).shape
    ) as number[];
    const starts = primals.slice(1) as Tracer[];

    let curr = ct;
    const currShape = [...sliceSizes];
    for (let ax = starts.length - 1; ax >= 0; ax--) {
      if (sliceSizes[ax] === origShape[ax]) continue;

      currShape[ax] = origShape[ax];
      using target = fullInternal(
        new ShapedArray(currShape, (ct as Tracer).dtype, false),
        0,
      );

      const maxStart = (origShape[ax] as number) - sliceSizes[ax];
      using zeroArr = array(0, { dtype: starts[ax].dtype });
      using maxArr = array(maxStart, { dtype: starts[ax].dtype });

      const clampedStart1 = min(starts[ax], maxArr) as Tracer;
      const start = max(zeroArr, clampedStart1) as Tracer;

      using indicesBase = array(new Int32Array(range(sliceSizes[ax])), {
        dtype: starts[ax].dtype,
      });
      const indices = add(indicesBase, start);

      const nextCurr = scatterAdd(target, indices, curr, ax) as Tracer;

      clampedStart1.dispose();
      start.dispose();
      indices.dispose();
      if (curr !== ct) curr.dispose();
      curr = nextCurr;
    }

    return [curr, ...starts.map(() => null)];
  },
  [Primitive.ForiLoop](
    cts,
    args,
    { jaxpr, numConsts, lower: lowerDim, upper: upperDim, isJvpTransformed },
  ) {
    // ForiLoop backward pass: unrolled forward + backward.
    //
    // After JVP+PE the fori_loop carries both primal and tangent halves.
    // The backward pass builds compile-time jaxprs (primal forward, tangent
    // body, transposed tangent body), then:
    //   1. Runs primal forward N times to collect intermediate primal carries
    //   2. Runs transposed tangent body N times in reverse via evalJaxpr
    //
    // This mirrors the scan backward approach: evalJaxpr returns fresh arrays,
    // avoiding aliasing complications from evalJaxprTransposed.

    if (!isJvpTransformed) {
      throw new Error(
        "ForiLoop backward pass requires isJvpTransformed (from JVP rule). " +
          "Use jit(grad(f)) instead of grad(jit(f)).",
      );
    }

    const N = dimSub(upperDim, lowerDim);
    const numCarryTotal = args.length - numConsts;
    const numCarryHalf = numCarryTotal / 2;

    // Body invars: [consts(numConsts), i, carry(numCarryTotal)]
    const iDtype = jaxpr.inBinders[numConsts].aval.dtype;

    // ---- Identify unknown (tangent) positions ----
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < jaxpr.inBinders.length; i++) {
      if (i < numConsts) {
        bodyUndefPrimals.push(args[i] instanceof UndefPrimal);
      } else if (i === numConsts) {
        bodyUndefPrimals.push(false); // i: always known
      } else {
        const carryIdx = i - numConsts - 1;
        bodyUndefPrimals.push(carryIdx >= numCarryHalf);
      }
    }

    const numTangentConsts = args
      .slice(0, numConsts)
      .filter((a) => a instanceof UndefPrimal).length;

    // ---- Build primal forward jaxpr ----
    const primalForwardInTypes = jaxpr.inBinders
      .filter((_, i) => !bodyUndefPrimals[i])
      .map((v) => v.aval);

    const { jaxpr: primalForwardJaxpr } = makeJaxpr(
      (...primalInputs: Tracer[]): Tracer[] => {
        const fullInputs: Tracer[] = [];
        const localTemps: JaxArray[] = [];
        let primalIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            const zero = zerosInternal(
              jaxpr.inBinders[i].aval.shape,
              jaxpr.inBinders[i].aval.dtype,
            );
            if (zero instanceof JaxArray) localTemps.push(zero);
            fullInputs.push(zero);
          } else {
            fullInputs.push(primalInputs[primalIdx++]);
          }
        }
        const outs = evalJaxpr(jaxpr, fullInputs);
        const primalOuts = outs.slice(0, numCarryHalf);
        for (let i = numCarryHalf; i < outs.length; i++) outs[i].dispose();
        for (const temp of localTemps) temp.dispose();
        return primalOuts;
      },
      { validateRefs: false },
    )(...primalForwardInTypes);

    // ---- Build tangent body jaxpr ----
    const tangentBodyInAvals = [
      ...jaxpr.inBinders
        .filter((_, i) => !bodyUndefPrimals[i])
        .map((v) => v.aval),
      ...jaxpr.inBinders
        .filter((_, i) => bodyUndefPrimals[i])
        .map((v) => v.aval),
    ];

    const numTangentCarry = numCarryHalf;

    const { jaxpr: tangentBody } = makeJaxpr(
      (...tangentBodyArgs: Tracer[]): Tracer[] => {
        const numPrimalInputs = jaxpr.inBinders.filter(
          (_, i) => !bodyUndefPrimals[i],
        ).length;
        const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
        const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);

        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        let tangentIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            fullInputs.push(tangentInputs[tangentIdx++]);
          } else {
            fullInputs.push(primalResiduals[primalIdx++]);
          }
        }

        const fullOuts = evalJaxpr(jaxpr, fullInputs);
        // Extract tangent carry outputs (second half)
        const tangentOuts = fullOuts.slice(numCarryHalf);
        for (let i = 0; i < numCarryHalf; i++) fullOuts[i].dispose();
        return tangentOuts;
      },
      { validateRefs: false },
    )(...tangentBodyInAvals);

    // ---- Transpose the tangent body (cache-owned) ----
    const tangentBodyUndefPrimals = [
      ...Array(
        tangentBody.jaxpr.inBinders.length -
          (numTangentConsts + numTangentCarry),
      ).fill(false),
      ...Array(numTangentConsts + numTangentCarry).fill(true),
    ];

    const transposedBody = transposeJaxpr(
      tangentBody.jaxpr,
      tangentBodyUndefPrimals,
    );

    // ---- Forward pass: collect primal carry at each step via Scan ----
    const constResiduals = args
      .slice(0, numConsts)
      .filter((_, i) => !bodyUndefPrimals[i]) as Tracer[];
    const primalCarryInit = args.slice(
      numConsts,
      numConsts + numCarryHalf,
    ) as Tracer[];

    // Build Forward Scan body:
    // Takes: [i, ...primalCarry]
    // Returns: [iNext, ...nextPrimalCarry, i, ...primalCarry]  <- stacked ys are the history
    const { jaxpr: fwdScanBody } = makeJaxpr(
      (iIn: Tracer, ...carryIn: Tracer[]) => {
        const forwardOuts = evalJaxpr(primalForwardJaxpr.jaxpr, [
          ...primalForwardJaxpr.consts,
          ...constResiduals,
          iIn,
          ...carryIn,
        ]);
        using one = array(1, { dtype: iDtype });
        const iNext = bind(Primitive.Add, [iIn, one])[0];
        return [iNext, ...forwardOuts, iIn, ...carryIn];
      },
    )(
      new ShapedArray([], iDtype, false) as any,
      ...primalCarryInit.map((c) => c.aval as any),
    );

    const initialILoc = typeof lowerDim === "number" ? lowerDim : 0;
    const initialIArr = array(initialILoc, { dtype: iDtype });
    const fwdScanOuts = bind(
      Primitive.Scan,
      [...fwdScanBody.consts, initialIArr, ...primalCarryInit],
      {
        jaxpr: fwdScanBody.jaxpr,
        numConsts: fwdScanBody.consts.length,
        numCarry: 1 + numCarryHalf,
        length: N,
        reverse: false,
      },
    ) as Tracer[];

    // Dispose forward scan carry outputs (final i + final primal carries)
    const ysStart = 1 + numCarryHalf;
    for (let i = 0; i < ysStart; i++) fwdScanOuts[i].dispose();
    initialIArr.dispose();
    const historyI = fwdScanOuts[ysStart];
    const historyPrimals = fwdScanOuts.slice(ysStart + 1);

    // ---- Backward pass via Scan ----
    const ctCarryInit = cts.slice(numCarryHalf);

    const tangentConstsAvals = tangentBodyInAvals.slice(0, numTangentConsts);

    const { jaxpr: bwdScanBody } = makeJaxpr((...bwdArgs: Tracer[]) => {
      const ctConstsIn = bwdArgs.slice(0, numTangentConsts);
      const ctCarryIn = bwdArgs.slice(
        numTangentConsts,
        numTangentConsts + numTangentCarry,
      );
      const iArr_k = bwdArgs[numTangentConsts + numTangentCarry];
      const primals_k = bwdArgs.slice(numTangentConsts + numTangentCarry + 1);

      const transposedInputs = [
        ...transposedBody.constEnv.values,
        ...constResiduals,
        iArr_k,
        ...primals_k,
        ...ctCarryIn,
      ];

      const transposedOuts = evalJaxpr(transposedBody.jaxpr, transposedInputs);

      const ctConstsStep = transposedOuts.slice(0, numTangentConsts);
      const ctCarryStep = transposedOuts.slice(
        numTangentConsts,
        numTangentConsts + numTangentCarry,
      );

      const ctConstsOut = ctConstsIn.map(
        (c, idx) => bind(Primitive.Add, [c, ctConstsStep[idx]])[0],
      );

      return [...ctConstsOut, ...ctCarryStep];
    })(
      ...Array.from(
        { length: numTangentConsts },
        (_, i) =>
          new ShapedArray(
            tangentConstsAvals[i].shape,
            tangentConstsAvals[i].dtype,
            false,
          ) as any,
      ),
      ...ctCarryInit.map((c, i) =>
        c
          ? (c.aval as any)
          : (new ShapedArray(
              primalCarryInit[i].shape,
              primalCarryInit[i].dtype,
              false,
            ) as any),
      ),
      new ShapedArray([], iDtype, false) as any,
      ...primalCarryInit.map((c) => c.aval as any),
    );

    const ctConstsZeros = Array.from({ length: numTangentConsts }, (_, i) =>
      zerosInternal(tangentConstsAvals[i].shape, tangentConstsAvals[i].dtype),
    );
    const ctCarrySafeInit = ctCarryInit.map(
      (c, i) =>
        c || zerosInternal(primalCarryInit[i].shape, primalCarryInit[i].dtype),
    );

    const bwdScanOuts = bind(
      Primitive.Scan,
      [
        ...bwdScanBody.consts,
        ...ctConstsZeros,
        ...ctCarrySafeInit,
        historyI,
        ...historyPrimals,
      ],
      {
        jaxpr: bwdScanBody.jaxpr,
        numConsts: bwdScanBody.consts.length,
        numCarry: numTangentConsts + numTangentCarry,
        length: N,
        reverse: true,
      },
    ) as Tracer[];

    const ctConstsFinal = bwdScanOuts.slice(0, numTangentConsts);
    const ctCarryFinal = bwdScanOuts.slice(
      numTangentConsts,
      numTangentConsts + numTangentCarry,
    );

    const resultCts: (Tracer | null)[] = [];
    let ctConstsIdx = 0;
    for (let i = 0; i < numConsts; i++) {
      resultCts.push(bodyUndefPrimals[i] ? ctConstsFinal[ctConstsIdx++] : null);
    }

    let ctCarryIdx = 0;
    for (let i = 0; i < numCarryTotal; i++) {
      if (i < numCarryHalf) {
        resultCts.push(null);
      } else if (args[numConsts + i] instanceof UndefPrimal) {
        resultCts.push(ctCarryFinal[ctCarryIdx++]);
      } else {
        // JVP tangent carry that isn't actually UndefPrimal (e.g., zero init
        // created inside the function) — dispose computed ct, return null.
        ctCarryFinal[ctCarryIdx++].dispose();
        resultCts.push(null);
      }
    }

    // Dispose intermediate arrays (only locally-created ones)
    historyI.dispose();
    for (const t of historyPrimals) t.dispose();
    for (const t of ctConstsZeros) t.dispose();
    // Only dispose zeros we created, not caller-owned cts
    for (let i = 0; i < ctCarrySafeInit.length; i++) {
      if (ctCarrySafeInit[i] !== ctCarryInit[i]) ctCarrySafeInit[i].dispose();
    }

    // Dispose locally-created ClosedJaxpr objects. Their consts were .ref'd
    // by the scan body builders; disposing balances those refs. In tracing
    // mode the outer builder also .ref'd them, so they survive; in eager
    // mode the scans have finished and the consts are freed.
    // transposedBody is cache-owned (transposeJaxprCache) — do NOT dispose.
    primalForwardJaxpr.dispose();
    tangentBody.dispose();
    fwdScanBody.dispose();
    bwdScanBody.dispose();

    return resultCts;
  },
};

const transposeJaxprCache = new Map<Jaxpr, Map<string, CachedJaxprArtifact>>();

// Register for cleanup during checkLeaks.stop() to avoid leaking
// cached const environments across test boundaries.
_registerJitCacheDisposer(() => {
  for (const inner of transposeJaxprCache.values()) {
    for (const artifact of inner.values()) artifact.constEnv.dispose();
  }
  transposeJaxprCache.clear();
});
_registerCacheSizeGetter("transposeJaxpr", () => {
  let total = 0;
  for (const inner of transposeJaxprCache.values()) total += inner.size;
  return total;
});

function transposeJaxpr(
  jaxpr: Jaxpr,
  undefPrimals: boolean[],
): CachedJaxprArtifact {
  // Include backend type: consts in the cached const environment are concrete arrays
  // living on whichever device was active at first transpose. Cross-device reuse
  // would read stale data from the wrong backend.
  const cacheKey = defaultDevice() + "," + JSON.stringify(undefPrimals);
  const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;

  // This handles grad-of-jit or transpose-of-jit. To do this, it needs to
  // evaluate the Jaxpr transposed and then retrace it. See the comment in
  // jvpJaxpr() to explain more about what's going on here.
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);

  // Need to remove the UndefPrimals from the input types, as they are not
  // inputs to the Jaxpr while tracing.
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
  const artifact = buildCachedJaxprArtifact(
    makeJaxpr(
      (forwardIn: Tracer[], cotangents: Tracer[]) => {
        const args: (Tracer | UndefPrimal)[] = [];
        let forwardInIdx = 0; // index in forwardIn
        for (let i = 0; i < undefPrimals.length; i++) {
          if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
          else args.push(forwardIn[forwardInIdx++]);
        }
        return evalJaxprTransposed(jaxpr, args, cotangents);
      },
      { validateRefs: false },
    )(forwardInTypes, outTypes),
  );
  typecheckJaxpr(artifact.jaxpr); // sanity check

  if (!transposeJaxprCache.has(jaxpr))
    transposeJaxprCache.set(jaxpr, new Map());
  transposeJaxprCache.get(jaxpr)!.set(cacheKey, artifact);
  return artifact;
}

/**
 * Phase 2 of partial evaluation: transpose a forward jaxpr to produce a
 * backward jaxpr suitable for VJP/grad.
 *
 * This is a thin wrapper around `transposeJaxpr` that constructs the
 * `undefPrimals` mask from the forward jaxpr's input count and the number
 * of primals. In the forward jaxpr, the inputs are `[...consts, ...primals, ...tangents]`.
 * The `undefPrimals` mask marks which inputs should be treated as
 * unknowns in the transpose (i.e., the tangent slots → true, primal slots → false).
 *
 * Extracted from `vjpFlat` in M1.2 to enable composition with
 * `buildForwardJaxpr` and future artifact types.
 */
export function buildBackwardJaxpr(forwardJaxpr: ClosedJaxpr): ClosedJaxpr {
  // The forward jaxpr's inBinders are [...constVars, ...tangentVars].
  // constVars correspond to the residuals from partial evaluation (known
  // in the backward pass). tangentVars correspond to the JVP tangent
  // inputs (unknown — we want gradients w.r.t. them).
  const numConsts = forwardJaxpr.consts.length;
  const undefPrimals = forwardJaxpr.jaxpr.inBinders.map(
    (_, i) => i >= numConsts,
  );

  // Build a fresh backward jaxpr. We bypass transposeJaxprCache because
  // the returned ClosedJaxpr will be owned (and disposed) by the caller
  // (PullbackArtifact). Caching would create stale-entry hazards.
  const jaxpr = forwardJaxpr.jaxpr;
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);

  const { jaxpr: newJaxpr } = makeJaxpr(
    (forwardIn: Tracer[], cotangents: Tracer[]) => {
      const args: (Tracer | UndefPrimal)[] = [];
      let forwardInIdx = 0;
      for (let i = 0; i < undefPrimals.length; i++) {
        if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
        else args.push(forwardIn[forwardInIdx++]);
      }
      return evalJaxprTransposed(jaxpr, args, cotangents);
    },
    { validateRefs: false },
  )(forwardInTypes, outTypes);
  typecheckJaxpr(newJaxpr.jaxpr);
  return newJaxpr;
}

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
  auxStore?: { value: any },
): [Tracer[], (...cotangents: Tracer[]) => Tracer[], () => void] {
  // Phase 1: JVP + partial evaluation → forward jaxpr + primal outputs.
  // We always need this, regardless of the transposition strategy.
  const {
    primalsOut,
    jaxpr: forwardJaxpr,
    peIntermediates,
  } = linearizeFlatUtil(f, primalsIn);

  // Phase 2: Dispose PE intermediates.
  // Protect consts whose only remaining ref is the CJ's .ref (rc≤1).
  // Consts at rc>1 still carry a ref from instantiateConst that the
  // collector must clean up; protecting them would leak that extra ref.
  const protectedVals = new Set<Tracer>(primalsOut);
  for (const c of forwardJaxpr.consts) {
    if (c.refCount <= 1) protectedVals.add(c);
  }
  if (auxStore?.value != null) {
    for (const arr of collectConcreteArrays(auxStore.value)) {
      protectedVals.add(arr);
    }
  }
  disposePeIntermediates(peIntermediates, protectedVals);

  // CRITICAL: Flush primals' pending backend dispatches. PE tracing created
  // concrete forward-pass arrays with lazy PendingExecute chains. Submitting
  // them here prevents orphaned Slot references.
  flushPendingConcreteArrays(primalsOut);

  // Phase 3: Call-time transposition.
  //
  // evalJaxprTransposed runs with concrete arrays when fVjp is invoked.
  // The scan transpose rule emits in-graph Primitive.Scan calls for the
  // backward pass.  evalJaxprTransposed applies per-equation transpose
  // rules directly and preserves caller-owned args and cotangents.
  const fVjp = (...cotangents: Tracer[]) => {
    const transposeInputs = [
      ...forwardJaxpr.consts,
      ...primalsIn.map((t) => new UndefPrimal(t.aval)),
    ];
    return evalJaxprTransposed(forwardJaxpr.jaxpr, transposeInputs, cotangents);
  };
  const dispose = () => forwardJaxpr.dispose();
  return [primalsOut, fVjp, dispose];
}

export function vjp(
  f: (...primals: any) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...cotangents: any) => any>, any?] {
  // --- custom_vjp fast path ---
  // If `f` was created by `customVjp(fwd, bwd)`, bypass the standard
  // linearize + transpose pipeline and use the user-supplied backward pass.
  const cvd = getCustomVjpDef(f) as CustomVjpDef | undefined;
  if (cvd) {
    return customVjpPath(cvd, primalsIn, hasAux);
  }

  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  // Wrap scalar primals to Arrays; track which are newly created for disposal.
  const wrappedPrimals = primalsInFlat.map(pureArray);
  const [primalsOutFlat, fVjpFlat, innerDispose] = vjpFlat(
    fFlat,
    wrappedPrimals,
    hasAux ? aux : undefined,
  );
  // Dispose newly-created pureArray wrappers. After vjpFlat returns, the
  // wrappers are only used for .aval (shape/dtype metadata), which is safe
  // to read after disposal. Skip wrappers that appear in primalsOutFlat
  // (identity function case: output IS the input primal).
  if (!insideAbstractTrace()) {
    const primalsOutSet = new Set(primalsOutFlat);
    for (let i = 0; i < wrappedPrimals.length; i++) {
      if (
        wrappedPrimals[i] !== primalsInFlat[i] &&
        !primalsOutSet.has(wrappedPrimals[i])
      ) {
        wrappedPrimals[i].dispose();
      }
    }
  }
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = ((cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    // Wrap scalar cotangents to Arrays; dispose wrappers after transpose.
    const wrappedCots = cotangentsOutFlat.map(pureArray);
    const cotangentsInFlat = fVjpFlat(...wrappedCots);
    if (!insideAbstractTrace()) {
      for (let i = 0; i < wrappedCots.length; i++) {
        if (wrappedCots[i] !== cotangentsOutFlat[i]) {
          wrappedCots[i].dispose();
        }
      }
    }
    return treeUnflatten(inTree, cotangentsInFlat);
  }) as OwnedFunction<(...cotangents: any) => any>;
  fVjp.dispose = innerDispose;
  fVjp[Symbol.dispose] = innerDispose;

  if (hasAux) {
    return [primalsOut, fVjp, lowerAux(aux!.value)];
  }
  return [primalsOut, fVjp];
}

/** @inline */
export type GradOpts = {
  /**
   * Integer or sequence of integers. Specifies which positional argument(s) to
   * differentiate with respect to.
   *
   * Defaults to `0` (the first argument).
   */
  argnums?: number | number[];

  /**
   * The input function returns a pair of `[out, aux]` including an auxiliary
   * value. This `aux` is not differentiated, but is returned alongside the
   * gradient when evaluating the function.
   */
  hasAux?: boolean;
};

export function grad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const valueAndGradFn = valueAndGrad(f, opts);
  return (...x: any) => {
    if (opts?.hasAux) {
      const [[y, aux], dx] = valueAndGradFn(...x);
      if (!insideAbstractTrace()) y.dispose();
      return [dx, aux];
    } else {
      const [y, dx] = valueAndGradFn(...x);
      if (!insideAbstractTrace()) y.dispose();
      return dx;
    }
  };
}

export function valueAndGrad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const argnums = opts?.argnums ?? 0; // By default, differentiate w.r.t. first arg.
  const hasAux = opts?.hasAux ?? false;
  checkInts(argnums);
  const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
  return (...x: any) => {
    if (x.length === 0) {
      throw new Error("grad requires at least one argument to differentiate");
    }
    // Differentiate only with respect to the argnums.
    // Track stopGradient results for disposal after vjp completes.
    // We track (sgResult, original) pairs so we only dispose sg results that
    // are distinct from their inputs (stopGradient returns the same object in
    // eager mode, so disposing it would dispose the user's input).
    const sgArrays: Tracer[] = [];
    const sgOriginals = new Set<Tracer>();
    for (let i = 0; i < x.length; i++) {
      if (!argnumsSet.has(i)) {
        x[i] = treeMap((leaf: any) => {
          const sg = stopGradient(leaf);
          if (sg instanceof Tracer) {
            if (leaf instanceof Tracer) sgOriginals.add(leaf);
            sgArrays.push(sg);
          }
          return sg;
        }, x[i]);
      }
    }
    const [y, fVjp, aux] = vjp(f, x, { hasAux });
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      if (!insideAbstractTrace()) {
        fVjp.dispose();
        treeDispose(y);
        if (hasAux) treeDispose(aux);
        for (const a of sgArrays) {
          if (!sgOriginals.has(a)) a.dispose();
        }
      }
      throw new TypeError("grad requires a scalar output");
    }
    if (!isFloatDtype(y.dtype)) {
      if (!insideAbstractTrace()) {
        fVjp.dispose();
        treeDispose(y);
        if (hasAux) treeDispose(aux);
        for (const a of sgArrays) {
          if (!sgOriginals.has(a)) a.dispose();
        }
      }
      throw new TypeError("grad only supports floating-point dtypes");
    }
    const seed = onesLike(y);
    const cts = fVjp(seed); // backprop from scalar 1
    let seedEscapes = false;
    for (const ct of cts) {
      if (ct === seed) {
        seedEscapes = true;
        break;
      }
      for (const arr of collectConcreteArrays(ct)) {
        if (arr === seed) {
          seedEscapes = true;
          break;
        }
      }
      if (seedEscapes) break;
    }
    const shouldDisposeSeed = !seedEscapes;
    if (!insideAbstractTrace()) {
      for (const a of sgArrays) {
        if (!sgOriginals.has(a)) a.dispose();
      }
    }
    fVjp.dispose();
    if (shouldDisposeSeed) {
      if (!insideAbstractTrace()) {
        seed.dispose();
      } else if (
        currentTraceLevel() === 1 &&
        seed instanceof JaxArray &&
        seed.refCount > 0
      ) {
        seed.dispose();
      }
    }
    for (let i = 0; i < cts.length; i++) {
      if (!argnumsSet.has(i)) treeDispose(cts[i]);
    }
    const grads =
      typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
    return hasAux ? [[y, aux], grads] : [y, grads];
  };
}

// See also: jacfwd()
export function jacrev(f: any) {
  return function jacobianReverse(x: Tracer) {
    if (x.shape.length !== 1) {
      throw new TypeError("jacrev only supports 1D inputs");
    }
    const [size] = x.shape;
    const pullback = (ct: Tracer) => {
      const [y, fVjp] = vjp(f, [x]);
      y.dispose();
      const [ret] = fVjp(ct);
      fVjp.dispose();
      return ret;
    };
    const eyeMatrix = eye(size, undefined, { dtype: x.dtype });
    const result = vmap(pullback, [1])(eyeMatrix);
    eyeMatrix.dispose();
    return result;
  };
}

// See also: jacfwd()
export function hessian(f: any) {
  return jacfwd(grad(f));
}

// ---------------------------------------------------------------------------
// custom_vjp: user-defined VJP rules
// ---------------------------------------------------------------------------

/**
 * Internal descriptor attached to functions wrapped by {@link customVjp}.
 * @internal
 */
interface CustomVjpDef {
  fwd: (...args: any[]) => [any, any];
  bwd: (residuals: any, cotangents: any) => any;
}

/**
 * Internal path used by {@link vjp} when the target function carries a
 * `__customVjpDef` descriptor.
 *
 * 1. Calls `fwd(primals...)` → `[outputs, residuals]`.
 * 2. Applies `stopGradient` to every Array leaf in `residuals` so that
 *    the standard autodiff machinery treats them as constants.
 * 3. Returns `[outputs, vjpFn]` where `vjpFn(g) = bwd(residuals, g)`.
 *
 * `hasAux` is intentionally unsupported here — `customVjp` wraps a
 * *pair* of functions (`fwd`/`bwd`), not a single function with aux.
 *
 * @internal
 */
function customVjpPath(
  cvd: CustomVjpDef,
  primalsIn: any[],
  hasAux: boolean,
): [any, OwnedFunction<(...cotangents: any) => any>, any?] {
  if (hasAux) {
    throw new Error("customVjp functions do not support hasAux in vjp()");
  }

  // Flatten primal inputs to capture the input tree for cotangent validation.
  const [, primalsInTree] = treeFlatten(primalsIn);

  // Forward: call user's fwd function normally.
  const [outputs, residuals] = cvd.fwd(...primalsIn);

  // Independently retain residual Array leaves so the pullback survives even
  // if the caller disposes the primal outputs (which may alias residuals).
  // Detach from the trace via stopGradient so residuals become constants in
  // any outer differentiation context.
  const retainedResiduals: Tracer[] = [];
  const detachedResiduals = treeMap((leaf: any) => {
    if (leaf instanceof Tracer) {
      const sg = stopGradient(leaf);
      if (sg !== leaf) {
        // Tracing mode: stopGradient produced a new wrapper — own it.
        retainedResiduals.push(sg as Tracer);
        return sg;
      }
      // Eager mode: stopGradient is identity. Take an independent ref so
      // the residual stays alive regardless of what the caller does with
      // the output.
      // jax-js-lint: allow-ref — residual retention for pullback lifetime
      const retained = leaf.ref;
      retainedResiduals.push(retained);
      return retained;
    }
    return leaf;
  }, residuals);

  // Build the VJP function.
  const vjpFn = ((cotangents: any) => {
    const rawCts = cvd.bwd(detachedResiduals, cotangents);
    // Validate cotangent structure: must match the primal input tree.
    const ctsArray = globalThis.Array.isArray(rawCts) ? rawCts : [rawCts];
    const [ctsFlat, ctsTree] = treeFlatten(ctsArray);
    if (!primalsInTree.equals(ctsTree)) {
      throw new TreeMismatchError("customVjp backward", primalsInTree, ctsTree);
    }
    return ctsFlat;
  }) as OwnedFunction<(...cotangents: any) => any>;

  // Release retained residuals when the vjpFn is disposed.
  vjpFn.dispose = () => {
    if (!insideAbstractTrace()) {
      for (const a of retainedResiduals) a.dispose();
    }
  };
  vjpFn[Symbol.dispose] = vjpFn.dispose;

  return [outputs, vjpFn];
}

/**
 * Create a function with a custom vector-Jacobian product (VJP) rule.
 *
 * This is the jax-js equivalent of JAX's `jax.custom_vjp`. It lets you
 * define exactly how reverse-mode differentiation should behave for a
 * function, which is essential for:
 *
 * - **Implicit differentiation** (optimizers, fixed-point solvers)
 * - **Numerically stable gradients** (log-sum-exp, softmax)
 * - **Memory-efficient backprop** (checkpointing, not saving intermediates)
 *
 * @param fwd  Forward function. Called with the same arguments as the
 *             original function. Must return `[outputs, residuals]` where
 *             `residuals` is any pytree of values needed by the backward
 *             pass. The `outputs` are returned to the caller.
 *
 * @param bwd  Backward function. Called with `(residuals, cotangents)` where
 *             `residuals` is the second element returned by `fwd`, and
 *             `cotangents` has the same pytree structure as `outputs`.
 *             Must return an array (or pytree) of cotangents with the same
 *             structure as the *positional arguments* to `fwd`.
 *
 * @returns A function with the same signature as `(...args) => outputs`.
 *          When differentiated via `vjp()`, `grad()`, or `valueAndGrad()`,
 *          the custom `bwd` function is invoked instead of the standard
 *          autodiff backward pass.
 *
 * @example
 * ```ts
 * // Numerically stable log1pexp with custom gradient
 * const log1pexp = customVjp(
 *   // fwd: compute output + save residuals for backward
 *   (x: np.Array) => {
 *     const out = np.log(np.add(np.array(1), np.exp(x)));
 *     return [out, x];  // save x as residual
 *   },
 *   // bwd: compute gradient using residuals
 *   (x: np.Array, g: np.Array) => {
 *     return np.multiply(g, np.subtract(np.array(1), np.exp(np.negative(x))));
 *   },
 * );
 *
 * const dx = grad(log1pexp)(np.array(100.0));
 * // → 1.0 (correct), not NaN (what autodiff of log(1+exp(100)) gives)
 * ```
 *
 * @example
 * ```ts
 * // Implicit differentiation for an optimizer
 * const solve = customVjp(
 *   (params: np.Array) => {
 *     const solution = runOptimizer(params);  // opaque solve
 *     return [solution, { solution, params }];
 *   },
 *   (res, g) => {
 *     // Use the implicit function theorem: dx_star/dtheta = -(dF/dx)^-1 (dF/dtheta)
 *     const { solution, params } = res;
 *     const implicitGrad = computeImplicitGrad(solution, params, g);
 *     return implicitGrad;
 *   },
 * );
 * ```
 */
export function customVjp<F extends (...args: any[]) => any>(
  fwd: (...args: Parameters<F>) => [ReturnType<F>, any],
  bwd: (residuals: any, cotangents: ReturnType<F>) => any,
): (...args: Parameters<F>) => ReturnType<F> {
  const wrapped = (...args: Parameters<F>): ReturnType<F> => {
    const [outputs] = fwd(...args);
    return outputs;
  };

  // Tag the function so vjp/grad/valueAndGrad can detect it.
  setCustomVjpDef(wrapped, { fwd, bwd } satisfies CustomVjpDef);

  return wrapped as (...args: Parameters<F>) => ReturnType<F>;
}

// ---------------------------------------------------------------------------
// custom_jvp: user-defined JVP rules
// ---------------------------------------------------------------------------

/**
 * Internal descriptor attached to functions wrapped by {@link customJvp}.
 * @internal
 */
interface CustomJvpDef {
  fn: (...args: any[]) => any;
  jvpRule: (primals: any[], tangents: any[]) => [any, any];
}

/**
 * Create a function with a custom Jacobian-vector product (JVP) rule.
 *
 * This is the jax-js equivalent of JAX's `jax.custom_jvp`. It lets you
 * define exactly how forward-mode differentiation should behave for a
 * function, which is essential for:
 *
 * - **Numerically stable derivatives** (log-sum-exp, softmax)
 * - **Custom local linearization** rules
 * - **Both forward- and reverse-mode** — a custom JVP rule automatically
 *   provides a custom VJP via transpose
 *
 * @param fn   The primal function to wrap.
 *
 * @param jvpRule  The custom JVP rule. Called with `(primals, tangents)` where
 *                 each is an array of the same positional arguments as `fn`.
 *                 Must return `[primalOut, tangentOut]`.
 *
 * @returns A function with the same signature as `fn`. When differentiated
 *          via `jvp()`, `jacfwd()`, `grad()`, or `vjp()`, the custom
 *          `jvpRule` is invoked instead of tracing through `fn`.
 *
 * @example
 * ```ts
 * // Numerically stable log1pexp with custom JVP
 * const log1pexp = customJvp(
 *   (x: np.Array) => np.log(np.add(np.array(1), np.exp(x))),
 *   ([x], [dx]) => {
 *     const primal = np.log(np.add(np.array(1), np.exp(x)));
 *     const tangent = np.multiply(dx, np.subtract(np.array(1), np.exp(np.negative(x))));
 *     return [primal, tangent];
 *   },
 * );
 *
 * // Works with both forward- and reverse-mode AD:
 * const [y, dy] = jvp(log1pexp, [np.array(100.0)], [np.array(1.0)]);
 * const dx = grad(log1pexp)(np.array(100.0));  // also works via transpose
 * ```
 */
export function customJvp<F extends (...args: any[]) => any>(
  fn: F,
  jvpRule: (
    primals: Parameters<F>,
    tangents: Parameters<F>,
  ) => [ReturnType<F>, ReturnType<F>],
): F {
  const wrapped = ((...args: any[]) => {
    return fn(...args);
  }) as unknown as F;

  // Tag the function so jvp/jacfwd can detect it. vjp/grad also benefit
  // because the standard linearize→transpose pipeline will trace through
  // the custom JVP rule, producing correct reverse-mode derivatives.
  setCustomJvpDef(wrapped, { fn, jvpRule } as unknown as CustomJvpDef);

  return wrapped;
}

// ---------------------------------------------------------------------------
// customLinearSolve: implicit differentiation for linear solves
// ---------------------------------------------------------------------------

/**
 * Options for {@link customLinearSolve}.
 */
export interface CustomLinearSolveOpts {
  /**
   * Transpose solve function. Called as `transposeSolve(vecmat, g)` during
   * reverse-mode differentiation, where `vecmat` is the transpose of `matvec`
   * and `g` is the output cotangent.
   *
   * Required when `symmetric` is false. For symmetric operators, the forward
   * `solve` function is reused for the transpose.
   */
  transposeSolve?: (vecmat: (x: any) => any, b: any) => any;
  /** If true, `matvec` is known to be symmetric (A = A^T). */
  symmetric?: boolean;
}

/**
 * Solve a linear system with implicit differentiation.
 *
 * Computes `x = solve(matvec, b)` such that `matvec(x) ≈ b`, and provides
 * an analytically correct VJP that avoids differentiating through the
 * solve implementation. This is essential for:
 *
 * - **Matrix-free solves** (CG, GMRES, iterative methods)
 * - **Implicit layers** (DEQ, fixed-point networks)
 * - **Optimization** (Newton steps, trust-region subproblems)
 *
 * ### Mathematical background
 *
 * Given `A x = b` where `A` is the linear operator represented by `matvec`:
 * - Forward: `x = A^{-1} b` (computed by `solve`)
 * - VJP: for output cotangent `g`, the cotangent of `b` is `A^{-T} g`
 *   (computed by `transposeSolve` or `solve` if symmetric)
 *
 * The key insight is that we never need to differentiate through the
 * solver's internal iterations — only through `matvec` (to get its
 * transpose) and through one solve call in the backward pass.
 *
 * @param matvec  Linear operator `A`: maps `x → A x`. Must be a pure
 *                function of `x` (no side effects).
 * @param b       Right-hand side of the linear system.
 * @param solve   Forward solver: `solve(matvec, b) → x` such that
 *                `matvec(x) ≈ b`.
 * @param opts    Optional settings for transpose solve and symmetry.
 *
 * @returns The solution `x`.
 *
 * @example
 * ```ts
 * // Conjugate gradient solve with implicit differentiation
 * const x = customLinearSolve(
 *   (v) => np.matmul(A, v),  // matvec: x → A x
 *   b,                        // right-hand side
 *   (matvec, b) => cg(matvec, b, { maxIter: 100 }),
 * );
 * // grad() through this avoids differentiating through CG iterations
 * ```
 */
export function customLinearSolve(
  matvec: (x: any) => any,
  b: any,
  solve: (matvec: (x: any) => any, b: any) => any,
  opts?: CustomLinearSolveOpts,
): any {
  const symmetric = opts?.symmetric ?? false;
  const transposeSolve = opts?.transposeSolve;

  // Wrap the solve as a customVjp function of b so that reverse-mode
  // differentiation uses the implicit function theorem instead of
  // differentiating through the solver's internal iterations.
  //
  // For A x = b:
  //   VJP w.r.t. b:  g → A^{-T} g  (transpose solve)
  const implicitSolve = customVjp(
    // fwd: solve the system, save nothing (we close over matvec/solve)
    (b_in: any) => {
      const x_out = solve(matvec, b_in);
      return [x_out, null];
    },
    // bwd: solve the transpose system
    (_res: null, g: any) => {
      if (symmetric) {
        // For symmetric A (A = A^T), transpose solve is the same as forward.
        return solve(matvec, g);
      }
      if (!transposeSolve) {
        throw new Error(
          "customLinearSolve requires opts.transposeSolve when opts.symmetric is false",
        );
      }
      // For non-symmetric A, build the transpose operator via vjp of matvec,
      // then call the user-provided transposeSolve. Since matvec is linear,
      // one transpose artifact can be reused for the entire backward solve.
      const transposeSeed = treeMap((leaf: any) => zerosLike(leaf), g);
      const [, vecmatVjp] = vjp(matvec, [transposeSeed]);
      treeDispose(transposeSeed);
      try {
        const vecmat = (v: any) => {
          const [result] = vecmatVjp(v);
          return result;
        };
        return transposeSolve(vecmat, g);
      } finally {
        vecmatVjp.dispose();
      }
    },
  );

  return implicitSolve(b);
}

/**
 * Solve a root-finding problem with implicit differentiation.
 *
 * Computes `x* = solve(f, initialGuess)` such that `f(x*) ≈ 0`, and provides
 * an analytically correct VJP via the Implicit Function Theorem (IFT) that
 * avoids differentiating through the solver's internal iterations.
 *
 * ### Mathematical background
 *
 * Given the optimality condition `f(x*, θ) = 0` where `θ` are parameters
 * captured in `f`'s closure:
 *
 * - By the IFT: `dx∗/dθ = -(∂f/∂x)⁻¹ · (∂f/∂θ)`
 * - VJP: for output cotangent `g`, the cotangent of `θ` is
 *   `-(∂f/∂θ)ᵀ · (∂f/∂x)⁻ᵀ · g`
 *
 * The key insight is that we never differentiate through the solver — only
 * through `f` (to access its closure parameters) and through one linear
 * solve in the backward pass.
 *
 * ### Implementation
 *
 * Uses the identity: `x* = x* - (∂f/∂x)⁻¹ · f(x*)` (a Newton step
 * that is a no-op at convergence since `f(x*) ≈ 0`). The `stopGradient`
 * on `x*` blocks gradient flow through the solver, while evaluating
 * `f(stopGradient(x*))` establishes the AD dependency on `f`'s closure
 * parameters. The `customLinearSolve` handles the `(∂f/∂x)⁻¹` factor
 * and its transpose in the backward pass.
 *
 * @param f  Optimality function: `f(x*) ≈ 0` at the solution. May close
 *           over differentiable parameters — gradients flow through them
 *           via the IFT.
 * @param initialGuess  Starting point for the solver.
 * @param solve  Forward solver: `solve(f, x0) → x*` such that `f(x*) ≈ 0`.
 *               Not differentiated through.
 * @param tangentSolve  Linear solver: `tangentSolve(g, y) → x` such that
 *                      `g(x) = y`. Used both in the forward pass (with
 *                      `g = ∂f/∂x`) and in the backward pass (with
 *                      `g = (∂f/∂x)ᵀ`).
 *
 * @returns The solution `x*`.
 *
 * @example
 * ```ts
 * // Find x such that A·x - b = 0 (i.e., solve A·x = b)
 * const A = np.array([[2, 1], [1, 3]]);
 * const b = np.array([5, 7]);
 *
 * const x = customRoot(
 *   (x) => np.subtract(np.matmul(A, x), b),   // f(x) = Ax - b
 *   np.zeros([2]),                              // initial guess
 *   (f, x0) => iterativeSolver(f, x0),         // any root-finding solver
 *   (g, y) => iterativeSolver(g, y),            // linear system solver
 * );
 * // grad() through this avoids differentiating through the solver
 * ```
 */
export function customRoot(
  f: (x: any) => any,
  initialGuess: any,
  solve: (f: (x: any) => any, initialGuess: any) => any,
  tangentSolve: (g: (x: any) => any, y: any) => any,
): any {
  // Step 1: Solve f(x) = 0. This is opaque to AD.
  const xStar = solve(f, initialGuess);

  // Step 2: Freeze x* to block gradient flow through the solver.
  const xFrozen = treeMap((leaf: any) => stopGradient(leaf), xStar);

  // Step 3: Evaluate f at frozen x*. This carries the AD graph through
  // f's closure parameters — the only path for gradients to reach them.
  // Since f(x*) ≈ 0 at convergence, this doesn't change the output value.
  const fAtStar = f(xFrozen);
  const [xFrozenFlat] = treeFlatten(xFrozen);
  const borrowedPrimalLeaves = new Set<Tracer>();
  for (const leaf of xFrozenFlat) {
    if (leaf instanceof Tracer) borrowedPrimalLeaves.add(leaf);
  }

  // Step 4: Define the Jacobian-vector product ∂f/∂x · v at x*.
  // This is the "linear operator" for customLinearSolve.
  const matvec = (v: any) => {
    const [primals, tangentOut] = jvp((x: any) => f(x), [xFrozen], [v]);
    const [primalsFlat] = treeFlatten(primals);
    const seen = new Set<Tracer>();
    for (const leaf of primalsFlat) {
      if (
        leaf instanceof Tracer &&
        !borrowedPrimalLeaves.has(leaf) &&
        !seen.has(leaf)
      ) {
        seen.add(leaf);
        leaf.dispose();
      }
    }
    return tangentOut;
  };

  // Step 5: Solve (∂f/∂x) · correction = f(x*) via customLinearSolve.
  // Forward: correction ≈ 0 (since f(x*) ≈ 0).
  // Backward (VJP with cotangent g):
  //   cotangent of f(x*) = (∂f/∂x)^{-T} · (-g)
  //   → flows through f to closure params θ: cotangent(θ) = (∂f/∂θ)^T · (∂f/∂x)^{-T} · (-g)
  //   = -(dx*/dθ)^T · g  — exactly the IFT VJP. ✓
  const correction = customLinearSolve(matvec, fAtStar, tangentSolve, {
    transposeSolve: tangentSolve,
  });

  // Step 6: result = x* - correction ≈ x* - 0 = x* (value unchanged).
  // The AD graph through fAtStar → customLinearSolve → subtraction
  // implements the IFT for gradient computation.
  const result = treeMap(
    (x: any, c: any) => {
      const negC = neg(c);
      const sum = add(x, negC);
      negC.dispose();
      return sum;
    },
    xFrozen,
    correction,
  );

  // Dispose intermediates. In eager mode these are concrete arrays;
  // in tracing mode, dispose is always safe (non-consuming model).
  treeDispose(fAtStar);
  treeDispose(correction);
  treeDispose(xFrozen);

  return result;
}
