import {
  accessorAluExp,
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DataArray,
  DType,
  dtypedArray,
  dtypedJsArray,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Device, Executable, getBackend, Slot } from "../backend";
import { Routine } from "../routine";
import {
  concreteDim,
  Pair,
  resolveShape,
  ShapeTracker,
  unravelAlu,
} from "../shape";
import {
  deepEqual,
  generalBroadcast,
  isPermutation,
  prod,
  range,
  RecursiveArray,
  recursiveFlatten,
  rep,
  type ScanPath,
} from "../utils";
import {
  _lastRefMap,
  _leakTrackingEnabled,
  _leakTrackingMap,
  _trackRefsEnabled,
} from "./check-leaks";
import {
  checkConvShape,
  pool,
  poolTranspose,
  prepareConv,
} from "./convolution";
import {
  _associativeScanCoreImpl,
  AbstractValue,
  CompareOp,
  add as coreAdd,
  concatenate as coreConcatenate,
  exp as coreExp,
  mul as coreMul,
  pad as corePad,
  shrink as coreShrink,
  getAval,
  inMakeJaxprBody,
  inPartialEvalScope,
  ndim,
  newMain,
  Primitive,
  PrimitiveParams,
  promoteAvals,
  registerPartialEvalCreatedArray,
  routinePrimitives,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  UseAfterFreeError,
  where,
} from "./core";
import { abstractEvalRules, evalJaxpr } from "./jaxpr";
import { jitCompile } from "./jit";
import { executeScan } from "./scan-executor";
import { planScan } from "./scan-plan";

const JsArray = globalThis.Array;

// Don't realize expression arrays smaller than this size.
const inlineArrayLimit = 128;

export type ArrayLike = Array | number | boolean;

/** Version of pureArray with fudged types. */
export const fudgeArray = pureArray as (x: ArrayLike) => Array;

/**
 * An executable operation that will be dispatched to the backend.
 *
 * This holds a reference to all input buffers used in the operation. After the
 * operation is dispatched, the references should be released.
 */
export interface IPendingExecute {
  readonly backend: Backend;
  prepared: any;
  submitted: boolean;
  updateRc(delta: number): void;
  prepare(): Promise<void>;
  prepareSync(): void;
  submit(): void;
}

export class PendingExecute implements IPendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare
  #rc = 1; // since this could be held by multiple arrays, cancel when it hits 0

  constructor(
    readonly backend: Backend,
    readonly source: Kernel | Routine,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
    readonly dynamicParams?: number[],
  ) {
    // Take a reference to all I/O buffers, while this execution is pending.
    // The reference is dropped after submit() or cancellation.
    for (const slot of inputs) this.backend.incRef(slot);
    for (const slot of outputs) this.backend.incRef(slot);
  }

  // Change the reference count of the PendingExecute object.
  // Used when copying the object to a new Array, or disposing an array.
  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // Cancel operation, release the references held to all input buffers.
      for (const slot of this.inputs) this.backend.decRef(slot);
      for (const slot of this.outputs) this.backend.decRef(slot);
    }
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      if (this.source instanceof Kernel) {
        this.prepared = await this.backend.prepareKernel(this.source);
      } else {
        this.prepared = await this.backend.prepareRoutine(this.source);
      }
    })();
    await this.#promise;
  }

  prepareSync() {
    if (this.prepared) return;
    if (this.source instanceof Kernel) {
      this.prepared = this.backend.prepareKernelSync(this.source);
    } else {
      this.prepared = this.backend.prepareRoutineSync(this.source);
    }
  }

  submit() {
    if (this.submitted) return;
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    if (!this.prepared) throw new Error("internal: Not prepared yet");
    this.submitted = true;
    this.backend.dispatch(
      this.prepared,
      this.inputs,
      this.outputs,
      this.dynamicParams,
    );
    for (const slot of this.inputs) this.backend.decRef(slot);
    for (const slot of this.outputs) this.backend.decRef(slot);
  }
}

/** @inline */
export type DTypeAndDevice = { dtype?: DType; device?: Device };

/** @inline */
export type DTypeShapeAndDevice = {
  dtype?: DType;
  shape?: number[];
  device?: Device;
};

type ArrayConstructorArgs = {
  source: AluExp | Slot;
  st: ShapeTracker;
  dtype: DType;
  weakType: boolean;
  backend: Backend;
  committed: boolean;
  pending?: Iterable<IPendingExecute>;
  creationRefOwner?: CreationRefOwner;
};

/** Known claimants of an Array's creation ref. */
export type CreationRefClaimant = "jvp-lifted-tangents" | "jacfwd-eye-matrix";

/** All possible states of an Array's creation-ref owner field. */
export type CreationRefOwner =
  | "none"
  | "partial-eval"
  | "makeJaxpr-body"
  | CreationRefClaimant
  | "balanced";

/**
 * A multidimensional numeric array with data stored on CPU or GPU.
 *
 * This is the library's core data type. Equivalent to `jax.Array` from JAX, or
 * `torch.Tensor`.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
export class Array extends Tracer {
  #dtype: DType;
  #weakType: boolean;
  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #committed: boolean; // if array is committed to device (passed explicitly)
  #rc: number; // reference count for this specific Array object
  /**
   * Named owner of this Array's creation ref (rc=1 at birth).
   *
   * - "none": no subsystem claimed the creation ref yet.
   * - "partial-eval": born during PE scope; PE cleanup path manages it.
   * - "makeJaxpr-body": born during makeJaxpr body execution; if captured as
   *   a const and still outstanding at build time, makeJaxpr must balance it.
   * - claimant states: a specific subsystem owns the creation ref and must
   *   balance it on its own cleanup path.
   * - "balanced": makeJaxpr already paid the compensating dispose.
   */
  #creationRefOwner: CreationRefOwner;
  #pendingSet: Set<IPendingExecute> | null; // only if source is `Slot`

  /**
   * @ignore
   * Constructs an array from source, shape and backend. Note that if the source
   * is a backend `Slot`, this constructor _takes ownership_ of the slot. It
   * will be freed when the array is disposed.
   */
  constructor(args: ArrayConstructorArgs) {
    super(baseArrayTrace);
    this.#dtype = args.dtype;
    this.#weakType = args.weakType;
    this.#source = args.source;
    this.#st = args.st;
    this.#backend = args.backend;
    this.#committed = args.committed;
    this.#rc = 1;
    this.#creationRefOwner = args.creationRefOwner ?? "none";

    this.#pendingSet = new Set(args.pending);
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
    } else if (this.#source instanceof AluExp) {
      throw new Error("internal: AluExp source cannot have pending executes");
    }

    // Leak tracking: record this Array and capture stack frames (V8 defers
    // the expensive .stack string formatting until first access — cold path).
    if (_leakTrackingEnabled) {
      _leakTrackingMap.set(this, new Error());
    }

    // Track Arrays created during PE scope for intermediate disposal.
    // Anonymous constants (np.array([...]) inside grad body) bypass bind()
    // and PE tracking, so this constructor hook catches them.
    // When inside both PE and makeJaxpr, the inner makeJaxpr wins (below).
    if (args.creationRefOwner === undefined) {
      if (inPartialEvalScope()) {
        this.#creationRefOwner = "partial-eval";
        if (!inMakeJaxprBody()) registerPartialEvalCreatedArray(this);
      }

      // Track Arrays created during makeJaxpr body execution.
      // These have an unowned creation ref that must be balanced when the
      // outermost makeJaxpr builds its ClosedJaxpr — builders hold independent
      // .ref handles, but the creation rc=1 has no owner.
      if (inMakeJaxprBody()) this.#creationRefOwner = "makeJaxpr-body";
    }
  }

  /** @ignore */
  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }

  get device(): Device {
    return this.#backend.type;
  }

  #check() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
  }

  /** Construct an array, copying fields from `this`. */
  #newArrayFrom(args: Partial<ArrayConstructorArgs>) {
    return new Array({
      source: args.source ?? this.#source,
      st: args.st ?? this.#st,
      dtype: args.dtype ?? this.#dtype,
      weakType: this.#weakType,
      backend: args.backend ?? this.#backend,
      committed: args.committed ?? this.#committed,
      pending: args.pending ?? this.#pending ?? undefined,
      creationRefOwner: args.creationRefOwner,
    });
  }

  get ref() {
    this.#check();
    this.#rc++;
    if (_trackRefsEnabled) _lastRefMap.set(this, new Error());
    return this;
  }

  /**
   * Internal-only retained handle that returns an independent Array wrapper.
   * Unlike `.ref`, this does not share wrapper-local rc state with the
   * original object; it retains the underlying backend/pending resources and
   * hands that ownership to a fresh wrapper.
   */
  get retainedHandle() {
    return this.#reshape(this.#st, "balanced");
  }

  /** Get the current reference count (for debugging memory management). */
  get refCount(): number {
    return this.#rc;
  }

  override isAliveForCleanup(): boolean {
    return this.#rc > 0;
  }

  /**
   * Whether this Array has an unowned creation ref (rc=1 from birth)
   * that needs balancing by makeJaxpr after building the ClosedJaxpr.
   */
  get hasUnownedCreationRef(): boolean {
    return this.#creationRefOwner === "makeJaxpr-body";
  }

  /**
   * True if this array was created inside a partial-eval scope
   * (i.e., during grad/vjp forward tracing).
   */
  get hasPartialEvalCreationRef(): boolean {
    return this.#creationRefOwner === "partial-eval";
  }

  /**
   * Debug-only view of the current creation-ref owner. Used by ownership
   * characterization tests to pin claimant-specific transitions.
   */
  get _creationRefOwnerForDebug(): CreationRefOwner {
    return this.#creationRefOwner;
  }

  /**
   * Claim ownership of this Array's creation ref for a specific subsystem.
   * After this call, makeJaxpr will NOT balance the creation ref.
   */
  claimCreationRef(owner: CreationRefClaimant): void {
    if (
      this.#creationRefOwner === "none" ||
      this.#creationRefOwner === "partial-eval" ||
      this.#creationRefOwner === "makeJaxpr-body"
    ) {
      this.#creationRefOwner = owner;
      return;
    }
    throw new Error(
      `internal: invalid creation-ref claim transition ${this.#creationRefOwner} -> ${owner}`,
    );
  }

  /**
   * Mark that makeJaxpr paid the compensating dispose for a body-created
   * array's creation ref. Prevents double-balancing by outer builders.
   */
  markCreationRefBalancedByMakeJaxpr(): void {
    if (this.#creationRefOwner !== "makeJaxpr-body")
      throw new Error(
        `internal: invalid makeJaxpr balance transition from ${this.#creationRefOwner}`,
      );
    this.#creationRefOwner = "balanced";
  }

  dispose() {
    this.#check();
    if (--this.#rc === 0) {
      // Leak tracking: remove from map (no-op when map is empty / tracking off).
      if (_leakTrackingMap.size > 0) _leakTrackingMap.delete(this);
      // Free any pending executables that haven't been submitted yet.
      // WARNING: This cancels lazy PendingExecute without running the kernel.
      // In benchmarks/profiling, call .dataSync() or .blockUntilReady() first
      // to force materialization, otherwise the operation never executes.
      for (const exe of this.#pending) exe.updateRc(-1);
      // If this has an array source, free it from the backend.
      if (typeof this.#source === "number") {
        this.#backend.decRef(this.#source);
      }
    }
  }

  /**
   * Override Tracer's no-op to actually dispose concrete Arrays.
   * Called by `using` declarations at block scope exit.
   */
  [Symbol.dispose]() {
    if (this.#rc > 0) this.dispose();
  }

  /** Get the pending executes as a list, trimming if already submitted. */
  get #pending(): IPendingExecute[] {
    if (!this.#pendingSet) return [];
    for (const p of this.#pendingSet) {
      if (p.submitted) this.#pendingSet.delete(p);
    }
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
      return [];
    } else {
      return [...this.#pendingSet];
    }
  }

  /**
   * Convert this array into a primitive value.
   *
   * This only works for scalars (0-dimensional arrays). It lets you get values
   * "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
   * evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
   *
   * This method is also called for `==` equality.
   */
  [Symbol.toPrimitive](): any {
    if (this.ndim === 0) {
      return this.dataSync()[0];
    } else {
      throw new Error(
        `Cannot convert non-scalar array to primitive: ${this.toString()}`,
      );
    }
  }

  #reshape(st: ShapeTracker, creationRefOwner?: CreationRefOwner): Array {
    this.#check();
    const pending = this.#pending;
    for (const exe of pending) exe.updateRc(+1);
    if (typeof this.#source === "number") this.#backend.incRef(this.#source);
    return this.#newArrayFrom({ st, pending, creationRefOwner });
  }

  /**
   * Underlying implementation of the Gather primitive. This indexes an array
   * and extracts slices based on indices in other integer arrays.
   */
  #gather(indices: Array[], axis: number[], outDim: number): Array {
    this.#check();

    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length) {
      throw new TypeError("Gather axis must not have duplicates");
    }

    if (indices.some((a) => a.#committed && a.#backend !== this.#backend)) {
      throw new TypeError(
        `Gather indices must have the same backend: ${this.#backend.type}`,
      );
    }
    indices = indices.map((ar) => ar._putSync(this.#backend));

    // First, broadcast each integer array in `indices`.
    indices = Array.#broadcastArrays(indices);
    const indexShape = indices[0].shape;
    const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const inputs: Slot[] = [];
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < this.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }
    for (const [i, ar] of indices.entries()) {
      if (ar.#source instanceof AluExp) {
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          accessorAluExp(ar.#source, ar.#st, idxAxis),
        );
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis),
        );
      }
    }

    let exp: AluExp;
    if (this.#source instanceof AluExp) {
      // This is an AluExp, not an actual array, so turn it into an expression.
      exp = accessorAluExp(this.#source, this.#st, src);
    } else {
      let gid = inputs.indexOf(this.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(this.#source);
      }
      exp = accessorGlobal(this.#dtype, gid, this.#st, src);
    }

    const kernel = Kernel.single(inputs.length, prod(finalShape), exp);
    const output = this.#backend.malloc(kernel.outputs[0].bytes as number);
    const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(finalShape),
      pending,
    });
  }

  /** Move axes to the rightmost dimension of the shape. */
  #moveAxesDown(axis: number[]): Array {
    this.#check();
    if (axis.length === 0) return this.reshape(this.shape.concat(1));
    const newShape: number[] = [];
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    for (let i = 0; i < this.#st.shape.length; i++) {
      if (axis.includes(i)) {
        shiftedAxes.push(i);
      } else {
        keptAxes.push(i);
        newShape.push(this.#st.shape[i]);
      }
    }
    newShape.push(-1);
    const transposed = this.#transpose(keptAxes.concat(shiftedAxes));
    const result = transposed.reshape(newShape);
    transposed.dispose();
    return result;
  }

  #transpose(perm: number[]): Array {
    this.#check();
    if (!isPermutation(perm, this.ndim))
      throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    return this.#reshape(this.#st.permute(perm));
  }

  #unary(op: AluOp, dtypeOutput?: DType) {
    const weakType = !dtypeOutput && this.#weakType;
    dtypeOutput ??= this.#dtype; // Default to current dtype unless changed.

    this.#check();
    // Short circuit if the array is already AluExp.
    if (this.#source instanceof AluExp) {
      const exp = new AluExp(op, dtypeOutput, [this.#source]);
      return this.#newArrayFrom({
        source: exp.simplify(),
        dtype: dtypeOutput,
        weakType,
      });
    }

    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp = new AluExp(op, dtypeOutput, [
      AluExp.globalView(this.#dtype, 0, this.#st, indices),
    ]);
    const kernel = Kernel.single(1, this.#st.size, exp);
    const output = this.#backend.malloc(kernel.outputs[0].bytes as number);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(
      new PendingExecute(this.#backend, kernel, [this.#source], [output]),
    );

    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(this.shape),
      dtype: dtypeOutput,
      weakType,
      pending,
    });
  }

  #binary(op: AluOp, other: Array): Array {
    const custom = (src: AluExp[]) => new AluExp(op, src[0].dtype, src);
    return Array.#naryCustom(op, custom, [this, other]);
  }

  static #naryCustom(
    name: string,
    custom: (src: AluExp[]) => AluExp,
    arrays: Array[],
    {
      dtypeOverride,
      strongTypeOutput,
      reduceAxis,
    }: {
      dtypeOverride?: (DType | undefined)[];
      strongTypeOutput?: boolean;
      reduceAxis?: boolean;
    } = {},
  ): Array {
    const n = arrays.length;
    if (n === 0) throw new TypeError(`No inputs for ${name}`);

    for (const ar of arrays) ar.#check();

    let castDtype: DType | undefined;
    let castWeakType = true;
    for (let i = 0; i < n; i++) {
      if (dtypeOverride?.[i]) {
        if (arrays[i].#dtype !== dtypeOverride[i]) {
          throw new TypeError(
            `Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`,
          );
        }
      } else {
        // Try to cast with dtype of other arguments in the operation.
        if (castDtype === undefined) {
          castDtype = arrays[i].#dtype;
          castWeakType = arrays[i].#weakType;
        } else {
          ({ dtype: castDtype, weakType: castWeakType } = promoteAvals(
            new ShapedArray([], castDtype, castWeakType),
            arrays[i].aval.scalar(),
          ));
        }
      }
    }
    const weakType = castWeakType && !strongTypeOutput;

    const { backend, committed } = Array.#computeBackend(name, arrays);
    const originalArrays = [...arrays];
    arrays = arrays.map((ar) => ar._putSync(backend));
    // Track pre-broadcast arrays to dispose broadcast views afterward.
    const preBroadcast = [...arrays];
    arrays = Array.#broadcastArrays(arrays);
    const newShape = [...arrays[0].shape];

    // Short circuit if all are already AluExp.
    if (arrays.every((ar) => ar.#source instanceof AluExp) && !reduceAxis) {
      const sources = arrays.map((ar, i) => {
        if (!dtypeOverride?.[i]) {
          return AluExp.cast(castDtype!, ar.#source as AluExp);
        } else {
          return ar.#source as AluExp;
        }
      });
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        // All are AluExp and have the same shape tracker.
        const exp = custom(sources);
        // Dispose broadcast views (no-op for AluExp-backed, safe).
        for (let i = 0; i < arrays.length; i++)
          if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
        // Dispose _putSync copies that differ from original input arrays.
        for (let i = 0; i < originalArrays.length; i++)
          if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();
        return new Array({
          source: exp.simplify(),
          st: arrays[0].#st,
          dtype: exp.dtype,
          weakType: weakType,
          backend,
          committed,
        });
      }
      // If their shape trackers are different, we need to normalize them.
      const exp = custom(
        arrays.map((ar, i) => {
          const src = sources[i];
          if (ar.#st.contiguous) return src;
          return accessorAluExp(src, ar.#st, unravelAlu(newShape, AluVar.idx));
        }),
      );
      const st = ShapeTracker.fromShape(newShape);
      // Dispose broadcast views (no-op for AluExp-backed, safe).
      for (let i = 0; i < arrays.length; i++)
        if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
      // Dispose _putSync copies that differ from original input arrays.
      for (let i = 0; i < originalArrays.length; i++)
        if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();
      return new Array({
        source: exp.simplify(),
        st,
        dtype: exp.dtype,
        weakType,
        backend,
        committed,
      });
    }

    let indices: AluExp[];
    if (!reduceAxis) {
      indices = unravelAlu(newShape, AluVar.gidx);
    } else {
      const contractedShape = newShape.slice(0, -1);
      indices = [...unravelAlu(contractedShape, AluVar.gidx), AluVar.ridx];
    }

    const inputs: Slot[] = [];
    const src: AluExp[] = [];
    for (const [i, ar] of arrays.entries()) {
      let nextSrc: AluExp;
      if (ar.#source instanceof AluExp) {
        nextSrc = accessorAluExp(ar.#source, ar.#st, indices);
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        nextSrc = AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
      }
      if (!dtypeOverride?.[i]) nextSrc = AluExp.cast(castDtype!, nextSrc);
      src.push(nextSrc);
    }

    const exp = custom(src);
    let re: Reduction | undefined = undefined;
    if (reduceAxis) {
      const [axisSize] = newShape.splice(-1, 1); // Remove the contracted axis.
      re = new Reduction(exp.dtype, AluOp.Add, axisSize);
    }
    const kernel = Kernel.single(inputs.length, prod(newShape), exp, re);
    const output = backend.malloc(kernel.outputs[0].bytes as number);
    const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
    for (const exe of pending) exe.updateRc(+1);
    pending.add(new PendingExecute(backend, kernel, inputs, [output]));

    // Dispose broadcast views created by #broadcastArrays.
    for (let i = 0; i < arrays.length; i++)
      if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
    // Dispose _putSync copies that differ from original input arrays.
    for (let i = 0; i < originalArrays.length; i++)
      if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();

    return new Array({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      dtype: kernel.outputs[0].dtype,
      weakType,
      backend,
      committed,
      pending,
    });
  }

  /** Reduce the last dimension of the array by an operation. */
  #reduce(op: AluOp): Array {
    const shape = this.shape;
    const reduction = new Reduction(this.#dtype, op, shape[shape.length - 1]);
    const newShape = shape.slice(0, -1); // first n-1 axes are in the shape
    const newSize = prod(newShape);

    const indices = [...unravelAlu(newShape, AluVar.gidx), AluVar.ridx];

    let exp: AluExp;
    const inputs: Slot[] = [];
    if (this.#source instanceof AluExp) {
      exp = accessorAluExp(this.#source, this.#st, indices);
    } else {
      inputs.push(this.#source);
      exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
    }

    const kernel = Kernel.single(inputs.length, newSize, exp, reduction);
    const output = this.#backend.malloc(kernel.outputs[0].bytes as number);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      pending,
    });
  }

  /** Apply an operation with custom lowering to this array. */
  static #routine<P extends Primitive>(prim: P): ImplRule<P> {
    return (arrays: Array[], params: PrimitiveParams<P>) => {
      const { backend, committed } = Array.#computeBackend(prim, arrays);
      for (const ar of arrays) ar.#realize();

      const avals = arrays.map((ar) => ar.aval);
      const avalsOut = abstractEvalRules[prim](avals, params);
      const routine = new Routine(
        routinePrimitives.get(prim)!,
        {
          inputShapes: avals.map((a) => a.shape as number[]),
          inputDtypes: avals.map((a) => a.dtype),
          outputShapes: avalsOut.map((a) => a.shape as number[]),
          outputDtypes: avalsOut.map((a) => a.dtype),
        },
        params,
      );

      const inputs = arrays.map((ar) => ar.#source as Slot);
      const outputs = avalsOut.map((x) =>
        backend.malloc(byteWidth(x.dtype) * x.size),
      );
      // Deduplicate inherited PEs with a Set (like #naryCustom) — the same PE
      // can appear in multiple input arrays' pending lists when an array is
      // used as input to multiple operations. Without dedup, updateRc(+1)
      // would be called multiple times for the same PE, but the output's
      // Set-based #pendingSet only calls updateRc(-1) once on dispose.
      const inherited = new Set(arrays.flatMap((ar) => ar.#pending));
      for (const exe of inherited) exe.updateRc(+outputs.length);
      const routinePE = new PendingExecute(backend, routine, inputs, outputs);
      routinePE.updateRc(+outputs.length - 1);
      const pending = [...inherited, routinePE];

      return outputs.map(
        (output, i) =>
          new Array({
            source: output,
            st: ShapeTracker.fromShape(avalsOut[i].shape as number[]),
            dtype: avalsOut[i].dtype,
            weakType: avalsOut[i].weakType,
            backend,
            committed,
            pending,
          }),
      );
    };
  }

  /**
   * Normalizes this array into one backed by a `Slot`.
   *
   * This mutates the array in-place, turning it into an equivalent array whose
   * source is actual, contiguous data on device.
   *
   * Calling this twice is a no-op.
   */
  #realize(): void {
    this.#check();
    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    if (this.#source instanceof AluExp) {
      const exp = accessorAluExp(this.#source, this.#st, indices);
      const kernel = Kernel.single(0, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.outputs[0].bytes as number);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [],
        [output],
      );
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = new Set([pendingItem]);
    } else {
      // Only realize if the ShapeTracker is non-contiguous.
      if (this.#st.contiguous) return;
      const exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
      const kernel = Kernel.single(1, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.outputs[0].bytes as number);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [this.#source],
        [output],
      );
      this.#backend.decRef(this.#source);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= new Set();
      this.#pendingSet.add(pendingItem);
    }
  }

  #dataInline(): DataArray {
    this.#check();
    if (!(this.#source instanceof AluExp))
      throw new Error("internal: #dataInline called on non-AluExp source");
    using ar = this.#newArrayFrom({ backend: getBackend("cpu") });
    return ar.dataSync();
  }

  static #broadcastArrays(arrays: Array[]): Array[] {
    // Broadcast all arrays to the same shape.
    if (arrays.length === 0)
      throw new Error("Need at least one array to broadcast");
    if (arrays.length === 1) return arrays;
    const newShape = arrays
      .map((a) => a.shape)
      .reduce((a, b) => generalBroadcast(a, b) as number[]);
    return arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape)) return ar;
      return ar.#reshape(
        ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)),
      );
    });
  }

  static #computeBackend(
    name: string,
    arrays: Array[],
  ): {
    backend: Backend;
    committed: boolean;
  } {
    // First, check if any arrays are committed.
    const committed = arrays.filter((ar) => ar.#committed);
    if (committed.length > 0) {
      const backend = committed[0].#backend;
      for (const ar of committed) {
        if (ar.#backend !== backend) {
          throw new Error(
            `Device mismatch in ${name} between committed arrays on ` +
              `(${backend.type}, ${ar.#backend.type}), ` +
              `please move to the same device with devicePut()`,
          );
        }
      }
      return { backend, committed: true };
    } else {
      // No committed arrays, pick the backend of the first operand.
      const backend = arrays.length > 0 ? arrays[0].#backend : getBackend();
      return { backend, committed: false };
    }
  }

  /** Realize the array and return it as data. */
  async data(): Promise<DataArray> {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      this.#backend.beginBatch?.();
      try {
        for (const p of pending) p.submit();
      } finally {
        this.#backend.endBatch?.();
      }
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = await this.#backend.read(this.#source as Slot, 0, byteCount);
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Wait for this array to finish evaluation.
   *
   * Operations and data loading in jax-js are lazy, so this function ensures
   * that pending operations are dispatched and fully executed before it
   * returns.
   *
   * If you are mapping from `data()` or `dataSync()`, it will also trigger
   * dispatch of operations as well.
   *
   * **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
   * asynchronously for multiple arrays.
   */
  async blockUntilReady(): Promise<Array> {
    this.#check();
    if (this.#source instanceof AluExp) return this;
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      this.#backend.beginBatch?.();
      try {
        for (const p of pending) p.submit();
      } finally {
        this.#backend.endBatch?.();
      }
    }
    // Lightweight GPU fence — resolves when all submitted work completes.
    // Falls back to zero-byte read on backends without fence() (CPU, WASM).
    if (this.#backend.fence) {
      await this.#backend.fence();
    } else {
      await this.#backend.read(this.#source, 0, 0);
    }
    return this;
  }

  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): DataArray {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    const pending = this.#pending;
    if (pending.length > 0) {
      this.#backend.beginBatch?.();
      try {
        for (const p of pending) {
          p.prepareSync();
          p.submit();
        }
      } finally {
        this.#backend.endBatch?.();
      }
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = this.#backend.readSync(this.#source as Slot, 0, byteCount);
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Return this array as a WebGPU buffer (with `STORAGE | COPY_SRC`).
   *
   * Only available on the WebGPU backend. The array remains alive; dispose it
   * when you no longer need the buffer.
   * You _should not_ mutate the buffer's contents.
   *
   * Note that the GPU buffer may be slightly larger than the array's size; it
   * will always be aligned to 4 bytes.
   */
  async gpuBuffer(): Promise<GPUBuffer> {
    if (this.device !== "webgpu")
      throw new Error(`gpuBuffer() is only available on WebGPU backend`);
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    const backend = this.#backend as import("../backend/webgpu").WebGPUBackend;
    const { buffer } = backend.buffers.get(this.#source as Slot)!;
    return buffer;
  }

  /** Synchronous version of `Array.gpuBuffer()`. */
  gpuBufferSync(): GPUBuffer {
    if (this.device !== "webgpu")
      throw new Error(`gpuBufferSync() is only available on WebGPU backend`);
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    const backend = this.#backend as import("../backend/webgpu").WebGPUBackend;
    const { buffer } = backend.buffers.get(this.#source as Slot)!;
    return buffer;
  }

  /**
   * Convert this array into a JavaScript object.
   *
   * This is a blocking operation that will compile all of the shaders and wait
   * for execution to complete, synchronously. No other JavaScript code on the
   * site will be run during shader execution.
   *
   * To avoid blocking, prefer `jsAsync()` when possible.
   */
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
  }

  /**
   * Read the underlying data and dispose this array in one step.
   *
   * This is a convenience for the very common "extract and dispose" pattern
   * at the boundary between jax-js and plain JavaScript:
   * ```ts
   * const floats = await arr.consumeData();
   * // arr is now disposed — no separate .dispose() needed
   * ```
   */
  async consumeData(): Promise<DataArray> {
    const result = await this.data();
    this.dispose();
    return result;
  }

  /**
   * Synchronous variant of `consumeData()`. Reads the data and disposes this
   * array. Prefer the async `consumeData()` for better performance.
   */
  consumeDataSync(): DataArray {
    const result = this.dataSync();
    this.dispose();
    return result;
  }

  /**
   * Copy an element of an array to a numeric scalar and return it.
   *
   * Throws an error if the array does not have a single element. The array must
   * either be rank-0, or all dimensions of the shape are 1.
   */
  item(): number {
    if (this.size !== 1) {
      throw new Error(`item() can only be called on arrays of size 1`);
    }
    return this.dataSync()[0];
  }

  //
  // Internal methods follow, not public API. Do not use.
  //

  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules(): typeof implRules {
    return {
      [Primitive.Add]([x, y]) {
        return [x.#binary(AluOp.Add, y)];
      },
      [Primitive.Mul]([x, y]) {
        return [x.#binary(AluOp.Mul, y)];
      },
      [Primitive.Idiv]([x, y]) {
        return [x.#binary(AluOp.Idiv, y)];
      },
      [Primitive.Mod]([x, y]) {
        return [x.#binary(AluOp.Mod, y)];
      },
      [Primitive.Min]([x, y]) {
        return [x.#binary(AluOp.Min, y)];
      },
      [Primitive.Max]([x, y]) {
        return [x.#binary(AluOp.Max, y)];
      },
      [Primitive.BitCombine]([x, y], { op }) {
        const custom = (src: AluExp[]) => AluExp.bitCombine(src[0], src[1], op);
        return [Array.#naryCustom("bit_combine", custom, [x, y])];
      },
      [Primitive.BitShift]([x, y], { op }) {
        const custom = (src: AluExp[]) => AluExp.bitShift(src[0], src[1], op);
        return [
          Array.#naryCustom("bit_shift", custom, [x, y], {
            dtypeOverride: [undefined, y.dtype],
          }),
        ];
      },
      [Primitive.Neg]([x]) {
        return [zerosLike(x).#binary(AluOp.Sub, x)];
      },
      [Primitive.Reciprocal]([x]) {
        return [x.#unary(AluOp.Reciprocal)];
      },
      [Primitive.Floor]([x]) {
        return [x.#unary(AluOp.Floor)];
      },
      [Primitive.Ceil]([x]) {
        return [x.#unary(AluOp.Ceil)];
      },
      [Primitive.StopGradient]([x]) {
        return [x]; // Stop gradient is a no-op, just return the input.
      },
      [Primitive.Cast]([x], { dtype }) {
        return [x.#unary(AluOp.Cast, dtype)];
      },
      [Primitive.Bitcast]([x], { dtype }) {
        if (x.dtype === dtype) return [x];
        if (x.dtype === DType.Bool || dtype === DType.Bool) {
          throw new TypeError("Bitcast to/from bool is not allowed");
        }
        if (byteWidth(x.dtype) !== byteWidth(dtype)) {
          throw new TypeError(
            `Bitcast from ${x.dtype} to ${dtype} with different byte width`,
          );
        }
        if (x.#source instanceof AluExp) {
          return [x.#unary(AluOp.Bitcast, dtype)];
        } else {
          // Just keep the same data / source, but change the dtype.
          x.#backend.incRef(x.#source);
          const pending = x.#pending;
          for (const exe of pending) exe.updateRc(+1);
          return [x.#newArrayFrom({ dtype, weakType: false, pending })];
        }
      },
      [Primitive.Sin]([x]) {
        return [x.#unary(AluOp.Sin)];
      },
      [Primitive.Cos]([x]) {
        return [x.#unary(AluOp.Cos)];
      },
      [Primitive.Asin]([x]) {
        return [x.#unary(AluOp.Asin)];
      },
      [Primitive.Atan]([x]) {
        return [x.#unary(AluOp.Atan)];
      },
      [Primitive.Exp]([x]) {
        return [x.#unary(AluOp.Exp)];
      },
      [Primitive.Log]([x]) {
        return [x.#unary(AluOp.Log)];
      },
      [Primitive.Erf]([x]) {
        return [x.#unary(AluOp.Erf)];
      },
      [Primitive.Erfc]([x]) {
        return [x.#unary(AluOp.Erfc)];
      },
      [Primitive.Sqrt]([x]) {
        return [x.#unary(AluOp.Sqrt)];
      },
      [Primitive.Reduce]([x], { op, axis }) {
        if (axis.length === 0) return [x];
        const moved = x.#moveAxesDown(axis);
        const result = moved.#reduce(op);
        moved.dispose();
        return [result];
      },
      [Primitive.Pool]([x], { window, strides }) {
        const st = pool(x.#st, window, strides);
        return [x.#reshape(st)];
      },
      [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
        const n = inShape.length;
        let st = poolTranspose(x.#st, inShape, window, strides);
        st = st.reshape([...st.shape.slice(0, n), prod(st.shape.slice(n))]);
        const reshaped = x.#reshape(st);
        const result = reshaped.#reduce(AluOp.Add);
        reshaped.dispose();
        return [result];
      },
      [Primitive.Dot]([x, y]) {
        return [
          Array.#naryCustom(
            "dot",
            ([x, y]: AluExp[]) => AluExp.mul(x, y),
            [x, y],
            { reduceAxis: true },
          ),
        ];
      },
      [Primitive.Conv]([x, y], params) {
        checkConvShape(x.shape, y.shape, params);
        const [stX, stY] = prepareConv(x.#st, y.#st, params);
        const xView = x.#reshape(stX);
        const yView = y.#reshape(stY);
        const result = Array.#naryCustom(
          "conv",
          ([x, y]: AluExp[]) => AluExp.mul(x, y),
          [xView, yView],
          { reduceAxis: true },
        );
        xView.dispose();
        yView.dispose();
        return [result];
      },
      [Primitive.Compare]([x, y], { op }) {
        const custom = ([x, y]: AluExp[]) => aluCompare(x, y, op);
        return [
          Array.#naryCustom("compare", custom, [x, y], {
            strongTypeOutput: true, // outputs strongly typed bool
          }),
        ];
      },
      [Primitive.Where]([cond, x, y]) {
        const custom = ([cond, x, y]: AluExp[]) => AluExp.where(cond, x, y);
        return [
          Array.#naryCustom("where", custom, [cond, x, y], {
            dtypeOverride: [DType.Bool],
          }),
        ];
      },
      [Primitive.Concatenate](xs, { axis }) {
        // Lower to a pad of all the arrays onto the final size, then sum.
        const ndim = xs[0].ndim;
        const sizes = xs.map((x) => x.shape[axis]);
        const finalSize = sizes.reduce((a, b) => a + b, 0);
        const makePadAxis = (start: number, end: number): [number, number][] =>
          range(ndim).map((i) => (i === axis ? [start, end] : [0, 0]));
        let cum = 0;
        const xsPadded: Array[] = [];
        for (let i = 0; i < xs.length; i++) {
          const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
          xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
          cum += sizes[i];
        }
        const custom = (exps: AluExp[]) => exps.reduce(AluExp.add);
        const result = Array.#naryCustom("concatenate", custom, xsPadded);
        for (const p of xsPadded) p.dispose();
        return [result];
      },
      [Primitive.Split]([x], { axis, sizes }) {
        const outputs: Array[] = [];
        for (let i = 0, start = 0; i < sizes.length; i++) {
          const slice = range(x.ndim).map<Pair>((d) =>
            d === axis ? [start, start + sizes[i]] : [0, x.shape[d]],
          );
          outputs.push(x.#reshape(x.#st.shrink(slice)));
          start += sizes[i];
        }
        return outputs;
      },
      [Primitive.RandomBits]([k0, k1], { shape, mode }) {
        const keyShape = k0.shape;
        const genShape = shape.slice(keyShape.length);
        // Arrays of size >2^32 won't fit into browser memory anyway, so it's
        // okay to take lazy iota this way for counters.
        using c0 = full(genShape, 0, {
          dtype: DType.Uint32,
          device: k0.device,
        });
        using c1 = arange(0, prod(genShape), 1, {
          dtype: DType.Uint32,
          device: k0.device,
        }).reshape(genShape);
        const k0View = k0.#reshape(
          k0.#st.reshape(keyShape.concat(rep(genShape.length, 1))),
        );
        const k1View = k1.#reshape(
          k1.#st.reshape(keyShape.concat(rep(genShape.length, 1))),
        );
        const custom = ([k0, k1, c0, c1]: AluExp[]) =>
          AluExp.threefry2x32(k0, k1, c0, c1, mode);
        const result = Array.#naryCustom("random_bits", custom, [
          k0View,
          k1View,
          c0,
          c1,
        ]);
        k0View.dispose();
        k1View.dispose();
        return [result];
      },
      [Primitive.Gather]([x, ...indices], { axis, outDim }) {
        return [x.#gather(indices, axis, outDim)];
      },
      [Primitive.Transpose]([x], { perm }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast]([x], { shape, axis }) {
        return [x.#reshape(x.#st.broadcast(shape as number[], axis))];
      },
      [Primitive.Reshape]([x], { shape }) {
        return [x.#reshape(x.#st.reshape(shape))];
      },
      [Primitive.Flip]([x], { axis }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis) arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      [Primitive.Reverse]([x], { axis }) {
        const arg = rep(x.ndim, false);
        arg[axis] = true;
        const flipped = x.#reshape(x.#st.flip(arg));
        flipped.#realize();
        return [flipped];
      },
      [Primitive.Shrink]([x], { slice }) {
        return [x.#reshape(x.#st.shrink(slice))];
      },
      [Primitive.Pad]([x], { width }) {
        return [x.#reshape(x.#st.pad(width))];
      },
      [Primitive.Sort]: Array.#routine(Primitive.Sort),
      [Primitive.Argsort]: Array.#routine(Primitive.Argsort),
      [Primitive.TriangularSolve]: Array.#routine(Primitive.TriangularSolve),
      [Primitive.Cholesky]: Array.#routine(Primitive.Cholesky),
      [Primitive.LU]: Array.#routine(Primitive.LU),
      [Primitive.QR]: Array.#routine(Primitive.QR),
      [Primitive.Jit](args, { jaxpr, name, numConsts, dynamicAxes }) {
        if (jaxpr.inBinders.length !== args.length) {
          throw new Error(
            `jit expects ${jaxpr.inBinders.length} args, got ${args.length}`,
          );
        }

        const { backend, committed } = Array.#computeBackend("jit", args);

        args = args.map((ar) => ar._putSync(backend));

        // Realize all input slots upfront.
        const slots = args.map((x) => x._realizeSource());

        // Batch input pending ops into one queue.submit() instead of
        // submitting each dispatch individually. This eliminates per-dispatch
        // GPU submission overhead when chaining jit calls (e.g., 6
        // transformer layers go from ~50 submissions down to ~12).
        backend.beginBatch?.();
        try {
          // Flush all input pending ops to ensure data is materialized
          // before JIT execution. This is critical for scan's fallback loop
          // which reads slots synchronously within the JIT program.
          for (const ar of args) {
            for (const exe of ar.#pending) {
              exe.prepareSync();
              exe.submit();
            }
          }
        } finally {
          backend.endBatch?.();
        }

        // Build dimension bindings from concrete input shapes if the Jaxpr
        // was traced with symbolic dims.
        let dimBindings: ReadonlyMap<string, number> | undefined;
        if (dynamicAxes) {
          const bindings = new Map<string, number>();
          for (const [axisStr, dimName] of Object.entries(dynamicAxes)) {
            const axis = Number(axisStr);
            // Use the first dynamic arg's concrete shape to resolve the binding.
            // Skip captured constants — dynamic_axes refers to user input args.
            const concreteVal = args[numConsts]?.shape[axis] as number;
            bindings.set(dimName, concreteVal);
          }
          dimBindings = bindings;
        }

        const jp = jitCompile(backend, jaxpr, dimBindings, name);
        const { outputs, pending } = jp.execute(slots, dimBindings);

        // Batch output pending ops into one submit so the next jit call
        // finds no unflushed ops (#pending getter filters submitted ops).
        if (pending.length > 0) {
          backend.beginBatch?.();
          try {
            for (const exe of pending) {
              exe.prepareSync();
              exe.submit();
            }
          } finally {
            backend.endBatch?.();
          }
          for (const exe of pending) exe.updateRc(+outputs.length - 1);
        }

        return outputs.map((source, i) => {
          const outShape = dimBindings
            ? resolveShape(jaxpr.outs[i].aval.shape, dimBindings)
            : (jaxpr.outs[i].aval.shape as number[]);
          return new Array({
            source,
            st: ShapeTracker.fromShape(outShape),
            dtype: jaxpr.outs[i].aval.dtype,
            weakType: jaxpr.outs[i].aval.weakType,
            backend,
            committed,
            pending,
          });
        });
      },
      [Primitive.DynamicUpdateSlice]([dst, src], { offset, axis }) {
        const dstShape = dst.shape;
        const srcShape = src.shape;

        // Mode 1: same rank, axis in range
        if (dstShape.length === srcShape.length) {
          for (let i = 0; i < dstShape.length; i++) {
            if (i === axis) continue;
            if (dstShape[i] !== srcShape[i]) {
              throw new Error(
                "dynamicUpdateSlice: dst and src must match on non-updated axes",
              );
            }
          }
          if (offset + srcShape[axis] > dstShape[axis]) {
            throw new Error(
              "dynamicUpdateSlice: offset + src.shape[axis] out of bounds",
            );
          }
          const innerBefore = prod(dstShape.slice(axis + 1));
          const outerBefore = prod(dstShape.slice(0, axis));
          const dstData = dst.dataSync();
          const srcData = src.dataSync();
          for (let out = 0; out < outerBefore; out++) {
            for (let i = 0; i < srcShape[axis]; i++) {
              const srcStart = (out * srcShape[axis] + i) * innerBefore;
              const dstStart =
                (out * dstShape[axis] + offset + i) * innerBefore;
              dstData.set(
                srcData.subarray(srcStart, srcStart + innerBefore),
                dstStart,
              );
            }
          }
          return [
            array(dstData, {
              shape: dstShape,
              dtype: dst.dtype,
              device: dst.device,
            }),
          ];
        }

        // Mode 2: stacked dst at axis=0 — dst has one extra leading dim
        if (axis === 0 && dstShape.length === srcShape.length + 1) {
          for (let i = 0; i < srcShape.length; i++) {
            if (dstShape[i + 1] !== srcShape[i]) {
              throw new Error(
                "dynamicUpdateSlice: dst and src must match on non-updated axes (stacked mode)",
              );
            }
          }
          if (offset + 1 > dstShape[0]) {
            throw new Error(
              "dynamicUpdateSlice: offset out of bounds for stacked dst",
            );
          }

          // Fast path: backend buffer copy
          const { backend } = Array.#computeBackend("dynamicUpdateSlice", [
            dst,
            src,
          ]);
          if (backend.copyBufferToBuffer) {
            const dstSlot = dst._realizeSource();
            const srcSlot = src._realizeSource();
            const innerSizeBytes = prod(srcShape) * byteWidth(dst.dtype);
            const dstOffsetBytes = offset * innerSizeBytes;
            const canBufferCopy =
              dstOffsetBytes % 4 === 0 && innerSizeBytes % 4 === 0;
            if (canBufferCopy) {
              // Flush pending ops on both
              for (const p of dst.#pending) {
                p.prepareSync();
                p.submit();
              }
              for (const p of src.#pending) {
                p.prepareSync();
                p.submit();
              }
              backend.copyBufferToBuffer(
                srcSlot,
                0,
                dstSlot,
                dstOffsetBytes,
                innerSizeBytes,
              );
              backend.incRef(dstSlot);
              return [
                new Array({
                  source: dstSlot,
                  st: ShapeTracker.fromShape(dstShape),
                  dtype: dst.dtype,
                  weakType: dst.weakType,
                  backend,
                  committed: true,
                  pending: [],
                }),
              ];
            }
          }

          // CPU fallback: read, patch, create new array
          const dstData = dst.dataSync();
          const srcData = src.dataSync();
          const innerSize = prod(srcShape);
          const dstStart = offset * innerSize;
          (dstData as any).set(srcData as any, dstStart);
          return [
            array(dstData, {
              shape: dstShape,
              dtype: dst.dtype,
              device: dst.device,
            }),
          ];
        }

        throw new Error(
          "dynamicUpdateSlice: unsupported dst/src shapes for update",
        );
      },
      [Primitive.DynamicUpdateSliceGeneral](
        [dst, src],
        { startIndices }: { startIndices: number[] },
      ) {
        // ND eager update: copy dst, then patch the ND sub-block with src.
        // Uses stride-based addressing for O(srcElements) copy cost.
        const dstShape = dst.shape as number[];
        const srcShape = src.shape as number[];
        const rank = dstShape.length;

        // Bounds validation (mirrors jaxpr.ts abstract eval)
        for (let i = 0; i < rank; i++) {
          if (startIndices[i] + srcShape[i] > dstShape[i]) {
            throw new Error(
              `dynamicUpdateSliceGeneral: out of bounds on axis ${i}: ` +
                `start ${startIndices[i]} + src extent ${srcShape[i]} > dst extent ${dstShape[i]}`,
            );
          }
        }

        const dstData = dst.dataSync();
        const srcData = src.dataSync();
        // resultData is a copy of dstData
        const TypedArr = dstData.constructor as any;
        const resultData = new TypedArr(dstData);

        // Precompute dst strides (row-major)
        const dstStrides = new JsArray(rank);
        dstStrides[rank - 1] = 1;
        for (let i = rank - 2; i >= 0; i--) {
          dstStrides[i] = dstStrides[i + 1] * dstShape[i + 1];
        }

        // Walk src elements using stride-based addressing
        const srcElements = prod(srcShape);
        const coords: number[] = new JsArray(rank).fill(0);
        for (let flat = 0; flat < srcElements; flat++) {
          // Compute dst offset from coords + startIndices
          let dstOff = 0;
          for (let d = 0; d < rank; d++) {
            dstOff += (coords[d] + startIndices[d]) * dstStrides[d];
          }
          resultData[dstOff] = srcData[flat];

          // Increment coords (innermost first)
          for (let d = rank - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < srcShape[d]) break;
            coords[d] = 0;
          }
        }

        return [
          array(resultData, {
            shape: dstShape,
            dtype: dst.dtype,
            device: dst.device,
          }),
        ];
      },
      [Primitive.ScatterAdd]([target, indices, updates], { axis }) {
        // Eager scatter_add: target[indices[i]] += updates[i] along axis.
        // Returns a new array — target is NOT consumed.
        const targetShape = target.shape;
        const indicesData = indices.dataSync() as Int32Array;
        const updatesData = updates.dataSync();
        const resultData = target.dataSync();
        const n = indicesData.length;
        const innerBefore = prod(targetShape.slice(axis + 1));
        const outerBefore = prod(targetShape.slice(0, axis));
        // For each update position, walk the outer × inner grid.
        for (let out = 0; out < outerBefore; out++) {
          for (let j = 0; j < n; j++) {
            const idx = indicesData[j];
            for (let inner = 0; inner < innerBefore; inner++) {
              const targetOfs =
                (out * targetShape[axis] + idx) * innerBefore + inner;
              const updateOfs = (out * n + j) * innerBefore + inner;
              (resultData as any)[targetOfs] += (updatesData as any)[updateOfs];
            }
          }
        }
        return [
          array(resultData, {
            shape: targetShape,
            dtype: target.dtype,
            device: target.device,
          }),
        ];
      },
      [Primitive.Scan](
        args,
        { jaxpr, numCarry, numConsts, length: lengthDim, reverse, acceptPath },
      ) {
        // Scan primitive: executes jaxpr in a loop, threading carry state
        // Args layout: [...consts, ...initCarry, ...xs]
        // jaxpr inputs: [...consts, ...carry, ...x_slice]
        // jaxpr outputs: [...newCarry, ...y_slice]
        const length = concreteDim(lengthDim, "scan eager impl");

        const _consts = args.slice(0, numConsts);
        const _initCarry = args.slice(numConsts, numConsts + numCarry);
        const xs = args.slice(numConsts + numCarry);

        const numX = xs.length;
        const numY = jaxpr.outs.length - numCarry;

        if (jaxpr.inBinders.length !== numConsts + numCarry + numX) {
          throw new Error(
            `scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`,
          );
        }

        const { backend, committed } = Array.#computeBackend("scan", args);
        // Move all args to the same backend
        const allArgs = args.map((ar) => ar._putSync(backend));
        const constArgs = allArgs.slice(0, numConsts);
        const initCarryArgs = allArgs.slice(numConsts, numConsts + numCarry);
        const xsArgs = allArgs.slice(numConsts + numCarry);

        // JIT-compile body for fused per-iteration execution
        const bodyProgram = jitCompile(backend, jaxpr, undefined, "scan_body");

        // Determine scan plan (P0: always fallback)
        const plan = planScan(
          backend,
          bodyProgram,
          jaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          acceptPath as ScanPath | ScanPath[] | undefined,
        );

        // Realize all input slots and flush pending ops
        const constSlots = constArgs.map((c) => {
          const slot = c._realizeSource();
          for (const p of c.#pending) {
            p.prepareSync();
            p.submit();
          }
          return slot;
        });
        const initCarrySlots = initCarryArgs.map((c) => {
          const slot = c._realizeSource();
          for (const p of c.#pending) {
            p.prepareSync();
            p.submit();
          }
          return slot;
        });
        const xsSlots = xsArgs.map((x) => {
          const slot = x._realizeSource();
          for (const p of x.#pending) {
            p.prepareSync();
            p.submit();
          }
          return slot;
        });

        // Compute xs avals (per-slice shape, without leading length dim)
        const xsAvals = xsArgs.map(
          (x) => new ShapedArray(x.shape.slice(1), x.dtype, x.aval.weakType),
        );

        // Preallocate output slots: [carry_out..., stacked_ys...]
        const outputSlots: Slot[] = [];
        const bodyOutAvals = jaxpr.outs.map((v) => v.aval);

        // Carry output slots
        for (let ci = 0; ci < numCarry; ci++) {
          const aval = bodyOutAvals[ci];
          const sizeBytes =
            prod(aval.shape as number[]) * byteWidth(aval.dtype);
          outputSlots.push(backend.malloc(sizeBytes > 0 ? sizeBytes : 1));
        }

        // Stacked Y output slots
        for (let yi = 0; yi < numY; yi++) {
          const aval = bodyOutAvals[numCarry + yi];
          const totalSizeBytes =
            length * prod(aval.shape as number[]) * byteWidth(aval.dtype);
          outputSlots.push(
            backend.malloc(totalSizeBytes > 0 ? totalSizeBytes : 1),
          );
        }

        // IncRef slots that executeScan borrows (consts and xs)
        for (const s of constSlots) backend.incRef(s);
        for (const s of xsSlots) backend.incRef(s);

        const result = executeScan({
          backend,
          plan,
          bodyProgram,
          bodyJaxpr: jaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          constSlots,
          initCarrySlots,
          xsSlots,
          xsAvals,
          outputSlots,
        });

        // Flush any remaining pending ops
        for (const p of result.pending) {
          p.prepareSync();
          p.submit();
        }

        // Dispose original args (consts, xs refs consumed above)
        for (const s of constSlots) backend.decRef(s);
        for (const s of xsSlots) backend.decRef(s);

        // Wrap output slots as Arrays
        const outputs: Array[] = [];

        // Carry outputs
        for (let ci = 0; ci < numCarry; ci++) {
          const aval = bodyOutAvals[ci];
          outputs.push(
            new Array({
              source: result.outputs[ci],
              st: ShapeTracker.fromShape(aval.shape as number[]),
              dtype: aval.dtype,
              weakType: aval.weakType,
              backend,
              committed,
              pending: [],
            }),
          );
        }

        // Stacked Y outputs
        for (let yi = 0; yi < numY; yi++) {
          const aval = bodyOutAvals[numCarry + yi];
          outputs.push(
            new Array({
              source: result.outputs[numCarry + yi],
              st: ShapeTracker.fromShape([length, ...(aval.shape as number[])]),
              dtype: aval.dtype,
              weakType: aval.weakType,
              backend,
              committed,
              pending: [],
            }),
          );
        }

        return outputs;
      },
      [Primitive.AssociativeScan](
        args,
        { jaxpr: bodyJaxpr, numLeaves, axis, reverse },
      ) {
        // Eager impl: delegate to the Kogge-Stone core algorithm
        // registered by lax-associative-scan.ts.
        // The core impl uses vmap internally to vectorize the element-level
        // body jaxpr over the batch dimension.
        const numConsts = args.length - numLeaves;
        const consts = args.slice(0, numConsts);
        const elemsLeaves = args.slice(numConsts);

        if (!_associativeScanCoreImpl) {
          throw new Error(
            "AssociativeScan: core implementation not registered. " +
              "Import lax-associative-scan.ts first.",
          );
        }

        return _associativeScanCoreImpl(
          bodyJaxpr,
          consts,
          elemsLeaves,
          numLeaves,
          axis,
          reverse,
        );
      },
      [Primitive.BlockIndex]() {
        throw new Error("blockIndex() is only valid inside a block_map body");
      },
      [Primitive.BlockMap](
        args,
        {
          jaxpr: bodyJaxpr,
          blockShape,
          inAxes,
          outAxes,
          numConsts,
          numInputs,
          halo,
        },
      ) {
        // Eager impl: slice/pad/concat loop over the grid of blocks.
        const consts = args.slice(0, numConsts);
        const inputs = args.slice(numConsts, numConsts + numInputs);

        // Compute grid shape from inputs + inAxes (from ORIGINAL dims, not halo-expanded)
        const gridRank = blockShape.length;
        const gridShape: number[] = new JsArray(gridRank).fill(0);
        for (let i = 0; i < inputs.length; i++) {
          const axes = inAxes[i];
          for (let g = 0; g < gridRank; g++) {
            if (axes[g] !== null) {
              const dim = inputs[i].shape[axes[g]!] as number;
              gridShape[g] = Math.ceil(dim / blockShape[g]);
            }
          }
        }

        // Total number of blocks
        const numBlocks = gridShape.reduce((a, b) => a * b, 1);

        // Collect block outputs per output index
        const numOutputs = bodyJaxpr.outs.length;
        const outputBlocks: Array[][] = JsArray.from(
          { length: numOutputs },
          () => [],
        );

        // Iterate over all block index tuples (row-major order)
        for (let flatIdx = 0; flatIdx < numBlocks; flatIdx++) {
          // Convert flat index to grid coordinates
          const blockIdx: number[] = new JsArray(gridRank);
          let remaining = flatIdx;
          for (let g = gridRank - 1; g >= 0; g--) {
            blockIdx[g] = remaining % gridShape[g];
            remaining = Math.floor(remaining / gridShape[g]);
          }

          // Slice each input along its mapped axes, pad if non-divisible or halo
          const blockInputs: Array[] = [];
          for (let i = 0; i < inputs.length; i++) {
            const inputArr = inputs[i];
            const axes = inAxes[i];
            const nd = inputArr.ndim;

            // Build the shrink slice for all dims at once
            const slice: Pair[] = range(nd).map((d) => [
              0,
              inputArr.shape[d] as number,
            ]);
            // Desired tile size per axis (including halo)
            const tileSize: Map<number, number> = new Map();
            for (let g = 0; g < gridRank; g++) {
              if (axes[g] !== null) {
                const ax = axes[g]!;
                const [lo, hi] = halo ? halo[i][g] : [0, 0];
                const start = blockIdx[g] * blockShape[g] - lo;
                const fullDim = inputArr.shape[ax] as number;
                const end = (blockIdx[g] + 1) * blockShape[g] + hi;
                // Clamp to valid range
                slice[ax] = [Math.max(0, start), Math.min(end, fullDim)];
                tileSize.set(ax, blockShape[g] + lo + hi);
              }
            }

            let current = coreShrink(inputArr, slice) as Array;

            // Pad to target tile size (blockShape + halo) if needed
            // Left-pad when start < 0 (halo extends before array), right-pad for edge blocks
            const padWidth: Pair[] = range(current.ndim).map(
              () => [0, 0] as Pair,
            );
            let anyPad = false;
            for (let g = 0; g < gridRank; g++) {
              if (axes[g] !== null) {
                const ax = axes[g]!;
                const [lo, _hi] = halo ? halo[i][g] : [0, 0];
                const start = blockIdx[g] * blockShape[g] - lo;
                const target = tileSize.get(ax)!;
                const actual = current.shape[ax] as number;
                const leftPad = start < 0 ? -start : 0;
                const rightPad = target - actual - leftPad;
                if (leftPad > 0 || rightPad > 0) {
                  padWidth[ax] = [leftPad, rightPad];
                  anyPad = true;
                }
              }
            }
            if (anyPad) {
              const padded = corePad(current, padWidth) as Array;
              current.dispose();
              current = padded;
            }

            blockInputs.push(current);
          }

          // Execute body jaxpr on this block
          // evalJaxpr is non-consuming — consts and blockInputs stay alive.
          const bodyOuts = evalJaxpr(bodyJaxpr, [...consts, ...blockInputs]);

          // Dispose sliced block inputs
          for (const bi of blockInputs) bi.dispose();

          // Collect outputs
          for (let o = 0; o < numOutputs; o++) {
            outputBlocks[o].push(bodyOuts[o] as Array);
          }
        }

        // Concatenate block outputs and trim padding
        const results: Array[] = [];
        for (let o = 0; o < numOutputs; o++) {
          const axes = outAxes[o];

          if (numBlocks === 0) {
            const bodyOutAval = bodyJaxpr.outs[o].aval;
            const fullShape = [...bodyOutAval.shape] as number[];
            for (let g = 0; g < gridRank; g++) {
              if (axes[g] !== null) fullShape[axes[g]!] = 0;
            }
            results.push(
              zeros(fullShape, { dtype: bodyOutAval.dtype }) as Array,
            );
            continue;
          }

          // Hierarchical concatenation/reduction: from last grid axis to first.
          // blocks[o] are in row-major order over the grid.
          // For each grid axis (from innermost): concat mapped axes, sum null axes.
          let current = outputBlocks[o] as Array[];
          for (let g = gridRank - 1; g >= 0; g--) {
            const stride = gridShape[g];
            if (axes[g] === null) {
              // Null outAxis: reduce (sum) blocks along this grid dimension.
              // This handles backward-pass accumulation for unmapped input axes.
              if (stride <= 1) continue;
              const reduced: Array[] = [];
              for (let start = 0; start < current.length; start += stride) {
                let acc = coreAdd(current[start], current[start + 1]) as Array;
                for (let d = 2; d < stride; d++) {
                  const next = coreAdd(acc, current[start + d]) as Array;
                  acc.dispose();
                  acc = next;
                }
                reduced.push(acc);
              }
              if (current !== outputBlocks[o]) {
                for (const c of current) c.dispose();
              }
              current = reduced;
              continue;
            }
            const concatAxis = axes[g]!;
            const grouped: Array[] = [];
            for (let start = 0; start < current.length; start += stride) {
              const group = current.slice(start, start + stride);
              const merged = coreConcatenate(group, concatAxis) as Array;
              grouped.push(merged);
            }
            // Dispose intermediates from previous level (not original blocks)
            if (current !== outputBlocks[o]) {
              for (const c of current) c.dispose();
            }
            current = grouped;
          }

          let result =
            current.length === 1
              ? current[0]
              : (coreConcatenate(current, 0) as Array);

          // Dispose intermediate concat results (but not if result is one of them)
          if (current !== outputBlocks[o] && current.length > 1) {
            for (const c of current) {
              if (c !== result) c.dispose();
            }
          }

          // Trim padding if input dimension wasn't evenly divisible
          const resultShape = result.shape as number[];
          let needsTrim = false;
          const trimSlice: Pair[] = resultShape.map((s) => [0, s] as Pair);
          for (let g = 0; g < gridRank; g++) {
            if (axes[g] !== null) {
              const ax = axes[g]!;
              for (let i = 0; i < inputs.length; i++) {
                if (inAxes[i][g] !== null) {
                  const origDim = inputs[i].shape[inAxes[i][g]!] as number;
                  if (origDim < resultShape[ax]) {
                    trimSlice[ax] = [0, origDim];
                    needsTrim = true;
                  }
                  break;
                }
              }
            }
          }

          if (needsTrim) {
            const trimmed = coreShrink(result, trimSlice) as Array;
            result.dispose();
            result = trimmed;
          }

          // Dispose original block outputs
          for (const b of outputBlocks[o]) b.dispose();

          results.push(result);
        }

        return results;
      },
      [Primitive.ForiLoop](
        args,
        { jaxpr, numConsts, lower: lowerDim, upper: upperDim },
      ) {
        const lower = concreteDim(lowerDim, "foriLoop eager");
        const upper = concreteDim(upperDim, "foriLoop eager");
        const consts = args.slice(0, numConsts);
        let carry = args.slice(numConsts);
        for (let i = lower; i < upper; i++) {
          using idx = array(i, { dtype: DType.Int32 }) as Array;
          // non-consuming
          const nextCarry = evalJaxpr(jaxpr, [
            ...consts,
            idx,
            ...carry,
          ]) as Array[];
          for (const c of carry) {
            // only dispose intermediate carries, not original inputs
            if (!args.includes(c)) c.dispose();
          }
          carry = nextCarry;
        }
        // The outputs of a primitive MUST be freshly-owned references.
        // If an output is just an input passed through, we must .ref it.
        // jax-js-lint: allow-ref
        return carry.map((c) => (args.includes(c) ? c.ref : c));
      },
      [Primitive.WorkgroupAssociativeScan](args, { jaxpr, numConsts }) {
        // Eager impl: delegate to the Kogge-Stone core algorithm
        // registered by lax-associative-scan.ts, same as AssociativeScan.
        const consts = args.slice(0, numConsts);
        const elemsLeaves = args.slice(numConsts);

        if (!_associativeScanCoreImpl) {
          throw new Error(
            "WorkgroupAssociativeScan: core implementation not registered. " +
              "Import lax-associative-scan.ts first.",
          );
        }

        return _associativeScanCoreImpl(
          jaxpr,
          consts,
          elemsLeaves,
          elemsLeaves.length,
          0,
          false,
        );
      },
      [Primitive.DynamicSlice](args, { sliceSizes }) {
        const operand = args[0];
        const indices = args.slice(1);
        const ndim = operand.ndim;
        const startIndicesVal: number[] = new JsArray(ndim);
        for (let i = 0; i < ndim; i++) {
          // Unconditionally clamped to [0, dimSize - sliceSize]
          const maxStart = (operand.shape[i] as number) - sliceSizes[i];
          const rawStart = indices[i].dataSync()[0] as number;
          startIndicesVal[i] = Math.max(0, Math.min(rawStart, maxStart));
        }
        const slicePair = startIndicesVal.map(
          (start, i) => [start, start + sliceSizes[i]] as Pair,
        );
        return [coreShrink(operand, slicePair) as Array];
      },
      [Primitive.UncheckedDynamicSlice](args, { sliceSizes }) {
        const operand = args[0];
        const indices = args.slice(1);
        const ndim = operand.ndim;
        const startIndicesVal: number[] = new JsArray(ndim);
        for (let i = 0; i < ndim; i++) {
          // No clamping — caller guarantees in-bounds
          startIndicesVal[i] = indices[i].dataSync()[0] as number;
        }
        const slicePair = startIndicesVal.map(
          (start, i) => [start, start + sliceSizes[i]] as Pair,
        );
        return [coreShrink(operand, slicePair) as Array];
      },
    };
  }

  /** @private */
  _realizeSource() {
    this.#realize();
    return this.#source as number; // Because #realize() was called.
  }

  /**
   * Submit all pending backend dispatches without reading data back.
   * Used by evalJaxprTransposed to flush result arrays' PE chains before
   * disposing intermediates — prevents orphaned Slot references.
   * @private
   */
  _flushPendingSync(): void {
    this.#realize();
    const pending = this.#pending;
    if (pending.length === 0) return;
    this.#backend.beginBatch?.();
    try {
      for (const p of pending) {
        p.prepareSync();
        p.submit();
      }
    } finally {
      this.#backend.endBatch?.();
    }
  }

  /** @private Put this array on a new backend, asynchronously. */
  async _put(backend: Backend): Promise<Array> {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      // Not realized yet, just dump the AluExp on the target backend.
      return this.#newArrayFrom({ backend, committed: true });
    } else {
      // Realize the array (handles non-contiguous ShapeTrackers) and copy data
      // to the new backend. Using data() instead of raw read() ensures the
      // ShapeTracker is applied, so transposed/reshaped views are linearized.
      const data = await this.data();
      return arrayFromData(
        data,
        this.shape,
        { dtype: this.#dtype, device: backend.type },
        this.#weakType,
      );
    }
  }

  /** @private Put this array on a new backend, synchronously. */
  _putSync(backend: Backend): Array {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      // Not realized yet, just dump the AluExp on the target backend.
      return this.#newArrayFrom({ backend, committed: true });
    } else {
      // Realize the array (handles non-contiguous ShapeTrackers) and copy data
      // to the new backend. Using dataSync() instead of raw readSync() ensures
      // the ShapeTracker is applied, so transposed/reshaped views are linearized.
      const data = this.dataSync();
      return arrayFromData(
        data,
        this.shape,
        { dtype: this.#dtype, device: backend.type },
        this.#weakType,
      );
    }
  }
}

/** Constructor for creating a new array from data. */
export function array(
  values: Array | DataArray | RecursiveArray<number> | RecursiveArray<boolean>,
  { shape, dtype, device }: { shape?: number[] } & DTypeAndDevice = {},
): Array {
  if (values instanceof Tracer) {
    if (shape && !deepEqual(values.shape, shape)) {
      values = values.reshape(shape);
    }
    if (dtype && values.dtype !== dtype) {
      values = values.astype(dtype);
    }
    return values;
  } else if (ArrayBuffer.isView(values)) {
    return arrayFromData(values, shape ?? [values.length], {
      dtype,
      device,
    });
  } else {
    // Assume this is a nested array object, infer the shape.
    if (!shape) {
      shape = [];
      let cur = values;
      while (JsArray.isArray(cur)) {
        shape.push(cur.length);
        cur = cur[0];
      }
    }
    const size = prod(shape);
    const flat = recursiveFlatten(values);
    if (flat.length !== size) {
      throw new Error(
        `Jagged shape: ${JSON.stringify(shape)} vs ${flat.length}`,
      );
    }
    if (size === 0) return zeros(shape, { dtype, device });
    if (size === 1) return full(shape, flat[0], { dtype, device });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? DType.Bool;
      const data = new Int32Array(flat.map((x) => (x ? 1 : 0)));
      return arrayFromData(data, shape, { dtype, device });
    } else {
      const weakType = dtype == undefined && shape.length === 0;
      dtype = dtype ?? DType.Float32;
      const data = dtypedJsArray(dtype, flat as number[]);
      return arrayFromData(data, shape, { dtype, device }, weakType);
    }
  }
}

function arrayFromData(
  data: DataArray,
  shape: number[],
  { dtype, device }: DTypeAndDevice,
  weakType = false,
): Array {
  if (data instanceof Float32Array) {
    if (dtype && dtype !== DType.Float32)
      throw new Error("Float32Array must have float32 type");
    dtype ??= DType.Float32;
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== DType.Int32 && dtype !== DType.Bool)
      throw new Error("Int32Array must have int32 or bool type");
    dtype ??= DType.Int32;
  } else if (data instanceof Uint32Array) {
    if (dtype && dtype !== DType.Uint32)
      throw new Error("Uint32Array must have uint32 type");
    dtype ??= DType.Uint32;
  } else if (data instanceof Float16Array) {
    if (dtype && dtype !== DType.Float16)
      throw new Error("Float16Array must have float16 type");
    dtype ??= DType.Float16;
  } else if (data instanceof Float64Array) {
    if (dtype && dtype !== DType.Float64)
      throw new Error("Float64Array must have float64 type");
    dtype ??= DType.Float64;
  } else {
    throw new Error(
      "Unsupported data array type: " + (data as any).constructor.name,
    );
  }

  if (data.length < inlineArrayLimit) {
    // Check if all elements are of the same value and short-circuit.
    let allEqual = true;
    for (let i = 1; i < data.length; i++) {
      if (data[i] !== data[0]) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) {
      // If all elements are equal, we can use a constant expression.
      const sa = new ShapedArray(shape, dtype, weakType);
      return fullInternal(sa, data[0], device);
    }
  }

  const backend = getBackend(device);
  const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const slot = backend.malloc(data.byteLength, buf);
  return new Array({
    source: slot,
    st: ShapeTracker.fromShape(shape),
    dtype,
    weakType,
    backend,
    committed: device != undefined,
  });
}

function dataToJs(
  dtype: DType,
  data: DataArray,
  shape: number[],
): RecursiveArray<number> | RecursiveArray<boolean> | any {
  if (shape.length === 0) {
    return dtype === DType.Bool ? Boolean(data[0]) : data[0];
  }
  const [first, ...rest] = shape;
  const restSize = prod(rest);
  const ret: any[] = [];
  for (let i = 0; i < first; i++) {
    const subarray = data.slice(i * restSize, (i + 1) * restSize);
    ret.push(dataToJs(dtype, subarray, rest));
  }
  return ret;
}

/** If x is a value, lift it into an array, otherwise leave it be. */
export function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) return x;
  return array(x);
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: Array[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack: must be level 0.
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));

type ImplRule<P extends Primitive> = (
  tracers: Array[],
  params: PrimitiveParams<P>,
) => Array[];
const implRules: { [P in Primitive]: ImplRule<P> } = Array._implRules();

export function fullInternal(
  aval: AbstractValue,
  fillValue: number | boolean,
  device?: Device,
) {
  return new Array({
    source: AluExp.const(aval.dtype, fillValue),
    st: ShapeTracker.fromShape(aval.shape as number[]),
    dtype: aval.dtype,
    weakType: aval.weakType,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

export function zerosLike(val: TracerValue, opts?: DTypeShapeAndDevice): Array {
  return fullLike(val, 0, opts);
}

export function onesLike(val: TracerValue, opts?: DTypeShapeAndDevice): Array {
  return fullLike(val, 1, opts);
}

export function fullLike(
  val: TracerValue,
  fillValue: number | boolean | Array,
  { dtype, shape, device }: DTypeShapeAndDevice = {},
): Array {
  const aval = getAval(val);
  if (fillValue instanceof Tracer) {
    // TODO: Full can also take an array as a fill value. This is equivalent to
    // expanding the array.
    throw new Error("numpy.fullLike() with array argument not implemented yet");
  }
  const sa = new ShapedArray(
    shape ?? aval.shape,
    dtype ?? aval.dtype,
    aval.weakType && dtype === undefined,
  );
  return fullInternal(sa, fillValue, device);
}

/** Return a new array of given shape and type, filled with zeros. */
export function zeros(shape: number[], opts?: DTypeAndDevice): Array {
  return full(shape, 0, opts);
}

/** Return a new array of given shape and type, filled with ones. */
export function ones(shape: number[], opts?: DTypeAndDevice): Array {
  return full(shape, 1, opts);
}

/** Return a new array of given shape and type, filled with `fill_value`. */
export function full(
  shape: number[],
  fillValue: number | boolean | Array,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  let weakType = dtype == undefined && shape.length === 0;
  if (typeof fillValue === "number") {
    dtype = dtype ?? DType.Float32;
  } else if (typeof fillValue === "boolean") {
    dtype = dtype ?? DType.Bool;
    weakType = false; // booleans are never weakly typed
  } else if (fillValue instanceof Tracer) {
    // TODO: Full can also take an array as a fill value. This is equivalent to
    // expanding the array.
    throw new Error("numpy.full() with array argument not implemented yet");
  } else {
    throw new TypeError(`Invalid type for full: ${fillValue}`);
  }
  return fullInternal(
    new ShapedArray(shape, dtype, weakType),
    fillValue,
    device,
  );
}

/**
 * Create an identity matrix.
 *
 * If numCols is not provided, it defaults to numRows, i.e., a square identity
 * matrix with ones on the diagonal.
 *
 * Supports both `eye(N, M, { dtype })` and `eye(N, { dtype })` calling
 * conventions. If the second argument is an object (not a number), it is treated
 * as the options parameter and `numCols` defaults to `numRows`.
 */
export function eye(
  numRows: number,
  numCols?: number | DTypeAndDevice,
  opts?: DTypeAndDevice,
): Array {
  // Handle eye(N, { dtype }) calling convention: second arg is options, not numCols
  if (
    typeof numCols === "object" &&
    numCols !== null &&
    !globalThis.Array.isArray(numCols)
  ) {
    opts = numCols as DTypeAndDevice;
    numCols = undefined;
  }
  const { dtype: dtypeOpt, device } = opts ?? {};
  numCols = (numCols as number | undefined) ?? numRows;
  let dtype = dtypeOpt;
  const weakType = dtype == undefined;
  dtype = dtype ?? DType.Float32;

  if (numCols < numRows) {
    // If less columns than rows, take the transpose since it's no longer a
    // simple modular arithmetic expression.
    const arr = eye(numCols, numRows, { dtype, device });
    return arr.transpose();
  }
  if (numRows === 0) {
    return zeros([0, numCols], { dtype, device });
  }

  const exp = AluExp.cmplt(
    AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)),
    AluExp.i32(1),
  );
  return new Array({
    source: AluExp.cast(dtype, exp),
    st: ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    weakType,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/** Return the identity matrix, with ones on the main diagonal. */
export function identity(
  n: number,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return eye(n, n, { dtype, device });
}

/**
 * Return evenly spaced values within a given interval.
 *
 * This can be called with a varying number of arguments, just like the range()
 * builtin function in Python.
 *
 * - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
 * - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
 * - `arange(start, stop, step)` creates an array starting at `start`, ending
 *   before `stop`, with a step size of `step`.
 *
 * Defaults to an integer data type. This can produce unintended results when
 * using a non-integer step, so prefer linspace() in those cases.
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Int32; // default to int32 for arange

  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  if (step === 0) {
    throw new RangeError(
      `Invalid step for arange: ${step}. Step must be non-zero.`,
    );
  }
  const size = Math.max(0, Math.ceil((stop - start) / step));
  if (size === 0) {
    return zeros([0], { dtype, device });
  }

  const exp = AluExp.add(
    AluExp.const(dtype, start),
    AluExp.mul(AluExp.cast(dtype, AluVar.idx), AluExp.const(dtype, step)),
  );
  const st = ShapeTracker.fromShape([size]);
  return new Array({
    source: exp,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/**
 * Return an array with ones on and below the diagonal and zeros elsewhere.
 *
 * If `k` is provided, it specifies the sub-diagonal on and below which the
 * array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
 * `k>0` is above it.
 */
export function tri(
  n: number,
  m?: number,
  k: number = 0,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  m ??= n;
  dtype ??= DType.Float32;
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`tri: n must be a non-negative integer, got ${n}`);
  }
  if (!Number.isInteger(m) || m < 0) {
    throw new Error(`tri: m must be a non-negative integer, got ${m}`);
  }
  if (!Number.isInteger(k)) {
    throw new Error(`tri: k must be an integer, got ${k}`);
  }
  using rows = arange(k, n + k, 1, { dtype: DType.Int32, device });
  using cols = arange(0, m, 1, { dtype: DType.Int32, device });
  using rowsReshaped = rows.reshape([n, 1]);
  // Don't use `using` on ge — astype() returns `this` for same dtype,
  // so `using ge` would dispose the returned value on block exit.
  const ge = rowsReshaped.greaterEqual(cols);
  if (ge.dtype === dtype) return ge;
  using _ge = ge;
  return ge.astype(dtype);
}

/** Return the lower triangle of an array. Must be of dimension >= 2. */
export function tril(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new Error(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  using mask = tri(n, m, k, { dtype: DType.Bool });
  using zeros = zerosLike(a);
  return where(mask, a, zeros) as Array;
}

/** Return the upper triangle of an array. Must be of dimension >= 2. */
export function triu(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new Error(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  using mask = tri(n, m, k - 1, { dtype: DType.Bool });
  using zeros = zerosLike(a);
  return where(mask, zeros, a) as Array;
}

/**
 * Return evenly spaced numbers over a specified interval.
 *
 * Returns _num_ evenly spaced samples, calculated over the interval
 * [`start`, `stop`]. The endpoint `stop` is included in the result by default,
 * but this is controlled by the `endpoint` parameter.
 *
 * The default data type is Float32. Use arange() for integer steps.
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  endpoint: boolean = true,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Float32; // default to float32 for linspace

  if (num < 0 || !Number.isInteger(num)) {
    throw new RangeError(
      `Invalid num for linspace: ${num}. Must be non-negative integer.`,
    );
  } else if (num === 0) {
    return zeros([0], { dtype, device });
  } else if (num === 1) {
    return full([1], start, { dtype, device });
  } else if (start === stop) {
    return full([num], start, { dtype, device });
  }

  // Now num >= 2, there are at least 2 points.
  const delta = stop - start;
  const denom = endpoint ? num - 1 : num;
  const exp = AluExp.cast(
    dtype,
    AluExp.add(
      AluExp.f32(start),
      AluExp.mul(
        AluExp.f32(delta / denom),
        AluExp.cast(DType.Float32, AluVar.idx),
      ),
    ),
  );
  const st = ShapeTracker.fromShape([num]);
  return new Array({
    source: exp,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/**
 * Return numbers spaced evenly on a log scale.
 *
 * In linear space, the sequence starts at `base ** start` and ends at
 * `base ** stop` (see `endpoint` below).
 *
 * @param start - `base ** start` is the starting value of the sequence.
 * @param stop - `base ** stop` is the final value of the sequence, unless `endpoint` is false.
 * @param num - Number of samples to generate. Default is 50.
 * @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
 * @param base - The base of the log space. Default is 10.
 * @returns Array of evenly spaced values on a log scale.
 */
export function logspace(
  start: number,
  stop: number,
  num: number = 50,
  endpoint: boolean = true,
  base: number = 10,
  { dtype, device }: DTypeAndDevice = {},
) {
  const y = linspace(start, stop, num, endpoint, { dtype, device });
  // base ** y = exp(log(base) * y)
  const logBase = Math.log(base);
  return coreExp(coreMul(y, logBase)) as Array;
}

export function aluCompare(a: AluExp, b: AluExp, op: CompareOp): AluExp {
  // Translate a `CompareOp` into an `AluExp` on two sub-expressions.
  switch (op) {
    case CompareOp.Less:
      return AluExp.cmplt(a, b);
    case CompareOp.Equal:
      // `x == y` is always equivalent to `!(x != y)`, including NaN
      return AluExp.cmpne(a, b).not();
    case CompareOp.NotEqual:
      return AluExp.cmpne(a, b);
    case CompareOp.LessEqual:
      // `x <= y` is equivalent to `x < y || x == y`
      return AluExp.add(AluExp.cmplt(a, b), AluExp.cmpne(a, b).not());
  }
}
