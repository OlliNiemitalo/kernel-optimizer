import {
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
} from "../alu";
import {
  _emitCodeCapture,
  _isCodeCaptureEnabled,
  Backend,
  type BackendCapabilities,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { isSymbolicSize, type SizeExpr } from "../dim";
import { Routine, Routines, runCpuRoutine } from "../routine";
import { emitTrace, isTracing, traceSourceInfo } from "../tracing";
import { tuneNullopt } from "../tuner";
import { DEBUG, FpHash, mapSetUnion, rep, runWithCache } from "../utils";
import { WasmAllocator } from "./wasm/allocator";
import {
  wasm_asin,
  wasm_atan,
  wasm_cos,
  wasm_erf,
  wasm_erfc,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./wasm/builtins";
import type { WasmMegaModule } from "./wasm/mega-module";
import { OrchestratorWorker } from "./wasm/orchestrator";
import {
  getArgsortModule,
  getCholeskyModule,
  getLUModule,
  getQRModule,
  getScatterAddModule,
  getSortModule,
  getTriangularSolveModule,
} from "./wasm/routine-provider";
import {
  configureMemoryImport,
  MAX_SHARED_PAGES,
  setUseSharedMemory,
} from "./wasm/shared-memory-config";
import { CodeGenerator } from "./wasm/wasmblr";
import { WasmWorkerPool } from "./wasm/worker-pool";

export { configureMemoryImport } from "./wasm/shared-memory-config";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmProgram {
  module: WebAssembly.Module;
}

/** A single step in the native scan body (after analysis). */
export interface GeneralScanStep {
  source: Kernel | Routine;
  inputSlots: number[];
  outputInternalIdx: number;
  outputInternalIndices?: number[];
  routineCallInfo?: {
    routineInfoIdx: number;
    staticParams: number[];
  };
}

/** Where a carry output comes from. */
export type CarryOutputSource =
  | { type: "passthrough"; carryIdx: number }
  | { type: "internal"; internalIdx: number };

/** Where a Y output comes from. */
export type YOutputSource =
  | { type: "passthrough"; carryIdx: number }
  | { type: "xs-passthrough"; xsIdx: number }
  | { type: "internal"; internalIdx: number };

/** Routine info for WASM imports in scan codegen. */
export interface ScanRoutineInfo {
  routine: Routines;
  exportName: string;
  numParams: number;
  dtype: "f32" | "f64";
  sizeParams: number[];
  unitDiagonal?: boolean;
  lower?: boolean;
}

/** Parameters for the general native scan codegen. */
export interface NativeScanGeneralParams {
  numConsts: number;
  constSizes: number[];
  numCarry: number;
  carrySizes: number[];
  numX: number;
  xsStrides: number[];
  numY: number;
  ysStrides: number[];
  internalSizes: number[];
  steps: GeneralScanStep[];
  carryOutSources: CarryOutputSource[];
  yOutputSources: YOutputSource[];
  reverse: boolean;
  auxBufferSize?: number;
  elementSize?: 4 | 8;
  routineInfos?: ScanRoutineInfo[];
}

/** Parameters for the compiled block-map WASM codegen. */
export interface BlockMapWasmParams {
  numConsts: number;
  numInputs: number;
  numOutputs: number;
  gridRank: number;
  gridShape: number[];
  blockShape: number[];
  /** [numInputs][gridRank] — which input dim maps to each grid dim, or null. */
  inAxes: (number | null)[][];
  /** [numOutputs][gridRank] — which output dim maps to each grid dim, or null. */
  outAxes: (number | null)[][];
  inputShapes: number[][];
  outputShapes: number[][];
  /** Byte sizes of each constant buffer. */
  constSizes: number[];
  /** Byte sizes of each body-input block buffer (block-shaped). */
  blockInputSizes: number[];
  /** Byte sizes of each body-output block buffer (block-shaped). */
  blockOutputSizes: number[];
  /** Internal scratch buffer sizes produced by body kernel steps. */
  internalSizes: number[];
  /** Reindexed body kernel steps. */
  steps: GeneralScanStep[];
  /** Map from body output index to the producing internal buffer index. */
  outputSources: number[];
  /** Strides (in bytes) per dimension for each input. */
  inputStrides: number[][];
  /** Strides (in bytes) per dimension for each output. */
  outputStrides: number[][];
  /**
   * Per-input per-grid-axis overlap [lo, hi]. When present, codegen emits
   * halo-shifted reads with interior fast path (skip memory.fill).
   * `halo[i][g] = [lo, hi]` — extra elements before/after tile on grid axis g.
   */
  halo?: [number, number][][];
}

const moduleCache = new Map<string, WebAssembly.Module>();

// ---------------------------------------------------------------------------
// Shared memory support (M5.1)
// ---------------------------------------------------------------------------

/**
 * Minimum number of elements in a kernel before parallel dispatch is used.
 * Below this threshold, the overhead of worker coordination exceeds the
 * benefit of parallelism.
 */
const PARALLEL_THRESHOLD = 4096;

/** Backend that compiles into WebAssembly bytecode for immediate execution. */
export class WasmBackend implements Backend {
  readonly type: Device = "wasm";
  readonly maxArgs = 64; // Arbitrary choice
  readonly capabilities: BackendCapabilities = {
    atomicF32Add: false,
    shaderF16: false,
    // SharedArrayBuffer is available when either:
    //   (a) crossOriginIsolated is true (COOP + COEP headers), or
    //   (b) the browser enables it unconditionally (e.g., Chromium
    //       --enable-features=SharedArrayBuffer; Node.js)
    // We test constructability rather than relying on crossOriginIsolated
    // so that the Chromium flag and non-browser runtimes work too.
    sharedMemory: (() => {
      try {
        return new SharedArrayBuffer(1).byteLength === 1;
      } catch {
        return false;
      }
    })(),
    multiOutputKernel: true,
  };

  #memory: WebAssembly.Memory;
  #nextSlot: number;
  #allocator: WasmAllocator;
  #buffers: Map<Slot, WasmBuffer>;
  /** Cache WebAssembly instances keyed by module for reuse in dispatch. */
  #instanceCache: WeakMap<WebAssembly.Module, WebAssembly.Instance>;
  /**
   * Whether this thread can spin-wait for Workers to complete.
   *
   * On browser main threads, `Atomics.wait` is forbidden (throws TypeError)
   * and — critically — `postMessage` delivery to Worker threads requires the
   * main thread's event loop to process the message.  A tight `Atomics.load`
   * spin-loop blocks the event loop, so the Worker never receives the message,
   * causing deadlock.  This affects **all** browser engines (Chrome, Firefox,
   * Safari) running on the main thread.
   *
   * On Node.js, `Atomics.wait` works on the main thread and message
   * delivery is independent of the event loop, so spin-waits work correctly.
   *
   * We detect this by probing `Atomics.wait` — if it throws, we're on a
   * browser main thread where spin-wait Worker patterns are broken.
   */
  #canSpinWaitWorkers: boolean;
  /**
   * Worker pool for parallel kernel dispatch (only when shared memory).
   * Created lazily on first parallel dispatch to avoid spawning Workers
   * when they are never needed (most tests use arrays < PARALLEL_THRESHOLD).
   * `undefined` = not yet attempted, `null` = attempted but failed/unsupported.
   */
  #workerPool: WasmWorkerPool | null | undefined = undefined;
  /**
   * Orchestrator worker for off-main-thread mega-module execution (M6.2b).
   * Created lazily on first mega-module dispatch.
   * `undefined` = not yet attempted, `null` = attempted but failed/unsupported.
   */
  #orchestrator: OrchestratorWorker | null | undefined = undefined;

  constructor() {
    const shared = this.capabilities.sharedMemory;
    setUseSharedMemory(shared);
    this.#memory = shared
      ? new WebAssembly.Memory({
          initial: 0,
          maximum: MAX_SHARED_PAGES,
          shared: true,
        })
      : new WebAssembly.Memory({ initial: 0 });
    this.#allocator = new WasmAllocator(this.#memory);
    this.#nextSlot = 1;
    this.#buffers = new Map();
    this.#instanceCache = new WeakMap();

    // Probe whether spin-wait Worker patterns work on this thread.
    // Atomics.wait throws on browser main threads — use that as a proxy.
    this.#canSpinWaitWorkers = (() => {
      if (!shared) return false;
      try {
        const probe = new Int32Array(new SharedArrayBuffer(4));
        // value !== 0, so returns "not-equal" immediately (no actual wait)
        Atomics.wait(probe, 0, 1, 0);
        return true;
      } catch {
        return false;
      }
    })();
    // Workers/orchestrator are created lazily — see #getWorkerPool(), #getOrchestrator().
  }

  /**
   * Lazily create or return the WasmWorkerPool.
   * Returns null when SharedArrayBuffer, Workers, or main-thread spin-waits
   * are unavailable (browsers forbid Atomics.wait on the main thread, and
   * postMessage delivery to Workers requires the event loop to process —
   * a main-thread spin-loop prevents message delivery, causing deadlock).
   */
  #getWorkerPool(): WasmWorkerPool | null {
    if (this.#workerPool !== undefined) return this.#workerPool;
    if (
      !this.capabilities.sharedMemory ||
      typeof Worker === "undefined" ||
      !this.#canSpinWaitWorkers
    ) {
      this.#workerPool = null;
      return null;
    }
    try {
      this.#workerPool = new WasmWorkerPool(this.#memory);
    } catch {
      // Worker creation can fail (e.g., CSP restrictions); fall back silently
      this.#workerPool = null;
    }
    return this.#workerPool;
  }

  /**
   * Lazily create or return the OrchestratorWorker.
   * Returns null when SharedArrayBuffer, Workers, or main-thread spin-waits
   * are unavailable (see #getWorkerPool comment for rationale).
   */
  #getOrchestrator(): OrchestratorWorker | null {
    if (this.#orchestrator !== undefined) return this.#orchestrator;
    if (
      !this.capabilities.sharedMemory ||
      typeof Worker === "undefined" ||
      !this.#canSpinWaitWorkers
    ) {
      this.#orchestrator = null;
      return null;
    }
    try {
      this.#orchestrator = new OrchestratorWorker(this.#memory);
    } catch {
      this.#orchestrator = null;
    }
    return this.#orchestrator;
  }

  /** The worker pool, if available. Exposed for testing. */
  get workerPool(): WasmWorkerPool | null {
    return this.#workerPool ?? null;
  }

  /** The orchestrator worker, if available. Exposed for testing. */
  get orchestrator(): OrchestratorWorker | null {
    return this.#orchestrator ?? null;
  }

  /** Whether spin-wait Worker patterns work on this thread. Exposed for testing. */
  get canSpinWaitWorkers(): boolean {
    return this.#canSpinWaitWorkers;
  }

  /**
   * Tear down background workers (worker pool + orchestrator).
   * Called during test teardown to allow vitest/Playwright to exit cleanly.
   * In production, browser tab close terminates workers automatically.
   */
  destroyWorkers(): void {
    if (this.#workerPool) {
      this.#workerPool.destroy();
      this.#workerPool = null;
    }
    if (this.#orchestrator) {
      this.#orchestrator.destroy();
      this.#orchestrator = null;
    }
  }

  slotCount(): number {
    return this.#buffers.size;
  }

  /** Return WASM allocator statistics for diagnostics and testing. */
  allocatorStats(): {
    totalAllocated: number;
    freeListSizes: Map<number, number>;
  } {
    return this.#allocator.getStats();
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    const ptr = this.#allocator.malloc(size);

    if (initialData) {
      if (initialData.byteLength !== size)
        throw new Error("initialData size does not match buffer size");
      new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { ptr, size, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.#allocator.free(buffer.ptr);
      this.#buffers.delete(slot);
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    return this.#readData(slot, start, count);
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    return this.#readData(slot, start, count);
  }

  #readData(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    if (this.capabilities.sharedMemory) {
      // SharedArrayBuffer-backed memory: must copy to a plain ArrayBuffer
      // because callers expect a transferable/detachable buffer.
      return new Uint8Array(buffer.slice(start, start + count));
    } else {
      return buffer.slice(start, start + count);
    }
  }

  copyBufferToBuffer(
    src: Slot,
    srcOffset: number,
    dst: Slot,
    dstOffset: number,
    size: number,
  ): void {
    const srcBuf = this.#getBuffer(src);
    const dstBuf = this.#getBuffer(dst);
    const srcView = new Uint8Array(
      srcBuf.buffer,
      srcBuf.byteOffset + srcOffset,
      size,
    );
    const dstView = new Uint8Array(
      dstBuf.buffer,
      dstBuf.byteOffset + dstOffset,
      size,
    );
    dstView.set(srcView);
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<WasmProgram>> {
    const exe = this.prepareKernelSync(kernel);
    // Pre-register module on workers for parallel dispatch
    const pool = this.#getWorkerPool();
    if (pool && exe.data?.module) {
      await pool.registerModule(exe.data.module);
    }
    return exe;
  }

  prepareKernelSync(kernel: Kernel): Executable<WasmProgram> {
    const kernelHash = FpHash.hash(kernel);
    const captureEnabled = _isCodeCaptureEnabled();
    const module = runWithCache(moduleCache, kernelHash.toString(), () => {
      const { bytes, wat } = codegenWasm(kernel, captureEnabled);
      if (captureEnabled) {
        const re = kernel.outputs[0]?.reduction;
        _emitCodeCapture({
          backend: "wasm",
          kind: "kernel",
          label: kernel.label,
          code: wat,
          metadata: {
            numInputs: kernel.nargs,
            numOutputs: kernel.numOutputs,
            dtype: kernel.outputs[0]?.dtype,
            reduction: re != null,
            simd: false, // set below if applicable
          },
        });
      }
      return new WebAssembly.Module(bytes);
    });
    return new Executable(kernel, { module });
  }

  async prepareRoutine(routine: Routine): Promise<Executable<WasmProgram>> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable<WasmProgram> {
    // WASM routines use size-specialized wasmblr modules dispatched in dispatch().
    // For non-float or unsupported routines, dispatch() falls back to CPU.
    return new Executable(routine, undefined as any);
  }

  dispatch(
    exe: Executable<WasmProgram>,
    inputs: Slot[],
    outputs: Slot[],
    dynamicParams?: number[],
  ): void {
    const tracing = isTracing();
    const start = tracing ? performance.now() : 0;

    if (exe.source instanceof Routine) {
      const routine = exe.source;
      // Determine element size from dtype (f32=4, f64=8)
      const dtype = routine.type.inputDtypes[0];
      const isF32 = dtype === DType.Float32;
      const isF64 = dtype === DType.Float64;
      if (isF32 || isF64) {
        const elementSize: 4 | 8 = isF32 ? 4 : 8;
        switch (routine.name) {
          case Routines.Cholesky:
            this.#dispatchCholesky(routine, inputs, outputs, elementSize);
            break;
          case Routines.TriangularSolve:
            this.#dispatchTriangularSolve(
              routine,
              inputs,
              outputs,
              elementSize,
            );
            break;
          case Routines.LU:
            this.#dispatchLU(routine, inputs, outputs, elementSize);
            break;
          case Routines.Sort:
            this.#dispatchSort(routine, inputs, outputs, elementSize);
            break;
          case Routines.Argsort:
            this.#dispatchArgsort(routine, inputs, outputs, elementSize);
            break;
          // QR: fall through to CPU fallback (WASM routine has correctness issues)
          default:
            // Fall back to CPU for unimplemented routines
            runCpuRoutine(
              routine,
              inputs.map((slot) => this.#getBuffer(slot)),
              outputs.map((slot) => this.#getBuffer(slot)),
            );
            break;
        }
      } else {
        // Fall back to CPU for non-float routines
        runCpuRoutine(
          routine,
          inputs.map((slot) => this.#getBuffer(slot)),
          outputs.map((slot) => this.#getBuffer(slot)),
        );
      }
    } else {
      // Reuse cached instance if available
      let instance = this.#instanceCache.get(exe.data.module);
      if (!instance) {
        instance = new WebAssembly.Instance(exe.data.module, {
          env: { memory: this.#memory },
        });
        this.#instanceCache.set(exe.data.module, instance);
      }
      const ptrs = [...inputs, ...outputs].map(
        (slot) => this.#buffers.get(slot)!.ptr,
      );
      // Kernel signature: (start, end, ...ptrs, [reduceSize]).
      // For symbolic kernels, dynamicParams[0] is the resolved total size.
      // dynamicParams[1] is the resolved reduction size (when kernel has symbolic reduction).
      const totalSize = dynamicParams?.[0] ?? (exe.source.size as number);
      const extraArgs =
        dynamicParams && dynamicParams.length > 1 ? dynamicParams.slice(1) : [];

      // Parallel dispatch: use workers when pool is available, module is registered,
      // and array is large enough for the overhead to pay off.
      const pool = this.#workerPool ?? null; // use pool if already created (don't force-create here)
      if (
        pool &&
        totalSize >= PARALLEL_THRESHOLD &&
        pool.isModuleReady(exe.data.module)
      ) {
        const moduleId = pool.getModuleId(exe.data.module)!;
        pool.dispatchSync(moduleId, instance, totalSize, [
          ...ptrs,
          ...extraArgs,
        ]);
      } else {
        const func = instance.exports.kernel as (...args: number[]) => void;
        func(0, totalSize, ...ptrs, ...extraArgs);
      }
    }

    if (tracing) {
      const info = traceSourceInfo(exe.source);
      emitTrace("wasm", info, start, performance.now());
    }
  }

  /** Get or create a WASM instance for a size-specialized routine module. */
  #getRoutineInstanceForModule(
    module: WebAssembly.Module,
  ): WebAssembly.Instance {
    let instance = this.#instanceCache.get(module);
    if (!instance) {
      instance = new WebAssembly.Instance(module, {
        env: { memory: this.#memory },
      });
      this.#instanceCache.set(module, instance);
    }
    return instance;
  }

  /** Get the size-specialized routine module for a scan routine info. */
  #getRoutineModuleForScan(info: ScanRoutineInfo): WebAssembly.Module {
    const { routine, dtype, sizeParams, unitDiagonal, lower } = info;

    switch (routine) {
      case Routines.Cholesky: {
        const [n] = sizeParams;
        return getCholeskyModule({ n, dtype });
      }
      case Routines.Sort: {
        const [n] = sizeParams;
        return getSortModule({ n, dtype });
      }
      case Routines.Argsort: {
        const [n] = sizeParams;
        return getArgsortModule({ n, dtype });
      }
      case Routines.TriangularSolve: {
        const [n, batchRows] = sizeParams;
        return getTriangularSolveModule({
          n,
          batchRows,
          dtype,
          unitDiagonal: unitDiagonal ?? false,
          lower: lower ?? true,
        });
      }
      case Routines.LU: {
        const [m, n] = sizeParams;
        return getLUModule({ m, n, dtype });
      }
      case Routines.QR: {
        const [m, n] = sizeParams;
        return getQRModule({ m, n, dtype });
      }
      default:
        throw new Error(`Unsupported routine for scan: ${Routines[routine]}`);
    }
  }

  #dispatchCholesky(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
    const dtype = elementSize === 4 ? "f32" : "f64";

    const module = getCholeskyModule({ n, dtype });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.cholesky_batched as (
      i: number,
      o: number,
      b: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      batchSize,
    );
  }

  #dispatchTriangularSolve(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const aShape = routine.type.inputShapes[0];
    const bShape = routine.type.inputShapes[1];
    const n = aShape[aShape.length - 1];
    const batchRows = bShape[bShape.length - 2]; // number of rows in B
    const numBatches = aShape.slice(0, -2).reduce((a, b) => a * b, 1);
    const dtype = elementSize === 4 ? "f32" : "f64";
    const unitDiagonal = routine.params?.unitDiagonal ?? false;
    const lower = routine.params?.lower ?? false;

    const module = getTriangularSolveModule({
      n,
      batchRows,
      dtype,
      unitDiagonal,
      lower,
    });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.triangular_solve_batched as (
      a: number,
      b: number,
      x: number,
      numBatches: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(inputs[1])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      numBatches,
    );
  }

  #dispatchLU(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const m = shape[shape.length - 2];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
    const dtype = elementSize === 4 ? "f32" : "f64";

    const module = getLUModule({ m, n, dtype });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.lu_batched as (
      a: number,
      lu: number,
      piv: number,
      perm: number,
      batchSize: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      this.#buffers.get(outputs[2])!.ptr,
      batchSize,
    );
  }

  #dispatchSort(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const totalSize = n * batchSize * elementSize;
    const dtype = elementSize === 4 ? "f32" : "f64";

    // Copy input to output (sort is in-place)
    const inBuf = this.#buffers.get(inputs[0])!;
    const outBuf = this.#buffers.get(outputs[0])!;
    new Uint8Array(this.#memory.buffer, outBuf.ptr, totalSize).set(
      new Uint8Array(this.#memory.buffer, inBuf.ptr, totalSize),
    );

    // Allocate auxiliary buffer
    const auxPtr = this.#allocator.malloc(n * elementSize);

    // Call in-place sort on output buffer
    const module = getSortModule({ n, dtype });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.sort_batched as (
      data: number,
      aux: number,
      batchSize: number,
    ) => void;
    func(outBuf.ptr, auxPtr, batchSize);

    this.#allocator.free(auxPtr);
  }

  #dispatchArgsort(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const dtype = elementSize === 4 ? "f32" : "f64";

    // Allocate auxiliary buffers (aux uses 4 bytes for indices)
    const auxPtr = this.#allocator.malloc(n * 4);

    const module = getArgsortModule({ n, dtype });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.argsort_batched as (
      data: number,
      out: number,
      idx: number,
      aux: number,
      batchSize: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      auxPtr,
      batchSize,
    );

    this.#allocator.free(auxPtr);
  }

  // eslint-disable-next-line no-unused-private-class-members
  #dispatchQR(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const m = shape[shape.length - 2];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
    const dtype = elementSize === 4 ? "f32" : "f64";

    // Allocate m×n scratch buffer for Householder transformations
    const workPtr = this.#allocator.malloc(m * n * elementSize);

    const module = getQRModule({ m, n, dtype });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.qr_batched as (
      a: number,
      q: number,
      r: number,
      work: number,
      batchSize: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      workPtr,
      batchSize,
    );

    this.#allocator.free(workPtr);
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }

  #getPtr(slot: Slot): number {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.ptr;
  }

  // ---------------------------------------------------------------------------
  // ScatterAdd dispatch (M2: scatter_add primitive)
  // ---------------------------------------------------------------------------

  dispatchScatterAdd(
    output: Slot,
    indices: Slot,
    updates: Slot,
    axis: number,
    targetShape: number[],
    updatesLen: number,
    dtype: DType,
  ): void {
    const ndim = targetShape.length;
    const innerSize =
      ndim > 0 ? targetShape.slice(axis + 1).reduce((a, b) => a * b, 1) : 1;
    const outerSize =
      ndim > 0 ? targetShape.slice(0, axis).reduce((a, b) => a * b, 1) : 1;
    const axisSize = ndim > 0 ? targetShape[axis] : 1;

    let wasmDtype: "f32" | "f64" | "i32";
    if (dtype === DType.Float32) wasmDtype = "f32";
    else if (dtype === DType.Float64) wasmDtype = "f64";
    else if (dtype === DType.Int32) wasmDtype = "i32";
    else throw new Error(`ScatterAdd: unsupported dtype ${dtype}`);

    const module = getScatterAddModule({
      outerSize,
      updatesLen,
      innerSize,
      axisSize,
      dtype: wasmDtype,
    });
    const instance = this.#getRoutineInstanceForModule(module);
    const func = instance.exports.scatter_add as (
      outPtr: number,
      idxPtr: number,
      updPtr: number,
    ) => void;
    func(
      this.#buffers.get(output)!.ptr,
      this.#buffers.get(indices)!.ptr,
      this.#buffers.get(updates)!.ptr,
    );
  }

  /**
   * Prepare a native scan WASM module from the given params.
   * Returns an Executable whose data is a WasmProgram.
   */
  prepareNativeScanGeneral(
    params: NativeScanGeneralParams,
  ): Executable<WasmProgram> {
    const captureEnabled = _isCodeCaptureEnabled();
    const { bytes, wat } = codegenNativeScanGeneral(params, captureEnabled);
    if (captureEnabled) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "scan",
        code: wat,
        metadata: {
          numCarry: params.numCarry,
          numY: params.numY,
          reverse: params.reverse,
          numSteps: params.steps.length,
          byteLength: bytes.byteLength,
        },
      });
    }
    const module = new WebAssembly.Module(bytes);
    return new Executable(null as any, { module });
  }

  /**
   * Dispatch a native scan, executing the compiled WASM loop.
   *
   * Slots layout:
   *   [...consts, ...carryIn, ...xs, ...carryOut, ...ysStacked]
   * where carryOut and ysStacked are preallocated output buffers.
   */
  dispatchNativeScanGeneral(
    exe: Executable<WasmProgram>,
    params: NativeScanGeneralParams,
    length: number,
    constSlots: Slot[],
    carryInSlots: Slot[],
    xsSlots: Slot[],
    carryOutSlots: Slot[],
    ysStackedSlots: Slot[],
  ): void {
    const { internalSizes, auxBufferSize } = params;

    // Allocate internal scratch buffers
    const internalPtrs: number[] = [];
    for (const size of internalSizes) {
      internalPtrs.push(this.#allocator.malloc(size));
    }

    // Allocate aux buffer if needed (for sort/argsort)
    let auxPtr = 0;
    if (auxBufferSize && auxBufferSize > 0) {
      auxPtr = this.#allocator.malloc(auxBufferSize);
    }

    // Copy carryIn to carryOut (initial carry values)
    const { carrySizes } = params;
    for (let c = 0; c < params.numCarry; c++) {
      const srcBuf = this.#getBuffer(carryInSlots[c]);
      const dstBuf = this.#getBuffer(carryOutSlots[c]);
      dstBuf.set(srcBuf.subarray(0, carrySizes[c]));
    }

    // Build args: [length, consts, carryOut, xs, carryOut, ysStacked, internals, aux?]
    // The scan function arguments are:
    //   [length, ...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
    //    ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal), aux?]
    // BUT carryIn is also carryOut in this layout (the codegen copies carryIn to carryOut
    // at step 1, so we pass carryOut pointers for both carryIn and carryOut positions).
    const args: number[] = [length];

    // consts
    for (const slot of constSlots) args.push(this.#getPtr(slot));
    // carryIn (we pass carryOut ptrs here — codegen step 1 copies carryIn→carryOut first,
    // but we already copied above, so carryOut IS the working carry buffer)
    for (const slot of carryOutSlots) args.push(this.#getPtr(slot));
    // xs
    for (const slot of xsSlots) args.push(this.#getPtr(slot));
    // carryOut (same ptrs as carryIn above — the scan reads/writes to carryOut)
    for (const slot of carryOutSlots) args.push(this.#getPtr(slot));
    // ysStacked
    for (const slot of ysStackedSlots) args.push(this.#getPtr(slot));
    // internals
    args.push(...internalPtrs);
    // aux
    if (auxBufferSize && auxBufferSize > 0) args.push(auxPtr);

    // Instantiate and run (reuse cached instance)
    let instance = this.#instanceCache.get(exe.data.module);
    if (!instance) {
      const imports: WebAssembly.Imports = { env: { memory: this.#memory } };

      // Add routine function imports if needed (using size-specialized modules)
      if (params.routineInfos && params.routineInfos.length > 0) {
        const routineImports: Record<string, WebAssembly.ExportValue> = {};
        for (const info of params.routineInfos) {
          const routineModule = this.#getRoutineModuleForScan(info);
          const routineInstance =
            this.#getRoutineInstanceForModule(routineModule);
          routineImports[info.exportName] = routineInstance.exports[
            info.exportName
          ] as WebAssembly.ExportValue;
        }
        imports.routines = routineImports;
      }

      instance = new WebAssembly.Instance(exe.data.module, imports);
      this.#instanceCache.set(exe.data.module, instance);
    }
    const scanFunc = instance.exports.scan as (...args: number[]) => void;
    scanFunc(...args);

    // Free scratch buffers
    for (const ptr of internalPtrs) this.#allocator.free(ptr);
    if (auxPtr) this.#allocator.free(auxPtr);
  }

  // -------------------------------------------------------------------------
  // Blocked associative scan (Phase 5)
  // -------------------------------------------------------------------------

  prepareBlockedAssociativeScan(
    params: NativeAssocScanBlockedParams,
  ): Executable<WasmProgram> {
    const captureEnabled = _isCodeCaptureEnabled();
    const { bytes, wat } = codegenBlockedAssociativeScan(
      params,
      captureEnabled,
    );
    if (captureEnabled) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "assoc-scan",
        code: wat,
        metadata: {
          numLeaves: params.numLeaves,
          blockSize: params.blockSize,
          reverse: params.reverse,
          numSteps: params.steps.length,
          byteLength: bytes.byteLength,
        },
      });
    }
    const module = new WebAssembly.Module(bytes);
    return new Executable(null as any, { module });
  }

  dispatchBlockedAssociativeScan(
    exe: Executable<WasmProgram>,
    params: NativeAssocScanBlockedParams,
    N: number,
    constSlots: Slot[],
    inputLeafSlots: Slot[],
    outputLeafSlots: Slot[],
  ): void {
    if (N === 0) return;

    const { leafElemSizes, internalSizes, blockSize: B } = params;
    const totalLeafElemSize = leafElemSizes.reduce((a, b) => a + b, 0);
    const M = Math.ceil(N / B);
    const pingPongSize = totalLeafElemSize * N;
    const summarySize = totalLeafElemSize * M;

    const pingPtr = this.#allocator.malloc(pingPongSize);
    const pongPtr = this.#allocator.malloc(pingPongSize);
    const sPingPtr = this.#allocator.malloc(summarySize);
    const sPongPtr = this.#allocator.malloc(summarySize);

    const internalPtrs: number[] = [];
    for (const size of internalSizes) {
      internalPtrs.push(this.#allocator.malloc(size));
    }

    let instance = this.#instanceCache.get(exe.data.module);
    if (!instance) {
      const imports: WebAssembly.Imports = { env: { memory: this.#memory } };
      instance = new WebAssembly.Instance(exe.data.module, imports);
      this.#instanceCache.set(exe.data.module, instance);
    }

    const args: number[] = [N];
    for (const slot of constSlots) args.push(this.#getPtr(slot));
    for (const slot of inputLeafSlots) args.push(this.#getPtr(slot));
    for (const slot of outputLeafSlots) args.push(this.#getPtr(slot));
    args.push(pingPtr, pongPtr, sPingPtr, sPongPtr);
    args.push(...internalPtrs);

    const scanFunc = instance.exports.blocked_assoc_scan as (
      ...args: number[]
    ) => void;
    scanFunc(...args);

    for (const ptr of internalPtrs) this.#allocator.free(ptr);
    this.#allocator.free(sPongPtr);
    this.#allocator.free(sPingPtr);
    this.#allocator.free(pongPtr);
    this.#allocator.free(pingPtr);
  }

  // ---------------------------------------------------------------------------
  // Mega-Module dispatch (M6.1)
  // ---------------------------------------------------------------------------

  /**
   * Execute a compiled mega-module with the given input Slots.
   *
   * The mega-module function allocates its own intermediates (via imported
   * env.alloc/env.free) and writes output pointers to a result buffer.
   * This method reads those pointers and creates proper backend Slots.
   *
   * For pass-through outputs (outputSizes[i] === 0), the output pointer
   * matches an input pointer — the corresponding input Slot is incRef'd.
   */
  executeMegaModule(megaModule: WasmMegaModule, inputSlots: Slot[]): Slot[] {
    // Map input Slots → raw pointers
    const inputPtrs: number[] = inputSlots.map((s) => this.#getPtr(s));

    // Allocate result buffer (numOutputs * 4 bytes for i32 pointers)
    const resultBufSize = megaModule.numOutputs * 4;
    const resultBufPtr = this.#allocator.malloc(resultBufSize);

    const orch = this.#getOrchestrator();
    if (orch && inputPtrs.length <= 64) {
      // --- Orchestrator path (M6.2b): off-main-thread execution ---
      // Register the module if not already known to the orchestrator.
      const moduleId = orch.registerModuleSync(megaModule.module);
      // Dispatch to orchestrator. The main thread spin-waits here,
      // servicing alloc/free proxy requests via the control buffer.
      orch.dispatch(
        moduleId,
        inputPtrs,
        resultBufPtr,
        (size: number) => this.#allocator.malloc(size),
        (ptr: number) => this.#allocator.free(ptr),
      );
    } else {
      // --- Direct execution path (fallback / no orchestrator) ---
      // Get or create cached instance (with alloc/free imports)
      let instance = this.#instanceCache.get(megaModule.module);
      if (!instance) {
        instance = new WebAssembly.Instance(megaModule.module, {
          env: {
            memory: this.#memory,
            alloc: (size: number) => this.#allocator.malloc(size),
            free: (ptr: number) => this.#allocator.free(ptr),
          },
        });
        this.#instanceCache.set(megaModule.module, instance);
      }

      // Call mega_execute(input0_ptr, ..., inputN_ptr, resultBufPtr)
      const func = instance.exports.mega_execute as (...args: number[]) => void;
      func(...inputPtrs, resultBufPtr);
    }

    // Read output pointers from result buffer and create Slots.
    // All outputs are new allocations (pass-through outputs are rejected
    // at compile time by compileToMegaModule).
    const view = new DataView(this.#memory.buffer, resultBufPtr, resultBufSize);
    const outputSlots: Slot[] = [];

    for (let i = 0; i < megaModule.numOutputs; i++) {
      const ptr = view.getInt32(i * 4, true); // little-endian
      const slot = this.#nextSlot++;
      this.#buffers.set(slot, {
        ptr,
        size: megaModule.outputSizes[i],
        ref: 1,
      });
      outputSlots.push(slot);
    }

    // Free the result buffer
    this.#allocator.free(resultBufPtr);

    return outputSlots;
  }

  // ---------------------------------------------------------------------------
  // Parallel mega-module dispatch (M6.2c)
  // ---------------------------------------------------------------------------

  /**
   * Register a mega-module on the worker pool for parallel dispatch (M6.2c).
   * Workers get stub alloc/free imports (they only call kernel_N functions).
   * Must be awaited before calling executeMegaModuleParallelSync.
   */
  async registerMegaModuleOnPool(megaModule: WasmMegaModule): Promise<void> {
    const pool = this.#getWorkerPool();
    if (!pool) throw new Error("Worker pool not available");
    if (!pool.isModuleReady(megaModule.module)) {
      await pool.registerMegaModule(megaModule.module);
    }
  }

  /**
   * Execute a mega-module with JS-driven step execution, dispatching large
   * kernels in parallel via the WasmWorkerPool (M6.2c).
   *
   * Instead of calling `mega_execute` (which runs all steps sequentially in
   * WASM), this method walks the step metadata on the main thread and calls
   * individual extracted kernel functions. Kernels with size >= PARALLEL_THRESHOLD
   * are dispatched across workers; smaller kernels run on the main thread only.
   *
   * Requires: the mega-module has been registered on the worker pool via
   * `registerMegaModuleOnPool()`.
   */
  executeMegaModuleParallelSync(
    megaModule: WasmMegaModule,
    inputSlots: Slot[],
  ): Slot[] {
    const pool = this.#workerPool;
    if (!pool)
      throw new Error("Worker pool not available for parallel dispatch");

    const moduleId = pool.getModuleId(megaModule.module)!;

    // Get or create a main-thread instance (with real alloc/free imports)
    let instance = this.#instanceCache.get(megaModule.module);
    if (!instance) {
      instance = new WebAssembly.Instance(megaModule.module, {
        env: {
          memory: this.#memory,
          alloc: (size: number) => this.#allocator.malloc(size),
          free: (ptr: number) => this.#allocator.free(ptr),
        },
      });
      this.#instanceCache.set(megaModule.module, instance);
    }

    // Initialize locals array: map input JitId indices → pointers
    const locals = new Array<number>(megaModule.numLocals).fill(0);
    for (let i = 0; i < inputSlots.length; i++) {
      locals[i] = this.#getPtr(inputSlots[i]);
    }

    // Walk step metadata, executing each step
    for (const step of megaModule.stepInfos) {
      switch (step.type) {
        case "malloc": {
          const ptr = this.#allocator.malloc(step.size);
          locals[step.outputIdx] = ptr;
          if (step.initialData) {
            new Uint8Array(
              this.#memory.buffer,
              ptr,
              step.initialData.byteLength,
            ).set(step.initialData);
          }
          break;
        }

        case "free":
          this.#allocator.free(locals[step.inputIdx]);
          locals[step.inputIdx] = 0;
          break;

        case "recycle":
          locals[step.toIdx] = locals[step.fromIdx];
          locals[step.fromIdx] = 0;
          break;

        case "kernel": {
          const args = [...step.inputIdxs, ...step.outputIdxs].map(
            (idx) => locals[idx],
          );

          if (step.kernelSize >= PARALLEL_THRESHOLD) {
            // Parallel dispatch across workers
            pool.dispatchSync(
              moduleId,
              instance,
              step.kernelSize,
              args,
              step.kernelIdx,
            );
          } else {
            // Serial dispatch on main thread only
            const fn = instance.exports[`kernel_${step.kernelIdx}`] as (
              ...a: number[]
            ) => void;
            fn(0, step.kernelSize, ...args);
          }
          break;
        }
      }
    }

    // Read output pointers from locals and create Slots
    const outputSlots: Slot[] = [];
    for (let i = 0; i < megaModule.numOutputs; i++) {
      const ptr = locals[megaModule.outputLocalIdxs[i]];
      const slot = this.#nextSlot++;
      this.#buffers.set(slot, {
        ptr,
        size: megaModule.outputSizes[i],
        ref: 1,
      });
      outputSlots.push(slot);
    }

    return outputSlots;
  }

  /**
   * Check whether a mega-module should use the parallel execution path.
   * Returns true if SharedArrayBuffer is available AND at least one kernel
   * has size >= PARALLEL_THRESHOLD.
   *
   * Pure check — does NOT lazily create the worker pool. Pool creation
   * is deferred to {@link registerMegaModuleOnPool}.
   */
  shouldUseParallelMegaModule(megaModule: WasmMegaModule): boolean {
    if (
      !megaModule.canParallelize ||
      !this.capabilities.sharedMemory ||
      typeof Worker === "undefined" ||
      !this.#canSpinWaitWorkers
    ) {
      return false;
    }
    return megaModule.stepInfos.some(
      (s) => s.type === "kernel" && s.kernelSize >= PARALLEL_THRESHOLD,
    );
  }

  // -------------------------------------------------------------------------
  // Compiled block-map loop
  // -------------------------------------------------------------------------

  /**
   * Compile a block-map loop into a single WASM module.
   * Returns an Executable whose data is a WasmProgram, or null if the body
   * cannot be compiled (e.g. contains routines or unsupported steps).
   */
  prepareBlockMapWasm(
    params: BlockMapWasmParams,
  ): Executable<WasmProgram> | null {
    try {
      const captureEnabled = _isCodeCaptureEnabled();
      const { bytes, wat } = codegenBlockMapLoop(params, captureEnabled);
      if (captureEnabled) {
        _emitCodeCapture({
          backend: "wasm",
          kind: "block-map",
          code: wat,
          metadata: {
            gridShape: params.gridShape,
            blockShape: params.blockShape,
            numInputs: params.numInputs,
            numOutputs: params.numOutputs,
            numSteps: params.steps.length,
            byteLength: bytes.byteLength,
          },
        });
      }
      const module = new WebAssembly.Module(bytes);
      return new Executable(null as any, { module });
    } catch (e) {
      if (DEBUG >= 2) console.warn("[wasm-block-map] codegen failed:", e);
      return null;
    }
  }

  /**
   * Dispatch a compiled block-map WASM module.
   *
   * Allocates internal scratch buffers and block-input/output scratch buffers,
   * runs the compiled loop, then frees scratch.
   */
  dispatchBlockMapWasm(
    exe: Executable<WasmProgram>,
    params: BlockMapWasmParams,
    constSlots: Slot[],
    inputSlots: Slot[],
    outputSlots: Slot[],
  ): void {
    const { blockInputSizes, blockOutputSizes, internalSizes } = params;

    // Allocate scratch buffers for block inputs (body reads from these)
    const scratchInputPtrs: number[] = [];
    for (const size of blockInputSizes) {
      scratchInputPtrs.push(this.#allocator.malloc(size));
    }

    // Allocate scratch buffers for block outputs (body writes to these)
    const scratchOutputPtrs: number[] = [];
    for (const size of blockOutputSizes) {
      scratchOutputPtrs.push(this.#allocator.malloc(size));
    }

    // Allocate internal scratch buffers (kernel intermediates)
    const internalPtrs: number[] = [];
    for (const size of internalSizes) {
      internalPtrs.push(this.#allocator.malloc(size));
    }

    // Build args: [consts, inputs, outputs, scratchInputs, scratchOutputs, internals]
    const args: number[] = [];
    for (const slot of constSlots) args.push(this.#getPtr(slot));
    for (const slot of inputSlots) args.push(this.#getPtr(slot));
    for (const slot of outputSlots) args.push(this.#getPtr(slot));
    args.push(...scratchInputPtrs);
    args.push(...scratchOutputPtrs);
    args.push(...internalPtrs);

    // Instantiate and run
    let instance = this.#instanceCache.get(exe.data.module);
    if (!instance) {
      const imports: WebAssembly.Imports = { env: { memory: this.#memory } };
      instance = new WebAssembly.Instance(exe.data.module, imports);
      this.#instanceCache.set(exe.data.module, instance);
    }
    const blockMapFunc = instance.exports.block_map_loop as (
      ...args: number[]
    ) => void;
    blockMapFunc(...args);

    // Free scratch
    for (const ptr of scratchInputPtrs) this.#allocator.free(ptr);
    for (const ptr of scratchOutputPtrs) this.#allocator.free(ptr);
    for (const ptr of internalPtrs) this.#allocator.free(ptr);
  }
}

// ---------------------------------------------------------------------------
// Shared WASM helper function imports
// ---------------------------------------------------------------------------

/**
 * Import WASM helper functions (sin, cos, exp, etc.) needed by a set of AluOps.
 * Shared by regular kernel codegen and scan codegen.
 */
export function importWasmHelperFuncs(
  cg: CodeGenerator,
  ops: Set<AluOp> | Map<AluOp, Set<DType>>,
): Record<string, number> {
  const funcs: Record<string, number> = {};
  const hasOp = (op: AluOp) => (ops instanceof Map ? ops.has(op) : ops.has(op));
  if (hasOp(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (hasOp(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (hasOp(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (hasOp(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (hasOp(AluOp.Exp) || hasOp(AluOp.Erf) || hasOp(AluOp.Erfc))
    funcs.exp = wasm_exp(cg);
  if (hasOp(AluOp.Log)) funcs.log = wasm_log(cg);
  if (hasOp(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (hasOp(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (hasOp(AluOp.Threefry2x32)) funcs.threefry2x32 = wasm_threefry2x32(cg);
  return funcs;
}

/**
 * Number of range parameters prepended to the kernel function signature.
 * All kernels use (start: i32, end: i32, ...ptrs) -> () so that parallel
 * dispatch (M5.3) can pass a sub-range to each worker. Single-threaded
 * dispatch calls kernel(0, totalSize, ...ptrs).
 */
const RANGE_PARAMS = 2;

// ---------------------------------------------------------------------------
// WASM SIMD vectorization (Improvement 5)
// ---------------------------------------------------------------------------

/** Ops that have direct f32x4 / i32x4 SIMD equivalents. */
const SIMD_OK_OPS = new Set([
  AluOp.Add,
  AluOp.Sub,
  AluOp.Mul,
  AluOp.Min,
  AluOp.Max,
  AluOp.Sqrt,
  AluOp.Floor,
  AluOp.Ceil,
  AluOp.Reciprocal,
  AluOp.Const,
  AluOp.Special,
]);

/**
 * Check if an expression can be vectorized using WASM SIMD (f32x4).
 * Returns true only for f32 elementwise expressions with contiguous access.
 */
export function canVectorizeSimd(exp: AluExp, dtype: DType): boolean {
  if (dtype !== DType.Float32) return false;
  return !exp.some((e) => {
    if (SIMD_OK_OPS.has(e.op)) return false; // OK → don't stop
    if (e.op === AluOp.GlobalIndex) {
      // After tuneNullopt, contiguous arrays have indexExp = Special(gidx)
      // (the simplifier eliminates redundant Mod when gidx range < dim).
      // Broadcast arrays (stride=0) have indexExp = Const(0).
      return e.src[0].op !== AluOp.Special && e.src[0].op !== AluOp.Const;
    }
    return true; // Unknown op → not vectorizable
  });
}

/**
 * Translate an AluExp to WASM SIMD instructions (f32x4, 4-wide).
 * Handles only the SIMD-compatible subset of AluOps. The caller must
 * verify eligibility with `canVectorizeSimd` first.
 *
 * @param getInputLocal - Returns the WASM local index holding the base pointer
 *   for input buffer `gid`. In codegenWasm: `gid + RANGE_PARAMS`.
 *   In mega-module: `inputLocals[gid]`.
 *
 * Leaves a v128 result on the WASM stack.
 */
export function translateExpCoreSimd(
  cg: CodeGenerator,
  exp: AluExp,
  gidxLocal: number,
  getInputLocal: (gid: number) => number,
): void {
  // CSE: count references for local.tee caching (same as scalar path)
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (e: AluExp) => {
    references.set(e, (references.get(e) ?? 0) + 1);
    if (!seen.has(e)) {
      seen.add(e);
      for (const src of e.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (e: AluExp): void => {
    if (expContext.has(e)) {
      cg.local.get(expContext.get(e)!);
      return;
    }
    const { op, src, arg } = e;

    if (op === AluOp.Add) {
      gen(src[0]);
      gen(src[1]);
      cg.f32x4.add();
    } else if (op === AluOp.Sub) {
      gen(src[0]);
      gen(src[1]);
      cg.f32x4.sub();
    } else if (op === AluOp.Mul) {
      gen(src[0]);
      gen(src[1]);
      cg.f32x4.mul();
    } else if (op === AluOp.Min) {
      gen(src[0]);
      gen(src[1]);
      cg.f32x4.min();
    } else if (op === AluOp.Max) {
      gen(src[0]);
      gen(src[1]);
      cg.f32x4.max();
    } else if (op === AluOp.Sqrt) {
      gen(src[0]);
      cg.f32x4.sqrt();
    } else if (op === AluOp.Floor) {
      gen(src[0]);
      cg.f32x4.floor();
    } else if (op === AluOp.Ceil) {
      gen(src[0]);
      cg.f32x4.ceil();
    } else if (op === AluOp.Reciprocal) {
      cg.f32.const(1);
      cg.f32x4.splat();
      gen(src[0]);
      cg.f32x4.div();
    } else if (op === AluOp.Const) {
      cg.f32.const(arg as number);
      cg.f32x4.splat();
    } else if (op === AluOp.Special) {
      // gidx → not directly needed as v128; handled via GlobalIndex loads
      // This case shouldn't be reached as a top-level node in practice
      cg.local.get(gidxLocal);
      cg.f32.convert_i32_s();
      cg.f32x4.splat();
    } else if (op === AluOp.GlobalIndex) {
      const [gid, _len] = arg as [number, number];
      if (src[0].op === AluOp.Const) {
        // Broadcast: load one f32 at constant offset, splat to all 4 lanes
        const constIdx = src[0].arg as number;
        cg.local.get(getInputLocal(gid));
        cg.f32.load(2, constIdx * 4);
        cg.f32x4.splat();
      } else {
        // Contiguous v128.load: base + gidx * 4
        cg.local.get(getInputLocal(gid));
        cg.local.get(gidxLocal);
        cg.i32.const(4); // byteWidth(Float32)
        cg.i32.mul();
        cg.i32.add();
        cg.v128.load(2, 0); // align hint = 4 bytes
      }
    } else {
      throw new Error(`SIMD: unsupported op ${op}`);
    }

    // CSE: cache in v128 local if referenced multiple times
    if ((references.get(e) ?? 0) > 1) {
      const local = cg.local.declare(cg.v128);
      cg.local.tee(local);
      expContext.set(e, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

function codegenWasm(
  kernel: Kernel,
  traceEnabled: boolean,
): { bytes: Uint8Array<ArrayBuffer>; wat?: string } {
  if (kernel.isMultiOutput) {
    return codegenWasmMulti(kernel, traceEnabled);
  }

  const tune = tuneNullopt(kernel);

  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.outputs[0].exp}\ntune.exp: ${tune.exp}`);
  }

  // Check SIMD eligibility: f32 elementwise, no reduction, contiguous access
  const re = kernel.outputs[0].reduction;
  const useSimd = !re && canVectorizeSimd(tune.exp, kernel.outputs[0].dtype);

  if (DEBUG >= 2 && useSimd) {
    console.info(`wasm: SIMD f32x4 path for kernel (nargs=${kernel.nargs})`);
  }

  const cg = new CodeGenerator();
  cg.trace = traceEnabled;
  configureMemoryImport(cg);

  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  const funcs = importWasmHelperFuncs(cg, distinctOps);

  // Signature: (start, end, ptr0, ..., ptrOut, [reduceSize]) -> ()
  // start and end are the element range for parallel dispatch (M5.3).
  // Single-threaded: kernel(0, totalSize, ...ptrs, [reduceSize])
  // When the kernel has a symbolic reduction, an extra i32 param holds the
  // resolved reduction size (passed at dispatch time via dynamicParams[1]).
  const hasSymbolicReduction = re != null && isSymbolicSize(re.size);
  const nParams =
    RANGE_PARAMS + kernel.nargs + 1 + (hasSymbolicReduction ? 1 : 0);
  // Index of the reduction size parameter (valid only when hasSymbolicReduction)
  const reduceSizeParamIdx = RANGE_PARAMS + kernel.nargs + 1;

  const kernelFunc = cg.function(rep(nParams, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);

    if (useSimd) {
      // ----- SIMD path: main f32x4 loop + scalar tail -----
      const outParam = RANGE_PARAMS + kernel.nargs;

      // SIMD main loop: process 4 f32 elements per iteration
      cg.local.get(0); // start
      cg.local.set(gidx);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        // Break if gidx + 4 > end  (i.e. fewer than 4 elements left)
        cg.local.get(gidx);
        cg.i32.const(4);
        cg.i32.add();
        cg.local.get(1); // end
        cg.i32.gt_u();
        cg.br_if(0);

        // Push output address: outBase + gidx * 4
        cg.local.get(outParam);
        cg.local.get(gidx);
        cg.i32.const(4); // byteWidth(Float32)
        cg.i32.mul();
        cg.i32.add();

        // Emit SIMD expression → v128 on stack
        translateExpCoreSimd(cg, tune.exp, gidx, (gid) => gid + RANGE_PARAMS);

        // v128.store (pops value, then address)
        cg.v128.store(2, 0);

        // gidx += 4
        cg.local.get(gidx);
        cg.i32.const(4);
        cg.i32.add();
        cg.local.set(gidx);

        cg.br(1);
        cg.end();
      }
      cg.end();

      // Scalar tail: process remaining elements one at a time
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(gidx);
        cg.local.get(1); // end
        cg.i32.ge_u();
        cg.br_if(0);

        // Push output address
        cg.local.get(outParam);
        cg.local.get(gidx);
        cg.i32.const(4);
        cg.i32.mul();
        cg.i32.add();

        // Scalar expression
        translateExp(cg, funcs, tune.exp, { gidx }, RANGE_PARAMS);

        // f32.store
        cg.f32.store(2);

        // gidx++
        cg.local.get(gidx);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(gidx);

        cg.br(1);
        cg.end();
      }
      cg.end();
    } else {
      // ----- Scalar path (existing) -----
      emitKernelBody({
        cg,
        funcs,
        kernel,
        gidx,
        startLocal: 0,
        endLocal: 1,
        reduceSizeLocal: hasSymbolicReduction ? reduceSizeParamIdx : undefined,
        emitOutputAddr: () => {
          cg.local.get(RANGE_PARAMS + kernel.nargs);
          cg.local.get(gidx);
          cg.i32.const(byteWidth(kernel.outputs[0].dtype));
          cg.i32.mul();
          cg.i32.add();
        },
        emitExp: (exp, { ridx, acc }) => {
          const vars: Record<string, number> = { gidx };
          if (ridx !== undefined) vars.ridx = ridx;
          if (acc !== undefined) vars.acc = acc;
          translateExp(cg, funcs, exp, vars, RANGE_PARAMS);
        },
      });
    }
  });
  cg.export(kernelFunc, "kernel");

  const bytes = cg.finish();
  return { bytes, wat: traceEnabled ? cg.toWat() : undefined };
}

/**
 * Codegen for multi-output kernels. Generates a single gidx loop that
 * evaluates and stores each output expression sequentially per element.
 * Function signature: (input0, ..., inputN-1, output0, ..., outputM-1).
 */
function codegenWasmMulti(
  kernel: Kernel,
  traceEnabled: boolean,
): { bytes: Uint8Array<ArrayBuffer>; wat?: string } {
  const numOutputs = kernel.numOutputs;
  const tunes = kernel.outputs.map((o) => {
    // Build a temporary single-output kernel for tuning
    const tmpKernel = Kernel.single(
      kernel.nargs,
      kernel.size,
      o.exp,
      o.reduction,
    );
    return tuneNullopt(tmpKernel);
  });

  const cg = new CodeGenerator();
  cg.trace = traceEnabled;
  configureMemoryImport(cg);

  // Collect all distinct ops across all output expressions
  let allOps: Map<AluOp, Set<DType>> = new Map();
  for (const tune of tunes) {
    allOps = mapSetUnion(allOps, tune.exp.distinctOps());
    if (tune.epilogue)
      allOps = mapSetUnion(allOps, tune.epilogue.distinctOps());
  }
  const funcs = importWasmHelperFuncs(cg, allOps);

  // Function params: start, end, nargs inputs + numOutputs outputs, all i32
  // When any output has a symbolic reduction, an extra i32 param holds the
  // resolved reduction size (passed at dispatch time via dynamicParams[1]).
  // Signature: (start, end, ptr0, ..., ptrN-1, out0, ..., outM-1, [reduceSize])
  const hasSymbolicReduction = kernel.outputs.some(
    (o) => o.reduction != null && isSymbolicSize(o.reduction.size),
  );
  const nParams =
    RANGE_PARAMS + kernel.nargs + numOutputs + (hasSymbolicReduction ? 1 : 0);
  const reduceSizeParamIdx = RANGE_PARAMS + kernel.nargs + numOutputs;

  const kernelFunc = cg.function(rep(nParams, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);

    // gidx loop: start from param 0 (start), loop until param 1 (end)
    cg.local.get(0); // start
    cg.local.set(gidx);
    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(gidx);
      cg.local.get(1); // end
      cg.i32.ge_u();
      cg.br_if(0);

      // For each output: compute expression, store result
      for (let oi = 0; oi < numOutputs; oi++) {
        const tune = tunes[oi];
        const out = kernel.outputs[oi];
        const outParamIdx = RANGE_PARAMS + kernel.nargs + oi;
        const storeAlign = Math.log2(byteWidth(out.dtype));

        // Push output address: outParam + gidx * elemSize
        cg.local.get(outParamIdx);
        cg.local.get(gidx);
        cg.i32.const(byteWidth(out.dtype));
        cg.i32.mul();
        cg.i32.add();

        if (out.reduction) {
          // Reduction: accumulator + inner ridx loop
          const re = out.reduction;
          const acc = cg.local.declare(dty(cg, null, out.exp.dtype));
          dty(cg, null, out.exp.dtype).const(re.identity);
          cg.local.set(acc);

          const useKahan = re.dtype === DType.Float64 && re.op === AluOp.Add;
          let kahanComp: number | undefined;
          if (useKahan) {
            kahanComp = cg.local.declare(cg.f64);
            cg.f64.const(0);
            cg.local.set(kahanComp);
          }

          const ridx = cg.local.declare(cg.i32);
          cg.i32.const(0);
          cg.local.set(ridx);

          cg.loop(cg.void);
          {
            cg.block(cg.void);
            cg.local.get(ridx);
            if (hasSymbolicReduction && isSymbolicSize(re.size)) {
              cg.local.get(reduceSizeParamIdx);
            } else {
              cg.i32.const(re.size as number);
            }
            cg.i32.ge_u();
            cg.br_if(0);

            const vars: Record<string, number> = { gidx, ridx };
            translateExp(cg, funcs, tune.exp, vars, RANGE_PARAMS);
            codegenReductionAccumulate(cg, re, acc, kahanComp);

            cg.local.get(ridx);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(ridx);

            cg.br(1);
            cg.end();
          }
          cg.end();

          // Emit epilogue
          const epilogueVars: Record<string, number> = { gidx, acc };
          translateExp(cg, funcs, tune.epilogue!, epilogueVars, RANGE_PARAMS);
        } else {
          // No reduction: just translate the expression
          const vars: Record<string, number> = { gidx };
          translateExp(cg, funcs, tune.exp, vars, RANGE_PARAMS);
        }

        // Store result
        dty(cg, null, out.dtype).store(storeAlign);
      }

      // gidx++
      cg.local.get(gidx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(gidx);

      cg.br(1);
      cg.end();
    }
    cg.end();
  });
  cg.export(kernelFunc, "kernel");

  return { bytes: cg.finish(), wat: traceEnabled ? cg.toWat() : undefined };
}

// ---------------------------------------------------------------------------
// Unified AluExp translation
// ---------------------------------------------------------------------------

/**
 * Context for translating AluExp to WASM.
 * The handleGlobalIndex callback is called to emit code that loads a value
 * from a buffer. After it returns, the value should be on the WASM stack.
 */
export interface TranslateExpContext {
  /** Get the value of a variable (e.g., "gidx", "ridx", "acc") */
  getVariable: (name: string) => number | undefined;
  /** Emit code to handle GlobalIndex. Should leave the loaded value on stack. */
  handleGlobalIndex: (
    cg: CodeGenerator,
    gen: (e: AluExp) => void,
    gid: number,
    len: number,
    indexExp: AluExp,
    dtype: DType,
  ) => void;
}

/**
 * Translate an AluExp tree to WASM code.
 *
 * This is the core expression translation shared by regular kernels and scan.
 * The context provides callbacks for variable resolution and GlobalIndex handling.
 */
export function translateExpCore(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: TranslateExpContext,
): void {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (e: AluExp) => {
    references.set(e, (references.get(e) ?? 0) + 1);
    if (!seen.has(e)) {
      seen.add(e);
      for (const src of e.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (e: AluExp): void => {
    if (expContext.has(e)) {
      cg.local.get(expContext.get(e)!);
      return;
    }
    const { op, src, dtype, arg } = e;

    // Some of these cases early `return` to force-inline them (no local.set).
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      gen(src[0]);
      gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) cg.i32.or();
        else dty(cg, op, dtype).add();
      } else if (op === AluOp.Sub) {
        dty(cg, op, dtype).sub();
      } else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) cg.i32.and();
        else dty(cg, op, dtype).mul();
      } else if (op === AluOp.Idiv) {
        if (isFloatDtype(dtype)) {
          dtyF(cg, op, dtype).div();
          dtyF(cg, op, dtype).trunc();
        } else if (dtype === DType.Uint32) cg.i32.div_u();
        else if (dtype === DType.Int32) cg.i32.div_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Mod) {
        if (isFloatDtype(dtype)) {
          // Emulate a % b = a - trunc(a/b)*b
          const dt = dtyF(cg, op, dtype);
          const a = cg.local.declare(dt);
          const b = cg.local.declare(dt);
          cg.local.set(b);
          cg.local.tee(a); // stack: a
          cg.local.get(a);
          cg.local.get(b);
          dt.div();
          dt.trunc(); // stack: a, trunc(a/b)
          cg.local.get(b);
          dt.mul(); // stack: a, trunc(a/b)*b
          dt.sub();
        } else if (dtype === DType.Uint32) cg.i32.rem_u();
        else if (dtype === DType.Int32) cg.i32.rem_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Min || op === AluOp.Max) {
        if (isFloatDtype(dtype)) {
          if (op === AluOp.Min) dtyF(cg, op, dtype).min();
          else dtyF(cg, op, dtype).max();
        } else if (
          dtype === DType.Int32 ||
          dtype === DType.Uint32 ||
          dtype === DType.Bool
        ) {
          // Wasm has no i32.min, so emulate with select.
          const a = cg.local.declare(cg.i32);
          const b = cg.local.declare(cg.i32);
          cg.local.set(b);
          cg.local.tee(a);
          cg.local.get(b);
          cg.local.get(a);
          cg.local.get(b);
          if (dtype === DType.Int32) {
            if (op === AluOp.Min) cg.i32.lt_s();
            else cg.i32.gt_s();
          } else {
            if (op === AluOp.Min) cg.i32.lt_u();
            else cg.i32.gt_u();
          }
          cg.select();
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.BitCombine) {
        if (arg === "and") cg.i32.and();
        else if (arg === "or") cg.i32.or();
        else cg.i32.xor();
      } else if (op === AluOp.BitShift) {
        if (arg === "shl") cg.i32.shl();
        else cg.i32.shr_u();
      } else if (op === AluOp.Cmplt) {
        const srcDtype = src[0].dtype;
        if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
        else if (srcDtype === DType.Int32) cg.i32.lt_s();
        else if (srcDtype === DType.Uint32) cg.i32.lt_u();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Cmpne) dty(cg, op, src[0].dtype).ne();
      else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      // TODO: Our intrinsics are only implemented in f32 precision currently,
      // so we cast to f32 first for other floating-point inputs.
      const callFuncF32 = (func: number): void => {
        if (dtype !== DType.Float32) {
          if (dtype === DType.Float64) cg.f32.demote_f64();
          else throw new UnsupportedOpError(op, dtype, "wasm");
        }
        cg.call(func);
        if (dtype === DType.Float64) cg.f64.promote_f32();
      };
      if (op === AluOp.Sin) (gen(src[0]), callFuncF32(funcs.sin));
      else if (op === AluOp.Cos) (gen(src[0]), callFuncF32(funcs.cos));
      else if (op === AluOp.Asin) (gen(src[0]), callFuncF32(funcs.asin));
      else if (op === AluOp.Atan) (gen(src[0]), callFuncF32(funcs.atan));
      else if (op === AluOp.Exp) (gen(src[0]), callFuncF32(funcs.exp));
      else if (op === AluOp.Log) (gen(src[0]), callFuncF32(funcs.log));
      else if (op === AluOp.Erf) (gen(src[0]), callFuncF32(funcs.erf));
      else if (op === AluOp.Erfc) (gen(src[0]), callFuncF32(funcs.erfc));
      else if (op === AluOp.Sqrt) (gen(src[0]), dtyF(cg, op, dtype).sqrt());
      else if (op === AluOp.Reciprocal) {
        const dt = dtyF(cg, op, dtype);
        (dt.const(1), gen(src[0]), dt.div());
      } else if (op === AluOp.Floor) (gen(src[0]), dtyF(cg, op, dtype).floor());
      else if (op === AluOp.Ceil) (gen(src[0]), dtyF(cg, op, dtype).ceil());
      else if (op === AluOp.Cast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr =
          dtype0 === DType.Int32 ||
          dtype0 === DType.Uint32 ||
          dtype0 === DType.Bool;
        if (dtype === DType.Int32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_s();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_s();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Uint32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_u();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_u();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float32) {
          if (dtype0 === DType.Float32) void 0;
          else if (dtype0 === DType.Float64) cg.f32.demote_f64();
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f32.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f32.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float64) {
          if (dtype0 === DType.Float32) cg.f64.promote_f32();
          else if (dtype0 === DType.Float64) void 0;
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f64.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f64.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Bool) {
          if (dtype0 === DType.Bool) void 0;
          else if (i32repr) (cg.i32.const(0), cg.i32.ne());
          else if (dtype0 === DType.Float32) (cg.f32.const(0), cg.f32.ne());
          else if (dtype0 === DType.Float64) (cg.f64.const(0), cg.f64.ne());
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Bitcast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        if (dtype !== dtype0) {
          const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32;
          if (dtype === DType.Int32 || dtype === DType.Uint32) {
            if (dtype0 === DType.Float32) cg.i32.reinterpret_f32();
            else if (i32repr) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else if (dtype === DType.Float32) {
            if (i32repr) cg.f32.reinterpret_i32();
            else if (dtype0 === DType.Float32) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else throw new UnsupportedOpError(op, dtype, "wasm");
        }
      } else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (op === AluOp.Where) {
      // Cost-based decision: use if/else true branching when at least one
      // arm is expensive (contains transcendental function calls like exp,
      // log, sin, erf). Branch overhead is ~5 cycles on WASM; a function
      // call costs ~20-100 cycles. Skipping the expensive arm when not
      // taken is a net win per element.
      const costT = src[1].estimateCost();
      const costF = src[2].estimateCost();
      if (Math.max(costT, costF) >= 15) {
        // 1) Evaluate condition first — leaves i32 on stack for `if`.
        gen(src[0]);

        // 2) Pre-evaluate shared subexpressions in the arms to prevent
        //    CSE locals from being uninitialized in the untaken branch.
        //    Any arm node with refcount > 1 might be cached by CSE on
        //    first eval — if that first eval is inside one branch, the
        //    other branch (or code after the Where) would read default-zero
        //    from the uninitialised local. Evaluating them here, before
        //    the branch, guarantees the local is always set.
        const armNodes = new Set<AluExp>();
        const collectArmNodes = (node: AluExp) => {
          if (armNodes.has(node)) return;
          armNodes.add(node);
          for (const s of node.src) collectArmNodes(s);
        };
        collectArmNodes(src[1]);
        collectArmNodes(src[2]);
        for (const node of armNodes) {
          if (
            (references.get(node) ?? 0) > 1 &&
            !expContext.has(node) &&
            node.op !== AluOp.Const &&
            node.op !== AluOp.Variable &&
            node.op !== AluOp.Special
          ) {
            gen(node);
            cg.drop();
          }
        }

        // 3) Emit if/else — only the taken branch executes.
        cg.if(dty(cg, null, dtype));
        gen(src[1]); // true arm
        cg.else();
        gen(src[2]); // false arm
        cg.end();
      } else {
        // Branchless path: both arms are cheap — select is faster.
        gen(src[1]); // t
        gen(src[2]); // f
        gen(src[0]); // cond
        cg.select();
      }
    } else if (op === AluOp.Threefry2x32) {
      for (let i = 0; i < 4; i++) gen(src[i]);
      cg.call(funcs.threefry2x32);
      if (arg === "xor") cg.i32.xor();
      else if (arg === 0) cg.drop();
      else if (arg === 1) {
        const local = cg.local.declare(cg.i32);
        cg.local.set(local);
        cg.drop();
        cg.local.get(local);
      } else throw new UnsupportedOpError(op, dtype, "wasm", arg);
    } else if (op === AluOp.Const) {
      return dty(cg, op, dtype).const(arg as number);
    } else if (op === AluOp.Special) {
      const resolved = ctx.getVariable(arg[0] as string);
      if (resolved === undefined) throw new Error(`unknown special: ${arg[0]}`);
      return cg.local.get(resolved);
    } else if (op === AluOp.Variable) {
      const resolved = ctx.getVariable(arg as string);
      if (resolved === undefined) throw new Error(`unknown variable: ${arg}`);
      return cg.local.get(resolved);
    } else if (op === AluOp.GlobalIndex || op === AluOp.GlobalView) {
      const [gid, len] = arg as [number, number];
      ctx.handleGlobalIndex(cg, gen, gid, len, src[0], dtype);
    } else throw new UnsupportedOpError(op, dtype, "wasm");

    if ((references.get(e) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(e, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

/**
 * Translate an AluExp to WASM code for a regular kernel.
 * This is a thin wrapper around translateExpCore with kernel-specific GlobalIndex handling.
 */
function translateExp(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: Record<string, number>,
  /**
   * Offset to add to GlobalIndex buffer indices when looking up function
   * params. Accounts for the (start, end) prefix in kernel signatures.
   * Default 0 (scan context — no prefix).
   */
  paramOffset = 0,
) {
  translateExpCore(cg, funcs, exp, {
    getVariable: (name) => ctx[name],
    handleGlobalIndex: (cg, gen, gid, len, indexExp, dtype) => {
      gen(indexExp);

      // If value is out-of-bounds, just set it to be zero.
      // This extra bounds-check is needed in Wasm because otherwise we will get
      // out-of-bounds memory access traps. WebGPU just silently returns 0.
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.i32.const(0);
      cg.local.get(local);
      cg.i32.const(len);
      cg.i32.lt_u();
      cg.select();

      cg.i32.const(byteWidth(dtype));
      cg.i32.mul();
      cg.local.get(gid + paramOffset); // base offset of array (shifted by RANGE_PARAMS for kernels)
      cg.i32.add();
      dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(byteWidth(dtype)));
    },
  });
}

// ---------------------------------------------------------------------------
// Reduction accumulate helper (shared by kernel and scan codegen)
// ---------------------------------------------------------------------------

export function codegenReductionAccumulate(
  cg: CodeGenerator,
  re: { op: AluOp; dtype: DType; size: SizeExpr; identity: number },
  acc: number,
  kahanComp?: number,
): void {
  if (re.op === AluOp.Add) {
    if (kahanComp !== undefined && re.dtype === DType.Float64) {
      // Kahan compensated summation for Float64:
      //   y = val - comp;  t = acc + y;  comp = (t - acc) - y;  acc = t;
      const val = cg.local.declare(cg.f64);
      cg.local.set(val); // pop expression result into val

      // y = val - comp
      const y = cg.local.declare(cg.f64);
      cg.local.get(val);
      cg.local.get(kahanComp);
      cg.f64.sub();
      cg.local.set(y);

      // t = acc + y
      const t = cg.local.declare(cg.f64);
      cg.local.get(acc);
      cg.local.get(y);
      cg.f64.add();
      cg.local.set(t);

      // comp = (t - acc) - y
      cg.local.get(t);
      cg.local.get(acc);
      cg.f64.sub();
      cg.local.get(y);
      cg.f64.sub();
      cg.local.set(kahanComp);

      // acc = t
      cg.local.get(t);
      cg.local.set(acc);
      return;
    }
    cg.local.get(acc);
    if (re.dtype === DType.Bool) cg.i32.or();
    else dty(cg, re.op, re.dtype).add();
  } else if (re.op === AluOp.Mul) {
    cg.local.get(acc);
    if (re.dtype === DType.Bool) cg.i32.and();
    else dty(cg, re.op, re.dtype).mul();
  } else if (re.op === AluOp.Min || re.op === AluOp.Max) {
    if (isFloatDtype(re.dtype)) {
      cg.local.get(acc);
      if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
      else dtyF(cg, re.op, re.dtype).max();
    } else if ([DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)) {
      // WASM has no i32.min/max, so emulate with select
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.local.get(acc);
      cg.local.get(local);
      cg.local.get(acc);
      if (re.op === AluOp.Min) {
        if (re.dtype === DType.Int32) cg.i32.lt_s();
        else cg.i32.lt_u();
      } else {
        if (re.dtype === DType.Int32) cg.i32.gt_s();
        else cg.i32.gt_u();
      }
      cg.select();
    } else {
      throw new Error(`invalid reduction min/max over ${re.dtype}`);
    }
  } else {
    throw new Error(`invalid wasm reduction op: ${re.op}`);
  }
  cg.local.set(acc);
}

// ---------------------------------------------------------------------------
// Shared kernel body: gidx loop + reduction + store (used by kernel & scan)
// ---------------------------------------------------------------------------

/**
 * Emit the inner per-element loop for a single-output kernel.
 *
 * Shared between `codegenWasm()` (regular kernels) and
 * `codegenNativeScanGeneral()` (scan kernel steps). Callers inject
 * backend-specific behavior via three callbacks:
 *
 * - `emitOutputAddr`: push the store address for element [gidx] onto the stack
 * - `emitExp`: translate an expression, leaving result on stack
 * - `emitStore`: (optional) custom store logic; default is a simple typed store
 */
function emitKernelBody(opts: {
  cg: CodeGenerator;
  funcs: Record<string, number>;
  kernel: Kernel;
  gidx: number;
  /** Emit code to push the output base address for element [gidx] onto stack. */
  emitOutputAddr: () => void;
  /** Translate an expression (leaves result on WASM stack). */
  emitExp: (exp: AluExp, extra: { ridx?: number; acc?: number }) => void;
  /** Custom store logic. If omitted, a simple typed store is emitted. */
  emitStore?: () => void;
  /**
   * Optional WASM local holding the loop start index.
   * When provided, gidx is initialised from `local.get(startLocal)` instead
   * of `i32.const(0)` — enabling parallel dispatch (M5.3).
   */
  startLocal?: number;
  /**
   * Optional WASM local holding the loop end (exclusive) element count.
   * When provided, uses `local.get(endLocal)` instead of
   * `i32.const(kernel.size)` for the gidx loop bound — enabling
   * parameterized / parallel kernels whose range is determined at dispatch.
   */
  endLocal?: number;
  /**
   * Optional WASM local holding the reduction size (loop bound for ridx).
   * When provided, uses `local.get(reduceSizeLocal)` instead of
   * `i32.const(re.size)` — enabling parameterized kernels where the
   * reduction axis has a symbolic dimension resolved at dispatch time.
   */
  reduceSizeLocal?: number;
}): void {
  const {
    cg,
    kernel,
    gidx,
    emitOutputAddr,
    emitExp,
    emitStore,
    startLocal,
    endLocal,
    reduceSizeLocal,
  } = opts;
  const tune = tuneNullopt(kernel);
  const re = kernel.outputs[0].reduction;
  const storeAlign = Math.log2(byteWidth(kernel.outputs[0].dtype));

  if (startLocal !== undefined) {
    cg.local.get(startLocal);
  } else {
    cg.i32.const(0);
  }
  cg.local.set(gidx);
  cg.loop(cg.void);
  {
    // if (gidx >= end) break;
    cg.block(cg.void);
    cg.local.get(gidx);
    if (endLocal !== undefined) {
      cg.local.get(endLocal);
    } else {
      cg.i32.const(kernel.size as number);
    }
    cg.i32.ge_u();
    cg.br_if(0);

    // Push output address for this element.
    emitOutputAddr();

    if (re) {
      // Reduction: accumulator + inner ridx loop.
      const acc = cg.local.declare(dty(cg, null, kernel.outputs[0].exp.dtype));
      dty(cg, null, kernel.outputs[0].exp.dtype).const(re.identity);
      cg.local.set(acc);

      // Kahan compensation local for Float64 Add reductions.
      const useKahan = re.dtype === DType.Float64 && re.op === AluOp.Add;
      let kahanComp: number | undefined;
      if (useKahan) {
        kahanComp = cg.local.declare(cg.f64);
        cg.f64.const(0);
        cg.local.set(kahanComp);
      }

      const ridx = cg.local.declare(cg.i32);
      cg.i32.const(0);
      cg.local.set(ridx);

      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(ridx);
        if (reduceSizeLocal !== undefined) {
          cg.local.get(reduceSizeLocal);
        } else {
          cg.i32.const(re.size as number);
        }
        cg.i32.ge_u();
        cg.br_if(0);

        emitExp(tune.exp, { ridx });
        codegenReductionAccumulate(cg, re, acc, kahanComp);

        cg.local.get(ridx);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(ridx);

        cg.br(1);
        cg.end();
      }
      cg.end();

      emitExp(tune.epilogue!, { acc });
    } else {
      emitExp(tune.exp, {});
    }

    // Store result.
    if (emitStore) {
      emitStore();
    } else {
      dty(cg, null, kernel.outputs[0].dtype).store(storeAlign);
    }

    // gidx++
    cg.local.get(gidx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(gidx);

    cg.br(1); // continue gidx loop
    cg.end();
  }
  cg.end();
}

// ---------------------------------------------------------------------------
// tuneNulloptExp helper for scan (size-specific tuning without full Kernel)
// ---------------------------------------------------------------------------

function _tuneNulloptExp(exp: AluExp, size: number): AluExp {
  const gidx = AluExp.special(DType.Int32, "gidx", size);
  return exp.substitute({ gidx }).rewriteGlobalViews().simplify();
}

// ---------------------------------------------------------------------------
// General scan context for WASM expression translation
// ---------------------------------------------------------------------------

/** Context for general scan expression translation. */
interface GeneralScanContext {
  gidx: number;
  iter: number;
  dataIdx: number;
  ridx: number;
  acc?: number;
  constsBase: number;
  constSizes: number[];
  numConsts: number;
  xsBase: number;
  xsStrides: number[];
  carryBase: number;
  carrySizes: number[];
  numCarry: number;
  internalsBase: number;
  internalSizes: number[];
  numInternal: number;
  /** Total number of jaxpr inputs (consts + carry + xs). */
  numInputs: number;
}

/**
 * Translate an AluExp to WASM code within a general scan context.
 * Thin wrapper around translateExpCore with scan-specific GlobalIndex handling.
 */
function translateExpWithGeneralScanContext(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: GeneralScanContext,
) {
  translateExpCore(cg, funcs, exp, {
    getVariable: (name) => {
      if (name === "gidx") return ctx.gidx;
      if (name === "ridx") {
        if (ctx.ridx < 0)
          throw new Error("ridx used but not in reduction context");
        return ctx.ridx;
      }
      if (name === "acc") {
        if (ctx.acc === undefined)
          throw new Error("acc used but not in epilogue context");
        return ctx.acc;
      }
      return undefined;
    },
    handleGlobalIndex: (cg, gen, gid, _len, indexExp, dtype) => {
      const bw = byteWidth(dtype);

      if (gid < ctx.numConsts) {
        // Constant input (no iteration offset)
        cg.local.get(ctx.constsBase + gid);
      } else if (gid < ctx.numConsts + ctx.numCarry) {
        // Carry input (read from carryOut which has current carry values)
        const carryIdx = gid - ctx.numConsts;
        cg.local.get(ctx.carryBase + carryIdx);
      } else if (gid < ctx.numInputs) {
        // X input with iteration offset (use dataIdx for reverse support)
        const xIdx = gid - ctx.numConsts - ctx.numCarry;
        cg.local.get(ctx.xsBase + xIdx);
        cg.local.get(ctx.dataIdx);
        cg.i32.const(ctx.xsStrides[xIdx]);
        cg.i32.mul();
        cg.i32.add();
      } else {
        // Internal buffer (result from previous step)
        const internalIdx = gid - ctx.numInputs;
        cg.local.get(ctx.internalsBase + internalIdx);
      }

      // Add element index offset
      gen(indexExp);
      cg.i32.const(bw);
      cg.i32.mul();
      cg.i32.add();

      // Load the value
      dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(bw));
    },
  });
}

// ---------------------------------------------------------------------------
// Native scan codegen (WASM)
// ---------------------------------------------------------------------------

/**
 * Generate a complete WASM module for a native scan loop.
 *
 * The generated module exports a single `scan` function that:
 * 1. Copies carryIn to carryOut (working buffer)
 * 2. Loops over iterations, executing body steps (kernels) per iteration
 * 3. Copies Y outputs to ysStacked at iteration offset
 * 4. Updates carry from internal buffers
 *
 * Function arguments:
 *   [...consts, ...carryIn, ...xs, ...carryOut, ...ysStacked, ...internals, aux?]
 */
function codegenNativeScanGeneral(
  params: NativeScanGeneralParams,
  traceEnabled: boolean,
): { bytes: Uint8Array<ArrayBuffer>; wat?: string } {
  const {
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    internalSizes,
    steps,
    carryOutSources,
    yOutputSources,
    reverse,
    routineInfos,
  } = params;
  const numInternal = internalSizes.length;

  // ---- Direct-write optimization analysis ----
  // Checks if internal buffers can write directly to carryOut/ysStacked.
  const numInputs = numConsts + numCarry + numX;

  function collectCarryReads(exp: AluExp): Set<number> {
    const result = new Set<number>();
    exp.fold((e: AluExp) => {
      if (e.op === AluOp.GlobalIndex || e.op === AluOp.GlobalView) {
        const gid = (e.arg as number[])[0];
        if (gid >= numConsts && gid < numConsts + numCarry) {
          result.add(gid - numConsts);
        }
      }
    });
    return result;
  }

  const internalReadByStep = new Set<number>();
  for (const step of steps) {
    for (const slotIdx of step.inputSlots) {
      if (slotIdx >= numInputs) {
        internalReadByStep.add(slotIdx - numInputs);
      }
    }
  }

  const stepCarryReads: Set<number>[] = [];
  for (const step of steps) {
    const reads = new Set<number>();
    if (step.source instanceof Kernel) {
      // Single-output kernel: check exp and epilogue
      for (const c of collectCarryReads(step.source.outputs[0].exp))
        reads.add(c);
      if (step.source.outputs[0].reduction?.epilogue) {
        for (const c of collectCarryReads(
          step.source.outputs[0].reduction.epilogue,
        ))
          reads.add(c);
      }
    }
    stepCarryReads.push(reads);
  }

  const internalToCarry = new Map<number, number>();
  for (let c = 0; c < numCarry; c++) {
    const src = carryOutSources[c];
    if (src.type === "internal") {
      internalToCarry.set(src.internalIdx, c);
    }
  }

  const internalToY = new Map<number, number>();
  for (let y = 0; y < numY; y++) {
    const src = yOutputSources[y];
    if (src.type === "internal") {
      internalToY.set(src.internalIdx, y);
    }
  }

  const yPassthroughCarries = new Set<number>();
  for (let y = 0; y < numY; y++) {
    const src = yOutputSources[y];
    if (src.type === "passthrough") {
      yPassthroughCarries.add(src.carryIdx);
    }
  }

  interface DirectWrite {
    carryIdx: number;
    yIdx?: number;
  }
  const directWriteMap = new Map<number, DirectWrite>();

  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    if (step.source instanceof Kernel) {
      if (step.source.outputs[0].reduction) continue; // no reduction allowed for direct-write

      const indices = [step.outputInternalIdx];

      for (const intIdx of indices) {
        if (!internalToCarry.has(intIdx)) continue;
        if (internalReadByStep.has(intIdx)) continue;

        const carryIdx = internalToCarry.get(intIdx)!;
        if (yPassthroughCarries.has(carryIdx)) continue;

        let laterStepReadsCarry = false;
        for (let s = stepIdx + 1; s < steps.length; s++) {
          if (stepCarryReads[s].has(carryIdx)) {
            laterStepReadsCarry = true;
            break;
          }
        }
        if (laterStepReadsCarry) continue;

        const dw: DirectWrite = { carryIdx };
        if (internalToY.has(intIdx)) {
          dw.yIdx = internalToY.get(intIdx)!;
        }
        directWriteMap.set(intIdx, dw);
      }
    }
  }

  if (DEBUG >= 2 && directWriteMap.size > 0) {
    console.log(
      `[wasm-scan] direct-write optimization: ${directWriteMap.size} internal buffers redirected`,
      [...directWriteMap.entries()].map(([intIdx, dw]) => ({
        intIdx,
        carryIdx: dw.carryIdx,
        yIdx: dw.yIdx,
      })),
    );
  }

  // ---- Code generation ----
  const cg = new CodeGenerator();
  cg.trace = traceEnabled;
  configureMemoryImport(cg);

  // Import routine functions from the "routines" module
  const routineFuncIndices: number[] = [];
  if (routineInfos) {
    for (const info of routineInfos) {
      const funcIdx = cg.importFunction(
        "routines",
        info.exportName,
        rep(info.numParams, cg.i32),
        [],
      );
      routineFuncIndices.push(funcIdx);
    }
  }

  // Collect all helper functions needed by kernels
  const allOps = new Set<AluOp>();
  for (const step of steps) {
    if (step.source instanceof Kernel) {
      const tune = tuneNullopt(step.source);
      for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
      if (tune.epilogue) {
        for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
      }
    }
  }

  const funcs = importWasmHelperFuncs(cg, allOps);

  // Function arguments layout:
  // [length, ...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
  //  ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal), aux?]
  const needsAux = (params.auxBufferSize ?? 0) > 0;
  const numArgs =
    1 +
    numConsts +
    numCarry +
    numX +
    numCarry +
    numY +
    numInternal +
    (needsAux ? 1 : 0);
  const auxArgIdx = needsAux
    ? 1 + numConsts + numCarry + numX + numCarry + numY + numInternal
    : -1;

  const scanFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    // Local variables
    const iter = cg.local.declare(cg.i32);
    const gidx = cg.local.declare(cg.i32);
    const dataIdx = cg.local.declare(cg.i32);

    // Argument indices (length is arg 0, everything else shifted by 1)
    const lengthArg = 0;
    const constsBase = 1;
    const carryInBase = 1 + numConsts;
    const xsBase = 1 + numConsts + numCarry;
    const carryOutBase = 1 + numConsts + numCarry + numX;
    const ysStackedBase = 1 + numConsts + numCarry + numX + numCarry;
    const internalsBase = 1 + numConsts + numCarry + numX + numCarry + numY;

    // Step 1: Copy carryIn to carryOut (working buffer)
    for (let c = 0; c < numCarry; c++) {
      const size = carrySizes[c];
      cg.local.get(carryOutBase + c);
      cg.local.get(carryInBase + c);
      cg.i32.const(size);
      cg.memory.copy();
    }

    // Step 2: Main scan loop
    cg.i32.const(0);
    cg.local.set(iter);

    const makeScanContext = (): GeneralScanContext => ({
      gidx,
      iter,
      dataIdx,
      ridx: -1,
      constsBase,
      constSizes,
      numConsts,
      xsBase,
      xsStrides,
      carryBase: carryOutBase,
      carrySizes,
      numCarry,
      internalsBase,
      internalSizes,
      numInternal,
      numInputs: numConsts + numCarry + numX,
    });

    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(iter);
      cg.local.get(lengthArg);
      cg.i32.ge_u();
      cg.br_if(0);

      // Compute dataIdx = reverse ? (length - 1 - iter) : iter
      if (reverse) {
        cg.local.get(lengthArg);
        cg.i32.const(1);
        cg.i32.sub();
        cg.local.get(iter);
        cg.i32.sub();
        cg.local.set(dataIdx);
      } else {
        cg.local.get(iter);
        cg.local.set(dataIdx);
      }

      // Step 2a: Execute each step (Kernel), writing to internal buffers
      for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
        const step = steps[stepIdx];
        const internalIdx = step.outputInternalIdx;

        if (step.source instanceof Kernel) {
          // Single-output Kernel step — delegate to shared emitKernelBody.
          const kernel = step.source;
          const dw = directWriteMap.get(internalIdx);
          const bw = byteWidth(kernel.outputs[0].dtype);
          const needsDualStore = dw && dw.yIdx !== undefined;

          emitKernelBody({
            cg,
            funcs,
            kernel,
            gidx,
            emitOutputAddr: () => {
              if (dw) {
                cg.local.get(carryOutBase + dw.carryIdx);
              } else {
                cg.local.get(internalsBase + internalIdx);
              }
              cg.local.get(gidx);
              cg.i32.const(bw);
              cg.i32.mul();
              cg.i32.add();
            },
            emitExp: (exp, extra) => {
              const scanCtx = makeScanContext();
              if (extra.ridx !== undefined) scanCtx.ridx = extra.ridx;
              if (extra.acc !== undefined) scanCtx.acc = extra.acc;
              translateExpWithGeneralScanContext(cg, funcs, exp, scanCtx);
            },
            emitStore: needsDualStore
              ? () => {
                  const storeAlign = Math.log2(bw);
                  const tmpVal = cg.local.declare(
                    dty(cg, null, kernel.outputs[0].dtype),
                  );
                  cg.local.tee(tmpVal);
                  dty(cg, null, kernel.outputs[0].dtype).store(storeAlign);
                  // Store to ysStacked
                  cg.local.get(ysStackedBase + dw!.yIdx!);
                  cg.local.get(dataIdx);
                  cg.i32.const(ysStrides[dw!.yIdx!]);
                  cg.i32.mul();
                  cg.i32.add();
                  cg.local.get(gidx);
                  cg.i32.const(bw);
                  cg.i32.mul();
                  cg.i32.add();
                  cg.local.get(tmpVal);
                  dty(cg, null, kernel.outputs[0].dtype).store(storeAlign);
                }
              : undefined,
          });
        }
        // Routine step: call the imported routine function
        if (step.source instanceof Routine) {
          const callInfo = step.routineCallInfo!;
          const funcIdx = routineFuncIndices[callInfo.routineInfoIdx];
          const routineType = routineInfos![callInfo.routineInfoIdx].routine;

          // Helper to push a slot pointer onto the stack
          const pushSlotPtr = (slotIdx: number) => {
            if (slotIdx < numConsts) {
              cg.local.get(constsBase + slotIdx);
            } else if (slotIdx < numConsts + numCarry) {
              cg.local.get(carryOutBase + (slotIdx - numConsts));
            } else if (slotIdx < numConsts + numCarry + numX) {
              // xs input: base + dataIdx * stride
              const xIdx = slotIdx - numConsts - numCarry;
              cg.local.get(xsBase + xIdx);
              cg.local.get(dataIdx);
              cg.i32.const(xsStrides[xIdx]);
              cg.i32.mul();
              cg.i32.add();
            } else {
              // Internal buffer
              const intIdx = slotIdx - numConsts - numCarry - numX;
              cg.local.get(internalsBase + intIdx);
            }
          };

          if (routineType === Routines.Cholesky) {
            pushSlotPtr(step.inputSlots[0]); // inPtr
            cg.local.get(internalsBase + internalIdx); // outPtr
          } else if (routineType === Routines.Sort) {
            // Copy input to internal buffer first (sort is in-place)
            const sortSize = callInfo.staticParams[0];
            const elemSize = params.elementSize ?? 4;
            const copySize = sortSize * elemSize;

            cg.local.get(internalsBase + internalIdx); // dst
            pushSlotPtr(step.inputSlots[0]); // src
            cg.i32.const(copySize); // len
            cg.memory.copy();

            cg.local.get(internalsBase + internalIdx); // dataPtr (in-place)
            cg.local.get(auxArgIdx); // auxPtr
          } else if (routineType === Routines.TriangularSolve) {
            pushSlotPtr(step.inputSlots[0]); // aPtr
            pushSlotPtr(step.inputSlots[1]); // bPtr
            cg.local.get(internalsBase + internalIdx); // xPtr (output)
          } else if (routineType === Routines.LU) {
            const outIndices = step.outputInternalIndices!;
            pushSlotPtr(step.inputSlots[0]); // aPtr
            cg.local.get(internalsBase + outIndices[0]); // luPtr
            cg.local.get(internalsBase + outIndices[1]); // pivPtr
            cg.local.get(internalsBase + outIndices[2]); // permPtr
          } else if (routineType === Routines.Argsort) {
            const outIndices = step.outputInternalIndices!;
            pushSlotPtr(step.inputSlots[0]); // dataPtr
            cg.local.get(internalsBase + outIndices[0]); // outPtr
            cg.local.get(internalsBase + outIndices[1]); // idxPtr
            cg.local.get(auxArgIdx); // auxPtr
          } else {
            pushSlotPtr(step.inputSlots[0]);
            cg.local.get(internalsBase + internalIdx);
          }

          cg.call(funcIdx);
        }
      }

      // Step 2b: Copy Y outputs to ysStacked at iteration offset
      // NOTE: Must run BEFORE carry update (2c) so passthrough reads OLD carry values
      for (let y = 0; y < numY; y++) {
        const source = yOutputSources[y];

        // Skip if this Y output was already direct-written by the kernel
        if (
          source.type === "internal" &&
          directWriteMap.has(source.internalIdx) &&
          directWriteMap.get(source.internalIdx)!.yIdx === y
        ) {
          continue;
        }

        const yStride = ysStrides[y];

        if (source.type === "passthrough") {
          const srcArgIdx = carryOutBase + source.carryIdx;
          const size = carrySizes[source.carryIdx];
          // dst = ysStacked[y] + dataIdx * yStride
          cg.local.get(ysStackedBase + y);
          cg.local.get(dataIdx);
          cg.i32.const(yStride);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(srcArgIdx);
          cg.i32.const(size);
          cg.memory.copy();
        } else if (source.type === "xs-passthrough") {
          const xsPassthroughIdx = source.xsIdx;
          const size = xsStrides[xsPassthroughIdx];
          // dst = ysStacked[y] + dataIdx * yStride
          cg.local.get(ysStackedBase + y);
          cg.local.get(dataIdx);
          cg.i32.const(yStride);
          cg.i32.mul();
          cg.i32.add();
          // src = xs[xsIdx] + dataIdx * xsStrides[xsIdx]
          cg.local.get(xsBase + xsPassthroughIdx);
          cg.local.get(dataIdx);
          cg.i32.const(xsStrides[xsPassthroughIdx]);
          cg.i32.mul();
          cg.i32.add();
          cg.i32.const(size);
          cg.memory.copy();
        } else {
          // internal
          const srcArgIdx = internalsBase + source.internalIdx;
          const size = internalSizes[source.internalIdx];
          // dst = ysStacked[y] + dataIdx * yStride
          cg.local.get(ysStackedBase + y);
          cg.local.get(dataIdx);
          cg.i32.const(yStride);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(srcArgIdx);
          cg.i32.const(size);
          cg.memory.copy();
        }
      }

      // Step 2c: Copy carry outputs from internal buffers to carryOut
      for (let c = 0; c < numCarry; c++) {
        const source = carryOutSources[c];

        // Skip if this carry output was already direct-written
        if (
          source.type === "internal" &&
          directWriteMap.has(source.internalIdx)
        ) {
          continue;
        }

        const size = carrySizes[c];
        const srcLocal =
          source.type === "passthrough"
            ? carryInBase + source.carryIdx
            : internalsBase + source.internalIdx;

        cg.local.get(carryOutBase + c);
        cg.local.get(srcLocal);
        cg.i32.const(size);
        cg.memory.copy();
      }

      // iter++
      cg.local.get(iter);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(iter);

      cg.br(1);
      cg.end();
    }
    cg.end();
  });

  cg.export(scanFunc, "scan");
  const bytes = cg.finish();
  return { bytes, wat: traceEnabled ? cg.toWat() : undefined };
}

// ---------------------------------------------------------------------------
// Compiled block-map loop codegen (WASM)
// ---------------------------------------------------------------------------

/** Context for block-map WASM expression translation. */
interface BlockMapContext {
  gidx: number;
  ridx: number;
  acc?: number;
  constsBase: number;
  numConsts: number;
  scratchInputsBase: number;
  numInputs: number;
  internalsBase: number;
  numInternal: number;
}

/**
 * Translate an AluExp to WASM code within a block-map context.
 * Body kernels have GIDs mapped as:
 *   [0..numConsts) → constants
 *   [numConsts..numConsts+numInputs) → scratch block-input buffers
 *   [numConsts+numInputs..) → internal buffers
 */
function translateExpWithBlockMapContext(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: BlockMapContext,
) {
  translateExpCore(cg, funcs, exp, {
    getVariable: (name) => {
      if (name === "gidx") return ctx.gidx;
      if (name === "ridx") {
        if (ctx.ridx < 0)
          throw new Error("ridx used but not in reduction context");
        return ctx.ridx;
      }
      if (name === "acc") {
        if (ctx.acc === undefined)
          throw new Error("acc used but not in epilogue context");
        return ctx.acc;
      }
      return undefined;
    },
    handleGlobalIndex: (cg, gen, gid, _len, indexExp, dtype) => {
      const bw = byteWidth(dtype);

      if (gid < ctx.numConsts) {
        // Constant input
        cg.local.get(ctx.constsBase + gid);
      } else if (gid < ctx.numConsts + ctx.numInputs) {
        // Block input (scratch buffer — already sliced)
        cg.local.get(ctx.scratchInputsBase + (gid - ctx.numConsts));
      } else {
        // Internal buffer
        const intIdx = gid - ctx.numConsts - ctx.numInputs;
        cg.local.get(ctx.internalsBase + intIdx);
      }

      // Add element index offset
      gen(indexExp);
      cg.i32.const(bw);
      cg.i32.mul();
      cg.i32.add();

      // Load the value
      dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(bw));
    },
  });
}

/**
 * Generate a WASM module for a compiled block-map loop.
 *
 * The generated module exports a single `block_map_loop` function that
 * iterates over all blocks, copies input slices into scratch buffers,
 * executes inlined body kernels, and copies outputs into the result buffers.
 *
 * Function arguments:
 *   [...constPtrs, ...inputPtrs, ...outputPtrs,
 *    ...scratchInputPtrs, ...scratchOutputPtrs, ...internalPtrs]
 */
function codegenBlockMapLoop(
  params: BlockMapWasmParams,
  traceEnabled: boolean,
): { bytes: Uint8Array<ArrayBuffer>; wat?: string } {
  const {
    numConsts,
    numInputs,
    numOutputs,
    gridRank,
    gridShape,
    blockShape,
    inAxes,
    outAxes,
    inputShapes,
    outputShapes,
    blockInputSizes,
    blockOutputSizes,
    internalSizes,
    steps,
    outputSources,
    inputStrides,
    outputStrides,
    halo,
  } = params;
  const numInternal = internalSizes.length;
  const numBlocks = gridShape.reduce((a, b) => a * b, 1);

  // ---- Code generation ----
  const cg = new CodeGenerator();
  cg.trace = traceEnabled;
  configureMemoryImport(cg);

  // Collect all helper functions needed by body kernels
  const allOps = new Set<AluOp>();
  for (const step of steps) {
    if (step.source instanceof Kernel) {
      const tune = tuneNullopt(step.source);
      for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
      if (tune.epilogue) {
        for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
      }
    }
  }
  const funcs = importWasmHelperFuncs(cg, allOps);

  // Function arguments layout:
  //   [consts(numConsts), inputs(numInputs), outputs(numOutputs),
  //    scratchInputs(numInputs), scratchOutputs(numOutputs), internals(numInternal)]
  const numArgs =
    numConsts + numInputs + numOutputs + numInputs + numOutputs + numInternal;

  const constsBase = 0;
  const inputsBase = numConsts;
  const outputsBase = numConsts + numInputs;
  const scratchInputsBase = numConsts + numInputs + numOutputs;
  const _scratchOutputsBase = numConsts + numInputs + numOutputs + numInputs;
  const internalsBase =
    numConsts + numInputs + numOutputs + numInputs + numOutputs;

  const mainFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    // Local variables
    const flatIdx = cg.local.declare(cg.i32);
    const gidx = cg.local.declare(cg.i32);
    // Block coordinates (one per grid rank)
    const blockCoords: number[] = [];
    for (let g = 0; g < gridRank; g++) {
      blockCoords.push(cg.local.declare(cg.i32));
    }
    const remaining = cg.local.declare(cg.i32);
    // Temp locals for byte offsets
    const srcOffset = cg.local.declare(cg.i32);
    const dstOffset = cg.local.declare(cg.i32);
    const copySize = cg.local.declare(cg.i32);

    // Main loop: for flatIdx = 0 to numBlocks - 1
    cg.i32.const(0);
    cg.local.set(flatIdx);

    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(flatIdx);
      cg.i32.const(numBlocks);
      cg.i32.ge_u();
      cg.br_if(0);

      // Decompose flatIdx to grid coordinates (row-major: last dim fastest)
      cg.local.get(flatIdx);
      cg.local.set(remaining);
      for (let g = gridRank - 1; g >= 0; g--) {
        cg.local.get(remaining);
        cg.i32.const(gridShape[g]);
        cg.i32.rem_u();
        cg.local.set(blockCoords[g]);
        cg.local.get(remaining);
        cg.i32.const(gridShape[g]);
        cg.i32.div_u();
        cg.local.set(remaining);
      }

      // ---- Step 1: Copy input slices into scratch buffers ----
      // Declare rawStart local only if any input has halo (compile-time check)
      const _anyHalo =
        halo != null &&
        halo.some((h, idx) =>
          inAxes[idx].some(
            (ax, g) => ax !== null && (h[g][0] !== 0 || h[g][1] !== 0),
          ),
        );
      const rawStart = _anyHalo ? cg.local.declare(cg.i32) : -1;

      for (let i = 0; i < numInputs; i++) {
        const axes = inAxes[i];
        const inputShape = inputShapes[i];
        const iStrides = inputStrides[i];
        const elemBytes = iStrides[inputShape.length - 1];
        const totalBytes = inputShape.reduce((a, b) => a * b, 1) * elemBytes;

        // Find the single mapped grid axis (at most one due to contiguity
        // check in buildBlockMapWasmParams). Determine per-input halo.
        let mappedG = -1;
        let lo = 0;
        let hi = 0;
        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            mappedG = g;
            if (halo) {
              lo = halo[i][g][0];
              hi = halo[i][g][1];
            }
          }
        }

        if (mappedG === -1) {
          // Broadcast input (no mapped axis) — copy full input to scratch.
          // blockInputSizes[i] === totalBytes for broadcast inputs.
          cg.local.get(scratchInputsBase + i);
          cg.local.get(inputsBase + i);
          cg.i32.const(blockInputSizes[i]);
          cg.memory.copy();
          continue;
        }

        const ax = axes[mappedG]!;
        const axStride = iStrides[ax];

        if (lo === 0 && hi === 0) {
          // ---- Non-halo input: interior fast path ----
          // srcOffset = blockCoord * blockShape * axStride
          cg.local.get(blockCoords[mappedG]);
          cg.i32.const(blockShape[mappedG] * axStride);
          cg.i32.mul();
          cg.local.set(srcOffset);

          // Interior test: available = totalBytes - srcOffset >= blockInputSizes?
          cg.i32.const(totalBytes);
          cg.local.get(srcOffset);
          cg.i32.sub();
          cg.i32.const(blockInputSizes[i]);
          cg.i32.ge_u();
          cg.if(cg.void);
          {
            // Fast path: copy only, no fill
            cg.local.get(scratchInputsBase + i);
            cg.local.get(inputsBase + i);
            cg.local.get(srcOffset);
            cg.i32.add();
            cg.i32.const(blockInputSizes[i]);
            cg.memory.copy();
          }
          cg.else();
          {
            // Boundary path: fill + clamped copy
            cg.local.get(scratchInputsBase + i);
            cg.i32.const(0);
            cg.i32.const(blockInputSizes[i]);
            cg.memory.fill();

            // copySize = totalBytes - srcOffset (available bytes)
            cg.i32.const(totalBytes);
            cg.local.get(srcOffset);
            cg.i32.sub();
            cg.local.set(copySize);

            cg.local.get(scratchInputsBase + i); // dst
            cg.local.get(inputsBase + i);
            cg.local.get(srcOffset);
            cg.i32.add(); // src
            cg.local.get(copySize);
            cg.memory.copy();
          }
          cg.end();
        } else {
          // ---- Halo input: signed interior test ----
          const inputDim = inputShape[ax];
          const tileElems = blockShape[mappedG] + lo + hi;

          // rawStart (elements) = blockCoord * blockShape - lo
          cg.local.get(blockCoords[mappedG]);
          cg.i32.const(blockShape[mappedG]);
          cg.i32.mul();
          cg.i32.const(lo);
          cg.i32.sub();
          cg.local.set(rawStart);

          // Interior: rawStart >= 0 AND rawStart + tileElems <= inputDim
          cg.local.get(rawStart);
          cg.i32.const(0);
          cg.i32.ge_s();
          cg.local.get(rawStart);
          cg.i32.const(inputDim - tileElems);
          cg.i32.le_s();
          cg.i32.and();
          cg.if(cg.void);
          {
            // Fast path: copy full tile, no fill
            cg.local.get(scratchInputsBase + i);
            cg.local.get(inputsBase + i);
            cg.local.get(rawStart);
            cg.i32.const(axStride);
            cg.i32.mul();
            cg.i32.add(); // src = input + rawStart * axStride
            cg.i32.const(blockInputSizes[i]);
            cg.memory.copy();
          }
          cg.else();
          {
            // Boundary path: fill + clamped copy with dstSkip
            cg.local.get(scratchInputsBase + i);
            cg.i32.const(0);
            cg.i32.const(blockInputSizes[i]);
            cg.memory.fill();

            // clampedStart = max(0, rawStart) via select
            cg.local.get(rawStart); // val1 (kept if cond true)
            cg.i32.const(0); // val2 (kept if cond false)
            cg.local.get(rawStart);
            cg.i32.const(0);
            cg.i32.gt_s(); // cond: rawStart > 0
            cg.select();
            cg.local.set(srcOffset); // srcOffset = clampedStart

            // clampedEnd = min(inputDim, rawStart + tileElems) via select
            cg.local.get(rawStart);
            cg.i32.const(tileElems);
            cg.i32.add();
            cg.local.set(dstOffset); // dstOffset = rawEnd (temp)

            cg.i32.const(inputDim); // val1
            cg.local.get(dstOffset); // val2 (rawEnd)
            cg.i32.const(inputDim);
            cg.local.get(dstOffset);
            cg.i32.lt_s(); // cond: inputDim < rawEnd
            cg.select();
            cg.local.set(dstOffset); // dstOffset = clampedEnd

            // copyLen = (clampedEnd - clampedStart) * axStride
            cg.local.get(dstOffset);
            cg.local.get(srcOffset);
            cg.i32.sub();
            cg.i32.const(axStride);
            cg.i32.mul();
            cg.local.set(copySize);

            // dst = scratch + (clampedStart - rawStart) * axStride
            cg.local.get(scratchInputsBase + i);
            cg.local.get(srcOffset);
            cg.local.get(rawStart);
            cg.i32.sub();
            cg.i32.const(axStride);
            cg.i32.mul();
            cg.i32.add();

            // src = input + clampedStart * axStride
            cg.local.get(inputsBase + i);
            cg.local.get(srcOffset);
            cg.i32.const(axStride);
            cg.i32.mul();
            cg.i32.add();

            // memory.copy(dst, src, copyLen)
            cg.local.get(copySize);
            cg.memory.copy();
          }
          cg.end();
        }
      }

      // ---- Step 2: Execute body kernel steps ----
      for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
        const step = steps[stepIdx];
        if (step.source instanceof Kernel) {
          const kernel = step.source;
          const internalIdx = step.outputInternalIdx;
          const bw = byteWidth(kernel.outputs[0].dtype);

          emitKernelBody({
            cg,
            funcs,
            kernel,
            gidx,
            emitOutputAddr: () => {
              cg.local.get(internalsBase + internalIdx);
              cg.local.get(gidx);
              cg.i32.const(bw);
              cg.i32.mul();
              cg.i32.add();
            },
            emitExp: (exp, extra) => {
              const ctx: BlockMapContext = {
                gidx,
                ridx: -1,
                constsBase,
                numConsts,
                scratchInputsBase,
                numInputs,
                internalsBase,
                numInternal,
              };
              if (extra.ridx !== undefined) ctx.ridx = extra.ridx;
              if (extra.acc !== undefined) ctx.acc = extra.acc;
              translateExpWithBlockMapContext(cg, funcs, exp, ctx);
            },
          });
        }
      }

      // ---- Step 3: Copy body outputs to the result buffers ----
      for (let o = 0; o < numOutputs; o++) {
        const axes = outAxes[o];
        const outputShape = outputShapes[o];
        const oStrides = outputStrides[o];
        const srcInternalIdx = outputSources[o];

        // Compute destination byte offset
        cg.i32.const(0);
        cg.local.set(dstOffset);
        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            const ax = axes[g]!;
            cg.local.get(dstOffset);
            cg.local.get(blockCoords[g]);
            cg.i32.const(blockShape[g] * oStrides[ax]);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(dstOffset);
          }
        }

        // Copy from internal buffer to output, clamping at boundary
        cg.local.get(outputsBase + o); // dst base
        cg.local.get(dstOffset);
        cg.i32.add(); // dst = outputPtr + dstOffset

        cg.local.get(internalsBase + srcInternalIdx); // src = internal buffer

        // copySize = min(blockOutputSizes[o], outputTotalBytes - dstOffset)
        const outElemBytes = oStrides[outputShape.length - 1];
        const outTotalBytes =
          outputShape.reduce((a, b) => a * b, 1) * outElemBytes;
        cg.i32.const(outTotalBytes);
        cg.local.get(dstOffset);
        cg.i32.sub(); // available
        cg.local.tee(copySize);
        cg.i32.const(blockOutputSizes[o]);
        cg.i32.gt_u();
        cg.if(cg.void);
        cg.i32.const(blockOutputSizes[o]);
        cg.local.set(copySize);
        cg.end();

        cg.local.get(copySize);
        cg.memory.copy();
      }

      // flatIdx++
      cg.local.get(flatIdx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(flatIdx);

      cg.br(1); // continue loop
      cg.end();
    }
    cg.end();
  });

  cg.export(mainFunc, "block_map_loop");
  const bytes = cg.finish();
  return { bytes, wat: traceEnabled ? cg.toWat() : undefined };
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

export function dty(cg: CodeGenerator, op: AluOp | null, dtype: DType) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}

export function dtyF(
  cg: CodeGenerator,
  op: AluOp | null,
  dtype: DType,
): CodeGenerator["f32" | "f64"] {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}

// ---------------------------------------------------------------------------
// Native associative scan codegen (WASM) — M7.2
// ---------------------------------------------------------------------------

/**
 * Parameters for generating a WASM module that runs the full Kogge-Stone
 * associative scan ladder in a single invocation.
 *
 * The generated module accepts N as a runtime i32 parameter, so a single
 * compilation can be reused for different input lengths (polymorphic length).
 */
export interface NativeAssocScanParams {
  /** Number of constant inputs (shared across all elements). */
  numConsts: number;
  /** Byte size of each constant input. */
  constSizes: number[];
  /** Number of leaves in the pytree (== numLeaves). */
  numLeaves: number;
  /** Per-element byte size of each leaf (e.g. 4 for f32 scalar). */
  leafElemSizes: number[];
  /** Body kernel steps (compiled from the body jaxpr). */
  steps: GeneralScanStep[];
  /** Byte sizes of internal buffers used between steps. */
  internalSizes: number[];
  /** Whether to reverse the scan direction. */
  reverse: boolean;
  /**
   * Mapping from output leaf index to internal buffer index.
   * leafToInternalIdx[k] = the internal buffer index that produces
   * the k-th output leaf of the body function.
   */
  leafToInternalIdx: number[];
}

// ---------------------------------------------------------------------------
// Blocked associative scan (Phase 5)
// ---------------------------------------------------------------------------

/**
 * Params for the blocked associative scan WASM module.
 * Same body kernel steps as the flat version, plus a block size.
 */
export interface NativeAssocScanBlockedParams extends NativeAssocScanParams {
  /** Block size for the blocked Kogge-Stone decomposition. */
  blockSize: number;
}

/**
 * Generate a WASM module for a blocked associative scan.
 *
 * Three-level algorithm:
 *   Level 1: Per-block prefix scans (Kogge-Stone with stride < B)
 *   Level 2: Flat Kogge-Stone on M block summaries
 *   Level 3: Apply scanned summaries to blocks 1..M-1
 *
 * Function signature:
 *   (N: i32, ...constPtrs, ...inputLeafPtrs, ...outputLeafPtrs,
 *    pingPtr, pongPtr, summaryPingPtr, summaryPongPtr, ...internalPtrs) -> ()
 */
function codegenBlockedAssociativeScan(
  params: NativeAssocScanBlockedParams,
  traceEnabled: boolean,
): { bytes: Uint8Array<ArrayBuffer>; wat?: string } {
  const {
    numConsts,
    numLeaves,
    leafElemSizes,
    steps,
    internalSizes,
    reverse,
    blockSize: B,
  } = params;
  const numInternal = internalSizes.length;
  const numInputs = numConsts + 2 * numLeaves;

  // Precompute leaf byte offsets within ping/pong buffers
  const leafElemOffsets: number[] = [];
  let offset = 0;
  for (let k = 0; k < numLeaves; k++) {
    leafElemOffsets.push(offset);
    offset += leafElemSizes[k];
  }

  // Collect all ops for helper function imports
  const allOps = new Set<AluOp>();
  for (const step of steps) {
    if (step.source instanceof Kernel) {
      const tune = tuneNullopt(step.source);
      for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
      if (tune.epilogue) {
        for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
      }
    }
  }

  const cg = new CodeGenerator();
  cg.trace = traceEnabled;
  configureMemoryImport(cg);
  const funcs = importWasmHelperFuncs(cg, allOps);

  // Arguments: (N, ...consts, ...inputs, ...outputs, ping, pong, sPing, sPong, ...internals)
  const NArg = 0;
  const constsBase = 1;
  const inputsBase = 1 + numConsts;
  const outputsBase = 1 + numConsts + numLeaves;
  const pingArg = 1 + numConsts + numLeaves + numLeaves;
  const pongArg = pingArg + 1;
  const sPingArg = pongArg + 1;
  const sPongArg = sPingArg + 1;
  const internalsBase = sPongArg + 1;
  const numArgs = internalsBase + numInternal;

  // Helper: emit the body kernel steps for a single element combine.
  // `emitAAddr(leafIdx)` pushes the base address of the a-operand for leaf k.
  // `emitBAddr(leafIdx)` pushes the base address of the b-operand for leaf k.
  // `emitOutAddr(leafIdx)` pushes the base address of the output for leaf k.
  // After calling, internal buffers hold intermediate results too.
  function emitCombineAndCopy(
    gidx: number,
    emitALeafAddr: (k: number) => void,
    emitBLeafAddr: (k: number) => void,
    emitCopyDstAddr: (k: number) => void,
  ) {
    for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
      const step = steps[stepIdx];
      if (!(step.source instanceof Kernel)) {
        throw new Error("blocked assoc_scan: only Kernel steps supported");
      }
      const kernel = step.source;
      const internalIdx = step.outputInternalIdx;
      const bw = byteWidth(kernel.outputs[0].dtype);

      emitKernelBody({
        cg,
        funcs,
        kernel,
        gidx,
        emitOutputAddr: () => {
          cg.local.get(internalsBase + internalIdx);
          cg.local.get(gidx);
          cg.i32.const(bw);
          cg.i32.mul();
          cg.i32.add();
        },
        emitExp: (exp, extra) => {
          translateExpCore(cg, funcs, exp, {
            getVariable: (name) => {
              if (name === "gidx") return gidx;
              if (name === "ridx" && extra.ridx !== undefined)
                return extra.ridx;
              if (name === "acc" && extra.acc !== undefined) return extra.acc;
              return undefined;
            },
            handleGlobalIndex: (cg, gen, gid, _len, indexExp, dtype) => {
              const elemBw = byteWidth(dtype);
              if (gid < numConsts) {
                cg.local.get(constsBase + gid);
              } else if (gid < numConsts + numLeaves) {
                // a-leaf
                emitALeafAddr(gid - numConsts);
              } else if (gid < numInputs) {
                // b-leaf
                emitBLeafAddr(gid - numConsts - numLeaves);
              } else {
                // internal buffer
                cg.local.get(internalsBase + (gid - numInputs));
              }
              gen(indexExp);
              cg.i32.const(elemBw);
              cg.i32.mul();
              cg.i32.add();
              dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(elemBw));
            },
          });
        },
      });
    }
    // Copy from internal buffers to output
    for (let k = 0; k < numLeaves; k++) {
      emitCopyDstAddr(k);
      cg.local.get(internalsBase + params.leafToInternalIdx[k]);
      emitCopy(leafElemSizes[k]);
    }
  }

  // Helper: emit an inline byte copy for small compile-time-known sizes.
  // Expects [dst, src] on the WASM operand stack. Uses i32/v128 load/store
  // (not f32) for bit-exact raw copies that preserve all bit patterns.
  let tmpCopyDst: number | undefined;
  let tmpCopySrc: number | undefined;
  function emitCopy(size: number) {
    if (size === 4) {
      cg.i32.load(2, 0);
      cg.i32.store(2, 0);
    } else if (size === 16) {
      cg.v128.load(2, 0);
      cg.v128.store(2, 0);
    } else if (size === 8) {
      tmpCopyDst ??= cg.local.declare(cg.i32);
      tmpCopySrc ??= cg.local.declare(cg.i32);
      cg.local.set(tmpCopySrc);
      cg.local.tee(tmpCopyDst);
      cg.local.get(tmpCopySrc);
      cg.i32.load(2, 0);
      cg.i32.store(2, 0);
      cg.local.get(tmpCopyDst);
      cg.local.get(tmpCopySrc);
      cg.i32.load(2, 4);
      cg.i32.store(2, 4);
    } else if (size === 32) {
      tmpCopyDst ??= cg.local.declare(cg.i32);
      tmpCopySrc ??= cg.local.declare(cg.i32);
      cg.local.set(tmpCopySrc);
      cg.local.tee(tmpCopyDst);
      cg.local.get(tmpCopySrc);
      cg.v128.load(2, 0);
      cg.v128.store(2, 0);
      cg.local.get(tmpCopyDst);
      cg.local.get(tmpCopySrc);
      cg.v128.load(2, 16);
      cg.v128.store(2, 16);
    } else {
      cg.i32.const(size);
      cg.memory.copy();
    }
  }

  const mainFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    const stride = cg.local.declare(cg.i32);
    const i = cg.local.declare(cg.i32);
    const gidx = cg.local.declare(cg.i32);
    const curPing = cg.local.declare(cg.i32);
    const curPong = cg.local.declare(cg.i32);
    const curSPing = cg.local.declare(cg.i32);
    const curSPong = cg.local.declare(cg.i32);
    const tmp = cg.local.declare(cg.i32);
    const M = cg.local.declare(cg.i32); // number of blocks
    const posInBlock = cg.local.declare(cg.i32);

    // Initialize pointers
    cg.local.get(pingArg);
    cg.local.set(curPing);
    cg.local.get(pongArg);
    cg.local.set(curPong);
    cg.local.get(sPingArg);
    cg.local.set(curSPing);
    cg.local.get(sPongArg);
    cg.local.set(curSPong);

    // M = ceil(N / B) = (N + B - 1) / B
    cg.local.get(NArg);
    cg.i32.const(B - 1);
    cg.i32.add();
    cg.i32.const(B);
    cg.i32.div_u();
    cg.local.set(M);

    // ---- Copy inputs → ping ----
    for (let k = 0; k < numLeaves; k++) {
      cg.local.get(curPing);
      cg.i32.const(leafElemOffsets[k]);
      cg.local.get(NArg);
      cg.i32.mul();
      cg.i32.add();
      cg.local.get(inputsBase + k);
      cg.local.get(NArg);
      cg.i32.const(leafElemSizes[k]);
      cg.i32.mul();
      cg.memory.copy();
    }

    // ---- If reverse: reverse each leaf in ping ----
    if (reverse) {
      const j = cg.local.declare(cg.i32);
      const halfN = cg.local.declare(cg.i32);
      cg.local.get(NArg);
      cg.i32.const(1);
      cg.i32.shr_u();
      cg.local.set(halfN);
      for (let k = 0; k < numLeaves; k++) {
        const bw = leafElemSizes[k];
        cg.i32.const(0);
        cg.local.set(j);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(j);
          cg.local.get(halfN);
          cg.i32.ge_u();
          cg.br_if(0);
          // addr_left = curPing + leafOffset*N + j*bw
          cg.local.get(curPing);
          cg.i32.const(leafElemOffsets[k]);
          cg.local.get(NArg);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(j);
          cg.i32.const(bw);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(tmp);
          // addr_right = curPing + leafOffset*N + (N-1-j)*bw
          cg.local.get(curPing);
          cg.i32.const(leafElemOffsets[k]);
          cg.local.get(NArg);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(NArg);
          cg.i32.const(1);
          cg.i32.sub();
          cg.local.get(j);
          cg.i32.sub();
          cg.i32.const(bw);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(posInBlock); // reuse as temp
          // swap via pong temp
          cg.local.get(curPong);
          cg.local.get(tmp);
          emitCopy(bw);
          cg.local.get(tmp);
          cg.local.get(posInBlock);
          emitCopy(bw);
          cg.local.get(posInBlock);
          cg.local.get(curPong);
          emitCopy(bw);
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(j);
          cg.br(1);
          cg.end();
        }
        cg.end();
      }
    }

    // ==== LEVEL 1: Block-local Kogge-Stone (stride < B) ====
    cg.i32.const(1);
    cg.local.set(stride);

    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(stride);
      cg.i32.const(B);
      cg.i32.ge_u();
      cg.br_if(0); // break if stride >= B

      // Inner loop: for i = 0..N-1
      cg.i32.const(0);
      cg.local.set(i);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(i);
        cg.local.get(NArg);
        cg.i32.ge_u();
        cg.br_if(0);

        // posInBlock = i % B
        cg.local.get(i);
        cg.i32.const(B);
        cg.i32.rem_u();
        cg.local.set(posInBlock);

        // if posInBlock >= stride → combine; else → copy
        cg.local.get(posInBlock);
        cg.local.get(stride);
        cg.i32.ge_u();
        cg.if(cg.void);
        {
          // Combine: fn(ping[i-stride], ping[i]) → internal → pong[i]
          emitCombineAndCopy(
            gidx,
            (k) => {
              // a-leaf addr: curPing + leafOffset*N + (i-stride)*leafElemSize
              cg.local.get(curPing);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(NArg);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.local.get(stride);
              cg.i32.sub();
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
            (k) => {
              // b-leaf addr: curPing + leafOffset*N + i*leafElemSize
              cg.local.get(curPing);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(NArg);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
            (k) => {
              // output addr: curPong + leafOffset*N + i*leafElemSize
              cg.local.get(curPong);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(NArg);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
          );
        }
        cg.else();
        {
          // Copy: pong[i] = ping[i] for all leaves
          for (let k = 0; k < numLeaves; k++) {
            const size = leafElemSizes[k];
            cg.local.get(curPong);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(NArg);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(size);
            cg.i32.mul();
            cg.i32.add();
            // src
            cg.local.get(curPing);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(NArg);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(size);
            cg.i32.mul();
            cg.i32.add();
            emitCopy(size);
          }
        }
        cg.end();

        cg.local.get(i);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);
        cg.br(1);
        cg.end();
      }
      cg.end();

      // Swap ping/pong
      cg.local.get(curPing);
      cg.local.set(tmp);
      cg.local.get(curPong);
      cg.local.set(curPing);
      cg.local.get(tmp);
      cg.local.set(curPong);

      // stride *= 2
      cg.local.get(stride);
      cg.i32.const(1);
      cg.i32.shl();
      cg.local.set(stride);
      cg.br(1);
      cg.end();
    }
    cg.end();

    // After Level 1, curPing has per-block prefix sums.
    // Skip Levels 2-3 if M <= 1.
    cg.local.get(M);
    cg.i32.const(1);
    cg.i32.gt_u();
    cg.if(cg.void);
    {
      // ==== LEVEL 2: Extract summaries + flat Kogge-Stone on M elements ====

      // Extract: for b = 0..M-1, summary[b] = ping[last element of block b]
      // last element of block b = min((b+1)*B, N) - 1
      cg.i32.const(0);
      cg.local.set(i); // reuse i as b
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(i);
        cg.local.get(M);
        cg.i32.ge_u();
        cg.br_if(0);

        for (let k = 0; k < numLeaves; k++) {
          const size = leafElemSizes[k];
          // dst = curSPing + leafOffset*M + b*size
          cg.local.get(curSPing);
          cg.i32.const(leafElemOffsets[k]);
          cg.local.get(M);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(i);
          cg.i32.const(size);
          cg.i32.mul();
          cg.i32.add();
          // src = curPing + leafOffset*N + (min((b+1)*B, N) - 1)*size
          // Compute min((b+1)*B, N): (b+1)*B
          cg.local.get(curPing);
          cg.i32.const(leafElemOffsets[k]);
          cg.local.get(NArg);
          cg.i32.mul();
          cg.i32.add();
          // index = min((i+1)*B, N) - 1
          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
          cg.i32.const(B);
          cg.i32.mul();
          cg.local.set(tmp); // tmp = (b+1)*B
          // min with N
          cg.local.get(tmp);
          cg.local.get(NArg);
          cg.i32.gt_u();
          cg.if(cg.void);
          cg.local.get(NArg);
          cg.local.set(tmp);
          cg.end();
          // index = tmp - 1
          cg.local.get(tmp);
          cg.i32.const(1);
          cg.i32.sub();
          cg.i32.const(size);
          cg.i32.mul();
          cg.i32.add();
          emitCopy(size);
        }

        cg.local.get(i);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);
        cg.br(1);
        cg.end();
      }
      cg.end();

      // Flat Kogge-Stone on summaries: stride = 1..M-1
      cg.i32.const(1);
      cg.local.set(stride);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(stride);
        cg.local.get(M);
        cg.i32.ge_u();
        cg.br_if(0);

        // Combine: for b = stride..M-1
        cg.local.get(stride);
        cg.local.set(i);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(i);
          cg.local.get(M);
          cg.i32.ge_u();
          cg.br_if(0);

          emitCombineAndCopy(
            gidx,
            (k) => {
              // a-leaf: curSPing + leafOffset*M + (b-stride)*size
              cg.local.get(curSPing);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(M);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.local.get(stride);
              cg.i32.sub();
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
            (k) => {
              // b-leaf: curSPing + leafOffset*M + b*size
              cg.local.get(curSPing);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(M);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
            (k) => {
              // output: curSPong + leafOffset*M + b*size
              cg.local.get(curSPong);
              cg.i32.const(leafElemOffsets[k]);
              cg.local.get(M);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(i);
              cg.i32.const(leafElemSizes[k]);
              cg.i32.mul();
              cg.i32.add();
            },
          );

          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(i);
          cg.br(1);
          cg.end();
        }
        cg.end();

        // Copy prefix: for b = 0..stride-1
        cg.i32.const(0);
        cg.local.set(i);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(i);
          cg.local.get(stride);
          cg.i32.ge_u();
          cg.br_if(0);
          for (let k = 0; k < numLeaves; k++) {
            const size = leafElemSizes[k];
            cg.local.get(curSPong);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(M);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(size);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(curSPing);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(M);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(size);
            cg.i32.mul();
            cg.i32.add();
            emitCopy(size);
          }
          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(i);
          cg.br(1);
          cg.end();
        }
        cg.end();

        // Swap summary ping/pong
        cg.local.get(curSPing);
        cg.local.set(tmp);
        cg.local.get(curSPong);
        cg.local.set(curSPing);
        cg.local.get(tmp);
        cg.local.set(curSPong);

        cg.local.get(stride);
        cg.i32.const(1);
        cg.i32.shl();
        cg.local.set(stride);
        cg.br(1);
        cg.end();
      }
      cg.end();

      // ==== LEVEL 3: Apply scanned summaries to blocks 1..M-1 ====
      // For i = B..N-1: pong[i] = fn(summary[(i/B)-1], ping[i])
      // For i = 0..B-1: pong[i] = ping[i]

      // Copy block 0 as-is
      for (let k = 0; k < numLeaves; k++) {
        const size = leafElemSizes[k];
        // dst = curPong + leafOffset*N
        cg.local.get(curPong);
        cg.i32.const(leafElemOffsets[k]);
        cg.local.get(NArg);
        cg.i32.mul();
        cg.i32.add();
        // src = curPing + leafOffset*N
        cg.local.get(curPing);
        cg.i32.const(leafElemOffsets[k]);
        cg.local.get(NArg);
        cg.i32.mul();
        cg.i32.add();
        // min(B, N) * size
        cg.i32.const(B);
        cg.local.get(NArg);
        cg.i32.gt_u();
        cg.if(cg.i32);
        cg.local.get(NArg);
        cg.else();
        cg.i32.const(B);
        cg.end();
        cg.i32.const(size);
        cg.i32.mul();
        cg.memory.copy();
      }

      // Apply summaries to remaining elements
      cg.i32.const(B);
      cg.local.set(i);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(i);
        cg.local.get(NArg);
        cg.i32.ge_u();
        cg.br_if(0);

        // block_idx = i / B; summary index = block_idx - 1
        emitCombineAndCopy(
          gidx,
          (k) => {
            // a-leaf: curSPing + leafOffset*M + (i/B - 1)*size
            cg.local.get(curSPing);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(M);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(B);
            cg.i32.div_u();
            cg.i32.const(1);
            cg.i32.sub();
            cg.i32.const(leafElemSizes[k]);
            cg.i32.mul();
            cg.i32.add();
          },
          (k) => {
            // b-leaf: curPing + leafOffset*N + i*size
            cg.local.get(curPing);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(NArg);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(leafElemSizes[k]);
            cg.i32.mul();
            cg.i32.add();
          },
          (k) => {
            // output: curPong + leafOffset*N + i*size
            cg.local.get(curPong);
            cg.i32.const(leafElemOffsets[k]);
            cg.local.get(NArg);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(i);
            cg.i32.const(leafElemSizes[k]);
            cg.i32.mul();
            cg.i32.add();
          },
        );

        cg.local.get(i);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);
        cg.br(1);
        cg.end();
      }
      cg.end();

      // Swap ping/pong so curPing has the final result
      cg.local.get(curPing);
      cg.local.set(tmp);
      cg.local.get(curPong);
      cg.local.set(curPing);
      cg.local.get(tmp);
      cg.local.set(curPong);
    }
    cg.end(); // end if M > 1

    // ---- Copy results to output ----
    if (reverse) {
      const j = cg.local.declare(cg.i32);
      for (let k = 0; k < numLeaves; k++) {
        const bw = leafElemSizes[k];
        cg.i32.const(0);
        cg.local.set(j);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(j);
          cg.local.get(NArg);
          cg.i32.ge_u();
          cg.br_if(0);
          cg.local.get(outputsBase + k);
          cg.local.get(j);
          cg.i32.const(bw);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(curPing);
          cg.i32.const(leafElemOffsets[k]);
          cg.local.get(NArg);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(NArg);
          cg.i32.const(1);
          cg.i32.sub();
          cg.local.get(j);
          cg.i32.sub();
          cg.i32.const(bw);
          cg.i32.mul();
          cg.i32.add();
          emitCopy(bw);
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(j);
          cg.br(1);
          cg.end();
        }
        cg.end();
      }
    } else {
      for (let k = 0; k < numLeaves; k++) {
        cg.local.get(outputsBase + k);
        cg.local.get(curPing);
        cg.i32.const(leafElemOffsets[k]);
        cg.local.get(NArg);
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(NArg);
        cg.i32.const(leafElemSizes[k]);
        cg.i32.mul();
        cg.memory.copy();
      }
    }
  });

  cg.export(mainFunc, "blocked_assoc_scan");
  const bytes = cg.finish();
  return { bytes, wat: traceEnabled ? cg.toWat() : undefined };
}

export function getScanRoutineInfo(routine: Routine): ScanRoutineInfo | null {
  const routineName = routine.name as Routines;
  const isF64 = routine.type.inputDtypes[0] === DType.Float64;
  const dtype: "f32" | "f64" = isF64 ? "f64" : "f32";

  if (routineName === Routines.Cholesky) {
    const inputShape = routine.type.inputShapes[0];
    const n = inputShape[inputShape.length - 1];
    return {
      routine: routineName,
      exportName: "cholesky",
      numParams: 2,
      dtype,
      sizeParams: [n],
    };
  } else if (routineName === Routines.Sort) {
    const inputShape = routine.type.inputShapes[0];
    const n = inputShape[inputShape.length - 1];
    return {
      routine: routineName,
      exportName: "sort",
      numParams: 2,
      dtype,
      sizeParams: [n],
    };
  } else if (routineName === Routines.TriangularSolve) {
    const aShape = routine.type.inputShapes[0];
    const bShape = routine.type.inputShapes[1];
    const n = aShape[aShape.length - 1];
    const batchRows = bShape[bShape.length - 1];
    return {
      routine: routineName,
      exportName: "triangular_solve",
      numParams: 3,
      dtype,
      sizeParams: [n, batchRows],
      unitDiagonal: routine.params?.unitDiagonal ?? false,
      lower: false,
    };
  } else if (routineName === Routines.LU) {
    const inputShape = routine.type.inputShapes[0];
    const m = inputShape[inputShape.length - 2];
    const n = inputShape[inputShape.length - 1];
    return {
      routine: routineName,
      exportName: "lu",
      numParams: 4,
      dtype,
      sizeParams: [m, n],
    };
  } else if (routineName === Routines.Argsort) {
    const inputShape = routine.type.inputShapes[0];
    const n = inputShape[inputShape.length - 1];
    return {
      routine: routineName,
      exportName: "argsort",
      numParams: 4,
      dtype,
      sizeParams: [n],
    };
  } else if (routineName === Routines.QR) {
    const inputShape = routine.type.inputShapes[0];
    const m = inputShape[inputShape.length - 2];
    const n = inputShape[inputShape.length - 1];
    return {
      routine: routineName,
      exportName: "qr",
      numParams: 4,
      dtype,
      sizeParams: [m, n],
    };
  }
  return null;
}
