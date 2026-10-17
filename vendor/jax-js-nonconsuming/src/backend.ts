// Shared interfaces and code for the low-level backend API.
//
// Think of each backend as a _connector_ to a specific hardware or software
// implementation of the array API.
//
// Backends do not share any of the built-in operational semantics of the
// library. This is a private API. You must allocate and free buffers manually,
// and dispatch happens on the level of each shader. Buffers are untyped.
//
// The "cpu" backend is very slow and used for debugging. Prefer "wasm".

import { AluOp, DType, Kernel } from "./alu";
import { CpuBackend } from "./backend/cpu";
import { WasmBackend } from "./backend/wasm";
import type { WebGPUBackend } from "./backend/webgpu";
import { Routine, Routines } from "./routine";

export type Device = "cpu" | "wasm" | "webgpu" | "webgl";

/** Hardware and environment capabilities exposed by each backend. */
export interface BackendCapabilities {
  /** WebGPU: true if shader-f32-atomic-add extension is available. */
  readonly atomicF32Add: boolean;
  /** WebGPU: true if shader-f16 extension is available. */
  readonly shaderF16: boolean;
  /** WebGPU: true if subgroups extension is available. */
  readonly subgroups?: boolean;
  /** Wasm: true if crossOriginIsolated (SharedArrayBuffer available). */
  readonly sharedMemory: boolean;
  /** Whether the backend supports multi-output kernel dispatch. */
  readonly multiOutputKernel: boolean;
  /** WebGPU: maximum workgroup size along X (from device.limits). */
  readonly maxComputeWorkgroupSizeX?: number;
  /** WebGPU: max threads per workgroup (from device.limits). */
  readonly maxComputeInvocationsPerWorkgroup?: number;
  /** WebGPU: max shared memory bytes per workgroup (from device.limits). */
  readonly maxComputeWorkgroupStorageSize?: number;
  /** WebGPU: adapter architecture string (e.g. "gen-9", "xe-lpg"). */
  readonly adapterArchitecture?: string;
  /** WebGPU: adapter vendor string (e.g. "intel", "nvidia", "apple"). */
  readonly adapterVendor?: string;
  /** WebGPU: true if timestamp-query feature is available. */
  readonly timestampQuery?: boolean;

  /** Measured or injected memory bandwidth in GB/s. Absent before calibration in the current runtime path. */
  readonly bandwidthGBs?: number;
  /** Measured or injected compute throughput in TFLOPS. Absent before calibration in the current runtime path. */
  readonly tflops?: number;
  /** Measured or injected base dispatch overhead in microseconds. Absent before calibration in the current runtime path. */
  readonly dispatchOverheadUs?: number;
  /**
   * Measured or injected per-thread register budget (words) before spill likely
   * begins. Absent before calibration in the current runtime path.
   */
  readonly rOptWords?: number;
  /** Measured or injected relative cost multiplier for workgroup barriers. */
  readonly barrierCostFactor?: number;

  /**
   * Whether microbenchmark calibration has been applied to this backend.
   * When false, the tuner falls back to conservative built-in defaults while
   * still using the real hardware limits and feature flags above.
   */
  readonly calibrated?: boolean;
}

/**
 * A stable fingerprint of immutable device capabilities used in cache keys.
 * Includes only structural limits that don't change across calibration runs.
 * Optional coarse classification is included only when present.
 */
export function staticCapabilityFingerprint(caps: BackendCapabilities): string {
  return [
    caps.maxComputeWorkgroupSizeX ?? 0,
    caps.maxComputeInvocationsPerWorkgroup ?? 0,
    caps.maxComputeWorkgroupStorageSize ?? 0,
    caps.shaderF16 ? 1 : 0,
    caps.subgroups ? 1 : 0,
    caps.atomicF32Add ? 1 : 0,
  ].join(",");
}

export const devices: Device[] = ["cpu", "wasm", "webgpu", "webgl"];

// ── Code capture API (Phase A0) ──────────────────────────────────────────
// Registers a callback invoked on every compiled code unit (WGSL/WASM).
// Disabled by default (null). Zero overhead when not installed.

/** A compiled code unit captured by `setCodeCapture`. */
export type CodeCaptureEntry = {
  backend: "webgpu" | "wasm" | "cpu" | "webgl";
  kind:
    | "kernel"
    | "mega-module"
    | "scan"
    | "assoc-scan"
    | "block-map"
    | "routine"
    | "program";
  label?: string;
  /** WGSL source (WebGPU) or WAT source (WASM). */
  code?: string;
  /** WebGPU only: workgroup size of the compute shader. */
  workgroupSize?: [number, number, number];
  /** Structured metadata (both backends). */
  metadata?: {
    size?: number;
    simd?: boolean;
    numInputs?: number;
    numOutputs?: number;
    dtype?: number | string;
    reduction?: boolean;
    numSteps?: number;
    numKernels?: number;
    byteLength?: number;
    [key: string]: unknown;
  };
};

let codeCaptureCallback: ((entry: CodeCaptureEntry) => void) | null = null;

/**
 * Register a callback invoked on every compiled code unit (WGSL shader or WASM
 * module). Pass `null` to disable. Disabled by default (zero overhead).
 */
export function setCodeCapture(
  cb: ((entry: CodeCaptureEntry) => void) | null,
): void {
  codeCaptureCallback = cb;
}

/** Internal: invoke the code capture callback if installed. */
export function _emitCodeCapture(entry: CodeCaptureEntry): void {
  if (codeCaptureCallback) codeCaptureCallback(entry);
}

/** Internal: check if code capture is currently enabled. */
export function _isCodeCaptureEnabled(): boolean {
  return codeCaptureCallback !== null;
}

const initializedBackends = new Map<Device, Backend>();

// Default backends, initialized at startup.
initializedBackends.set("cpu", new CpuBackend());
if (typeof WebAssembly !== "undefined") {
  initializedBackends.set("wasm", new WasmBackend());
}

let defaultBackend: Device = initializedBackends.has("wasm") ? "wasm" : "cpu";

/** Configure the default device for arrays. */
export function defaultDevice(device?: Device): Device {
  if (device !== undefined) {
    if (initializedBackends.has(device)) {
      defaultBackend = device;
    } else {
      throw new Error(`Backend not initialized: ${device}`);
    }
  }
  return defaultBackend;
}

/**
 * Initialize `jax-js` library backends.
 *
 * By default, this will initialize all available backends. If one or more
 * backends is provided, only attempt to initialize those. Returns a list of
 * available backends.
 */
export async function init(...devicesToInit: Device[]): Promise<Device[]> {
  if (devicesToInit.length === 0) {
    devicesToInit = devices;
  }
  const requestedDevices = new Set(devicesToInit);
  const promises: Promise<void>[] = [];
  for (const device of requestedDevices) {
    if (!initializedBackends.has(device)) {
      promises.push(
        (async () => {
          const backend = await createBackend(device);
          if (backend) {
            initializedBackends.set(device, backend);
          }
        })(),
      );
    }
  }
  await Promise.all(promises);

  // WebGPU calibration must happen on an async boundary before sync JIT
  // compilation becomes reachable. `init("webgpu")` is that boundary.
  if (requestedDevices.has("webgpu") && initializedBackends.has("webgpu")) {
    await _ensureCalibrated();
  }

  return Array.from(initializedBackends.keys());
}

/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(device: Device): Promise<Backend | null> {
  if (device === "cpu") {
    return new CpuBackend();
  } else if (device === "wasm") {
    if (typeof WebAssembly === "undefined") return null; // WebAssembly is not available.
    return new WasmBackend();
  } else if (device === "webgpu") {
    if (!navigator.gpu) return null; // WebGPU is not available.
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) return null;

    const { WebGPUBackend } = await import("./backend/webgpu");

    const importantLimits: Exclude<keyof GPUSupportedLimits, "__brand">[] = [
      "maxBufferSize",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", // All of our workgroups use X or Y.
      "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ",
      "maxComputeWorkgroupStorageSize",
      "maxComputeWorkgroupsPerDimension", // Grid size limited to 65535 due to AMD storage in u16.
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
      "maxStorageTexturesPerShaderStage",
    ];

    const requestedFeatures: GPUFeatureName[] = [
      "shader-f16", // "enable f16;" feature support for f16 data type
      "timestamp-query", // Performance timing queries.
      "subgroups" as GPUFeatureName, // SIMD-width operations (shuffle, reduce within wave).
    ];

    try {
      const device = await adapter.requestDevice({
        requiredLimits: Object.fromEntries(
          importantLimits.map((limit) => [limit, adapter.limits[limit]]),
        ),
        requiredFeatures: requestedFeatures.filter((feature) =>
          adapter.features.has(feature),
        ),
      });
      return new WebGPUBackend(device);
    } catch (error) {
      // Browsers can throw a TypeError if features are not supported by the
      // adapter, or limits have not been set properly.
      console.error("Unexpected error requesting WebGPU device:", error);
      return null;
    }
  } else if (device === "webgl") {
    if (typeof WebGL2RenderingContext === "undefined") return null; // WebGL2 is not available.
    const canvas = new OffscreenCanvas(0, 0);
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true,
    });
    if (!gl) return null;
    // Required extension for rendering to float textures.
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    const { WebGLBackend } = await import("./backend/webgl");
    return new WebGLBackend(gl);
  } else {
    device satisfies never;
    throw new Error(`Backend not found: ${device}`);
  }
}

/** Retrieve a backend that has been initialized. */
export function getBackend(device?: Device): Backend {
  device = device ?? defaultBackend;
  const backend = initializedBackends.get(device);
  if (!backend) {
    throw new Error(`${device} backend not ready, call init() first`);
  }
  return backend;
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
  /** The name of the backend as a string. */
  readonly type: Device;

  /** Maximum number of arguments per dispatched kernel. */
  readonly maxArgs: number;

  /** Hardware and environment capabilities. */
  readonly capabilities: BackendCapabilities;

  /** Number of live backend slots (allocated buffers with refcount > 0). */
  slotCount(): number;

  /** Allocate a new slot with reference count 1. */
  malloc(size: number, initialData?: Uint8Array): Slot;

  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;

  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;

  /** Read a range of bytes from a buffer. */
  read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>>;

  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): Uint8Array<ArrayBuffer>;

  /** Copy bytes between two device buffers. */
  copyBufferToBuffer(
    src: Slot,
    srcOffset: number,
    dst: Slot,
    dstOffset: number,
    size: number,
  ): void;

  /** Prepare an expression to be executed later. */
  prepareKernel(kernel: Kernel): Promise<Executable>;

  /** Prepare an expression to be executed later, blocking variant. */
  prepareKernelSync(kernel: Kernel): Executable;

  /** Prepare an advanced routine to be executed later. */
  prepareRoutine(routine: Routine): Promise<Executable>;

  /** Prepare an advanced routine to be executed later, blocking variant. */
  prepareRoutineSync(routine: Routine): Executable;

  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immediately, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   *
   * @param dynamicParams - Optional resolved runtime parameters for
   * parameterized kernels (symbolic dims). For WASM, appended as extra i32
   * function args. For WebGPU, written to a uniform buffer.
   */
  dispatch(
    exe: Executable,
    inputs: Slot[],
    outputs: Slot[],
    dynamicParams?: number[],
  ): void;

  /**
   * Optional: prepare the backend's buffer pool for an upcoming JitProgram
   * execution. The pool should evict entries whose sizes aren't in
   * `hints.mallocSizes` and cap total retained bytes at `hints.peakBytes`.
   * This ensures physical peak memory never exceeds peak live memory.
   */
  configurePool?(hints: {
    readonly peakBytes: number;
    readonly mallocSizes: ReadonlySet<number>;
  }): void;

  /**
   * Optional: begin batching dispatches into a single command submission.
   * When active, `dispatch()` accumulates into a shared command encoder
   * instead of submitting individually. Call `endBatch()` to submit.
   * No-op for backends without dispatch submission overhead (WASM, CPU).
   */
  beginBatch?(): void;

  /**
   * Optional: submit all batched dispatches accumulated since `beginBatch()`.
   * Must be called after `beginBatch()` to flush the accumulated commands.
   */
  endBatch?(): void;

  /**
   * Optional: submit the current batch encoder mid-batch without changing depth.
   * Materializes all pending GPU commands so that subsequent non-batch operations
   * (e.g., body program execution) can read the results.
   * No-op when no batch is active.
   */
  flushBatch?(): void;

  /**
   * Optional: lightweight GPU fence — resolves when all previously submitted
   * work completes. Uses `queue.onSubmittedWorkDone()` on WebGPU.
   * Backends without async dispatch (CPU, WASM) resolve immediately.
   */
  fence?(): Promise<void>;

  /**
   * Optional: dispatch a scatter-add operation.
   * Accumulates `updates` into `target` (already copied to output) at
   * positions given by `indices` along `axis`.
   *
   * For duplicate indices, values are summed (order-independent).
   * The output slot must already contain a copy of the target data.
   */
  dispatchScatterAdd?(
    output: Slot,
    indices: Slot,
    updates: Slot,
    axis: number,
    targetShape: number[],
    updatesLen: number,
    dtype: DType,
  ): void;

  /**
   * Optional: reverse a buffer along a single axis.
   * Copies `axisSize` contiguous slices of `innerBytes` in reverse order
   * from `input` to `output`.
   */
  reverseBuffer?(
    input: Slot,
    output: Slot,
    axisSize: number,
    innerBytes: number,
    dtype: DType,
  ): void;

  /**
   * Optional: Decoupled Fallback prefix scan — single-dispatch O(N) scan.
   * Uses inter-workgroup atomics with bounded spin + work-stealing fallback.
   * WebGPU only. Phase 1: scalar binary ops (add/mul/min/max) on f32.
   */
  dispatchDecoupledFallbackScan?(
    input: Slot,
    output: Slot,
    N: number,
    op: AluOp,
    dtype: DType,
    blockSize: number,
  ): void;

  /**
   * Optional: begin GPU timestamp profiling.
   * Subsequent compute passes will record per-pass timestamps.
   * Call `stopProfiling()` to retrieve timing data.
   */
  startProfiling?(): void;

  /**
   * Optional: stop GPU timestamp profiling and return timing results.
   * Resolves timestamps, reads them back, and returns per-pass GPU timing.
   */
  stopProfiling?(): Promise<GpuTimingResult>;
}

/** GPU timing data returned by `profileGpu`. */
export interface GpuTimingResult {
  /** Per-compute-pass GPU duration (nanosecond precision). */
  passes: {
    durationMs: number;
    /** Dispatch grid dimensions (if available). */
    grid?: [number, number];
    /** Workgroup size (if available). */
    workgroupSize?: number | number[];
    /** Shader label or hash (if available). */
    label?: string;
  }[];
  /** Wall-clock GPU time from first pass start to last pass end. */
  totalMs: number;
  /** True if more compute passes were dispatched than the profiling buffer could record. */
  truncated: boolean;
}

export class Executable<T = any> {
  constructor(
    /** The `Kernel` or `Routine` that was prepared. */
    readonly source: Kernel | Routine,
    /** Extra data specific to the backend running this executable. */
    readonly data: T,
  ) {}
}

export class SlotError extends Error {
  constructor(slot: Slot) {
    super(`Used a buffer that is invalid or already freed: ${slot}`);
  }
}

export class UnsupportedOpError extends Error {
  constructor(op: AluOp | null, dtype: DType, device: Device, arg?: any) {
    let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
    if (arg !== undefined) msg += ` with arg ${JSON.stringify(arg)}`;
    super(msg);
  }
}

export class UnsupportedRoutineError extends Error {
  constructor(name: Routines, device: Device) {
    super(`routine '${name}' is not supported in ${device} backend`);
  }
}

// Backend-specific functions are below this line.

/**
 * If the WebGPU backend has been initialized, return the `GPUDevice` that this
 * backend runs on. This is useful for sharing buffers.
 */
export function getWebGPUDevice(): GPUDevice {
  const backend = initializedBackends.get("webgpu") as
    | WebGPUBackend
    | undefined;
  if (!backend) {
    throw new Error(
      "WebGPU backend not initialized, call init('webgpu') first",
    );
  }
  return backend.device;
}

// ── Microbenchmark calibration (P10) ─────────────────────────────────────

/**
 * Calibration state for the WebGPU backend.
 * - "pending": not yet calibrated, will run during init("webgpu")
 * - "running": calibration in progress
 * - "done": calibration complete, belief state frozen
 * - "off": calibration disabled
 */
let _calibrationState: "pending" | "running" | "done" | "off" = "pending";

/** Internal test hook for forcing calibration state. Not a stable public API. */
export function _setCalibrationState(state: "pending" | "off"): void {
  _calibrationState = state;
  // Reset the calibrated flag on the WebGPU backend for test isolation.
  const backend = initializedBackends.get("webgpu") as
    | import("./backend/webgpu").WebGPUBackend
    | undefined;
  if (backend && backend.capabilities.calibrated) {
    backend.resetCalibration();
  }
}

/**
 * Run microbenchmark calibration on the WebGPU backend. Measures fundamental
 * hardware characteristics (bandwidth, TFLOPS, dispatch overhead, register
 * budget) and updates the backend capabilities with empirical values.
 *
 * This runs automatically during `init("webgpu")` when calibration is still
 * pending. Call it explicitly when you want deterministic control over when
 * calibration happens.
 */
export async function calibrateGpu(): Promise<void> {
  if (_calibrationState === "done" || _calibrationState === "off") return;

  const backend = initializedBackends.get("webgpu") as
    | WebGPUBackend
    | undefined;
  if (!backend) {
    throw new Error(
      "WebGPU backend not initialized, call init('webgpu') first",
    );
  }

  _calibrationState = "running";
  try {
    const { runMicrobenchmarks } = await import("./microbench");
    const beliefState = await runMicrobenchmarks(backend.device);
    backend.applyCalibration(beliefState);
    _calibrationState = "done";
  } catch (e) {
    // Calibration failure is non-fatal — keep real hardware limits and use
    // conservative built-in performance defaults.
    console.warn(
      "[microbench] calibration failed, using conservative defaults:",
      e,
    );
    _calibrationState = "done";
  }
}

/**
 * Ensure calibration has run before sync WebGPU JIT compilation becomes
 * reachable. Called internally from the async WebGPU initialization path.
 */
export async function _ensureCalibrated(): Promise<void> {
  if (_calibrationState === "pending") {
    await calibrateGpu();
  }
}
