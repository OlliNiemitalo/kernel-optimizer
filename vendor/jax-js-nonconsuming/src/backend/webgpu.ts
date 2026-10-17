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
  type GpuTimingResult,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import type { IPendingExecute } from "../frontend/array.js";
import type { JitId, JitStep } from "../frontend/jit";
import { Routine } from "../routine";
import { isSymbolicSize } from "../shape";
import { tuneNullopt, tuneWebgpu, type WebGPUTuneResult } from "../tuner";
import {
  DEBUG,
  findPow2,
  FpHash,
  mapSetUnion,
  prod,
  range,
  strip1,
} from "../utils";
import {
  blockMapFusedShaderSource,
  type BlockMapShaderParams,
} from "./webgpu/block-map";
import { erfSrc, threefrySrc } from "./webgpu/builtins";
import {
  calculateGrid,
  castSaturateWgsl,
  constToWgsl,
  dtypeToWgsl,
  gridOffsetY,
  ShaderInfo,
  withNanInfHeader,
} from "./webgpu/codegen";
import {
  type ArenaSlab,
  type ArenaSlabEntry,
  buildConflictGraphAndColor,
  type ConstSlabEntry,
  type TapeDUS,
  type TapeOp,
  type TapeReverse,
  type TapeScan,
  type TapeScatterAdd,
  type WebGPUCommandTape,
} from "./webgpu/command-tape";
import {
  type DFScanDtype,
  type DFScanOp,
  generateDecoupledFallbackScanShader,
} from "./webgpu/decoupled-fallback-scan";
import { SyncReader } from "./webgpu/reader";
import { createRoutineShader } from "./webgpu/routines";
import {
  createAllIterationsOffsetsBuffer,
  type ScanBindingInfo,
  wrapRoutineForScan,
} from "./webgpu/scan-wrapper";
import { maybeAcquireTracingSlot, recordTrace } from "./webgpu/tracing";
import { createWgslGen, type ResolveGlobalIndex } from "./webgpu/wgsl-gen";

interface ShaderDispatch extends ShaderInfo {
  pipeline: GPUComputePipeline; // Compiled pipeline for the shader.
}

// ---------------------------------------------------------------------------
// Types for WebGPU native scan (multi-kernel shader)
// ---------------------------------------------------------------------------

/**
 * Where a WebGPU scan Y output comes from.
 *
 * All three scan paths classify Y outputs into these categories via
 * `classifyWebGPUYSources()` and store the result in the prepared struct.
 * Executors dispatch Y stacking from source type, not raw JitIds:
 * - compiled-loop: carry-snapshot/xs → writeback phase; carry-live/internal → inline step writes
 * - preencoded-routine: carry-live only (carry → ysStacked copy)
 * - preencoded-multi-step: all types (source-based buffer resolution per iteration)
 */
export type WebGPUYOutputSource = { yi: number } & (
  | { type: "carry-snapshot"; carryIdx: number }
  | { type: "carry-live"; carryIdx: number }
  | { type: "xs"; xsIdx: number }
  | { type: "internal"; internalIdx: number }
);

export interface NativeScanMultiStep {
  /** The kernel to execute. */
  kernel: Kernel;
  /** Input mapping: indices into [consts, carry, xs, internals] flattened. */
  inputs: number[];
  /** Which carry slot this kernel writes to (0..numCarry-1), or -1. */
  outputCarryIdx: number;
  /** Which internal local this step defines (0..numInternal-1), or -1. */
  outputInternalIdx: number;
  /** Which Y outputs this kernel writes to directly. */
  outputYIdxs?: number[];
  /** Size of output in elements (not bytes). */
  outputSize: number;
}

/** Parameters for multi-kernel native scan execution on WebGPU. */
export interface NativeScanMultiParams {
  length: number;
  numConsts: number;
  constSizes: number[];
  constDtypes: DType[];
  numCarry: number;
  carrySizes: number[];
  carryDtypes: DType[];
  numX: number;
  xsStrides: number[];
  xsDtypes: DType[];
  numY: number;
  ysStrides: number[];
  ysDtypes: DType[];
  /** Elements per Y output (explicit, instead of relying on source structure). */
  yElemCounts: number[];
  steps: NativeScanMultiStep[];
  reverse?: boolean;
  /** Declarative passthrough Y-output source descriptors (not handled by steps). */
  yOutputSources: WebGPUYOutputSource[];
  /** Number of internal intermediate locals (from steps with internal deps). */
  numInternal: number;
  /** Per-internal element count (typed, not bytes). */
  internalElemCounts: number[];
  /** Per-internal dtype. */
  internalDtypes: DType[];
}

// ---------------------------------------------------------------------------
// Types for WebGPU preencoded-routine scan (P4)
// ---------------------------------------------------------------------------

/** Parameters for preencoded scan execution on WebGPU (routine body like matmul). */
export interface PreencodedScanParams {
  /** Number of scan iterations. */
  length: number;
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Strides (in ELEMENTS) along axis 0 for each xs input. */
  xsElemStrides: number[];
  /** Strides (in ELEMENTS) along axis 0 for each stacked y output. */
  ysElemStrides: number[];
  /** The prepared routine executable for the body. */
  bodyRoutine: Executable<ShaderDispatch[]>;
  /** Number of carry arrays. */
  numCarry: number;
  /** Number of xs inputs. */
  numX: number;
  /** Number of ys outputs. */
  numY: number;
  /** Number of const inputs (bound before carry). */
  numConsts: number;
  /** Whether to scan in reverse order. */
  reverse?: boolean;
  /** For each routine input binding i, the body jaxpr JitId, used to classify as const/carry/xs. */
  routineInputJitIds: number[];
  /** For each routine output binding i, the body output index. */
  routineOutputJitIds: number[];
  /** Classified Y-output sources (all carry-live for preencoded-routine). */
  ySources: WebGPUYOutputSource[];
}

/** Prepared preencoded scan with wrapped shaders and offset buffer. */
export interface PreparedPreencodedScan {
  params: PreencodedScanParams;
  /** Shaders with uniform offset support. */
  wrappedShaders: ShaderDispatch[];
  /** GPU buffer containing all iteration offsets. */
  offsetBuffer: GPUBuffer;
  /** Alignment of each iteration's offset data in the buffer. */
  offsetAlignment: number;
  /** Per-carry copy strategy for ys stacking (true = use shader copy). */
  copyUsesShader: boolean[];
  /** Bind group layout for the uniform offset group (with dynamic offset). */
  uniformLayout: GPUBindGroupLayout;
}

// ---------------------------------------------------------------------------
// Types for WebGPU preencoded multi-step scan (Phase 2)
// ---------------------------------------------------------------------------

export type WebGPUCarryOutputSource =
  | { type: "internal"; internalIdx: number }
  | { type: "carry-passthrough"; carryIdx: number }
  | { type: "const"; constIdx: number }
  | { type: "xs"; xIdx: number };

/** A single prepared step in a multi-step preencoded scan body. */
export interface PreencodedMultiStepEntry {
  /** The compiled pipeline(s) for this execute step. */
  dispatches: ShaderDispatch[];
  /** Input JitIds for this step (used to build bind groups). */
  inputJitIds: number[];
  /** Output JitIds for this step. */
  outputJitIds: number[];
  /** Per-step offset buffer for xs offsets (null if step has no xs inputs). */
  offsetBuffer: GPUBuffer | null;
  /** Alignment between iterations in the offset buffer (0 if no offsets). */
  offsetAlignment: number;
}

/** Prepared preencoded multi-step scan with per-step pipelines and layout. */
export interface PreparedPreencodedMultiStep {
  /** Scan iteration count. */
  length: number;
  /** One entry per execute step in the body program. */
  stepEntries: PreencodedMultiStepEntry[];
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Sizes of each internal buffer in bytes (pre-allocated scratch). */
  internalSizes: number[];
  /** Map from body JitId to internal buffer index. */
  internalMap: Map<number, number>;
  /** Number of carry, const, xs, ys arrays. */
  numCarry: number;
  numConsts: number;
  numX: number;
  numY: number;
  /** Whether to scan in reverse order. */
  reverse: boolean;
  /** Per-ys copy strategy for ys stacking (true = use shader copy for non-4b-aligned). */
  copyUsesShader: boolean[];
  /** Bind group layout for the uniform offset group (with dynamic offset). */
  uniformLayout: GPUBindGroupLayout;
  /** Classified carry-output sources for carry-copy loop. */
  carrySources: WebGPUCarryOutputSource[];
  /** Classified Y-output sources from the shared vocabulary. */
  ySources: WebGPUYOutputSource[];
  /** Per-ys byte sizes for ys stacking. */
  ysSizes: number[];
}

const COPY_WORKGROUP_SIZE = 64;

const COPY_SHADER_CODE = String.raw`
struct CopyParams {
  srcOffset: u32,
  dstOffset: u32,
  size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(1) @binding(0) var<uniform> params: CopyParams;

fn byte_mask(n: u32) -> u32 {
  if (n >= 4u) { return 0xffffffffu; }
  return (1u << (n * 8u)) - 1u;
}

fn load_unaligned(offset: u32) -> u32 {
  let word = offset >> 2u;
  let shift = (offset & 3u) * 8u;
  if (shift == 0u) {
    return src[word];
  }
  let low = src[word];
  let high = src[word + 1u];
  return (low >> shift) | (high << (32u - shift));
}

@compute @workgroup_size(${COPY_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let gid = id.x + id.y * ${gridOffsetY}u;
  let firstDstWord = params.dstOffset >> 2u;
  let wordIdx = firstDstWord + gid;
  let lastDstWord = (params.dstOffset + params.size + 3u) >> 2u;
  if (wordIdx >= lastDstWord) { return; }

  let wordByteStart = wordIdx * 4u;
  let copyStart = max(params.dstOffset, wordByteStart);
  let copyEnd = min(params.dstOffset + params.size, wordByteStart + 4u);
  let nbytes = copyEnd - copyStart;
  let srcByteOff = params.srcOffset + (copyStart - params.dstOffset);
  let value = load_unaligned(srcByteOff);

  if (nbytes == 4u) {
    dst[wordIdx] = value;
  } else {
    let shift = (copyStart & 3u) * 8u;
    let mask = byte_mask(nbytes) << shift;
    let cur = dst[wordIdx];
    dst[wordIdx] = (cur & ~mask) | ((value << shift) & mask);
  }
}
`.trim();

// --- GPU dispatch counter (module-level for pipelineSubmit access) ---
let _dispatchCount = 0;

// --- GPU timestamp profiling state (passed explicitly from backend instance) ---

/** Per-pass metadata recorded during detailed profiling. */
interface _ProfilingPassMeta {
  grid?: [number, number];
  workgroupSize?: number | number[];
  label?: string;
}

interface _ProfilingState {
  querySet: GPUQuerySet;
  passIdx: number;
  overflow: boolean;
  /** Per-pass metadata — populated at each dispatch. */
  passMeta: _ProfilingPassMeta[];
}
const _MAX_PROFILING_PASSES = 4096;

/** Return timestampWrites descriptor for the current pass, or undefined. */
function _profilingTimestampWrites(
  profiling: _ProfilingState | null,
  meta?: _ProfilingPassMeta,
): GPUComputePassTimestampWrites | undefined {
  if (!profiling) return undefined;
  if (profiling.passIdx >= _MAX_PROFILING_PASSES) {
    profiling.overflow = true;
    return undefined;
  }
  const idx = profiling.passIdx++;
  profiling.passMeta.push(meta ?? {});
  return {
    querySet: profiling.querySet,
    beginningOfPassWriteIndex: idx * 2,
    endOfPassWriteIndex: idx * 2 + 1,
  };
}

/** Begin a compute pass with optional profiling timestamps and metadata. */
function _beginComputePass(
  encoder: GPUCommandEncoder,
  profiling: _ProfilingState | null = null,
  meta?: _ProfilingPassMeta,
  fallbackTsw?: GPUComputePassTimestampWrites,
): GPUComputePassEncoder {
  const tsw = _profilingTimestampWrites(profiling, meta) ?? fallbackTsw;
  return encoder.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
}

/** Implementation of `Backend` that uses WebGPU in browsers. */

class _PendingCommandTape implements IPendingExecute {
  prepared: any = null;
  submitted = false;
  #promise: Promise<void> | null = null;
  #rc = 1;

  constructor(
    readonly backend: WebGPUBackend,
    readonly tape: WebGPUCommandTape,
    readonly inputSlots: Slot[],
    readonly outputSlots: Slot[],
    readonly outputTableIdxs: number[],
  ) {}

  updateRc(delta: number) {
    if (this.#rc <= 0)
      throw new Error("internal: PendingCommandTape used rc<=0");
    this.#rc += delta;
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      const promises: Promise<any>[] = [];
      for (const op of this.tape.ops) {
        if (op.type === "dispatch") {
          promises.push(this.backend.pipelines.prepare(op.dispatch.shader));
        } else if (op.type === "scan") {
          const scan = op.scan as any;
          if (scan.mode === "compiled-loop" && scan.nativeShader)
            promises.push(this.backend.pipelines.prepare(scan.nativeShader));
          if (scan.mode === "preencoded-multi-step") {
            if (scan.passShaders)
              scan.passShaders.forEach((s: any) =>
                promises.push(this.backend.pipelines.prepare(s)),
              );
            if (scan.stepPipelines)
              scan.stepPipelines.forEach((s: any) =>
                promises.push(this.backend.pipelines.prepare(s.shader)),
              );
            if (scan.yStackPipeline)
              promises.push(
                this.backend.pipelines.prepare(scan.yStackPipeline.shader),
              );
            if (scan.reductionPipeline)
              promises.push(
                this.backend.pipelines.prepare(scan.reductionPipeline.shader),
              );
          }
          if (scan.mode === "preencoded-routine") {
            if (scan.passShaders)
              scan.passShaders.forEach((s: any) =>
                promises.push(this.backend.pipelines.prepare(s)),
              );
          }
        }
      }
      await Promise.all(promises);
      this.prepared = true;
    })();
    return this.#promise;
  }

  prepareSync() {}

  submit() {
    if (this.submitted) return;
    this.submitted = true;
    if (this.#rc <= 0) return;

    this.backend._executeCommandTapeDeferred(
      this.tape,
      this.inputSlots,
      this.outputSlots,
      this.outputTableIdxs,
    );
  }
}

export class WebGPUBackend implements Backend {
  readonly type: Device = "webgpu";
  readonly maxArgs: number;
  readonly capabilities: BackendCapabilities;

  // Added dynamically by JIT calibration

  resetCalibration() {
    // Clear all calibrated metrics to prevent stale values from poisoning
    // cost model decisions across test isolation boundaries.
    const caps = this.capabilities as any;
    caps.calibrated = false;
    delete caps.bandwidthGBs;
    delete caps.tflops;
    delete caps.dispatchOverheadUs;
    delete caps.rOptWords;
    delete caps.barrierCostFactor;
  }

  applyCalibration(beliefState: any) {
    Object.assign(this.capabilities as any, beliefState);
    (this.capabilities as any).calibrated = true;
  }

  readonly pipelines: ShaderPipelineCache;
  readonly syncReader: SyncReader;
  readonly buffers: Map<
    Slot,
    {
      ref: number;
      size: number; // Refers to "true size" requested, less padding.
      buffer: GPUBuffer;
    }
  >;
  nextSlot: number;

  #cachedShaderMap = new Map<bigint, ShaderInfo>();
  #reusableZsb: GPUBuffer;
  #copyPipeline: GPUComputePipeline | null = null;

  /**
   * Buffer pool: recently-freed GPUBuffers indexed by padded byte size.
   * Avoids expensive device.createBuffer() / buffer.destroy() cycles for
   * same-size allocations, which are very common in JIT-compiled programs.
   *
   * **Peak-memory guarantee:** Before each JIT execution, `configurePool()`
   * evicts pool entries whose sizes won't be needed and caps total retained
   * bytes at the program's peak live bytes. This ensures physical peak memory
   * never exceeds what was already required during execution — the pool is
   * free from a peak-memory perspective.
   *
   * A per-size-class cap (MAX_POOL_PER_SIZE) limits redundant same-size
   * entries. MAX_POOL_BYTES_DEFAULT is a fallback byte budget used when
   * no JitProgram has configured the pool yet (e.g., in eager mode).
   */
  #bufferPool = new Map<number, GPUBuffer[]>();
  #poolBudgetBytes: number = 64 * 1024 * 1024; // default: 64 MB (eager mode)
  #poolCurrentBytes: number = 0;
  static readonly MAX_POOL_PER_SIZE = 4;
  /** Fallback budget for eager mode (no JitProgram to derive peak from). */
  static readonly MAX_POOL_BYTES_DEFAULT = 64 * 1024 * 1024; // 64 MB

  /**
   * Total bytes of GPU storage buffers currently allocated by jax-js
   * (live + pooled). Incremented on createBuffer, decremented on destroy.
   * Staging/read buffers are excluded since they're transient.
   */
  #gpuAllocatedBytes: number = 0;

  /**
   * High-water mark: the maximum value `#gpuAllocatedBytes` ever reached.
   * Useful for verifying that optimizations (fusion, recycling, pooling)
   * reduce peak GPU memory. Reset via `resetPeakGpuAllocatedBytes()`.
   */
  #gpuPeakBytes: number = 0;

  // --- Batch dispatch state ---
  // When non-null, dispatch() encodes into this shared encoder instead of
  // creating a new one per call. endBatch() submits and cleans up.
  #batchEncoder: GPUCommandEncoder | null = null;
  #batchDepth = 0;
  #batchUniformsToDestroy: GPUBuffer[] = [];
  #batchDeferredFreeBuffers: GPUBuffer[] = [];

  // --- GPU timestamp profiling state (T3/P9) ---
  #profiling: _ProfilingState | null = null;

  constructor(readonly device: GPUDevice) {
    if (DEBUG >= 3 && device.adapterInfo) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
    this.capabilities = {
      atomicF32Add: device.features.has(
        "shader-f32-atomic-add" as GPUFeatureName,
      ),
      shaderF16: device.features.has("shader-f16" as GPUFeatureName),
      subgroups: device.features.has("subgroups" as GPUFeatureName),
      sharedMemory: false,
      multiOutputKernel: true,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
      adapterArchitecture: device.adapterInfo?.architecture,
      adapterVendor: device.adapterInfo?.vendor,
      timestampQuery: device.features.has("timestamp-query"),
    };
    this.pipelines = new ShaderPipelineCache(device);
    this.syncReader = new SyncReader(device);
    this.buffers = new Map();
    this.nextSlot = 1;

    // Special "zero-size buffer" that's reused across all allocations of size
    // zero, backing slots for those allocations.
    //
    // WebGPU allows creating buffers of size 0, but you cannot actually make
    // bindings of size 0 when calling `createBindGroup()`. The simplest way to
    // handle this is to just create a buffer of minimum size (4 bytes) and
    // reuse that across all zero-size allocations.
    this.#reusableZsb = this.#createBuffer(4);

    device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured error in WebGPU backend:", event.error.message);
    });
  }

  /** Number of live backend slots (excludes pooled buffers). */
  slotCount(): number {
    return this.buffers.size;
  }

  /** Total GPU bytes held by jax-js: live buffer bytes + pooled buffer bytes. */
  gpuAllocatedBytes(): number {
    return this.#gpuAllocatedBytes;
  }

  /**
   * High-water mark: the maximum `gpuAllocatedBytes()` ever reached.
   * Useful for verifying that fusion, recycling, and pooling reduce peak memory.
   */
  peakGpuAllocatedBytes(): number {
    return this.#gpuPeakBytes;
  }

  /** Reset the peak memory watermark to the current allocation level. */
  resetPeakGpuAllocatedBytes(): void {
    this.#gpuPeakBytes = this.#gpuAllocatedBytes;
  }

  /** Number of GPU compute dispatches since last reset. */
  get dispatchCount(): number {
    return _dispatchCount;
  }

  /** Reset the dispatch counter to zero. */
  resetDispatchCount(): void {
    _dispatchCount = 0;
  }

  // ---------------------------------------------------------------------------
  // GPU timestamp profiling (T3/P9)
  // ---------------------------------------------------------------------------

  /** Begin GPU timestamp profiling. Requires `timestamp-query` feature. */
  startProfiling(): void {
    if (!this.device.features.has("timestamp-query")) {
      throw new Error("timestamp-query feature not available on this device");
    }
    if (this.#profiling) {
      throw new Error("Profiling already active — call stopProfiling() first");
    }
    this.#profiling = {
      querySet: this.device.createQuerySet({
        type: "timestamp",
        count: _MAX_PROFILING_PASSES * 2,
      }),
      passIdx: 0,
      overflow: false,
      passMeta: [],
    };
  }

  /** Stop profiling and return per-pass GPU timing results. */
  async stopProfiling(): Promise<GpuTimingResult> {
    const state = this.#profiling;
    this.#profiling = null;

    if (!state || state.passIdx === 0) {
      state?.querySet.destroy();
      return { passes: [], totalMs: 0, truncated: state?.overflow ?? false };
    }

    const entryCount = state.passIdx * 2;
    const resolveSize = entryCount * 8; // BigUint64 = 8 bytes each

    const resolveBuffer = this.device.createBuffer({
      size: resolveSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: resolveSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.resolveQuerySet(state.querySet, 0, entryCount, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, resolveSize);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(readBuffer.getMappedRange());

    const passes: GpuTimingResult["passes"] = [];
    for (let i = 0; i < state.passIdx; i++) {
      const startNs = timestamps[i * 2];
      const endNs = timestamps[i * 2 + 1];
      const meta = state.passMeta[i];
      passes.push({
        durationMs: Number(endNs - startNs) / 1_000_000,
        ...(meta?.grid && { grid: meta.grid }),
        ...(meta?.workgroupSize != null && {
          workgroupSize: meta.workgroupSize,
        }),
        ...(meta?.label && { label: meta.label }),
      });
    }
    const wallNs = timestamps[entryCount - 1] - timestamps[0];
    const totalMs = Number(wallNs) / 1_000_000;

    readBuffer.unmap();
    readBuffer.destroy();
    resolveBuffer.destroy();
    state.querySet.destroy();

    return { passes, totalMs, truncated: state.overflow };
  }

  /** Buffer pool diagnostic: pooled buffer count, pooled bytes, and byte budget. */
  poolStats(): {
    pooledBuffers: number;
    pooledBytes: number;
    budgetBytes: number;
  } {
    let pooledBuffers = 0;
    for (const list of this.#bufferPool.values()) pooledBuffers += list.length;
    return {
      pooledBuffers,
      pooledBytes: this.#poolCurrentBytes,
      budgetBytes: this.#poolBudgetBytes,
    };
  }

  malloc(size: number, initialData?: Uint8Array<ArrayBuffer>): Slot {
    let buffer: GPUBuffer;
    // All GPUBuffer must be a multiple of 4 bytes in length, to support copy
    // operations. Pad it to a multiple of 4. Minimum 16 bytes ensures uniform
    // bindings work (vec4 in shaders requires ≥16-byte buffers). This floor
    // applies to all allocations, not just uniforms, because threading an
    // `isUniform` flag through malloc would add complexity for negligible
    // gain — only scalar-promotion mallocs (2–8 bytes) are affected.
    const paddedSize = Math.max(Math.ceil(size / 4) * 4, 16);
    if (size === 0) {
      buffer = this.#reusableZsb;
    } else if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      // Try to reuse a pooled buffer for initial data too.
      const pooled = this.#poolPop(paddedSize);
      if (pooled) {
        buffer = pooled;
        this.#writeBufferUnaligned(buffer, initialData);
      } else if (initialData.byteLength < 4096) {
        buffer = this.#createBuffer(paddedSize, { mapped: true });
        new Uint8Array(buffer.getMappedRange(), 0, size).set(initialData);
        buffer.unmap();
      } else {
        // getMappedRange() seems slower for large buffers, use writeBuffer() instead.
        buffer = this.#createBuffer(paddedSize);
        this.#writeBufferUnaligned(buffer, initialData);
      }
    } else {
      // No initial data — try the pool first.
      buffer = this.#poolPop(paddedSize) ?? this.#createBuffer(paddedSize);
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, size, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.buffers.delete(slot);
      // Pending async slots have buffer=undefined until submit() materializes
      // them. If the array is disposed before submit(), just drop the slot
      // entry — there's no GPUBuffer to reclaim.
      if (!buffer.buffer) return;
      if (buffer.buffer !== this.#reusableZsb) {
        if (this.#batchEncoder) {
          // Defer pool-or-destroy: buffer may still be referenced by
          // commands in the batch encoder that haven't been submitted yet.
          this.#batchDeferredFreeBuffers.push(buffer.buffer);
        } else if (!this.#poolPush(buffer.buffer)) {
          this.#gpuAllocatedBytes -= buffer.buffer.size;
          buffer.buffer.destroy();
        }
      }
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;

    // Need a GPUBuffer with MAP_READ usage when transfering data to host.
    const paddedSize = Math.ceil(count / 4) * 4;
    const staging = this.#createBuffer(paddedSize, { read: true });
    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
      this.device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      return new Uint8Array(arrayBuffer.slice(), 0, count);
    } finally {
      staging.destroy();
    }
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;
    return this.syncReader.read(buffer, start, count);
  }

  copyBufferToBuffer(
    src: Slot,
    srcOffset: number,
    dst: Slot,
    dstOffset: number,
    size: number,
  ): void {
    if (size === 0) return;
    const srcBuf = this.#getBuffer(src);
    const dstBuf = this.#getBuffer(dst);
    // Use batch encoder when a batch is active (e.g., fallback scan loop),
    // otherwise create a standalone command encoder.
    const encoder = this.#batchEncoder ?? this.device.createCommandEncoder();
    const ownEncoder = !this.#batchEncoder;
    const uniformBuf = this.#encodeCopyAuto(
      encoder,
      srcBuf.buffer,
      srcOffset,
      dstBuf.buffer,
      dstOffset,
      size,
    );
    if (ownEncoder) {
      this.device.queue.submit([encoder.finish()]);
      if (uniformBuf) uniformBuf.destroy();
    } else if (uniformBuf) {
      // Defer uniform buffer destruction until batch ends
      this.#batchUniformsToDestroy.push(uniformBuf);
    }
  }

  /**
   * Write data to a GPU buffer, handling non-4-byte-aligned sizes.
   * WebGPU's writeBuffer requires 4-byte-aligned size; this pads the
   * trailing bytes when necessary.
   */
  #writeBufferUnaligned(
    buffer: GPUBuffer,
    data: Uint8Array<ArrayBuffer>,
  ): void {
    if (data.byteLength % 4 === 0) {
      this.device.queue.writeBuffer(buffer, 0, data);
    } else {
      const aligned = data.byteLength - (data.byteLength % 4);
      this.device.queue.writeBuffer(buffer, 0, data, 0, aligned);
      const remainder = new Uint8Array(4);
      remainder.set(data.subarray(aligned));
      this.device.queue.writeBuffer(buffer, aligned, remainder);
    }
  }

  #cachedShader(kernel: Kernel): ShaderInfo {
    const cacheKey = FpHash.hash(kernel);
    let result = this.#cachedShaderMap.get(cacheKey);
    if (!result) {
      result = pipelineSource(this.device, kernel, this.capabilities);
      if (kernel.label) result.label = kernel.label;
      this.#cachedShaderMap.set(cacheKey, result);
    } else if (!result.label && kernel.label) {
      result.label = kernel.label;
    }
    return result;
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<ShaderDispatch[]>> {
    const shader = this.#cachedShader(kernel);
    const pipeline = await this.pipelines.prepare(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  prepareKernelSync(kernel: Kernel): Executable<ShaderDispatch[]> {
    const shader = this.#cachedShader(kernel);
    const pipeline = this.pipelines.prepareSync(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  async prepareRoutine(
    routine: Routine,
  ): Promise<Executable<ShaderDispatch[]>> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = await Promise.all(
      shaders.map(async (shader) => {
        const pipeline = await this.pipelines.prepare(shader);
        return { ...shader, pipeline };
      }),
    );
    return new Executable(routine, dispatches);
  }

  prepareRoutineSync(routine: Routine): Executable<ShaderDispatch[]> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = shaders.map((shader) => {
      const pipeline = this.pipelines.prepareSync(shader);
      return { ...shader, pipeline };
    });
    return new Executable(routine, dispatches);
  }

  dispatch(
    exe: Executable<ShaderDispatch[]>,
    inputs: Slot[],
    outputs: Slot[],
    dynamicParams?: number[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
    pipelineSubmit(
      this.device,
      exe.data,
      inputBuffers,
      outputBuffers,
      dynamicParams,
      this.#batchEncoder ?? undefined,
      this.#batchEncoder ? this.#batchUniformsToDestroy : undefined,
      this.#profiling,
      exe.source,
    );
  }

  // ---------------------------------------------------------------------------
  // Native scan methods (P3: WebGPU multi-kernel scan)
  // ---------------------------------------------------------------------------

  /**
   * Prepare a multi-kernel native scan shader.
   * Returns null if codegen fails.
   */
  prepareNativeScanMulti(
    params: NativeScanMultiParams,
  ): Executable<ShaderDispatch[]> | null {
    const { steps } = params;
    if (!steps || steps.length === 0) return null;

    try {
      const shader = nativeScanMultiShaderSource(this.device, params);
      const pipeline = this.pipelines.prepareSync(shader);
      const firstKernel = steps[0].kernel;
      return new Executable(firstKernel, [{ ...shader, pipeline }]);
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("WebGPU native scan multi codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a native scan shader.
   *
   * Copies initCarry → carryOut before dispatch (shader has no initCarry bindings).
   * Buffer binding order: [consts, xs, carryOut, ysStacked]
   */
  dispatchNativeScanGeneral(
    exe: Executable<ShaderDispatch[]>,
    params: NativeScanMultiParams,
    consts: (Slot | GPUBuffer)[],
    initCarry: (Slot | GPUBuffer)[],
    xs: (Slot | GPUBuffer)[],
    carryOut: (Slot | GPUBuffer)[],
    ysStacked: (Slot | GPUBuffer)[],
    tapeCtx?: {
      encoder: GPUCommandEncoder;
      deferredDestroys: GPUBuffer[];
      copyUniforms: GPUBuffer[];
      pooledDuringTape: Set<GPUBuffer>;
    },
  ): void {
    const getBuf = (s: Slot | GPUBuffer) =>
      s instanceof GPUBuffer ? s : this.#getBuffer(s).buffer;
    const commandEncoder =
      tapeCtx?.encoder ??
      this.#batchEncoder ??
      this.device.createCommandEncoder();
    const ownEncoder = !tapeCtx && !this.#batchEncoder;

    // Pre-copy initCarry → carryOut (shader reads carry as pre-initialized)
    for (let i = 0; i < initCarry.length; i++) {
      const initBuf = getBuf(initCarry[i]);
      const carryBuf = getBuf(carryOut[i]);
      if (initBuf !== carryBuf) {
        commandEncoder.copyBufferToBuffer(
          initBuf,
          0,
          carryBuf,
          0,
          params.carrySizes[i],
        );
      }
    }

    // Binding order: [consts(read), xs(read), carry(read_write), ys(read_write)]
    const allBuffers = [
      ...consts.map(getBuf),
      ...xs.map(getBuf),
      ...carryOut.map(getBuf),
      ...ysStacked.map(getBuf),
    ];

    for (const { pipeline, ...shader } of exe.data) {
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: allBuffers.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
      });

      for (const { grid } of shader.passes) {
        if (prod(grid) === 0) continue;
        const wgs = shader.workgroupSize;
        const passEncoder = _beginComputePass(commandEncoder, this.#profiling, {
          grid,
          workgroupSize:
            wgs == null
              ? undefined
              : typeof wgs === "number"
                ? wgs
                : (wgs.filter((x) => x != null) as number[]),
        });
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(grid[0], grid[1]);
        _dispatchCount++;
        passEncoder.end();
      }
    }

    if (ownEncoder) {
      this.device.queue.submit([commandEncoder.finish()]);
    }
  }
  // ---------------------------------------------------------------------------
  // Preencoded-routine scan methods (P4)
  // ---------------------------------------------------------------------------

  #getCopyPipeline(): GPUComputePipeline {
    if (this.#copyPipeline) return this.#copyPipeline;
    const shader: ShaderInfo = {
      code: COPY_SHADER_CODE,
      numInputs: 1,
      numOutputs: 1,
      hasUniform: true,
      passes: [{ grid: [1, 1], uniform: new Uint8Array(16) }],
    };
    this.#copyPipeline = this.pipelines.prepareSync(shader);
    return this.#copyPipeline;
  }

  #encodeCopyWithShader(
    commandEncoder: GPUCommandEncoder,
    srcBuf: GPUBuffer,
    srcOffset: number,
    dstBuf: GPUBuffer,
    dstOffset: number,
    size: number,
  ): GPUBuffer | null {
    if (size <= 0) return null;
    const firstDstWord = dstOffset >>> 2;
    const lastDstWord = (dstOffset + size + 3) >>> 2;
    const words = lastDstWord - firstDstWord;
    const workgroups = Math.ceil(words / COPY_WORKGROUP_SIZE);
    if (workgroups === 0) return null;

    const [gridX, gridY] = calculateGrid(workgroups);
    const pipeline = this.#getCopyPipeline();

    const storageBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: dstBuf } },
      ],
    });

    const uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    new Uint32Array(uniformBuffer.getMappedRange()).set([
      srcOffset,
      dstOffset,
      size,
      0,
    ]);
    uniformBuffer.unmap();

    const uniformBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    const passEncoder = _beginComputePass(commandEncoder, this.#profiling, {
      grid: [gridX, gridY],
      label: "copy",
    });
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, storageBindGroup);
    passEncoder.setBindGroup(1, uniformBindGroup);
    passEncoder.dispatchWorkgroups(gridX, gridY);
    _dispatchCount++;
    passEncoder.end();

    return uniformBuffer;
  }

  /**
   * Encode a buffer copy that works for both aligned and unaligned sizes.
   * Operates on raw GPUBuffer/GPUCommandEncoder — no Slot dependency.
   *
   * - Aligned (offsets and size divisible by 4): native copyBufferToBuffer.
   * - Unaligned: WGSL copy shader dispatch (stays on GPU).
   *
   * Returns the temporary uniform buffer to destroy after queue.submit(),
   * or null if no temporary was needed (aligned path or zero size).
   */
  #encodeCopyAuto(
    encoder: GPUCommandEncoder,
    srcBuf: GPUBuffer,
    srcOffset: number,
    dstBuf: GPUBuffer,
    dstOffset: number,
    size: number,
  ): GPUBuffer | null {
    if (size === 0) return null;
    if (srcOffset % 4 === 0 && dstOffset % 4 === 0 && size % 4 === 0) {
      encoder.copyBufferToBuffer(srcBuf, srcOffset, dstBuf, dstOffset, size);
      return null;
    }
    return this.#encodeCopyWithShader(
      encoder,
      srcBuf,
      srcOffset,
      dstBuf,
      dstOffset,
      size,
    );
  }

  /**
   * Returns the minimum uniform buffer offset alignment for preencoded scan.
   */
  getPreencodedScanAlignment(): number {
    return this.device.limits.minUniformBufferOffsetAlignment ?? 256;
  }

  /**
   * Prepare a preencoded scan operation for routine bodies (matmul, etc.).
   *
   * Wraps routine shaders with uniform-based offset bindings for xs/ys,
   * creates the combined offsets buffer, and compiles pipelines.
   * Returns null if the routine can't be preencoded.
   */
  preparePreencodedScan(
    params: PreencodedScanParams,
  ): PreparedPreencodedScan | null {
    const {
      xsElemStrides,
      ysElemStrides,
      bodyRoutine,
      numConsts,
      numCarry,
      numX,
      numY,
      length,
      reverse,
      carrySizes,
      routineInputJitIds,
      routineOutputJitIds,
    } = params;

    if (!bodyRoutine || bodyRoutine.data.length === 0) {
      if (DEBUG >= 2) console.log("Preencoded scan: invalid routine");
      return null;
    }

    // Skip if no xs/ys need offsets
    if (numX === 0 && numY === 0) {
      if (DEBUG >= 2) console.log("Preencoded scan: no xs/ys, skipping");
      return null;
    }

    const scanInfo: ScanBindingInfo = {
      numConsts,
      numCarry,
      routineInputJitIds,
      routineOutputJitIds,
    };

    // Wrap each shader with scan offset support
    const wrappedShaders: ShaderDispatch[] = [];

    // Shared layout for group(1) with dynamic offset support — identical
    // for all shaders (single uniform binding for scan offsets).
    const uniformLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });

    for (const shader of bodyRoutine.data) {
      // Shaders that already use uniforms (like Sort) conflict with our offset uniform
      if (shader.hasUniform) {
        if (DEBUG >= 2)
          console.log("Preencoded scan: shader already has uniform, skipping");
        return null;
      }

      const wrapped = wrapRoutineForScan(shader, scanInfo);
      if (!wrapped.hasUniform) {
        if (DEBUG >= 2)
          console.log("Preencoded scan: shader doesn't need offsets");
        return null;
      }

      const module = this.device.createShaderModule({ code: wrapped.code });

      // Create explicit group(0) layout from binding counts. Cannot use
      // auto-layout extraction: WebGPU forbids reusing a bind group layout
      // created as part of a pipeline's default layout.
      const group0Layout = this.device.createBindGroupLayout({
        entries: range(wrapped.numInputs + wrapped.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: (i < wrapped.numInputs
              ? "read-only-storage"
              : "storage") as GPUBufferBindingType,
          },
        })),
      });
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [group0Layout, uniformLayout],
      });
      const pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });

      wrappedShaders.push({
        ...shader,
        code: wrapped.code,
        hasUniform: true,
        pipeline,
      });
    }

    // Create combined uniform buffer with offsets for all iterations
    const alignment = this.getPreencodedScanAlignment();
    const { buffer: offsetData, alignment: offsetAlignment } =
      createAllIterationsOffsetsBuffer(
        numX,
        numY,
        length,
        xsElemStrides,
        ysElemStrides,
        alignment,
        reverse,
      );

    const offsetBuffer = this.device.createBuffer({
      size: offsetData.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(offsetBuffer.getMappedRange()).set(offsetData);
    offsetBuffer.unmap();

    if (DEBUG >= 1) {
      console.log(
        `Preencoded scan: prepared for ${length} iterations with uniform offsets`,
      );
    }

    const copyUsesShader = carrySizes.map((size) => size % 4 !== 0);

    return {
      params,
      wrappedShaders,
      offsetBuffer,
      offsetAlignment,
      copyUsesShader,
      uniformLayout,
    };
  }

  /**
   * Dispatch a preencoded scan with routine body.
   *
   * Uses ping-pong buffers for carry and uniform-based offsets for xs/ys.
   * All iteration dispatches are encoded in a single command buffer.
   */
  dispatchPreencodedScan(
    prepared: PreparedPreencodedScan,
    constSlots: (Slot | GPUBuffer)[],
    initCarrySlots: (Slot | GPUBuffer)[],
    xsSlots: (Slot | GPUBuffer)[],
    carryOutSlots: (Slot | GPUBuffer)[],
    ysStackedSlots: (Slot | GPUBuffer)[],
    tapeCtx?: {
      encoder: GPUCommandEncoder;
      deferredDestroys: GPUBuffer[];
      copyUniforms: GPUBuffer[];
      pooledDuringTape: Set<GPUBuffer>;
    },
  ): void {
    const {
      params,
      wrappedShaders,
      offsetBuffer,
      offsetAlignment,
      copyUsesShader,
      uniformLayout,
    } = prepared;
    const {
      length,
      carrySizes,
      numCarry,
      numConsts,
      reverse,
      routineInputJitIds,
      routineOutputJitIds,
      ySources,
    } = params;

    const getBuf = (s: Slot | GPUBuffer) =>
      s instanceof GPUBuffer ? s : this.#getBuffer(s).buffer;
    const constBuffers = constSlots.map(getBuf);
    const initCarryBuffers = initCarrySlots.map(getBuf);
    const xsBuffers = xsSlots.map(getBuf);
    const carryOutBuffers = carryOutSlots.map(getBuf);
    const ysStackedBuffers = ysStackedSlots.map(getBuf);

    // Create ping-pong buffers for carry state (prefer pool)
    const carryPing = carrySizes.map((size) => {
      const padded = Math.max(size, 4);
      return this.#poolPop(padded) ?? this.#createBuffer(padded);
    });
    const carryPong = carrySizes.map((size) => {
      const padded = Math.max(size, 4);
      return this.#poolPop(padded) ?? this.#createBuffer(padded);
    });

    const commandEncoder =
      tapeCtx?.encoder ??
      this.#batchEncoder ??
      this.device.createCommandEncoder();
    const ownEncoder = !tapeCtx && !this.#batchEncoder;
    const copyUniformBuffers: GPUBuffer[] = [];

    // Copy initCarry to carryPing
    for (let i = 0; i < numCarry; i++) {
      if (carrySizes[i] > 0) {
        commandEncoder.copyBufferToBuffer(
          initCarryBuffers[i],
          0,
          carryPing[i],
          0,
          carrySizes[i],
        );
      }
    }

    const xsStart = numConsts + numCarry;

    for (const shader of wrappedShaders) {
      const { pipeline, passes } = shader;

      // Helper to create storage bind group for a ping-pong configuration
      const createStorageBindGroup = (
        readCarry: GPUBuffer[],
        writeCarry: GPUBuffer[],
      ): GPUBindGroup => {
        const entries: GPUBindGroupEntry[] = [];
        let binding = 0;

        // Input bindings: classify by scan buffer role
        for (let i = 0; i < routineInputJitIds.length; i++) {
          const jitId = routineInputJitIds[i];
          let buffer: GPUBuffer;

          if (jitId < numConsts) {
            buffer = constBuffers[jitId];
          } else if (jitId < xsStart) {
            const carryIdx = jitId - numConsts;
            buffer = readCarry[carryIdx];
          } else {
            const xIdx = jitId - xsStart;
            buffer = xsBuffers[xIdx];
          }

          entries.push({ binding: binding++, resource: { buffer } });
        }

        // Output bindings: in passthrough pattern, all go to carry (write)
        for (let i = 0; i < routineOutputJitIds.length; i++) {
          const buffer = writeCarry[i];
          if (!buffer) {
            throw new Error(
              `Preencoded scan: routine output ${i} has no carry buffer ` +
                `(writeCarry.length=${writeCarry.length})`,
            );
          }
          entries.push({ binding: binding++, resource: { buffer } });
        }

        return this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
      };

      const pingBindGroup = createStorageBindGroup(carryPing, carryPong);
      const pongBindGroup = createStorageBindGroup(carryPong, carryPing);

      // Create single uniform bind group with dynamic offset support
      const uniformBindGroup = this.device.createBindGroup({
        layout: uniformLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: offsetBuffer,
              offset: 0,
              size: offsetAlignment,
            },
          },
        ],
      });

      const filteredPasses = passes.filter(({ grid }) => prod(grid) > 0);

      for (let iter = 0; iter < length; iter++) {
        const storageBindGroup = iter % 2 === 0 ? pingBindGroup : pongBindGroup;

        for (const { grid } of filteredPasses) {
          const passEncoder = _beginComputePass(
            commandEncoder,
            this.#profiling,
          );
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, storageBindGroup);
          passEncoder.setBindGroup(1, uniformBindGroup, [
            iter * offsetAlignment,
          ]);
          passEncoder.dispatchWorkgroups(grid[0], grid[1]);
          _dispatchCount++;
          passEncoder.end();
        }

        // Stack ys: dispatch by classified Y source type.
        // Preencoded-routine only accepts carry-live (validated by planner).
        const ysIdx = reverse ? length - 1 - iter : iter;
        const currentCarryBuffers = iter % 2 === 0 ? carryPong : carryPing;
        for (const src of ySources) {
          let srcBuf: GPUBuffer;
          let copySize: number;
          switch (src.type) {
            case "carry-live":
              srcBuf = currentCarryBuffers[src.carryIdx];
              copySize = carrySizes[src.carryIdx];
              break;
            /* c8 ignore next 3 */
            default:
              throw new Error(
                `Unexpected Y source type "${src.type}" in preencoded-routine`,
              );
          }
          if (copySize <= 0) continue;
          const yOffset = ysIdx * copySize;
          if (!copyUsesShader[src.yi]) {
            commandEncoder.copyBufferToBuffer(
              srcBuf,
              0,
              ysStackedBuffers[src.yi],
              yOffset,
              copySize,
            );
          } else {
            const uniformBuf = this.#encodeCopyWithShader(
              commandEncoder,
              srcBuf,
              0,
              ysStackedBuffers[src.yi],
              yOffset,
              copySize,
            );
            if (uniformBuf) copyUniformBuffers.push(uniformBuf);
          }
        }
      }
    }

    // Copy final carry to carryOut
    const finalCarry = length % 2 === 0 ? carryPing : carryPong;
    for (let i = 0; i < numCarry; i++) {
      const copySize = carrySizes[i];
      if (copySize <= 0) continue;
      if (!copyUsesShader[i]) {
        commandEncoder.copyBufferToBuffer(
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          copySize,
        );
      } else {
        const uniformBuf = this.#encodeCopyWithShader(
          commandEncoder,
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          copySize,
        );
        if (uniformBuf) copyUniformBuffers.push(uniformBuf);
      }
    }

    if (ownEncoder) {
      this.device.queue.submit([commandEncoder.finish()]);
    }

    // Clean up temporary buffers
    if (tapeCtx) {
      for (const buf of copyUniformBuffers) tapeCtx.copyUniforms.push(buf);
      for (const buf of [...carryPing, ...carryPong]) {
        if (!this.#poolPush(buf)) {
          tapeCtx.deferredDestroys.push(buf);
        } else {
          tapeCtx.pooledDuringTape.add(buf);
        }
      }
    } else {
      for (const buf of copyUniformBuffers) buf.destroy();
      for (const buf of [...carryPing, ...carryPong]) {
        if (!this.#poolPush(buf)) {
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
        }
      }
    }
    // offsetBuffer is NOT destroyed — owned by PreparedPreencodedScan for reuse
  }

  // -------------------------------------------------------------------------
  // Preencoded multi-step scan (Phase 2c)
  // -------------------------------------------------------------------------

  /**
   * Prepare a preencoded multi-step scan: wrap each kernel's shader with
   * per-iteration xs offsets, compile pipelines, and create offset buffers.
   */
  preparePreencodedMultiStepScan(params: {
    length: number;
    executeSteps: {
      source: Kernel | Routine;
      inputs: number[];
      outputs: number[];
    }[];
    carrySizes: number[];
    internalSizes: number[];
    internalMap: Map<number, number>;
    numCarry: number;
    numConsts: number;
    numX: number;
    numY: number;
    reverse: boolean;
    xsElemStrides: number[];
    ysElemStrides: number[];
    ysSizes: number[];
    carrySources: WebGPUCarryOutputSource[];
    ySources: WebGPUYOutputSource[];
  }): PreparedPreencodedMultiStep | null {
    const {
      length,
      executeSteps,
      carrySizes,
      internalSizes,
      internalMap,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
      xsElemStrides,
      ysSizes,
      carrySources,
      ySources,
    } = params;

    if (length === 0) return null;

    // Shared layout for group(1) with dynamic offset support
    const uniformLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });

    const minAlignment = this.getPreencodedScanAlignment();
    const xsStart = numConsts + numCarry;
    const stepEntries: PreencodedMultiStepEntry[] = [];

    for (const step of executeSteps) {
      // Obtain shader info: kernels use the cached shader, routines use
      // createRoutineShader which returns ShaderInfo[] (one per sub-shader).
      // All current non-sort routines produce exactly 1 ShaderInfo.
      let shaderInfo: ShaderInfo;
      if (step.source instanceof Kernel) {
        shaderInfo = this.#cachedShader(step.source);
      } else {
        const routine = step.source as Routine;
        const shaderInfos = createRoutineShader(this.device, routine);
        if (shaderInfos.length !== 1) {
          if (DEBUG >= 2)
            console.log(
              "preparePreencodedMultiStepScan: multi-shader routine not supported",
            );
          return null;
        }
        shaderInfo = shaderInfos[0];
      }

      // Reject steps that already use uniforms (symbolic dims or Sort)
      if (shaderInfo.hasUniform) {
        if (DEBUG >= 2)
          console.log(
            "preparePreencodedMultiStepScan: step already has uniform",
          );
        return null;
      }

      // Build ScanBindingInfo for this step — only count this step's
      // actual xs inputs, excluding internal buffer JitIds.
      const scanInfo: ScanBindingInfo = {
        numConsts,
        numCarry,
        numX,
        routineInputJitIds: step.inputs,
        routineOutputJitIds: step.outputs,
      };

      // Wrap the shader with xs offset support
      const wrapped = wrapRoutineForScan(shaderInfo, scanInfo);

      let dispatches: ShaderDispatch[];
      let offsetBuffer: GPUBuffer | null = null;
      let offsetAlignment = 0;

      if (wrapped.hasUniform) {
        // Step has xs inputs — needs offset uniform at group(1).
        // Create explicit group(0) layout from binding counts. Cannot use
        // auto-layout extraction: WebGPU forbids reusing a bind group layout
        // created as part of a pipeline's default layout.
        const module = this.device.createShaderModule({
          code: wrapped.code,
        });
        const group0Layout = this.device.createBindGroupLayout({
          entries: range(wrapped.numInputs + wrapped.numOutputs).map((i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: (i < wrapped.numInputs
                ? "read-only-storage"
                : "storage") as GPUBufferBindingType,
            },
          })),
        });
        const pipelineLayout = this.device.createPipelineLayout({
          bindGroupLayouts: [group0Layout, uniformLayout],
        });
        const pipeline = this.device.createComputePipeline({
          layout: pipelineLayout,
          compute: { module, entryPoint: "main" },
        });

        dispatches = [{ ...wrapped, pipeline }];

        // Compute per-step xs element strides (in order of binding appearance)
        const stepXsStrides: number[] = [];
        for (const jitId of step.inputs) {
          if (jitId >= xsStart && jitId < xsStart + numX) {
            stepXsStrides.push(xsElemStrides[jitId - xsStart]);
          }
        }

        // Create per-step offset buffer
        const { buffer: offsetData, alignment } =
          createAllIterationsOffsetsBuffer(
            stepXsStrides.length,
            0, // no ys offsets in shader — ys stacked via copy
            length,
            stepXsStrides,
            [],
            minAlignment,
            reverse,
          );

        offsetBuffer = this.device.createBuffer({
          size: Math.max(offsetData.length, 4),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        });
        new Uint8Array(offsetBuffer.getMappedRange()).set(offsetData);
        offsetBuffer.unmap();
        offsetAlignment = alignment;
      } else {
        // Step has no xs inputs — use normal pipeline without group(1)
        const pipeline = this.pipelines.prepareSync(shaderInfo);
        dispatches = [{ ...shaderInfo, pipeline }];
      }

      stepEntries.push({
        dispatches,
        inputJitIds: step.inputs,
        outputJitIds: step.outputs,
        offsetBuffer,
        offsetAlignment,
      });
    }

    const copyUsesShader = ysSizes.map((size) => size % 4 !== 0);

    if (DEBUG >= 1) {
      const wrappedCount = stepEntries.filter(
        (e) => e.offsetBuffer !== null,
      ).length;
      console.log(
        `Preencoded multi-step scan: prepared ${stepEntries.length} steps ` +
          `(${wrappedCount} with xs offsets), ${length} iterations`,
      );
    }

    return {
      length,
      stepEntries,
      carrySizes,
      internalSizes,
      internalMap,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
      copyUsesShader,
      uniformLayout,
      carrySources,
      ySources,
      ysSizes,
    };
  }

  /**
   * Dispatch a preencoded multi-step scan.
   *
   * Encodes ALL iterations into a single command buffer: for each iteration,
   * dispatches all kernel steps, then copies carry outputs and stacks ys.
   */
  dispatchPreencodedMultiStepScan(
    prepared: PreparedPreencodedMultiStep,
    constSlots: (Slot | GPUBuffer)[],
    initCarrySlots: (Slot | GPUBuffer)[],
    xsSlots: (Slot | GPUBuffer)[],
    carryOutSlots: (Slot | GPUBuffer)[],
    ysStackedSlots: (Slot | GPUBuffer)[],
    tapeCtx?: {
      encoder: GPUCommandEncoder;
      deferredDestroys: GPUBuffer[];
      copyUniforms: GPUBuffer[];
      pooledDuringTape: Set<GPUBuffer>;
    },
  ): void {
    const getBuf = (s: Slot | GPUBuffer) =>
      s instanceof GPUBuffer ? s : this.#getBuffer(s).buffer;
    const {
      length,
      stepEntries,
      carrySizes,
      internalSizes,
      internalMap,
      numCarry,
      numConsts,
      numX,
      reverse: _reverse,
      copyUsesShader,
      uniformLayout,
      carrySources,
      ySources,
      ysSizes,
    } = prepared;

    // Resolve slots to GPU buffers
    const constBuffers = constSlots.map(getBuf);
    const initCarryBuffers = initCarrySlots.map(getBuf);
    const xsBuffers = xsSlots.map(getBuf);
    const carryOutBuffers = carryOutSlots.map(getBuf);
    const ysStackedBuffers = ysStackedSlots.map(getBuf);

    // Create transient ping-pong carry buffers (prefer pool)
    const carryPing = carrySizes.map((sz) => {
      const padded = Math.max(sz, 4);
      return this.#poolPop(padded) ?? this.#createBuffer(padded);
    });
    const carryPong = carrySizes.map((sz) => {
      const padded = Math.max(sz, 4);
      return this.#poolPop(padded) ?? this.#createBuffer(padded);
    });

    // Create transient internal scratch buffers (prefer pool)
    const internalBuffers = internalSizes.map((sz) => {
      const padded = Math.max(sz, 4);
      return this.#poolPop(padded) ?? this.#createBuffer(padded);
    });

    const commandEncoder =
      tapeCtx?.encoder ??
      this.#batchEncoder ??
      this.device.createCommandEncoder();
    const ownEncoder = !tapeCtx && !this.#batchEncoder;
    const copyUniformBuffers: GPUBuffer[] = [];

    // Copy initCarry → carryPing
    for (let i = 0; i < numCarry; i++) {
      if (carrySizes[i] > 0) {
        commandEncoder.copyBufferToBuffer(
          initCarryBuffers[i],
          0,
          carryPing[i],
          0,
          carrySizes[i],
        );
      }
    }

    const xsStart = numConsts + numCarry;

    // Helper: resolve a body JitId to a GPU buffer.
    // Used ONLY for bind-group construction. Carry and Y propagation use
    // typed source arrays instead of resolving raw JitIds per iteration.
    // carryRead is the current iteration's carry read side (ping or pong).
    const resolveBuffer = (
      jitId: number,
      carryRead: GPUBuffer[],
    ): GPUBuffer => {
      if (jitId < numConsts) return constBuffers[jitId];
      if (jitId < xsStart) return carryRead[jitId - numConsts];
      if (jitId < xsStart + numX) return xsBuffers[jitId - xsStart];
      const internalIdx = internalMap.get(jitId);
      if (internalIdx !== undefined) return internalBuffers[internalIdx];
      throw new Error(
        `dispatchPreencodedMultiStepScan: unknown JitId ${jitId}`,
      );
    };

    // Build per-step bind groups (two per step: ping=carry-read-from-ping,
    // pong=carry-read-from-pong). Uniform bind groups are created separately.
    const pingBindGroups: GPUBindGroup[] = [];
    const pongBindGroups: GPUBindGroup[] = [];
    const uniformBindGroups: (GPUBindGroup | null)[] = [];

    for (const entry of stepEntries) {
      const pipeline = entry.dispatches[0].pipeline;
      const layout0 = pipeline.getBindGroupLayout(0);

      const buildStorageBG = (carryRead: GPUBuffer[]): GPUBindGroup => {
        const entries: GPUBindGroupEntry[] = [];
        let binding = 0;
        for (const jitId of entry.inputJitIds) {
          entries.push({
            binding: binding++,
            resource: { buffer: resolveBuffer(jitId, carryRead) },
          });
        }
        for (const jitId of entry.outputJitIds) {
          // Outputs always go to internal buffers
          const internalIdx = internalMap.get(jitId);
          if (internalIdx === undefined) {
            throw new Error(
              `dispatchPreencodedMultiStepScan: output JitId ${jitId} not in internalMap`,
            );
          }
          entries.push({
            binding: binding++,
            resource: { buffer: internalBuffers[internalIdx] },
          });
        }
        return this.device.createBindGroup({ layout: layout0, entries });
      };

      pingBindGroups.push(buildStorageBG(carryPing));
      pongBindGroups.push(buildStorageBG(carryPong));

      // Uniform bind group for xs offsets (if this step has offset data)
      if (entry.offsetBuffer) {
        uniformBindGroups.push(
          this.device.createBindGroup({
            layout: uniformLayout,
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: entry.offsetBuffer,
                  offset: 0,
                  size: Math.max(entry.offsetAlignment, 4),
                },
              },
            ],
          }),
        );
      } else {
        uniformBindGroups.push(null);
      }
    }

    // Encode all iterations
    for (let iter = 0; iter < length; iter++) {
      const readPing = iter % 2 === 0;
      const carryRead = readPing ? carryPing : carryPong;
      const carryWrite = readPing ? carryPong : carryPing;

      // Dispatch each body step (kernel or routine)
      for (let si = 0; si < stepEntries.length; si++) {
        const entry = stepEntries[si];
        const storageBG = readPing ? pingBindGroups[si] : pongBindGroups[si];
        const ubg = uniformBindGroups[si];

        for (const dispatch of entry.dispatches) {
          for (const { grid } of dispatch.passes) {
            if (grid[0] === 0 || grid[1] === 0) continue;
            const pass = _beginComputePass(commandEncoder, this.#profiling);
            pass.setPipeline(dispatch.pipeline);
            pass.setBindGroup(0, storageBG);
            if (ubg) {
              pass.setBindGroup(1, ubg, [iter * entry.offsetAlignment]);
            }
            pass.dispatchWorkgroups(grid[0], grid[1]);
            _dispatchCount++;
            pass.end();
          }
        }
      }

      const iterIdx = _reverse ? length - 1 - iter : iter;

      // Copy carry outputs: dispatch by classified carry source type
      for (let ci = 0; ci < numCarry; ci++) {
        const src = carrySources[ci];
        let srcBuf: GPUBuffer;
        let srcOffset = 0;

        switch (src.type) {
          case "internal":
            srcBuf = internalBuffers[src.internalIdx];
            break;
          case "carry-passthrough":
            srcBuf = carryRead[src.carryIdx];
            break;
          case "const":
            srcBuf = constBuffers[src.constIdx];
            break;
          case "xs":
            srcBuf = xsBuffers[src.xIdx];
            srcOffset = iterIdx * carrySizes[ci];
            break;
        }

        const sz = carrySizes[ci];
        if (sz <= 0) continue;
        if (sz % 4 === 0) {
          commandEncoder.copyBufferToBuffer(
            srcBuf,
            srcOffset,
            carryWrite[ci],
            0,
            sz,
          );
        } else {
          const ub = this.#encodeCopyWithShader(
            commandEncoder,
            srcBuf,
            srcOffset,
            carryWrite[ci],
            0,
            sz,
          );
          if (ub) copyUniformBuffers.push(ub);
        }
      }

      // Stack ys: dispatch by classified Y source type.
      // Carry copy above has already written carryWrite, so carry-live reads
      // from carryWrite (post-update). Carry-snapshot reads from carryRead
      // (pre-update). xs reads from the stacked xs buffer at iteration offset.
      // Internal reads from the scratch buffer.
      const ysIdx = _reverse ? length - 1 - iter : iter;
      for (const src of ySources) {
        let srcBuf: GPUBuffer;
        let srcOffset = 0;
        switch (src.type) {
          case "carry-snapshot":
            srcBuf = carryRead[src.carryIdx];
            break;
          case "carry-live":
            srcBuf = carryWrite[src.carryIdx];
            break;
          case "xs":
            srcBuf = xsBuffers[src.xsIdx];
            // xs buffer is the full stacked array; read the iteration's slice
            srcOffset = ysIdx * ysSizes[src.yi];
            break;
          case "internal":
            srcBuf = internalBuffers[src.internalIdx];
            break;
        }
        const sz = ysSizes[src.yi];
        if (sz <= 0) continue;
        const yOffset = ysIdx * sz;
        if (!copyUsesShader[src.yi]) {
          commandEncoder.copyBufferToBuffer(
            srcBuf,
            srcOffset,
            ysStackedBuffers[src.yi],
            yOffset,
            sz,
          );
        } else {
          const ub = this.#encodeCopyWithShader(
            commandEncoder,
            srcBuf,
            srcOffset,
            ysStackedBuffers[src.yi],
            yOffset,
            sz,
          );
          if (ub) copyUniformBuffers.push(ub);
        }
      }
    }

    // Copy final carry → carryOut
    const finalCarry = length % 2 === 0 ? carryPing : carryPong;
    for (let i = 0; i < numCarry; i++) {
      const sz = carrySizes[i];
      if (sz <= 0) continue;
      if (sz % 4 === 0) {
        commandEncoder.copyBufferToBuffer(
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          sz,
        );
      } else {
        const ub = this.#encodeCopyWithShader(
          commandEncoder,
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          sz,
        );
        if (ub) copyUniformBuffers.push(ub);
      }
    }

    if (ownEncoder) {
      this.device.queue.submit([commandEncoder.finish()]);
    }

    // Clean up transient buffers
    if (tapeCtx) {
      for (const buf of copyUniformBuffers) tapeCtx.copyUniforms.push(buf);
      for (const buf of [...carryPing, ...carryPong, ...internalBuffers]) {
        if (!this.#poolPush(buf)) {
          tapeCtx.deferredDestroys.push(buf);
        } else {
          tapeCtx.pooledDuringTape.add(buf);
        }
      }
    } else {
      for (const buf of copyUniformBuffers) buf.destroy();
      for (const buf of [...carryPing, ...carryPong, ...internalBuffers]) {
        if (!this.#poolPush(buf)) {
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
        }
      }
    }
    // Per-step offset buffers are NOT destroyed — owned by prepared for reuse
  }

  #getBuffer(slot: Slot): { buffer: GPUBuffer; size: number } {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return { buffer: buffer.buffer, size: buffer.size };
  }

  /**
   * Create a GPU buffer.
   *
   * By default, this creates a general-purpose buffer with the given size.
   *
   * - If `mapped` is true, initialize the buffer in mapped mode so that it can
   *   be populated with data from the CPU. (Call `.unmap()` later.)
   * - If `read` is true, create a staging buffer for returning data to CPU.
   *   (Call `.mapAsync()` later.)
   */
  /** Pop a buffer of the given padded byte size from the pool, or null. */
  #poolPop(paddedSize: number): GPUBuffer | null {
    const list = this.#bufferPool.get(paddedSize);
    if (list && list.length > 0) {
      const buf = list.pop()!;
      this.#poolCurrentBytes -= buf.size;
      return buf;
    }
    return null;
  }

  /** Push a freed buffer into the pool. Returns false if pool is full. */
  #poolPush(buffer: GPUBuffer): boolean {
    const paddedSize = buffer.size;
    // Per-size-class cap.
    let list = this.#bufferPool.get(paddedSize);
    if (!list) {
      list = [];
      this.#bufferPool.set(paddedSize, list);
    }
    if (list.length >= WebGPUBackend.MAX_POOL_PER_SIZE) return false;
    // Byte budget cap.
    if (this.#poolCurrentBytes + paddedSize > this.#poolBudgetBytes) {
      return false;
    }
    list.push(buffer);
    this.#poolCurrentBytes += paddedSize;
    return true;
  }

  /** Remove a specific buffer from the pool (error cleanup). */
  #poolRemove(buffer: GPUBuffer): void {
    const list = this.#bufferPool.get(buffer.size);
    if (!list) return;
    const idx = list.indexOf(buffer);
    if (idx === -1) return;
    list.splice(idx, 1);
    this.#poolCurrentBytes -= buffer.size;
  }

  /**
   * Prepare the pool for the next JitProgram execution:
   * 1. Evict entries whose sizes aren't needed (stale cross-program buffers).
   * 2. Set the byte budget to the program's peak live bytes.
   *
   * This guarantees physical peak memory ≤ peak live memory: the pool only
   * retains buffers that will be reused, and total retained bytes can't exceed
   * what the program already needs at its peak point.
   */
  configurePool(hints: {
    readonly peakBytes: number;
    readonly mallocSizes: ReadonlySet<number>;
  }): void {
    // Evict pool entries whose sizes won't be needed.
    for (const [size, list] of this.#bufferPool) {
      if (!hints.mallocSizes.has(size)) {
        for (const buf of list) {
          this.#poolCurrentBytes -= buf.size;
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
        }
        this.#bufferPool.delete(size);
      }
    }
    // Update byte budget to this program's peak.
    this.#poolBudgetBytes = hints.peakBytes;
    // If current pool exceeds new budget, evict excess (LIFO per size).
    while (this.#poolCurrentBytes > this.#poolBudgetBytes) {
      let evicted = false;
      for (const [_size, list] of this.#bufferPool) {
        if (list.length > 0) {
          const buf = list.pop()!;
          this.#poolCurrentBytes -= buf.size;
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
          evicted = true;
          break;
        }
      }
      if (!evicted) break;
    }
  }

  // ---------------------------------------------------------------------------
  // BlockMap fused shader (Phase 3: single-dispatch shared-memory compiler)
  // ---------------------------------------------------------------------------

  /**
   * Compile a block_map body into a fused WGSL compute shader.
   * Returns a reusable Executable, or null if the body is not eligible.
   */
  prepareBlockMapFused(
    params: BlockMapShaderParams,
  ): Executable<ShaderDispatch[]> | null {
    try {
      const shader = blockMapFusedShaderSource(
        this.device,
        params,
        this.capabilities,
      );
      if (!shader) return null;
      const pipeline = this.pipelines.prepareSync(shader);
      return new Executable(null as any, [{ ...shader, pipeline }]);
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("WebGPU block_map fused shader codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a fused block_map shader.
   * Buffer binding order: [bodyInputs (consts + block inputs), outputs]
   *
   * When the shader uses uniform constants (numUniformConsts > 0), the first
   * N input buffers are bound to @group(1) as uniform buffers instead of
   * @group(0) storage. The remaining inputs + outputs go to @group(0).
   */
  dispatchBlockMapFused(
    exe: Executable<ShaderDispatch[]>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const shader = exe.data[0];
    const nuc = shader.numUniformConsts ?? 0;
    if (nuc > 0) {
      // Split: first nuc inputs → group(1) uniform, rest → group(0) storage
      const constBuffers = inputs
        .slice(0, nuc)
        .map((slot) => this.#getBuffer(slot).buffer);
      const storageInputBuffers = inputs
        .slice(nuc)
        .map((slot) => this.#getBuffer(slot).buffer);
      const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
      const commandEncoder =
        this.#batchEncoder ?? this.device.createCommandEncoder();
      const bindGroup0 = this.device.createBindGroup({
        layout: shader.pipeline.getBindGroupLayout(0),
        entries: [
          ...storageInputBuffers.map((buffer, i) => ({
            binding: i,
            resource: { buffer },
          })),
          ...outputBuffers.map((buffer, i) => ({
            binding: storageInputBuffers.length + i,
            resource: { buffer },
          })),
        ],
      });
      const bindGroup1 = this.device.createBindGroup({
        layout: shader.pipeline.getBindGroupLayout(1),
        entries: constBuffers.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
      });
      const grid = shader.passes[0].grid;
      const passEncoder = _beginComputePass(commandEncoder, this.#profiling);
      passEncoder.setPipeline(shader.pipeline);
      passEncoder.setBindGroup(0, bindGroup0);
      passEncoder.setBindGroup(1, bindGroup1);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      _dispatchCount++;
      passEncoder.end();
      if (!this.#batchEncoder) {
        this.device.queue.submit([commandEncoder.finish()]);
      }
    } else {
      const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
      const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
      pipelineSubmit(
        this.device,
        exe.data,
        inputBuffers,
        outputBuffers,
        undefined,
        this.#batchEncoder ?? undefined,
        this.#batchEncoder ? this.#batchUniformsToDestroy : undefined,
        this.#profiling,
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Batch dispatch: encode multiple dispatches into one command submission
  // ---------------------------------------------------------------------------

  beginBatch(): void {
    if (this.#batchDepth === 0) {
      this.#batchEncoder = this.device.createCommandEncoder();
      this.#batchUniformsToDestroy = [];
      this.#batchDeferredFreeBuffers = [];
    }
    this.#batchDepth++;
  }

  endBatch(): void {
    if (this.#batchDepth === 0) return;
    this.#batchDepth--;
    if (this.#batchDepth === 0) {
      const encoder = this.#batchEncoder;
      if (!encoder) return;
      this.#batchEncoder = null;
      this.device.queue.submit([encoder.finish()]);
      // Destroy uniform buffers created during batch encoding
      for (const buf of this.#batchUniformsToDestroy) buf.destroy();
      this.#batchUniformsToDestroy = [];
      // Process deferred buffer frees (delayed to avoid destroying buffers
      // that are referenced by commands in the batch encoder)
      for (const gpuBuf of this.#batchDeferredFreeBuffers) {
        if (!this.#poolPush(gpuBuf)) {
          this.#gpuAllocatedBytes -= gpuBuf.size;
          gpuBuf.destroy();
        }
      }
      this.#batchDeferredFreeBuffers = [];
    }
  }

  /**
   * Submit the current batch encoder mid-batch and replace it with a fresh one.
   * Does NOT change #batchDepth — the batch nesting structure is preserved.
   * This is needed when inner code (e.g., body program execution) must read
   * buffers that were written by commands still in the batch encoder.
   * No-op when no batch is active.
   */
  flushBatch(): void {
    const encoder = this.#batchEncoder;
    if (!encoder) return;
    this.device.queue.submit([encoder.finish()]);
    // Destroy any uniform buffers from the flushed encoder
    for (const buf of this.#batchUniformsToDestroy) buf.destroy();
    this.#batchUniformsToDestroy = [];
    // Note: deferred frees are NOT processed here — they are still referenced
    // by the outer batch scope and will be freed when depth reaches 0.
    // Create a fresh encoder for subsequent commands.
    this.#batchEncoder = this.device.createCommandEncoder();
  }

  async fence(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  // ---------------------------------------------------------------------------
  // Command tape: pre-compiled dispatch sequence (O8)
  // ---------------------------------------------------------------------------

  /**
   * Compile a JitProgram's steps into a WebGPU command tape.
   *
   * Pre-resolves pipelines, bind group layouts, grid dimensions, and uniform
   * bind groups. The resulting tape can be executed repeatedly with different
   * input slots via `executeCommandTape()`.
   *
   * Caller must verify eligibility via `canCompileToCommandTape()` first.
   */
  compileCommandTape(
    steps: JitStep[],
    inputIds: JitId[],
    outputIds: JitId[],
  ): WebGPUCommandTape {
    // 1. Build JitId → table index mapping
    const idToIdx = new Map<JitId, number>();
    let nextIdx = 0;

    // Assign indices to inputs (deduplicate — same JitId may appear twice)
    const inputTableIdxs: number[] = [];
    for (const id of inputIds) {
      if (!idToIdx.has(id)) idToIdx.set(id, nextIdx++);
      inputTableIdxs.push(idToIdx.get(id)!);
    }

    // 2. Walk steps in order, emitting TapeOps that preserve the original
    //    malloc/free/recycle/dispatch interleaving.  This lets freed buffers
    //    return to the pool *before* subsequent mallocs, reducing peak VRAM.
    const ops: TapeOp[] = [];
    const allocatedIdxs: number[] = [];
    const uniformBuffers: GPUBuffer[] = [];
    const knownSizes = new Map<number, number>();

    // Indices that must not be freed (external inputs + final outputs)
    const inputIdxSet = new Set(inputTableIdxs);

    for (const step of steps) {
      switch (step.type) {
        case "execute": {
          const inputIdxs = step.inputs.map((id) => idToIdx.get(id)!);
          const outputIdxs = step.outputs.map((id) => idToIdx.get(id)!);

          // Resolve pipeline
          const exe =
            step.source instanceof Kernel
              ? this.prepareKernelSync(step.source)
              : this.prepareRoutineSync(step.source);

          // Create tape dispatches for each shader dispatch in the executable
          for (const { pipeline, ...shader } of exe.data) {
            const filteredPasses = shader.passes.filter(
              (p) => prod(p.grid) > 0,
            );
            if (filteredPasses.length === 0) continue;

            const bindGroupLayout = pipeline.getBindGroupLayout(0);

            // Pre-build uniform bind group for static uniforms
            let uniformBindGroup: GPUBindGroup | null = null;
            let uniformAlignment = 0;

            if (shader.hasUniform) {
              const uniforms = filteredPasses.map((p) => p.uniform!);
              const [uniformBuffer, alignment] = combineUniforms(
                this.device,
                uniforms,
              );
              uniformBuffers.push(uniformBuffer);
              uniformAlignment = alignment;
              uniformBindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(1),
                entries: [
                  {
                    binding: 0,
                    resource: { buffer: uniformBuffer, size: alignment },
                  },
                ],
              });
            }

            ops.push({
              type: "dispatch",
              dispatch: {
                pipeline,
                shader,
                bindGroupLayout,
                inputIdxs,
                outputIdxs,
                passes: filteredPasses.map((p) => ({ grid: p.grid })),
                uniformBindGroup,
                uniformAlignment,
              },
            });
          }
          break;
        }
        case "malloc": {
          const idx = nextIdx++;
          idToIdx.set(step.output, idx);
          const size = step.size as number;
          const paddedSize = Math.ceil(size / 4) * 4;
          knownSizes.set(idx, size);
          allocatedIdxs.push(idx);
          ops.push({
            type: "malloc",
            malloc: {
              tableIdx: idx,
              paddedSize,
              originalSize: size,
              initialData:
                (step.initialData as Uint8Array<ArrayBuffer>) ?? null,
            },
          });
          break;
        }
        case "free": {
          const idx = idToIdx.get(step.input)!;
          // Don't free external inputs or outputs
          if (!inputIdxSet.has(idx)) {
            ops.push({ type: "free", tableIdx: idx });
          }
          break;
        }
        case "recycle": {
          const fromIdx = idToIdx.get(step.input)!;
          const toIdx = nextIdx++;
          idToIdx.set(step.output, toIdx);
          const fromSize = knownSizes.get(fromIdx);
          if (fromSize !== undefined) knownSizes.set(toIdx, fromSize);
          ops.push({ type: "recycle", fromIdx, toIdx });
          break;
        }
        case "dus": {
          const dstIdx = idToIdx.get(step.dst)!;
          const srcIdx = idToIdx.get(step.src)!;
          const outIdx = idToIdx.get(step.output)!;
          const dus: TapeDUS = {
            dstIdx,
            srcIdx,
            outIdx,
            dstSizeBytes: step.dstSizeBytes as number,
            offsetBytes: step.offsetBytes,
            sliceBytes: step.sliceBytes as number,
            outerFibers: step.outerFibers,
            srcFiberBytes: step.srcFiberBytes,
            dstFiberBytes: step.dstFiberBytes,
            srcBaseOffset: step.srcBaseOffset ?? 0,
            ndOuterExtents: step.ndOuterExtents,
            ndDstStrides: step.ndDstStrides,
          };
          ops.push({ type: "dus", dus });
          break;
        }
        case "scatter_add": {
          const targetIdx = idToIdx.get(step.target)!;
          const indicesIdx = idToIdx.get(step.indices)!;
          const updatesIdx = idToIdx.get(step.updates)!;
          const outIdx = idToIdx.get(step.output)!;
          const targetBytes = prod(step.targetShape) * byteWidth(step.dtype);
          const { pipeline, grid } = this.#resolveScatterAddPipeline(
            step.axis,
            step.targetShape,
            step.updatesLen,
            step.dtype,
          );
          const scatterAdd: TapeScatterAdd = {
            targetIdx,
            indicesIdx,
            updatesIdx,
            outIdx,
            targetBytes,
            pipeline,
            bindGroupLayout: pipeline.getBindGroupLayout(0),
            grid,
          };
          ops.push({ type: "scatter_add", scatterAdd });
          break;
        }
        case "reverse": {
          const inputIdx = idToIdx.get(step.input)!;
          const outIdx = idToIdx.get(step.output)!;
          const reverse: TapeReverse = {
            inputIdx,
            outIdx,
            axisSize: step.axisSize as number,
            innerBytes: step.innerBytes,
          };
          ops.push({ type: "reverse", reverse });
          break;
        }
        case "scan": {
          const tapeScan: TapeScan = {
            step,
            constIdxs: step.consts.map((id) => idToIdx.get(id)!),
            initCarryIdxs: step.initCarry.map((id) => idToIdx.get(id)!),
            xsIdxs: step.xs.map((id) => idToIdx.get(id)!),
            carryOutIdxs: step.outputs
              .slice(0, step.numCarry)
              .map((id) => idToIdx.get(id)!),
            ysStackedIdxs: step.outputs
              .slice(step.numCarry)
              .map((id) => idToIdx.get(id)!),
          };
          ops.push({ type: "scan", scan: tapeScan });
          break;
        }
      }
    }

    // 3. Build output mapping and filter out output frees from ops
    const outputTableIdxs = outputIds.map((id) => idToIdx.get(id)!);
    const outputIdxSet = new Set(outputTableIdxs);
    const safeOps = ops.filter(
      (op) => op.type !== "free" || !outputIdxSet.has(op.tableIdx),
    );

    // 4. Build constants slab (O9c): pack all initialData mallocs into a
    //    single GPU buffer with 256-byte-aligned offsets. This eliminates
    //    per-invocation mapped buffer creation for scalar-promoted constants.
    let constSlab: { buffer: GPUBuffer; entries: ConstSlabEntry[] } | null =
      null;
    {
      const SLAB_ALIGN = 256; // minStorageBufferOffsetAlignment
      const entries: ConstSlabEntry[] = [];
      const dataChunks: { data: Uint8Array<ArrayBuffer>; offset: number }[] =
        [];
      let slabSize = 0;

      for (const op of safeOps) {
        if (op.type !== "malloc" || !op.malloc.initialData) continue;
        const m = op.malloc;
        const data = m.initialData!;
        entries.push({
          tableIdx: m.tableIdx,
          offset: slabSize,
          bindSize: m.paddedSize,
          originalSize: m.originalSize,
        });
        dataChunks.push({ data, offset: slabSize });
        // Advance by 256-byte-aligned stride
        slabSize += Math.ceil(m.paddedSize / SLAB_ALIGN) * SLAB_ALIGN;
        // Mark as slab-allocated so executeTape skips per-invocation creation
        m.slabAllocated = true;
      }

      if (entries.length > 0) {
        slabSize = Math.max(slabSize, 4); // WebGPU minimum buffer size
        const slabBuffer = this.#createBuffer(slabSize, { mapped: true });
        const mapped = new Uint8Array(slabBuffer.getMappedRange());
        for (const { data, offset } of dataChunks) {
          mapped.set(data, offset);
        }
        slabBuffer.unmap();
        constSlab = { buffer: slabBuffer, entries };
      }
    }

    // 5. Build colored arena slabs (O9a-v2): partition internal intermediates
    //    by conflict-graph color so that no single GPUBuffer appears in both
    //    read-only-storage and storage bindings within a dispatch. Each color
    //    gets one persistent GPUBuffer slab; entries are 256-byte aligned.
    //    This makes buffer identities stable across invocations, enabling
    //    100% bind group cache hits for internal-only dispatches.
    let arenaSlabs: ArenaSlab[] | null = null;
    {
      // Build a temporary tape object for the coloring function.
      const tempTape: WebGPUCommandTape = {
        ops: safeOps,
        tableSize: nextIdx,
        inputTableIdxs,
        outputTableIdxs,
        outputSizes: outputTableIdxs.map((idx) => knownSizes.get(idx)!),
        allocatedIdxs,
        uniformBuffers,
        constSlab,
        arenaSlabs: null,
      };
      const coloring = buildConflictGraphAndColor(tempTape);

      if (coloring.numColors > 0) {
        const SLAB_ALIGN = 256; // minStorageBufferOffsetAlignment
        const MAX_COLORS = 4; // spill to discrete pool above this

        // Collect padded sizes for each table index from malloc ops.
        const idxPaddedSize = new Map<number, number>();
        const idxOriginalSize = new Map<number, number>();
        const idxToMalloc = new Map<number, TapeOp & { type: "malloc" }>();
        for (const op of safeOps) {
          if (op.type === "malloc" && !op.malloc.slabAllocated) {
            idxPaddedSize.set(op.malloc.tableIdx, op.malloc.paddedSize);
            idxOriginalSize.set(op.malloc.tableIdx, op.malloc.originalSize);
            idxToMalloc.set(
              op.malloc.tableIdx,
              op as TapeOp & { type: "malloc" },
            );
          }
        }

        const slabs: ArenaSlab[] = [];
        for (let c = 0; c < coloring.numColors && c < MAX_COLORS; c++) {
          const group = coloring.colorGroups[c];
          const entries: ArenaSlabEntry[] = [];
          let slabSize = 0;

          for (const idx of group) {
            const paddedSize = idxPaddedSize.get(idx);
            if (paddedSize === undefined || paddedSize === 0) continue;
            entries.push({
              tableIdx: idx,
              offset: slabSize,
              bindSize: paddedSize,
              originalSize: idxOriginalSize.get(idx)!,
            });
            // Advance by 256-byte-aligned stride
            slabSize += Math.ceil(paddedSize / SLAB_ALIGN) * SLAB_ALIGN;
          }

          if (entries.length === 0) continue;
          slabSize = Math.max(slabSize, 4); // WebGPU minimum buffer size
          const slabBuffer = this.#createBuffer(slabSize);
          slabs.push({ buffer: slabBuffer, entries });

          // Mark mallocs as arena-allocated
          for (const entry of entries) {
            const mallocOp = idxToMalloc.get(entry.tableIdx);
            if (mallocOp) mallocOp.malloc.arenaAllocated = true;
          }
        }

        arenaSlabs = slabs.length > 0 ? slabs : null;
      }
    }

    return {
      ops: safeOps,
      tableSize: nextIdx,
      inputTableIdxs,
      outputTableIdxs,
      outputSizes: outputTableIdxs.map((idx) => knownSizes.get(idx)!),
      allocatedIdxs,
      uniformBuffers,
      constSlab,
      arenaSlabs,
    };
  }

  /**
   * Destroy all GPU resources owned by a command tape.
   *
   * Called during cache eviction (`clearCaches()`) to prevent GPU memory leaks.
   * Destroys uniform buffers, constants slab, and arena slabs. After this call
   * the tape must not be executed again.
   */
  destroyCommandTapeResources(tape: WebGPUCommandTape): void {
    for (const buf of tape.uniformBuffers) {
      this.#gpuAllocatedBytes -= buf.size;
      buf.destroy();
    }
    tape.uniformBuffers.length = 0;

    if (tape.constSlab) {
      this.#gpuAllocatedBytes -= tape.constSlab.buffer.size;
      tape.constSlab.buffer.destroy();
      tape.constSlab = null;
    }

    if (tape.arenaSlabs) {
      for (const slab of tape.arenaSlabs) {
        this.#gpuAllocatedBytes -= slab.buffer.size;
        slab.buffer.destroy();
      }
      tape.arenaSlabs = null;
    }
  }

  /**
   * Execute a pre-compiled command tape with the given input slots.
   *
   * Each intermediate gets its own pooled GPUBuffer. Dispatches use bind group
   * caching (O9b) — when the pool returns the same GPUBuffer objects as the
   * previous invocation (common with LIFO pool ordering), the cached bind
   * group is reused, skipping `device.createBindGroup()`.
   *
   * Error-safe: if any step throws, all allocated GPU buffers are cleaned up.
   */
  executeCommandTape(
    tape: WebGPUCommandTape,
    inputSlots: Slot[],
  ): { outputs: Slot[]; pending: IPendingExecute[] } {
    // Execute the tape immediately and return results.
    // Must NOT use PendingCommandTape/beginBatch — deferred execution breaks
    // intra-tape pool reuse (ownEncoder=false → freed buffers deferred →
    // subsequent mallocs can't reuse → extra createBuffer per fori iteration).
    const outputs = this._executeCommandTapeImmediate(tape, inputSlots);
    return { outputs, pending: [] };
  }

  /**
   * Execute a command tape immediately with ownEncoder=true, creating output
   * slots directly.  This is the hot path for cached JitProgram execution.
   */
  _executeCommandTapeImmediate(
    tape: WebGPUCommandTape,
    inputSlots: Slot[],
  ): Slot[] {
    // Parallel arrays indexed by table position:
    //   buffers[i]   — GPUBuffer for this table entry
    //   sizes[i]     — original (unpadded) byte size for output slot creation
    //   offsets[i]   — byte offset within buffer for bind group entry (O9c slab)
    //   bindSizes[i] — bind size for bind group entry (0 = use whole buffer)
    const buffers: GPUBuffer[] = new globalThis.Array(tape.tableSize);
    const sizes: number[] = new globalThis.Array(tape.tableSize);
    const offsets: number[] = new globalThis.Array(tape.tableSize).fill(0);
    const bindSizes: number[] = new globalThis.Array(tape.tableSize).fill(0);

    // Pre-populate constants slab entries (O9c): these are stable across
    // invocations — same GPUBuffer at same offsets — so O9b cache hits are
    // guaranteed for dispatches that only reference slab + stable pool buffers.
    const slabBuf = tape.constSlab?.buffer;
    if (tape.constSlab) {
      for (const e of tape.constSlab.entries) {
        buffers[e.tableIdx] = slabBuf!;
        sizes[e.tableIdx] = e.originalSize;
        offsets[e.tableIdx] = e.offset;
        bindSizes[e.tableIdx] = e.bindSize;
      }
    }

    // Pre-populate colored arena slab entries (O9a-v2): persistent GPUBuffers
    // at fixed offsets make all internal dispatches hit the O9b bind group
    // cache after the first invocation.
    const arenaBufferSet = new Set<GPUBuffer>();
    if (tape.arenaSlabs) {
      for (const slab of tape.arenaSlabs) {
        arenaBufferSet.add(slab.buffer);
        for (const e of slab.entries) {
          buffers[e.tableIdx] = slab.buffer;
          sizes[e.tableIdx] = e.originalSize;
          offsets[e.tableIdx] = e.offset;
          bindSizes[e.tableIdx] = e.bindSize;
        }
      }
    }

    // Map external inputs
    for (let i = 0; i < inputSlots.length; i++) {
      const { buffer, size } = this.#getBuffer(inputSlots[i]);
      const idx = tape.inputTableIdxs[i];
      buffers[idx] = buffer;
      sizes[idx] = size;
      // offsets[idx] stays 0, bindSizes[idx] stays 0 (whole buffer)
    }

    // Track allocated buffers for error cleanup.
    const allocs: GPUBuffer[] = [];
    // Buffers freed mid-tape must NOT be destroyed until after queue.submit()
    // because earlier dispatches in the same GPUCommandEncoder still reference
    // them.  Defer destruction to post-submit.
    const deferredDestroys: GPUBuffer[] = [];
    // Temporary uniform buffers created by #encodeCopyAuto for unaligned
    // copies (O8e).  Must survive until after queue.submit().
    const copyUniforms: GPUBuffer[] = [];
    // Track buffers pushed to pool during this tape execution so we can
    // remove them on error (prevents pool poisoning with destroyed refs).
    const pooledDuringTape = new Set<GPUBuffer>();
    let submitted = false;
    const ownEncoder = !this.#batchEncoder;

    try {
      const encoder = this.#batchEncoder ?? this.device.createCommandEncoder();

      for (const op of tape.ops) {
        switch (op.type) {
          case "malloc": {
            const m = op.malloc;
            // O9c: slab-allocated constants are pre-populated above.
            if (m.slabAllocated) break;
            // O9a-v2: arena-allocated intermediates are pre-populated above.
            if (m.arenaAllocated) break;
            if (m.paddedSize === 0) {
              buffers[m.tableIdx] = this.#reusableZsb;
            } else {
              let buf: GPUBuffer;
              if (m.initialData) {
                // Fallback for non-slab initialData (shouldn't happen with
                // O9c, but kept for safety).
                buf = this.#createBuffer(m.paddedSize, { mapped: true });
                new Uint8Array(
                  buf.getMappedRange(),
                  0,
                  m.initialData.byteLength,
                ).set(m.initialData);
                buf.unmap();
              } else {
                buf =
                  this.#poolPop(m.paddedSize) ??
                  this.#createBuffer(m.paddedSize);
              }
              buffers[m.tableIdx] = buf;
              allocs.push(buf);
            }
            sizes[m.tableIdx] = m.originalSize;
            break;
          }
          case "free": {
            const buf = buffers[op.tableIdx];
            // Skip slab buffer — owned by tape, not per-invocation.
            if (buf === slabBuf) break;
            // Skip arena buffers — owned by tape, not per-invocation.
            if (arenaBufferSet.has(buf)) break;
            if (buf && buf !== this.#reusableZsb) {
              if (ownEncoder) {
                if (!this.#poolPush(buf)) {
                  deferredDestroys.push(buf);
                } else {
                  pooledDuringTape.add(buf);
                }
              } else {
                this.#batchDeferredFreeBuffers.push(buf);
              }
            }
            break;
          }
          case "recycle": {
            // O9a-v2: if toIdx is arena-allocated, it already has a
            // pre-assigned buffer — skip the recycle. fromIdx is also arena
            // (recycle chains bridging arena/external are excluded from
            // coloring), so its "free" was already a no-op.
            const toBuf = buffers[op.toIdx];
            if (toBuf && arenaBufferSet.has(toBuf)) break;
            buffers[op.toIdx] = buffers[op.fromIdx];
            sizes[op.toIdx] = sizes[op.fromIdx];
            offsets[op.toIdx] = offsets[op.fromIdx];
            bindSizes[op.toIdx] = bindSizes[op.fromIdx];
            break;
          }
          case "dispatch": {
            const d = op.dispatch;
            const numIn = d.inputIdxs.length;
            const numOut = d.outputIdxs.length;
            const totalBindings = numIn + numOut;

            // O9b: bind group cache — reuse when all referenced GPUBuffers
            // are the same objects as the previous invocation. With O9c slab,
            // constant inputs always match (same GPUBuffer), so cache hit rate
            // increases significantly.
            let bindGroup: GPUBindGroup;
            let cacheHit = false;
            const cached = d._bgCache;
            if (cached && cached.key.length === totalBindings) {
              cacheHit = true;
              for (let j = 0; j < numIn; j++) {
                if (cached.key[j] !== buffers[d.inputIdxs[j]]) {
                  cacheHit = false;
                  break;
                }
              }
              if (cacheHit) {
                for (let j = 0; j < numOut; j++) {
                  if (cached.key[numIn + j] !== buffers[d.outputIdxs[j]]) {
                    cacheHit = false;
                    break;
                  }
                }
              }
            }

            if (cacheHit) {
              bindGroup = cached!.value;
            } else {
              const entries: GPUBindGroupEntry[] = new globalThis.Array(
                totalBindings,
              );
              for (let i = 0; i < numIn; i++) {
                const idx = d.inputIdxs[i];
                const bsz = bindSizes[idx];
                entries[i] = {
                  binding: i,
                  resource:
                    bsz > 0
                      ? {
                          buffer: buffers[idx],
                          offset: offsets[idx],
                          size: bsz,
                        }
                      : { buffer: buffers[idx] },
                };
              }
              for (let i = 0; i < numOut; i++) {
                const idx = d.outputIdxs[i];
                const bsz = bindSizes[idx];
                entries[numIn + i] = {
                  binding: numIn + i,
                  resource:
                    bsz > 0
                      ? {
                          buffer: buffers[idx],
                          offset: offsets[idx],
                          size: bsz,
                        }
                      : { buffer: buffers[idx] },
                };
              }
              bindGroup = this.device.createBindGroup({
                layout: d.bindGroupLayout,
                entries,
              });

              // Cache the bind group keyed by GPUBuffer identity
              const key: GPUBuffer[] = new globalThis.Array(totalBindings);
              for (let i = 0; i < numIn; i++) key[i] = buffers[d.inputIdxs[i]];
              for (let i = 0; i < numOut; i++)
                key[numIn + i] = buffers[d.outputIdxs[i]];
              d._bgCache = { key, value: bindGroup };
            }

            for (let i = 0; i < d.passes.length; i++) {
              const pe = _beginComputePass(encoder, this.#profiling, {
                grid: d.passes[i].grid,
              });
              pe.setPipeline(d.pipeline);
              pe.setBindGroup(0, bindGroup);
              if (d.uniformBindGroup) {
                pe.setBindGroup(1, d.uniformBindGroup, [
                  i * d.uniformAlignment,
                ]);
              }
              pe.dispatchWorkgroups(d.passes[i].grid[0], d.passes[i].grid[1]);
              _dispatchCount++;
              pe.end();
            }
            break;
          }
          case "dus": {
            const d = op.dus;
            const dstBuf = buffers[d.dstIdx];
            const dstOff = offsets[d.dstIdx];
            const srcBuf = buffers[d.srcIdx];
            const srcOff = offsets[d.srcIdx];
            const outBuf = buffers[d.outIdx];
            const outOff = offsets[d.outIdx];

            // Copy dst → output (skip if recycled: same buffer + offset)
            if (dstBuf !== outBuf || dstOff !== outOff) {
              const u = this.#encodeCopyAuto(
                encoder,
                dstBuf,
                dstOff,
                outBuf,
                outOff,
                d.dstSizeBytes,
              );
              if (u) copyUniforms.push(u);
            }
            // Copy src slice into output at offsetBytes
            if (d.outerFibers === 1) {
              // Contiguous fast path (axis=0)
              const u = this.#encodeCopyAuto(
                encoder,
                srcBuf,
                srcOff + d.srcBaseOffset,
                outBuf,
                outOff + d.offsetBytes,
                d.sliceBytes,
              );
              if (u) copyUniforms.push(u);
            } else if (d.ndOuterExtents) {
              // ND fiber loop: coordinate decomposition for non-constant dst stride
              const extents = d.ndOuterExtents;
              const dstStrides = d.ndDstStrides!;
              const rowBytes = d.srcFiberBytes;
              // Precompute radix divisors: divisors[k] = prod(extents[k+1:])
              const divisors = new Array<number>(extents.length);
              divisors[extents.length - 1] = 1;
              for (let k = extents.length - 2; k >= 0; k--)
                divisors[k] = divisors[k + 1] * extents[k + 1];
              for (let i = 0; i < d.outerFibers; i++) {
                let rem = i;
                let dstFiberOff = d.offsetBytes;
                for (let k = 0; k < extents.length; k++) {
                  const coord = Math.floor(rem / divisors[k]);
                  rem = rem % divisors[k];
                  dstFiberOff += coord * dstStrides[k];
                }
                const u = this.#encodeCopyAuto(
                  encoder,
                  srcBuf,
                  srcOff + i * rowBytes,
                  outBuf,
                  outOff + dstFiberOff,
                  rowBytes,
                );
                if (u) copyUniforms.push(u);
              }
            } else {
              // Fiber-by-fiber copy for non-contiguous axis > 0
              for (let i = 0; i < d.outerFibers; i++) {
                const u = this.#encodeCopyAuto(
                  encoder,
                  srcBuf,
                  srcOff + d.srcBaseOffset + i * d.srcFiberBytes,
                  outBuf,
                  outOff + i * d.dstFiberBytes + d.offsetBytes,
                  d.srcFiberBytes,
                );
                if (u) copyUniforms.push(u);
              }
            }
            break;
          }
          case "scatter_add": {
            const sa = op.scatterAdd;
            const targetBuf = buffers[sa.targetIdx];
            const targetOff = offsets[sa.targetIdx];
            const outBuf = buffers[sa.outIdx];
            const outOff = offsets[sa.outIdx];

            // Copy target → output (skip if recycled)
            if (targetBuf !== outBuf || targetOff !== outOff) {
              const u = this.#encodeCopyAuto(
                encoder,
                targetBuf,
                targetOff,
                outBuf,
                outOff,
                sa.targetBytes,
              );
              if (u) copyUniforms.push(u);
            }

            // Dispatch scatter_add kernel
            const idxBuf = buffers[sa.indicesIdx];
            const idxOff = offsets[sa.indicesIdx];
            const idxBsz = bindSizes[sa.indicesIdx];
            const updBuf = buffers[sa.updatesIdx];
            const updOff = offsets[sa.updatesIdx];
            const updBsz = bindSizes[sa.updatesIdx];
            const outBsz = bindSizes[sa.outIdx];

            const bindGroup = this.device.createBindGroup({
              layout: sa.bindGroupLayout,
              entries: [
                {
                  binding: 0,
                  resource:
                    outBsz > 0
                      ? { buffer: outBuf, offset: outOff, size: outBsz }
                      : { buffer: outBuf },
                },
                {
                  binding: 1,
                  resource:
                    idxBsz > 0
                      ? { buffer: idxBuf, offset: idxOff, size: idxBsz }
                      : { buffer: idxBuf },
                },
                {
                  binding: 2,
                  resource:
                    updBsz > 0
                      ? { buffer: updBuf, offset: updOff, size: updBsz }
                      : { buffer: updBuf },
                },
              ],
            });
            const pe = _beginComputePass(encoder, this.#profiling, {
              grid: sa.grid,
              label: "scatter_add",
            });
            pe.setPipeline(sa.pipeline);
            pe.setBindGroup(0, bindGroup);
            pe.dispatchWorkgroups(sa.grid[0], sa.grid[1]);
            _dispatchCount++;
            pe.end();
            break;
          }
          case "reverse": {
            const r = op.reverse;
            const inBuf = buffers[r.inputIdx];
            const inOff = offsets[r.inputIdx];
            const outBuf = buffers[r.outIdx];
            const outOff = offsets[r.outIdx];

            // Copy slices in reverse order
            for (let i = 0; i < r.axisSize; i++) {
              const u = this.#encodeCopyAuto(
                encoder,
                inBuf,
                inOff + i * r.innerBytes,
                outBuf,
                outOff + (r.axisSize - 1 - i) * r.innerBytes,
                r.innerBytes,
              );
              if (u) copyUniforms.push(u);
            }
            break;
          }
          case "scan": {
            const { scan } = op;
            const consts = scan.constIdxs.map((i) => buffers[i]);
            const initCarry = scan.initCarryIdxs.map((i) => buffers[i]);
            const xs = scan.xsIdxs.map((i) => buffers[i]);
            const carryOut = scan.carryOutIdxs.map((i) => buffers[i]);
            const ysStacked = scan.ysStackedIdxs.map((i) => buffers[i]);

            const tapeCtx = {
              encoder,
              deferredDestroys,
              copyUniforms,
              pooledDuringTape,
            };

            const plan = scan.step.plan;
            if (plan.path === "compiled-loop") {
              this.dispatchNativeScanGeneral(
                plan.executable,
                plan.params as NativeScanMultiParams,
                consts,
                initCarry,
                xs,
                carryOut,
                ysStacked,
                tapeCtx,
              );
            } else if (plan.path === "preencoded-multi-step") {
              this.dispatchPreencodedMultiStepScan(
                plan.prepared as PreparedPreencodedMultiStep,
                consts,
                initCarry,
                xs,
                carryOut,
                ysStacked,
                tapeCtx,
              );
            } else if (plan.path === "preencoded-routine") {
              this.dispatchPreencodedScan(
                plan.preencodedParams as PreparedPreencodedScan,
                consts,
                initCarry,
                xs,
                carryOut,
                ysStacked,
                tapeCtx,
              );
            }
            break;
          }
        }
      }

      if (ownEncoder) {
        this.device.queue.submit([encoder.finish()]);
      }
      submitted = true;
    } finally {
      if (!submitted) {
        // Remove buffers pushed to pool during this (failed) tape execution
        // to prevent pool poisoning (destroyed GPUBuffer refs in pool).
        for (const buf of pooledDuringTape) {
          this.#poolRemove(buf);
        }
        for (const buf of allocs) {
          if (buf !== this.#reusableZsb) {
            this.#gpuAllocatedBytes -= buf.size;
            buf.destroy();
          }
        }
        // Destroy any copy-shader uniforms created before the failure.
        for (const buf of copyUniforms) buf.destroy();
      }
    }

    if (ownEncoder) {
      // Now safe to destroy buffers that couldn't fit in the pool — the command
      // buffer has been submitted, so references are no longer live.
      for (const buf of deferredDestroys) {
        this.#gpuAllocatedBytes -= buf.size;
        buf.destroy();
      }
      // Destroy temporary uniform buffers from unaligned copy shader (O8e).
      for (const buf of copyUniforms) buf.destroy();
    } else {
      // Pass copy-shader uniforms to batch cleanup.
      this.#batchUniformsToDestroy.push(...copyUniforms);
    }

    // Create output slots
    const outputs: Slot[] = new globalThis.Array(tape.outputTableIdxs.length);
    for (let i = 0; i < tape.outputTableIdxs.length; i++) {
      const idx = tape.outputTableIdxs[i];
      const slot = this.nextSlot++;
      this.buffers.set(slot, {
        buffer: buffers[idx],
        size: sizes[idx],
        ref: 1,
      });
      outputs[i] = slot;
    }
    return outputs;
  }

  /**
   * Deferred command tape execution — used by PendingCommandTape.submit().
   * Populates pre-created output slots instead of creating new ones.
   */
  _executeCommandTapeDeferred(
    tape: WebGPUCommandTape,
    inputSlots: Slot[],
    outputSlots: Slot[],
    _outputTableIdxs: number[],
  ) {
    // Delegate to the immediate path, then transfer buffers to pre-created slots.
    const immediateOutputs = this._executeCommandTapeImmediate(
      tape,
      inputSlots,
    );
    for (let i = 0; i < outputSlots.length; i++) {
      const entry = this.buffers.get(outputSlots[i]);
      if (entry) {
        const immediateEntry = this.buffers.get(immediateOutputs[i]);
        if (immediateEntry) {
          entry.buffer = immediateEntry.buffer;
          entry.size = immediateEntry.size;
          // Remove the immediate slot — ownership transferred to pre-created slot
          this.buffers.delete(immediateOutputs[i]);
        }
      } else {
        // Pre-created slot was disposed — clean up the immediate slot
        this.decRef(immediateOutputs[i]);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // ScatterAdd dispatch (M2: scatter_add primitive)
  // ---------------------------------------------------------------------------
  #scatterAddPipelineCache = new Map<string, GPUComputePipeline>();

  /**
   * Resolve (get or create) a scatter_add pipeline for the given parameters.
   * Returns the pipeline and pre-computed grid dimensions.
   */
  #resolveScatterAddPipeline(
    axis: number,
    targetShape: number[],
    updatesLen: number,
    dtype: DType,
  ): { pipeline: GPUComputePipeline; grid: [number, number] } {
    const ndim = targetShape.length;
    const innerSize =
      ndim > 0 ? targetShape.slice(axis + 1).reduce((a, b) => a * b, 1) : 1;
    const outerSize =
      ndim > 0 ? targetShape.slice(0, axis).reduce((a, b) => a * b, 1) : 1;
    const axisSize = ndim > 0 ? targetShape[axis] : 1;

    const totalUpdates = updatesLen * outerSize * innerSize;
    const [gridX, gridY] = calculateGrid(Math.ceil(totalUpdates / 64));

    const useNativeF32Atomic =
      dtype === DType.Float32 && this.capabilities.atomicF32Add;

    const cacheKey = `scatter_add_${dtype}_${ndim}_${axis}_${outerSize}_${innerSize}_${axisSize}_${updatesLen}_${useNativeF32Atomic ? "native" : "cas"}`;
    let pipeline = this.#scatterAddPipelineCache.get(cacheKey);

    if (!pipeline) {
      const wgslType = dtypeToWgsl(dtype);
      const isFloat = dtype === DType.Float32 || dtype === DType.Float16;
      const atomicType = useNativeF32Atomic
        ? "f32"
        : isFloat
          ? "u32"
          : wgslType === "i32"
            ? "i32"
            : "u32";

      let code = useNativeF32Atomic ? "enable shader_f32_atomic_add;\n" : "";
      code += `
@group(0) @binding(0) var<storage, read_write> output: array<atomic<${atomicType}>>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<${wgslType}>;

const INNER: u32 = ${innerSize}u;
const OUTER: u32 = ${outerSize}u;
const AXIS_SIZE: u32 = ${axisSize}u;
const UPDATES_LEN: u32 = ${updatesLen}u;
const TOTAL: u32 = ${totalUpdates}u;
const TARGET_INNER_STRIDE: u32 = ${axisSize * innerSize}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let flat = gid.x + gid.y * ${gridX}u;
  if (flat >= TOTAL) { return; }

  let inner = flat % INNER;
  let tmp = flat / INNER;
  let updateIdx = tmp % UPDATES_LEN;
  let outer = tmp / UPDATES_LEN;

  let targetAxisIdx = u32(indices[updateIdx]);
  if (targetAxisIdx >= AXIS_SIZE) { return; }

  let outFlat = outer * TARGET_INNER_STRIDE + targetAxisIdx * INNER + inner;

  let val = updates[flat];
`;

      if (useNativeF32Atomic) {
        code += `
  atomicAdd(&output[outFlat], val);
`;
      } else if (isFloat) {
        code += `
  var old_bits = atomicLoad(&output[outFlat]);
  loop {
    let old_val = bitcast<${wgslType}>(old_bits);
    let new_val = old_val + val;
    let new_bits = bitcast<u32>(new_val);
    let result = atomicCompareExchangeWeak(&output[outFlat], old_bits, new_bits);
    if (result.exchanged) { break; }
    old_bits = result.old_value;
  }
`;
      } else {
        code += `
  atomicAdd(&output[outFlat], ${atomicType === "u32" ? "bitcast<u32>(val)" : "val"});
`;
      }

      code += `}\n`;

      const shaderModule = this.device.createShaderModule({ code });
      const layout = this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.device.createBindGroupLayout({
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
              },
            ],
          }),
        ],
      });
      pipeline = this.device.createComputePipeline({
        layout,
        compute: { module: shaderModule, entryPoint: "main" },
      });
      this.#scatterAddPipelineCache.set(cacheKey, pipeline);
    }

    return { pipeline, grid: [gridX, gridY] };
  }

  dispatchScatterAdd(
    output: Slot,
    indices: Slot,
    updates: Slot,
    axis: number,
    targetShape: number[],
    updatesLen: number,
    dtype: DType,
  ): void {
    if (dtype === DType.Float64) {
      throw new Error("ScatterAdd: Float64 not supported on WebGPU");
    }

    const { pipeline, grid } = this.#resolveScatterAddPipeline(
      axis,
      targetShape,
      updatesLen,
      dtype,
    );

    const outBuf = this.#getBuffer(output).buffer;
    const idxBuf = this.#getBuffer(indices).buffer;
    const updBuf = this.#getBuffer(updates).buffer;

    const commandEncoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outBuf } },
        { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: updBuf } },
      ],
    });
    const pass = _beginComputePass(commandEncoder, this.#profiling);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid[0], grid[1]);
    _dispatchCount++;
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  // ---------------------------------------------------------------------------
  // Decoupled Fallback prefix scan (T0: single-dispatch O(N) scan)
  // ---------------------------------------------------------------------------
  #dfScanPipelineCache = new Map<string, GPUComputePipeline>();

  dispatchDecoupledFallbackScan(
    input: Slot,
    output: Slot,
    N: number,
    op: AluOp,
    dtype: DType,
    blockSize: number,
  ): void {
    const M = Math.ceil(N / blockSize);

    // Get or create pipeline
    const cacheKey = `df_scan_${op}_${dtype}_${blockSize}`;
    let pipeline = this.#dfScanPipelineCache.get(cacheKey);
    if (!pipeline) {
      const code = generateDecoupledFallbackScanShader(
        op as DFScanOp,
        dtype as DFScanDtype,
        blockSize,
      );
      const shaderModule = this.device.createShaderModule({ code });
      const layout = this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.device.createBindGroupLayout({
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
              },
              {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" },
              },
            ],
          }),
        ],
      });
      pipeline = this.device.createComputePipeline({
        layout,
        compute: { module: shaderModule, entryPoint: "main" },
      });
      this.#dfScanPipelineCache.set(cacheKey, pipeline);
    }

    // Allocate descriptor buffer (M u32s, will be zeroed via clearBuffer)
    const descBytes = Math.max(M * 4, 4); // at least 4 bytes
    const descSlot = this.malloc(descBytes);

    // Allocate uniform buffer for params (N)
    // Minimum allocation: 256 bytes (minUniformBufferOffsetAlignment)
    const uniformSlot = this.malloc(256);
    const uniformBuf = this.#getBuffer(uniformSlot).buffer;
    this.device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([N]).buffer);

    // Build command buffer: clearBuffer(descriptors) → computePass
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.clearBuffer(this.#getBuffer(descSlot).buffer, 0, descBytes);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#getBuffer(input).buffer } },
        { binding: 1, resource: { buffer: this.#getBuffer(output).buffer } },
        {
          binding: 2,
          resource: {
            buffer: this.#getBuffer(descSlot).buffer,
            size: descBytes,
          },
        },
        { binding: 3, resource: { buffer: uniformBuf, size: 256 } },
      ],
    });
    const pass = _beginComputePass(commandEncoder, this.#profiling);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(M);
    _dispatchCount++;
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Free temporary buffers
    this.decRef(descSlot);
    this.decRef(uniformSlot);
  }

  #createBuffer(
    size: number,
    { mapped = false, read = false } = {},
  ): GPUBuffer {
    if (read && mapped) {
      throw new Error("mapped and read cannot both be true");
    }
    const buffer = this.device.createBuffer({
      size,
      usage: read
        ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.UNIFORM,
      mappedAtCreation: mapped,
    });
    if (!read) {
      this.#gpuAllocatedBytes += size;
      if (this.#gpuAllocatedBytes > this.#gpuPeakBytes) {
        this.#gpuPeakBytes = this.#gpuAllocatedBytes;
      }
    }
    return buffer;
  }
}

/**
 * Compiles a multi-output kernel into a single WebGPU shader.
 * Multi-output kernels are always non-reduction (enforced by jitCompile batching).
 * Generates one result binding per output and one store per output per gidx.
 */
function pipelineSourceMulti(
  device: GPUDevice,
  kernel: Kernel,
  caps?: BackendCapabilities,
): ShaderInfo {
  const { nargs } = kernel;
  const numOutputs = kernel.numOutputs;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // Tune each output individually
  const tunes = kernel.outputs.map((o) => {
    const tmp = Kernel.single(nargs, kernel.size, o.exp, o.reduction);
    return tuneWebgpu(tmp, caps);
  });

  // All outputs share the same threadCount (same size, no reductions)
  const threadCount = tunes[0].threadCount;

  const shader: string[] = [];
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };

  // Check for f16 across all outputs
  if (tunes.some((t) => t.exp.some((exp) => exp.dtype === DType.Float16))) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  // Collect all distinct ops across all output expressions
  let allOps: Map<AluOp, Set<DType>> = new Map();
  for (const tune of tunes) {
    allOps = mapSetUnion(allOps, tune.exp.distinctOps());
  }
  if (allOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc)) emit(erfSrc);
  emit("");

  // Find used args across all outputs
  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  for (const tune of tunes) {
    tune.exp.fold((exp) => {
      if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
    });
  }

  // Input bindings
  for (let i = 0; i < nargs; i++) {
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  // Output bindings (one per output)
  for (let oi = 0; oi < numOutputs; oi++) {
    const resultTy = dtypeToWgsl(kernel.outputs[oi].dtype, true);
    emit(
      `@group(0) @binding(${nargs + oi}) var<storage, read_write> result${oi} : array<${resultTy}>;`,
    );
  }

  const symbolic = isSymbolicSize(threadCount);

  let workgroupSize: number;
  let gridX: number;
  let gridY: number;

  if (symbolic) {
    workgroupSize = 256;
    gridX = 1;
    gridY = 1;
    emit(
      "",
      "struct Dims { total_size: u32 }",
      "@group(1) @binding(0) var<uniform> dims: Dims;",
    );
  } else {
    const tc = threadCount as number;
    workgroupSize = findPow2(tc, 256);
    const gridSize = Math.ceil(tc / workgroupSize);
    [gridX, gridY] = calculateGrid(gridSize);
  }

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    pushIndent,
  );

  if (symbolic) {
    const sizeX = gridOffsetY * workgroupSize;
    emit(
      `let linear_idx: u32 = ${sizeX}u * id.y + id.x;`,
      "if (linear_idx >= dims.total_size) { return; }",
      "let gidx: i32 = i32(linear_idx);",
    );
  } else {
    const tc = threadCount as number;
    if (gridY === 1) {
      emit(`if (id.x >= ${tc}) { return; }`, "let gidx: i32 = i32(id.x);");
    } else {
      const sizeX = gridX * workgroupSize;
      emit(
        `if (${sizeX} * id.y + id.x >= ${tc}) { return; }`,
        `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
      );
    }
  }

  // CSE infrastructure (shared across all outputs for subexpression sharing)
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;
  const isGensym = (text: string) => text.match(/^alu[0-9]+$/);

  // Phony assignments for unused args
  if (args.length > 0) {
    emit(args.map((arg) => `_ = &${arg};`).join(" "));
  }

  // Count references across ALL output expressions for correct CSE
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };
  for (const tune of tunes) {
    countReferences(tune.exp);
  }

  // AluExp → WGSL translator with CSE
  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expContext.has(exp)) return expContext.get(exp)!;
    const { op, src, dtype, arg } = exp;

    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `(${a} + ${b})`;
      } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `(${a} * ${b})`;
      } else if (op === AluOp.Idiv)
        source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `(${a} % ${b})`;
      else if (op === AluOp.Min) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `min(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Max) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `max(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) {
        if (isFloatDtype(src[0].dtype)) {
          const x = isGensym(a) ? a : gensym();
          if (x !== a) emit(`let ${x} = ${a};`);
          source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
        } else {
          source = `(${a} != ${b})`;
        }
      }
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
        const a = gen(src[0].src[0]);
        source = `inverseSqrt(${a})`;
      } else {
        const a = gen(src[0]);
        if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
        else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
        else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
        else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
        else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
        else if (op === AluOp.Log) source = `log(${strip1(a)})`;
        else if (op === AluOp.Erf || op === AluOp.Erfc) {
          const funcName = op === AluOp.Erf ? "erf" : "erfc";
          if (dtype !== DType.Float32) {
            source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
          } else {
            source = `${funcName}(${strip1(a)})`;
          }
        } else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
        else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
        else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
        else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
        else if (op === AluOp.Cast)
          source = castSaturateWgsl(strip1(a), src[0].dtype, dtype);
        else if (op === AluOp.Bitcast)
          source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
      }
    } else if (op === AluOp.Where) {
      source = `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    } else if (op === AluOp.Threefry2x32) {
      const x = gensym();
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
      if (dtype === DType.Bool) source = `(${source} != 0)`;
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    const typeName = dtypeToWgsl(dtype);
    if ((references.get(exp) ?? 0) > 1) {
      const name = gensym();
      expContext.set(exp, name);
      emit(`let ${name}: ${typeName} = ${strip1(source)};`);
      return name;
    } else {
      expContext.set(exp, source);
      return source;
    }
  };

  // Generate stores for each output
  for (let oi = 0; oi < numOutputs; oi++) {
    const resultTy = dtypeToWgsl(kernel.outputs[oi].dtype, true);
    let rhs = strip1(gen(tunes[oi].exp));
    if (resultTy !== dtypeToWgsl(tunes[oi].exp.dtype))
      rhs = `${resultTy}(${rhs})`;
    emit(`result${oi}[gidx] = ${rhs};`);
  }

  emit(popIndent, "}");
  return {
    code: withNanInfHeader(shader.join("\n")),
    numInputs: nargs,
    numOutputs: numOutputs,
    hasUniform: symbolic,
    passes: [{ grid: [gridX, gridY] }],
    isSymbolic: symbolic || undefined,
    workgroupSize: symbolic ? workgroupSize : undefined,
  };
}

/**
 * Compiles an expression into WebGPU shader source code.
 *
 * Returns the shader source and the number of workgroups to dispatch along x
 * and y axes, to run the kernel.
 */
function pipelineSource(
  device: GPUDevice,
  kernel: Kernel,
  caps?: BackendCapabilities,
): ShaderInfo {
  if (kernel.isMultiOutput) {
    return pipelineSourceMulti(device, kernel, caps);
  }

  const tune = tuneWebgpu(kernel, caps);
  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.outputs[0].exp}\ntune.exp: ${tune.exp}`);
  }

  const { nargs } = kernel;
  const re = kernel.outputs[0].reduction;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // Pre-compute subgroup eligibility (needed for enable directive).
  const _groupSizeForSg = re
    ? ((tune as WebGPUTuneResult).size.groups ?? 1)
    : 1;
  const useSubgroups =
    _groupSizeForSg > 1 &&
    re!.dtype !== DType.Bool &&
    device.features.has("subgroups");

  // binding(0..n-1): input buffers
  // binding(n): output buffer

  const shader: string[] = []; // line-separated
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };

  if (
    tune.exp.some((exp) => exp.dtype === DType.Float16) ||
    tune.epilogue?.some((exp) => exp.dtype === DType.Float16)
  ) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }
  if (useSubgroups) {
    emit("enable subgroups;");
  }

  // Global functions at the start of the shader.
  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  if (distinctOps.has(AluOp.Threefry2x32)) {
    emit(threefrySrc);
  }
  if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  // End global function definitions.
  emit("");

  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });
  tune.epilogue?.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });

  for (let i = 0; i < nargs; i++) {
    // If not used, just assume float32, all that matters is size / alignment.
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  const resultTy = dtypeToWgsl(kernel.outputs[0].dtype, true);
  emit(
    `@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`,
  );

  const symbolic = isSymbolicSize(tune.threadCount);
  const symbolicReduce = isSymbolicSize(tune.size.reduce);

  // For symbolic kernels, use a uniform buffer to pass the total element count
  // at dispatch time. For concrete kernels, bake the count into the shader.
  // When the reduction axis is symbolic, also pass reduce_size via uniform.
  let workgroupSize: number;
  let gridX: number;
  let gridY: number;

  if (symbolic || symbolicReduce) {
    workgroupSize = symbolic ? 256 : findPow2(tune.threadCount as number, 256);
    if (symbolic) {
      // Grid resolved at dispatch time from dynamicParams.
      gridX = 1;
      gridY = 1;
    } else {
      const tc = tune.threadCount as number;
      const gridSize = Math.ceil(tc / workgroupSize);
      [gridX, gridY] = calculateGrid(gridSize);
    }
    // Build struct Dims with the fields we need
    const dimsFields: string[] = [];
    if (symbolic) dimsFields.push("total_size: u32");
    if (symbolicReduce) dimsFields.push("reduce_size: u32");
    emit(
      "",
      `struct Dims { ${dimsFields.join(", ")} }`,
      "@group(1) @binding(0) var<uniform> dims: Dims;",
    );
  } else {
    const threadCount = tune.threadCount as number;
    const tuneLocal = tune.size.local ?? 1;
    const tuneGroups = (tune as WebGPUTuneResult).size.groups ?? 1;
    const maxWgSize = device.limits.maxComputeWorkgroupSizeX;
    if (tuneGroups > 1) {
      workgroupSize = Math.min(tuneGroups, maxWgSize);
    } else if (tuneLocal > 1) {
      workgroupSize = Math.min(tuneLocal, maxWgSize);
    } else {
      workgroupSize = findPow2(threadCount, Math.min(256, maxWgSize));
    }
    const gridSize = Math.ceil(threadCount / workgroupSize);
    [gridX, gridY] = calculateGrid(gridSize);
  }

  // Shared memory for cooperative group reductions.
  const groupSize = re ? ((tune as WebGPUTuneResult).size.groups ?? 1) : 1;
  const useSharedMem = groupSize > 1;
  if (useSharedMem) {
    const upcast = (tune as WebGPUTuneResult).size.upcast ?? 1;
    const shmemTy = dtypeToWgsl(re!.dtype);
    emit(`var<workgroup> shmem: array<${shmemTy}, ${groupSize * upcast}>;`);
  }

  emit("", `@compute @workgroup_size(${workgroupSize})`);
  if (useSubgroups) {
    emit(
      "fn main(@builtin(global_invocation_id) id : vec3<u32>, @builtin(subgroup_size) sg_size: u32) {",
      pushIndent,
    );
  } else {
    emit(
      "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
      pushIndent,
    );
  }

  if (symbolic) {
    // For symbolic: always use 2D formula (works for 1D when id.y=0).
    const sizeX = gridOffsetY * workgroupSize;
    emit(
      `let linear_idx: u32 = ${sizeX}u * id.y + id.x;`,
      "if (linear_idx >= dims.total_size) { return; }",
      "let gidx: i32 = i32(linear_idx);",
    );
  } else {
    const threadCount = tune.threadCount as number;
    if (gridY === 1) {
      if (useSharedMem) {
        // Use a validity flag instead of early return so all threads
        // reach workgroupBarrier() (WGSL uniform control flow requirement).
        emit(
          `let _valid: bool = id.x < ${threadCount}u;`,
          `let gidx: i32 = i32(id.x / ${groupSize}u);`,
          `let group: i32 = i32(id.x % ${groupSize}u);`,
        );
      } else {
        emit(`if (id.x >= ${threadCount}u) { return; }`);
        emit("let gidx: i32 = i32(id.x);");
      }
    } else {
      const sizeX = gridX * workgroupSize;
      if (useSharedMem) {
        // Use a validity flag instead of early return so all threads
        // reach workgroupBarrier() (WGSL uniform control flow requirement).
        emit(
          `let _tid: u32 = ${sizeX}u * id.y + id.x;`,
          `let _valid: bool = _tid < ${threadCount}u;`,
          `let gidx: i32 = i32(_tid / ${groupSize}u);`,
          `let group: i32 = i32(_tid % ${groupSize}u);`,
        );
      } else {
        emit(
          `if (${sizeX} * id.y + id.x >= ${threadCount}) { return; }`,
          `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
        );
      }
    }
  }

  // Generate code for each AluExp operation.
  // Some expressions may be used twice, so we keep track of them.
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;
  const isGensym = (text: string) => text.match(/^alu[0-9]+$/);

  // Insert phony assignments, in case some inputs are not in use.
  // https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686
  if (args.length > 0) {
    emit(args.map((arg) => `_ = &${arg};`).join(" "));
  }

  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expContext.has(exp)) return expContext.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them.
    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `(${a} + ${b})`;
      } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `(${a} * ${b})`;
      } else if (op === AluOp.Idiv)
        source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `(${a} % ${b})`;
      else if (op === AluOp.Min) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `min(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Max) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `max(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.BitCombine) {
        if (arg === "and") source = `(${a} & ${b})`;
        else if (arg === "or") source = `(${a} | ${b})`;
        else source = dtype === DType.Bool ? `(${a} != ${b})` : `(${a} ^ ${b})`;
      } else if (op === AluOp.BitShift) {
        if (arg === "shl") source = `(${a} << ${b})`;
        else source = `(${a} >> ${b})`;
      } else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) {
        // Edge case: WebGPU doesn't handle NaN correctly, it's unspecified.
        // Use bitcast to check the IEEE 754 bit pattern instead:
        // f32 NaN: exponent all-1s (0x7F800000) AND mantissa non-zero.
        // f16 NaN: exponent all-1s (0x7C00) AND mantissa non-zero.
        if (isFloatDtype(src[0].dtype)) {
          const x = isGensym(a) ? a : gensym();
          if (x !== a) emit(`let ${x} = ${a};`);
          const y = isGensym(b) ? b : gensym();
          if (y !== b) emit(`let ${y} = ${b};`);
          if (src[0].dtype === DType.Float16) {
            const bitsX = gensym();
            const bitsY = gensym();
            emit(
              `let ${bitsX} = bitcast<u32>(vec2<f16>(${x}, f16(0.0))) & 0xFFFFu;`,
            );
            emit(
              `let ${bitsY} = bitcast<u32>(vec2<f16>(${y}, f16(0.0))) & 0xFFFFu;`,
            );
            const isNanX = `((${bitsX} & 0x7C00u) == 0x7C00u && (${bitsX} & 0x03FFu) != 0u)`;
            const isNanY = `((${bitsY} & 0x7C00u) == 0x7C00u && (${bitsY} & 0x03FFu) != 0u)`;
            source = `(${x} != ${y} || ${isNanX} || ${isNanY})`;
          } else {
            const bitsX = gensym();
            const bitsY = gensym();
            emit(`let ${bitsX} = bitcast<u32>(${x});`);
            emit(`let ${bitsY} = bitcast<u32>(${y});`);
            const isNanX = `((${bitsX} & 0x7F800000u) == 0x7F800000u && (${bitsX} & 0x007FFFFFu) != 0u)`;
            const isNanY = `((${bitsY} & 0x7F800000u) == 0x7F800000u && (${bitsY} & 0x007FFFFFu) != 0u)`;
            source = `(${x} != ${y} || ${isNanX} || ${isNanY})`;
          }
        } else {
          source = `(${a} != ${b})`;
        }
      }
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
        // Special case: 1/sqrt(x) is optimized as rsqrt(x)
        const a = gen(src[0].src[0]);
        source = `inverseSqrt(${a})`;
      } else {
        const a = gen(src[0]);
        if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
        else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
        else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
        else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
        else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
        else if (op === AluOp.Log) source = `log(${strip1(a)})`;
        else if (op === AluOp.Erf || op === AluOp.Erfc) {
          const funcName = op === AluOp.Erf ? "erf" : "erfc";
          if (dtype !== DType.Float32) {
            // Always compute special functions in f32 for precision.
            source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
          } else {
            source = `${funcName}(${strip1(a)})`;
          }
        } else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
        else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
        else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
        else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
        else if (op === AluOp.Cast)
          source = castSaturateWgsl(strip1(a), src[0].dtype, dtype);
        else if (op === AluOp.Bitcast)
          source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
      }
    } else if (op === AluOp.Where) {
      // select(f, t, cond) -> cond ? t : f
      source = `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    } else if (op === AluOp.Threefry2x32) {
      const x = gensym(); // temporary to hold the `vec2<u32>(x0, x1)`
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
      if (dtype === DType.Bool) source = `(${source} != 0)`; // bool is represented as i32
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    const typeName = dtypeToWgsl(dtype);
    if ((references.get(exp) ?? 0) > 1) {
      const name = gensym();
      expContext.set(exp, name);
      emit(`let ${name}: ${typeName} = ${strip1(source)};`);
      return name;
    } else {
      expContext.set(exp, source);
      return source;
    }
  };

  if (!re) {
    countReferences(tune.exp);
    let rhs = strip1(gen(tune.exp));
    if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
    emit(`result[gidx] = ${rhs};`);
  } else {
    const unroll = tune.size.unroll ?? 1;
    const upcast = tune.size.upcast ?? 1;

    const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
    for (let i = 0; i < upcast; i++) {
      emit(
        `var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`,
      ); // Initialize accumulators.
    }

    const reduceBound = symbolicReduce
      ? "i32(dims.reduce_size)"
      : `${tune.size.reduce}`;
    emit(
      `for (var ridx: i32 = 0; ridx < ${reduceBound}; ridx++) {`,
      pushIndent,
    );
    // Guard loop body so out-of-bounds threads keep identity accumulators
    // but still reach workgroupBarrier() after the loop.
    if (useSharedMem) emit("if (_valid) {", pushIndent);

    // Now generate (shared) expressions for each accumulator and unroll value.
    const exps: AluExp[][] = [];
    const cache = new Map<bigint, AluExp>();
    for (let up = 0; up < upcast; up++) {
      exps.push([]);
      for (let un = 0; un < unroll; un++) {
        const exp = tune.exp.substitute({
          upcast: AluExp.i32(up),
          unroll: AluExp.i32(un),
        });
        exps[up].push(exp.simplify(cache));
        countReferences(exps[up][un]);
      }
    }

    // After references are counted, we can generate the code.
    const items = exps.map((ar) => ar.map(gen).map(strip1));
    for (let i = 0; i < upcast; i++) {
      let rhs = items[i][0];
      for (let j = 1; j < unroll; j++) {
        if (re.op === AluOp.Add) rhs = `${rhs} + ${items[i][j]}`;
        else if (re.op === AluOp.Mul) rhs = `${rhs} * ${items[i][j]}`;
        else if (re.op === AluOp.Min) {
          // For booleans, min is AND; for numerics, use min()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} && ${items[i][j]})`
              : `min(${rhs}, ${items[i][j]})`;
        } else if (re.op === AluOp.Max) {
          // For booleans, max is OR; for numerics, use max()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} || ${items[i][j]})`
              : `max(${rhs}, ${items[i][j]})`;
        } else throw new Error(`Unsupported reduction op: ${re.op}`);
      }
      if (re.op === AluOp.Add) emit(`${acc[i]} += ${rhs};`);
      else if (re.op === AluOp.Mul) emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === AluOp.Min) {
        // For booleans, min is AND; for numerics, use min()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
        else emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      } else if (re.op === AluOp.Max) {
        // For booleans, max is OR; for numerics, use max()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
        else emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      } else throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    if (useSharedMem) emit(popIndent, "}"); // close _valid guard
    emit(popIndent, "}");

    // Shared-memory tree reduction: each thread wrote its partial into acc[i],
    // now combine across workgroup threads via shmem.
    if (useSharedMem) {
      if (useSubgroups) {
        // Subgroup-accelerated reduction:
        // Phase 1: reduce within each subgroup (no barriers needed).
        const sgOp =
          re.op === AluOp.Add
            ? "subgroupAdd"
            : re.op === AluOp.Mul
              ? "subgroupMul"
              : re.op === AluOp.Min
                ? "subgroupMin"
                : "subgroupMax";
        for (let i = 0; i < upcast; i++) {
          emit(`${acc[i]} = ${sgOp}(${acc[i]});`);
        }
        // Phase 2: leaders store packed subgroup results to shmem.
        emit("let _sg_leader: bool = (group % i32(sg_size)) == 0;");
        emit("let _sg_idx: i32 = group / i32(sg_size);");
        emit("if (_sg_leader) {");
        emit(pushIndent);
        for (let i = 0; i < upcast; i++) {
          emit(`shmem[${i * groupSize} + _sg_idx] = ${acc[i]};`);
        }
        emit(popIndent, "}");
        emit("workgroupBarrier();");
        // Phase 3: inter-subgroup tree reduction.
        // num_subgroups = groupSize / sg_size (runtime). Unroll all
        // possible strides; guard each with stride * sg_size < groupSize.
        for (let stride = groupSize >> 1; stride >= 1; stride >>= 1) {
          emit(`if (${stride}u * sg_size < ${groupSize}u) {`);
          emit(pushIndent);
          emit(`if (_sg_leader && _sg_idx < ${stride}) {`);
          emit(pushIndent);
          for (let i = 0; i < upcast; i++) {
            const thisSlot = `shmem[${i * groupSize} + _sg_idx]`;
            const otherSlot = `shmem[${i * groupSize} + _sg_idx + ${stride}]`;
            if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
            else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
            else if (re.op === AluOp.Min)
              emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
            else if (re.op === AluOp.Max)
              emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
          }
          emit(popIndent, "}");
          emit("workgroupBarrier();");
          emit(popIndent, "}");
        }
      } else {
        // Store each thread's partial into shared memory.
        for (let i = 0; i < upcast; i++) {
          emit(`shmem[${i * groupSize} + group] = ${acc[i]};`);
        }
        emit("workgroupBarrier();");

        // Tree reduction with halving stride.
        for (let stride = groupSize >> 1; stride >= 1; stride >>= 1) {
          emit(`if (group < ${stride}) {`, pushIndent);
          for (let i = 0; i < upcast; i++) {
            const thisSlot = `shmem[${i * groupSize} + group]`;
            const otherSlot = `shmem[${i * groupSize} + group + ${stride}]`;
            if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
            else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
            else if (re.op === AluOp.Min) {
              if (re.dtype === DType.Bool)
                emit(`${thisSlot} = ${thisSlot} && ${otherSlot};`);
              else emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
            } else if (re.op === AluOp.Max) {
              if (re.dtype === DType.Bool)
                emit(`${thisSlot} = ${thisSlot} || ${otherSlot};`);
              else emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
            }
          }
          emit(popIndent, "}");
          emit("workgroupBarrier();");
        }
      }

      // Thread 0 reads the final reduced value back into accumulators.
      emit("if (group == 0) {", pushIndent);
      for (let i = 0; i < upcast; i++) {
        emit(`${acc[i]} = shmem[${i * groupSize}];`);
      }
    }

    // Exited the reduction loop scope. Erase any local variables.
    expContext.clear();
    references.clear();
    seen.clear();

    const outputIdxExps: AluExp[] = [];
    const fusionExps: AluExp[] = [];
    for (let i = 0; i < upcast; i++) {
      const exp = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
      outputIdxExps.push(exp.simplify(cache));
      countReferences(outputIdxExps[i]);
      fusionExps.push(
        tune
          .epilogue!.substitute({
            acc: AluExp.variable(re.dtype, acc[i]),
            upcast: AluExp.i32(i),
          })
          .simplify(cache),
      );
      countReferences(fusionExps[i]);
    }
    for (let i = 0; i < upcast; i++) {
      const index = strip1(gen(outputIdxExps[i]));
      let rhs = strip1(gen(fusionExps[i]));
      if (resultTy !== dtypeToWgsl(fusionExps[i].dtype))
        rhs = `${resultTy}(${rhs})`;
      emit(`result[${index}] = ${rhs};`);
    }

    // Close the thread-0 guard for shared-memory reductions.
    if (useSharedMem) {
      emit(popIndent, "}");
    }
  }

  emit(popIndent, "}");

  const sharedBytes = useSharedMem
    ? groupSize *
      ((tune as WebGPUTuneResult).size.upcast ?? 1) *
      byteWidth(re!.dtype)
    : undefined;
  return {
    code: withNanInfHeader(shader.join("\n")),
    numInputs: nargs,
    numOutputs: 1,
    hasUniform: symbolic || symbolicReduce,
    passes: [{ grid: [gridX, gridY] }],
    isSymbolic: symbolic || undefined,
    workgroupSize: symbolic ? workgroupSize : undefined,
    hasSymbolicReduction: symbolicReduce || undefined,
    sharedMemoryBytes: sharedBytes,
  };
}

// ---------------------------------------------------------------------------
// Scan shader codegen (P3: WebGPU multi-kernel scan)
// ---------------------------------------------------------------------------

type EmitFn = (...lines: (string | symbol)[]) => void;
const PUSH_INDENT = Symbol("pushIndent");
const POP_INDENT = Symbol("popIndent");

function createShaderEmitter(): {
  emit: EmitFn;
  pushIndent: symbol;
  popIndent: symbol;
  getCode: () => string;
} {
  const shader: string[] = [];
  let indent = "";
  const emit: EmitFn = (...lines) => {
    for (const line of lines) {
      if (line === PUSH_INDENT) indent += "  ";
      else if (line === POP_INDENT) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };
  return {
    emit,
    pushIndent: PUSH_INDENT,
    popIndent: POP_INDENT,
    getCode: () => shader.join("\n"),
  };
}

/**
 * Convert a WGSL expression to storage representation when types differ.
 * WGSL Bool maps to `bool` in expressions but `i32` in storage buffers.
 */
function wgslToStorage(expr: string, dtype: DType): string {
  const storageTy = dtypeToWgsl(dtype, true);
  const exprTy = dtypeToWgsl(dtype);
  return storageTy !== exprTy ? `${storageTy}(${expr})` : expr;
}

/**
 * Generate a WGSL shader for native scan with multiple kernel steps.
 *
 * Uses carry snapshot locals to avoid read-after-write hazards: at each
 * iteration start carries are loaded into `c_i` locals, all expressions
 * read from those locals, and results are written back to carry buffers.
 * Internal intermediates (step outputs consumed by later steps) are also
 * kept as locals, eliminating the need for extra storage bindings.
 *
 * InitCarry values are copied to carry buffers externally (copyBufferToBuffer)
 * before the shader dispatch, so no initCarry bindings are needed.
 *
 * Buffer layout:
 *   - binding 0..numConsts-1: constants (read)
 *   - binding numConsts..numConsts+numX-1: xs (read)
 *   - binding +numX..+numCarry: carry (read_write, pre-initialized)
 *   - binding +numCarry..+numY: ysStacked (read_write)
 */
function nativeScanMultiShaderSource(
  device: GPUDevice,
  params: NativeScanMultiParams,
): ShaderInfo {
  const {
    length,
    numConsts,
    carrySizes,
    xsStrides,
    ysStrides,
    steps,
    numCarry,
    numX,
    numY,
    numInternal,
    reverse,
    internalElemCounts,
    internalDtypes,
  } = params;

  // Determine dtype from first kernel step
  const dtype = steps[0]?.kernel.outputs[0].dtype ?? DType.Float32;
  const resultTy = dtypeToWgsl(dtype, true);
  const elemSize = byteWidth(dtype);

  // Per-buffer WGSL type names
  const constTys = (params.constDtypes ?? []).map((d) => dtypeToWgsl(d, true));
  const xsTys = (params.xsDtypes ?? []).map((d) => dtypeToWgsl(d, true));
  const carryTys = (params.carryDtypes ?? []).map((d) => dtypeToWgsl(d, true));
  const ysTys = (params.ysDtypes ?? []).map((d) => dtypeToWgsl(d, true));
  const internalTys = (internalDtypes ?? []).map((d) => dtypeToWgsl(d, true));

  // Compute element-level strides (byte strides / per-xs element size)
  const xsElemStrides = xsStrides.map((s, i) => {
    const xElemSize =
      params.xsDtypes && params.xsDtypes[i]
        ? byteWidth(params.xsDtypes[i])
        : elemSize;
    return s / xElemSize;
  });

  // Carry element counts for snapshot / writeback
  const carryElemCounts = carrySizes.map((s, i) => {
    const cElemSize =
      params.carryDtypes && params.carryDtypes[i]
        ? byteWidth(params.carryDtypes[i])
        : elemSize;
    return s / cElemSize;
  });

  // Ys element strides for output indexing
  const ysElemStrides = ysStrides.map((s, i) => {
    const yElemSize =
      params.ysDtypes && params.ysDtypes[i]
        ? byteWidth(params.ysDtypes[i])
        : elemSize;
    return s / yElemSize;
  });

  // Find the maximum kernel size across all steps — determines workgroup size
  const maxKernelSize = Math.max(
    ...steps.map((s) => s.kernel.size as number),
    1,
  );

  const { emit, pushIndent, popIndent, getCode } = createShaderEmitter();

  // Check for f16 requirement
  if (dtype === DType.Float16) {
    if (!device.features.has("shader-f16")) {
      throw new Error("WebGPU device does not support shader-f16 feature");
    }
    emit("enable f16;");
  }

  // Global function definitions needed by all kernels
  const allDistinctOps = new Set<AluOp>();
  for (const step of steps) {
    const tune = tuneNullopt(step.kernel);
    for (const [op] of tune.exp.distinctOps()) allDistinctOps.add(op);
    if (tune.epilogue) {
      for (const [op] of tune.epilogue.distinctOps()) allDistinctOps.add(op);
    }
  }
  if (allDistinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allDistinctOps.has(AluOp.Erf) || allDistinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  emit("");

  // Buffer declarations — no initCarry (copied externally)
  let bindingIdx = 0;

  for (let i = 0; i < numConsts; i++) {
    const ty = constTys[i] ?? resultTy;
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> const${i}: array<${ty}>;`,
    );
  }
  for (let i = 0; i < numX; i++) {
    const ty = xsTys[i] ?? resultTy;
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> xs${i}: array<${ty}>;`,
    );
  }
  for (let i = 0; i < numCarry; i++) {
    const ty = carryTys[i] ?? resultTy;
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> carry${i}: array<${ty}>;`,
    );
  }
  for (let i = 0; i < numY; i++) {
    const ty = ysTys[i] ?? resultTy;
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> ys${i}: array<${ty}>;`,
    );
  }

  // Compute shader entry point — single thread per scan (sequential loop)
  const workgroupSize = 1;
  const [gridX, gridY] = calculateGrid(1);

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
    pushIndent,
  );

  // Declare var<private> arrays for carry snapshots
  for (let i = 0; i < numCarry; i++) {
    const cTy = carryTys[i] ?? resultTy;
    const count = carryElemCounts[i];
    emit(`var c_${i}: array<${cTy}, ${count}>;`);
  }

  // Declare var<private> arrays for internal intermediates
  for (let i = 0; i < (internalElemCounts?.length ?? 0); i++) {
    const count = internalElemCounts[i];
    const ty = internalTys[i] ?? resultTy;
    emit(`var internal_${i}: array<${ty}, ${count}>;`);
  }

  emit("");

  // Main scan loop
  emit(`for (var iter: u32 = 0u; iter < ${length}u; iter++) {`, pushIndent);

  if (reverse) {
    emit(`let dataIdx = ${length - 1}u - iter;`);
  } else {
    emit(`let dataIdx = iter;`);
  }

  // Snapshot carry into local arrays (avoids RAW hazard across steps)
  emit("");
  emit("// Snapshot carry values for this iteration");
  for (let i = 0; i < numCarry; i++) {
    const count = carryElemCounts[i];
    if (count > 1) {
      emit(
        `for (var ci: i32 = 0; ci < ${count}; ci++) { c_${i}[ci] = carry${i}[ci]; }`,
      );
    } else {
      emit(`c_${i}[0] = carry${i}[0];`);
    }
  }

  // Build scan-specific resolveGlobalIndex callback
  // gid space: [consts | carry(snapshot) | xs | internals | carry-live]
  const numInputs = numConsts + numCarry + numX;
  const carryLiveGidBase = numInputs + numInternal;
  const scanResolve: ResolveGlobalIndex = (gid, idxCode, _dtype) => {
    if (gid < numConsts) {
      return `const${gid}[${idxCode}]`;
    } else if (gid < numConsts + numCarry) {
      const ci = gid - numConsts;
      // Use carry snapshot array (value at start of iteration)
      return `c_${ci}[${idxCode}]`;
    } else if (gid < numConsts + numCarry + numX) {
      const xi = gid - numConsts - numCarry;
      const stride = xsElemStrides[xi];
      return `xs${xi}[i32(dataIdx) * ${stride} + ${idxCode}]`;
    } else if (gid < carryLiveGidBase) {
      // Internal intermediate — array access with proper index
      const ii = gid - numInputs;
      return `internal_${ii}[${idxCode}]`;
    } else {
      // Carry-live: read UPDATED carry buffer (not snapshot).
      // Used by Y-only steps that depend on carry values computed
      // earlier in the same iteration.
      const ci = gid - carryLiveGidBase;
      return `carry${ci}[${idxCode}]`;
    }
  };

  // Helper: emit store(s) for a step's computed value.
  // Steps write to carry, internal, and/or direct Y outputs in a single pass.
  // Passthrough Y outputs use a separate writeback phase driven by yOutputSources.
  const emitStepStore = (step: NativeScanMultiStep, valExpr: string) => {
    if (step.outputCarryIdx >= 0) {
      emit(`carry${step.outputCarryIdx}[eidx] = ${valExpr};`);
    }
    if (step.outputInternalIdx >= 0) {
      emit(`internal_${step.outputInternalIdx}[eidx] = ${valExpr};`);
    }
    if (step.outputYIdxs) {
      for (const yi of step.outputYIdxs) {
        const ysStride = ysElemStrides[yi] ?? 0;
        if (ysStride > 0) {
          emit(`ys${yi}[i32(dataIdx) * ${ysStride} + eidx] = ${valExpr};`);
        }
      }
    }
  };

  // Execute each step using createWgslGen + eidx loops
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    const kernel = step.kernel;
    const kernelSize = kernel.size as number;

    const targets: string[] = [];
    if (step.outputCarryIdx >= 0) targets.push(`carry${step.outputCarryIdx}`);
    if (step.outputInternalIdx >= 0)
      targets.push(`internal_${step.outputInternalIdx}`);

    emit("");
    emit(`// Step ${stepIdx}: writes to ${targets.join(", ")}`);

    const gidxOverride = AluExp.special(DType.Int32, "eidx", kernelSize);
    const gen = createWgslGen({
      kernel,
      prefix: `sc_s${stepIdx}`,
      resolveGlobalIndex: scanResolve,
      emit,
      blockSize: maxKernelSize,
      gidxOverride,
    });

    const re = kernel.outputs[0].reduction;
    const outDtype = kernel.outputs[0].dtype;
    if (re) {
      // Reduction kernel: eidx loop × ridx loop
      // Use non-storage type for accumulator (bool stays bool during computation)
      const accTy = dtypeToWgsl(re.dtype);
      const redSize =
        typeof re.size === "number"
          ? re.size
          : (re.concreteHint ?? Number(re.size));

      if (kernelSize > 1) {
        emit(
          `for (var eidx: i32 = 0; eidx < ${kernelSize}; eidx++) {`,
          pushIndent,
        );
      } else {
        emit(`{`, pushIndent, `let eidx: i32 = 0;`);
      }
      emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
      emit(`for (var ridx: i32 = 0; ridx < ${redSize}; ridx++) {`, pushIndent);
      emit(`let val = ${strip1(gen(kernel.outputs[0].exp))};`);

      if (re.op === AluOp.Add) emit(`acc = acc + val;`);
      else if (re.op === AluOp.Mul) emit(`acc = acc * val;`);
      else if (re.op === AluOp.Min) emit(`acc = min(acc, val);`);
      else if (re.op === AluOp.Max) emit(`acc = max(acc, val);`);
      else throw new Error(`Unsupported reduction op: ${re.op}`);

      emit(popIndent, "}");

      // Epilogue + store (convert bool→i32 for storage buffers)
      const epilogueVal = wgslToStorage(
        strip1(gen(kernel.outputs[0].reduction!.epilogue)),
        outDtype,
      );
      emitStepStore(step, epilogueVal);
      emit(popIndent, "}");
    } else {
      // Elementwise kernel
      if (kernelSize > 1) {
        emit(
          `for (var eidx: i32 = 0; eidx < ${kernelSize}; eidx++) {`,
          pushIndent,
        );
      } else {
        emit(`{`, pushIndent, `let eidx: i32 = 0;`);
      }

      // Convert bool→i32 for storage buffers
      const val = wgslToStorage(strip1(gen(kernel.outputs[0].exp)), outDtype);
      emitStepStore(step, val);
      emit(popIndent, "}");
    }
  }

  // Y output writeback phase — copy from source to stacked Y buffers.
  // In compiled-loop, only passthrough sources (carry-snapshot, xs) appear here;
  // carry-live and internal are written inline by steps via outputYIdxs.
  const yOutputSources = params.yOutputSources;
  if (yOutputSources.length > 0) {
    emit("");
    emit("// Y output writeback (passthroughs)");
    for (let i = 0; i < yOutputSources.length; i++) {
      const src = yOutputSources[i];
      const yi = src.yi;
      const ysStride = ysElemStrides[yi] ?? 0;
      if (ysStride === 0) continue;

      let srcExpr: string;
      const elemCount = params.yElemCounts[yi];
      switch (src.type) {
        case "carry-snapshot": {
          srcExpr = `c_${src.carryIdx}[eidx]`;
          break;
        }
        case "xs": {
          const xsStride = xsElemStrides[src.xsIdx];
          srcExpr = `xs${src.xsIdx}[i32(dataIdx) * ${xsStride} + eidx]`;
          break;
        }
        // carry-live and internal are handled by outputYIdxs in compiled-loop,
        // but the type union is shared with preencoded paths that use all variants.
        /* c8 ignore next 2 */
        case "carry-live":
        case "internal":
          throw new Error(
            `Unexpected Y source type "${src.type}" in compiled-loop writeback`,
          );
      }

      if (elemCount > 1) {
        emit(
          `for (var eidx: i32 = 0; eidx < ${elemCount}; eidx++) { ys${yi}[i32(dataIdx) * ${ysStride} + eidx] = ${srcExpr}; }`,
        );
      } else {
        emit(
          `{ let eidx: i32 = 0; ys${yi}[i32(dataIdx) * ${ysStride} + eidx] = ${srcExpr}; }`,
        );
      }
    }
  }

  emit(popIndent, "}");
  emit(popIndent, "}");

  const numReadOnlyInputs = numConsts + numX;
  const numReadWriteOutputs = numCarry + numY;

  return {
    code: withNanInfHeader(getCode()),
    numInputs: numReadOnlyInputs,
    numOutputs: numReadWriteOutputs,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

function pipelineSubmit(
  device: GPUDevice,
  pipelines: ShaderDispatch[],
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
  dynamicParams?: number[],
  batchEncoder?: GPUCommandEncoder,
  batchUniformCollector?: GPUBuffer[],
  profiling?: _ProfilingState | null,
  source?: Kernel | Routine,
) {
  const commandEncoder = batchEncoder ?? device.createCommandEncoder();
  const uniformBuffersToDestroy: GPUBuffer[] = [];
  for (const { pipeline, ...shader } of pipelines) {
    if (
      inputs.length !== shader.numInputs ||
      outputs.length !== shader.numOutputs
    ) {
      throw new Error(
        `webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, ` +
          `got ${inputs.length} inputs and ${outputs.length} outputs`,
      );
    }

    const filteredPasses = shader.passes.filter(({ grid }) => prod(grid) > 0);
    if (filteredPasses.length === 0) continue; // No work to do.

    const slot = maybeAcquireTracingSlot(device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...inputs.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
        ...outputs.map((buffer, i) => ({
          binding: inputs.length + i,
          resource: { buffer },
        })),
      ],
    });

    let uniformBindGroup: GPUBindGroup | null = null;
    let uniformAlignment = 0;

    // Symbolic shaders: compute grid dynamically from dynamicParams and
    // create a uniform buffer with resolved dimension values.
    // struct Dims layout matches the shader:
    //   isSymbolic + hasSymbolicReduction → { total_size: u32, reduce_size: u32 }
    //   isSymbolic only                   → { total_size: u32 }
    //   hasSymbolicReduction only         → { reduce_size: u32 }
    let symbolicGrid: [number, number] | null = null;
    const needsSymbolicUniform =
      (shader.isSymbolic || shader.hasSymbolicReduction) &&
      dynamicParams &&
      dynamicParams.length > 0;
    if (needsSymbolicUniform) {
      if (shader.isSymbolic) {
        const totalSize = dynamicParams[0];
        const wgSize =
          typeof shader.workgroupSize === "number"
            ? shader.workgroupSize
            : (shader.workgroupSize?.[0] ?? 256);
        const gridSize = Math.ceil(totalSize / wgSize);
        symbolicGrid = calculateGrid(gridSize);
      }

      // Build uniform data matching the shader's struct Dims field order
      const uniformFields: number[] = [];
      if (shader.isSymbolic) uniformFields.push(dynamicParams[0]); // total_size
      if (shader.hasSymbolicReduction) uniformFields.push(dynamicParams[1]); // reduce_size
      const dataSize = uniformFields.length * 4;
      const uniformData = new Uint8Array(dataSize);
      const dv = new DataView(uniformData.buffer);
      for (let fi = 0; fi < uniformFields.length; fi++) {
        dv.setUint32(fi * 4, uniformFields[fi], true);
      }

      const minAlign = device.limits.minUniformBufferOffsetAlignment;
      const alignment = Math.ceil(dataSize / minAlign) * minAlign;
      const uniformBuffer = device.createBuffer({
        size: alignment,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      });
      new Uint8Array(uniformBuffer.getMappedRange()).set(uniformData);
      uniformBuffer.unmap();
      uniformBuffersToDestroy.push(uniformBuffer);
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, size: alignment } },
        ],
      });
    } else if (shader.hasUniform) {
      // Non-symbolic shader with uniforms: create a shared buffer with uniform
      // values for each pass of the shader (use dynamic offsets).
      const uniforms = filteredPasses.map(({ uniform }) => uniform!);
      const [uniformBuffer, alignment] = combineUniforms(device, uniforms);
      uniformBuffersToDestroy.push(uniformBuffer);
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, size: alignment } },
        ],
      });
    }

    for (let i = 0; i < filteredPasses.length; i++) {
      const grid = symbolicGrid ?? filteredPasses[i].grid;
      const tracingTsw =
        slot && !profiling
          ? {
              querySet: slot.batch.querySet,
              beginningOfPassWriteIndex: i === 0 ? slot.beginIndex : undefined,
              endOfPassWriteIndex:
                i === filteredPasses.length - 1 ? slot.endIndex : undefined,
            }
          : undefined;
      const passEncoder = _beginComputePass(
        commandEncoder,
        profiling,
        undefined,
        tracingTsw,
      );
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      if (uniformBindGroup)
        passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      _dispatchCount++;
      passEncoder.end();
    }

    if (slot && source) {
      recordTrace(device, slot, source, filteredPasses.length, shader.code);
    }
  }
  if (batchEncoder) {
    // In batch mode: collect uniforms for caller to destroy after submit
    batchUniformCollector!.push(...uniformBuffersToDestroy);
  } else {
    device.queue.submit([commandEncoder.finish()]);
    for (const buf of uniformBuffersToDestroy) buf.destroy();
  }
}

function combineUniforms(
  device: GPUDevice,
  uniforms: Uint8Array<ArrayBuffer>[],
): [GPUBuffer, number] {
  for (const buf of uniforms) {
    if (
      !buf ||
      buf.byteLength === 0 ||
      buf.byteLength !== uniforms[0].byteLength
    ) {
      throw new Error("webgpu: Uniform mismatch between shader passes");
    }
  }
  const minAlign = device.limits.minUniformBufferOffsetAlignment;
  const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
  const buffer = device.createBuffer({
    size: alignment * uniforms.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const bufferMapped = new Uint8Array(buffer.getMappedRange());
  for (let i = 0; i < uniforms.length; i++)
    bufferMapped.set(uniforms[i], i * alignment);
  buffer.unmap();
  return [buffer, alignment];
}

/**
 * A cache for compiled GPU compute pipelines, keyed by the shader source.
 *
 * This supports both async compilation (recommended) and a synchronous variant.
 * If the pipeline is not in the cache, it will be compiled and added. For async
 * compilation, only one compilation will be in progress at a time for a given
 * shader source.
 */
class ShaderPipelineCache {
  cache: Map<string, GPUComputePipeline>;
  inProgress: Map<string, Promise<GPUComputePipeline>>;
  #layoutCache: Map<string, GPUPipelineLayout>;
  #bindGroup0Cache: Map<string, GPUBindGroupLayout> = new Map();
  #bindGroup1Cache: Map<string, GPUBindGroupLayout> = new Map();

  constructor(readonly device: GPUDevice) {
    this.cache = new Map();
    this.inProgress = new Map();
    this.#layoutCache = new Map();
  }

  getLayout(shader: ShaderInfo): GPUBindGroupLayout {
    const key = `${shader.numInputs}:${shader.numOutputs}`;
    let cached = this.#bindGroup0Cache.get(key);
    if (!cached) {
      cached = this.device.createBindGroupLayout({
        entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: i < shader.numInputs ? "read-only-storage" : "storage",
          },
        })),
      });
      this.#bindGroup0Cache.set(key, cached);
    }
    return cached;
  }

  getUniformLayout(shader: ShaderInfo): GPUBindGroupLayout {
    const nuc = shader.numUniformConsts ?? 0;
    const key = `${shader.hasUniform ? 1 : 0}:${nuc}`;
    let cached = this.#bindGroup1Cache.get(key);
    if (!cached) {
      if (nuc > 0) {
        cached = this.device.createBindGroupLayout({
          entries: range(nuc).map((i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          })),
        });
      } else if (shader.hasUniform) {
        cached = this.device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform", hasDynamicOffset: true },
            },
          ],
        });
      } else {
        throw new Error("Shader has no uniform layout");
      }
      this.#bindGroup1Cache.set(key, cached);
    }
    return cached;
  }

  #getLayout(shader: ShaderInfo): GPUPipelineLayout {
    if (
      shader.numInputs + shader.numOutputs >
      this.device.limits.maxStorageBuffersPerShaderStage
    ) {
      const actual = shader.numInputs + shader.numOutputs;
      const max = this.device.limits.maxStorageBuffersPerShaderStage;
      throw new Error(
        `Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`,
      );
    }
    // Cache by signature: most JIT programs share the same layout shape.
    const nuc = shader.numUniformConsts ?? 0;
    const key = `${shader.numInputs}:${shader.numOutputs}:${shader.hasUniform ? 1 : 0}:${nuc}`;
    const cached = this.#layoutCache.get(key);
    if (cached) return cached;

    const bindGroupLayouts =
      shader.hasUniform || (shader.numUniformConsts ?? 0) > 0
        ? [this.getLayout(shader), this.getUniformLayout(shader)]
        : [this.getLayout(shader)];

    const layout = this.device.createPipelineLayout({ bindGroupLayouts });
    this.#layoutCache.set(key, layout);
    return layout;
  }

  async prepare(shader: ShaderInfo): Promise<GPUComputePipeline> {
    // Deno's WebGPU (wgpu-rs) doesn't support createComputePipelineAsync
    // reliably. Fall back to synchronous pipeline creation.
    // @ts-expect-error Deno global
    if (typeof Deno !== "undefined") return this.prepareSync(shader);

    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    const existingPromise = this.inProgress.get(shader.code);
    if (existingPromise) return await existingPromise;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "webgpu",
        kind: "kernel",
        label: shader.label,
        code: shader.code,
        workgroupSize: Array.isArray(shader.workgroupSize)
          ? (shader.workgroupSize as [number, number, number])
          : shader.workgroupSize
            ? [shader.workgroupSize, 1, 1]
            : undefined,
        metadata: {
          numInputs: shader.numInputs,
          numOutputs: shader.numOutputs,
        },
      });
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    const promise = (async () => {
      this.device.pushErrorScope("validation");
      try {
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.#getLayout(shader),
          compute: {
            module: shaderModule,
            entryPoint: "main",
          },
        });
        await this.device.popErrorScope();
        return pipeline;
      } catch (_error: unknown) {
        // This can race with other compilations, but it shouldn't happen in
        // correct code. Any validation error here is a bug in `jax-js`.
        const scope = await this.device.popErrorScope();
        const emsg = await compileError(shaderModule, scope, shader.code);
        throw new Error(emsg);
      }
    })();
    this.inProgress.set(shader.code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(shader.code, pipeline);
    return pipeline;
  }

  prepareSync(shader: ShaderInfo): GPUComputePipeline {
    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "webgpu",
        kind: "kernel",
        label: shader.label,
        code: shader.code,
        workgroupSize: Array.isArray(shader.workgroupSize)
          ? (shader.workgroupSize as [number, number, number])
          : shader.workgroupSize
            ? [shader.workgroupSize, 1, 1]
            : undefined,
        metadata: {
          numInputs: shader.numInputs,
          numOutputs: shader.numOutputs,
        },
      });
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    // Deno's wgpu-rs doesn't support pushErrorScope/popErrorScope reliably.
    // @ts-expect-error Deno global
    const hasScopeApi = typeof Deno === "undefined";
    if (hasScopeApi) this.device.pushErrorScope("validation");
    const pipeline = this.device.createComputePipeline({
      layout: this.#getLayout(shader),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    if (hasScopeApi) {
      this.device.popErrorScope().then(async (scope) => {
        // This happens asynchronously, so we can't throw here. But shader syntax
        // validation errors should never occur in correct code. Any issues here
        // reflect bugs in jax-js.
        if (scope !== null) {
          const emsg = await compileError(shaderModule, scope, shader.code);
          console.error(emsg);
        }
      });
    }
    this.cache.set(shader.code, pipeline);
    return pipeline;
  }
}

/** Gather information about a compilation error and format it. */
async function compileError(
  shaderModule: GPUShaderModule,
  scope: GPUError | null,
  code: string,
): Promise<string> {
  let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
  const info = await shaderModule.getCompilationInfo();
  for (const msg of info.messages) {
    message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
  }
  if (code) {
    message += `\n\n${code}`;
  }
  return message;
}
