/**
 * @file BlockMap JIT executor — runs the block_map body program over a grid
 * of blocks, slicing inputs and assembling outputs.
 *
 * The fallback path iterates in JS, calling bodyProgram.execute() per block.
 * The fused path (WebGPU only) compiles the body into a single WGSL shader
 * where each workgroup processes one block using shared memory.
 */

import { byteWidth, type DType, Kernel, Reduction } from "../alu";
import { type Backend, Executable, type Slot } from "../backend";
import type { BlockMapWasmParams, GeneralScanStep } from "../backend/wasm";
import type { PackedLeafLayout } from "../backend/webgpu/codegen";
import { DEBUG, prod } from "../utils";
import type { IPendingExecute } from "./array";
import { _registerJitCacheDisposer } from "./check-leaks";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram, JitStep } from "./jit";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ExecuteBlockMapParams {
  backend: Backend;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  blockShape: number[];
  inAxes: (number | null)[][];
  outAxes: (number | null)[][];
  numConsts: number;
  numInputs: number;
  gridShape: number[];
  /** Original (untrimmed) input shapes. */
  inputShapes: number[][];
  /** Original (untrimmed) output shapes. */
  outputShapes: number[][];
  constSlots: Slot[];
  inputSlots: Slot[];
  /** Preallocated output slots — sized to match outputShapes. */
  outputSlots: Slot[];
  /** Register tiling dimensions. */
  threadTile?: number[];
  /** Body uses BlockIndex: compiled body has one extra input for the block index. */
  hasBlockIndex?: boolean;
  /** Per-constant info for uniform buffer migration (one entry per numConsts input). */
  constInfos?: { elemCount: number; dtype: DType; bytes: number }[];
  /** Point-mode inputs: one element per grid point (see BlockMapShaderParams). */
  pointInputs?: boolean[];
  /** Grid offset for mapped input/output base offsets (see BlockMapShaderParams). */
  gridOffset?: number;
  /**
   * Enable leaf packing — all non-const inputs packed into a single storage buffer
   * (and similarly for outputs) to stay within maxStorageBuffersPerShaderStage.
   * The shader codegen computes concrete element offsets from shapes.
   */
  needsLeafPacking?: boolean;
  /**
   * Per-input per-grid-axis overlap [lo, hi].
   * When present, the body receives expanded tiles (blockShape[g] + lo + hi)
   * and global reads that fall outside the input bounds return zero.
   * inputShapes must be the ORIGINAL (unpadded) shapes.
   */
  halo?: [number, number][][];
}

export interface ExecuteBlockMapResult {
  outputs: Slot[];
  pending: IPendingExecute[];
}

/**
 * Execute a block_map. Tries the fused WebGPU path first (single dispatch,
 * shared-memory intermediates), then falls back to the JS loop executor.
 */
export function executeBlockMap(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult {
  // Try fused WebGPU shader path
  if (params.backend.type === "webgpu") {
    const result = tryExecuteBlockMapFused(params);
    if (result) return result;
  }
  // Try compiled WASM loop path
  if (params.backend.type === "wasm") {
    const result = tryExecuteBlockMapWasm(params);
    if (result) return result;
  }
  return executeBlockMapFallback(params);
}

// ---------------------------------------------------------------------------
// Fused cache compile counter — test observability for cache partitioning
// ---------------------------------------------------------------------------

/** Number of times prepareBlockMapFused was actually called (cache miss). */
let _fusedCompileCount = 0;

/**
 * Return the current fused-cache compile count and optionally reset it.
 * Test-only — verifies cache hit/miss behaviour.
 */
export function _fusedCacheCompileCount(reset = false): number {
  const count = _fusedCompileCount;
  if (reset) _fusedCompileCount = 0;
  return count;
}

// ---------------------------------------------------------------------------
// Fused WebGPU path: single-dispatch shared-memory shader
// ---------------------------------------------------------------------------

// Cache fused Executable by bodyProgram identity + codegen-specialization key.
// The JitProgram is identity-stable (cached by jitCompile) and may be
// polymorphic (via dynamic_axes / SymDim). A single polymorphic JitProgram
// can require multiple fused executables when concrete grid geometry differs,
// because the WGSL bakes grid decomposition, boundary guards, strides, and
// buffer indexing as compile-time constants.
// Uses a regular Map (not WeakMap) so that clearCaches() deterministically
// invalidates entries — avoiding non-deterministic GC timing issues with
// checkLeaks slot counting.
const blockMapFusedCache = new Map<
  JitProgram,
  Map<string, Executable | null>
>();
_registerJitCacheDisposer(() => blockMapFusedCache.clear());

/**
 * Codegen-specialization key for fused block-map executables.
 * Includes only fields that affect the generated WGSL — excludes runtime
 * data (slots, backend). If future work makes a field runtime-driven
 * (e.g., grid extent as a uniform), remove it from the key at that time.
 */
function blockMapSpecKey(params: ExecuteBlockMapParams): string {
  const parts = [
    params.blockShape.join(","),
    params.gridShape.join(","),
    params.inAxes.map((a) => a.map((v) => v ?? "_").join(",")).join(";"),
    params.outAxes.map((a) => a.map((v) => v ?? "_").join(",")).join(";"),
    params.numConsts,
    params.numInputs,
    params.inputShapes.map((s) => s.join(",")).join(";"),
    params.outputShapes.map((s) => s.join(",")).join(";"),
    params.threadTile?.join(",") ?? "-",
    params.hasBlockIndex ? "bi" : "-",
    params.pointInputs?.map((p) => (p ? "P" : "-")).join("") ?? "-",
    params.gridOffset ?? 0,
    params.needsLeafPacking ? "LP" : "-",
    params.halo
      ? params.halo
          .map((h) => h.map(([l, r]) => `${l}_${r}`).join(","))
          .join(";")
      : "-",
  ];
  return parts.join("|");
}

function tryExecuteBlockMapFused(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult | null {
  // Lazy import to avoid circular dependency at module load time.
  // The scan-executor uses the same pattern (type import + cast).
  const webgpuBackend =
    params.backend as import("../backend/webgpu").WebGPUBackend;

  const specKey = blockMapSpecKey(params);
  let inner = blockMapFusedCache.get(params.bodyProgram);
  if (!inner) {
    inner = new Map();
    blockMapFusedCache.set(params.bodyProgram, inner);
  }
  let exe = inner.get(specKey);
  if (exe === undefined) {
    _fusedCompileCount++;
    exe =
      webgpuBackend.prepareBlockMapFused({
        bodyProgram: params.bodyProgram,
        blockShape: params.blockShape,
        gridShape: params.gridShape,
        inAxes: params.inAxes,
        outAxes: params.outAxes,
        numConsts: params.numConsts,
        numInputs: params.numInputs,
        inputShapes: params.inputShapes,
        outputShapes: params.outputShapes,
        threadTile: params.threadTile,
        hasBlockIndex: params.hasBlockIndex,
        constInfos: params.constInfos,
        pointInputs: params.pointInputs,
        gridOffset: params.gridOffset,
        needsLeafPacking: params.needsLeafPacking,
        halo: params.halo,
      }) ?? null;
    inner.set(specKey, exe);
  }

  if (!exe) return null;

  if (DEBUG >= 1) {
    console.info("block_map: using fused WebGPU shader path");
  }

  // --- Leaf packing: pack/unpack buffers around dispatch ---
  const layout: PackedLeafLayout | undefined = exe.data[0].packedLeafLayout;
  if (layout) {
    const bw = byteWidth(layout.dtype);
    const backend = params.backend;

    // Allocate packed input buffer and copy leaf inputs into it
    const packedInBytes = layout.totalInputElems * bw;
    const packedInSlot = backend.malloc(packedInBytes);
    for (let i = 0; i < params.inputSlots.length; i++) {
      const offsetBytes = layout.inputOffsets[i] * bw;
      const sizeBytes = prod(params.inputShapes[i]) * bw;
      backend.copyBufferToBuffer(
        params.inputSlots[i],
        0,
        packedInSlot,
        offsetBytes,
        sizeBytes,
      );
    }

    // Allocate packed output buffer
    const packedOutBytes = layout.totalOutputElems * bw;
    const packedOutSlot = backend.malloc(packedOutBytes);

    // Dispatch with packed buffers
    webgpuBackend.dispatchBlockMapFused(
      exe,
      [...params.constSlots, packedInSlot],
      [packedOutSlot],
    );

    // Unpack output buffer into individual output slots
    for (let o = 0; o < params.outputSlots.length; o++) {
      const offsetBytes = layout.outputOffsets[o] * bw;
      const sizeBytes = prod(params.outputShapes[o]) * bw;
      backend.copyBufferToBuffer(
        packedOutSlot,
        offsetBytes,
        params.outputSlots[o],
        0,
        sizeBytes,
      );
    }

    // Free packed temporaries
    backend.decRef(packedInSlot);
    backend.decRef(packedOutSlot);

    return { outputs: params.outputSlots, pending: [] };
  }

  // Dispatch: inputs = [consts, blockInputs], outputs = [outputSlots]
  const inputs = [...params.constSlots, ...params.inputSlots];
  webgpuBackend.dispatchBlockMapFused(exe, inputs, params.outputSlots);

  return { outputs: params.outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Compiled WASM block-loop path
// ---------------------------------------------------------------------------

function tryExecuteBlockMapWasm(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult | null {
  const wasmBackend = params.backend as import("../backend/wasm").WasmBackend;

  const wasmParams = buildBlockMapWasmParams(params);
  if (!wasmParams) return null;

  const exe = wasmBackend.prepareBlockMapWasm(wasmParams);
  if (!exe) return null;

  if (DEBUG >= 1) {
    console.info("block_map: using compiled WASM loop path");
  }

  wasmBackend.dispatchBlockMapWasm(
    exe,
    wasmParams,
    params.constSlots,
    params.inputSlots,
    params.outputSlots,
  );

  return { outputs: params.outputSlots, pending: [] };
}

/**
 * Build BlockMapWasmParams from the body program, or return null if the body
 * can't be compiled to WASM (routines, loops, non-contiguous slices, etc.).
 */
function buildBlockMapWasmParams(
  params: ExecuteBlockMapParams,
): BlockMapWasmParams | null {
  const {
    bodyProgram,
    bodyJaxpr,
    blockShape,
    inAxes,
    outAxes,
    numConsts,
    numInputs,
    gridShape,
    inputShapes,
    outputShapes,
  } = params;
  const gridRank = blockShape.length;
  const numOutputs = bodyJaxpr.outs.length;

  // Only support kernel-only bodies (no routines, nested scans, etc.)
  type ExecuteStep = Extract<JitStep, { type: "execute" }>;
  const executeSteps: ExecuteStep[] = [];
  for (const step of bodyProgram.steps) {
    if (step.type === "execute") {
      if (!(step.source instanceof Kernel)) {
        if (DEBUG >= 2)
          console.log("[wasm-block-map] rejected: non-kernel execute step");
        return null;
      }
      executeSteps.push(step as ExecuteStep);
    } else if (
      step.type === "malloc" ||
      step.type === "free" ||
      step.type === "recycle" ||
      step.type === "incref"
    ) {
      // These are fine — they're resource management
    } else {
      // scan, assoc_scan, block_map, fori_loop, etc. — can't inline
      if (DEBUG >= 2)
        console.log(
          `[wasm-block-map] rejected: unsupported step type ${step.type}`,
        );
      return null;
    }
  }

  if (executeSteps.length === 0) {
    if (DEBUG >= 2) console.log("[wasm-block-map] rejected: no execute steps");
    return null;
  }

  // Check contiguity: for each input with axis mapping, the slice must be contiguous.
  // A slice is contiguous if all mapped axes are "leading" — i.e., for mapped axis `ax`,
  // all dimensions d > ax match the full input shape (not sub-sliced by a different grid dim).
  for (let i = 0; i < numInputs; i++) {
    const axes = inAxes[i];
    const shape = inputShapes[i];
    // Collect mapped dims
    const mappedDims = new Set<number>();
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) mappedDims.add(axes[g]!);
    }
    // For contiguity: if dim `d` is mapped, then all dims d+1..nd-1 must also
    // be mapped (so the block is a contiguous chunk of memory).
    // Actually, for a single mapped dim, the slice is contiguous as long as
    // it slices a contiguous inner extent. This is true when the mapped dim
    // is the outermost non-trivial dim: all inner dims are fully included.
    // For simplicity: require that mapped dims form a prefix {0} or {0,1} etc.
    // OR that there's only one mapped dim and blockShape matches inner dims
    // after that dim.

    // Simple check: for each mapped dim ax, all dims (ax+1..nd-1) must NOT
    // be differently sliced. Since the body expects blockShape dimensions on
    // mapped axes and full input dims on unmapped axes, this is satisfied when
    // the inner dimensions match.
    for (const ax of mappedDims) {
      for (let d = ax + 1; d < shape.length; d++) {
        if (!mappedDims.has(d)) {
          // Inner dim `d` is unmapped — the body reads it at full size.
          // The slice includes the full extent in this dim — contiguous.
          continue;
        }
        // If inner dim is also mapped, then the slice is a 2D sub-block
        // of a 2D array. This is NOT contiguous in general for row-major.
        // Exception: if it's the last (innermost) dim, the contiguous
        // row covers blockShape[g] elements, and we'd need row-by-row copy.
        // For now, reject multi-dim sub-blocking.
        if (DEBUG >= 2)
          console.log(
            "[wasm-block-map] rejected: non-contiguous multi-dim slicing",
          );
        return null;
      }
    }
  }

  // Same check for outputs
  for (let o = 0; o < numOutputs; o++) {
    const axes = outAxes[o];
    const shape = outputShapes[o];
    const mappedDims = new Set<number>();
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) mappedDims.add(axes[g]!);
    }
    for (const ax of mappedDims) {
      for (let d = ax + 1; d < shape.length; d++) {
        if (mappedDims.has(d)) {
          if (DEBUG >= 2)
            console.log(
              "[wasm-block-map] rejected: non-contiguous multi-dim output slicing",
            );
          return null;
        }
      }
    }
  }

  // Build slot-to-internal mapping
  const numBodyInputs = numConsts + numInputs;
  const slotToInternal = new Map<number, number>();
  const internalSizes: number[] = [];

  for (const step of executeSteps) {
    const kernel = step.source as Kernel;
    const internalIdx = internalSizes.length;
    slotToInternal.set(step.outputs[0], internalIdx);
    internalSizes.push(
      (kernel.size as number) * byteWidth(kernel.outputs[0].dtype),
    );
  }

  // Build reindexed kernel steps
  const reindexedSteps: GeneralScanStep[] = [];
  for (const step of executeSteps) {
    const kernel = step.source as Kernel;

    // Map each input JitId to a body-local index
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numBodyInputs) {
        inputSlots.push(inputId);
      } else {
        const intIdx = slotToInternal.get(inputId);
        if (intIdx === undefined) {
          if (DEBUG >= 2)
            console.log(`[wasm-block-map] rejected: unmapped input ${inputId}`);
          return null;
        }
        inputSlots.push(numBodyInputs + intIdx);
      }
    }

    // Reindex kernel expressions
    const reindexMap = inputSlots;
    const reindexedExp = kernel.outputs[0].exp.reindexGids(reindexMap);
    const reindexedReduction = kernel.outputs[0].reduction
      ? new Reduction(
          kernel.outputs[0].reduction.dtype,
          kernel.outputs[0].reduction.op,
          kernel.outputs[0].reduction.size,
          kernel.outputs[0].reduction.epilogue.reindexGids(reindexMap),
        )
      : undefined;
    const reindexedKernel = Kernel.single(
      numBodyInputs + internalSizes.length,
      kernel.size,
      reindexedExp,
      reindexedReduction,
    );

    reindexedSteps.push({
      source: reindexedKernel,
      inputSlots,
      outputInternalIdx: slotToInternal.get(step.outputs[0])!,
    });
  }

  // Determine which internal buffer each output comes from
  const outputSources: number[] = [];
  for (const outSlot of bodyProgram.outputs) {
    const intIdx = slotToInternal.get(outSlot);
    if (intIdx !== undefined) {
      outputSources.push(intIdx);
    } else {
      // Output is a passthrough from input — not supported in compiled path
      if (DEBUG >= 2)
        console.log("[wasm-block-map] rejected: passthrough output");
      return null;
    }
  }

  // Compute byte sizes and strides
  const bodyInputAvals = bodyJaxpr.inBinders
    .slice(numConsts)
    .map((v) => v.aval);
  const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);

  const constSizes: number[] = [];
  for (let c = 0; c < numConsts; c++) {
    const aval = bodyJaxpr.inBinders[c].aval;
    constSizes.push((aval.size as number) * byteWidth(aval.dtype));
  }

  const blockInputSizes: number[] = [];
  for (let i = 0; i < numInputs; i++) {
    blockInputSizes.push(
      (bodyInputAvals[i].size as number) * byteWidth(bodyInputAvals[i].dtype),
    );
  }

  const blockOutputSizes: number[] = [];
  for (let o = 0; o < numOutputs; o++) {
    blockOutputSizes.push(
      (bodyOutAvals[o].size as number) * byteWidth(bodyOutAvals[o].dtype),
    );
  }

  const inputStridesArr: number[][] = [];
  for (let i = 0; i < numInputs; i++) {
    const shape = inputShapes[i];
    const elemBytes = byteWidth(bodyInputAvals[i].dtype);
    inputStridesArr.push(computeStrides(shape, elemBytes));
  }

  const outputStridesArr: number[][] = [];
  for (let o = 0; o < numOutputs; o++) {
    const shape = outputShapes[o];
    const elemBytes = byteWidth(bodyOutAvals[o].dtype);
    outputStridesArr.push(computeStrides(shape, elemBytes));
  }

  return {
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
    constSizes,
    blockInputSizes,
    blockOutputSizes,
    internalSizes,
    steps: reindexedSteps,
    outputSources,
    inputStrides: inputStridesArr,
    outputStrides: outputStridesArr,
    halo: params.halo,
  };
}

// ---------------------------------------------------------------------------
// Fallback: JS loop calling bodyProgram.execute() per block
// ---------------------------------------------------------------------------

function executeBlockMapFallback(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult {
  const {
    backend,
    bodyProgram,
    bodyJaxpr,
    blockShape,
    inAxes,
    outAxes,
    numConsts: _numConsts,
    numInputs,
    gridShape,
    inputShapes,
    outputShapes,
    constSlots,
    inputSlots,
    outputSlots,
    hasBlockIndex,
    gridOffset: _gridOffset,
  } = params;
  const gridOffset = _gridOffset ?? 0;

  const gridRank = blockShape.length;
  const numBlocks = gridShape.reduce((a, b) => a * b, 1);
  const numOutputs = bodyJaxpr.outs.length;
  const pending: IPendingExecute[] = [];

  // When hasBlockIndex, the rewritten body jaxpr has one extra inBinder
  // at the end for the block index. Exclude it from input avals.
  const bodyInputAvals = bodyJaxpr.inBinders
    .slice(_numConsts, _numConsts + numInputs)
    .map((v) => v.aval);
  const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);

  const BATCH_SIZE = 64;
  const useBatching = backend.beginBatch != null && numBlocks > 1;
  if (useBatching) backend.beginBatch!();

  // Shared zero buffer for halo/boundary blocks that need zero-padded tiles.
  // Hoisted outside the loop to avoid per-block Uint8Array GC pressure.
  let sharedZeros: Uint8Array | undefined;

  try {
    for (let flatIdx = 0; flatIdx < numBlocks; flatIdx++) {
      // Convert flat index to grid coordinates (row-major)
      const blockIdx: number[] = new Array(gridRank);
      let remaining = flatIdx;
      for (let g = gridRank - 1; g >= 0; g--) {
        blockIdx[g] = remaining % gridShape[g];
        remaining = Math.floor(remaining / gridShape[g]);
      }

      // Flush pending before each body invocation
      flushPending(pending, backend);

      // Determine if this is a boundary block (last block on any axis where
      // the input dimension is not evenly divisible by blockShape).
      const _isBoundary = blockIdx.some((bi, g) => bi === gridShape[g] - 1);

      // Create block input slots by slicing from full inputs
      const blockInputSlots: Slot[] = [];
      const ownedSlices: Slot[] = [];

      for (let i = 0; i < numInputs; i++) {
        const axes = inAxes[i];
        const blockAval = bodyInputAvals[i];
        const nd = blockAval.shape.length;
        const elemBytes = byteWidth(blockAval.dtype);
        const blockBytes = blockAval.size * elemBytes;

        // Compute the slice region in the original input, accounting for halo.
        // rawStart may be negative (halo before start of input), rawEnd may
        // exceed inputDim (halo past end). We clamp and zero-pad.
        const sliceStarts: number[] = new Array(nd).fill(0);
        const sliceSizes: number[] = [...blockAval.shape] as number[];
        const dstStarts: number[] = new Array(nd).fill(0);
        let needsPad = false;

        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            const ax = axes[g]!;
            const [lo] = params.halo?.[i]?.[g] ?? [0, 0];
            const bodyDim = blockAval.shape[ax] as number;
            const rawStart = (blockIdx[g] + gridOffset) * blockShape[g] - lo;
            const rawEnd = rawStart + bodyDim;
            const srcStart = Math.max(0, rawStart);
            const srcEnd = Math.min(rawEnd, inputShapes[i][ax]);
            sliceStarts[ax] = srcStart;
            sliceSizes[ax] = Math.max(0, srcEnd - srcStart);
            dstStarts[ax] = srcStart - rawStart; // offset in dst for valid data
            if (dstStarts[ax] > 0 || sliceSizes[ax] < bodyDim) {
              needsPad = true;
            }
          }
        }

        if (needsPad) {
          // Allocate zero-filled and copy valid portion at correct offset.
          // Reuse a single shared zero buffer across blocks to avoid GC churn.
          if (!sharedZeros || sharedZeros.byteLength < blockBytes) {
            sharedZeros = new Uint8Array(blockBytes);
          }
          const blockSlot = backend.malloc(
            blockBytes,
            sharedZeros.byteLength === blockBytes
              ? sharedZeros
              : sharedZeros.subarray(0, blockBytes),
          );

          const validBytes = sliceSizes.reduce((a, b) => a * b, 1) * elemBytes;
          if (validBytes > 0) {
            copyBlock(
              backend,
              inputSlots[i],
              inputShapes[i],
              sliceStarts,
              sliceSizes,
              blockSlot,
              blockAval.shape as number[],
              dstStarts,
              elemBytes,
            );
          }

          blockInputSlots.push(blockSlot);
          ownedSlices.push(blockSlot);
        } else {
          // Full block — allocate and copy
          const blockSlot = backend.malloc(blockBytes);
          const contiguous = isSliceContiguous(
            sliceStarts,
            sliceSizes,
            inputShapes[i],
          );
          if (contiguous) {
            const byteOffset = computeLinearOffset(
              sliceStarts,
              inputShapes[i],
              elemBytes,
            );
            backend.copyBufferToBuffer(
              inputSlots[i],
              byteOffset,
              blockSlot,
              0,
              blockBytes,
            );
          } else {
            copyBlock(
              backend,
              inputSlots[i],
              inputShapes[i],
              sliceStarts,
              sliceSizes,
              blockSlot,
              blockAval.shape as number[],
              new Array(nd).fill(0) as number[],
              elemBytes,
            );
          }
          blockInputSlots.push(blockSlot);
          ownedSlices.push(blockSlot);
        }
      }

      // IncRef consts (body borrows them)
      for (const slot of constSlots) backend.incRef(slot);

      // Build body inputs: [consts, blockInputs, (blockIndex if needed)]
      const bodyInputs: Slot[] = [...constSlots, ...blockInputSlots];

      // Provide block index as extra input when body uses BlockIndex.
      let blockIndexSlot: Slot | undefined;
      if (hasBlockIndex) {
        const idxData = new Int32Array([flatIdx]);
        blockIndexSlot = backend.malloc(4, new Uint8Array(idxData.buffer));
        bodyInputs.push(blockIndexSlot);
      }

      // Flush the batch encoder before body execution: the input slice copies
      // above are encoded in the batch encoder but not yet submitted. If the
      // body program uses the command tape fast path (WebGPU O8), it creates
      // its own GPUCommandEncoder and submits independently — so the copies
      // must be submitted first or the body reads uninitialized buffers.
      // Use flushBatch() to submit mid-batch without changing the depth —
      // this works correctly even when a parent scope (e.g.,
      // executeAssocScanBlockMap) has started a batch around us.
      // For backends without flushBatch, fall back to endBatch/beginBatch.
      if (backend.flushBatch) {
        backend.flushBatch();
      } else if (useBatching) {
        backend.endBatch!();
        backend.beginBatch!();
      }

      // Execute body program
      const bodyResult = bodyProgram.execute(bodyInputs);
      pending.push(...bodyResult.pending);

      // Flush pending from body execution
      flushPending(pending, backend);

      // Release consts, block input slices, and block index slot
      for (const slot of constSlots) backend.decRef(slot);
      for (const slot of ownedSlices) backend.decRef(slot);
      if (blockIndexSlot) backend.decRef(blockIndexSlot);

      // Copy body outputs into the preallocated output buffers
      for (let o = 0; o < numOutputs; o++) {
        const axes = outAxes[o];
        const blockAval = bodyOutAvals[o];
        const nd = blockAval.shape.length;
        const elemBytes = byteWidth(blockAval.dtype);

        // Compute the destination region in the output buffer
        const dstStarts: number[] = new Array(nd).fill(0);
        const copySizes: number[] = [...blockAval.shape] as number[];
        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            const ax = axes[g]!;
            dstStarts[ax] = (blockIdx[g] + gridOffset) * blockShape[g];
            // Clamp copy size to output extent
            const remaining = outputShapes[o][ax] - dstStarts[ax];
            if (remaining < blockShape[g]) {
              copySizes[ax] = Math.max(0, remaining);
            }
          }
        }

        const totalCopyElems = copySizes.reduce((a, b) => a * b, 1);
        if (totalCopyElems > 0) {
          copyBlock(
            backend,
            bodyResult.outputs[o],
            blockAval.shape as number[],
            new Array(nd).fill(0) as number[], // source starts at 0
            copySizes,
            outputSlots[o],
            outputShapes[o],
            dstStarts,
            elemBytes,
          );
        }

        backend.decRef(bodyResult.outputs[o]);
      }

      // Periodic batch flush
      if (
        useBatching &&
        flatIdx < numBlocks - 1 &&
        (flatIdx + 1) % BATCH_SIZE === 0
      ) {
        backend.endBatch!();
        backend.beginBatch!();
      }
    }
  } finally {
    if (useBatching) backend.endBatch!();
  }

  flushPending(pending, backend);
  return { outputs: outputSlots, pending };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Check if a slice is contiguous in memory (can be copied in one go). */
export function isSliceContiguous(
  starts: number[],
  sizes: number[],
  fullShape: number[],
): boolean {
  // In row-major layout, a slice is contiguous iff for every dimension d
  // where sizes[d] > 1, ALL inner dimensions d' > d have
  // sizes[d'] == fullShape[d']. This ensures the stride matches.
  const nd = starts.length;
  for (let d = 0; d < nd; d++) {
    if (sizes[d] <= 1) continue;
    for (let d2 = d + 1; d2 < nd; d2++) {
      if (sizes[d2] !== fullShape[d2]) return false;
    }
    break; // Only need to check the outermost non-trivial dim
  }
  return true;
}

/** Compute linear byte offset for a multi-index into a row-major shape. */
export function computeLinearOffset(
  starts: number[],
  fullShape: number[],
  elemBytes: number,
): number {
  const nd = starts.length;
  let offset = 0;
  let stride = elemBytes;
  for (let d = nd - 1; d >= 0; d--) {
    offset += starts[d] * stride;
    stride *= fullShape[d];
  }
  return offset;
}

/** Flush pending operations. */
export function flushPending(
  pending: IPendingExecute[],
  backend: Backend,
): void {
  if (pending.length === 0) return;
  backend.beginBatch?.();
  try {
    for (const p of pending) {
      p.prepareSync();
      p.submit();
    }
  } finally {
    backend.endBatch?.();
  }
  pending.length = 0;
}

/**
 * Copy a block between buffers with arbitrary source/dest offsets and shapes.
 * Handles both contiguous (single copy) and strided (row-by-row) cases.
 */
export function copyBlock(
  backend: Backend,
  srcSlot: Slot,
  srcFullShape: number[],
  srcStarts: number[],
  copySizes: number[],
  dstSlot: Slot,
  dstFullShape: number[],
  dstStarts: number[],
  elemBytes: number,
): void {
  const nd = copySizes.length;

  // Check contiguity on both sides
  const srcContiguous = isSliceContiguous(srcStarts, copySizes, srcFullShape);
  const dstContiguous = isSliceContiguous(dstStarts, copySizes, dstFullShape);

  if (srcContiguous && dstContiguous) {
    // Single copy
    const srcOffset = computeLinearOffset(srcStarts, srcFullShape, elemBytes);
    const dstOffset = computeLinearOffset(dstStarts, dstFullShape, elemBytes);
    const totalBytes = copySizes.reduce((a, b) => a * b, 1) * elemBytes;
    backend.copyBufferToBuffer(
      srcSlot,
      srcOffset,
      dstSlot,
      dstOffset,
      totalBytes,
    );
    return;
  }

  // Strided: copy innermost-dimension rows
  const innermostBytes = copySizes[nd - 1] * elemBytes;
  const numRows = copySizes.slice(0, -1).reduce((a, b) => a * b, 1) || 1;

  const srcStrides = computeStrides(srcFullShape, elemBytes);
  const dstStrides = computeStrides(dstFullShape, elemBytes);

  const idx = new Array(Math.max(nd - 1, 0)).fill(0);
  for (let row = 0; row < numRows; row++) {
    let srcOffset = 0;
    let dstOffset = 0;
    for (let d = 0; d < nd - 1; d++) {
      srcOffset += (srcStarts[d] + idx[d]) * srcStrides[d];
      dstOffset += (dstStarts[d] + idx[d]) * dstStrides[d];
    }
    srcOffset += srcStarts[nd - 1] * elemBytes;
    dstOffset += dstStarts[nd - 1] * elemBytes;

    backend.copyBufferToBuffer(
      srcSlot,
      srcOffset,
      dstSlot,
      dstOffset,
      innermostBytes,
    );

    // Increment multi-index (row-major)
    for (let d = nd - 2; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < copySizes[d]) break;
      idx[d] = 0;
    }
  }
}

/** Compute byte strides for a shape (row-major, last dim is innermost). */
export function computeStrides(shape: number[], elemBytes: number): number[] {
  const nd = shape.length;
  const strides = new Array(nd);
  strides[nd - 1] = elemBytes;
  for (let d = nd - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * shape[d + 1];
  }
  return strides;
}

// ---------------------------------------------------------------------------
// Generic blocked-data-movement primitives
// ---------------------------------------------------------------------------

/**
 * Gather one element per block along a given axis.
 *
 * For each source buffer, allocates an output where axis `axis` has size
 * `M = ceil(axisLen / blockSize)` and copies the element at index
 * `min((i+1)*blockSize - 1, axisLen - 1)` to position `i`.
 */
export function gatherAxisPoints(
  backend: Backend,
  srcSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axis: number,
  blockSize: number,
  axisLen: number,
): Slot[] {
  const M = Math.ceil(axisLen / blockSize);
  const outSlots: Slot[] = [];

  backend.beginBatch?.();
  for (let k = 0; k < srcSlots.length; k++) {
    const shape = srcShapes[k];
    const elemBytes = byteWidth(dtypes[k]);
    const innerBytes =
      shape.slice(axis + 1).reduce((a, b) => a * b, 1) * elemBytes;

    if (axis === 0) {
      // Fast path: contiguous elements
      const outBytes = M * innerBytes;
      const outSlot = backend.malloc(outBytes);
      for (let i = 0; i < M; i++) {
        const srcIdx = Math.min((i + 1) * blockSize - 1, axisLen - 1);
        backend.copyBufferToBuffer(
          srcSlots[k],
          srcIdx * innerBytes,
          outSlot,
          i * innerBytes,
          innerBytes,
        );
      }
      outSlots.push(outSlot);
    } else {
      // General case: fiber-by-fiber for non-contiguous axis
      const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
      const srcOuterStride = shape[axis] * innerBytes;
      const dstOuterStride = M * innerBytes;
      const outTotalBytes = outerSize * M * innerBytes;
      const outSlot = backend.malloc(outTotalBytes);
      for (let i = 0; i < M; i++) {
        const srcIdx = Math.min((i + 1) * blockSize - 1, axisLen - 1);
        for (let f = 0; f < outerSize; f++) {
          backend.copyBufferToBuffer(
            srcSlots[k],
            f * srcOuterStride + srcIdx * innerBytes,
            outSlot,
            f * dstOuterStride + i * innerBytes,
            innerBytes,
          );
        }
      }
      outSlots.push(outSlot);
    }
  }
  backend.endBatch?.();
  return outSlots;
}

/**
 * Copy a contiguous range along a given axis across multi-leaf slot arrays.
 *
 * Copies elements `[axisStart, axisStart + axisLen)` along `axis` from
 * each src to the corresponding position in dst.
 */
export function copyAxisRange(
  backend: Backend,
  srcSlots: Slot[],
  dstSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axis: number,
  axisStart: number,
  axisLen: number,
): void {
  for (let k = 0; k < srcSlots.length; k++) {
    const shape = srcShapes[k];
    const elemBytes = byteWidth(dtypes[k]);
    const innerBytes =
      shape.slice(axis + 1).reduce((a, b) => a * b, 1) * elemBytes;

    if (axis === 0) {
      // Fast path: contiguous
      const offset = axisStart * innerBytes;
      const copyBytes = axisLen * innerBytes;
      backend.copyBufferToBuffer(
        srcSlots[k],
        offset,
        dstSlots[k],
        offset,
        copyBytes,
      );
    } else {
      // General case: fiber-by-fiber
      const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
      const outerStride = shape[axis] * innerBytes;
      const offset = axisStart * innerBytes;
      const copyBytes = axisLen * innerBytes;
      for (let f = 0; f < outerSize; f++) {
        backend.copyBufferToBuffer(
          srcSlots[k],
          f * outerStride + offset,
          dstSlots[k],
          f * outerStride + offset,
          copyBytes,
        );
      }
    }
  }
}

/** Declarative input description for {@link mapOverBlocks}. */
export interface BlockInput {
  slots: Slot[];
  shapes: number[][];
  dtypes: DType[];
  /** "point": gather one element at `blockIdx + indexOffset`. "block": gather `[B, …]` slice. */
  mode: "point" | "block";
  /** For "point" mode: offset added to blockIdx to compute source index. */
  indexOffset?: number;
}

/**
 * Run a compiled body program over a grid of blocks, materializing
 * point-gathered and block-gathered inputs per iteration.
 *
 * For each block in `[gridStart, gridEnd)`: materializes each input group
 * according to its mode, runs the body, copies only valid output elements
 * to the destination.
 *
 * **Partial trailing block contract:** Block-mode inputs are always
 * materialized as full `[blockSize, …]` temporaries even when fewer than
 * `blockSize` elements remain. Only the valid prefix is copied in; the
 * suffix is uninitialized padding. Only the valid prefix of the body result
 * is written back to the destination — padding elements are never exposed.
 *
 * The body program must tolerate uninitialized padding in block-mode inputs
 * for the trailing block. Elementwise and vmapped bodies satisfy this
 * naturally because the padding outputs are discarded.
 *
 * @param constSlots  Constant slots retained/released symmetrically per iteration.
 * @param inputs      Declarative input groups — see {@link BlockInput}.
 * @param numConsts   Number of leading constant slots in the body program signature.
 *                    Must equal `constSlots.length`.
 */
export function mapOverBlocks(
  backend: Backend,
  bodyProgram: JitProgram,
  constSlots: Slot[],
  inputs: BlockInput[],
  outputSlots: Slot[],
  outputShapes: number[][],
  outputDtypes: DType[],
  axis: number,
  blockSize: number,
  axisLen: number,
  gridStart: number,
  gridEnd: number,
  numConsts: number,
  dimBindings?: ReadonlyMap<string, number>,
): void {
  if (numConsts !== constSlots.length) {
    throw new Error(
      `mapOverBlocks: numConsts (${numConsts}) !== constSlots.length (${constSlots.length})`,
    );
  }
  if (gridStart < 0 || gridEnd < gridStart) {
    throw new Error(
      `mapOverBlocks: invalid grid range [${gridStart}, ${gridEnd})`,
    );
  }
  if (outputSlots.length !== outputShapes.length) {
    throw new Error(
      `mapOverBlocks: outputSlots.length (${outputSlots.length}) !== outputShapes.length (${outputShapes.length})`,
    );
  }
  if (outputSlots.length !== outputDtypes.length) {
    throw new Error(
      `mapOverBlocks: outputSlots.length (${outputSlots.length}) !== outputDtypes.length (${outputDtypes.length})`,
    );
  }
  const numOutputs = outputSlots.length;
  const numBlocks = gridEnd - gridStart;
  const useBatching = backend.beginBatch != null && numBlocks > 1;
  if (useBatching) backend.beginBatch!();

  for (let blockIdx = gridStart; blockIdx < gridEnd; blockIdx++) {
    const blockStart = blockIdx * blockSize;
    const blockEnd = Math.min(blockStart + blockSize, axisLen);
    const blockLen = blockEnd - blockStart;

    // Materialize inputs for this block
    const bodyInputSlots: Slot[] = [];
    const tempSlots: Slot[] = []; // track temporaries for cleanup

    for (const input of inputs) {
      for (let k = 0; k < input.slots.length; k++) {
        const elemBytes = byteWidth(input.dtypes[k]);
        const shape = input.shapes[k];
        const innerBytes =
          shape.slice(axis + 1).reduce((a, b) => a * b, 1) * elemBytes;
        const outerSize =
          axis === 0 ? 1 : shape.slice(0, axis).reduce((a, b) => a * b, 1);
        const srcOuterStride = shape[axis] * innerBytes;

        if (input.mode === "point") {
          const srcIdx = blockIdx + (input.indexOffset ?? 0);
          const srcAxisLen = shape[axis];
          if (srcIdx < 0 || srcIdx >= srcAxisLen) {
            throw new Error(
              `mapOverBlocks: point-mode source index ${srcIdx} out of range [0, ${srcAxisLen}) for input leaf ${k} at blockIdx=${blockIdx}`,
            );
          }
          const totalBytes = outerSize * innerBytes;
          const slot = backend.malloc(totalBytes);
          if (axis === 0) {
            backend.copyBufferToBuffer(
              input.slots[k],
              srcIdx * innerBytes,
              slot,
              0,
              innerBytes,
            );
          } else {
            const dstOuterStride = innerBytes; // output has no axis dim
            for (let f = 0; f < outerSize; f++) {
              backend.copyBufferToBuffer(
                input.slots[k],
                f * srcOuterStride + srcIdx * innerBytes,
                slot,
                f * dstOuterStride,
                innerBytes,
              );
            }
          }
          bodyInputSlots.push(slot);
          tempSlots.push(slot);
        } else {
          // "block" mode: full [.., B, …] temporary, only valid prefix copied in
          const fullBytes = outerSize * blockSize * innerBytes;
          const slot = backend.malloc(fullBytes);
          const copyBytes = blockLen * innerBytes;
          if (axis === 0) {
            backend.copyBufferToBuffer(
              input.slots[k],
              blockStart * innerBytes,
              slot,
              0,
              copyBytes,
            );
          } else {
            const dstOuterStride = blockSize * innerBytes;
            for (let f = 0; f < outerSize; f++) {
              backend.copyBufferToBuffer(
                input.slots[k],
                f * srcOuterStride + blockStart * innerBytes,
                slot,
                f * dstOuterStride,
                copyBytes,
              );
            }
          }
          bodyInputSlots.push(slot);
          tempSlots.push(slot);
        }
      }
    }

    // IncRef consts for body execution
    for (const s of constSlots) backend.incRef(s);

    // Execute body: [consts, ...inputs] -> [outputs]
    const allInputs = [...constSlots, ...bodyInputSlots];
    const bodyResult = bodyProgram.execute(allInputs, dimBindings);
    if (bodyResult.pending.length > 0)
      flushPending(bodyResult.pending, backend);

    // DecRef consts and temporary inputs
    for (const s of constSlots) backend.decRef(s);
    for (const s of tempSlots) backend.decRef(s);

    // Copy valid output elements to destination
    for (let k = 0; k < numOutputs; k++) {
      const elemBytes = byteWidth(outputDtypes[k]);
      const shape = outputShapes[k];
      const innerBytes =
        shape.slice(axis + 1).reduce((a, b) => a * b, 1) * elemBytes;

      if (axis === 0) {
        backend.copyBufferToBuffer(
          bodyResult.outputs[k],
          0,
          outputSlots[k],
          blockStart * innerBytes,
          blockLen * innerBytes,
        );
      } else {
        const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
        const srcOuterStride = blockSize * innerBytes; // body output has blockSize along axis
        const dstOuterStride = shape[axis] * innerBytes;
        for (let f = 0; f < outerSize; f++) {
          backend.copyBufferToBuffer(
            bodyResult.outputs[k],
            f * srcOuterStride,
            outputSlots[k],
            f * dstOuterStride + blockStart * innerBytes,
            blockLen * innerBytes,
          );
        }
      }
      backend.decRef(bodyResult.outputs[k]);
    }
  }

  if (useBatching) backend.endBatch!();
}
