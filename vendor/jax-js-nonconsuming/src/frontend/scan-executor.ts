/**
 * @file Unified scan executor — single execution path for all scan strategies.
 *
 * This replaces v1's dual loop (eager Primitive.Scan + JIT scanRunner) with one
 * function that handles ownership, flush, and dispatch for all backends and all
 * plan paths (fallback, compiled-loop, preencoded-routine).
 */

import { byteWidth } from "../alu";
import type { Backend, Slot } from "../backend";
import type { IPendingExecute } from "./array";
import { Array as JaxArray } from "./array";
import { _associativeScanCoreImpl, ShapedArray } from "./core";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram } from "./jit";
import { type AssocScanPlan, type ScanPlan } from "./scan-plan";
import type { WasmBackend } from "../backend/wasm";
import type { NativeScanMultiParams, WebGPUBackend } from "../backend/webgpu";
import {
  concreteDim,
  concreteShape,
  type Dim,
  hasSymbolicDims,
  isSymbolicDim,
  resolveShape,
  ShapeTracker,
} from "../shape";
import {
  copyAxisRange,
  executeBlockMap,
  type ExecuteBlockMapParams,
  gatherAxisPoints,
  mapOverBlocks,
} from "./block-map-executor";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ExecuteScanParams {
  backend: Backend;
  plan: ScanPlan;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  length: number;
  numCarry: number;
  numConsts: number;
  numX: number;
  numY: number;
  reverse: boolean;
  constSlots: Slot[];
  initCarrySlots: Slot[];
  xsSlots: Slot[];
  xsAvals: ShapedArray[];
  /** Preallocated output slots: [carry_out..., stacked_ys...] */
  outputSlots: Slot[];
  /** Dim bindings for resolving symbolic sizes in body program execution */
  dimBindings?: ReadonlyMap<string, number>;
}

export interface ExecuteScanResult {
  outputs: Slot[];
  pending: IPendingExecute[];
}

/**
 * Execute a scan loop. Dispatches to the appropriate strategy based on the plan.
 *
 * Ownership contract:
 * - constSlots: borrowed (incRef'd before each body call, not consumed)
 * - initCarrySlots: consumed (absorbed into first iteration)
 * - xsSlots: borrowed (sliced per iteration via createView or readSync)
 * - outputSlots: filled in-place (carry outputs + stacked Y outputs)
 */
export function executeScan(params: ExecuteScanParams): ExecuteScanResult {
  switch (params.plan.path) {
    case "fallback":
      return executeScanFallback(params);
    case "compiled-loop":
      return executeScanCompiledLoop(params);
    case "preencoded-routine":
      return executeScanPreencodedRoutine(params);
    case "preencoded-multi-step":
      return executeScanPreencodedMultiStep(params);
  }
}

// ---------------------------------------------------------------------------
// Fallback: JS loop calling bodyProgram.execute() per iteration
// ---------------------------------------------------------------------------

function executeScanFallback(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts: _numConsts,
    numX: _numX,
    numY,
    reverse,
    constSlots,
    initCarrySlots,
    xsSlots,
    xsAvals,
    outputSlots,
    dimBindings,
  } = params;

  // Helper: resolve size of an aval, handling symbolic shapes via dimBindings
  const resolveAvalSize = (aval: ShapedArray): number => {
    if (hasSymbolicDims(aval.shape)) {
      if (!dimBindings) throw new Error("Symbolic shape but no dimBindings");
      const shape = resolveShape(aval.shape, dimBindings);
      return shape.reduce((a, b) => a * b, 1);
    }
    return aval.size as number;
  };

  // Compute per-xs byte strides (size of one iteration's slice)
  const xsStrides = xsAvals.map(
    (aval) => resolveAvalSize(aval) * byteWidth(aval.dtype),
  );

  // Compute per-y byte strides from body jaxpr outputs
  const yOutAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const ysStrides = yOutAvals.map(
    (aval) => resolveAvalSize(aval) * byteWidth(aval.dtype),
  );

  // Current carry slots — start with initCarry.
  // IncRef so the loop can uniformly decRef old carry each iteration
  // (initCarrySlots are borrowed from the caller who frees them separately).
  let carry = initCarrySlots.slice();
  for (const slot of carry) backend.incRef(slot);

  // Y output slots from the preallocated outputs
  const ysOutputSlots = outputSlots.slice(numCarry);

  // Track pending operations
  const pending: IPendingExecute[] = [];

  // Sub-batch GPU commands: instead of one queue.submit() per iteration,
  // batch SCAN_BATCH_SIZE iterations into one submit. This reduces O(2N)
  // submits to O(N/SCAN_BATCH_SIZE) while limiting deferred buffer
  // accumulation. The batch encoder accumulates dispatches and copies;
  // nested flushPending calls become no-ops (depth tracking).
  const SCAN_BATCH_SIZE = 256;
  const useBatching = backend.beginBatch != null && length > 1;

  if (useBatching) backend.beginBatch!();

  try {
    for (let step = 0; step < length; step++) {
      const i = reverse ? length - 1 - step : step;

      // Invariant 1: Flush pending ops before each body invocation
      flushPending(pending);

      // Slice xs for this iteration
      const xSlices = sliceXsAtIteration(
        backend,
        xsSlots,
        xsStrides,
        xsAvals,
        i,
      );

      // IncRef consts so body can consume them
      for (const slot of constSlots) backend.incRef(slot);

      // Build body inputs: [consts, carry, xSlices]
      // carry is consumed (body takes ownership)
      const bodyInputs = [...constSlots, ...carry, ...xSlices];

      // Execute body
      const bodyResult = bodyProgram.execute(bodyInputs, dimBindings);
      pending.push(...bodyResult.pending);

      // Flush pending ops from body execution before reading output slots
      // (the body's kernels must be dispatched before we can copy from them)
      flushPending(pending);

      const newCarry = bodyResult.outputs.slice(0, numCarry);
      const ySlices = bodyResult.outputs.slice(numCarry);

      // Release borrowed consts and created x slice slots.
      // Note: JitProgram.execute() already inserts incref steps for any output
      // that is a passthrough from an input or appears multiple times in the
      // output list, so each output position has its own reference. No extra
      // alias-protection incRef is needed here — the JIT's refs protect outputs
      // from being prematurely freed by these input decRefs.
      for (const slot of constSlots) backend.decRef(slot);
      for (const slot of xSlices) backend.decRef(slot);

      // Invariant 3: Y stacking — copy y slices into preallocated output buffers
      for (let yi = 0; yi < numY; yi++) {
        if (ysStrides[yi] > 0) {
          copySliceToBuffer(
            backend,
            ysOutputSlots[yi],
            ySlices[yi],
            i,
            ysStrides[yi],
            ysStrides[yi],
          );
        }
        // Free the y slice (it's been copied into the output buffer)
        backend.decRef(ySlices[yi]);
      }

      // Invariant 2: Carry lifecycle — body.execute() borrows inputs (does not
      // consume them). We must explicitly release old carry slots. The JIT's
      // incref for passthrough/duplicate outputs ensures that any carry slot
      // reappearing in newCarry has an extra ref, so this decRef is safe.
      for (const slot of carry) backend.decRef(slot);
      carry = newCarry;

      // Periodic flush: end current batch and start a new one to release
      // deferred buffers and limit GPU memory accumulation.
      if (
        useBatching &&
        step < length - 1 &&
        (step + 1) % SCAN_BATCH_SIZE === 0
      ) {
        backend.endBatch!();
        backend.beginBatch!();
      }
    }
  } finally {
    if (useBatching) backend.endBatch!();
  }

  // Flush any remaining pending ops before writing final carry
  flushPending(pending);

  // Write final carry to output slots
  const carryOutputSlots = outputSlots.slice(0, numCarry);
  for (let ci = 0; ci < numCarry; ci++) {
    if (carry[ci] === carryOutputSlots[ci]) {
      // Slot is already the output (shouldn't normally happen in fallback)
      continue;
    }
    // Copy carry data into the preallocated output slot
    const carryAval = bodyJaxpr.outs[ci].aval;
    const carrySize = resolveAvalSize(carryAval) * byteWidth(carryAval.dtype);
    copySliceToBuffer(
      backend,
      carryOutputSlots[ci],
      carry[ci],
      0,
      0,
      carrySize,
    );
    backend.decRef(carry[ci]);
  }

  return { outputs: outputSlots, pending };
}

// ---------------------------------------------------------------------------
// Compiled-loop: WASM or WebGPU native scan
// ---------------------------------------------------------------------------

function executeScanCompiledLoop(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "compiled-loop") throw new Error("unreachable");
  if (!plan.params) throw new Error("compiled-loop plan missing params");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  if (backend.type === "webgpu") {
    // WebGPU native scan — no internal slots needed
    const webgpuBackend = backend as WebGPUBackend;
    webgpuBackend.dispatchNativeScanGeneral(
      plan.executable,
      plan.params as NativeScanMultiParams,
      constSlots,
      initCarrySlots,
      xsSlots,
      carryOutSlots,
      ysStackedSlots,
    );
  } else {
    // WASM native scan
    const wasmBackend = backend as WasmBackend;
    wasmBackend.dispatchNativeScanGeneral(
      plan.executable,
      plan.params as import("../backend/wasm").NativeScanGeneralParams,
      params.length,
      constSlots,
      initCarrySlots,
      xsSlots,
      carryOutSlots,
      ysStackedSlots,
    );
  }

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Preencoded-routine: WebGPU routine scan with uniform offsets (P4)
// ---------------------------------------------------------------------------

function executeScanPreencodedRoutine(
  params: ExecuteScanParams,
): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "preencoded-routine") throw new Error("unreachable");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  const webgpuBackend = backend as WebGPUBackend;
  webgpuBackend.dispatchPreencodedScan(
    plan.preencodedParams,
    constSlots,
    initCarrySlots,
    xsSlots,
    carryOutSlots,
    ysStackedSlots,
  );

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Preencoded multi-step: WebGPU multi-kernel scan with per-step offsets (P2c)
// ---------------------------------------------------------------------------

function executeScanPreencodedMultiStep(
  params: ExecuteScanParams,
): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "preencoded-multi-step") throw new Error("unreachable");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  const webgpuBackend = backend as WebGPUBackend;
  webgpuBackend.dispatchPreencodedMultiStepScan(
    plan.prepared,
    constSlots,
    initCarrySlots,
    xsSlots,
    carryOutSlots,
    ysStackedSlots,
  );

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Associative scan executor (Kogge-Stone)
// ---------------------------------------------------------------------------

export interface ExecuteAssocScanParams {
  backend: Backend;
  plan: AssocScanPlan;
  bodyJaxpr: Jaxpr;
  numLeaves: number;
  numConsts: number;
  axis: number;
  reverse: boolean;
  constSlots: Slot[];
  elemSlots: Slot[];
  constAvals: ShapedArray[];
  elemAvals: ShapedArray[];
  /** Preallocated output slots (may be replaced in fallback path). */
  outputSlots: Slot[];
  /** Dimension bindings for symbolic shape resolution. */
  dimBindings?: ReadonlyMap<string, number>;
}

export interface ExecuteAssocScanResult {
  /** Final output slots — same as input for native paths, replaced for fallback. */
  outputs: Slot[];
  pending: IPendingExecute[];
}

/**
 * Execute an associative scan. Dispatches based on plan path:
 * - compiled-loop: WASM-compiled Kogge-Stone ladder
 * - webgpu-fused: fused WGSL shader per round
 * - fallback: JS Kogge-Stone loop via vmap + evalJaxpr
 */
export function executeAssociativeScan(
  params: ExecuteAssocScanParams,
): ExecuteAssocScanResult {
  const {
    backend,
    plan,
    bodyJaxpr,
    numLeaves,
    numConsts: _numConsts,
    axis,
    reverse,
    constSlots,
    elemSlots,
    constAvals,
    elemAvals,
    outputSlots,
    dimBindings,
  } = params;

  if (plan.path === "compiled-loop-blocked") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    if (axis === 0) {
      // Fast path: scan axis is contiguous — dispatch directly.
      (backend as WasmBackend).dispatchBlockedAssociativeScan(
        plan.executable,
        plan.params,
        N,
        constSlots,
        elemSlots,
        outputSlots,
      );
    } else {
      // Strided boundary copies: transpose scan axis to 0, run WASM, transpose back.
      const tempIn: Slot[] = [];
      const tempOut: Slot[] = [];
      for (let k = 0; k < numLeaves; k++) {
        const shape = resolveLeafShape(elemAvals[k].shape, dimBindings);
        const totalBytes = shapeProduct(shape) * byteWidth(elemAvals[k].dtype);
        tempIn.push(backend.malloc(totalBytes));
        tempOut.push(backend.malloc(totalBytes));
      }
      // Gather: input (scan at `axis`) → temp (scan at axis 0)
      for (let k = 0; k < numLeaves; k++) {
        transposeAxisToFront(
          backend,
          elemSlots[k],
          tempIn[k],
          resolveLeafShape(elemAvals[k].shape, dimBindings),
          axis,
          byteWidth(elemAvals[k].dtype),
        );
      }
      (backend as WasmBackend).dispatchBlockedAssociativeScan(
        plan.executable,
        plan.params,
        N,
        constSlots,
        tempIn,
        tempOut,
      );
      // Scatter: temp (scan at axis 0) → output (scan at `axis`)
      for (let k = 0; k < numLeaves; k++) {
        transposeAxisFromFront(
          backend,
          tempOut[k],
          outputSlots[k],
          resolveLeafShape(elemAvals[k].shape, dimBindings),
          axis,
          byteWidth(elemAvals[k].dtype),
        );
      }
      for (const s of tempIn) backend.decRef(s);
      for (const s of tempOut) backend.decRef(s);
    }
    return { outputs: outputSlots, pending: [] };
  }

  if (plan.path === "webgpu-block-map") {
    return executeAssocScanBlockMap(params);
  }

  if (plan.path === "decoupled-fallback") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    (backend as WebGPUBackend).dispatchDecoupledFallbackScan(
      elemSlots[0],
      outputSlots[0],
      N,
      plan.op,
      plan.dtype,
      plan.blockSize,
    );
    return { outputs: outputSlots, pending: [] };
  }

  // Fallback: JS Kogge-Stone loop
  const resolveAvalShape = (shape: Dim[]): number[] =>
    hasSymbolicDims(shape)
      ? concreteShape(resolveShape(shape, dimBindings!))
      : (shape as number[]);

  const constArrays = constSlots.map((slot, i) => {
    backend.incRef(slot);
    const aval = constAvals[i];
    return new JaxArray({
      source: slot,
      st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
      dtype: aval.dtype,
      weakType: aval.weakType,
      backend,
      committed: false,
    });
  });

  const elemArrays = elemSlots.map((slot, i) => {
    backend.incRef(slot);
    const aval = elemAvals[i];
    return new JaxArray({
      source: slot,
      st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
      dtype: aval.dtype,
      weakType: aval.weakType,
      backend,
      committed: false,
    });
  });

  const results = _associativeScanCoreImpl!(
    bodyJaxpr,
    constArrays,
    elemArrays,
    numLeaves,
    axis,
    reverse,
  );

  // Extract slots from results, replacing pre-allocated output buffers.
  const finalOutputs: Slot[] = [];
  for (let i = 0; i < numLeaves; i++) {
    const resultArr = results[i] as JaxArray;
    resultArr._flushPendingSync();
    const slot = resultArr._realizeSource();
    backend.incRef(slot);
    // Free the pre-allocated output buffer
    backend.decRef(outputSlots[i]);
    finalOutputs.push(slot);
    resultArr.dispose();
  }

  // Dispose input Array wrappers
  for (const a of constArrays) a.dispose();
  for (const a of elemArrays) a.dispose();

  return { outputs: finalOutputs, pending: [] };
}

// ---------------------------------------------------------------------------
// Block-map–based associative scan execution
// ---------------------------------------------------------------------------

/**
 * Execute an associative scan using the block-map decomposition:
 * 1. Local scan: block_map with WorkgroupAssociativeScan body
 * 2. If M > 1: gather block summaries → recursive scan → apply
 *
 * This path reuses block-map.ts's fused shader infrastructure for the
 * local scan, eliminating the standalone WebGPU assocScan shaders.
 */
function executeAssocScanBlockMap(
  params: ExecuteAssocScanParams,
): ExecuteAssocScanResult {
  const { backend, plan } = params;

  if (plan.path !== "webgpu-block-map") {
    throw new Error("executeAssocScanBlockMap: expected webgpu-block-map plan");
  }

  // Batch all GPU operations (dispatches + buffer copies) into a single
  // queue.submit(). Without this, each dispatchBlockMapFused /
  // copyBufferToBuffer creates its own command encoder + submit, causing
  // massive overhead (e.g., 78 submits for N=3200).
  backend.beginBatch?.();
  try {
    return executeAssocScanBlockMapInner(params);
  } finally {
    backend.endBatch?.();
  }
}

function executeAssocScanBlockMapInner(
  params: ExecuteAssocScanParams,
): ExecuteAssocScanResult {
  const {
    backend,
    plan,
    numLeaves,
    axis,
    constSlots,
    elemSlots,
    elemAvals,
    constAvals,
    outputSlots,
    dimBindings,
  } = params;

  if (plan.path !== "webgpu-block-map") {
    throw new Error("executeAssocScanBlockMap: expected webgpu-block-map plan");
  }

  const {
    localScan,
    scanBodyJaxpr,
    blockSize: B,
    numConsts,
    applyVmapProgram,
  } = plan;
  const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
  const M = Math.ceil(N / B);

  // Resolve elem shapes once — elemAvals may carry SymDim when dynamic_axes is
  // used. Every subsequent shape access uses this concrete resolved array.
  const resolvedElemShapes: number[][] = elemAvals.map((a) =>
    dimBindings ? resolveShape(a.shape, dimBindings) : (a.shape as number[]),
  );

  const dtypes = elemAvals.map((a) => a.dtype);

  // --- Phase 1: Local scan via block_map ---
  const gridShape = [M];
  const blockShape = [B];

  const inputShapes = resolvedElemShapes;
  const outputShapes = resolvedElemShapes;

  // Allocate local scan output buffers
  const localScanSlots: Slot[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const totalBytes =
      resolvedElemShapes[k].reduce((a, b) => a * b, 1) *
      byteWidth(elemAvals[k].dtype);
    localScanSlots.push(backend.malloc(totalBytes));
  }

  const blockMapParams: ExecuteBlockMapParams = {
    backend,
    bodyProgram: localScan.bodyProgram,
    bodyJaxpr: localScan.bodyJaxpr,
    blockShape,
    inAxes: localScan.inAxes,
    outAxes: localScan.outAxes,
    numConsts: localScan.numConsts,
    numInputs: localScan.numInputs,
    gridShape,
    inputShapes,
    outputShapes,
    constSlots,
    inputSlots: elemSlots,
    outputSlots: localScanSlots,
    constInfos: localScan.constInfos,
    needsLeafPacking: plan.needsLeafPacking,
  };

  const bmResult = executeBlockMap(blockMapParams);
  // Flush any pending block map operations
  if (bmResult.pending.length > 0) {
    flushPending(bmResult.pending);
  }

  if (M === 1) {
    // Single block: local scan output IS the final output.
    copyAxisRange(
      backend,
      localScanSlots,
      outputSlots,
      resolvedElemShapes,
      dtypes,
      axis,
      0,
      N,
    );
    for (const s of localScanSlots) backend.decRef(s);
    return { outputs: outputSlots, pending: [] };
  }

  // --- Phase 2: Gather block summaries ---
  const summarySlots = gatherAxisPoints(
    backend,
    localScanSlots,
    resolvedElemShapes,
    dtypes,
    axis,
    B,
    N,
  );

  // --- Phase 3: Recursive prefix scan of summaries ---
  // Recurse through public dispatch with the same plan (self-similar).
  // For M ≤ B the recursive call is single-block; for M > B it decomposes.
  const scannedSummarySlots: Slot[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const summaryShape = [...resolvedElemShapes[k]];
    summaryShape[axis] = M;
    scannedSummarySlots.push(
      backend.malloc(
        summaryShape.reduce((a, b) => a * b, 1) * byteWidth(elemAvals[k].dtype),
      ),
    );
  }

  const summaryElemAvals = elemAvals.map((a, k) => {
    const shape = [...resolvedElemShapes[k]];
    shape[axis] = M;
    return new ShapedArray(shape, a.dtype, a.weakType);
  });

  const summaryResult = executeAssociativeScan({
    backend,
    plan,
    bodyJaxpr: scanBodyJaxpr,
    numLeaves,
    numConsts,
    axis,
    reverse: false,
    constSlots,
    elemSlots: summarySlots,
    constAvals,
    elemAvals: summaryElemAvals,
    outputSlots: scannedSummarySlots,
    dimBindings,
  });
  if (summaryResult.pending.length > 0) flushPending(summaryResult.pending);

  // The summary scan may replace outputSlots (fallback path).
  const finalScannedSummarySlots = summaryResult.outputs;

  // Free raw summaries
  for (const s of summarySlots) backend.decRef(s);

  // --- Phase 4: Copy block 0 + apply blocks 1..M-1 ---
  copyAxisRange(
    backend,
    localScanSlots,
    outputSlots,
    resolvedElemShapes,
    dtypes,
    axis,
    0,
    Math.min(B, N),
  );

  if (M > 1 && axis === 0) {
    // Fused Phase 4: single block_map dispatch over M-1 blocks.
    // Uses pointInputs for per-workgroup prefix access and gridOffset=1
    // to start mapped inputs/outputs from block 1.
    //
    // Body signature: [consts, prefix_0..L (broadcast), block_0..L (mapped)]
    //              -> [result_0..L]
    // Point inputs (prefix): shape [d1,d2,...] per element, indexed by workgroup_id
    //   - Buffer is scannedSummary[M, d1, d2, ...], workgroup i reads summary[i]
    // Mapped inputs (block): shape [N, d1, d2, ...], gridOffset=1 shifts to block 1+
    // Mapped outputs: shape [N, d1, d2, ...], gridOffset=1 writes from block 1+

    // Per-element shapes for point inputs (prefix): strip the scan axis
    const prefixElemShapes = resolvedElemShapes.map((s) => {
      const shape = [...s];
      shape.splice(axis, 1);
      return shape;
    });

    // inAxes: prefix leaves=null (point mode handles offset),
    //         block leaves=mapped on axis
    // Note: inAxes covers only non-const inputs (numInputs entries).
    // Constants are handled separately via constSlots/numConsts.
    const inAxes: (number | null)[][] = [
      ...prefixElemShapes.map(() => [null as number | null]),
      ...resolvedElemShapes.map(() => [axis as number | null]),
    ];
    const outAxes: (number | null)[][] = resolvedElemShapes.map(() => [axis]);

    // pointInputs: true for prefix leaves, false for block leaves
    // (covers only non-const inputs, same length as inAxes)
    const pointInputs = [
      ...prefixElemShapes.map(() => true),
      ...resolvedElemShapes.map(() => false),
    ];

    const inputShapes = [...prefixElemShapes, ...resolvedElemShapes];

    const phase4Params: ExecuteBlockMapParams = {
      backend,
      bodyProgram: applyVmapProgram,
      bodyJaxpr: plan.applyVmapJaxpr,
      blockShape: [B],
      inAxes,
      outAxes,
      numConsts,
      numInputs: numLeaves * 2, // prefix + block leaves
      gridShape: [M - 1],
      inputShapes,
      outputShapes: resolvedElemShapes,
      constSlots,
      inputSlots: [
        ...finalScannedSummarySlots, // prefix (point-mode)
        ...localScanSlots, // block (mapped with gridOffset)
      ],
      outputSlots,
      constInfos: plan.localScan.constInfos,
      pointInputs,
      gridOffset: 1,
      needsLeafPacking: plan.needsLeafPacking,
    };

    const phase4Result = executeBlockMap(phase4Params);
    if (phase4Result.pending.length > 0) {
      flushPending(phase4Result.pending);
    }
  } else if (M > 1) {
    // axis > 0: fall back to per-block mapOverBlocks (non-contiguous data)
    const summaryShapes = resolvedElemShapes.map((s) => {
      const shape = [...s];
      shape[axis] = M;
      return shape;
    });

    mapOverBlocks(
      backend,
      applyVmapProgram,
      constSlots,
      [
        {
          slots: finalScannedSummarySlots,
          shapes: summaryShapes,
          dtypes,
          mode: "point",
          indexOffset: -1,
        },
        {
          slots: localScanSlots,
          shapes: resolvedElemShapes,
          dtypes,
          mode: "block",
        },
      ],
      outputSlots,
      resolvedElemShapes,
      dtypes,
      axis,
      B,
      N,
      1,
      M,
      numConsts,
      dimBindings,
    );
  }

  // Cleanup
  for (const s of localScanSlots) backend.decRef(s);
  for (const s of finalScannedSummarySlots) backend.decRef(s);

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolve the scan axis dimension N, handling symbolic dims. */
function resolveAxisN(
  shape: Dim[],
  axis: number,
  dimBindings?: ReadonlyMap<string, number>,
): number {
  const nDim = shape[axis];
  return isSymbolicDim(nDim)
    ? concreteDim(resolveShape([nDim], dimBindings!)[0], "assoc_scan N")
    : (nDim as number);
}

/** Resolve a possibly-symbolic shape to concrete numbers. */
function resolveLeafShape(
  shape: Dim[],
  dimBindings?: ReadonlyMap<string, number>,
): number[] {
  return hasSymbolicDims(shape)
    ? concreteShape(resolveShape(shape, dimBindings!))
    : (shape as number[]);
}

/** Product of all elements in a shape array. */
function shapeProduct(shape: number[]): number {
  let p = 1;
  for (let i = 0; i < shape.length; i++) p *= shape[i];
  return p;
}

/**
 * Copy data from `src` (row-major, scan at `axis`) to `dst` (row-major, scan at axis 0).
 * Equivalent to moveaxis(src, axis, 0).
 */
function transposeAxisToFront(
  backend: Backend,
  src: Slot,
  dst: Slot,
  shape: number[],
  axis: number,
  elemBytes: number,
): void {
  const N = shape[axis];
  let outerSize = 1;
  for (let i = 0; i < axis; i++) outerSize *= shape[i];
  let innerBytes = elemBytes;
  for (let i = axis + 1; i < shape.length; i++) innerBytes *= shape[i];
  const srcFiberStride = N * innerBytes;
  const dstElemStride = outerSize * innerBytes;
  for (let f = 0; f < outerSize; f++) {
    for (let n = 0; n < N; n++) {
      backend.copyBufferToBuffer(
        src,
        f * srcFiberStride + n * innerBytes,
        dst,
        n * dstElemStride + f * innerBytes,
        innerBytes,
      );
    }
  }
}

/**
 * Copy data from `src` (row-major, scan at axis 0) to `dst` (row-major, scan at `axis`).
 * Inverse of transposeAxisToFront.
 */
function transposeAxisFromFront(
  backend: Backend,
  src: Slot,
  dst: Slot,
  shape: number[],
  axis: number,
  elemBytes: number,
): void {
  const N = shape[axis];
  let outerSize = 1;
  for (let i = 0; i < axis; i++) outerSize *= shape[i];
  let innerBytes = elemBytes;
  for (let i = axis + 1; i < shape.length; i++) innerBytes *= shape[i];
  const dstFiberStride = N * innerBytes;
  const srcElemStride = outerSize * innerBytes;
  for (let f = 0; f < outerSize; f++) {
    for (let n = 0; n < N; n++) {
      backend.copyBufferToBuffer(
        src,
        n * srcElemStride + f * innerBytes,
        dst,
        f * dstFiberStride + n * innerBytes,
        innerBytes,
      );
    }
  }
}

/** Flush all pending GPU/WASM operations, batching dispatches when possible. */
function flushPending(pending: IPendingExecute[]): void {
  if (pending.length === 0) return;
  const backend = pending[0].backend;
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
 * Slice xs buffers at a given iteration index.
 * Returns new slots (views or copies) for each xs input.
 */
function sliceXsAtIteration(
  backend: Backend,
  xsSlots: Slot[],
  xsStrides: number[],
  xsAvals: ShapedArray[],
  iterIdx: number,
): Slot[] {
  const slices: Slot[] = [];
  for (let j = 0; j < xsSlots.length; j++) {
    const srcOffset = iterIdx * xsStrides[j];
    const sliceSize = xsStrides[j]; // stride = per-element bytes

    // Copy the xs slice into a new buffer. Prefer copyBufferToBuffer
    // (keeps data on-device, avoids readSync which needs OffscreenCanvas
    // on WebGPU). Fall back to readSync + malloc
    // for backends that don't implement copyBufferToBuffer (CPU).
    if (backend.copyBufferToBuffer) {
      const slot = backend.malloc(sliceSize);
      backend.copyBufferToBuffer(xsSlots[j], srcOffset, slot, 0, sliceSize);
      slices.push(slot);
    } else {
      const data = backend.readSync(xsSlots[j], srcOffset, sliceSize);
      const slot = backend.malloc(sliceSize, data);
      slices.push(slot);
    }
  }
  return slices;
}

/**
 * Copy a slice from src slot into dst slot at a given iteration offset.
 */
function copySliceToBuffer(
  backend: Backend,
  dst: Slot,
  src: Slot,
  iterIdx: number,
  strideBytes: number,
  sliceBytes: number,
): void {
  const dstOffset = iterIdx * strideBytes;
  backend.copyBufferToBuffer(src, 0, dst, dstOffset, sliceBytes);
}
