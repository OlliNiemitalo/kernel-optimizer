/**
 * @file Scan plan construction — determines the execution strategy for a scan.
 */

import {
  AluOp,
  byteWidth,
  detectScalarAssocOp,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import type { Backend, Executable } from "../backend";
import {
  getScanRoutineInfo,
  NativeAssocScanBlockedParams,
  NativeAssocScanParams,
  NativeScanGeneralParams,
  ScanRoutineInfo,
  WasmBackend,
} from "../backend/wasm";
import type {
  NativeScanMultiParams,
  NativeScanMultiStep,
  PreparedPreencodedMultiStep,
  PreparedPreencodedScan,
  WebGPUCarryOutputSource,
  WebGPUYOutputSource,
} from "../backend/webgpu";
import type { WebGPUBackend } from "../backend/webgpu";
import { Routine, Routines } from "../routine";
import {
  type Dim,
  hasSymbolicDims,
  isSymbolicDim,
  resolveShape,
} from "../shape";
import { type CostFeatures, evaluateTotalCost } from "../tuner";
import { DEBUG } from "../utils";
import type { ScanPath } from "../utils";
import { Primitive, ShapedArray } from "./core";
import { Jaxpr, JaxprEqn, Var } from "./jaxpr";
import { jitCompile, type JitId, type JitProgram, type JitStep } from "./jit";
import { vmapJaxpr } from "./vmap";

// ---------------------------------------------------------------------------
// ScanPlan: a discriminated union of execution strategies
// ---------------------------------------------------------------------------

export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | {
      path: "compiled-loop";
      executable: Executable;
      params?: NativeScanGeneralParams | NativeScanMultiParams;
      internalSizes?: number[];
    }
  | { path: "preencoded-routine"; preencodedParams: PreparedPreencodedScan }
  | {
      path: "preencoded-multi-step";
      prepared: PreparedPreencodedMultiStep;
    };

/**
 * Execution plan for `lax.associativeScan` (Kogge-Stone parallel prefix scan).
 *
 * - `compiled-loop-blocked`: Three-level blocked Kogge-Stone in a single WASM module.
 *   Reduces total work from O(N log N) to O(N log B) for large N.
 * - `webgpu-block-map`: Block-map–based three-level blocked Kogge-Stone on WebGPU.
 * - `fallback`: JS-driven Kogge-Stone loop calling body program per round.
 */
/**
 * Plan-time metadata for a block_map stage. Holds the compiled body and
 * axis mapping separately from runtime data (gridShape, slots, shapes).
 */
export interface BlockMapStage {
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  inAxes: (number | null)[][];
  outAxes: (number | null)[][];
  blockShape: number[];
  numConsts: number;
  numInputs: number;
  /** Per-constant info for uniform buffer migration. */
  constInfos?: { elemCount: number; dtype: DType; bytes: number }[];
}

export type AssocScanPlan =
  | {
      path: "compiled-loop-blocked";
      executable: Executable;
      params: NativeAssocScanBlockedParams;
    }
  | {
      /**
       * Block-map–based path: uses {@link Primitive.BlockMap} with a
       * {@link Primitive.WorkgroupAssociativeScan} body for the local scan
       * phase, then generic blocked-data-movement primitives for gather,
       * recursive summary scan, and apply.
       *
       * The plan is self-similar: the same plan object works at every
       * recursion level because all compiled programs are shape-independent.
       */
      path: "webgpu-block-map";
      blockSize: number;
      numLeaves: number;
      numConsts: number;
      /** Local scan: block_map with WorkgroupAssociativeScan body. */
      localScan: BlockMapStage;
      /** Per-element body Jaxpr for recursive summary scan dispatch. */
      scanBodyJaxpr: Jaxpr;
      /** Vmapped apply body: [consts, prefix, block[B,…]] → [result[B,…]]. */
      applyVmapProgram: JitProgram;
      /** Vmapped apply body Jaxpr (for fused block_map in Phase 4). */
      applyVmapJaxpr: Jaxpr;
      /**
       * Enable leaf packing — pack all non-const inputs into a single storage
       * buffer (and similarly for outputs) to stay within
       * maxStorageBuffersPerShaderStage.  When true, the block-map shader
       * codegen computes concrete offsets from shapes at compile time.
       */
      needsLeafPacking?: boolean;
    }
  | {
      /**
       * Decoupled Fallback single-dispatch prefix scan (WebGPU only).
       * O(N) work in one dispatch via inter-workgroup atomics with bounded
       * spin + work-stealing fallback (no FPG required).
       * Phase 1: scalar binary ops (add/mul/min/max) on f32.
       */
      path: "decoupled-fallback";
      op: AluOp;
      dtype: DType;
      blockSize: number;
    }
  | { path: "fallback" };

type ExecuteStep = Extract<JitStep, { type: "execute" }>;

// ---------------------------------------------------------------------------
// Path acceptance checking (for testing / debugging)
// ---------------------------------------------------------------------------

/**
 * Check if a chosen scan path satisfies the acceptPath constraint.
 * Returns an error message if the path is not allowed, or null if OK.
 *
 * Special case: an empty array `[]` always rejects, showing the chosen path.
 */
export function checkAcceptedPath(
  chosenPath: ScanPath,
  acceptPath: string | string[] | undefined,
  extraInfo?: string,
): string | null {
  if (!acceptPath) return null;

  const allowedPaths = Array.isArray(acceptPath) ? acceptPath : [acceptPath];
  const suffix = extraInfo ? ` (${extraInfo})` : "";

  if (allowedPaths.length === 0) {
    return `Scan path debug: chose "${chosenPath}"${suffix}`;
  }

  if (!allowedPaths.includes(chosenPath)) {
    return (
      `Scan acceptPath constraint not satisfied: ` +
      `got "${chosenPath}" but accepted paths are [${allowedPaths.map((p) => `"${p}"`).join(", ")}]${suffix}`
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Buffer size helpers (shared by backends in later phases)
// ---------------------------------------------------------------------------

/**
 * Extract buffer sizes and strides from body jaxpr for native scan codegen.
 * Shared by WebGPU and WASM native scan implementations.
 */
export function getScanBufferSizes(
  bodyJaxpr: Jaxpr,
  numConsts: number,
  numCarry: number,
  numX: number,
  dimBindings?: ReadonlyMap<string, number>,
) {
  const resolveSize = (a: { shape: readonly Dim[]; dtype: DType }) => {
    const shape =
      dimBindings && hasSymbolicDims(a.shape)
        ? resolveShape(a.shape, dimBindings)
        : (a.shape as number[]);
    return shape.reduce((acc, d) => acc * d, 1) * byteWidth(a.dtype);
  };
  const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry, numConsts + numCarry + numX)
    .map((v) => v.aval);
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);

  return {
    constSizes: constAvals.map(resolveSize),
    constDtypes: constAvals.map((a) => a.dtype),
    carrySizes: carryAvals.map(resolveSize),
    carryDtypes: carryAvals.map((a) => a.dtype),
    xsStrides: xAvals.map(resolveSize),
    xsDtypes: xAvals.map((a) => a.dtype),
    ysStrides: yAvals.map(resolveSize),
    ysDtypes: yAvals.map((a) => a.dtype),
  };
}

// ---------------------------------------------------------------------------
// Body step classification (shared structural analysis)
// ---------------------------------------------------------------------------

/** Classification of a scan body program's steps for planner use. */
export interface BodyStepClassification {
  /** Execute steps that produce Kernel outputs, with execute-step-relative index. */
  kernelSteps: { index: number; step: ExecuteStep }[];
  /** Execute steps that produce Routine outputs. */
  routineSteps: { index: number; step: ExecuteStep }[];
  /** Non-execute loop steps (scan / assoc_scan) in the body program. */
  loopSteps: { index: number; step: JitStep; kind: "scan" | "assoc_scan" }[];
  /** Internal dependency edges between execute steps. */
  internalDeps: { consumer: number; producer: number; slot: JitId }[];
  /** Carry output indices that are passthroughs (carry_out[i] === some carry_in[j]). */
  carryPassthroughs: number[];
  /** Map from output JitId → producing execute step index and output index. */
  outputToProducer: Map<JitId, { execIdx: number; outputIdx: number }>;
  /** All execute steps in body-program order. */
  executeSteps: ExecuteStep[];
  hasRoutines: boolean;
  hasLoops: boolean;
  hasInternalDeps: boolean;
  /** True when an internal dep has producer and consumer with different kernel sizes. */
  hasCrossGidxDeps: boolean;
}

/**
 * Classify a scan body program's steps for planner consumption.
 *
 * Extracts shared structural facts that both WASM and WebGPU planners need:
 * step type classification, output-to-producer mapping, internal dependencies,
 * carry passthroughs, and nested loop detection.
 */
export function classifyBodySteps(
  bodyProgram: JitProgram,
  numCarry: number,
  numConsts: number,
  _numX: number,
): BodyStepClassification {
  const executeSteps: ExecuteStep[] = [];
  const kernelSteps: BodyStepClassification["kernelSteps"] = [];
  const routineSteps: BodyStepClassification["routineSteps"] = [];
  const loopSteps: BodyStepClassification["loopSteps"] = [];

  let execIdx = 0;
  for (let i = 0; i < bodyProgram.steps.length; i++) {
    const step = bodyProgram.steps[i];
    if (step.type === "execute") {
      const es = step as ExecuteStep;
      executeSteps.push(es);
      if (es.source instanceof Kernel) {
        kernelSteps.push({ index: execIdx, step: es });
      } else if (es.source instanceof Routine) {
        routineSteps.push({ index: execIdx, step: es });
      }
      execIdx++;
    } else if (step.type === "scan") {
      loopSteps.push({ index: i, step, kind: "scan" });
    } else if (step.type === "assoc_scan") {
      loopSteps.push({ index: i, step, kind: "assoc_scan" });
    }
  }

  // Output-to-producer mapping
  const outputToProducer = new Map<
    JitId,
    { execIdx: number; outputIdx: number }
  >();
  for (let ei = 0; ei < executeSteps.length; ei++) {
    const step = executeSteps[ei];
    for (let oi = 0; oi < step.outputs.length; oi++) {
      outputToProducer.set(step.outputs[oi], { execIdx: ei, outputIdx: oi });
    }
  }

  // Internal dependencies
  const internalDeps: BodyStepClassification["internalDeps"] = [];
  let hasCrossGidxDeps = false;
  for (let ei = 0; ei < executeSteps.length; ei++) {
    const step = executeSteps[ei];
    for (const inputId of step.inputs) {
      const producer = outputToProducer.get(inputId);
      if (producer && producer.execIdx !== ei) {
        internalDeps.push({
          consumer: ei,
          producer: producer.execIdx,
          slot: inputId,
        });
        const prodStep = executeSteps[producer.execIdx];
        if (
          prodStep.source instanceof Kernel &&
          step.source instanceof Kernel &&
          prodStep.source.size !== step.source.size
        ) {
          hasCrossGidxDeps = true;
        }
      }
    }
  }

  // Carry passthroughs
  const carryPassthroughs: number[] = [];
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  for (let ci = 0; ci < numCarry; ci++) {
    if (carryInputIds.includes(carryOutIds[ci])) {
      carryPassthroughs.push(ci);
    }
  }

  return {
    kernelSteps,
    routineSteps,
    loopSteps,
    internalDeps,
    carryPassthroughs,
    outputToProducer,
    executeSteps,
    hasRoutines: routineSteps.length > 0,
    hasLoops: loopSteps.length > 0,
    hasInternalDeps: internalDeps.length > 0,
    hasCrossGidxDeps,
  };
}

// ---------------------------------------------------------------------------
// Decoupled Fallback detection
// ---------------------------------------------------------------------------

const DF_SUPPORTED_DTYPES = new Set<DType>([
  DType.Float32,
  // DType.Uint32 excluded: 30-bit descriptor packing silently truncates
  // values > 2^30-1.  Cumulative sums/products can exceed that range with no
  // compile-time guard, producing incorrect results.  Kogge-Stone handles u32
  // correctly at O(N log N).
  //
  // DType.Int32 excluded: same 30-bit descriptor truncation issue — signed
  // range ±2^29 (~537M) is insufficient for large cumulative sums.
  // Kogge-Stone block-map path handles i32 correctly (fixed in v0.8.4).
]);

/**
 * Detect if an assocScan body is a simple scalar binary op eligible for
 * the Decoupled Fallback single-dispatch path.
 *
 * Requirements:
 * - numLeaves == 1 (single leaf, not pytree)
 * - numConsts == 0 (no captured constants)
 * - Body has exactly 1 execute step producing a single-output scalar kernel
 * - Kernel expression is a supported binary op (Add/Mul/Min/Max) of the two inputs
 * - Dtype is f32
 */
function detectDecoupledFallbackOp(
  bodyProgram: JitProgram,
  numLeaves: number,
  numConsts: number,
): { op: AluOp; dtype: DType } | null {
  if (numLeaves !== 1 || numConsts !== 0) return null;

  // Find the single execute step
  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length !== 1) return null;

  const step = executeSteps[0];
  if (!(step.source instanceof Kernel)) return null;

  const kernel = step.source;

  // Use shared detection for Add/Mul (also used by subgroupInclusiveAdd path).
  // DF also supports Min/Max, so check those separately.
  const assocOp = detectScalarAssocOp(kernel);
  if (assocOp != null) {
    if (!DF_SUPPORTED_DTYPES.has(kernel.outputs[0].dtype)) return null;
    return { op: assocOp, dtype: kernel.outputs[0].dtype };
  }

  // Min/Max: not in detectScalarAssocOp (only Add/Mul have subgroup builtins)
  if (kernel.numOutputs !== 1 || kernel.size !== 1) return null;
  if (kernel.outputs[0].reduction) return null;
  const exp = kernel.outputs[0].exp;
  if (exp.op !== AluOp.Min && exp.op !== AluOp.Max) return null;
  if (exp.src.length !== 2) return null;
  if (!DF_SUPPORTED_DTYPES.has(exp.dtype)) return null;
  if (
    exp.src[0].op !== AluOp.GlobalIndex ||
    exp.src[1].op !== AluOp.GlobalIndex
  )
    return null;
  const g0 = exp.src[0].arg[0] as number;
  const g1 = exp.src[1].arg[0] as number;
  if (!((g0 === 0 && g1 === 1) || (g0 === 1 && g1 === 0))) return null;
  return { op: exp.op, dtype: exp.dtype };
}

// ---------------------------------------------------------------------------
// planScan: decide which execution strategy to use
// ---------------------------------------------------------------------------

/**
 * Try to prepare a WASM native scan executable.
 *
 * Builds GeneralScanStep[] from the body program's execute steps, maps
 * slot IDs to internal buffer indices, determines CarryOutputSource and
 * YOutputSource for each output, and calls backend.prepareNativeScanGeneral().
 *
 * Returns null if the body can't be compiled (e.g. unsupported step types,
 * unmapped slots).
 */
function tryPrepareWasmNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  classification: BodyStepClassification,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  dimBindings?: ReadonlyMap<string, number>,
): {
  executable: Executable;
  internalSizes: number[];
  params?: NativeScanGeneralParams;
} | null {
  if (DEBUG >= 2) {
    console.log(
      `[wasm-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${classification.executeSteps.length}`,
    );
  }

  const executeSteps = classification.executeSteps;

  // Check for unsupported routines via the classification's routine steps.
  for (const { step } of classification.routineSteps) {
    if (!getScanRoutineInfo(step.source as Routine)) {
      if (DEBUG >= 1) {
        console.log(
          `[wasm-scan] skipped, unsupported routine: ${(step.source as Routine).name}`,
        );
      }
      return null;
    }
  }

  const numInputs = numConsts + numCarry + numX;

  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
    dimBindings,
  );

  // Build mapping from JitId (output slot) to internal buffer index
  // Multi-output routines need multiple internal buffers
  const slotToInternal = new Map<JitId, number>();
  const stepToInternalBase = new Map<number, number>();
  const internalSizes: number[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;
    stepToInternalBase.set(i, internalSizes.length);

    if (source instanceof Kernel) {
      const internalIdx = internalSizes.length;
      slotToInternal.set(step.outputs[0], internalIdx);
      internalSizes.push(
        (source.size as number) * byteWidth(source.outputs[0].dtype),
      );
    } else if (source instanceof Routine) {
      // Routine: may have multiple outputs
      for (let outIdx = 0; outIdx < step.outputs.length; outIdx++) {
        const internalIdx = internalSizes.length;
        slotToInternal.set(step.outputs[outIdx], internalIdx);
        const outShape = source.type.outputShapes[outIdx];
        const outDtype = source.type.outputDtypes[outIdx];
        internalSizes.push(
          outShape.reduce((a, b) => a * b, 1) * byteWidth(outDtype),
        );
      }
    }
  }

  // Calculate aux buffer size for routines that need it
  let auxBufferSize = 0;
  let elementSize: 4 | 8 = 4;
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      const routine = step.source;
      const dtype = routine.type.inputDtypes[0];
      elementSize = byteWidth(dtype) as 4 | 8;
      if (routine.name === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
      } else if (routine.name === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * 4);
      }
    }
  }

  // Build routineInfos array for WASM imports (size-specialized)
  const routineInfos: ScanRoutineInfo[] = [];
  const stepToRoutineInfoIdx = new Map<number, number>();

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    if (step.source instanceof Routine) {
      const routine = step.source;
      const info = getScanRoutineInfo(routine);
      if (!info) {
        throw new Error(
          `[wasm-scan] Unexpected unsupported routine: ${routine.name}`,
        );
      }

      const routineInfoIdx = routineInfos.length;
      stepToRoutineInfoIdx.set(i, routineInfoIdx);
      routineInfos.push(info);
    }
  }

  // Build steps with reindexed inputs
  type LocalStep = import("../backend/wasm").GeneralScanStep;
  const steps: LocalStep[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;

    // Map each input JitId to either a jaxpr input index or an internal buffer
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputSlots.push(inputId);
      } else {
        const internalIdx = slotToInternal.get(inputId);
        if (internalIdx === undefined) {
          if (DEBUG >= 1)
            console.log(
              `[wasm-scan] skipped, input ${inputId} not found in slot mapping`,
            );
          return null;
        }
        inputSlots.push(numInputs + internalIdx);
      }
    }

    if (source instanceof Kernel) {
      // Reindex kernel expressions to use our inputSlots mapping
      const reindexMap = inputSlots;
      const reindexedExp = source.outputs[0].exp.reindexGids(reindexMap);
      const reindexedReduction = source.outputs[0].reduction
        ? new Reduction(
            source.outputs[0].reduction.dtype,
            source.outputs[0].reduction.op,
            source.outputs[0].reduction.size,
            source.outputs[0].reduction.epilogue.reindexGids(reindexMap),
          )
        : undefined;
      const reindexedKernel = Kernel.single(
        numInputs + internalSizes.length,
        source.size,
        reindexedExp,
        reindexedReduction,
      );

      const internalBase = stepToInternalBase.get(i)!;
      steps.push({
        source: reindexedKernel,
        inputSlots,
        outputInternalIdx: internalBase,
      });
    } else if (source instanceof Routine) {
      // Routine step: build routineCallInfo with static params
      const routine = source;
      const routineName = routine.name as Routines;
      const routineInfoIdx = stepToRoutineInfoIdx.get(i)!;
      const internalBase = stepToInternalBase.get(i)!;
      const numOutputs = routine.type.outputShapes.length;
      const outputInternalIndices: number[] = [];
      for (let outIdx = 0; outIdx < numOutputs; outIdx++) {
        outputInternalIndices.push(internalBase + outIdx);
      }

      // Build static params based on routine type
      let staticParams: number[] = [];
      if (routineName === Routines.Cholesky) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      } else if (routineName === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      } else if (routineName === Routines.TriangularSolve) {
        const aShape = routine.type.inputShapes[0];
        const bShape = routine.type.inputShapes[1];
        const n = aShape[aShape.length - 1];
        const batchRows = bShape[bShape.length - 1];
        const numBatches = 1;
        const unitDiagonal = routine.params?.unitDiagonal ? 1 : 0;
        const lower = 0;
        staticParams = [n, batchRows, numBatches, unitDiagonal, lower];
      } else if (routineName === Routines.LU) {
        const inputShape = routine.type.inputShapes[0];
        const m = inputShape[inputShape.length - 2];
        const n = inputShape[inputShape.length - 1];
        staticParams = [m, n];
      } else if (routineName === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      } else if (routineName === Routines.QR) {
        const inputShape = routine.type.inputShapes[0];
        const m = inputShape[inputShape.length - 2];
        const n = inputShape[inputShape.length - 1];
        staticParams = [m, n];
      }

      steps.push({
        source,
        inputSlots,
        outputInternalIdx: internalBase,
        outputInternalIndices,
        routineCallInfo: {
          routineInfoIdx,
          staticParams,
        },
      });
    }
  }

  // Determine carry output sources
  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  type LocalCarrySource = import("../backend/wasm").CarryOutputSource;
  const carryOutSources: LocalCarrySource[] = [];
  for (const slot of carryOutSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      carryOutSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(
          `[wasm-scan] skipped, carry output slot ${slot} not produced by any execute step`,
        );
      return null;
    }
    carryOutSources.push({ type: "internal", internalIdx });
  }

  // Determine Y output sources
  const xsInputSlots = bodyProgram.inputs.slice(
    numConsts + numCarry,
    numConsts + numCarry + numX,
  );
  const yOutputSlots = bodyProgram.outputs.slice(numCarry);

  type LocalYSource = import("../backend/wasm").YOutputSource;
  const yOutputSources: LocalYSource[] = [];
  for (const slot of yOutputSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      yOutputSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const xsIdx = xsInputSlots.indexOf(slot);
    if (xsIdx !== -1) {
      yOutputSources.push({ type: "xs-passthrough", xsIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(`[wasm-scan] skipped, Y output slot ${slot} not found`);
      return null;
    }
    yOutputSources.push({ type: "internal", internalIdx });
  }

  // Build params and prepare the native scan
  const wasmBackend = backend as WasmBackend;
  if (!wasmBackend.prepareNativeScanGeneral) {
    if (DEBUG >= 2)
      console.log("[wasm-scan] backend has no prepareNativeScanGeneral");
    return null;
  }

  const params: NativeScanGeneralParams = {
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
    auxBufferSize: auxBufferSize > 0 ? auxBufferSize : undefined,
    elementSize: auxBufferSize > 0 ? elementSize : undefined,
    routineInfos: routineInfos.length > 0 ? routineInfos : undefined,
  };

  try {
    const exe = wasmBackend.prepareNativeScanGeneral(params);
    if (exe) {
      if (DEBUG >= 1) {
        const hasRoutines = steps.some((s) => s.source instanceof Routine);
        console.log(
          `[wasm-scan] SUCCESS! Using WASM native scan with ${steps.length} steps` +
            (hasRoutines ? " (includes routines)" : ""),
        );
      }
      return { executable: exe, internalSizes, params };
    }
    return null;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[wasm-scan] preparation failed:", e);
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// WebGPU multi-kernel native scan (P3)
// ---------------------------------------------------------------------------

function classifyWebGPUCarrySources(
  bodyProgram: JitProgram,
  numCarry: number,
  numConsts: number,
  numX: number,
  jitIdToInternalIdx: ReadonlyMap<JitId, number>,
): WebGPUCarryOutputSource[] | null {
  const xsStart = numConsts + numCarry;
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );
  const xsInputIds = bodyProgram.inputs.slice(xsStart, xsStart + numX);
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  const sources: WebGPUCarryOutputSource[] = [];

  for (let ci = 0; ci < numCarry; ci++) {
    const outId = carryOutIds[ci];

    // 1. carry-passthrough: output JitId matches a carry input
    const passIdx = carryInputIds.indexOf(outId);
    if (passIdx !== -1) {
      sources.push({ type: "carry-passthrough", carryIdx: passIdx });
      continue;
    }

    // 2. internal: produced by a body step
    const intIdx = jitIdToInternalIdx.get(outId);
    if (intIdx !== undefined) {
      sources.push({ type: "internal", internalIdx: intIdx });
      continue;
    }

    // 3. const: output is a constant
    if (outId < numConsts) {
      sources.push({ type: "const", constIdx: outId });
      continue;
    }

    // 4. xs: output is an xs element
    const xIdx = xsInputIds.indexOf(outId);
    if (xIdx !== -1) {
      // INVARIANT: If the carry output is precisely an `xs` element,
      // Jaxpr identity semantics guarantee its shape, dtype, and byte
      // size are identical to the `xs` slice's properties. Because of
      // this, it's safe for the executor to use `iterIdx * carrySizes[ci]`
      // as the source byte offset into `xsBuffers[xIdx]`.
      sources.push({ type: "xs", xIdx });
      continue;
    }

    // Unrecognized source type — fail gracefully so the planner falls back
    return null;
  }
  return sources;
}

/**
 * Classify every Y output into one of four source categories.
 * Shared by all WebGPU scan paths (compiled-loop, preencoded-routine, preencoded-multi-step).
 *
 * Returns null if any Y output cannot be resolved (caller should reject/fall back).
 */
function classifyWebGPUYSources(
  bodyProgram: JitProgram,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  jitIdToCarryIdx: ReadonlyMap<JitId, number>,
  jitIdToInternalIdx?: ReadonlyMap<JitId, number>,
): WebGPUYOutputSource[] | null {
  if (numY === 0) return [];

  const yOutIds = bodyProgram.outputs.slice(numCarry, numCarry + numY);
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );
  const xsInputIds = bodyProgram.inputs.slice(
    numConsts + numCarry,
    numConsts + numCarry + numX,
  );

  const sources: WebGPUYOutputSource[] = [];
  for (let yi = 0; yi < numY; yi++) {
    const yId = yOutIds[yi];

    // 1. carry-snapshot: Y reads the pre-update carry value (body input)
    const snapIdx = carryInputIds.indexOf(yId);
    if (snapIdx !== -1) {
      sources.push({ yi, type: "carry-snapshot", carryIdx: snapIdx });
      continue;
    }

    // 2. carry-live: Y reads the updated carry output (produced by a step)
    const liveIdx = jitIdToCarryIdx.get(yId);
    if (liveIdx !== undefined) {
      sources.push({ yi, type: "carry-live", carryIdx: liveIdx });
      continue;
    }

    // 3. xs: Y passes through an xs input unchanged
    const xsIdx = xsInputIds.indexOf(yId);
    if (xsIdx !== -1) {
      sources.push({ yi, type: "xs", xsIdx });
      continue;
    }

    // 4. internal: Y reads a step-produced intermediate that isn't a carry
    if (jitIdToInternalIdx) {
      const intIdx = jitIdToInternalIdx.get(yId);
      if (intIdx !== undefined) {
        sources.push({ yi, type: "internal", internalIdx: intIdx });
        continue;
      }
    }

    // Unresolvable
    return null;
  }
  return sources;
}

/**
 * Try to prepare a WebGPU native scan.
 *
 * Uses carry snapshot locals to avoid read-after-write hazards and
 * local variables for internal intermediates — no extra storage bindings.
 *
 * Constraints:
 * - No routine steps (only kernel steps)
 * - Total storage bindings ≤ maxStorageBuffersPerShaderStage
 * - Y outputs must be produced by body steps or be carry passthroughs
 */
function tryPrepareWebGPUNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  classification: BodyStepClassification,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  dimBindings?: ReadonlyMap<string, number>,
): {
  executable: Executable;
  params: NativeScanMultiParams;
} | null {
  // No routine steps
  if (classification.hasRoutines) {
    if (DEBUG >= 1) console.log(`[webgpu-scan] skipped, routine in scan body`);
    return null;
  }

  const executeSteps = classification.executeSteps;
  const numInputs = numConsts + numCarry + numX;
  const {
    constSizes,
    constDtypes,
    carrySizes,
    carryDtypes,
    xsStrides,
    xsDtypes,
    ysStrides,
    ysDtypes,
  } = getScanBufferSizes(bodyJaxpr, numConsts, numCarry, numX, dimBindings);

  // Check binding count: consts + xs + carry(rw) + ys(rw) ≤ limit
  const totalBindings = numConsts + numX + numCarry + numY;
  const maxBindings =
    (backend as WebGPUBackend).device?.limits
      ?.maxStorageBuffersPerShaderStage ?? 8;
  if (totalBindings > maxBindings) {
    if (DEBUG >= 1)
      console.log(
        `[webgpu-scan] skipped, ${totalBindings} bindings > limit ${maxBindings}`,
      );
    return null;
  }

  // Map ALL step output JitIds to their producing step and output index.
  // Multi-output kernels may produce multiple outputs from one step.
  const outputToStepInfo = new Map<
    JitId,
    { step: ExecuteStep; outputIdx: number }
  >();
  for (const step of executeSteps) {
    for (let oi = 0; oi < step.outputs.length; oi++) {
      outputToStepInfo.set(step.outputs[oi], { step, outputIdx: oi });
    }
  }

  // Identify carry outputs and carry passthroughs
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  // Set of JitIds that are carry outputs (produced by steps, not passthroughs)
  const carryOutputJitIds = new Set<JitId>();
  for (let ci = 0; ci < numCarry; ci++) {
    const outId = carryOutIds[ci];
    if (!carryInputIds.includes(outId)) {
      // Not a passthrough — must be produced by a step
      if (!outputToStepInfo.has(outId)) {
        if (DEBUG >= 1)
          console.log(
            `[webgpu-scan] skipped, carry ${ci} not produced by execute step`,
          );
        return null;
      }
      carryOutputJitIds.add(outId);
    }
    // Passthrough carries are fine — carry buffer retains its value
  }

  // Map carry output JitIds → carry index
  const jitIdToCarryIdx = new Map<JitId, number>();
  for (let ci = 0; ci < numCarry; ci++) {
    if (carryOutputJitIds.has(carryOutIds[ci])) {
      jitIdToCarryIdx.set(carryOutIds[ci], ci);
    }
  }

  // Assign internal indices to step outputs that aren't carry outputs.
  // Internal intermediates become var<private> arrays in WGSL (no storage bindings).
  let nextInternalIdx = 0;
  const jitIdToInternalIdx = new Map<JitId, number>();
  const internalElemCounts: number[] = [];
  const internalDtypes: DType[] = [];
  for (const step of executeSteps) {
    const source = step.source as Kernel;
    for (let oi = 0; oi < step.outputs.length; oi++) {
      const outId = step.outputs[oi];
      if (!carryOutputJitIds.has(outId)) {
        const idx = nextInternalIdx++;
        jitIdToInternalIdx.set(outId, idx);
        const kOut = source.outputs[oi] ?? source.outputs[0];
        internalElemCounts[idx] = source.size as number;
        internalDtypes[idx] = kOut.dtype;
      }
    }
  }
  const numInternal = nextInternalIdx;

  // Classify Y outputs using the shared vocabulary.
  const allYSources = classifyWebGPUYSources(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
    numY,
    jitIdToCarryIdx,
    jitIdToInternalIdx,
  );
  if (!allYSources) {
    if (DEBUG >= 1)
      console.log(`[webgpu-scan] skipped, Y source not resolvable`);
    return null;
  }

  // Compiled-loop splits Y handling:
  // - Passthroughs (carry-snapshot, xs) → writeback phase via descriptors
  // - Computed (carry-live, internal) → inline step writes via outputYIdxs
  const yOutputSources = allYSources.filter(
    (s): s is WebGPUYOutputSource & { type: "carry-snapshot" | "xs" } =>
      s.type === "carry-snapshot" || s.type === "xs",
  );

  // Explicitly calculate element counts from output avals (break layout coupling)
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const yElemCounts = yAvals.map((a) => {
    const shape =
      dimBindings && hasSymbolicDims(a.shape)
        ? resolveShape(a.shape, dimBindings)
        : (a.shape as number[]);
    return shape.reduce((acc, d) => acc * d, 1);
  });

  // Map from output JitId to Y indices (for carry-live and internal inline writes)
  const yOutIds =
    numY > 0 ? bodyProgram.outputs.slice(numCarry, numCarry + numY) : [];
  const jitIdToYIdx = new Map<JitId, number[]>();
  for (const src of allYSources) {
    if (src.type === "carry-live" || src.type === "internal") {
      const yId = yOutIds[src.yi];
      if (!jitIdToYIdx.has(yId)) jitIdToYIdx.set(yId, []);
      jitIdToYIdx.get(yId)!.push(src.yi);
    }
  }

  // Budget check: reject if total private-memory internal arrays exceed 8KB.
  // var<private> has no spec limit but excessive register pressure hurts occupancy.
  const totalInternalBytes = internalElemCounts.reduce(
    (sum, count, i) => sum + count * byteWidth(internalDtypes[i]),
    0,
  );
  if (totalInternalBytes > 8192) {
    if (DEBUG >= 1)
      console.log(
        `[webgpu-scan] skipped compiled-loop, internal arrays ${totalInternalBytes}B > 8KB budget`,
      );
    return null;
  }

  // Build scan gid mapping: body JitId → scan gid space.
  // Gid space: [0..numConsts) const, [numConsts..+numCarry) carry (snapshot),
  // [+numCarry..+numX) xs, [+numX..+numInternal) internal,
  // [+numInternal..+numCarry) carry-live (updated value from this iteration)
  const jitIdToScanGid = new Map<JitId, number>();
  // Body inputs map directly (carry inputs → snapshot reads)
  for (let i = 0; i < numInputs; i++) {
    jitIdToScanGid.set(bodyProgram.inputs[i], i);
  }
  // Internal intermediates
  for (const [jitId, internalIdx] of jitIdToInternalIdx) {
    jitIdToScanGid.set(jitId, numInputs + internalIdx);
  }
  // Carry output JitIds → "carry-live" gids (reads the UPDATED carry buffer,
  // not the snapshot). Y-only steps depend on the carry value computed in
  // this iteration, so they must read from carry{ci} not c_{ci}.
  const carryLiveGidBase = numInputs + numInternal;
  for (const [jitId, ci] of jitIdToCarryIdx) {
    jitIdToScanGid.set(jitId, carryLiveGidBase + ci);
  }

  // Build NativeScanMultiStep[] from ALL step outputs, in dependency order
  const multiSteps: NativeScanMultiStep[] = [];

  for (const step of executeSteps) {
    const source = step.source as Kernel;

    for (let oi = 0; oi < step.outputs.length; oi++) {
      const outId = step.outputs[oi];
      const carryIdx = jitIdToCarryIdx.get(outId) ?? -1;
      const internalIdx = jitIdToInternalIdx.get(outId) ?? -1;

      // Build reindex map: local kernel arg → scan gid
      const scanReindexMap = step.inputs.map(
        (jitId) => jitIdToScanGid.get(jitId) ?? jitId,
      );

      const kernelOutput = source.outputs[oi] ?? source.outputs[0];

      const reindexedExp = kernelOutput.exp.reindexGids(scanReindexMap);
      const reindexedReduction = kernelOutput.reduction
        ? new Reduction(
            kernelOutput.reduction.dtype,
            kernelOutput.reduction.op,
            kernelOutput.reduction.size,
            kernelOutput.reduction.epilogue.reindexGids(scanReindexMap),
          )
        : undefined;
      const reindexedKernel = Kernel.single(
        numInputs + numInternal,
        source.size,
        reindexedExp,
        reindexedReduction,
      );

      const yIdxs = jitIdToYIdx.get(outId);

      multiSteps.push({
        kernel: reindexedKernel,
        inputs: scanReindexMap.slice(),
        outputCarryIdx: carryIdx,
        outputInternalIdx: internalIdx,
        outputYIdxs: yIdxs,
        outputSize: source.size as number,
      });
    }
  }

  const params: NativeScanMultiParams = {
    length,
    numConsts,
    constSizes,
    constDtypes,
    numCarry,
    carrySizes,
    carryDtypes,
    numX,
    xsStrides,
    xsDtypes,
    numY,
    ysStrides,
    ysDtypes,
    yElemCounts,
    steps: multiSteps,
    reverse,
    yOutputSources,
    numInternal,
    internalElemCounts,
    internalDtypes,
  };

  // Call backend
  const webgpuBackend = backend as WebGPUBackend;
  const exe = webgpuBackend.prepareNativeScanMulti(params);
  if (!exe) return null;

  if (DEBUG >= 1) {
    const ySrcTypes = allYSources.map((s) => s.type).join(", ");
    console.log(
      `[webgpu-scan] SUCCESS! Using WebGPU native scan with ${multiSteps.length} steps` +
        (numInternal > 0 ? ` (${numInternal} internal locals)` : "") +
        (carryOutputJitIds.size < numCarry
          ? ` (${numCarry - carryOutputJitIds.size} passthrough carries)`
          : "") +
        (numY > 0 ? ` (Y sources: ${ySrcTypes})` : ""),
    );
  }
  return { executable: exe, params };
}

/**
 * Try to prepare a native scan executable (any backend).
 */
function tryPrepareNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number | Dim,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  dimBindings?: ReadonlyMap<string, number>,
): {
  executable: Executable;
  internalSizes?: number[];
  params?: NativeScanGeneralParams | NativeScanMultiParams;
} | null {
  const classification = classifyBodySteps(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
  );
  if (classification.executeSteps.length === 0) {
    if (DEBUG >= 1) console.log("[compiled-loop] skipped, no execute steps");
    return null;
  }

  if (backend.type === "wasm") {
    return tryPrepareWasmNativeScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      classification,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
      dimBindings,
    );
  }

  if (backend.type === "webgpu") {
    if (isSymbolicDim(length)) {
      if (DEBUG >= 1)
        console.log(
          "[compiled-loop] skipped, symbolic length not supported on WebGPU yet",
        );
      return null;
    }
    return tryPrepareWebGPUNativeScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      classification,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
      dimBindings,
    );
  }

  if (DEBUG >= 1)
    console.log(
      `[compiled-loop] skipped, backend=${backend.type} not supported yet`,
    );
  return null;
}

// ---------------------------------------------------------------------------
// Preencoded-routine scan (P4: WebGPU routine bodies)
// ---------------------------------------------------------------------------

/**
 * Try to prepare a preencoded scan for routine bodies (matmul, cholesky, etc.).
 *
 * Requirements:
 * - WebGPU backend
 * - Exactly 1 execute step in body that is a Routine
 * - numCarry === numY (passthrough pattern)
 * - Routine shader must not already use uniforms (e.g. Sort excluded)
 */
function tryPreparePreencodedScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): PreparedPreencodedScan | null {
  if (backend.type !== "webgpu") {
    if (DEBUG >= 2)
      console.log("Preencoded scan: skipped, unsupported backend");
    return null;
  }

  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length !== 1) {
    if (DEBUG >= 2)
      console.log(
        `Preencoded scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`,
      );
    return null;
  }

  const execStep = executeSteps[0];
  if (!(execStep.source instanceof Routine)) {
    if (DEBUG >= 2) console.log("Preencoded scan: skipped, not a Routine");
    return null;
  }

  // Classify Y outputs. Preencoded-routine only supports carry-live (the executor
  // copies updated carry → ys after each step), with Y[i] mapping to carry[i].
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );
  const jitIdToCarryIdx = new Map<JitId, number>();
  for (let ci = 0; ci < numCarry; ci++) {
    const outId = carryOutIds[ci];
    if (!carryInputIds.includes(outId)) {
      jitIdToCarryIdx.set(outId, ci);
    }
  }

  const ySources = classifyWebGPUYSources(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
    numY,
    jitIdToCarryIdx,
  );
  if (!ySources) {
    if (DEBUG >= 2)
      console.log("Preencoded scan: skipped, Y source not resolvable");
    return null;
  }
  for (let i = 0; i < ySources.length; i++) {
    const src = ySources[i];
    if (src.type !== "carry-live" || src.carryIdx !== i) {
      if (DEBUG >= 2)
        console.log(
          `Preencoded scan: skipped, Y[${i}] is ${src.type} (need carry-live[${i}])`,
        );
      return null;
    }
  }

  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry)
    .map((v) => v.aval);

  const carrySizes = carryAvals.map((a) => a.size * byteWidth(a.dtype));
  const xsElemStrides = xAvals.map((a) => a.size);
  const ysElemStrides = carryAvals.map((a) => a.size);

  if (!backend.prepareRoutineSync) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no prepareRoutineSync",
      );
    return null;
  }

  let bodyRoutineExe;
  try {
    bodyRoutineExe = backend.prepareRoutineSync(execStep.source);
  } catch (e) {
    if (DEBUG >= 2)
      console.warn("Preencoded scan: prepareRoutineSync failed:", e);
    return null;
  }

  const webgpuBackend = backend as WebGPUBackend;
  if (!webgpuBackend.preparePreencodedScan) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no preparePreencodedScan",
      );
    return null;
  }

  const preencodedScanParams = {
    length,
    carrySizes,
    xsElemStrides,
    ysElemStrides,
    bodyRoutine: bodyRoutineExe,
    numCarry,
    numX,
    numY,
    numConsts,
    reverse,
    routineInputJitIds: execStep.inputs,
    routineOutputJitIds: execStep.outputs,
    ySources,
  };

  try {
    const prepared = webgpuBackend.preparePreencodedScan(preencodedScanParams);
    if (prepared && DEBUG >= 1) {
      console.log(
        `Preencoded scan: SUCCESS! Using WebGPU preencoded scan for ${execStep.source.name}`,
      );
    }
    return prepared;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("Preencoded scan preparation failed:", e);
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Preencoded multi-step scan (Phase 2: multi-kernel bodies)
// ---------------------------------------------------------------------------

/**
 * Try to prepare a preencoded multi-step scan for kernel-only bodies.
 *
 * Requirements:
 * - WebGPU backend
 * - Concrete length (no symbolic dims)
 * - All execute steps are Kernels (no Routines — Phase 3 adds routine support)
 * - No nested loop steps (scan/assoc_scan in body)
 * - No cross-gidx internal dependencies
 * - Each step's bindings fit within storage buffer limits
 * - No symbolic kernel sizes
 */
function tryPreparePreencodedMultiStep(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): PreparedPreencodedMultiStep | null {
  if (backend.type !== "webgpu") {
    if (DEBUG >= 2) console.log("Preencoded multi-step: not WebGPU");
    return null;
  }

  const classification = classifyBodySteps(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
  );

  if (classification.hasLoops) {
    if (DEBUG >= 2) console.log("Preencoded multi-step: skipped, has loops");
    return null;
  }

  // Cross-gidx deps are safe in preencoded-multi-step: each step dispatches
  // independently with its own grid size and internal buffers. The cross-gidx
  // restriction only applies to same-gidx fusion (compiled-loop Phase 1).

  const { executeSteps } = classification;
  if (executeSteps.length < 2) {
    // Single-step bodies are handled by compiled-loop or preencoded-routine
    if (DEBUG >= 2)
      console.log("Preencoded multi-step: skipped, < 2 execute steps");
    return null;
  }

  const webgpuBackend = backend as WebGPUBackend;
  const maxBindings = webgpuBackend.maxArgs + 1; // +1 for output

  // Collect malloc/recycle/free steps for pre-allocation
  const mallocSteps: { jitId: number; size: number }[] = [];
  const recycleSteps: { from: number; to: number }[] = [];
  const freeStepIds: number[] = [];

  for (const step of bodyProgram.steps) {
    if (step.type === "malloc") {
      if (typeof step.size !== "number") {
        if (DEBUG >= 2)
          console.log("Preencoded multi-step: skipped, symbolic malloc size");
        return null;
      }
      mallocSteps.push({ jitId: step.output, size: step.size });
    } else if (step.type === "recycle") {
      recycleSteps.push({ from: step.input, to: step.output });
    } else if (step.type === "free") {
      freeStepIds.push(step.input);
    } else if (step.type !== "execute" && step.type !== "incref") {
      if (DEBUG >= 2)
        console.log(
          `Preencoded multi-step: skipped, unsupported step type "${step.type}"`,
        );
      return null;
    }
  }

  // Build internal buffer map: JitId → internal index + size
  const internalMap = new Map<number, number>();
  const internalSizes: number[] = [];
  for (const { jitId, size } of mallocSteps) {
    internalMap.set(jitId, internalSizes.length);
    internalSizes.push(size);
  }
  // Recycle targets map to the same internal index as their source
  for (const { from, to } of recycleSteps) {
    const idx = internalMap.get(from);
    if (idx !== undefined) {
      internalMap.set(to, idx);
    }
  }

  // Check all steps have concrete sizes and no pre-existing uniforms
  for (const step of executeSteps) {
    // Only kernels can be symbolic — routines always have concrete sizes
    if (step.source instanceof Kernel && step.source.isSymbolic) {
      if (DEBUG >= 2)
        console.log("Preencoded multi-step: skipped, symbolic kernel size");
      return null;
    }

    // Count bindings: inputs + outputs must fit in storage limit
    const totalBindings = step.inputs.length + step.outputs.length;
    if (totalBindings > maxBindings) {
      if (DEBUG >= 2)
        console.log(
          `Preencoded multi-step: skipped, step needs ${totalBindings} bindings (max ${maxBindings})`,
        );
      return null;
    }
  }

  // Compute buffer sizes and strides
  const { carrySizes } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  // Element strides for xs offset computation
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry, numConsts + numCarry + numX)
    .map((v) => v.aval);
  const xsElemStrides = xAvals.map((a) => a.size);

  // Element strides for ys offset computation
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const ysElemStrides = yAvals.map((a) => a.size);
  const ysSizes = yAvals.map((a) => a.size * byteWidth(a.dtype));

  // Classify Y outputs using the shared vocabulary.
  // Preencoded-multi-step supports all source types (carry-snapshot, carry-live, xs, internal).
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );
  const jitIdToCarryIdx = new Map<JitId, number>();
  for (let ci = 0; ci < numCarry; ci++) {
    const outId = carryOutIds[ci];
    if (!carryInputIds.includes(outId)) {
      jitIdToCarryIdx.set(outId, ci);
    }
  }
  // Convert internalMap (JitId → buffer index) to JitId → internal index for classifier
  const jitIdToInternalIdx = new Map<JitId, number>();
  for (const [jitId, idx] of internalMap) {
    jitIdToInternalIdx.set(jitId, idx);
  }

  const carrySources = classifyWebGPUCarrySources(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
    jitIdToInternalIdx,
  );
  if (!carrySources) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded multi-step: skipped, carry source not resolvable",
      );
    return null;
  }

  const ySources = classifyWebGPUYSources(
    bodyProgram,
    numCarry,
    numConsts,
    numX,
    numY,
    jitIdToCarryIdx,
    jitIdToInternalIdx,
  );
  if (!ySources) {
    if (DEBUG >= 2)
      console.log("Preencoded multi-step: skipped, Y source not resolvable");
    return null;
  }

  if (!webgpuBackend.preparePreencodedMultiStepScan) {
    if (DEBUG >= 2)
      console.log("Preencoded multi-step: backend missing method");
    return null;
  }

  try {
    const prepared = webgpuBackend.preparePreencodedMultiStepScan({
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
      ysElemStrides,
      ysSizes,
      carrySources,
      ySources,
    });
    if (prepared && DEBUG >= 1) {
      const ySrcTypes = ySources.map((s) => s.type).join(", ");
      console.log(
        `Preencoded multi-step: SUCCESS! ${executeSteps.length} steps, ${length} iterations` +
          (numY > 0 ? ` (Y sources: ${ySrcTypes})` : ""),
      );
    }
    return prepared;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("Preencoded multi-step preparation failed:", e);
    }
    return null;
  }
}

/**
 * Choose a scan execution strategy.
 *
 * Priority: compiled-loop > preencoded-routine > preencoded-multi-step > fallback.
 */
export function planScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number | Dim,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  acceptPath?: ScanPath | ScanPath[],
  dimBindings?: ReadonlyMap<string, number>,
): ScanPlan {
  // Try compiled-loop (WASM native scan)
  const nativeScanResult = tryPrepareNativeScan(
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts,
    numX,
    numY,
    reverse,
    dimBindings,
  );

  if (nativeScanResult) {
    const pathError = checkAcceptedPath("compiled-loop", acceptPath);
    if (pathError) throw new Error(pathError);
    return {
      path: "compiled-loop",
      executable: nativeScanResult.executable,
      internalSizes: nativeScanResult.internalSizes,
      params: nativeScanResult.params,
    };
  }

  // P4: preencoded-routine for WebGPU routine bodies
  // WebGPU preencoded-routine requires concrete length (uniform offsets)
  if (typeof length === "number") {
    const preencodedResult = tryPreparePreencodedScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );

    if (preencodedResult) {
      const pathError = checkAcceptedPath("preencoded-routine", acceptPath);
      if (pathError) throw new Error(pathError);
      return {
        path: "preencoded-routine",
        preencodedParams: preencodedResult,
      };
    }

    // Phase 2: preencoded multi-step for WebGPU multi-kernel bodies
    const multiStepResult = tryPreparePreencodedMultiStep(
      backend,
      bodyProgram,
      bodyJaxpr,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );

    if (multiStepResult) {
      const pathError = checkAcceptedPath("preencoded-multi-step", acceptPath);
      if (pathError) throw new Error(pathError);
      return {
        path: "preencoded-multi-step",
        prepared: multiStepResult,
      };
    }
  }

  // Fallback: JS loop
  const dispatchCount = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ).length;
  const extraInfo = `${dispatchCount} dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`;

  const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
  if (pathError) throw new Error(pathError);

  return { path: "fallback", extraInfo };
}

// ---------------------------------------------------------------------------
// planAssociativeScan: decide execution strategy for associative scan (M7.2)
// ---------------------------------------------------------------------------

/**
 * Plan execution strategy for `lax.associativeScan`.
 *
 * Tries compiled-loop (WASM) first, falls back to JS Kogge-Stone.
 * The compiled WASM module takes N as a runtime parameter, enabling
 * polymorphic length: a single compilation can serve any input length.
 *
 * @param backend         Backend to compile for
 * @param bodyProgram     JIT-compiled body program (per-element, N-independent)
 * @param bodyJaxpr       Body jaxpr (per-element shapes, no scan axis)
 * @param numLeaves       Number of pytree leaves
 * @param numConsts       Number of constant inputs closed over by the body
 * @param reverse         Whether to reverse the scan direction
 * @param dimBindings     Dimension bindings from the enclosing jitCompile
 */
export function planAssociativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  numLeaves: number,
  numConsts: number,
  axis: number,
  reverse: boolean,
  dimBindings?: ReadonlyMap<string, number>,
): AssocScanPlan {
  // Only WASM and WebGPU backends support native assoc scan
  if (backend.type !== "wasm" && backend.type !== "webgpu") {
    if (DEBUG >= 1) {
      console.log(`[assoc-scan] skipping native: backend is ${backend.type}`);
    }
    return { path: "fallback" };
  }

  // Classify body steps (shared analysis)
  const classification = classifyBodySteps(
    bodyProgram,
    numLeaves,
    numConsts,
    numLeaves,
  );

  if (classification.executeSteps.length === 0) {
    if (DEBUG >= 1) {
      console.log("[assoc-scan] skipping compiled-loop: no execute steps");
    }
    return { path: "fallback" };
  }

  // Check that all steps are kernel-only (no routines for now)
  if (classification.hasRoutines) {
    if (DEBUG >= 1) {
      const rName = (classification.routineSteps[0].step.source as Routine)
        .name;
      console.log(
        `[assoc-scan] skipping compiled-loop: routine ${rName} in body`,
      );
    }
    return { path: "fallback" };
  }

  // Check for nested loop steps
  if (classification.hasLoops) {
    if (DEBUG >= 1) {
      console.log(
        `[assoc-scan] skipping compiled-loop: unsupported step type "${classification.loopSteps[0].kind}"`,
      );
    }
    return { path: "fallback" };
  }

  // Check for non-execute steps that would make compilation impossible
  for (const step of bodyProgram.steps) {
    if (
      step.type !== "execute" &&
      step.type !== "malloc" &&
      step.type !== "free" &&
      step.type !== "recycle" &&
      step.type !== "incref"
    ) {
      if (DEBUG >= 1) {
        console.log(
          `[assoc-scan] skipping compiled-loop: unsupported step type "${step.type}"`,
        );
      }
      return { path: "fallback" };
    }
  }

  const executeSteps = classification.executeSteps;

  const numInputs = numConsts + 2 * numLeaves;

  // Build slot-to-internal mapping
  const slotToInternal = new Map<JitId, number>();
  const stepToInternalBase = new Map<number, number>();
  const internalSizes: number[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;
    if (!(source instanceof Kernel)) continue; // already checked above
    const internalIdx = internalSizes.length;
    stepToInternalBase.set(i, internalIdx);
    // Map ALL outputs of multi-output kernels, not just the first
    for (let oi = 0; oi < source.numOutputs; oi++) {
      slotToInternal.set(step.outputs[oi], internalIdx + oi);
    }
    for (let oi = 0; oi < source.numOutputs; oi++) {
      internalSizes.push(
        (source.size as number) * byteWidth(source.outputs[oi].dtype),
      );
    }
  }

  // Determine leaf-to-internal mapping (which internal produced each output leaf)
  const leafToInternalIdx: number[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const outputId = bodyProgram.outputs[k];
    const internalIdx = slotToInternal.get(outputId);
    if (internalIdx === undefined) {
      // Output is a passthrough from input (not produced by a kernel).
      // This means fn(a,b) = a or fn(a,b) = b, which is unusual
      // but valid. Fall back for now.
      if (DEBUG >= 1) {
        console.log(
          `[assoc-scan] skipping compiled-loop: output leaf ${k} is passthrough`,
        );
      }
      return { path: "fallback" };
    }
    leafToInternalIdx.push(internalIdx);
  }

  // Build reindexed steps
  type LocalStep = import("../backend/wasm").GeneralScanStep;
  const steps: LocalStep[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source as Kernel;

    // Map each input JitId to a global input ID or internal buffer
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputSlots.push(inputId);
      } else {
        const intIdx = slotToInternal.get(inputId);
        if (intIdx === undefined) {
          if (DEBUG >= 1) {
            console.log(
              `[assoc-scan] skipping compiled-loop: unmapped slot ${inputId}`,
            );
          }
          return { path: "fallback" };
        }
        inputSlots.push(numInputs + intIdx);
      }
    }

    // Reindex kernel expression GlobalIndex IDs
    const reindexMap = inputSlots;
    const internalBase = stepToInternalBase.get(i)!;

    if (source.numOutputs > 1) {
      // Multi-output kernel: reindex all outputs
      const reindexedOutputs = source.outputs.map((out) => ({
        exp: out.exp.reindexGids(reindexMap),
        reduction: out.reduction
          ? new Reduction(
              out.reduction.dtype,
              out.reduction.op,
              out.reduction.size,
              out.reduction.epilogue.reindexGids(reindexMap),
            )
          : undefined,
        dtype: out.dtype,
      }));
      const reindexedKernel = Kernel.multi(
        numInputs + internalSizes.length,
        source.size,
        reindexedOutputs,
      );
      const indices: number[] = [];
      for (let oi = 0; oi < source.numOutputs; oi++)
        indices.push(internalBase + oi);
      steps.push({
        source: reindexedKernel,
        inputSlots,
        outputInternalIdx: internalBase,
        outputInternalIndices: indices,
      });
    } else {
      const reindexedExp = source.outputs[0].exp.reindexGids(reindexMap);
      const reindexedReduction = source.outputs[0].reduction
        ? new Reduction(
            source.outputs[0].reduction.dtype,
            source.outputs[0].reduction.op,
            source.outputs[0].reduction.size,
            source.outputs[0].reduction.epilogue.reindexGids(reindexMap),
          )
        : undefined;
      const reindexedKernel = Kernel.single(
        numInputs + internalSizes.length,
        source.size,
        reindexedExp,
        reindexedReduction,
      );
      steps.push({
        source: reindexedKernel,
        inputSlots,
        outputInternalIdx: internalBase,
      });
    }
  }

  // Build per-element leaf sizes (bytes) for WASM
  const leafElemSizes: number[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const aval = bodyJaxpr.inBinders[numConsts + k].aval;
    leafElemSizes.push(aval.size * byteWidth(aval.dtype));
  }

  // Build const sizes
  const constSizes: number[] = [];
  const constInfos: { elemCount: number; dtype: DType; bytes: number }[] = [];
  for (let k = 0; k < numConsts; k++) {
    const aval = bodyJaxpr.inBinders[k].aval;
    const bytes = aval.size * byteWidth(aval.dtype);
    constSizes.push(bytes);
    constInfos.push({ elemCount: aval.size, dtype: aval.dtype, bytes });
  }

  // --- WebGPU block-map path ---
  // Guard: storage buffer bindings must fit inside the per-stage limit.
  // Constants that fit in uniform buffers (≤64KB) are moved to @group(1),
  // freeing storage bindings.
  if (backend.type === "webgpu") {
    // --- Decoupled Fallback: single-dispatch O(N) scan for scalar binary ops ---
    if (axis === 0) {
      const dfOp = detectDecoupledFallbackOp(bodyProgram, numLeaves, numConsts);
      if (dfOp) {
        const BLOCK_SIZE = 256;
        if (DEBUG >= 1) {
          console.log(
            `[assoc-scan] SUCCESS! Using Decoupled Fallback path (${dfOp.op} ${dfOp.dtype} B=${BLOCK_SIZE})`,
          );
        }
        return {
          path: "decoupled-fallback",
          op: dfOp.op,
          dtype: dfOp.dtype,
          blockSize: BLOCK_SIZE,
        };
      }
    }

    const maxBindings = backend.maxArgs + 1; // maxStorageBuffersPerShaderStage
    const uniformConsts = constInfos.every((c) => c.bytes <= 65536)
      ? numConsts
      : 0;
    const neededBindings = numConsts - uniformConsts + 2 * numLeaves;

    // Leaf packing: when binding count exceeds the limit but all leaves share
    // the same dtype, pack all non-const inputs into one storage buffer and
    // all outputs into another.  Reduces binding count to uniformConsts + 2.
    let needsLeafPacking = false;
    if (neededBindings > maxBindings) {
      const packedBindings = numConsts - uniformConsts + 2;
      const leafDtypes = new Set<DType>();
      for (let i = 0; i < numLeaves; i++) {
        leafDtypes.add(bodyJaxpr.inBinders[numConsts + i].aval.dtype);
      }
      // Output dtypes must also match — packed output buffer has a single dtype.
      for (const out of bodyJaxpr.outs) {
        if (out instanceof Var) leafDtypes.add(out.aval.dtype);
      }
      if (leafDtypes.size === 1 && packedBindings <= maxBindings) {
        needsLeafPacking = true;
        if (DEBUG >= 1) {
          console.log(
            `[assoc-scan] leaf packing: ${neededBindings} bindings > ${maxBindings}, packed to ${packedBindings}`,
          );
        }
      } else {
        if (DEBUG >= 1) {
          console.log(
            `[assoc-scan] block-map rejected: ${neededBindings} bindings > device max ${maxBindings}` +
              (leafDtypes.size > 1
                ? " (mixed dtypes, packing not viable)"
                : ""),
          );
        }
        return { path: "fallback" };
      }
    }

    // Score the allowed block sizes based on Continuous Cost Modeling logic.
    // The fused shader's shmem usage scales linearly with blockSize.
    // Minimum B=32 keeps the Kogge-Stone round count ≤5.
    const candidates = [256, 128, 64, 32].map((b) => {
      // Rough estimation:
      const shmemUsage = b * neededBindings * 4;
      const features: CostFeatures = {
        nDispatch: 1,
        nBuffers: neededBindings,
        countAlu: b * neededBindings,
        countMem: b * neededBindings * 8, // read + write
        depthPriv: 16,
        sizeShmem: shmemUsage,
        sizeWgsl: 4096,
        parallelism: b,
        produceCount: b,
      };
      return { b, cost: evaluateTotalCost(features, backend.capabilities) };
    });

    candidates.sort((x, y) => x.cost - y.cost);

    for (const { b: BLOCK_SIZE } of candidates) {
      try {
        const blockMapPlan = tryBuildBlockMapAssocScanPlan(
          backend,
          bodyProgram,
          bodyJaxpr,
          numLeaves,
          numConsts,
          axis,
          reverse,
          BLOCK_SIZE,
          dimBindings,
          needsLeafPacking,
        );
        if (blockMapPlan) {
          if (DEBUG >= 1) {
            console.log(
              `[assoc-scan] SUCCESS! Using WebGPU block-map path (B=${BLOCK_SIZE}${needsLeafPacking ? ", leaf-packed" : ""})`,
            );
          }
          return blockMapPlan;
        }
      } catch (e) {
        if (DEBUG >= 2) {
          console.warn(
            `[assoc-scan] WebGPU block-map compilation failed (B=${BLOCK_SIZE}):`,
            e,
          );
        }
      }
    }
    return { path: "fallback" };
  }

  // --- WASM path: blocked compiled-loop ---
  const params: NativeAssocScanParams = {
    numConsts,
    constSizes,
    numLeaves,
    leafElemSizes,
    steps,
    internalSizes,
    reverse,
    leafToInternalIdx,
  };

  const wasmBackend = backend as WasmBackend;

  // Blocked Kogge-Stone: O(N log B) work instead of O(N log N).
  // Handles small N correctly (M=1 skips levels 2-3).
  const BLOCK_SIZE = 256;
  try {
    const blockedParams: NativeAssocScanBlockedParams = {
      ...params,
      blockSize: BLOCK_SIZE,
    };
    const exe = wasmBackend.prepareBlockedAssociativeScan(blockedParams);
    if (DEBUG >= 1) {
      console.log(
        `[assoc-scan] SUCCESS! Using WASM compiled-loop-blocked (B=${BLOCK_SIZE}) with ${steps.length} step(s)`,
      );
    }
    return {
      path: "compiled-loop-blocked",
      executable: exe,
      params: blockedParams,
    };
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[assoc-scan] blocked compilation failed:", e);
    }
    return { path: "fallback" };
  }
}

// ---------------------------------------------------------------------------
// Block-map–based associative scan plan builder
// ---------------------------------------------------------------------------

/**
 * Build a block-map–based plan for associative scan (WebGPU only).
 *
 * Constructs a body Jaxpr containing a single WorkgroupAssociativeScan
 * equation, then JIT-compiles it for use as a block_map body. The block-map
 * fused shader compiler in block-map.ts handles the Kogge-Stone codegen.
 *
 * Returns null if the block-map path is not viable.
 */
function tryBuildBlockMapAssocScanPlan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  numLeaves: number,
  numConsts: number,
  axis: number,
  reverse: boolean,
  blockSize: number,
  dimBindings?: ReadonlyMap<string, number>,
  needsLeafPacking?: boolean,
): AssocScanPlan | null {
  if (backend.type !== "webgpu") return null;

  // Build block-shaped element avals: insert blockSize at the scan axis.
  // Body always sees B at axis 0; inAxes/outAxes handle the mapping.
  const constAvals: ShapedArray[] = [];
  for (let i = 0; i < numConsts; i++) {
    constAvals.push(bodyJaxpr.inBinders[i].aval);
  }

  const elemAvals: ShapedArray[] = [];
  for (let i = 0; i < numLeaves; i++) {
    const perElem = bodyJaxpr.inBinders[numConsts + i].aval;
    const perShape = perElem.shape as number[];
    // Insert blockSize at position `axis` (where the scan axis was removed)
    const blockElemShape = [
      ...perShape.slice(0, axis),
      blockSize,
      ...perShape.slice(axis),
    ];
    elemAvals.push(
      new ShapedArray(blockElemShape, perElem.dtype, perElem.weakType),
    );
  }

  // Construct the body Jaxpr for the block_map:
  //   inputs: [const_0, ..., const_C, elem_0[B,...], ..., elem_L[B,...]]
  //   eqn:    [out_0,...,out_L] = WorkgroupAssociativeScan(all_inputs)
  //   outputs: [out_0, ..., out_L]
  const inBinders: Var[] = [];
  for (const a of constAvals) inBinders.push(new Var(a));
  for (const a of elemAvals) inBinders.push(new Var(a));

  const outBinders: Var[] = [];
  for (const a of elemAvals) outBinders.push(new Var(a));

  const wasEqn = new JaxprEqn(
    Primitive.WorkgroupAssociativeScan,
    inBinders,
    { jaxpr: bodyJaxpr, numConsts } as Record<string, any>,
    outBinders,
  );

  const localScanBodyJaxpr = new Jaxpr(inBinders, [wasEqn], outBinders);

  // JIT-compile the body. block-map.ts will see the workgroup_assoc_scan step
  // and generate inlined Kogge-Stone rounds in the fused shader.
  let localScanBodyProgram: JitProgram;
  try {
    localScanBodyProgram = jitCompile(
      backend,
      localScanBodyJaxpr,
      dimBindings,
      "workgroup_assoc_scan_body",
    );
  } catch (e) {
    if (DEBUG >= 1) {
      console.log("[assoc-scan] block-map path: body jitCompile failed:", e);
    }
    return null;
  }

  // Verify the body program contains a workgroup_assoc_scan step.
  const hasWAS = localScanBodyProgram.steps.some(
    (s) => s.type === "workgroup_assoc_scan",
  );
  if (!hasWAS) {
    if (DEBUG >= 1) {
      console.log(
        "[assoc-scan] block-map path: body has no workgroup_assoc_scan step",
      );
    }
    return null;
  }

  // --- Pre-estimate fused shader shmem to reject block sizes that overflow ---
  // This mirrors the accounting in blockMapFusedShaderSource: top-level mallocs
  // + WAS ping-pong + WAS body mallocs + reduction workspaces.
  const maxShmem = backend.capabilities.maxComputeWorkgroupStorageSize ?? 16384;
  let estimatedShmem = 0;
  for (const step of localScanBodyProgram.steps) {
    if (step.type === "malloc" && typeof step.size === "number") {
      estimatedShmem += step.size;
    } else if (step.type === "workgroup_assoc_scan") {
      // Ping-pong arrays: 2 × blockSize × elemCount × byteWidth per element
      for (const aval of step.elemAvals) {
        const elemCount = aval.size / (aval.shape[0] as number);
        estimatedShmem += blockSize * elemCount * byteWidth(aval.dtype) * 2;
      }
      // WAS body mallocs + reduction workspaces
      for (const bs of step.bodyProgram.steps) {
        if (bs.type === "malloc" && typeof bs.size === "number") {
          estimatedShmem += bs.size;
        } else if (bs.type === "execute" && bs.source instanceof Kernel) {
          for (const out of bs.source.outputs) {
            if (out.reduction) {
              estimatedShmem += blockSize * byteWidth(out.reduction.dtype);
            }
          }
        }
      }
    } else if (step.type === "execute" && step.source instanceof Kernel) {
      for (const out of step.source.outputs) {
        if (out.reduction) {
          estimatedShmem += blockSize * byteWidth(out.reduction.dtype);
        }
      }
    }
  }
  if (estimatedShmem > maxShmem) {
    if (DEBUG >= 1) {
      console.log(
        `[assoc-scan] block-map B=${blockSize}: shmem ~${estimatedShmem} > limit ${maxShmem}, trying smaller`,
      );
    }
    return null;
  }

  // Build vmapped apply body: processes a block of B elements at once.
  // Signature: (consts..., prefix_0, ..., prefix_L, block_0, ..., block_L)
  //         -> (result_0, ..., result_L)
  // where prefix leaves are broadcast (same for all B elements)
  // and block leaves are mapped on the scan axis.
  const applyVmapDims: (number | null)[] = [
    ...Array(numConsts).fill(null), // consts: broadcast
    ...Array(numLeaves).fill(null), // prefix: broadcast
    ...Array(numLeaves).fill(axis), // block elements: mapped on scan axis
  ];
  const applyVmapClosed = vmapJaxpr(bodyJaxpr, blockSize, applyVmapDims);

  // The vmapped body should be const-free when the source body is pure.
  // If it has captured consts, fall back — we don't propagate vmap const
  // slots through the plan.
  if (applyVmapClosed.constEnv.length > 0) {
    if (DEBUG >= 1) {
      console.log(
        "[assoc-scan] block-map path: vmapped apply body has unexpected consts",
      );
    }
    return null;
  }

  let applyVmapProgram: JitProgram;
  try {
    applyVmapProgram = jitCompile(
      backend,
      applyVmapClosed.jaxpr,
      dimBindings,
      "assoc_scan_apply_vmap",
    );
  } catch (e) {
    if (DEBUG >= 1) {
      console.log(
        "[assoc-scan] block-map path: applyVmap jitCompile failed:",
        e,
      );
    }
    return null;
  }

  // inAxes: per non-const input. Constants are broadcast (handled by
  // numConsts), so inAxes only covers the numLeaves element inputs.
  const inAxes: (number | null)[][] = elemAvals.map(() => [
    axis as number | null,
  ]);
  const outAxes: (number | null)[][] = elemAvals.map(() => [axis]);

  // Build constInfos for uniform buffer migration
  const constInfos: { elemCount: number; dtype: DType; bytes: number }[] = [];
  for (let i = 0; i < numConsts; i++) {
    const aval = bodyJaxpr.inBinders[i].aval;
    constInfos.push({
      elemCount: aval.size,
      dtype: aval.dtype,
      bytes: aval.size * byteWidth(aval.dtype),
    });
  }

  const localScan: BlockMapStage = {
    bodyProgram: localScanBodyProgram,
    bodyJaxpr: localScanBodyJaxpr,
    inAxes,
    outAxes,
    blockShape: [blockSize],
    numConsts,
    numInputs: numLeaves,
    constInfos,
  };

  return {
    path: "webgpu-block-map",
    localScan,
    scanBodyJaxpr: bodyJaxpr,
    blockSize,
    numLeaves,
    numConsts,
    applyVmapProgram,
    applyVmapJaxpr: applyVmapClosed.jaxpr,
    needsLeafPacking: needsLeafPacking || undefined,
  };
}
