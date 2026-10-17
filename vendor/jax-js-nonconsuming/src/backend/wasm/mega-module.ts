/**
 * Mega-Module compiler (M6.1): compiles an entire JitProgram's step list
 * into a single WASM function, eliminating JS↔WASM boundary crossings
 * between kernel dispatches.
 *
 * Architecture:
 * - Each JitId maps to a WASM local (i32 holding a memory pointer)
 * - Kernel bodies are inlined as gidx loops within the mega-function
 * - malloc/free are imported from the host WasmAllocator
 * - Routines are imported as separate WASM function calls
 * - incref is a no-op inside the module (handled at the JS Slot level)
 * - recycle is a zero-cost local rename
 *
 * The mega-module function signature:
 *   mega_execute(input0_ptr, ..., inputN_ptr, resultBufPtr) -> void
 *
 * Output pointers are written to the result buffer at known offsets.
 * The JS wrapper reads them and creates proper backend Slots.
 */

import { AluExp, AluOp, byteWidth, DType, Kernel } from "../../alu";
import { _emitCodeCapture, _isCodeCaptureEnabled } from "../../backend";
import type { JitId, JitStep } from "../../frontend/jit";
import { Routine } from "../../routine";
import { isSymbolicSize, type SizeExpr } from "../../shape";
import { tuneNullopt } from "../../tuner";
import { DEBUG, mapSetUnion, rep } from "../../utils";
import {
  canVectorizeSimd,
  codegenReductionAccumulate,
  configureMemoryImport,
  dty,
  importWasmHelperFuncs,
  translateExpCore,
  translateExpCoreSimd,
} from "../wasm";
import { CodeGenerator } from "./wasmblr";

/** Info about an extracted kernel function (M6.2a). */
export interface ExtractedKernelInfo {
  /** Export name (e.g., "kernel_0"). */
  readonly name: string;
  /** Number of elements this kernel processes. */
  readonly size: number;
  /** Whether this kernel has a reduction. */
  readonly isReduction: boolean;
  /** Number of input buffer params (after start, end). */
  readonly nInputs: number;
  /** Number of output buffer params. */
  readonly nOutputs: number;
}

/**
 * Step metadata for JS-driven parallel execution (M6.2c).
 * Describes one step in the mega-module's execution sequence so the
 * parallel dispatch path can replicate it without calling mega_execute.
 */
export type MegaStepInfo =
  | {
      type: "malloc";
      outputIdx: number;
      size: number;
      initialData?: Uint8Array;
    }
  | { type: "free"; inputIdx: number }
  | { type: "recycle"; fromIdx: number; toIdx: number }
  | {
      type: "kernel";
      kernelIdx: number;
      kernelSize: number;
      inputIdxs: number[];
      outputIdxs: number[];
    };

/** Result of compiling a JitProgram into a mega-module. */
export interface WasmMegaModule {
  /** The compiled WebAssembly module. */
  readonly module: WebAssembly.Module;
  /** Number of input parameters. */
  readonly numInputs: number;
  /** Number of output pointers written to the result buffer. */
  readonly numOutputs: number;
  /** Byte sizes of each output buffer (for creating Slots). */
  readonly outputSizes: number[];
  /**
   * Which step types are NOT supported (caused compilation to fail).
   * Empty if all steps were compiled.
   */
  readonly unsupportedSteps: string[];
  /**
   * Info about extracted kernel functions (M6.2a/c).
   * All kernels (elementwise and reduction) are extracted into separate
   * WASM functions with (start, end, ...bufs) signatures.
   */
  readonly kernelExports: ExtractedKernelInfo[];
  /**
   * Step metadata for JS-driven parallel execution (M6.2c).
   * JitId values are mapped to sequential indices (0..N). Input JitIds
   * use indices 0..numInputs-1. Other JitIds get indices >= numInputs.
   * The `stepInfos` array describes the full step sequence that
   * mega_execute would perform, enabling a JS-driven parallel path.
   */
  readonly stepInfos: MegaStepInfo[];
  /**
   * Total number of JitId slots needed for parallel execution.
   * This is the size of the locals array (maps JitId index → pointer).
   */
  readonly numLocals: number;
  /**
   * Mapping from output index (0..numOutputs-1) to the JitId local index
   * that holds that output pointer after execution completes.
   */
  readonly outputLocalIdxs: number[];
  /**
   * Whether the mega-module can be dispatched via the parallel stepInfos path.
   * False when the program contains steps (e.g., fori_loop) that are only
   * handled by mega_execute and have no parallel stepInfos representation.
   */
  readonly canParallelize: boolean;
}

/**
 * Check whether a JitProgram can be compiled to a mega-module.
 * Returns true if all steps are supported.
 */
export function canCompileToMegaModule(steps: JitStep[]): boolean {
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        // Symbolic malloc sizes require runtime resolution — not supported.
        if (isSymbolicSize(step.size)) return false;
        break;
      case "free":
      case "recycle":
        break;
      case "incref":
        // Incref requires refcount tracking inside WASM — not yet supported.
        // Without proper refcounting, a subsequent `free` would deallocate
        // memory that another local still references.
        return false;
      case "execute":
        // Support kernels (routines not yet supported)
        if (step.source instanceof Routine) return false;
        // Symbolic reduction sizes require runtime resolution via dynamicParams —
        // the mega-module inlines reduction loops with i32.const bounds, which
        // can't represent symbolic sizes. Reject and fall through to step-by-step.
        //
        // MIGRATION NOTE: To support symbolic reductions in mega-module, you would
        // need to: (1) accept a reduceSize i32 param in mega_execute's signature,
        // (2) use local.get instead of i32.const in emitReductionBody's loop bound,
        // (3) plumb dynamicParams through executeMegaModule. This mirrors the
        // approach used by codegenWasm() for regular kernels (see reduceSizeLocal).
        if (step.source instanceof Kernel && step.source.hasSymbolicReduction)
          return false;
        break;
      case "fori_loop": {
        // Accept fori_loop if bounds are concrete and body is mega-module-eligible.
        if (typeof step.lower !== "number" || typeof step.upper !== "number")
          return false;
        if (!canCompileToMegaModule(step.bodyProgram.steps)) return false;
        break;
      }
      case "scan":
      case "dus":
      case "scatter_add":
      case "reverse":
      case "assoc_scan":
      case "block_map":
      case "workgroup_assoc_scan":
        return false;
      default:
        return false;
    }
  }
  return true;
}

/**
 * Compile a JitProgram's step list into a single WASM mega-module.
 *
 * The compiled function takes input pointers as i32 params and a
 * resultBufPtr as the last param. Output pointers are written to
 * the result buffer.
 *
 * @param steps       JitProgram steps
 * @param inputIds    JitIds of program inputs
 * @param outputIds   JitIds of program outputs
 * @returns Compiled mega-module or null if unsupported steps are present
 */
export function compileToMegaModule(
  steps: JitStep[],
  inputIds: JitId[],
  outputIds: JitId[],
): WasmMegaModule | null {
  // Check all steps are supported
  if (!canCompileToMegaModule(steps)) return null;

  // Collect all JitIds referenced in the program
  const allJitIds = new Set<JitId>();
  for (const id of inputIds) allJitIds.add(id);
  for (const id of outputIds) allJitIds.add(id);
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        allJitIds.add(step.output);
        break;
      case "free":
        allJitIds.add(step.input);
        break;
      case "recycle":
        allJitIds.add(step.input);
        allJitIds.add(step.output);
        break;
      case "incref":
        allJitIds.add(step.input);
        break;
      case "execute":
        for (const id of step.inputs) allJitIds.add(id);
        for (const id of step.outputs) allJitIds.add(id);
        break;
      case "fori_loop":
        for (const id of step.consts) allJitIds.add(id);
        for (const id of step.initCarries) allJitIds.add(id);
        for (const id of step.outputs) allJitIds.add(id);
        // Body JitIds are handled separately (different namespace)
        break;
    }
  }

  // Track byte sizes of every buffer, including through recycle steps.
  // malloc creates a buffer; recycle transfers ownership (same size).
  const bufferSizes = new Map<JitId, SizeExpr>();
  for (const step of steps) {
    if (step.type === "malloc") {
      bufferSizes.set(step.output, step.size);
    } else if (step.type === "recycle") {
      const srcSize = bufferSizes.get(step.input);
      if (srcSize !== undefined) bufferSizes.set(step.output, srcSize);
    }
  }

  // Resolve output sizes (must be concrete for now)
  const inputIdSet = new Set(inputIds);
  const outputSizes: number[] = [];
  for (const id of outputIds) {
    const size = bufferSizes.get(id);
    if (size !== undefined) {
      if (isSymbolicSize(size)) return null; // Can't handle symbolic sizes yet
      outputSizes.push(size as number);
    } else if (inputIdSet.has(id)) {
      // Output is an input pass-through. Reject — pass-through programs are
      // trivial and don't benefit from mega-module compilation. The WASM
      // function's local for this JitId points to the input param, but other
      // steps (free, recycle) might overwrite it before the output pointer
      // is written to the result buffer. Rather than tracking which steps
      // modify which locals, we conservatively bail.
      return null;
    } else {
      // Output JitId is neither allocated nor an input — can't handle this.
      return null;
    }
  }

  // Collect all distinct AluOps across all kernels for importing helpers
  let allOps: Map<AluOp, Set<DType>> = new Map();

  const collectKernelOps = (stepList: JitStep[]) => {
    for (const s of stepList) {
      if (s.type === "execute" && s.source instanceof Kernel) {
        for (const out of s.source.outputs) {
          const tune = tuneNullopt(
            Kernel.single(
              s.source.nargs,
              s.source.size,
              out.exp,
              out.reduction,
            ),
          );
          allOps = mapSetUnion(allOps, tune.exp.distinctOps());
          if (tune.epilogue)
            allOps = mapSetUnion(allOps, tune.epilogue.distinctOps());
        }
      } else if (s.type === "fori_loop") {
        collectKernelOps(s.bodyProgram.steps);
      }
    }
  };
  collectKernelOps(steps);

  // --- Generate WASM module ---
  const traceEnabled = _isCodeCaptureEnabled();
  const cg = new CodeGenerator();
  cg.trace = traceEnabled;

  // Configure memory import (shared/non-shared)
  configureMemoryImport(cg);

  // Import allocator functions
  const allocFunc = cg.importFunction("env", "alloc", [cg.i32], [cg.i32]);
  const freeFunc = cg.importFunction("env", "free", [cg.i32], []);

  // Import math helper functions (reuse shared helper from wasm.ts)
  const funcs = importWasmHelperFuncs(cg, allOps);

  // --- M6.2a: Extract kernels into separate WASM functions ---
  // Each extracted function has signature: (start: i32, end: i32, ...bufs: i32[]) => void
  // mega_execute calls them via direct `call` with (0, size, ...bufLocals).
  // V8's TurboFan inlines these back in the serial path (zero overhead).
  // The exports enable M6.2b/c to call them from workers with sub-ranges.
  type ExtractedFunc = {
    funcRef: number;
    exportName: string;
    nInputs: number;
    nOutputs: number;
  };
  const extractedForStep = new Map<number, ExtractedFunc>();
  // Body kernel extractions keyed by fori_loop step index, then body step index.
  const extractedForBodyStep = new Map<number, Map<number, ExtractedFunc>>();
  const kernelExports: ExtractedKernelInfo[] = [];
  let kernelCounter = 0;

  const extractKernel = (
    kernel: Kernel,
    nInputs: number,
    nOutputs: number,
  ): ExtractedFunc => {
    const isReduction = kernel.hasReduction;
    const kernelSize = kernel.size as number;
    const exportName = `kernel_${kernelCounter}`;
    const funcRef = emitExtractedKernelFunc(
      cg,
      funcs,
      kernel,
      nInputs,
      nOutputs,
    );
    cg.export(funcRef, exportName);
    kernelExports.push({
      name: exportName,
      size: kernelSize,
      isReduction,
      nInputs,
      nOutputs,
    });
    kernelCounter++;
    return { funcRef, exportName, nInputs, nOutputs };
  };

  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    if (step.type === "execute" && step.source instanceof Kernel) {
      extractedForStep.set(
        stepIdx,
        extractKernel(step.source, step.inputs.length, step.outputs.length),
      );
    } else if (step.type === "fori_loop") {
      const bodyMap = new Map<number, ExtractedFunc>();
      const bodySteps = step.bodyProgram.steps;
      for (let bsi = 0; bsi < bodySteps.length; bsi++) {
        const bs = bodySteps[bsi];
        if (bs.type === "execute" && bs.source instanceof Kernel) {
          bodyMap.set(
            bsi,
            extractKernel(bs.source, bs.inputs.length, bs.outputs.length),
          );
        }
      }
      extractedForBodyStep.set(stepIdx, bodyMap);
    }
  }

  // Build function: mega_execute(input0..inputN, resultBufPtr) -> void
  const numInputParams = inputIds.length;
  const totalParams = numInputParams + 1; // +1 for resultBufPtr
  const paramTypes = rep(totalParams, cg.i32);

  const megaFunc = cg.function(paramTypes, [], () => {
    // Create a WASM local for each JitId.
    // Input JitIds are initialized from function params.
    // Others are initialized when created (malloc, recycle).
    const jitIdLocals = new Map<JitId, number>();

    // Map input JitIds to function params
    for (let i = 0; i < inputIds.length; i++) {
      jitIdLocals.set(inputIds[i], i); // param index
    }

    // Declare locals for non-input JitIds
    for (const id of allJitIds) {
      if (!jitIdLocals.has(id)) {
        const local = cg.local.declare(cg.i32);
        jitIdLocals.set(id, local);
      }
    }

    // Result buffer pointer is the last param
    const resultBufParam = numInputParams;

    // Emit code for each step
    for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
      const step = steps[stepIdx];

      switch (step.type) {
        case "malloc": {
          if (isSymbolicSize(step.size)) {
            throw new Error("Mega-module: symbolic malloc sizes not supported");
          }
          const size = step.size as number;
          // local.set(jitId, call $alloc(size))
          cg.i32.const(size);
          cg.call(allocFunc);
          cg.local.set(jitIdLocals.get(step.output)!);

          // If pre-filled constant data is present, emit inline i32.store
          // instructions to write it directly into the allocated buffer.
          // These are tiny scalar buffers (2–8 bytes), so 1–2 stores suffice.
          if (step.initialData) {
            const data = step.initialData;
            const dv = new DataView(
              data.buffer,
              data.byteOffset,
              data.byteLength,
            );
            // Write 4-byte chunks
            for (let off = 0; off + 4 <= data.byteLength; off += 4) {
              cg.local.get(jitIdLocals.get(step.output)!);
              cg.i32.const(dv.getInt32(off, true));
              cg.i32.store(2, off); // align=4
            }
            // Handle 2-byte remainder (Float16)
            const rem = data.byteLength % 4;
            if (rem === 2) {
              const off = data.byteLength - 2;
              cg.local.get(jitIdLocals.get(step.output)!);
              cg.i32.const(dv.getUint16(off, true));
              cg.i32.store16(1, off); // align=2
            }
          }
          break;
        }

        case "free": {
          // call $free(local.get(jitId))
          cg.local.get(jitIdLocals.get(step.input)!);
          cg.call(freeFunc);
          break;
        }

        case "recycle": {
          // local.set(new, local.get(old)) — zero cost
          cg.local.get(jitIdLocals.get(step.input)!);
          cg.local.set(jitIdLocals.get(step.output)!);
          break;
        }

        case "execute": {
          if (step.source instanceof Kernel) {
            const extracted = extractedForStep.get(stepIdx);
            if (extracted) {
              // M6.2a: call extracted kernel function with (0, size, ...bufs)
              const kernelSize = step.source.size as number;
              cg.i32.const(0); // start
              cg.i32.const(kernelSize); // end
              for (const id of step.inputs) {
                cg.local.get(jitIdLocals.get(id)!);
              }
              for (const id of step.outputs) {
                cg.local.get(jitIdLocals.get(id)!);
              }
              cg.call(extracted.funcRef);
            } else {
              // Reduction kernel: keep inlined in mega_execute
              emitInlinedKernel(
                cg,
                funcs,
                step.source,
                step.inputs.map((id) => jitIdLocals.get(id)!),
                step.outputs.map((id) => jitIdLocals.get(id)!),
              );
            }
          }
          break;
        }

        case "fori_loop": {
          emitForiLoop(
            cg,
            funcs,
            step,
            jitIdLocals,
            extractedForBodyStep.get(stepIdx)!,
            allocFunc,
            freeFunc,
          );
          break;
        }
      }
    }

    // Write output pointers to result buffer
    for (let i = 0; i < outputIds.length; i++) {
      const outputLocal = jitIdLocals.get(outputIds[i])!;
      // i32.store(resultBufPtr + i*4, local.get(outputJitId))
      cg.local.get(resultBufParam);
      if (i > 0) {
        cg.i32.const(i * 4);
        cg.i32.add();
      }
      cg.local.get(outputLocal);
      cg.i32.store(2); // align=4 (log2(4)=2)
    }
  });

  cg.export(megaFunc, "mega_execute");

  const bytes = cg.finish();
  const module = new WebAssembly.Module(bytes);

  // --- Debug logging ---
  if (DEBUG >= 1) {
    const reductions = kernelExports.filter((k) => k.isReduction).length;
    const elementwise = kernelExports.length - reductions;
    console.info(
      `mega-module: ${steps.length} steps, ${kernelExports.length} kernels ` +
        `(${elementwise} elementwise, ${reductions} reduction), all extracted, ${bytes.length} bytes`,
    );
  }
  if (DEBUG >= 2) {
    for (const ke of kernelExports) {
      console.info(
        `  ${ke.name}: size=${ke.size} ${ke.isReduction ? "reduction" : "elementwise"} ` +
          `in=${ke.nInputs} out=${ke.nOutputs}`,
      );
    }
  }
  if (DEBUG >= 4) {
    // Hex dump for wasm-dis / wasm-tools inspection
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(
      " ",
    );
    console.info(`mega-module WASM bytes (${bytes.length}):\n${hex}`);
  }

  // --- Code capture ---
  if (traceEnabled) {
    _emitCodeCapture({
      backend: "wasm",
      kind: "mega-module",
      code: cg.toWat(),
      metadata: {
        numSteps: steps.length,
        numKernels: kernelExports.length,
        byteLength: bytes.length,
      },
    });
  }

  // --- M6.2c: Build step metadata for JS-driven parallel execution ---
  // Map JitId → sequential index: inputs first (0..numInputs-1), then others.
  const jitIdToIdx = new Map<JitId, number>();
  for (let i = 0; i < inputIds.length; i++) {
    jitIdToIdx.set(inputIds[i], i);
  }
  let nextIdx = inputIds.length;
  for (const id of allJitIds) {
    if (!jitIdToIdx.has(id)) {
      jitIdToIdx.set(id, nextIdx++);
    }
  }
  const numLocals = nextIdx;

  const stepInfos: MegaStepInfo[] = [];
  let stepKernelCounter = 0;
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        stepInfos.push({
          type: "malloc",
          outputIdx: jitIdToIdx.get(step.output)!,
          size: step.size as number,
          ...(step.initialData && { initialData: step.initialData }),
        });
        break;
      case "free":
        stepInfos.push({
          type: "free",
          inputIdx: jitIdToIdx.get(step.input)!,
        });
        break;
      case "recycle":
        stepInfos.push({
          type: "recycle",
          fromIdx: jitIdToIdx.get(step.input)!,
          toIdx: jitIdToIdx.get(step.output)!,
        });
        break;
      case "execute":
        if (step.source instanceof Kernel) {
          stepInfos.push({
            type: "kernel",
            kernelIdx: stepKernelCounter,
            kernelSize: step.source.size as number,
            inputIdxs: step.inputs.map((id) => jitIdToIdx.get(id)!),
            outputIdxs: step.outputs.map((id) => jitIdToIdx.get(id)!),
          });
          stepKernelCounter++;
        }
        break;
      case "fori_loop":
        // fori_loop is inherently sequential — no parallel step metadata.
        // Advance kernel counter past body kernels so indices stay correct.
        for (const bs of step.bodyProgram.steps) {
          if (bs.type === "execute" && bs.source instanceof Kernel)
            stepKernelCounter++;
        }
        break;
    }
  }

  const outputLocalIdxs = outputIds.map((id) => jitIdToIdx.get(id)!);

  // fori_loop steps are only handled by mega_execute (not stepInfos-driven
  // parallel dispatch), so disable the parallel path if any are present.
  const hasForiLoop = steps.some((s) => s.type === "fori_loop");

  return {
    module,
    numInputs: numInputParams,
    numOutputs: outputIds.length,
    outputSizes,
    unsupportedSteps: [],
    kernelExports,
    stepInfos,
    numLocals,
    outputLocalIdxs,
    canParallelize: !hasForiLoop,
  };
}

// ---------------------------------------------------------------------------
// ForiLoop emission
// ---------------------------------------------------------------------------

/**
 * Emit a WASM loop implementing a fori_loop step.
 *
 * The loop runs from `lower` to `upper`, executing the body program's steps
 * each iteration. Body inputs are mapped to outer-scope locals (consts, idx
 * buffer, carry buffers). After each iteration, old carry buffers are freed
 * and carry locals updated to the body's outputs.
 *
 * At loop exit, final carries are memory.copy'd into the pre-allocated
 * output buffers, and the last iteration's carry buffers are freed.
 */
function emitForiLoop(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  step: Extract<JitStep, { type: "fori_loop" }>,
  outerLocals: Map<JitId, number>,
  bodyExtracted: Map<
    number,
    { funcRef: number; nInputs: number; nOutputs: number }
  >,
  allocFunc: number,
  freeFunc: number,
): void {
  const bodyProg = step.bodyProgram;
  const lower = step.lower as number;
  const upper = step.upper as number;
  const numConsts = step.numConsts;
  const numCarry = step.initCarries.length;

  // --- Allocate index buffer (4 bytes for i32 scalar) ---
  const idxBufLocal = cg.local.declare(cg.i32);
  cg.i32.const(4);
  cg.call(allocFunc);
  cg.local.set(idxBufLocal);

  // --- Carry tracking locals (hold current carry buffer pointers) ---
  const carryLocals: number[] = [];
  for (let k = 0; k < numCarry; k++) {
    const cl = cg.local.declare(cg.i32);
    cg.local.get(outerLocals.get(step.initCarries[k])!);
    cg.local.set(cl);
    carryLocals.push(cl);
  }

  // Old-carry locals (for freeing after body runs)
  const oldCarryLocals: number[] = [];
  for (let k = 0; k < numCarry; k++) {
    oldCarryLocals.push(cg.local.declare(cg.i32));
  }

  // ownsCarry flag: 0 = initial carries (parent-owned), 1 = body-allocated
  const ownsCarryLocal = cg.local.declare(cg.i32);
  cg.i32.const(0);
  cg.local.set(ownsCarryLocal);

  // --- Create WASM locals for all body JitIds ---
  const bodyLocals = new Map<JitId, number>();

  // Body input layout: [consts..., idx, carries...]
  // Map const inputs to outer's const locals
  for (let k = 0; k < numConsts; k++) {
    bodyLocals.set(bodyProg.inputs[k], outerLocals.get(step.consts[k])!);
  }
  // Map idx input to idxBufLocal
  bodyLocals.set(bodyProg.inputs[numConsts], idxBufLocal);
  // Map carry inputs to carryLocals
  for (let k = 0; k < numCarry; k++) {
    bodyLocals.set(bodyProg.inputs[numConsts + 1 + k], carryLocals[k]);
  }

  // Declare locals for non-input body JitIds
  const bodyAllIds = new Set<JitId>();
  for (const id of bodyProg.inputs) bodyAllIds.add(id);
  for (const id of bodyProg.outputs) bodyAllIds.add(id);
  for (const bs of bodyProg.steps) {
    switch (bs.type) {
      case "malloc":
        bodyAllIds.add(bs.output);
        break;
      case "free":
        bodyAllIds.add(bs.input);
        break;
      case "recycle":
        bodyAllIds.add(bs.input);
        bodyAllIds.add(bs.output);
        break;
      case "execute":
        for (const id of bs.inputs) bodyAllIds.add(id);
        for (const id of bs.outputs) bodyAllIds.add(id);
        break;
    }
  }
  for (const id of bodyAllIds) {
    if (!bodyLocals.has(id)) {
      bodyLocals.set(id, cg.local.declare(cg.i32));
    }
  }

  // --- Loop counter ---
  const iLocal = cg.local.declare(cg.i32);
  cg.i32.const(lower);
  cg.local.set(iLocal);

  // --- WASM loop ---
  cg.loop(cg.void);
  {
    cg.block(cg.void);
    // if (i >= upper) break
    cg.local.get(iLocal);
    cg.i32.const(upper);
    cg.i32.ge_s();
    cg.br_if(0);

    // Store loop index to idx buffer
    cg.local.get(idxBufLocal);
    cg.local.get(iLocal);
    cg.i32.store(2); // align=4

    // Update body carry-input locals from current carry locals
    // (needed because carryLocals are updated at end of each iteration)
    for (let k = 0; k < numCarry; k++) {
      cg.local.get(carryLocals[k]);
      cg.local.set(bodyLocals.get(bodyProg.inputs[numConsts + 1 + k])!);
    }

    // Save old carry pointers before body executes
    for (let k = 0; k < numCarry; k++) {
      cg.local.get(carryLocals[k]);
      cg.local.set(oldCarryLocals[k]);
    }

    // --- Emit body steps ---
    for (let bsi = 0; bsi < bodyProg.steps.length; bsi++) {
      const bs = bodyProg.steps[bsi];
      switch (bs.type) {
        case "malloc": {
          const size = bs.size as number;
          cg.i32.const(size);
          cg.call(allocFunc);
          cg.local.set(bodyLocals.get(bs.output)!);
          if (bs.initialData) {
            const data = bs.initialData;
            const dv = new DataView(
              data.buffer,
              data.byteOffset,
              data.byteLength,
            );
            for (let off = 0; off + 4 <= data.byteLength; off += 4) {
              cg.local.get(bodyLocals.get(bs.output)!);
              cg.i32.const(dv.getInt32(off, true));
              cg.i32.store(2, off);
            }
            const rem = data.byteLength % 4;
            if (rem === 2) {
              const off = data.byteLength - 2;
              cg.local.get(bodyLocals.get(bs.output)!);
              cg.i32.const(dv.getUint16(off, true));
              cg.i32.store16(1, off);
            }
          }
          break;
        }
        case "free": {
          cg.local.get(bodyLocals.get(bs.input)!);
          cg.call(freeFunc);
          break;
        }
        case "recycle": {
          cg.local.get(bodyLocals.get(bs.input)!);
          cg.local.set(bodyLocals.get(bs.output)!);
          break;
        }
        case "execute": {
          if (bs.source instanceof Kernel) {
            const extracted = bodyExtracted.get(bsi);
            if (extracted) {
              const kernelSize = bs.source.size as number;
              cg.i32.const(0);
              cg.i32.const(kernelSize);
              for (const id of bs.inputs) {
                cg.local.get(bodyLocals.get(id)!);
              }
              for (const id of bs.outputs) {
                cg.local.get(bodyLocals.get(id)!);
              }
              cg.call(extracted.funcRef);
            } else {
              emitInlinedKernel(
                cg,
                funcs,
                bs.source,
                bs.inputs.map((id) => bodyLocals.get(id)!),
                bs.outputs.map((id) => bodyLocals.get(id)!),
              );
            }
          }
          break;
        }
      }
    }

    // Free old carries if owned (skip first iteration — parent-owned)
    cg.local.get(ownsCarryLocal);
    cg.if(cg.void);
    for (let k = 0; k < numCarry; k++) {
      cg.local.get(oldCarryLocals[k]);
      cg.call(freeFunc);
    }
    cg.end();

    // Update carry locals to body's output pointers
    for (let k = 0; k < numCarry; k++) {
      cg.local.get(bodyLocals.get(bodyProg.outputs[k])!);
      cg.local.set(carryLocals[k]);
    }

    // Mark carries as owned after first iteration
    cg.i32.const(1);
    cg.local.set(ownsCarryLocal);

    // i++
    cg.local.get(iLocal);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(iLocal);

    cg.br(1); // continue loop
    cg.end(); // end block
  }
  cg.end(); // end loop

  // --- Copy final carries to pre-allocated output buffers ---
  for (let k = 0; k < numCarry; k++) {
    // memory.copy(dst, src, len)
    cg.local.get(outerLocals.get(step.outputs[k])!); // dst
    cg.local.get(carryLocals[k]); // src
    cg.i32.const(step.carrySizeBytes[k]); // len
    cg.memory.copy();
  }

  // Free last iteration's carry buffers (if loop executed at least once)
  cg.local.get(ownsCarryLocal);
  cg.if(cg.void);
  for (let k = 0; k < numCarry; k++) {
    cg.local.get(carryLocals[k]);
    cg.call(freeFunc);
  }
  cg.end();

  // Free index buffer
  cg.local.get(idxBufLocal);
  cg.call(freeFunc);
}

// ---------------------------------------------------------------------------
// Extracted kernel functions (M6.2a)
// ---------------------------------------------------------------------------

/**
 * Emit a separate WASM function for a non-reduction kernel.
 *
 * Signature: (start: i32, end: i32, inBuf0: i32, ..., outBuf0: i32, ...) => void
 *
 * The gidx loop runs [start, end) instead of [0, size). Buffer pointers
 * are function parameters, not mega_execute locals. This makes the function
 * independently callable by workers with sub-ranges for M6.2c.
 *
 * V8's TurboFan inlines direct `call` to these functions when called from
 * mega_execute, so serial performance is unchanged.
 */
function emitExtractedKernelFunc(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  nInputs: number,
  nOutputs: number,
): number {
  // Params: (start, end, inBuf0..inBufN, outBuf0..outBufM)
  const nBufParams = nInputs + nOutputs;
  const paramTypes = rep(2 + nBufParams, cg.i32); // start, end, ...bufs

  return cg.function(paramTypes, [], () => {
    // Param indices:
    // 0 = start, 1 = end
    // 2..2+nInputs-1 = input buffer pointers
    // 2+nInputs..2+nInputs+nOutputs-1 = output buffer pointers
    const startParam = 0;
    const endParam = 1;
    const inputLocals = Array.from({ length: nInputs }, (_, i) => 2 + i);
    const outputLocals = Array.from(
      { length: nOutputs },
      (_, i) => 2 + nInputs + i,
    );

    if (kernel.isMultiOutput) {
      emitExtractedMultiOutputBody(
        cg,
        funcs,
        kernel,
        inputLocals,
        outputLocals,
        startParam,
        endParam,
      );
    } else {
      emitExtractedSingleOutputBody(
        cg,
        funcs,
        kernel,
        inputLocals,
        outputLocals[0],
        startParam,
        endParam,
      );
    }
  });
}

/**
 * Emit the body of an extracted single-output kernel function.
 * gidx loop runs [startParam, endParam).
 */
function emitExtractedSingleOutputBody(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocal: number,
  startParam: number,
  endParam: number,
): void {
  const tune = tuneNullopt(kernel);
  const out = kernel.outputs[0];
  const re = out.reduction;
  const storeAlign = Math.log2(byteWidth(out.dtype));

  // SIMD eligibility: f32 elementwise, no reduction, contiguous access
  const useSimd = !re && canVectorizeSimd(tune.exp, out.dtype);

  if (DEBUG >= 2 && useSimd) {
    console.info(
      `mega-module: SIMD f32x4 path for extracted kernel (nargs=${kernel.nargs})`,
    );
  }

  const gidx = cg.local.declare(cg.i32);

  if (useSimd) {
    // ----- SIMD path: main f32x4 loop + scalar tail -----

    // SIMD main loop
    cg.local.get(startParam);
    cg.local.set(gidx);
    cg.loop(cg.void);
    {
      cg.block(cg.void);
      // Break if gidx + 4 > end
      cg.local.get(gidx);
      cg.i32.const(4);
      cg.i32.add();
      cg.local.get(endParam);
      cg.i32.gt_u();
      cg.br_if(0);

      // Push output address: outBase + gidx * 4
      cg.local.get(outputLocal);
      cg.local.get(gidx);
      cg.i32.const(4); // byteWidth(Float32)
      cg.i32.mul();
      cg.i32.add();

      // Emit SIMD expression
      translateExpCoreSimd(cg, tune.exp, gidx, (gid) => inputLocals[gid]);

      // v128.store
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

    // Scalar tail
    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(gidx);
      cg.local.get(endParam);
      cg.i32.ge_u();
      cg.br_if(0);

      cg.local.get(outputLocal);
      cg.local.get(gidx);
      cg.i32.const(4);
      cg.i32.mul();
      cg.i32.add();

      translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);

      cg.f32.store(2);

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

    // gidx = start
    cg.local.get(startParam);
    cg.local.set(gidx);

    // loop
    cg.loop(cg.void);
    {
      // if (gidx >= end) break
      cg.block(cg.void);
      cg.local.get(gidx);
      cg.local.get(endParam);
      cg.i32.ge_u();
      cg.br_if(0);

      // Output address: outputLocal + gidx * byteWidth
      cg.local.get(outputLocal);
      cg.local.get(gidx);
      cg.i32.const(byteWidth(out.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (re) {
        // Reduction: accumulator + inner ridx loop
        emitReductionBody(cg, funcs, tune, out, re, gidx, inputLocals);
      } else {
        // Translate expression
        translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
      }

      // Store result
      dty(cg, null, out.dtype).store(storeAlign);

      // gidx++
      cg.local.get(gidx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(gidx);

      cg.br(1);
      cg.end();
    }
    cg.end();
  }
}

/**
 * Emit the body of an extracted multi-output kernel function.
 * gidx loop runs [startParam, endParam).
 */
function emitExtractedMultiOutputBody(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocals: number[],
  startParam: number,
  endParam: number,
): void {
  const numOutputs = kernel.numOutputs;
  const tunes = kernel.outputs.map((o) => {
    const tmpKernel = Kernel.single(
      kernel.nargs,
      kernel.size,
      o.exp,
      o.reduction,
    );
    return tuneNullopt(tmpKernel);
  });

  const gidx = cg.local.declare(cg.i32);

  // gidx = start
  cg.local.get(startParam);
  cg.local.set(gidx);

  // loop
  cg.loop(cg.void);
  {
    // if (gidx >= end) break
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.local.get(endParam);
    cg.i32.ge_u();
    cg.br_if(0);

    // For each output: compute + store
    for (let oi = 0; oi < numOutputs; oi++) {
      const tune = tunes[oi];
      const out = kernel.outputs[oi];
      const storeAlign = Math.log2(byteWidth(out.dtype));

      // Output address: outputLocals[oi] + gidx * byteWidth
      cg.local.get(outputLocals[oi]);
      cg.local.get(gidx);
      cg.i32.const(byteWidth(out.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (out.reduction) {
        // Reduction output — each gidx independently reduces
        emitReductionBody(
          cg,
          funcs,
          tune,
          out,
          out.reduction,
          gidx,
          inputLocals,
        );
      } else {
        translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
      }

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
}

// ---------------------------------------------------------------------------
// Inline kernel emission (reduction kernels only after M6.2a)
// ---------------------------------------------------------------------------

/**
 * Emit an inlined kernel body within the mega-module.
 *
 * After M6.2a, this is only used for reduction kernels (which must run
 * single-threaded due to accumulator dependencies). Non-reduction kernels
 * are extracted into separate functions by emitExtractedKernelFunc.
 *
 * This reuses the gidx loop structure but references JitId locals
 * instead of function parameters for input/output buffers.
 */
function emitInlinedKernel(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocals: number[],
): void {
  if (kernel.isMultiOutput) {
    emitInlinedMultiOutputKernel(cg, funcs, kernel, inputLocals, outputLocals);
    return;
  }

  const tune = tuneNullopt(kernel);
  const out = kernel.outputs[0];
  const re = out.reduction;
  const storeAlign = Math.log2(byteWidth(out.dtype));

  const gidx = cg.local.declare(cg.i32);
  const kernelSize = kernel.size as number;

  // gidx = 0
  cg.i32.const(0);
  cg.local.set(gidx);

  // loop
  cg.loop(cg.void);
  {
    // if (gidx >= size) break
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.i32.const(kernelSize);
    cg.i32.ge_u();
    cg.br_if(0);

    // Output address: outputLocal + gidx * byteWidth
    cg.local.get(outputLocals[0]);
    cg.local.get(gidx);
    cg.i32.const(byteWidth(out.dtype));
    cg.i32.mul();
    cg.i32.add();

    if (re) {
      // Reduction: accumulator + inner ridx loop
      emitReductionBody(cg, funcs, tune, out, re, gidx, inputLocals);
    } else {
      // Non-reduction: translate expression directly
      translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
    }

    // Store result
    dty(cg, null, out.dtype).store(storeAlign);

    // gidx++
    cg.local.get(gidx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(gidx);

    cg.br(1);
    cg.end();
  }
  cg.end();
}

/**
 * Emit an inlined multi-output kernel body.
 * One gidx loop evaluates and stores all output expressions.
 */
function emitInlinedMultiOutputKernel(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocals: number[],
): void {
  const numOutputs = kernel.numOutputs;
  const tunes = kernel.outputs.map((o) => {
    const tmpKernel = Kernel.single(
      kernel.nargs,
      kernel.size,
      o.exp,
      o.reduction,
    );
    return tuneNullopt(tmpKernel);
  });

  const gidx = cg.local.declare(cg.i32);
  const kernelSize = kernel.size as number;

  // gidx = 0
  cg.i32.const(0);
  cg.local.set(gidx);

  // loop
  cg.loop(cg.void);
  {
    // if (gidx >= size) break
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.i32.const(kernelSize);
    cg.i32.ge_u();
    cg.br_if(0);

    // For each output: compute + store
    for (let oi = 0; oi < numOutputs; oi++) {
      const tune = tunes[oi];
      const out = kernel.outputs[oi];
      const storeAlign = Math.log2(byteWidth(out.dtype));

      // Output address: outputLocals[oi] + gidx * byteWidth
      cg.local.get(outputLocals[oi]);
      cg.local.get(gidx);
      cg.i32.const(byteWidth(out.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (out.reduction) {
        emitReductionBody(
          cg,
          funcs,
          tune,
          out,
          out.reduction,
          gidx,
          inputLocals,
        );
      } else {
        translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
      }

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
}

// ---------------------------------------------------------------------------
// Reduction body emission
// ---------------------------------------------------------------------------

/**
 * Emit reduction body: accumulator init, ridx loop, epilogue.
 * Shared between single-output and multi-output inlined kernels.
 */
function emitReductionBody(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  tune: ReturnType<typeof tuneNullopt>,
  out: {
    exp: AluExp;
    dtype: DType;
    reduction?: {
      op: AluOp;
      dtype: DType;
      size: SizeExpr;
      identity: number;
    } | null;
  },
  re: { op: AluOp; dtype: DType; size: SizeExpr; identity: number },
  gidx: number,
  inputLocals: number[],
): void {
  const acc = cg.local.declare(dty(cg, null, out.exp.dtype));
  dty(cg, null, out.exp.dtype).const(re.identity);
  cg.local.set(acc);

  // Kahan compensation for Float64 Add reductions
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

  // Inner reduction loop
  // Note: mega-module rejects symbolic sizes in canCompileToMegaModule(),
  // so re.size is guaranteed to be concrete here.
  cg.loop(cg.void);
  {
    cg.block(cg.void);
    cg.local.get(ridx);
    cg.i32.const(re.size as number);
    cg.i32.ge_u();
    cg.br_if(0);

    translateExpMega(cg, funcs, tune.exp, gidx, inputLocals, ridx);
    codegenReductionAccumulate(cg, re, acc, kahanComp);

    cg.local.get(ridx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(ridx);

    cg.br(1);
    cg.end();
  }
  cg.end();

  // Epilogue: apply post-reduction expression (e.g., divide by count for mean)
  translateExpMega(
    cg,
    funcs,
    tune.epilogue!,
    gidx,
    inputLocals,
    undefined,
    acc,
  );
}

// ---------------------------------------------------------------------------
// Expression translation for mega-module context
// ---------------------------------------------------------------------------

/**
 * Translate an AluExp within the mega-module context.
 * Delegates to the shared translateExpCore from wasm.ts, providing
 * mega-module-specific variable resolution and GlobalIndex handling.
 *
 * In the mega-module context, GlobalIndex buffer references use WASM locals
 * (inputLocals) directly, without the (start, end) parameter offset that
 * regular kernels have.
 */
function translateExpMega(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  gidx: number,
  inputLocals: number[],
  ridx?: number,
  acc?: number,
): void {
  translateExpCore(cg, funcs, exp, {
    getVariable: (name) => {
      if (name === "gidx") return gidx;
      if (name === "ridx") return ridx;
      if (name === "acc") return acc;
      return undefined;
    },
    handleGlobalIndex: (cg, gen, gid, len, indexExp, dtype) => {
      gen(indexExp);

      // Bounds check (same as regular kernel translation)
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.i32.const(0);
      cg.local.get(local);
      cg.i32.const(len);
      cg.i32.lt_u();
      cg.select();

      // Compute byte offset
      cg.i32.const(byteWidth(dtype));
      cg.i32.mul();

      // Add base pointer from the mega-module's input local
      // (no RANGE_PARAMS offset — mega-module maps JitIds directly)
      cg.local.get(inputLocals[gid]);
      cg.i32.add();

      // Load value
      dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(byteWidth(dtype)));
    },
  });
}
