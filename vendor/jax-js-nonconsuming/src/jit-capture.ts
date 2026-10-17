/**
 * JIT compilation capture facility.
 *
 * Captures the full compilation report for a jitted function: step list,
 * kernel details, shader/WASM code, tuning decisions, and command tape stats.
 * Useful for comparing compiler output across branches or hardware profiles.
 *
 * Usage:
 *   const report = captureJitReport(fn, ...args);
 *   console.log(formatJitReport(report));
 */

import { type DType, Kernel, type Reduction } from "./alu";
import {
  type BackendCapabilities,
  getBackend,
  setCodeCapture,
} from "./backend";
import type { ArrayLike } from "./frontend/array";
import { jit, makeJaxpr } from "./frontend/jaxpr";
import { jitCompile, JitProgram, type JitStepCounts } from "./frontend/jit";
import { Routine } from "./routine";
import {
  CONSERVATIVE_WEBGPU_PERF_DEFAULTS,
  resolvePerformanceBelief,
} from "./tuner";

// ── Report types ──────────────────────────────────────────────────────────

/** One kernel dispatch in the JIT program. */
export interface CapturedKernel {
  /** Step index in the program. */
  stepIdx: number;
  /** Human-readable label (e.g. "k0_add", "k1_reduce_mul"). */
  label?: string;
  /** Number of kernel inputs. */
  nargs: number;
  /** Kernel size expression (number or symbolic string). */
  size: string;
  /** Whether this is a multi-output kernel. */
  multiOutput: boolean;
  /** Number of outputs from this kernel. */
  numOutputs: number;
  /** Per-output details. */
  outputs: {
    dtype: string;
    bytes: string;
    hasReduction: boolean;
    reductionOp?: string;
    reductionSize?: string;
    /** The AluExp expression tree (toString). */
    expression: string;
  }[];
  /** Full Kernel.toString() for detailed inspection. */
  kernelString: string;
}

/** A routine dispatch in the JIT program. */
export interface CapturedRoutine {
  stepIdx: number;
  name: string;
  numInputs: number;
  numOutputs: number;
}

/** A sub-program (scan body, block_map body, fori_loop body, etc.). */
export interface CapturedSubProgram {
  stepIdx: number;
  type: string;
  /** Human-readable label of the body program. */
  label?: string;
  /** Summary of the sub-program's steps. */
  stepCounts: JitStepCounts;
  /** Recursive capture of the sub-program. */
  program: CapturedProgram;
}

/** Captured shader/WASM code emitted during compilation. */
export interface CapturedCode {
  backend: "webgpu" | "wasm" | "cpu" | "webgl";
  kind: string;
  label?: string;
  code?: string;
  workgroupSize?: [number, number, number];
  metadata?: Record<string, unknown>;
}

/** Full capture of a JitProgram. */
export interface CapturedProgram {
  /** Backend device type. */
  device: string;
  /** Human-readable label (e.g. "mandelbrot#a3f1"). */
  label?: string;
  /** Number of inputs. */
  numInputs: number;
  /** Number of outputs. */
  numOutputs: number;
  /** Total steps. */
  numSteps: number;
  /** Step type counts. */
  stepCounts: JitStepCounts;
  /** Pool hints (peak memory, malloc sizes). */
  poolHints: { peakBytes: number; mallocSizes: number[] };
  /** Command tape stats (WebGPU only, null if not compiled). */
  commandTapeStats: {
    hasConstSlab: boolean;
    constSlabEntries: number;
    arenaSlabCount: number;
    arenaEntryCount: number;
    recycleCount: number;
    dispatchCount: number;
    tableSize: number;
  } | null;
  /** All execute steps with kernel details. */
  kernels: CapturedKernel[];
  /** All routine dispatches. */
  routines: CapturedRoutine[];
  /** All sub-programs (scan, block_map, fori_loop bodies). */
  subPrograms: CapturedSubProgram[];
  /** Full JitProgram.toString() output. */
  programString: string;
}

/** Full report from captureJitReport. */
export interface JitReport {
  /** Backend capabilities at capture time. */
  capabilities: Partial<BackendCapabilities>;
  /** The compiled program. */
  program: CapturedProgram;
  /** Shader/WASM code emitted during compilation. */
  codeEntries: CapturedCode[];
}

// ── Implementation ────────────────────────────────────────────────────────

function dtypeName(dt: DType): string {
  return dt;
}

function captureProgram(prog: JitProgram): CapturedProgram {
  const kernels: CapturedKernel[] = [];
  const routines: CapturedRoutine[] = [];
  const subPrograms: CapturedSubProgram[] = [];

  for (let i = 0; i < prog.steps.length; i++) {
    const step = prog.steps[i];

    if (step.type === "execute") {
      if (step.source instanceof Kernel) {
        const k = step.source;
        kernels.push({
          stepIdx: i,
          label: k.label,
          nargs: k.nargs,
          size: String(k.size),
          multiOutput: k.isMultiOutput,
          numOutputs: k.numOutputs,
          outputs: k.outputs.map((o) => ({
            dtype: dtypeName(o.dtype),
            bytes: String(o.bytes),
            hasReduction: !!o.reduction,
            reductionOp: o.reduction ? reductionOpName(o.reduction) : undefined,
            reductionSize: o.reduction ? String(o.reduction.size) : undefined,
            expression: String(o.exp),
          })),
          kernelString: k.toString(),
        });
      } else {
        // Routine
        const r = step.source as Routine;
        routines.push({
          stepIdx: i,
          name: r.name,
          numInputs: step.inputs.length,
          numOutputs: step.outputs.length,
        });
      }
    }

    // Capture sub-programs
    if (step.type === "scan" && step.bodyProgram) {
      subPrograms.push({
        stepIdx: i,
        type: `scan(${step.plan.path})`,
        label: step.bodyProgram.label,
        stepCounts: step.bodyProgram.stepCounts(),
        program: captureProgram(step.bodyProgram),
      });
    } else if (step.type === "fori_loop" && step.bodyProgram) {
      subPrograms.push({
        stepIdx: i,
        type: `fori_loop(${step.lower}..${step.upper})`,
        label: step.bodyProgram.label,
        stepCounts: step.bodyProgram.stepCounts(),
        program: captureProgram(step.bodyProgram),
      });
    } else if (step.type === "block_map" && step.bodyProgram) {
      subPrograms.push({
        stepIdx: i,
        type: `block_map(blockShape=[${step.blockShape}])`,
        label: step.bodyProgram.label,
        stepCounts: step.bodyProgram.stepCounts(),
        program: captureProgram(step.bodyProgram),
      });
    } else if (step.type === "assoc_scan" && step.bodyProgram) {
      subPrograms.push({
        stepIdx: i,
        type: `assoc_scan(${step.plan.path})`,
        label: step.bodyProgram.label,
        stepCounts: step.bodyProgram.stepCounts(),
        program: captureProgram(step.bodyProgram),
      });
    }
  }

  // Force command tape compilation to get stats
  const tapeStats = prog.commandTapeStats();

  return {
    device: prog.backend.type,
    label: prog.label,
    numInputs: prog.inputs.length,
    numOutputs: prog.outputs.length,
    numSteps: prog.steps.length,
    stepCounts: prog.stepCounts(),
    poolHints: {
      peakBytes: prog.poolHints.peakBytes,
      mallocSizes: [...prog.poolHints.mallocSizes],
    },
    commandTapeStats: tapeStats,
    kernels,
    routines,
    subPrograms,
    programString: prog.toString(),
  };
}

function reductionOpName(r: Reduction): string {
  return String(r.op);
}

/**
 * Capture the full JIT compilation report for a function.
 *
 * This JIT-compiles and executes the function once with the given arguments,
 * capturing all compilation decisions: kernel details, shader/WASM code,
 * tuning choices, and step structure.
 *
 * **Note:** Shader/WASM code entries are only captured on JIT cache miss.
 * Call `clearCaches()` before this function if you want to force fresh
 * compilation and see the generated code.
 *
 * **Note:** This temporarily replaces any active `setCodeCapture` callback
 * for the duration of the call, restoring `null` afterwards.
 *
 * @param fn The function to capture (not pre-jitted — raw function)
 * @param args Arguments to trace with and execute
 * @returns Full JIT compilation report
 *
 * @example
 * ```ts
 * const report = captureJitReport(
 *   (x) => x.mul(2).add(1),
 *   np.ones([1024])
 * );
 * console.log(formatJitReport(report));
 * ```
 */
export function captureJitReport(
  fn: (...args: any[]) => any,
  ...args: ArrayLike[]
): JitReport {
  const backend = getBackend();

  // Collect emitted code entries. This temporarily replaces any active
  // capture callback. Code entries fire during execution (kernel compilation
  // is lazy), so we must actually execute the function.
  const codeEntries: CapturedCode[] = [];
  setCodeCapture((entry) => {
    codeEntries.push({
      backend: entry.backend,
      kind: entry.kind,
      label: entry.label,
      code: entry.code,
      workgroupSize: entry.workgroupSize,
      metadata: entry.metadata,
    });
  });

  let program: CapturedProgram;
  try {
    // Execute the jitted function once to trigger both tracing and kernel
    // compilation (which emits code capture entries).
    using jitted = jit(fn);
    const result = jitted(...args);
    // Dispose the result if it's an array or pytree of arrays
    disposeResult(result);

    // Now trace again to get the JitProgram structure. The jitCompile call
    // will hit cache (same jaxpr hash), so this is essentially free.
    const traced = makeJaxpr(fn)(...args);
    const closedJaxpr = traced.jaxpr;
    try {
      const jp = jitCompile(backend, closedJaxpr.jaxpr);
      program = captureProgram(jp);
    } finally {
      closedJaxpr.dispose();
    }
  } finally {
    setCodeCapture(null);
  }

  return {
    capabilities: { ...backend.capabilities },
    program,
    codeEntries,
  };
}

/** Dispose a result that may be an Array, array of Arrays, or pytree. */
function disposeResult(result: any): void {
  if (result == null) return;
  if (typeof result === "number" || typeof result === "boolean") return;
  if (typeof result[Symbol.dispose] === "function") {
    result[Symbol.dispose]();
    return;
  }
  if (globalThis.Array.isArray(result)) {
    for (const item of result) {
      disposeResult(item);
    }
    return;
  }
  if (typeof result === "object") {
    for (const key of Object.keys(result)) {
      disposeResult(result[key]);
    }
  }
}

// ── Formatting ────────────────────────────────────────────────────────────

/**
 * Format a JIT report as a human-readable string for comparison.
 *
 * The output is designed to be diff-friendly: sorted, deterministic,
 * and with stable formatting.
 */
export function formatJitReport(report: JitReport): string {
  const lines: string[] = [];
  const emit = (line: string) => lines.push(line);
  const indent = (n: number) => "  ".repeat(n);

  emit("=== JIT Compilation Report ===");
  emit("");

  // Backend info
  const caps = report.capabilities;
  emit("Backend:");
  emit(
    `${indent(1)}maxComputeWorkgroupSizeX: ${caps.maxComputeWorkgroupSizeX ?? "?"}`,
  );
  emit(
    `${indent(1)}maxComputeWorkgroupStorageSize: ${caps.maxComputeWorkgroupStorageSize ?? "?"}`,
  );
  emit(
    `${indent(1)}maxComputeInvocationsPerWorkgroup: ${caps.maxComputeInvocationsPerWorkgroup ?? "?"}`,
  );
  emit(`${indent(1)}shaderF16: ${caps.shaderF16 ?? false}`);
  emit(`${indent(1)}subgroups: ${caps.subgroups ?? false}`);
  const belief = resolvePerformanceBelief(caps);
  if (caps.calibrated) {
    emit(`${indent(1)}calibrated: true`);
    emit(`${indent(1)}bandwidthGBs: ${belief.bandwidthGBs}`);
    emit(`${indent(1)}tflops: ${belief.tflops}`);
    emit(`${indent(1)}dispatchOverheadUs: ${belief.dispatchOverheadUs}`);
    emit(`${indent(1)}rOptWords: ${belief.rOptWords}`);
  } else {
    emit(`${indent(1)}calibrated: false`);
    emit(`${indent(1)}performanceBelief: conservative-defaults`);
    emit(
      `${indent(1)}conservativeDefaults: dispatch=${CONSERVATIVE_WEBGPU_PERF_DEFAULTS.dispatchOverheadUs}us bw=${CONSERVATIVE_WEBGPU_PERF_DEFAULTS.bandwidthGBs}GB/s tflops=${CONSERVATIVE_WEBGPU_PERF_DEFAULTS.tflops} rOpt=${CONSERVATIVE_WEBGPU_PERF_DEFAULTS.rOptWords}`,
    );
  }
  emit("");

  // Program overview
  const { program: p } = report;
  emit("Program:");
  if (p.label) emit(`${indent(1)}label: "${p.label}"`);
  emit(`${indent(1)}device: ${p.device}`);
  emit(`${indent(1)}inputs: ${p.numInputs}`);
  emit(`${indent(1)}outputs: ${p.numOutputs}`);
  emit(`${indent(1)}steps: ${p.numSteps}`);
  emit(`${indent(1)}peakBytes: ${p.poolHints.peakBytes}`);
  emit("");

  // Step counts
  emit("Step Counts:");
  for (const [type, count] of Object.entries(p.stepCounts)) {
    if (count > 0) emit(`${indent(1)}${type}: ${count}`);
  }
  emit("");

  // Command tape stats
  if (p.commandTapeStats) {
    const ts = p.commandTapeStats;
    emit("Command Tape:");
    emit(`${indent(1)}dispatches: ${ts.dispatchCount}`);
    emit(`${indent(1)}tableSize: ${ts.tableSize}`);
    emit(`${indent(1)}recycleCount: ${ts.recycleCount}`);
    emit(
      `${indent(1)}constSlab: ${ts.hasConstSlab} (${ts.constSlabEntries} entries)`,
    );
    emit(
      `${indent(1)}arenaSlabs: ${ts.arenaSlabCount} (${ts.arenaEntryCount} entries)`,
    );
    emit("");
  }

  // Kernels
  if (p.kernels.length > 0) {
    emit("Kernels:");
    for (const k of p.kernels) {
      const kLabel = k.label ? ` "${k.label}"` : "";
      emit(
        `${indent(1)}[step ${k.stepIdx}]${kLabel} size=${k.size} nargs=${k.nargs} outputs=${k.numOutputs}`,
      );
      for (let oi = 0; oi < k.outputs.length; oi++) {
        const o = k.outputs[oi];
        let line = `${indent(2)}output[${oi}]: ${o.dtype} (${o.bytes} bytes)`;
        if (o.hasReduction) {
          line += ` REDUCTION ${o.reductionOp} size=${o.reductionSize}`;
        }
        emit(line);
        emit(`${indent(3)}expr: ${o.expression}`);
      }
    }
    emit("");
  }

  // Routines
  if (p.routines.length > 0) {
    emit("Routines:");
    for (const r of p.routines) {
      emit(
        `${indent(1)}[step ${r.stepIdx}] ${r.name} inputs=${r.numInputs} outputs=${r.numOutputs}`,
      );
    }
    emit("");
  }

  // Sub-programs
  if (p.subPrograms.length > 0) {
    emit("Sub-Programs:");
    for (const sp of p.subPrograms) {
      const spLabel = sp.label ? ` "${sp.label}"` : "";
      emit(`${indent(1)}[step ${sp.stepIdx}] ${sp.type}${spLabel}`);
      const sc = sp.stepCounts;
      const active = Object.entries(sc).filter(([, v]) => v > 0);
      emit(
        `${indent(2)}steps: ${active.map(([k, v]) => `${k}=${v}`).join(", ")}`,
      );
      // Recursively format body kernels
      if (sp.program.kernels.length > 0) {
        for (const k of sp.program.kernels) {
          const kLabel = k.label ? ` "${k.label}"` : "";
          emit(
            `${indent(2)}kernel:${kLabel} size=${k.size} nargs=${k.nargs} outputs=${k.numOutputs}`,
          );
          for (let oi = 0; oi < k.outputs.length; oi++) {
            const o = k.outputs[oi];
            let line = `${indent(3)}output[${oi}]: ${o.dtype}`;
            if (o.hasReduction) line += ` REDUCTION ${o.reductionOp}`;
            emit(line);
          }
        }
      }
    }
    emit("");
  }

  // Shader code
  if (report.codeEntries.length > 0) {
    emit("Compiled Code:");
    for (let i = 0; i < report.codeEntries.length; i++) {
      const c = report.codeEntries[i];
      emit(
        `${indent(1)}[${i}] ${c.backend}/${c.kind}${c.label ? ` "${c.label}"` : ""}`,
      );
      if (c.workgroupSize) {
        emit(`${indent(2)}workgroupSize: [${c.workgroupSize}]`);
      }
      if (c.metadata) {
        const meta = Object.entries(c.metadata)
          .filter(([, v]) => v !== undefined)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
        if (meta) emit(`${indent(2)}metadata: ${meta}`);
      }
      if (c.code) {
        emit(`${indent(2)}--- code (${c.code.length} chars) ---`);
        // Indent each line of code
        for (const codeLine of c.code.split("\n")) {
          emit(`${indent(2)}| ${codeLine}`);
        }
        emit(`${indent(2)}--- end ---`);
      }
    }
    emit("");
  }

  // Full program listing
  emit("Full Program Listing:");
  for (const line of p.programString.split("\n")) {
    emit(`${indent(1)}${line}`);
  }

  return lines.join("\n");
}
