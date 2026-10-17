// Command tape types and eligibility check for WebGPU dispatch acceleration.
//
// A command tape pre-compiles a JitProgram's dispatch sequence into a flat
// representation with pre-resolved pipelines, pre-computed buffer indices,
// and pre-built uniform bind groups. This eliminates per-step JS overhead
// (scope lookups, array allocation, refcounting, pipeline cache lookups)
// while producing a tight command-encoding loop over a flat GPUBuffer[] table.
//
// Ops are stored in original step order so that frees return buffers to the
// pool *before* subsequent mallocs, reducing peak VRAM compared to the
// bulk-malloc-then-bulk-free approach.

import { DType, Kernel } from "../../alu";
import type { JitStep } from "../../frontend/jit";
import { isSymbolicSize } from "../../shape";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Pre-resolved representation of a single shader dispatch (one per ShaderDispatch). */
export interface TapeDispatch {
  /** Pre-compiled GPU pipeline. */
  pipeline: GPUComputePipeline;
  /** Un-compiled raw shader struct (kept for PendingCommandTape.prepare). */
  shader: import("./codegen.js").ShaderInfo;
  /** Bind group layout for storage bindings (@group(0)). */
  bindGroupLayout: GPUBindGroupLayout;
  /** Indices into the flat buffer table for inputs. */
  inputIdxs: number[];
  /** Indices into the flat buffer table for outputs. */
  outputIdxs: number[];
  /** Per-pass grid dimensions. */
  passes: { grid: [number, number] }[];
  /** Pre-built uniform bind group (@group(1)), null if no uniforms. */
  uniformBindGroup: GPUBindGroup | null;
  /** Uniform buffer alignment (for dynamic offset calculation). */
  uniformAlignment: number;
  /**
   * Mutable bind group cache (O9b).  When the same GPUBuffer objects appear
   * at the same table indices across invocations (common with arena slab
   * pooling), the cached bind group is reused — skipping createBindGroup().
   */
  _bgCache?: { key: GPUBuffer[]; value: GPUBindGroup };
}

/** Pre-resolved buffer allocation. */
export interface TapeMalloc {
  /** Index in the flat buffer table. */
  tableIdx: number;
  /** Buffer size in bytes (padded to 4-byte alignment). */
  paddedSize: number;
  /** Original (unpadded) byte size for output slot creation. */
  originalSize: number;
  /** Pre-filled constant data (for O2 scalar promotion). Null if no data. */
  initialData: Uint8Array<ArrayBuffer> | null;
  /**
   * If true, this malloc is pre-allocated from the constants slab (O9c).
   * The buffer/offset/bindSize are set up before the main execution loop.
   */
  slabAllocated?: boolean;
  /**
   * If true, this malloc is pre-allocated from a colored arena slab (O9a-v2).
   * The buffer/offset/bindSize are set up before the main execution loop.
   */
  arenaAllocated?: boolean;
}

/** Entry in the constants slab (O9c). */
export interface ConstSlabEntry {
  /** Index in the flat buffer table. */
  tableIdx: number;
  /** Byte offset within the slab buffer (256-byte aligned). */
  offset: number;
  /** Bind size for this entry (4-byte aligned, the actual padded data size). */
  bindSize: number;
  /** Original (unpadded) byte size for output slot creation. */
  originalSize: number;
}

/** Entry in a colored arena slab (O9a-v2). */
export interface ArenaSlabEntry {
  /** Index in the flat buffer table. */
  tableIdx: number;
  /** Byte offset within the slab buffer (256-byte aligned). */
  offset: number;
  /** Bind size for this entry (4-byte aligned, the actual padded data size). */
  bindSize: number;
  /** Original (unpadded) byte size for output slot creation. */
  originalSize: number;
}

/** A single colored arena slab (one per color). */
export interface ArenaSlab {
  /** The GPU buffer for this color group. */
  buffer: GPUBuffer;
  /** Per-entry layout within this slab. */
  entries: ArenaSlabEntry[];
}

/** Pre-resolved DUS (dynamic update slice) buffer copy. */
export interface TapeDUS {
  dstIdx: number;
  srcIdx: number;
  outIdx: number;
  /** Total byte size of the destination (= output size). */
  dstSizeBytes: number;
  /** Per-fiber byte offset into dst where src is written. */
  offsetBytes: number;
  /** Byte size of the src slice (axis=0 contiguous fast path). */
  sliceBytes: number;
  /** Number of outer fibers (1 for axis=0). */
  outerFibers: number;
  /** Bytes per fiber in src (stride between fibers). */
  srcFiberBytes: number;
  /** Bytes per fiber in dst (stride between fibers). */
  dstFiberBytes: number;
  /** Byte offset into src where fiber reads begin (default 0). */
  srcBaseOffset: number;
  /** ND outer extents for non-constant dst stride (optional). */
  ndOuterExtents?: number[];
  /** ND dst byte strides per outer dim (optional, same length as ndOuterExtents). */
  ndDstStrides?: number[];
}

/** Pre-resolved scatter_add: copy target → output, then dispatch atomic kernel. */
export interface TapeScatterAdd {
  targetIdx: number;
  indicesIdx: number;
  updatesIdx: number;
  outIdx: number;
  /** Pre-computed target byte size for the initial copy. */
  targetBytes: number;
  /** Pre-compiled scatter pipeline. */
  pipeline: GPUComputePipeline;
  /** Bind group layout for the 3-binding scatter shader. */
  bindGroupLayout: GPUBindGroupLayout;
  /** Grid dimensions for the scatter dispatch. */
  grid: [number, number];
}

/** Pre-resolved reverse: copy slices in reverse order. */
export interface TapeReverse {
  inputIdx: number;
  outIdx: number;
  /** Number of elements along the reversed axis (concrete). */
  axisSize: number;
  /** Byte size of each axis slice (innerBytes = prod(shape[axis+1:]) * elemBytes). */
  innerBytes: number;
}

/** Pre-resolved scan operation. */
export interface TapeScan {
  step: Extract<JitStep, { type: "scan" }>;
  constIdxs: number[];
  initCarryIdxs: number[];
  xsIdxs: number[];
  carryOutIdxs: number[];
  ysStackedIdxs: number[];
}

/** A single tape operation in execution order. */
export type TapeOp =
  | { type: "malloc"; malloc: TapeMalloc }
  | { type: "free"; tableIdx: number }
  | { type: "recycle"; fromIdx: number; toIdx: number }
  | { type: "dispatch"; dispatch: TapeDispatch }
  | { type: "dus"; dus: TapeDUS }
  | { type: "scatter_add"; scatterAdd: TapeScatterAdd }
  | { type: "reverse"; reverse: TapeReverse }
  | { type: "scan"; scan: TapeScan };

/** A pre-compiled dispatch sequence for a kernel-only JitProgram. */
export interface WebGPUCommandTape {
  /** Operations in original step order (malloc/free/recycle/dispatch). */
  ops: TapeOp[];
  /** Number of entries in the flat buffer table. */
  tableSize: number;
  /** Mapping: external input position → table index. */
  inputTableIdxs: number[];
  /** Mapping: external output position → table index. */
  outputTableIdxs: number[];
  /** Sizes of the outputs */
  outputSizes: number[];
  /** Table indices that own allocated buffers (for cleanup on error). */
  allocatedIdxs: number[];
  /** Uniform buffers owned by this tape (kept alive for bind group references). */
  uniformBuffers: GPUBuffer[];
  /**
   * Constants slab (O9c): a single GPU buffer holding all initialData
   * constants packed with 256-byte alignment. Null if no constants.
   * Persists across invocations — constants are written once at compile time.
   */
  constSlab: { buffer: GPUBuffer; entries: ConstSlabEntry[] } | null;
  /**
   * Colored arena slabs (O9a-v2): one GPUBuffer per conflict-graph color.
   * Internal intermediates are pre-assigned to slab regions with 256-byte
   * alignment. Persists across invocations for stable bind group caching.
   * Null if no internal intermediates are eligible (e.g., all external).
   */
  arenaSlabs: ArenaSlab[] | null;
}

// ---------------------------------------------------------------------------
// Conflict graph coloring (O9a-v2 prototype)
// ---------------------------------------------------------------------------

/**
 * Result of conflict-graph analysis and greedy coloring.
 *
 * Each internal table index is assigned a color. Indices sharing a color can
 * safely reside in the same GPUBuffer slab because they never appear in
 * conflicting binding types (read-only-storage vs storage) within a single
 * dispatch.
 */
export interface ColoringResult {
  /** color[tableIdx] = color number (0-based), or -1 for external/uncolored. */
  colors: Int32Array;
  /** Number of distinct colors used. */
  numColors: number;
  /** Per-color list of table indices assigned to that color. */
  colorGroups: number[][];
}

/**
 * Build a conflict graph from a command tape and greedy-color it.
 *
 * **Conflict rules** (from WebGPU spec, buffer-identity-level validation):
 * - Outputs conflict with all inputs in the same dispatch (storage vs
 *   read-only-storage on the same buffer is a validation error).
 * - Outputs conflict with other outputs in the same dispatch (storage vs
 *   storage on the same buffer is a validation error).
 * - Inputs do NOT conflict with each other (read-only-storage + read-only-
 *   storage is valid).
 *
 * External inputs/outputs and slab-allocated constants are excluded from
 * coloring (they have fixed buffer assignments).
 *
 * @returns Coloring result with per-index color assignment.
 */
export function buildConflictGraphAndColor(
  tape: WebGPUCommandTape,
): ColoringResult {
  const n = tape.tableSize;
  const colors = new Int32Array(n).fill(-1);

  // Determine which indices are internal (eligible for arena allocation).
  // External inputs, external outputs, and slab-allocated constants are excluded.
  const externalSet = new Set<number>();
  for (const idx of tape.inputTableIdxs) externalSet.add(idx);
  for (const idx of tape.outputTableIdxs) externalSet.add(idx);
  if (tape.constSlab) {
    for (const e of tape.constSlab.entries) externalSet.add(e.tableIdx);
  }

  // Exclude recycle participants from arena allocation entirely.
  // A recycle replaces free(a) → malloc(b) with buffer sharing. If fromIdx
  // is arena-allocated, toIdx would inherit its slab buffer but won't have
  // its own arena entry — leading to buffer identity conflicts. Rather than
  // tracking this inheritance, we exclude all recycle ends from coloring.
  for (const op of tape.ops) {
    if (op.type === "recycle") {
      externalSet.add(op.fromIdx);
      externalSet.add(op.toIdx);
    } else if (op.type === "scan") {
      for (const idx of op.scan.constIdxs) externalSet.add(idx);
      for (const idx of op.scan.initCarryIdxs) externalSet.add(idx);
      for (const idx of op.scan.xsIdxs) externalSet.add(idx);
      for (const idx of op.scan.carryOutIdxs) externalSet.add(idx);
      for (const idx of op.scan.ysStackedIdxs) externalSet.add(idx);
    }
  }

  // Build adjacency lists (conflict graph).
  // Only track edges for internal indices.
  const adj: Set<number>[] = new Array(n);
  for (let i = 0; i < n; i++) adj[i] = new Set();

  for (const op of tape.ops) {
    if (op.type === "dispatch") {
      const d = op.dispatch;
      const outIdxs = d.outputIdxs;
      const inIdxs = d.inputIdxs;

      // Each output conflicts with every input in the same dispatch.
      for (const oi of outIdxs) {
        if (externalSet.has(oi)) continue;
        for (const ii of inIdxs) {
          if (ii === oi || externalSet.has(ii)) continue;
          adj[oi].add(ii);
          adj[ii].add(oi);
        }
      }
      // Each output conflicts with every other output in the same dispatch.
      for (let a = 0; a < outIdxs.length; a++) {
        if (externalSet.has(outIdxs[a])) continue;
        for (let b = a + 1; b < outIdxs.length; b++) {
          if (externalSet.has(outIdxs[b])) continue;
          adj[outIdxs[a]].add(outIdxs[b]);
          adj[outIdxs[b]].add(outIdxs[a]);
        }
      }
    } else if (op.type === "scatter_add") {
      // scatter_add dispatch: output (storage rw) conflicts with inputs (ro).
      const sa = op.scatterAdd;
      const outIdx = sa.outIdx;
      const inIdxs = [sa.indicesIdx, sa.updatesIdx];
      if (!externalSet.has(outIdx)) {
        for (const ii of inIdxs) {
          if (ii === outIdx || externalSet.has(ii)) continue;
          adj[outIdx].add(ii);
          adj[ii].add(outIdx);
        }
      }
    }
  }

  // Greedy coloring: process nodes by degree (most-constrained first).
  const internal: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!externalSet.has(i)) internal.push(i);
  }
  // Sort descending by degree (number of conflicts).
  internal.sort((a, b) => adj[b].size - adj[a].size);

  let numColors = 0;
  for (const idx of internal) {
    // Find the smallest color not used by any neighbor.
    const usedColors = new Set<number>();
    for (const neighbor of adj[idx]) {
      if (colors[neighbor] >= 0) usedColors.add(colors[neighbor]);
    }
    let c = 0;
    while (usedColors.has(c)) c++;
    colors[idx] = c;
    if (c >= numColors) numColors = c + 1;
  }

  // Build color groups.
  const colorGroups: number[][] = new Array(numColors);
  for (let c = 0; c < numColors; c++) colorGroups[c] = [];
  for (const idx of internal) {
    colorGroups[colors[idx]].push(idx);
  }

  return { colors, numColors, colorGroups };
}

// ---------------------------------------------------------------------------
// Eligibility check
// ---------------------------------------------------------------------------

/**
 * Check whether a JitProgram's steps can be compiled to a WebGPU command tape.
 *
 * Returns true if all steps are supported: execute (Kernel or Routine),
 * malloc (concrete size), free, recycle, DUS (concrete sizes), scatter_add
 * (non-f64), and reverse (concrete axis size). Unaligned copies use the WGSL
 * copy shader via `#encodeCopyAuto`. Programs with scan, block_map, or other
 * complex steps fall back to step-by-step.
 */
export function canCompileToCommandTape(steps: JitStep[]): boolean {
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        if (isSymbolicSize(step.size)) return false;
        break;
      case "free":
      case "recycle":
        break;
      case "execute":
        // Reject symbolic kernels — need runtime-resolved grid + uniform
        if (step.source instanceof Kernel && step.source.needsDynamicParams)
          return false;
        break;
      case "dus":
        // Reject symbolic sizes
        if (
          isSymbolicSize(step.sliceBytes) ||
          isSymbolicSize(step.dstSizeBytes)
        )
          return false;
        break;
      case "scatter_add":
        // Float64 not supported on WebGPU
        if (step.dtype === DType.Float64) return false;
        break;
      case "reverse":
        // Reject symbolic axis size or total bytes
        if (
          typeof step.axisSize !== "number" ||
          isSymbolicSize(step.totalBytes)
        )
          return false;
        break;
      case "scan":
        if (step.plan.path === "fallback") return false;
        // WebGPU paths are supported
        break;
      case "incref":
      case "assoc_scan":
      case "block_map":
      case "fori_loop":
      case "workgroup_assoc_scan":
        return false;
      default:
        return false;
    }
  }
  return true;
}
