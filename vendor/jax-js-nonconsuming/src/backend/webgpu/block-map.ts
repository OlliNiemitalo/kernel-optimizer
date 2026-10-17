/**
 * @file BlockMap fused shader compiler for WebGPU.
 *
 * Compiles kernel-only block_map bodies into single WGSL compute shaders
 * where each workgroup processes one block using `var<workgroup>` shared memory.
 *
 * Architecture:
 *   - Body JitProgram's malloc steps → var<workgroup> arrays
 *   - Body JitProgram's execute steps → inline WGSL via gen()
 *   - workgroupBarrier() between data-dependent steps
 *   - Thread i processes element i within the block
 *   - Block index from workgroup_id
 *
 * Returns null (→ fallback) if:
 *   - Body contains routine steps (Sort, Cholesky, etc.)
 *   - Shared memory exceeds device.limits.maxComputeWorkgroupStorageSize
 *   - prod(blockShape) exceeds device.limits.maxComputeInvocationsPerWorkgroup
 *   - Symbolic malloc sizes
 *   - More storage bindings than maxStorageBuffersPerShaderStage
 */

import { erfSrc, threefrySrc } from "./builtins";
import {
  calculateGrid,
  constToWgsl,
  dtypeToWgsl,
  type PackedLeafLayout,
  type ShaderInfo,
  withNanInfHeader,
} from "./codegen";
import {
  createWgslGen,
  type GlobalIndexRemapInfo,
  type ResolveGlobalIndex,
  wgslCastReductionRhs,
  wgslReductionAccumStmt,
} from "./wgsl-gen";
import {
  AluExp,
  AluOp,
  byteWidth,
  detectScalarAssocOp,
  DType,
  Kernel,
  Reduction,
} from "../../alu";
import type { JitId, JitProgram, JitStep } from "../../frontend/jit";
import { Routine } from "../../routine";
import { concreteDim, isSymbolicSize } from "../../shape";
import { DEBUG, mapSetUnion, prod, strip1 } from "../../utils";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export type { PackedLeafLayout } from "./codegen";

export interface BlockMapShaderParams {
  /** The compiled body program (its steps are analyzed). */
  bodyProgram: JitProgram;
  /** Block dimensions (always concrete numbers). */
  blockShape: number[];
  /** Grid dimensions (number of blocks per axis). */
  gridShape: number[];
  /** How body inputs map to grid axes. Per-input: axis index per grid dim, or null. */
  inAxes: (number | null)[][];
  /** How body outputs map to grid axes. Per-output: axis index per grid dim, or null. */
  outAxes: (number | null)[][];
  /** Number of constant inputs (first numConsts body inputs are consts). */
  numConsts: number;
  /** Number of non-const inputs. */
  numInputs: number;
  /** Original (untrimmed) input shapes. */
  inputShapes: number[][];
  /** Original (untrimmed) output shapes. */
  outputShapes: number[][];
  /** Register tiling dimensions. When set, each thread handles threadTile[g] outputs per axis. */
  threadTile?: number[];
  /** Body uses BlockIndex: compiled body has one extra input (block index scalar). */
  hasBlockIndex?: boolean;
  /** Per-constant info for uniform buffer migration (one entry per numConsts input). */
  constInfos?: { elemCount: number; dtype: DType; bytes: number }[];
  /**
   * Point-mode inputs: one element per grid point, indexed by workgroup_id.
   * Body sees these as broadcast (no block dimension), but each workgroup
   * reads a different element. inputShapes for point inputs should be the
   * per-element shape (excluding the grid-indexed dimension).
   */
  pointInputs?: boolean[];
  /**
   * Grid offset: shift mapped input/output base offsets by this many blocks.
   * Used when the block_map operates on a sub-range of the data (e.g.,
   * assocScan Phase 4 starts from block 1, not block 0).
   */
  gridOffset?: number;
  /**
   * Request leaf packing — when set, the shader codegen computes a
   * PackedLeafLayout from inputShapes/outputShapes and emits packed storage
   * bindings.  The computed layout is returned in ShaderInfo.packedLeafLayout.
   */
  needsLeafPacking?: boolean;
  /**
   * Per-input per-grid-axis overlap [lo, hi].
   * When present, bodies see expanded tiles (blockShape[g] + lo + hi) and
   * global reads that fall before index 0 or past the input dimension return
   * zero. inputShapes must contain ORIGINAL (unpadded) dimensions.
   */
  halo?: [number, number][][];
}

/**
 * Try to compile a block_map body into a fused WGSL compute shader.
 * Returns null if the body is not eligible for fusion.
 */
export function blockMapFusedShaderSource(
  device: GPUDevice,
  params: BlockMapShaderParams,
  _caps?: unknown,
): ShaderInfo | null {
  const {
    bodyProgram,
    blockShape,
    gridShape,
    numConsts,
    numInputs,
    threadTile,
    needsLeafPacking,
  } = params;
  const steps = bodyProgram.steps;

  // --- Register tiling: compute actual workgroup size ---
  // With threadTile, each thread handles threadTile[g] outputs along axis g.
  // wgShape[g] = blockShape[g] / threadTile[g], total threads = prod(wgShape).
  const wgShape = threadTile
    ? blockShape.map((bs, g) => bs / threadTile[g])
    : blockShape;
  const blockSize = prod(blockShape);
  const numThreads = prod(wgShape);

  // --- Guard: numThreads ≤ maxComputeInvocationsPerWorkgroup ---
  const maxInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  if (numThreads > maxInvocations) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: numThreads ${numThreads} > maxInvocations ${maxInvocations}, fallback`,
      );
    return null;
  }

  // --- Detect boundary blocks (non-divisible dimensions) ---
  // When dimensions are not evenly divisible by blockShape, the last block
  // along each axis has some invalid (out-of-bounds) threads. We emit a
  // per-thread `valid` flag and guard all global reads/writes.
  const gridRank = blockShape.length;

  // O11: Extract the tile inner (column) dimension from a GlobalView's last
  // index, which contains `gidx % cols` from the DynamicSlice unravel pattern.
  const findModBase = (exp: AluExp): number | null => {
    if (exp.op === AluOp.Mod && exp.src[1].op === AluOp.Const) {
      return exp.src[1].arg as number;
    }
    for (const s of exp.src) {
      const r = findModBase(s);
      if (r !== null) return r;
    }
    return null;
  };
  const extractTileInnerDim = (exp: AluExp): number | null => {
    if (exp.op === AluOp.GlobalView && exp.src.length >= 2) {
      return findModBase(exp.src[exp.src.length - 1]);
    }
    for (const s of exp.src) {
      const r = extractTileInnerDim(s);
      if (r !== null) return r;
    }
    return null;
  };
  // Per-axis: the original dimension along each grid axis (for validity checks)
  const axisDims: (number | null)[] = new Array(gridRank).fill(null);
  let hasBoundary = false;
  const gridOffset = params.gridOffset ?? 0;
  for (let g = 0; g < gridRank; g++) {
    for (let i = 0; i < numInputs; i++) {
      // Point inputs don't participate in axisDims/boundary detection — they
      // have one element per grid point, not blockShape elements.
      if (params.pointInputs?.[i]) continue;
      const axes = params.inAxes[i];
      if (axes[g] !== null) {
        const dim = params.inputShapes[i][axes[g]!];
        axisDims[g] = dim;
        if (dim % blockShape[g] !== 0) hasBoundary = true;
      }
    }
  }

  // --- Guard: all execute steps must use Kernel (no Routines) ---
  // --- Classify steps and compute shared memory budget ---

  /** Map from JitId → shared memory info for intermediates. */
  const shmemMap = new Map<
    JitId,
    { sizeBytes: number; dtype: DType; elemCount: number }
  >();
  /** Recycle aliases: recycled output → original malloc input. */
  const recycleAliases = new Map<JitId, JitId>();
  /** Steps that produce outputs (execute with Kernel source). */
  const kernelSteps: {
    step: Extract<JitStep, { type: "execute" }>;
    kernel: Kernel;
  }[] = [];
  /** Per-step reduction info (null if the step is elementwise). */
  const stepReductions: (Reduction | null)[] = [];

  /** Analyzed fori_loop steps. */
  interface ForiLoopInfo {
    foriStep: Extract<JitStep, { type: "fori_loop" }>;
    bodyKernels: {
      step: Extract<JitStep, { type: "execute" }>;
      kernel: Kernel;
    }[];
    bodyBarriers: Set<number>;
    /** P1: whether wrap-around dependencies require a barrier at end of loop body. */
    needWrapBarrier: boolean;
    bodyShmemMap: Map<JitId, { name: string; dtype: DType; elemCount: number }>;
    bodyShmemIds: Set<JitId>;
    /** P0a: size-1 kernels promoted from shmem to `let` bindings. */
    promotedScalars: Map<JitId, { name: string; dtype: DType }>;
    /**
     * O4: Body shmem buffers stored in `var<private>` arrays (per-thread registers)
     * instead of `var<workgroup>`. When threadTile is set, carry outputs and any
     * intermediate whose ALL consumers are also private get promoted to registers.
     * Each thread holds threadTile[0]*threadTile[1] elements per private buffer.
     */
    privateShmemIds: Set<JitId>;
    /**
     * O11: Bank padding for cooperative-loaded shmem arrays. Maps JitId → inner
     * (last) dimension of the 2D tile. Padded stride = innerDim + 1 eliminates
     * bank conflicts when multiple rows map to the same bank.
     */
    shmemBankPad: Map<JitId, number>;
    bodyInputIds: JitId[];
    bodyOutputIds: JitId[];
    numConsts: number;
    loopVar: string;
  }
  const foriLoops: ForiLoopInfo[] = [];

  /**
   * Analyzed workgroup_assoc_scan steps.
   *
   * NOTE: Scalar promotion (P0a) is not implemented for workgroup_assoc_scan.
   * The binary operator body kernels always take paired elements, so size-1
   * non-reduction kernels don't arise in practice. If future code paths need
   * this, add `promotedScalars` and `isScalar` mirroring ForiLoopInfo.
   */
  interface WorkgroupAssocScanInfo {
    wasStep: Extract<JitStep, { type: "workgroup_assoc_scan" }>;
    /** Body kernels (the binary operator fn). */
    bodyKernels: {
      step: Extract<JitStep, { type: "execute" }>;
      kernel: Kernel;
    }[];
    /** Body shmem intermediates. */
    bodyShmemMap: Map<JitId, { name: string; dtype: DType; elemCount: number }>;
    bodyShmemIds: Set<JitId>;
    bodyInputIds: JitId[];
    bodyOutputIds: JitId[];
    numConsts: number;
    numElems: number;
    /** Per-element dtype and count. */
    elemDtypes: DType[];
    elemCounts: number[];
    /** Ping/pong shmem names per elem: [pingName, pongName]. */
    pingPongNames: [string, string][];
    numRounds: number;
    /**
     * When the body is a scalar binary Add or Mul (single kernel, single elem,
     * elemCount=1, no reduction), this stores the op so codegen can emit
     * `subgroupInclusiveAdd` / `subgroupInclusiveMul` instead of iterated
     * Kogge-Stone rounds.
     */
    scalarOp?: AluOp.Add | AluOp.Mul;
  }
  const workgroupAssocScans: WorkgroupAssocScanInfo[] = [];

  /** Combined codegen entries in step order. */
  type CodegenEntry =
    | { type: "kernel"; kernelIdx: number }
    | { type: "fori_loop"; flIdx: number }
    | { type: "workgroup_assoc_scan"; wasIdx: number };
  const codegenEntries: CodegenEntry[] = [];

  let totalShmemBytes = 0;

  for (const step of steps) {
    switch (step.type) {
      case "execute": {
        if (step.source instanceof Routine) {
          if (DEBUG >= 1)
            console.info("block_map fused: routine in body, fallback");
          return null;
        }
        const kernel = step.source as Kernel;
        // Reject symbolic kernel sizes
        if (kernel.isSymbolic) {
          if (DEBUG >= 1)
            console.info("block_map fused: symbolic kernel size, fallback");
          return null;
        }
        // Multi-output reductions not supported in fused shader
        if (kernel.hasReduction && kernel.numOutputs > 1) {
          if (DEBUG >= 1)
            console.info("block_map fused: multi-output reduction, fallback");
          return null;
        }
        // Kernel size must be 1 (workgroup reduction) or a multiple of blockSize
        {
          const ks = kernel.size as number;
          if (ks !== 1 && ks % blockSize !== 0) {
            if (DEBUG >= 1)
              console.info(
                `block_map fused: kernel.size ${ks} not multiple of blockSize ${blockSize}, fallback`,
              );
            return null;
          }
        }
        const re = kernel.outputs[0]?.reduction ?? null;
        codegenEntries.push({ type: "kernel", kernelIdx: kernelSteps.length });
        kernelSteps.push({ step, kernel });
        stepReductions.push(re);
        break;
      }
      case "malloc": {
        if (isSymbolicSize(step.size)) {
          if (DEBUG >= 1)
            console.info("block_map fused: symbolic malloc size, fallback");
          return null;
        }
        const sizeBytes = step.size as number;
        shmemMap.set(step.output, {
          sizeBytes,
          dtype: DType.Float32,
          elemCount: sizeBytes / 4,
        });
        totalShmemBytes += sizeBytes;
        break;
      }
      case "free":
      case "incref":
        break;
      case "recycle": {
        // Alias the recycled output to the same shmem as the input.
        // Don't add to shmemMap (which drives var<workgroup> declarations) —
        // just record the alias so idToReadName maps both IDs to the same name.
        recycleAliases.set(step.output, step.input);
        break;
      }
      case "fori_loop": {
        // Analyze fori_loop body: must be kernel-only, elementwise-only
        const flIdx = foriLoops.length;
        const bodyProg = step.bodyProgram;
        const bodySteps = bodyProg.steps;
        const bodyKernels: ForiLoopInfo["bodyKernels"] = [];
        const bodyShmemMap = new Map<
          JitId,
          { name: string; dtype: DType; elemCount: number }
        >();
        const bodyShmemIds = new Set<JitId>();

        let valid = true;
        for (const bs of bodySteps) {
          switch (bs.type) {
            case "execute": {
              if (bs.source instanceof Routine) {
                if (DEBUG >= 1)
                  console.info(
                    "block_map fused: routine in fori_loop body, fallback",
                  );
                return null;
              }
              const bk = bs.source as Kernel;
              if (bk.isSymbolic) {
                valid = false;
                break;
              }
              // Reductions: allow per-thread contractions (kernel.size == blockSize)
              // but reject workgroup-level reductions (different thread counts).
              if (bk.hasReduction) {
                const re = bk.outputs[0]?.reduction;
                if (
                  !re ||
                  bk.numOutputs > 1 ||
                  (bk.size as number) !== blockSize
                ) {
                  if (DEBUG >= 1)
                    console.info(
                      "block_map fused: unsupported reduction in fori_loop body, fallback",
                    );
                  return null;
                }
                // Per-thread contraction: OK — each thread accumulates privately
              }
              // Body kernels larger than blockSize can't be processed 1:1
              if ((bk.size as number) > blockSize) {
                if (DEBUG >= 1)
                  console.info(
                    `block_map fused: fori_loop body kernel size ${bk.size} > blockSize ${blockSize}, fallback`,
                  );
                return null;
              }
              bodyKernels.push({
                step: bs as Extract<JitStep, { type: "execute" }>,
                kernel: bk,
              });
              break;
            }
            case "malloc": {
              if (isSymbolicSize(bs.size)) {
                valid = false;
                break;
              }
              const sz = bs.size as number;
              const sname = `fl${flIdx}_s${bs.output}`;
              bodyShmemMap.set(bs.output, {
                name: sname,
                dtype: DType.Float32,
                elemCount: sz / 4,
              });
              bodyShmemIds.add(bs.output);
              totalShmemBytes += sz;
              break;
            }
            case "free":
            case "incref":
              break;
            case "recycle": {
              const bExisting = bodyShmemMap.get(bs.input);
              if (bExisting) {
                bodyShmemMap.set(bs.output, bExisting);
                bodyShmemIds.add(bs.output);
              }
              break;
            }
            default:
              if (DEBUG >= 1)
                console.info(
                  `block_map fused: unsupported step in fori_loop body "${(bs as JitStep).type}", fallback`,
                );
              return null;
          }
          if (!valid) break;
        }
        if (!valid) {
          if (DEBUG >= 1)
            console.info("block_map fused: invalid fori_loop body, fallback");
          return null;
        }

        // Refine body shmem dtypes from body kernel outputs
        for (const { step: bs, kernel: bk } of bodyKernels) {
          for (let oi = 0; oi < bk.numOutputs; oi++) {
            const entry = bodyShmemMap.get(bs.outputs[oi]);
            if (entry) {
              entry.dtype = bk.outputs[oi].dtype;
              entry.elemCount = (entry.elemCount * 4) / byteWidth(entry.dtype);
            }
          }
        }

        // P0a: Scalar promotion — detect size-1 kernels that can be promoted
        // from shmem to `let` bindings. This eliminates the shmem allocation,
        // the `if (tidx < 1u)` guard, and the barrier.
        const promotedScalars = new Map<
          JitId,
          { name: string; dtype: DType }
        >();
        const bodyOutputSet = new Set(bodyProg.outputs);
        for (const { step: bs, kernel: bk } of bodyKernels) {
          if ((bk.size as number) !== 1 || bk.numOutputs !== 1) continue;
          if (bk.hasReduction) continue;
          const outId = bs.outputs[0];
          // Don't promote carry outputs — they must persist across iterations
          if (bodyOutputSet.has(outId)) continue;
          const shmemEntry = bodyShmemMap.get(outId);
          if (!shmemEntry) continue;
          const dtype = bk.outputs[0].dtype;
          promotedScalars.set(outId, {
            name: `fl${flIdx}_let_${outId}`,
            dtype,
          });
          // Remove from shmem tracking
          const removedBytes =
            shmemEntry.elemCount * byteWidth(shmemEntry.dtype);
          totalShmemBytes -= removedBytes;
          bodyShmemMap.delete(outId);
          bodyShmemIds.delete(outId);
        }

        // P1: Phase-based barrier scheduling
        // Build phases: maximal groups of consecutive steps where no step
        // reads a shmem location written by an earlier step in the same phase.
        // A barrier is needed at each phase boundary.
        const bStepWrites = new Map<number, Set<JitId>>();
        const bStepReads = new Map<number, Set<JitId>>();
        for (let bsi = 0; bsi < bodyKernels.length; bsi++) {
          const { step: bs } = bodyKernels[bsi];
          const w = new Set<JitId>();
          const r = new Set<JitId>();
          for (const oid of bs.outputs) {
            if (bodyShmemIds.has(oid)) w.add(oid);
          }
          for (const iid of bs.inputs) {
            if (bodyShmemIds.has(iid)) r.add(iid);
          }
          bStepWrites.set(bsi, w);
          bStepReads.set(bsi, r);
        }

        const phases: {
          startIdx: number;
          writes: Set<JitId>;
          reads: Set<JitId>;
        }[] = [];
        let curPhase = {
          startIdx: 0,
          writes: new Set<JitId>(),
          reads: new Set<JitId>(),
        };
        for (let bsi = 0; bsi < bodyKernels.length; bsi++) {
          const writes = bStepWrites.get(bsi)!;
          const reads = bStepReads.get(bsi)!;
          // RAW: does this step read something the current phase wrote?
          let conflict = false;
          for (const id of reads) {
            if (curPhase.writes.has(id)) {
              conflict = true;
              break;
            }
          }
          if (conflict) {
            phases.push(curPhase);
            curPhase = {
              startIdx: bsi,
              writes: new Set<JitId>(),
              reads: new Set<JitId>(),
            };
          }
          for (const id of writes) curPhase.writes.add(id);
          for (const id of reads) curPhase.reads.add(id);
        }
        phases.push(curPhase);

        // Barriers at phase boundaries (before each phase except the first)
        const bodyBarriers = new Set<number>();
        for (let pi = 1; pi < phases.length; pi++) {
          bodyBarriers.add(phases[pi].startIdx);
        }

        // Wrap-around: check cross-iteration dependencies between last and
        // first phases. RAW (first reads what last wrote) and WAR (last reads
        // what first writes in the next iteration) both require a barrier at
        // the end of the loop body.
        let needWrapBarrier = false;
        if (phases.length >= 2) {
          const firstPhase = phases[0];
          const lastPhase = phases[phases.length - 1];
          for (const id of firstPhase.reads) {
            if (lastPhase.writes.has(id)) {
              needWrapBarrier = true;
              break;
            }
          }
          if (!needWrapBarrier) {
            for (const id of lastPhase.reads) {
              if (firstPhase.writes.has(id)) {
                needWrapBarrier = true;
                break;
              }
            }
          }
        }

        // O4: Private buffer classification for register tiling.
        // When threadTile is set, identify body shmem buffers that can live in
        // per-thread private registers instead of workgroup shared memory.
        // Seed: carry outputs (bodyProg.outputs) are always private — each
        // thread owns threadTile[0]*threadTile[1] carry elements in registers.
        // Propagate: if ALL consumers of a body shmem buffer are themselves
        // private steps (all their outputs are private), promote that buffer
        // to private too. This chains through the dependency graph, e.g.
        // matmul: dot→prod (private) → add→carry (private), while tile loads
        // (shared by all threads) remain cooperative.
        const privateShmemIds = new Set<JitId>();
        if (threadTile) {
          // Seed: all carry output JitIds (body program outputs)
          for (const outId of bodyProg.outputs) {
            privateShmemIds.add(outId);
          }
          // Build a map: bodyJitId → list of kernel indices that read it
          const consumersOf = new Map<JitId, number[]>();
          for (let bsi = 0; bsi < bodyKernels.length; bsi++) {
            const { step: bs } = bodyKernels[bsi];
            for (const inId of bs.inputs) {
              if (bodyShmemIds.has(inId) || bodyOutputSet.has(inId)) {
                let list = consumersOf.get(inId);
                if (!list) {
                  list = [];
                  consumersOf.set(inId, list);
                }
                list.push(bsi);
              }
            }
          }
          // Propagate backward: mark a buffer as private if all its consumer
          // kernels have ALL their outputs in the private set.
          // CONSTRAINT: Do NOT privatize a buffer consumed by a reduction kernel.
          // Reductions access inputs at ridx-dependent positions that span the
          // entire tile buffer, not just this thread's tileElems positions.
          let changed = true;
          while (changed) {
            changed = false;
            for (const [bodyJitId] of bodyShmemMap) {
              if (privateShmemIds.has(bodyJitId)) continue;
              if (promotedScalars.has(bodyJitId)) continue;
              const consumers = consumersOf.get(bodyJitId);
              if (!consumers || consumers.length === 0) continue;
              let allPrivate = true;
              for (const ci of consumers) {
                const { step: cs, kernel: ck } = bodyKernels[ci];
                // Reduction kernels read inputs at ridx-dependent addresses —
                // the input must remain in shared memory.
                if (ck.hasReduction) {
                  allPrivate = false;
                  break;
                }
                for (const oid of cs.outputs) {
                  if (!privateShmemIds.has(oid) && !promotedScalars.has(oid)) {
                    allPrivate = false;
                    break;
                  }
                }
                if (!allPrivate) break;
              }
              if (allPrivate) {
                privateShmemIds.add(bodyJitId);
                changed = true;
              }
            }
          }
        }

        // O11: Identify cooperative-loaded shmem arrays eligible for bank padding.
        // For each cooperative body step output stored in shmem, extract the 2D
        // tile's inner dimension from the kernel expression's GlobalView.
        const shmemBankPad = new Map<JitId, number>();
        if (threadTile) {
          for (const { step: bs, kernel: bk } of bodyKernels) {
            const isSinglePromoted =
              bk.numOutputs === 1 && promotedScalars.has(bs.outputs[0]);
            const allPrivate =
              !isSinglePromoted &&
              bs.outputs.every(
                (oid) => privateShmemIds.has(oid) || promotedScalars.has(oid),
              );
            if (isSinglePromoted || allPrivate) continue;
            // Cooperative step — check outputs for 2D shmem tiles
            for (let oi = 0; oi < bk.numOutputs; oi++) {
              const outId = bs.outputs[oi];
              if (!bodyShmemMap.has(outId)) continue;
              const innerDim = extractTileInnerDim(bk.outputs[oi].exp);
              if (innerDim !== null && innerDim > 1) {
                shmemBankPad.set(outId, innerDim);
              }
            }
          }
        }

        foriLoops.push({
          foriStep: step as Extract<JitStep, { type: "fori_loop" }>,
          bodyKernels,
          bodyBarriers,
          needWrapBarrier,
          bodyShmemMap,
          bodyShmemIds,
          promotedScalars,
          privateShmemIds,
          shmemBankPad,
          bodyInputIds: bodyProg.inputs,
          bodyOutputIds: bodyProg.outputs,
          numConsts: step.numConsts,
          loopVar: `fl${flIdx}_i`,
        });
        codegenEntries.push({ type: "fori_loop", flIdx });
        break;
      }
      case "workgroup_assoc_scan": {
        // Analyze scan body: must be kernel-only, elementwise-only
        const wasIdx = workgroupAssocScans.length;
        const bodyProg = step.bodyProgram;
        const bodySteps = bodyProg.steps;
        const bodyKernels: WorkgroupAssocScanInfo["bodyKernels"] = [];
        const bodyShmemMap = new Map<
          JitId,
          { name: string; dtype: DType; elemCount: number }
        >();
        const bodyShmemIds = new Set<JitId>();

        // Pre-scan: collect byte sizes of every JitId in the body so we can
        // handle recycle steps whose input is a body input (not in bodyShmemMap).
        // Recycle replaces free+malloc; when the freed input is a body program
        // input (const/elem), bodyShmemMap won't have it, but we still need
        // a shmem entry for the recycled output.
        const bodyIdBytes = new Map<JitId, number>();
        // Body inputs: consts get sizes from parent shmemMap; elems from elemAvals
        for (let bi = 0; bi < bodyProg.inputs.length; bi++) {
          const bInId = bodyProg.inputs[bi];
          if (bi < step.numConsts) {
            const parentId = step.consts[bi];
            const parentInfo = shmemMap.get(parentId);
            if (parentInfo) {
              bodyIdBytes.set(bInId, parentInfo.sizeBytes);
            }
          } else {
            // a-elem or b-elem: index wraps around numElems
            const elemIdx = (bi - step.numConsts) % step.numElems;
            const aval = step.elemAvals[elemIdx];
            bodyIdBytes.set(
              bInId,
              prod(aval.shape as number[]) * byteWidth(aval.dtype),
            );
          }
        }
        // Pass 1: collect known sizes from mallocs and kernel outputs
        for (const bs2 of bodySteps) {
          if (bs2.type === "malloc" && typeof bs2.size === "number") {
            bodyIdBytes.set(bs2.output, bs2.size);
          } else if (
            bs2.type === "execute" &&
            !(bs2.source instanceof Routine)
          ) {
            const bk = bs2.source as Kernel;
            for (let oi = 0; oi < bk.numOutputs; oi++) {
              const b = bk.outputs[oi].bytes;
              if (typeof b === "number") bodyIdBytes.set(bs2.outputs[oi], b);
            }
          }
        }
        // Pass 2: propagate sizes through recycle aliases
        for (const bs2 of bodySteps) {
          if (bs2.type === "recycle") {
            const sz =
              bodyIdBytes.get(bs2.input) ?? bodyIdBytes.get(bs2.output);
            if (sz != null) {
              bodyIdBytes.set(bs2.input, sz);
              bodyIdBytes.set(bs2.output, sz);
            }
          }
        }

        let valid = true;
        for (const bs of bodySteps) {
          switch (bs.type) {
            case "execute": {
              if (bs.source instanceof Routine) {
                if (DEBUG >= 1)
                  console.info(
                    "block_map fused: routine in workgroup_assoc_scan body, fallback",
                  );
                return null;
              }
              const bk = bs.source as Kernel;
              if (bk.isSymbolic) {
                valid = false;
                break;
              }
              // Allow reduction kernels (e.g., matmul in DLM bodies)
              // but reject multi-output reductions and symbolic reduction sizes.
              // Multi-output rejection is required: the reduction's `for (var gidx`
              // loop would shadow the elementwise output's implicit gidx, and
              // the two outputs need different iteration structures.
              if (bk.hasReduction) {
                const re = bk.outputs[0]?.reduction;
                if (!re || isSymbolicSize(re.size) || bk.numOutputs > 1) {
                  valid = false;
                  break;
                }
              }
              bodyKernels.push({
                step: bs as Extract<JitStep, { type: "execute" }>,
                kernel: bk,
              });
              break;
            }
            case "malloc": {
              if (isSymbolicSize(bs.size)) {
                valid = false;
                break;
              }
              const sz = bs.size as number;
              const sname = `was${wasIdx}_s${bs.output}`;
              bodyShmemMap.set(bs.output, {
                name: sname,
                dtype: DType.Float32,
                elemCount: sz / 4,
              });
              bodyShmemIds.add(bs.output);
              totalShmemBytes += sz;
              break;
            }
            case "free":
            case "incref":
              break;
            case "recycle": {
              const bExisting = bodyShmemMap.get(bs.input);
              if (bExisting) {
                // Output reuses input's storage. Keep both mapped to same entry;
                // declaration site deduplicates by name.
                bodyShmemMap.set(bs.output, bExisting);
                bodyShmemIds.add(bs.output);
              } else {
                // Input is a body input (const/elem), not a prior shmem malloc.
                // The fused shader can't represent this — fall back.
                if (DEBUG >= 1)
                  console.info(
                    "block_map fused: recycle of body input in workgroup_assoc_scan body, fallback",
                  );
                valid = false;
              }
              break;
            }
            default:
              valid = false;
              break;
          }
          if (!valid) break;
        }
        if (!valid) {
          if (DEBUG >= 1)
            console.info(
              "block_map fused: invalid workgroup_assoc_scan body, fallback",
            );
          return null;
        }

        // Refine body shmem dtypes from body kernel outputs
        for (const { step: bs, kernel: bk } of bodyKernels) {
          for (let oi = 0; oi < bk.numOutputs; oi++) {
            const entry = bodyShmemMap.get(bs.outputs[oi]);
            if (entry) {
              entry.dtype = bk.outputs[oi].dtype;
              entry.elemCount = (entry.elemCount * 4) / byteWidth(entry.dtype);
            }
          }
        }

        const numElems = step.numElems;
        const numConsts = step.numConsts;

        // Determine elem dtypes from elemAvals
        const elemDtypes: DType[] = step.elemAvals.map((a) => a.dtype);
        const elemCounts: number[] = step.elemAvals.map(
          (a) => a.size / (a.shape[0] as number),
        );

        // Allocate ping-pong shmem per elem (each: blockSize * elemCount elements)
        const pingPongNames: [string, string][] = [];
        for (let e = 0; e < numElems; e++) {
          const count = blockSize * elemCounts[e];
          const bytes = count * byteWidth(elemDtypes[e]);
          const pingName = `was${wasIdx}_ping_${e}`;
          const pongName = `was${wasIdx}_pong_${e}`;
          pingPongNames.push([pingName, pongName]);
          totalShmemBytes += bytes * 2; // ping + pong
        }

        const numRounds = Math.ceil(Math.log2(blockSize));

        // Detect if body is a simple scalar Add or Mul — enables
        // subgroupInclusiveAdd / subgroupInclusiveMul fast path.
        let scalarOp: AluOp.Add | AluOp.Mul | undefined;
        if (
          numElems === 1 &&
          elemCounts[0] === 1 &&
          numConsts === 0 &&
          bodyKernels.length === 1
        ) {
          const detected = detectScalarAssocOp(bodyKernels[0].kernel);
          if (detected != null) scalarOp = detected;
        }

        workgroupAssocScans.push({
          wasStep: step as Extract<JitStep, { type: "workgroup_assoc_scan" }>,
          bodyKernels,
          bodyShmemMap,
          bodyShmemIds,
          bodyInputIds: bodyProg.inputs,
          bodyOutputIds: bodyProg.outputs,
          numConsts,
          numElems,
          elemDtypes,
          elemCounts,
          pingPongNames,
          numRounds,
          scalarOp,
        });
        codegenEntries.push({ type: "workgroup_assoc_scan", wasIdx });
        break;
      }
      default:
        // Unsupported step type (scan, dus, scatter_add, etc.)
        if (DEBUG >= 1)
          console.info(
            `block_map fused: unsupported step type "${(step as JitStep).type}", fallback`,
          );
        return null;
    }
  }

  // --- Account for reduction workspace shmem ---
  // Tree reductions (kernel.size == 1) need a workspace array of blockSize
  // elements. Per-element reductions (kernel.size > 1) use local accumulators
  // and don't need workspace shmem.
  const reduceWorkspaces: {
    stepIdx: number;
    dtype: DType;
    elemCount: number;
  }[] = [];
  for (let si = 0; si < kernelSteps.length; si++) {
    const re = stepReductions[si];
    if (re && (kernelSteps[si].kernel.size as number) === 1) {
      const wsBytes = blockSize * byteWidth(re.dtype);
      reduceWorkspaces.push({
        stepIdx: si,
        dtype: re.dtype,
        elemCount: blockSize,
      });
      totalShmemBytes += wsBytes;
    }
  }

  // Track whether any kernel needs a multi-element gidx loop per thread
  const hasMultiElemKernels = kernelSteps.some(
    ({ kernel }) => (kernel.size as number) > blockSize,
  );

  // --- Guard: shared memory budget ---
  const maxShmem = device.limits.maxComputeWorkgroupStorageSize;
  if (totalShmemBytes > maxShmem) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: shmem ${totalShmemBytes} > limit ${maxShmem}, fallback`,
      );
    return null;
  }

  // --- Determine body input/output buffer mapping ---
  // Body program inputs: [const0..constN, input0..inputM, (blockIndex?)]
  // Body program outputs: [out0..outK]
  const bodyInputIds = bodyProgram.inputs; // JitId[]
  const bodyOutputIds = bodyProgram.outputs; // JitId[]
  const hasBlockIndex = params.hasBlockIndex === true;
  // blockIndexInputId is the JitId of the synth block-index input (last input).
  const blockIndexInputId = hasBlockIndex
    ? bodyInputIds[bodyInputIds.length - 1]
    : -1;
  // numBodyInputs excludes the block index (it's not a storage buffer).
  const numBodyInputs = bodyInputIds.length - (hasBlockIndex ? 1 : 0);

  // --- Determine which constants can move to uniform buffers ---
  // Small constants (with constInfos provided) go to @group(1) var<uniform>,
  // freeing storage bindings. Each constant buffer must already have
  // GPUBufferUsage.UNIFORM. Only constants that fit in a single vec4 (≤4
  // elements) are eligible — this ensures all in{ci}[idx] usages (including
  // shmem initialization, which doesn't go through the resolve function)
  // correctly perform vec4 component access (returns scalar, not vec4).
  const constInfos = params.constInfos;
  const numUniformConsts =
    constInfos && numConsts > 0
      ? constInfos.reduce(
          (n, info) =>
            n +
            (info.elemCount <= 4 &&
            info.bytes <= device.limits.maxUniformBufferBindingSize
              ? 1
              : 0),
          0,
        )
      : 0;
  // Note: we only support moving ALL constants to uniform (not a partial subset)
  // to keep the binding renumbering simple.
  const effectiveUniformConsts = numUniformConsts === numConsts ? numConsts : 0;

  // Packed leaf layout is computed later (after inputDtypes is populated).
  // For now, derive binding counts assuming packing is viable when requested.
  const numOutputs = bodyOutputIds.length;
  const assumePacked = needsLeafPacking === true;
  const storageInputs = assumePacked
    ? 1
    : numBodyInputs - effectiveUniformConsts;
  const numOutputBindings = assumePacked ? 1 : numOutputs;
  const totalBindings = storageInputs + numOutputBindings;
  const maxBindings = device.limits.maxStorageBuffersPerShaderStage;
  if (totalBindings > maxBindings) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: ${totalBindings} storage bindings > max ${maxBindings}, fallback`,
      );
    return null;
  }

  // --- Refine shmem dtypes from kernel consumers ---
  // Walk kernel steps to determine what dtype each shmem buffer should use.
  // A kernel step's inputs map: step.inputs[j] → GlobalIndex arg[0] = j.
  // So step.inputs[bufferIndex] gives the JitId of the buffer.
  for (const { step, kernel } of kernelSteps) {
    // Check kernel outputs → determine dtype of their output JitIds
    for (let oi = 0; oi < kernel.numOutputs; oi++) {
      const outId = step.outputs[oi];
      const entry = shmemMap.get(outId);
      if (entry) {
        entry.dtype = kernel.outputs[oi].dtype;
        entry.elemCount = entry.sizeBytes / byteWidth(entry.dtype);
      }
    }
  }

  // Also refine carry shmem dtypes from fori_loop body outputs.
  // The carry output shmem (fs.outputs[ci]) is only written by the fori_loop
  // body, not by any top-level kernel, so the above loop misses it.
  for (const fl of foriLoops) {
    const numCarries = fl.foriStep.initCarries.length;
    for (let ci = 0; ci < numCarries; ci++) {
      const parentOutId = fl.foriStep.outputs[ci];
      const entry = shmemMap.get(parentOutId);
      if (!entry) continue;
      const bodyOutId = fl.bodyOutputIds[ci];
      // Body output is typically a body-malloc'd intermediate (already refined)
      const bodyShmemEntry = fl.bodyShmemMap.get(bodyOutId);
      if (bodyShmemEntry) {
        entry.dtype = bodyShmemEntry.dtype;
        entry.elemCount = entry.sizeBytes / byteWidth(entry.dtype);
      }
    }
  }

  // Refine shmem dtypes from workgroup_assoc_scan outputs.
  // WAS outputs are malloc'd parent shmem buffers written by WAS codegen,
  // not by any top-level kernel step, so the above loop misses them.
  for (const was of workgroupAssocScans) {
    for (let e = 0; e < was.numElems; e++) {
      const parentOutId = was.wasStep.outputs[e];
      const entry = shmemMap.get(parentOutId);
      if (entry) {
        entry.dtype = was.elemDtypes[e];
        entry.elemCount = entry.sizeBytes / byteWidth(entry.dtype);
      }
    }
  }

  // --- Build JitId → buffer name mapping ---
  // Body inputs (consts + block inputs) → global storage buffer names
  // Intermediates (malloc outputs) → shmem array names
  // Body outputs → global storage result buffer names
  //
  // Note: pass-through outputs (output JitId === input JitId) are tracked
  // separately and handled as explicit copies in the shader.

  /** Map from JitId → input buffer name. */
  const inputIdToName = new Map<JitId, string>();
  /** Map from JitId → shmem or input name (for reading). */
  const idToReadName = new Map<JitId, string>();
  const idIsShmem = new Set<JitId>();
  /** Outputs that are pass-through copies of inputs. */
  const passThroughOutputs = new Map<
    number,
    { inputName: string; inputIdx: number }
  >(); // resultIdx → input info

  // Global inputs: in0..in{numBodyInputs-1}
  // With leaf packing: uniform consts keep in{ci}, non-consts map to "in_packed".
  for (let i = 0; i < numBodyInputs; i++) {
    const name =
      assumePacked && i >= effectiveUniformConsts ? "in_packed" : `in${i}`;
    inputIdToName.set(bodyInputIds[i], name);
    idToReadName.set(bodyInputIds[i], name);
  }
  // BlockIndex input: not a storage buffer — resolved to `block_idx` inline.
  // Add sentinel to idToReadName so fori_loop const mapping can find it.
  const BLOCK_IDX_SENTINEL = "__block_idx__";
  if (hasBlockIndex) idToReadName.set(blockIndexInputId, BLOCK_IDX_SENTINEL);

  // Detect pass-through outputs
  for (let o = 0; o < numOutputs; o++) {
    const outId = bodyOutputIds[o];
    const bodyInputIdx = bodyInputIds.indexOf(outId);
    if (bodyInputIdx >= 0) {
      const ptInputName =
        assumePacked && bodyInputIdx >= effectiveUniformConsts
          ? "in_packed"
          : `in${bodyInputIdx}`;
      passThroughOutputs.set(o, {
        inputName: ptInputName,
        inputIdx: bodyInputIdx >= numConsts ? bodyInputIdx - numConsts : -1,
      });
    }
  }

  // Output variable name helper (packed: out_packed; normal: result{o})
  const resultVar = assumePacked
    ? (_o: number) => "out_packed"
    : (o: number) => `result${o}`;

  // Shared memory intermediates: shmem_{id}
  for (const [id] of shmemMap) {
    const name = `shmem_${id}`;
    idToReadName.set(id, name);
    idIsShmem.add(id);
  }

  // Recycle aliases: recycled output → same shmem array as original input.
  // Resolve through chains (recycle of a recycle) to find the root shmem name.
  for (const [recycledId, originalId] of recycleAliases) {
    let rootId = originalId;
    while (recycleAliases.has(rootId)) rootId = recycleAliases.get(rootId)!;
    const name = idToReadName.get(rootId);
    if (name) {
      idToReadName.set(recycledId, name);
      if (idIsShmem.has(rootId)) idIsShmem.add(recycledId);
    }
  }

  // Determine which kernel outputs write directly to global result buffers
  // vs which write to shared memory.
  const outputWrittenByStep = new Map<
    JitId,
    { stepIdx: number; kernelOutputIdx: number }
  >();
  for (let si = 0; si < kernelSteps.length; si++) {
    const { step, kernel } = kernelSteps[si];
    for (let oi = 0; oi < kernel.numOutputs; oi++) {
      const outId = step.outputs[oi];
      if (bodyOutputIds.includes(outId)) {
        outputWrittenByStep.set(outId, { stepIdx: si, kernelOutputIdx: oi });
      }
    }
  }

  // --- Build data dependencies for barrier placement ---
  // Track which JitIds each step writes and reads for barrier insertion.
  // Conservative: barrier after every step that writes shmem, if a subsequent
  // step reads that shmem.
  const stepWrites = new Map<number, Set<JitId>>(); // stepIdx → written shmem ids
  const stepReads = new Map<number, Set<JitId>>(); // stepIdx → read shmem ids

  for (let si = 0; si < kernelSteps.length; si++) {
    const { step } = kernelSteps[si];
    const writes = new Set<JitId>();
    const reads = new Set<JitId>();
    for (const outId of step.outputs) {
      if (idIsShmem.has(outId)) writes.add(outId);
    }
    for (const inId of step.inputs) {
      if (idIsShmem.has(inId)) reads.add(inId);
    }
    stepWrites.set(si, writes);
    stepReads.set(si, reads);
  }

  // Determine where barriers are needed: before step si if it reads shmem
  // that was written by a previous step.
  const needsBarrierBefore = new Set<number>();
  for (let si = 1; si < kernelSteps.length; si++) {
    const reads = stepReads.get(si)!;
    if (reads.size === 0) continue;
    for (let prev = 0; prev < si; prev++) {
      const writes = stepWrites.get(prev)!;
      for (const id of reads) {
        if (writes.has(id)) {
          needsBarrierBefore.add(si);
          break;
        }
      }
      if (needsBarrierBefore.has(si)) break;
    }
  }

  // --- Codegen ---
  const shader: string[] = [];
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : "");
    }
  };

  // Collect all distinct ops across all kernel steps for global functions
  let allOps: Map<AluOp, Set<DType>> = new Map();
  let needsF16 = false;
  const collectKernelOps = (kernel: Kernel) => {
    for (const output of kernel.outputs) {
      const ops = output.exp.distinctOps();
      allOps = mapSetUnion(allOps, ops);
      output.exp.fold((exp) => {
        if (exp.dtype === DType.Float16) needsF16 = true;
      });
    }
  };
  for (const { kernel } of kernelSteps) collectKernelOps(kernel);
  for (const fl of foriLoops) {
    for (const { kernel } of fl.bodyKernels) collectKernelOps(kernel);
  }
  for (const was of workgroupAssocScans) {
    for (const { kernel } of was.bodyKernels) collectKernelOps(kernel);
  }

  if (needsF16) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  // Check if subgroup reduction is available for non-Bool reductions.
  const hasSubgroups = device.features.has("subgroups");
  if (hasSubgroups) {
    emit("enable subgroups;");
  }

  if (allOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc)) emit(erfSrc);
  emit("");

  // --- Global storage bindings ---
  // Determine dtype of each body input from kernel consumers
  const inputDtypes: (DType | null)[] = new Array(numBodyInputs).fill(null);

  // Helper: record dtype for a GlobalIndex/GlobalView expression given its step's
  // input list and a mapping from step input JitIds back to parent bodyInputIds.
  // Note: kernel expressions may contain either GlobalView (pre-lowering) or
  // GlobalIndex (post-lowering). Both have the buffer index as arg[0].
  const recordGlobalIndexDtype = (
    step: { inputs: JitId[] },
    kernel: Kernel,
    mapToBodyInputIdx: (jitId: JitId) => number,
  ) => {
    for (const output of kernel.outputs) {
      output.exp.fold((exp) => {
        if (exp.op === AluOp.GlobalIndex || exp.op === AluOp.GlobalView) {
          const bufIdx = exp.arg[0] as number;
          if (bufIdx < step.inputs.length) {
            const jitId = step.inputs[bufIdx];
            const bodyInputIdx = mapToBodyInputIdx(jitId);
            if (bodyInputIdx >= 0) {
              inputDtypes[bodyInputIdx] = exp.dtype;
            }
          }
        }
      });
    }
  };

  // Top-level kernel steps
  for (const { step, kernel } of kernelSteps) {
    recordGlobalIndexDtype(step, kernel, (jitId) =>
      bodyInputIds.indexOf(jitId),
    );
  }

  // Fori_loop body kernels: trace through foriStep.consts to reach parent bodyInputIds
  for (const fl of foriLoops) {
    for (const { step: bStep, kernel: bKernel } of fl.bodyKernels) {
      recordGlobalIndexDtype(bStep, bKernel, (jitId) => {
        const biIdx = fl.bodyInputIds.indexOf(jitId);
        if (biIdx >= 0 && biIdx < fl.numConsts) {
          const parentJitId = fl.foriStep.consts[biIdx];
          return bodyInputIds.indexOf(parentJitId);
        }
        return -1;
      });
    }
  }

  // WorkgroupAssociativeScan body kernels: trace through wasStep consts
  for (const was of workgroupAssocScans) {
    for (const { step: bStep, kernel: bKernel } of was.bodyKernels) {
      recordGlobalIndexDtype(bStep, bKernel, (jitId) => {
        const biIdx = was.bodyInputIds.indexOf(jitId);
        if (biIdx >= 0 && biIdx < was.numConsts) {
          const parentJitId = was.wasStep.consts[biIdx];
          return bodyInputIds.indexOf(parentJitId);
        }
        return -1;
      });
    }
  }

  // Populate inputDtypes for WAS elem inputs: the scan body reads elem
  // data from ping-pong shared memory, not via GlobalIndex, so the above
  // tracing loop misses them. Use the known elemDtypes.
  for (const was of workgroupAssocScans) {
    for (let e = 0; e < was.numElems; e++) {
      const elemId = was.wasStep.elems[e];
      const bodyIdx = bodyInputIds.indexOf(elemId);
      if (bodyIdx >= 0 && inputDtypes[bodyIdx] === null) {
        inputDtypes[bodyIdx] = was.elemDtypes[e];
      }
    }
  }

  // Infer input dtypes from fori_loop init carries: the init carry input
  // is only used in the init→shmem copy (not by any kernel), so its dtype
  // must match the carry shmem dtype.
  for (const fl of foriLoops) {
    const numCarries = fl.foriStep.initCarries.length;
    for (let ci = 0; ci < numCarries; ci++) {
      const initId = fl.foriStep.initCarries[ci];
      const bodyIdx = bodyInputIds.indexOf(initId);
      if (bodyIdx >= 0 && inputDtypes[bodyIdx] === null) {
        const parentOutId = fl.foriStep.outputs[ci];
        const entry = shmemMap.get(parentOutId);
        if (entry) {
          inputDtypes[bodyIdx] = entry.dtype;
        }
      }
    }
  }

  // --- Compute packedLeafLayout now that inputDtypes is populated ---
  let packedLeafLayout: PackedLeafLayout | undefined;
  if (assumePacked) {
    const ncInputs = numBodyInputs - effectiveUniformConsts;
    let commonDtype: DType | null = null;
    for (let i = effectiveUniformConsts; i < numBodyInputs; i++) {
      const dt = inputDtypes[i] ?? DType.Float32;
      if (commonDtype === null) commonDtype = dt;
      else if (dt !== commonDtype) {
        commonDtype = null;
        break;
      }
    }
    if (commonDtype !== null && ncInputs > 0) {
      const inputOffsets: number[] = [];
      let totalIn = 0;
      for (let i = 0; i < ncInputs; i++) {
        inputOffsets.push(totalIn);
        totalIn += prod(params.inputShapes[i]);
      }
      const outputOffsets: number[] = [];
      let totalOut = 0;
      for (let o = 0; o < numOutputs; o++) {
        outputOffsets.push(totalOut);
        totalOut += prod(params.outputShapes[o]);
      }
      packedLeafLayout = {
        inputOffsets,
        totalInputElems: totalIn,
        outputOffsets,
        totalOutputElems: totalOut,
        dtype: commonDtype,
      };
    }
    if (!packedLeafLayout) {
      // Mixed dtypes — can't pack. Fall back.
      if (DEBUG >= 1)
        console.info(
          "block_map fused: leaf packing requested but dtypes differ, fallback",
        );
      return null;
    }
  }

  // --- Emit input bindings ---
  // Uniform constants go to @group(1); non-const inputs go to @group(0).
  // Variable names stay as in0, in1, ... regardless of binding group.
  // With packedLeafLayout: all non-const inputs share `in_packed : array<T>`.
  if (packedLeafLayout) {
    // Emit uniform constant bindings on @group(1) (same as non-packed path)
    for (let ci = 0; ci < effectiveUniformConsts; ci++) {
      const info = constInfos![ci];
      const ty = dtypeToWgsl(inputDtypes[ci] ?? info.dtype, true);
      emit(`@group(1) @binding(${ci}) var<uniform> in${ci} : vec4<${ty}>;`);
    }
    // Emit single packed input storage binding on @group(0) @binding(0)
    const pty = dtypeToWgsl(packedLeafLayout.dtype, true);
    emit(`@group(0) @binding(0) var<storage, read> in_packed : array<${pty}>;`);
  } else if (effectiveUniformConsts > 0) {
    // Emit uniform constant bindings on @group(1) as vec4<T>.
    // All uniform constants have ≤4 elements, so a single vec4 suffices.
    // Component access via in{ci}[idx] returns the scalar type.
    for (let ci = 0; ci < effectiveUniformConsts; ci++) {
      const info = constInfos![ci];
      const ty = dtypeToWgsl(inputDtypes[ci] ?? info.dtype, true);
      emit(`@group(1) @binding(${ci}) var<uniform> in${ci} : vec4<${ty}>;`);
    }
    // Emit non-const input bindings on @group(0), binding indices start at 0
    let storageIdx = 0;
    for (let i = effectiveUniformConsts; i < numBodyInputs; i++) {
      const ty = dtypeToWgsl(inputDtypes[i] ?? DType.Float32, true);
      emit(
        `@group(0) @binding(${storageIdx}) var<storage, read> in${i} : array<${ty}>;`,
      );
      storageIdx++;
    }
  } else {
    // No uniform constants — all inputs on @group(0) as before
    for (let i = 0; i < numBodyInputs; i++) {
      const ty = dtypeToWgsl(inputDtypes[i] ?? DType.Float32, true);
      emit(
        `@group(0) @binding(${i}) var<storage, read> in${i} : array<${ty}>;`,
      );
    }
  }

  // Output bindings
  // Determine output dtypes from kernels that write to them
  const outputDtypes: DType[] = [];
  for (let o = 0; o < numOutputs; o++) {
    const outId = bodyOutputIds[o];
    let dtype = DType.Float32;
    const written = outputWrittenByStep.get(outId);
    if (written) {
      dtype =
        kernelSteps[written.stepIdx].kernel.outputs[written.kernelOutputIdx]
          .dtype;
    } else {
      // Check if this output is a fori_loop carry output
      const carryShmem = shmemMap.get(outId);
      if (carryShmem) {
        dtype = carryShmem.dtype;
      } else {
        // Pass-through output: use the input's dtype
        const ptInfo = passThroughOutputs.get(o);
        if (ptInfo) {
          const inDtype = inputDtypes[bodyInputIds.indexOf(outId)];
          if (inDtype) dtype = inDtype;
        }
      }
    }
    outputDtypes.push(dtype);
    if (!packedLeafLayout) {
      const ty = dtypeToWgsl(dtype, true);
      emit(
        `@group(0) @binding(${storageInputs + o}) var<storage, read_write> result${o} : array<${ty}>;`,
      );
    }
  }
  // With packedLeafLayout: validate output dtypes match the packed dtype,
  // then emit a single packed output binding.
  if (packedLeafLayout) {
    for (let o = 0; o < numOutputs; o++) {
      if (outputDtypes[o] !== packedLeafLayout.dtype) {
        if (DEBUG >= 1)
          console.info(
            `block_map fused: output ${o} dtype ${outputDtypes[o]} != packed dtype ${packedLeafLayout.dtype}, fallback`,
          );
        return null;
      }
    }
    const pty = dtypeToWgsl(packedLeafLayout.dtype, true);
    emit(
      `@group(0) @binding(${storageInputs}) var<storage, read_write> out_packed : array<${pty}>;`,
    );
  }

  // --- Shared memory declarations ---
  // O4: Build set of parent shmem IDs that are privatized by register tiling.
  // These are carry outputs whose body-side counterparts are in privateShmemIds.
  const privatizedParentShmemIds = new Set<JitId>();
  for (const fl of foriLoops) {
    if (fl.privateShmemIds.size === 0) continue;
    const numCarries = fl.foriStep.initCarries.length;
    for (let ci = 0; ci < numCarries; ci++) {
      if (fl.privateShmemIds.has(fl.bodyOutputIds[ci])) {
        privatizedParentShmemIds.add(fl.foriStep.outputs[ci]);
      }
    }
  }
  for (const [id, info] of shmemMap) {
    if (!idIsShmem.has(id)) continue;
    if (privatizedParentShmemIds.has(id)) continue; // O4: private → var<private> later
    const ty = dtypeToWgsl(info.dtype, false);
    const name = idToReadName.get(id)!;
    emit(`var<workgroup> ${name}: array<${ty}, ${info.elemCount}>;`);
  }

  // Reduction workspace shmem arrays (one per reduction step)
  for (const ws of reduceWorkspaces) {
    const ty = dtypeToWgsl(ws.dtype, false);
    emit(
      `var<workgroup> reduce_ws_${ws.stepIdx}: array<${ty}, ${ws.elemCount}>;`,
    );
  }

  // Fori_loop body intermediate shmem arrays (skip private buffers)
  for (const fl of foriLoops) {
    const flDeclaredNames = new Set<string>();
    for (const [id, info] of fl.bodyShmemMap) {
      if (fl.privateShmemIds.has(id)) continue; // O4: private → declared in fn body
      if (flDeclaredNames.has(info.name)) continue; // recycle aliases
      flDeclaredNames.add(info.name);
      const ty = dtypeToWgsl(info.dtype, false);
      // O11: bank-padded shmem — add 1 extra element per row
      const pad = fl.shmemBankPad.get(id);
      const size = pad ? (info.elemCount / pad) * (pad + 1) : info.elemCount;
      emit(`var<workgroup> ${info.name}: array<${ty}, ${size}>;`);
    }
  }

  // WorkgroupAssociativeScan ping-pong shmem arrays + body intermediates
  for (const was of workgroupAssocScans) {
    for (let e = 0; e < was.numElems; e++) {
      const ty = dtypeToWgsl(was.elemDtypes[e], false);
      const count = blockSize * was.elemCounts[e];
      const [pingName, pongName] = was.pingPongNames[e];
      emit(`var<workgroup> ${pingName}: array<${ty}, ${count}>;`);
      emit(`var<workgroup> ${pongName}: array<${ty}, ${count}>;`);
    }
    const wasDeclaredNames = new Set<string>();
    for (const [, info] of was.bodyShmemMap) {
      if (wasDeclaredNames.has(info.name)) continue; // recycle aliases
      wasDeclaredNames.add(info.name);
      const ty = dtypeToWgsl(info.dtype, false);
      // Body intermediates are per-thread (each thread computes its own
      // compose result), so use var<private> — NOT var<workgroup>.
      // Using var<workgroup> with per-element sizes causes a race condition
      // where all threads write to the same few shared positions.
      emit(`var<private> ${info.name}: array<${ty}, ${info.elemCount}>;`);
    }
  }

  // --- Workgroup size and grid ---
  // Each workgroup = 1 block. Without threadTile, thread i processes element i.
  // With threadTile, each thread handles threadTile[g] elements per axis.
  // workgroup_id maps to block index in the grid.
  const wgSizeX = wgShape[0] ?? 1;
  const wgSizeY = wgShape.length > 1 ? wgShape[1] : 1;
  const wgSizeZ = wgShape.length > 2 ? wgShape[2] : 1;
  const totalBlocks = prod(gridShape);
  const [gridX, gridY] = calculateGrid(totalBlocks);

  // Build workgroup_size attribute
  let wgSizeStr = `${wgSizeX}`;
  if (wgSizeY > 1 || wgSizeZ > 1) wgSizeStr += `, ${wgSizeY}`;
  if (wgSizeZ > 1) wgSizeStr += `, ${wgSizeZ}`;

  // Determine if we can use subgroupShuffleUp for assocScan Kogge-Stone rounds.
  // Conservative: 3 rounds (stride ≤ 4) guaranteed within any subgroup (min sg_size=8).
  const useSubgroupShuffle = hasSubgroups && workgroupAssocScans.length > 0;

  emit("", `@compute @workgroup_size(${wgSizeStr})`);
  if (hasSubgroups) {
    emit(
      "fn main(",
      `  @builtin(local_invocation_index) tidx: u32,`,
      `  @builtin(workgroup_id) wg_id: vec3<u32>,`,
      `  @builtin(subgroup_size) sg_size: u32,`,
      ...(useSubgroupShuffle
        ? [`  @builtin(subgroup_invocation_id) sg_inv_id: u32,`]
        : []),
      ") {",
      pushIndent,
    );
  } else {
    emit(
      "fn main(",
      `  @builtin(local_invocation_index) tidx: u32,`,
      `  @builtin(workgroup_id) wg_id: vec3<u32>,`,
      ") {",
      pushIndent,
    );
  }

  // Compute flat block index from workgroup_id
  if (gridY === 1) {
    emit("let block_idx: u32 = wg_id.x;");
  } else {
    emit(`let block_idx: u32 = wg_id.y * ${gridX}u + wg_id.x;`);
  }

  // Guard: skip excess workgroups when grid is padded
  emit(`if (block_idx >= ${totalBlocks}u) { return; }`);

  // Compute per-axis block indices (row-major from block_idx)
  if (gridRank === 1) {
    emit("let block_i0: u32 = block_idx;");
  } else {
    emit("var _remaining: u32 = block_idx;");
    for (let g = gridRank - 1; g >= 0; g--) {
      if (g === 0) {
        emit(`let block_i${g}: u32 = _remaining;`);
      } else {
        emit(`let block_i${g}: u32 = _remaining % ${gridShape[g]}u;`);
        emit(`_remaining = _remaining / ${gridShape[g]}u;`);
      }
    }
  }

  // Compute per-axis thread indices within the block
  // With threadTile: tidx indexes into the wgShape grid (thread grid).
  // Without threadTile: tidx indexes into blockShape (1:1 element mapping).
  const tidxShape = wgShape;
  if (tidxShape.length === 1) {
    emit("let tidx_0: u32 = tidx;");
  } else if (tidxShape.length === 2) {
    // 2D: tidx = row * cols + col
    emit(`let tidx_0: u32 = tidx / ${tidxShape[1]}u;`); // row
    emit(`let tidx_1: u32 = tidx % ${tidxShape[1]}u;`); // col
  } else {
    // 3D: tidx = d0 * (d1*d2) + d1 * d2 + d2
    const d12 = tidxShape[1] * tidxShape[2];
    emit(`let tidx_0: u32 = tidx / ${d12}u;`);
    emit(`let tidx_1: u32 = (tidx % ${d12}u) / ${tidxShape[2]}u;`);
    emit(`let tidx_2: u32 = tidx % ${tidxShape[2]}u;`);
  }

  // --- Per-thread validity (boundary blocks) ---
  // For the last block along each axis, some threads may be out-of-bounds.
  // `valid` is true iff this thread maps to a real element for ALL axes.
  // Invalid threads read 0 from global inputs and don't write to global outputs,
  // matching the zero-pad semantics of the eagerness fallback.
  if (hasBoundary) {
    const terms: string[] = [];
    for (let g = 0; g < gridRank; g++) {
      if (axisDims[g] !== null) {
        if (gridOffset > 0) {
          terms.push(
            `((block_i${g} + ${gridOffset}u) * ${blockShape[g]}u + tidx_${g} < ${axisDims[g]}u)`,
          );
        } else {
          terms.push(
            `(block_i${g} * ${blockShape[g]}u + tidx_${g} < ${axisDims[g]}u)`,
          );
        }
      }
    }
    emit(`let valid: bool = ${terms.join(" && ")};`);
  }

  // Build global offset for each input and output.
  // For input i, axis g: globalOffset = block_i{g} * blockShape[g]
  // The kernel indexes with gidx (0..blockSize-1), but in the fused shader
  // we remap GlobalIndex reads to global buffer positions.

  // For each body input, precompute the global base offset
  // and detect stride mismatches between body-local and global layouts.
  //
  // When inAxes maps only a subset of an input's dimensions to grid axes,
  // the body sees a shape where mapped dims are shrunk to blockShape, creating
  // body-local strides that differ from the global buffer strides.
  // Example: B is [32,32] with inAxes=[null,1], blockShape=[16,16].
  //   Body shape = [32,16], body strides = [16,1], global strides = [32,1].
  //   Body-local flat index uses stride 16 per row, but global needs stride 32.
  //
  // When strides differ, resolveGlobalIndex must decompose the body-local flat
  // index into n-D coordinates (using body strides) and recompute with global
  // strides.
  const inBufStrides: number[][] = []; // global strides per input
  const inBodyStrides: number[][] = []; // body-local strides per input
  const inBodyShapes: number[][] = []; // body-local shapes per input
  const inNeedsRemap: boolean[] = []; // whether remapping is needed
  // Per-input halo metadata: shift in elements and total input element count.
  // haloShift[i] is the signed element offset to subtract from in_base so that
  // body-local index 0 maps to the first halo element (before the block start).
  const inHaloShift: number[] = []; // per-input: sum of lo * globalStride[ax]
  const inTotalElems: number[] = []; // per-input: product of original inputShape
  const inHasHalo: boolean[] = []; // per-input: any non-zero halo?
  const inPackOffset: number[] = []; // per-input: packed-buffer element offset (0 when no leaf packing)

  for (let i = 0; i < numInputs; i++) {
    const axes = params.inAxes[i];
    const isPoint = params.pointInputs?.[i] ?? false;
    const inShape = params.inputShapes[i];
    const nd = inShape.length;
    // Global strides
    const gStrides: number[] = new Array(nd);
    gStrides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      gStrides[d] = gStrides[d + 1] * inShape[d + 1];
    }
    inBufStrides.push(gStrides);
    // Body shape: replace mapped dims with blockShape (skip for point inputs).
    // With halo, the body sees blockShape[g] + lo + hi along mapped axes.
    const bShape = [...inShape];
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null && !isPoint) {
        const [lo, hi] = params.halo?.[i]?.[g] ?? [0, 0];
        bShape[axes[g]!] = blockShape[g] + lo + hi;
      }
    }
    inBodyShapes.push(bShape);
    // Body strides
    const bStrides: number[] = new Array(nd);
    bStrides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      bStrides[d] = bStrides[d + 1] * bShape[d + 1];
    }
    inBodyStrides.push(bStrides);
    // Detect mismatch
    let needsRemap = false;
    for (let d = 0; d < nd; d++) {
      if (bStrides[d] !== gStrides[d]) {
        needsRemap = true;
        break;
      }
    }
    inNeedsRemap.push(needsRemap);

    // Halo: compute signed element shift and total element count for OOB checks.
    let haloShift = 0;
    let hasHalo = false;
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null && !isPoint) {
        const [lo, hi] = params.halo?.[i]?.[g] ?? [0, 0];
        if (lo !== 0 || hi !== 0) {
          hasHalo = true;
          haloShift += lo * gStrides[axes[g]!];
        }
      }
    }
    inHaloShift.push(haloShift);
    inTotalElems.push(inShape.reduce((a, b) => a * b, 1));
    inHasHalo.push(hasHalo);

    // Compute base offset
    // With packedLeafLayout: add per-leaf element offset within packed buffer.
    // Track the offset for halo bounds checking in packed coordinates.
    const packInputOff =
      packedLeafLayout && !isPoint
        ? packedLeafLayout.inputOffsets[i]
        : packedLeafLayout && isPoint
          ? packedLeafLayout.inputOffsets[i]
          : 0;
    inPackOffset.push(packInputOff);
    const packPrefix = packInputOff > 0 ? `${packInputOff}u + ` : "";
    if (isPoint) {
      // Point-mode: 1 element per grid point, indexed by block_i0.
      // inputShapes[i] is the per-element shape (no grid dimension).
      const elemFlatSize = inShape.reduce((a, b) => a * b, 1);
      emit(`let in_base_${i}: u32 = ${packPrefix}block_i0 * ${elemFlatSize}u;`);
    } else {
      // Normal mapped: blockShape elements per grid point.
      const terms: string[] = [];
      for (let g = 0; g < gridRank; g++) {
        if (axes[g] !== null) {
          const ax = axes[g]!;
          const blockStride = blockShape[g] * gStrides[ax];
          if (gridOffset > 0) {
            terms.push(`(block_i${g} + ${gridOffset}u) * ${blockStride}u`);
          } else {
            terms.push(`block_i${g} * ${blockStride}u`);
          }
        }
      }
      const baseExpr = terms.length > 0 ? terms.join(" + ") : "0u";
      emit(`let in_base_${i}: u32 = ${packPrefix}${baseExpr};`);
    }
  }

  /**
   * Decompose a body-local flat index expression into per-axis coordinate
   * expressions for input i.  Returns an array indexed by axis dimension,
   * where each entry is the WGSL expression for that coordinate.
   *
   * Used by `inRemap` (global stride recomputation) and the halo bounds
   * check (per-axis OOB detection).  When both need coordinates for the
   * same input, the caller computes them once and passes the result to
   * `inRemap` via `precomputedCoords` to avoid duplicate GPU ALU.
   */
  function inCoordExprs(i: number, flatExpr: string): string[] {
    const bShape = inBodyShapes[i];
    const bStrides = inBodyStrides[i];
    const nd = bShape.length;
    const coords: string[] = new Array(nd);
    for (let d = 0; d < nd; d++) {
      if (bStrides[d] === 1) {
        coords[d] = `((${flatExpr}) % ${bShape[d]})`;
      } else if (d === 0) {
        coords[d] = `((${flatExpr}) / ${bStrides[d]})`;
      } else {
        coords[d] = `(((${flatExpr}) / ${bStrides[d]}) % ${bShape[d]})`;
      }
    }
    return coords;
  }

  /**
   * Remap a body-local flat index expression to a global flat index for input i.
   * When body strides match global strides, returns the expression unchanged.
   * When they differ, decomposes into n-D coords and recomputes with global strides.
   * Uses i32 arithmetic to match the gen() output type.
   *
   * When `precomputedCoords` is provided (from the halo path), those coordinate
   * expressions are used directly, avoiding a redundant inCoordExprs call.
   */
  function inRemap(
    i: number,
    flatExpr: string,
    precomputedCoords?: string[],
  ): string {
    if (!inNeedsRemap[i]) return flatExpr;
    const gStrides = inBufStrides[i];
    const coords = precomputedCoords ?? inCoordExprs(i, flatExpr);
    const nd = coords.length;
    // globalIdx = sum(coord[d] * globalStride[d])
    const terms: string[] = [];
    for (let d = 0; d < nd; d++) {
      if (gStrides[d] === 0) continue;
      if (gStrides[d] === 1) {
        terms.push(coords[d]);
      } else {
        terms.push(`${coords[d]} * ${gStrides[d]}`);
      }
    }
    return terms.length > 0 ? terms.join(" + ") : "0";
  }

  // For each body output, precompute the global base offset and strided tidx.
  // When the output block is a non-contiguous sub-rectangle of the output buffer
  // (e.g. a 16×16 tile in a 32×32 output), flat tidx ≠ strided output offset.
  const outBufStrides: number[][] = [];
  const outNeedsStrided: boolean[] = [];
  const outBodyMatchesPhysical: boolean[] = [];
  for (let o = 0; o < numOutputs; o++) {
    const axes = params.outAxes[o];
    const outShape = params.outputShapes[o];
    const nd = outShape.length;
    const strides: number[] = new Array(nd);
    strides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * outShape[d + 1];
    }
    outBufStrides.push(strides);
    const terms: string[] = [];
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) {
        const ax = axes[g]!;
        const blockStride = blockShape[g] * strides[ax];
        if (gridOffset > 0) {
          terms.push(`(block_i${g} + ${gridOffset}u) * ${blockStride}u`);
        } else {
          terms.push(`block_i${g} * ${blockStride}u`);
        }
      }
    }
    const baseExpr = terms.length > 0 ? terms.join(" + ") : "0u";
    // With packedLeafLayout: add per-output element offset in packed output.
    const packOutOff = packedLeafLayout ? packedLeafLayout.outputOffsets[o] : 0;
    const packOutPrefix = packOutOff > 0 ? `${packOutOff}u + ` : "";
    emit(`let out_base_${o}: u32 = ${packOutPrefix}${baseExpr};`);

    // Check if flat block-local index matches strided output offset.
    // They differ when the output buffer stride for any mapped axis
    // doesn't match the block's inner product for that axis.
    let needsStrided = false;
    let innerProd = 1;
    for (let g = gridRank - 1; g >= 0; g--) {
      if (axes[g] !== null && strides[axes[g]!] !== innerProd) {
        needsStrided = true;
      }
      innerProd *= blockShape[g];
    }

    // For multi-element kernels (elemsPerThread > 1), gidx encodes ALL
    // output dimensions (mapped + non-mapped), not just the mapped grid
    // coordinates. Check if the body output strides match the physical
    // buffer strides — if so, flat gidx IS the correct physical offset
    // and we can skip the strided decomposition for multi-element writes.
    // Body shape = outputShape with mapped axes replaced by blockShape.
    const bodyShape = [...outShape];
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) bodyShape[axes[g]!] = blockShape[g];
    }
    const bodyStrides: number[] = new Array(nd);
    bodyStrides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      bodyStrides[d] = bodyStrides[d + 1] * bodyShape[d + 1];
    }
    let bodyMatchesPhysical = true;
    for (let d = 0; d < nd; d++) {
      if (bodyStrides[d] !== strides[d]) {
        bodyMatchesPhysical = false;
        break;
      }
    }

    outNeedsStrided.push(needsStrided);
    outBodyMatchesPhysical.push(bodyMatchesPhysical);
    if (needsStrided) {
      const sTerms: string[] = [];
      for (let g = 0; g < gridRank; g++) {
        if (axes[g] !== null) {
          const stride = strides[axes[g]!];
          sTerms.push(stride !== 1 ? `tidx_${g} * ${stride}u` : `tidx_${g}`);
        }
      }
      emit(`let out_strided_tidx_${o}: u32 = ${sTerms.join(" + ")};`);
    }
  }

  /**
   * Convert a flat block-local index expression to a strided output offset.
   * For the common case (flatExpr === "tidx"), uses the precomputed variable.
   * For complex expressions (e.g., "tidx * 4u + _was_wi"), decomposes inline.
   */
  function outOffset(o: number, flatExpr: string = "tidx"): string {
    if (!outNeedsStrided[o]) return flatExpr;
    if (flatExpr === "tidx") return `out_strided_tidx_${o}`;
    // Inline decomposition for arbitrary flat expressions
    const axes = params.outAxes[o];
    const strides = outBufStrides[o];
    const oTerms: string[] = [];
    let innerProd = 1;
    for (let g = gridRank - 1; g >= 0; g--) {
      if (axes[g] !== null) {
        const axStride = strides[axes[g]!];
        const tidxG =
          innerProd === 1
            ? `((${flatExpr}) % ${blockShape[g]}u)`
            : `(((${flatExpr}) / ${innerProd}u) % ${blockShape[g]}u)`;
        oTerms.push(axStride !== 1 ? `${tidxG} * ${axStride}u` : tidxG);
      }
      innerProd *= blockShape[g];
    }
    return oTerms.reverse().join(" + ");
  }

  /**
   * Output offset for multi-element kernels (elemsPerThread > 1).
   * When body output strides match physical buffer strides, the flat gidx
   * IS the correct physical offset — no decomposition needed.
   * Otherwise, falls back to the standard `outOffset` decomposition.
   */
  function outOffsetMultiElem(o: number, flatExpr: string): string {
    if (outBodyMatchesPhysical[o]) return flatExpr;
    return outOffset(o, flatExpr);
  }

  // Declare gidx — the body kernel expressions reference this variable.
  // Each thread processes one element: gidx = tidx.
  // O4: When register tiling is active, gidx is mutable (reassigned per-element
  // in thread-tile loops and cooperative stride loops).
  // Multi-element kernels: gidx is mutable (reassigned per element in gidx loop).
  const hasRegisterTiling = foriLoops.some((fl) => fl.privateShmemIds.size > 0);
  const needsMutableGidx = hasRegisterTiling || hasMultiElemKernels;
  emit(
    needsMutableGidx
      ? "var gidx: i32 = i32(tidx);"
      : "let gidx: i32 = i32(tidx);",
  );

  // Helper: uniform-aware constant read.
  // All uniform constants are declared as vec4<T>, so component access
  // via in{ci}[idx] returns the scalar type — same as array indexing.
  // This function is a no-op identity but documents the intent.
  function readConst(ci: number, indexExpr: string): string {
    return `in${ci}[${indexExpr}]`;
  }

  // Phony assignments for unused inputs
  if (numBodyInputs > 0) {
    if (packedLeafLayout) {
      // With leaf packing: only keep-alive uniform consts + packed buffers
      const parts: string[] = [];
      for (let i = 0; i < effectiveUniformConsts; i++)
        parts.push(`_ = &in${i};`);
      parts.push("_ = &in_packed;");
      parts.push("_ = &out_packed;");
      emit(parts.join(" "));
    } else {
      emit(
        Array.from({ length: numBodyInputs }, (_, i) => `_ = &in${i};`).join(
          " ",
        ),
      );
    }
  }

  // --- Helper: create gen() function for a kernel step ---
  // Delegates to the shared createWgslGen, passing emit and blockSize
  // from the enclosing fused shader context.
  function createGen(
    kernel: Kernel,
    prefix: string,
    resolveGlobalIndex: ResolveGlobalIndex,
    variableOverrides?: Map<string, string>,
    gidxOverride?: AluExp,
    ridxOverride?: AluExp,
    globalIndexRemap?: Map<number, GlobalIndexRemapInfo>,
    bankPadDims?: Map<number, number>,
  ): (exp: AluExp) => string {
    return createWgslGen({
      kernel,
      prefix,
      resolveGlobalIndex,
      emit,
      blockSize,
      variableOverrides,
      gidxOverride,
      ridxOverride,
      globalIndexRemap,
      bankPadDims,
    });
  }

  // --- Generate codegen entries ---
  // Counter for unique WGSL variable names across kernel steps when emitting
  // shared halo coordinate let-bindings (avoids name collisions in shader scope).
  let haloCoordVarSeq = 0;
  for (const entry of codegenEntries) {
    if (entry.type === "kernel") {
      const si = entry.kernelIdx;
      if (needsBarrierBefore.has(si)) {
        emit("workgroupBarrier();");
      }

      const { step, kernel } = kernelSteps[si];

      // Build input name mapping for this step
      const stepInputNames: string[] = [];
      const stepInputIsGlobal: boolean[] = [];
      const stepInputBodyIdx: number[] = [];
      const stepInputIsBlockIndex: boolean[] = [];
      for (let j = 0; j < step.inputs.length; j++) {
        const jitId = step.inputs[j];
        if (hasBlockIndex && jitId === blockIndexInputId) {
          stepInputNames.push("_block_idx_scalar");
          stepInputIsGlobal.push(false);
          stepInputBodyIdx.push(-1);
          stepInputIsBlockIndex.push(true);
          continue;
        }
        const name = idToReadName.get(jitId) ?? inputIdToName.get(jitId);
        if (name) {
          stepInputNames.push(name);
          const isGlobalInput = inputIdToName.has(jitId);
          stepInputIsGlobal.push(isGlobalInput);
          const bodyIdx = bodyInputIds.indexOf(jitId);
          stepInputBodyIdx.push(
            bodyIdx >= numConsts ? bodyIdx - numConsts : -1,
          );
        } else {
          stepInputNames.push(`__unknown_${jitId}`);
          stepInputIsGlobal.push(false);
          stepInputBodyIdx.push(-1);
        }
        stepInputIsBlockIndex.push(false);
      }

      // Per kernel step: track emitted halo coordinate let-bindings so the
      // same (inputIdx, indexExpr) pair emits WGSL let declarations only once.
      const emittedHaloCoords = new Map<string, string[]>();

      const gen_resolve = (
        bufIdx: number,
        indexExpr: string,
        dtype: DType,
      ): string => {
        if (stepInputIsBlockIndex[bufIdx]) return `i32(block_idx)`;
        if (stepInputIsGlobal[bufIdx]) {
          const inputIdx = stepInputBodyIdx[bufIdx];
          if (inputIdx >= 0) {
            const name = stepInputNames[bufIdx];
            const zero = `${dtypeToWgsl(dtype)}(0)`;
            if (inHasHalo[inputIdx]) {
              // Zero-copy halo: per-axis coordinate bounds check.
              // When both stride remap and halo are needed (the conv case),
              // compute per-axis coordinates ONCE and emit WGSL let bindings
              // so the GPU executes the divide/modulo ALU only once.
              // Both the global-index remap and the per-axis OOB check then
              // reference the same bound variables.
              let coordVars: string[];
              if (inNeedsRemap[inputIdx]) {
                const coordKey = `${inputIdx}:${indexExpr}`;
                const cached = emittedHaloCoords.get(coordKey);
                if (cached) {
                  coordVars = cached;
                } else {
                  const rawCoords = inCoordExprs(inputIdx, indexExpr);
                  const tag = haloCoordVarSeq++;
                  coordVars = rawCoords.map((expr, d) => {
                    const vn = `_hc${tag}_${d}`;
                    emit(`let ${vn}: i32 = ${expr};`);
                    return vn;
                  });
                  emittedHaloCoords.set(coordKey, coordVars);
                }
              } else {
                // No remap needed; coordinates only used for halo check.
                coordVars = inCoordExprs(inputIdx, indexExpr);
              }
              // Build remapped index from shared coordinates.
              const remapped = inRemap(inputIdx, indexExpr, coordVars);

              const shift = inHaloShift[inputIdx];
              const total = inTotalElems[inputIdx];
              const packOff = inPackOffset[inputIdx];
              const lo = packOff; // lower bound in packed coords
              const hi = packOff + total; // exclusive upper bound
              const rawExpr = `(i32(in_base_${inputIdx}) + ${remapped} - ${shift})`;
              const safeExpr = `clamp(${rawExpr}, ${lo}, ${hi - 1})`;
              const readExpr = `${name}[${safeExpr}]`;

              const axes = params.inAxes[inputIdx];
              const validParts: string[] = [];
              for (let g = 0; g < gridRank; g++) {
                if (axes[g] === null) continue;
                const [loH, hiH] = params.halo?.[inputIdx]?.[g] ?? [0, 0];
                if (loH === 0 && hiH === 0) continue;
                const ax = axes[g]!;
                const globalDim = params.inputShapes[inputIdx][ax];
                const coordExpr = coordVars[ax];
                // Global coordinate = block_start + body_coord - halo_lo
                const blockStart =
                  gridOffset > 0
                    ? `i32((block_i${g} + ${gridOffset}u) * ${blockShape[g]}u)`
                    : `(i32(block_i${g}) * ${blockShape[g]})`;
                validParts.push(
                  `(${blockStart} + ${coordExpr} - ${loH} >= 0 && ${blockStart} + ${coordExpr} - ${loH} < ${globalDim})`,
                );
              }

              const haloValid =
                validParts.length > 0
                  ? validParts.join(" && ")
                  : `(${rawExpr} >= ${lo} && ${rawExpr} < ${hi})`;
              const validExpr = hasBoundary
                ? `(valid && ${haloValid})`
                : haloValid;
              return `select(${zero}, ${readExpr}, ${validExpr})`;
            }
            const remapped = inRemap(inputIdx, indexExpr);
            const readExpr = `${name}[i32(in_base_${inputIdx}) + ${remapped}]`;
            return hasBoundary
              ? `select(${zero}, ${readExpr}, valid)`
              : readExpr;
          } else {
            const ci = bodyInputIds.indexOf(step.inputs[bufIdx]);
            return readConst(ci, indexExpr);
          }
        } else {
          return `${stepInputNames[bufIdx]}[${indexExpr}]`;
        }
      };

      const gen = createGen(kernel, `s${si}`, gen_resolve);

      // Generate WGSL for each kernel output
      const re = stepReductions[si];
      const kernelSize = kernel.size as number;
      const elemsPerThread =
        kernelSize >= blockSize ? kernelSize / blockSize : 1;

      if (re && kernelSize === 1) {
        // --- Workgroup tree reduction (kernel.size == 1) ---
        // Each thread evaluates the expression at its own index (ridx → gidx,
        // i.e., tidx), then threads cooperatively tree-reduce the results.
        const treeGen = createGen(
          kernel,
          `s${si}`,
          gen_resolve,
          undefined,
          undefined,
          // ridxOverride: map ridx → gidx so each thread loads its own element
          AluExp.special(DType.Int32, "gidx", blockSize),
        );
        const outId = step.outputs[0];
        const rhs = strip1(treeGen(kernel.outputs[0].exp));
        const reDtype = re.dtype;
        const reTy = dtypeToWgsl(reDtype, false);
        const wsName = `reduce_ws_${si}`;

        const useSubgroupReduce = hasSubgroups && reDtype !== DType.Bool;

        if (useSubgroupReduce) {
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
          emit(`var _sg_val_${si}: ${reTy} = ${sgOp}(${reTy}(${rhs}));`);
          // Phase 2: leaders store packed subgroup results.
          emit(`let _sg_leader_${si}: bool = (tidx % sg_size) == 0u;`);
          emit(`let _sg_idx_${si}: u32 = tidx / sg_size;`);
          emit(`if (_sg_leader_${si}) {`);
          emit(pushIndent);
          emit(`${wsName}[_sg_idx_${si}] = _sg_val_${si};`);
          emit(popIndent, "}");
          emit("workgroupBarrier();");
          // Phase 3: inter-subgroup tree reduction.
          let startStride = 1;
          while (startStride * 2 < blockSize) startStride *= 2;
          for (let stride = startStride; stride >= 1; stride >>= 1) {
            emit(`if (${stride}u * sg_size < ${blockSize}u) {`);
            emit(pushIndent);
            emit(
              `if (_sg_leader_${si} && _sg_idx_${si} < ${stride}u && (_sg_idx_${si} + ${stride}u) * sg_size < ${blockSize}u) {`,
              pushIndent,
            );
            const thisSlot = `${wsName}[_sg_idx_${si}]`;
            const otherSlot = `${wsName}[_sg_idx_${si} + ${stride}u]`;
            if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
            else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
            else if (re.op === AluOp.Min)
              emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
            else if (re.op === AluOp.Max)
              emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
            emit(popIndent, "}");
            emit("workgroupBarrier();");
            emit(popIndent, "}");
          }
        } else {
          // Original tree reduction: store to shmem, barrier, halving
          // stride loop.
          emit(`${wsName}[tidx] = ${reTy}(${rhs});`);
          emit("workgroupBarrier();");

          let startStride = 1;
          while (startStride * 2 < blockSize) startStride *= 2;

          for (let stride = startStride; stride >= 1; stride >>= 1) {
            emit(
              `if (tidx < ${stride}u && tidx + ${stride}u < ${blockSize}u) {`,
              pushIndent,
            );
            const thisSlot = `${wsName}[tidx]`;
            const otherSlot = `${wsName}[tidx + ${stride}u]`;
            if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
            else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
            else if (re.op === AluOp.Min) {
              if (reDtype === DType.Bool)
                emit(`${thisSlot} = ${thisSlot} && ${otherSlot};`);
              else emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
            } else if (re.op === AluOp.Max) {
              if (reDtype === DType.Bool)
                emit(`${thisSlot} = ${thisSlot} || ${otherSlot};`);
              else emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
            }
            emit(popIndent, "}");
            emit("workgroupBarrier();");
          }
        }

        emit("if (tidx == 0u) {", pushIndent);

        const accVar = `${wsName}[0u]`;
        let finalValue: string;
        const isIdentityEpilogue =
          re.epilogue.op === AluOp.Variable && re.epilogue.arg === "acc";
        if (isIdentityEpilogue) {
          finalValue = accVar;
        } else {
          const epilogueGen = createGen(
            kernel,
            `s${si}_ep`,
            gen_resolve,
            new Map([["acc", accVar]]),
          );
          finalValue = strip1(epilogueGen(re.epilogue));
        }

        const resultIdx = bodyOutputIds.indexOf(outId);
        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          const castFinal =
            resultTy !== reTy ? `${resultTy}(${finalValue})` : finalValue;
          emit(
            `${resultVar(resultIdx)}[i32(out_base_${resultIdx})] = ${castFinal};`,
          );
        } else if (idIsShmem.has(outId)) {
          const shmemName = idToReadName.get(outId)!;
          emit(`${shmemName}[0u] = ${finalValue};`);
        }
        emit(popIndent, "}");
        emit("workgroupBarrier();");
      } else if (re) {
        // --- Per-element reduction: gidx loop + ridx accumulation ---
        // Each thread handles elemsPerThread output elements, each with its
        // own reduction loop. No shared-memory tree reduction needed.
        const outId = step.outputs[0];
        const reDtype = re.dtype;
        const reTy = dtypeToWgsl(reDtype, false);
        const bodyDtype = kernel.outputs[0].dtype;
        const reSize = re.size as number;
        const bodyExp = kernel.outputs[0].exp;
        const prefix = `s${si}`;
        const isIdentityEpilogue =
          re.epilogue.op === AluOp.Variable && re.epilogue.arg === "acc";

        // Open gidx loop
        emit(
          `for (var _ept_${si}: i32 = 0; _ept_${si} < ${elemsPerThread}; _ept_${si}++) {`,
          pushIndent,
        );
        emit(`gidx = i32(tidx) * ${elemsPerThread} + _ept_${si};`);

        // Accumulator
        const accName = `${prefix}_acc`;
        emit(`var ${accName}: ${reTy} = ${constToWgsl(reDtype, re.identity)};`);

        // Reduction loop — unroll for small sizes
        const UNROLL_THRESHOLD = 8;

        if (reSize <= UNROLL_THRESHOLD) {
          for (let ri = 0; ri < reSize; ri++) {
            const riGen = createGen(
              kernel,
              `${prefix}_r${ri}`,
              gen_resolve,
              undefined,
              undefined,
              AluExp.i32(ri),
            );
            const rhs = strip1(riGen(bodyExp));
            const castRhs = wgslCastReductionRhs(rhs, bodyDtype, reDtype);
            emit(wgslReductionAccumStmt(re.op, accName, castRhs));
          }
        } else {
          const ridxGen = createGen(kernel, `${prefix}_rl`, gen_resolve);
          emit(
            `for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
            pushIndent,
          );
          const rhs = strip1(ridxGen(bodyExp));
          const castRhs = wgslCastReductionRhs(rhs, bodyDtype, reDtype);
          emit(wgslReductionAccumStmt(re.op, accName, castRhs));
          emit(popIndent, "}");
        }

        // Apply epilogue
        let finalValue: string;
        if (isIdentityEpilogue) {
          finalValue = accName;
        } else {
          const epGen = createGen(
            kernel,
            `${prefix}_ep`,
            gen_resolve,
            new Map([["acc", accName]]),
          );
          finalValue = strip1(epGen(re.epilogue));
        }

        // Write result
        const resultIdx = bodyOutputIds.indexOf(outId);
        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          const castFinal =
            resultTy !== reTy ? `${resultTy}(${finalValue})` : finalValue;
          const reOutOff = outOffsetMultiElem(resultIdx, "u32(gidx)");
          if (hasBoundary) {
            emit(`if (valid) {`, pushIndent);
            emit(
              `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${reOutOff})] = ${castFinal};`,
            );
            emit(popIndent, "}");
          } else {
            emit(
              `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${reOutOff})] = ${castFinal};`,
            );
          }
        } else if (idIsShmem.has(outId)) {
          const shmemName = idToReadName.get(outId)!;
          emit(`${shmemName}[gidx] = ${finalValue};`);
        }

        // Close gidx loop
        emit(popIndent, "}");
        // Barrier so all threads finish before dependent steps read
        emit("workgroupBarrier();");
      } else {
        // --- Elementwise kernel ---
        // When elemsPerThread > 1, wrap in a gidx loop.
        const needsGidxLoop = elemsPerThread > 1;
        if (needsGidxLoop) {
          emit(
            `for (var _ept_${si}: i32 = 0; _ept_${si} < ${elemsPerThread}; _ept_${si}++) {`,
            pushIndent,
          );
          emit(`gidx = i32(tidx) * ${elemsPerThread} + _ept_${si};`);
        }

        for (let oi = 0; oi < kernel.numOutputs; oi++) {
          const outId = step.outputs[oi];
          const rhs = strip1(gen(kernel.outputs[oi].exp));

          const resultIdx = bodyOutputIds.indexOf(outId);
          if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
            const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
            const castRhs =
              resultTy !== dtypeToWgsl(kernel.outputs[oi].exp.dtype)
                ? `${resultTy}(${rhs})`
                : rhs;
            const offsetExpr = needsGidxLoop
              ? outOffsetMultiElem(resultIdx, "u32(gidx)")
              : outOffset(resultIdx);
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${offsetExpr})] = ${castRhs};`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${offsetExpr})] = ${castRhs};`,
              );
            }
          } else if (idIsShmem.has(outId)) {
            const shmemName = idToReadName.get(outId)!;
            emit(`${shmemName}[${needsGidxLoop ? "gidx" : "tidx"}] = ${rhs};`);
          }
        }

        if (needsGidxLoop) {
          emit(popIndent, "}");
        }
      }
    } else if (entry.type === "fori_loop") {
      // --- fori_loop step ---
      const fl = foriLoops[entry.flIdx];
      const fs = fl.foriStep;
      const foriLower = concreteDim(
        fs.lower,
        "block_map fused fori_loop lower",
      );
      const foriUpper = concreteDim(
        fs.upper,
        "block_map fused fori_loop upper",
      );
      const numBodyConsts = fl.numConsts;
      const numCarries = fs.initCarries.length;

      // Build mapping from body input JitIds to WGSL names + properties
      // Body inputs: [const0..constN, idx, carry0..carryM]
      const bodyInputInfo: {
        name: string;
        isGlobal: boolean;
        parentInputIdx: number; // for block offset
        isIndex: boolean;
        isScalar: boolean; // P0a: promoted scalar — use name directly, no indexing
      }[] = [];

      for (let bi = 0; bi < fl.bodyInputIds.length; bi++) {
        if (bi < numBodyConsts) {
          // Const input — maps to a parent JitId
          const parentJitId = fs.consts[bi];
          const parentName =
            idToReadName.get(parentJitId) ??
            inputIdToName.get(parentJitId) ??
            `__fl_unknown_${parentJitId}`;
          const isGlobalInput = inputIdToName.has(parentJitId);
          const parentBodyIdx = bodyInputIds.indexOf(parentJitId);
          bodyInputInfo.push({
            name: parentName,
            isGlobal: isGlobalInput && !idIsShmem.has(parentJitId),
            parentInputIdx:
              parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1,
            isIndex: false,
            isScalar: false,
          });
        } else if (bi === numBodyConsts) {
          // Loop index — scalar i32
          bodyInputInfo.push({
            name: fl.loopVar,
            isGlobal: false,
            parentInputIdx: -1,
            isIndex: true,
            isScalar: false,
          });
        } else {
          // Carry input — maps to the parent output shmem
          const carryIdx = bi - numBodyConsts - 1;
          const parentOutId = fs.outputs[carryIdx];
          const carryShmemName = idToReadName.get(parentOutId)!;
          bodyInputInfo.push({
            name: carryShmemName,
            isGlobal: false,
            parentInputIdx: -1,
            isIndex: false,
            isScalar: false,
          });
        }
      }

      // Build mapping from body JitIds to WGSL read names
      const bodyIdToName = new Map<JitId, string>();
      for (let bi = 0; bi < fl.bodyInputIds.length; bi++) {
        bodyIdToName.set(fl.bodyInputIds[bi], bodyInputInfo[bi].name);
      }
      for (const [bodyJitId, info] of fl.bodyShmemMap) {
        bodyIdToName.set(bodyJitId, info.name);
      }
      // P0a: add promoted scalar `let` names
      for (const [bodyJitId, info] of fl.promotedScalars) {
        bodyIdToName.set(bodyJitId, info.name);
      }
      // Map body output JitIds to carry output shmem names
      // (body outputs that are also body shmem intermediates already mapped)
      // For outputs that map to parent output shmem, add the mapping
      for (let ci = 0; ci < numCarries; ci++) {
        const bodyOutId = fl.bodyOutputIds[ci];
        const parentOutId = fs.outputs[ci];
        const carryShmemName = idToReadName.get(parentOutId)!;
        // If the body output is a body shmem intermediate, it already has
        // a body shmem name. We want it to write to the carry shmem instead.
        // Override the mapping:
        bodyIdToName.set(bodyOutId, carryShmemName);
      }

      // Resolve the shmem dtype for a body JitId. Checks carry shmem
      // (parent shmemMap) first, then body-internal bodyShmemMap.
      const bodyOutShmemDtype = (outId: JitId): DType | null => {
        for (let ci = 0; ci < numCarries; ci++) {
          if (fl.bodyOutputIds[ci] === outId) {
            const entry = shmemMap.get(fs.outputs[ci]);
            if (entry) return entry.dtype;
          }
        }
        const bEntry = fl.bodyShmemMap.get(outId);
        if (bEntry) return bEntry.dtype;
        return null;
      };

      // ================================================================
      // Carry initialization, loop body, and output write.
      // O4: register-tiled path when threadTile promotes carries to registers.
      // Standard 1:1 path otherwise.
      // ================================================================

      // Common resolveGlobalIndex factory for fori_loop body kernels.
      // When usePrivate=true and a buffer is in privateVarNames, reads
      // return the private register array indexed by _rt_idx.
      let bpCount = 0; // Counter for bank-pad CSE temp variables
      const makeBodyResolve =
        (
          bStepInputInfo: typeof bodyInputInfo,
          bStep: Extract<JitStep, { type: "execute" }>,
          privateVarNames: Map<JitId, string> | null,
          bankPadMap?: Map<JitId, number>,
        ) =>
        (
          bufIdx: number,
          indexExpr: string,
          dtype: DType,
          isGloballyStrided?: boolean,
          bankPadApplied?: boolean,
        ): string => {
          if (privateVarNames) {
            const jitId = bStep.inputs[bufIdx];
            const privName = privateVarNames.get(jitId);
            if (privName) return `${privName}[_rt_idx]`;
          }
          const info = bStepInputInfo[bufIdx];
          if (info.isIndex) return `${dtypeToWgsl(dtype)}(${info.name})`;
          if (info.isScalar) return `${dtypeToWgsl(dtype)}(${info.name})`;
          if (info.name === BLOCK_IDX_SENTINEL) return `i32(block_idx)`;
          if (info.isGlobal) {
            const inputIdx = info.parentInputIdx;
            if (inputIdx >= 0) {
              // When isGloballyStrided, the index was already remapped at the
              // AluExp level by createGen's globalIndexRemap — skip string inRemap.
              const remapped = isGloballyStrided
                ? indexExpr
                : inRemap(inputIdx, indexExpr);
              const readExpr = `${info.name}[i32(in_base_${inputIdx}) + ${remapped}]`;
              return hasBoundary
                ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
                : readExpr;
            }
            return `${info.name}[${indexExpr}]`;
          }
          // O11: bank-padded shmem read with CSE for complex index expressions
          // O11b: skip string-level bank-pad when already applied at AluExp level
          if (bankPadMap && !bankPadApplied) {
            const jitId = bStep.inputs[bufIdx];
            const innerDim = bankPadMap.get(jitId);
            if (innerDim !== undefined) {
              // Simple identifiers/numbers don't need a let binding
              if (/^\w+$/.test(indexExpr)) {
                return `${info.name}[${indexExpr} + ${indexExpr} / ${innerDim}]`;
              }
              const bp = `_bp${bpCount++}`;
              emit(`let ${bp}: i32 = ${indexExpr};`);
              return `${info.name}[${bp} + ${bp} / ${innerDim}]`;
            }
          }
          return `${info.name}[${indexExpr}]`;
        };

      // Common: build per-step input mapping for a body kernel
      const buildStepInputInfo = (
        bStep: Extract<JitStep, { type: "execute" }>,
      ) => {
        const bStepInputInfo: typeof bodyInputInfo = [];
        for (let j = 0; j < bStep.inputs.length; j++) {
          const jitId = bStep.inputs[j];
          const biIdx = fl.bodyInputIds.indexOf(jitId);
          if (biIdx >= 0) {
            bStepInputInfo.push(bodyInputInfo[biIdx]);
          } else {
            const sname = bodyIdToName.get(jitId);
            const isPromotedScalar = fl.promotedScalars.has(jitId);
            bStepInputInfo.push({
              name: sname ?? `__fl_unknown_${jitId}`,
              isGlobal: false,
              parentInputIdx: -1,
              isIndex: false,
              isScalar: isPromotedScalar,
            });
          }
        }
        return bStepInputInfo;
      };

      // Build AluExp-level GlobalIndex remap info for a body step.
      // For global inputs where body strides differ from buffer strides,
      // provides the shape/strides so createGen can remap at the AluExp level
      // (where simplify() can eliminate redundant div/mod).
      const buildStepRemap = (
        bStepInputInfo: typeof bodyInputInfo,
      ): Map<number, GlobalIndexRemapInfo> | undefined => {
        let remap: Map<number, GlobalIndexRemapInfo> | undefined;
        for (let j = 0; j < bStepInputInfo.length; j++) {
          const info = bStepInputInfo[j];
          if (
            info.isGlobal &&
            info.parentInputIdx >= 0 &&
            inNeedsRemap[info.parentInputIdx]
          ) {
            remap ??= new Map();
            remap.set(j, {
              bodyShape: inBodyShapes[info.parentInputIdx],
              bodyStrides: inBodyStrides[info.parentInputIdx],
              globalStrides: inBufStrides[info.parentInputIdx],
            });
          }
        }
        return remap;
      };

      const useRegisterTiling = !!(threadTile && fl.privateShmemIds.size > 0);

      // O11b: Build bufIdx → innerDim map for AluExp-level bank-pad rewrite.
      const buildStepBankPad = (
        bStep: Extract<JitStep, { type: "execute" }>,
      ): Map<number, number> | undefined => {
        if (!fl.shmemBankPad) return undefined;
        let dims: Map<number, number> | undefined;
        for (let j = 0; j < bStep.inputs.length; j++) {
          const innerDim = fl.shmemBankPad.get(bStep.inputs[j]);
          if (innerDim !== undefined) {
            dims ??= new Map();
            dims.set(j, innerDim);
          }
        }
        return dims;
      };

      if (useRegisterTiling) {
        // ==============================================================
        // O4: Register-tiled fori_loop
        // Each thread handles threadTile[g] elements per grid axis.
        // Carries and their dependency chain live in var arrays (registers).
        // Cooperative steps (tile loads, etc.) use stride loops.
        // ==============================================================
        const tileElems = prod(threadTile!);

        // Tile dimension constants (help Tint range analysis)
        for (let g = 0; g < gridRank; g++) {
          emit(`const RT_T${g}: u32 = ${threadTile![g]}u;`);
        }

        // Inner products for flat gidx: gidx = sum_g((tidx_g*T_g+_rt_g)*innerProd_g)
        const innerProds: number[] = new Array(gridRank);
        innerProds[gridRank - 1] = 1;
        for (let g = gridRank - 2; g >= 0; g--) {
          innerProds[g] = innerProds[g + 1] * blockShape[g + 1];
        }

        // O5: Structured gidx for AluExp simplification in register-tiled path.
        // Instead of a flat Special("gidx", size), represent gidx as
        // coord0 * stride0 + coord1 with bounded per-axis coords.
        // This lets the simplifier eliminate div/mod: (c0*S+c1)/S → c0.
        let structuredGidx: AluExp = AluExp.special(
          DType.Int32,
          `_rt_c${gridRank - 1}`,
          blockShape[gridRank - 1],
        );
        for (let g = gridRank - 2; g >= 0; g--) {
          const coord = AluExp.special(DType.Int32, `_rt_c${g}`, blockShape[g]);
          structuredGidx = AluExp.add(
            AluExp.mul(coord, AluExp.const(DType.Int32, innerProds[g])),
            structuredGidx,
          );
        }

        const flatGidxExpr = (): string => {
          const terms: string[] = [];
          for (let g = 0; g < gridRank; g++) {
            const pos = `(tidx_${g} * RT_T${g} + _rt_${g})`;
            if (innerProds[g] === 1) terms.push(pos);
            else terms.push(`${pos} * ${innerProds[g]}u`);
          }
          return terms.join(" + ");
        };

        // _rt_idx: flat index into private array [0, tileElems)
        const rtIdxExpr = (): string => {
          if (gridRank === 1) return `_rt_0`;
          const strides: number[] = new Array(gridRank);
          strides[gridRank - 1] = 1;
          for (let g = gridRank - 2; g >= 0; g--) {
            strides[g] = strides[g + 1] * threadTile![g + 1];
          }
          const terms: string[] = [];
          for (let g = 0; g < gridRank; g++) {
            if (strides[g] === 1) terms.push(`_rt_${g}`);
            else terms.push(`_rt_${g} * ${strides[g]}u`);
          }
          return terms.join(" + ");
        };

        const emitTileLoopOpen = () => {
          for (let g = 0; g < gridRank; g++) {
            emit(
              `for (var _rt_${g}: u32 = 0u; _rt_${g} < RT_T${g}; _rt_${g}++) {`,
              pushIndent,
            );
          }
          emit(`let _rt_idx: u32 = ${rtIdxExpr()};`);
        };
        const emitTileLoopClose = () => {
          for (let g = gridRank - 1; g >= 0; g--) {
            emit(popIndent, "}");
          }
        };

        // O12b: Pre-scan body kernels to detect carry-accumulating reductions.
        // When a fused reduction kernel outputs to a carry and its epilogue
        // is Add(acc_term, carry_read), we can accumulate directly into the
        // carry. For f16 matmul, this also promotes the carry to f32 for
        // precision (the inner reduction already uses an f32 accumulator).
        const carryPromotedDtype = new Map<number, DType>(); // ci → promoted dtype
        {
          // Build set of carry input JitIds for quick lookup
          const carryInputJitIds = new Set<JitId>();
          for (let ci = 0; ci < numCarries; ci++) {
            const carryInputIdx = numBodyConsts + 1 + ci;
            if (carryInputIdx < fl.bodyInputIds.length) {
              carryInputJitIds.add(fl.bodyInputIds[carryInputIdx]);
            }
          }
          // Check each body kernel for the carry-accumulation pattern
          for (const { step: bStep, kernel: bKernel } of fl.bodyKernels) {
            const bRe = bKernel.outputs[0]?.reduction ?? null;
            if (!bRe || bRe.op !== AluOp.Add) continue;
            const outId = bStep.outputs[0];
            // Must be a carry output
            const ci = fl.bodyOutputIds.indexOf(outId);
            if (ci < 0 || ci >= numCarries) continue;
            if (!fl.privateShmemIds.has(outId)) continue;
            // Detect epilogue pattern: Add(acc_term, carry_gv) or Add(carry_gv, acc_term)
            const ep = bRe.epilogue;
            if (ep.op !== AluOp.Add) continue;
            const isAccTerm = (e: AluExp): boolean =>
              (e.op === AluOp.Variable && e.arg === "acc") ||
              (e.op === AluOp.Cast && isAccTerm(e.src[0]));
            const isCarryGV = (e: AluExp): boolean => {
              if (e.op !== AluOp.GlobalView && e.op !== AluOp.GlobalIndex)
                return false;
              const gid = e.arg[0] as number;
              if (gid >= bStep.inputs.length) return false;
              return carryInputJitIds.has(bStep.inputs[gid]);
            };
            if (
              (isAccTerm(ep.src[0]) && isCarryGV(ep.src[1])) ||
              (isAccTerm(ep.src[1]) && isCarryGV(ep.src[0]))
            ) {
              // This carry's reduction accumulates f32 products into it.
              // Promote carry to f32 if reduction dtype is wider.
              if (bRe.dtype !== DType.Float32) continue; // only promote to f32
              const carryDtype =
                shmemMap.get(fs.outputs[ci])?.dtype ?? DType.Float32;
              if (carryDtype !== bRe.dtype) {
                carryPromotedDtype.set(ci, bRe.dtype);
              }
            }
          }
        }

        // Declare private register arrays for carries and private intermediates
        const privateVarNames = new Map<JitId, string>();
        for (let ci = 0; ci < numCarries; ci++) {
          const bodyOutId = fl.bodyOutputIds[ci];
          if (!fl.privateShmemIds.has(bodyOutId)) continue;
          const parentOutId = fs.outputs[ci];
          const varName = `rt_carry_${ci}`;
          privateVarNames.set(bodyOutId, varName);
          privateVarNames.set(parentOutId, varName);
          // Also map the carry input body JitId (read side in body program)
          const carryInputIdx = numBodyConsts + 1 + ci; // after consts + idx
          if (carryInputIdx < fl.bodyInputIds.length) {
            privateVarNames.set(fl.bodyInputIds[carryInputIdx], varName);
          }
          // O12b: Use promoted dtype (f32) for carries with f16 Add reductions
          const dtype =
            carryPromotedDtype.get(ci) ??
            shmemMap.get(parentOutId)?.dtype ??
            DType.Float32;
          emit(
            `var ${varName}: array<${dtypeToWgsl(dtype, false)}, ${tileElems}>;`,
          );
        }
        for (const [bodyJitId, info] of fl.bodyShmemMap) {
          if (!fl.privateShmemIds.has(bodyJitId)) continue;
          if (privateVarNames.has(bodyJitId)) continue;
          const varName = `rt_priv_${bodyJitId}`;
          privateVarNames.set(bodyJitId, varName);
          emit(
            `var ${varName}: array<${dtypeToWgsl(info.dtype, false)}, ${tileElems}>;`,
          );
        }

        // Initialize private carries from init values via tile loops
        for (let ci = 0; ci < numCarries; ci++) {
          const bodyOutId = fl.bodyOutputIds[ci];
          if (!fl.privateShmemIds.has(bodyOutId)) continue;
          const parentInitId = fs.initCarries[ci];
          const initName =
            idToReadName.get(parentInitId) ?? inputIdToName.get(parentInitId);
          const varName = privateVarNames.get(bodyOutId)!;
          const isGlobalInit =
            inputIdToName.has(parentInitId) && !idIsShmem.has(parentInitId);
          const parentBodyIdx = bodyInputIds.indexOf(parentInitId);
          const inIdx =
            parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;
          // O12b: wrap init value in a cast when carry is promoted to f32
          const promoted = carryPromotedDtype.get(ci);
          const wrapInit = (expr: string) =>
            promoted ? `${dtypeToWgsl(promoted, false)}(${expr})` : expr;

          if (initName) {
            emitTileLoopOpen();
            emit(`gidx = i32(${flatGidxExpr()});`);
            if (isGlobalInit && inIdx >= 0) {
              emit(
                `${varName}[_rt_idx] = ${wrapInit(`${initName}[i32(in_base_${inIdx}) + ${inRemap(inIdx, "gidx")}]`)};`,
              );
            } else {
              emit(
                `${varName}[_rt_idx] = ${wrapInit(`${initName}[u32(gidx)]`)};`,
              );
            }
            emitTileLoopClose();
          }
        }
        // Initialize non-private carries to shmem (standard copy)
        for (let ci = 0; ci < numCarries; ci++) {
          const bodyOutId = fl.bodyOutputIds[ci];
          if (fl.privateShmemIds.has(bodyOutId)) continue;
          const parentInitId = fs.initCarries[ci];
          const parentOutId = fs.outputs[ci];
          const initName =
            idToReadName.get(parentInitId) ?? inputIdToName.get(parentInitId);
          const carryName = idToReadName.get(parentOutId)!;
          if (initName && initName !== carryName) {
            if (
              inputIdToName.has(parentInitId) &&
              !idIsShmem.has(parentInitId)
            ) {
              const parentBodyIdx = bodyInputIds.indexOf(parentInitId);
              const inIdx =
                parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;
              if (inIdx >= 0) {
                const remapped = inRemap(inIdx, "i32(tidx)");
                const readExpr = `${initName}[i32(in_base_${inIdx}) + ${remapped}]`;
                emit(
                  `${carryName}[tidx] = ${hasBoundary ? `select(${dtypeToWgsl(shmemMap.get(parentOutId)?.dtype ?? DType.Float32)}(0), ${readExpr}, valid)` : readExpr};`,
                );
              } else {
                emit(`${carryName}[tidx] = ${initName}[i32(tidx)];`);
              }
            } else {
              emit(`${carryName}[tidx] = ${initName}[tidx];`);
            }
          }
        }
        emit("workgroupBarrier();");

        // For loop
        emit(
          `for (var ${fl.loopVar}: i32 = ${foriLower}; ${fl.loopVar} < ${foriUpper}; ${fl.loopVar}++) {`,
          pushIndent,
        );

        // Snapshot ALL carries at the start of each loop iteration.
        // Private (register) carries: full array copied to immutable `let`.
        //   Cost: tileElems (typically 16–64) register copies — negligible vs kernel work.
        //   WGSL `let` on small fixed-size arrays compiles to register moves that the
        //   driver can often elide entirely when the snapshot is consumed once.
        // Non-private (shmem) carries: scalar `let` snapshot (same pattern as standard path).
        // Prevents read-after-write hazards when multiple body kernels reference carries.
        for (let ci = 0; ci < numCarries; ci++) {
          const bodyOutId = fl.bodyOutputIds[ci];
          const carryInputIdx = numBodyConsts + 1 + ci;
          const snapName = `_rt_carry_snap_${ci}`;

          if (fl.privateShmemIds.has(bodyOutId)) {
            // Private carry: full array snapshot into registers
            emit(`let ${snapName} = rt_carry_${ci};`);
            if (carryInputIdx < fl.bodyInputIds.length) {
              privateVarNames.set(fl.bodyInputIds[carryInputIdx], snapName);
            }
          } else {
            // Non-private (shmem) carry: scalar snapshot.
            // Currently unreachable — all carries are seeded into privateShmemIds
            // when threadTile is set. Kept as a defensive path in case the
            // privatization seeding logic changes.
            const parentOutId = fs.outputs[ci];
            const dtype = shmemMap.get(parentOutId)?.dtype ?? DType.Float32;
            const ty = dtypeToWgsl(dtype);
            const shmemName = idToReadName.get(parentOutId)!;
            emit(`let ${snapName}: ${ty} = ${shmemName}[tidx];`);
            if (carryInputIdx < bodyInputInfo.length) {
              bodyInputInfo[carryInputIdx] = {
                ...bodyInputInfo[carryInputIdx],
                name: snapName,
                isScalar: true,
              };
            }
          }
        }

        // Body kernel steps
        for (let bsi = 0; bsi < fl.bodyKernels.length; bsi++) {
          if (fl.bodyBarriers.has(bsi)) {
            emit("workgroupBarrier();");
          }

          const { step: bStep, kernel: bKernel } = fl.bodyKernels[bsi];
          const bKernelSize = bKernel.size as number;
          const bStepInputInfo = buildStepInputInfo(bStep);

          const isPromoted =
            bKernel.numOutputs === 1 &&
            fl.promotedScalars.has(bStep.outputs[0]);
          const isPrivateStep =
            !isPromoted &&
            bStep.outputs.every(
              (oid) =>
                fl.privateShmemIds.has(oid) || fl.promotedScalars.has(oid),
            );

          if (isPromoted) {
            // P0a: promoted scalar let binding
            const gen = createGen(
              bKernel,
              `fl${entry.flIdx}_s${bsi}`,
              makeBodyResolve(bStepInputInfo, bStep, null, fl.shmemBankPad),
              undefined,
              undefined,
              undefined,
              buildStepRemap(bStepInputInfo),
              buildStepBankPad(bStep),
            );
            const outId = bStep.outputs[0];
            const rhs = strip1(gen(bKernel.outputs[0].exp));
            const promoted = fl.promotedScalars.get(outId)!;
            const ty = dtypeToWgsl(promoted.dtype);
            emit(`let ${promoted.name}: ${ty} = ${ty}(${rhs});`);
          } else if (isPrivateStep) {
            // O4: Private step — thread-tile loop
            const bRe = bKernel.outputs[0]?.reduction ?? null;

            if (bRe && bKernelSize === blockSize) {
              // O12: Outer-product private reduction.
              // ridx outermost, tile loops inside → the GPU compiler can CSE
              // shared memory loads across tile iterations. For threadTile=[T0,T1]
              // and Bk reduction elements, this reduces shmem reads from
              // T0*T1*Bk*2 to Bk*(T0+T1) — a (2*T0*T1)/(T0+T1)× improvement.
              const outId = bStep.outputs[0];
              const reDtype = bRe.dtype;
              const reTy = dtypeToWgsl(reDtype, false);
              const reSize = bRe.size as number;
              const prefix = `fl${entry.flIdx}_s${bsi}`;
              const gen = createGen(
                bKernel,
                prefix,
                makeBodyResolve(
                  bStepInputInfo,
                  bStep,
                  privateVarNames,
                  fl.shmemBankPad,
                ),
                undefined,
                structuredGidx,
                undefined,
                buildStepRemap(bStepInputInfo),
                buildStepBankPad(bStep),
              );

              const isIdentityEpilogue =
                bRe.epilogue.op === AluOp.Variable &&
                bRe.epilogue.arg === "acc";

              // O12a/O12b: Detect carry-accumulating Add reduction.
              // After JIT fusion, the matmul body is a single kernel whose
              // epilogue has structure: Add(acc_term, carry_read).
              // acc_term is Variable("acc") [f32] or Cast(f16, Variable("acc")) [f16].
              // carry_read is a GlobalView/GlobalIndex of the carry input buffer.
              //
              // When detected, skip the separate accumulator array and accumulate
              // directly into the carry: carry[i] += product.
              // For f16: the carry is promoted to f32 (O12b pre-scan), so all
              // accumulation happens in f32 with a single cast at final writeback.
              const privName = privateVarNames.get(outId);
              let canFuseIntoCarry = false;
              if (bRe.op === AluOp.Add && privName) {
                if (isIdentityEpilogue) {
                  // Original O12a: identity epilogue (non-fused bodies)
                  canFuseIntoCarry = true;
                } else if (bRe.epilogue.op === AluOp.Add) {
                  // O12b: fused epilogue Add(acc_term, carry_gv) pattern
                  const isAccTerm = (e: AluExp): boolean =>
                    (e.op === AluOp.Variable && e.arg === "acc") ||
                    (e.op === AluOp.Cast && isAccTerm(e.src[0]));
                  const isCarryRead = (e: AluExp): boolean =>
                    e.op === AluOp.GlobalView || e.op === AluOp.GlobalIndex;
                  const [a, b] = bRe.epilogue.src;
                  canFuseIntoCarry =
                    (isAccTerm(a) && isCarryRead(b)) ||
                    (isAccTerm(b) && isCarryRead(a));
                }
              }

              const bodyExp = bKernel.outputs[0].exp;
              const bodyDtype = bodyExp.dtype;

              if (canFuseIntoCarry) {
                // Direct carry accumulation: ridx loop + tile loop.
                // O5: Manual unrolling for small reduction sizes — bypasses
                // Chrome/Tint's broken @unroll attribute by emitting the loop
                // body N times with literal ridx values. Each iteration gets
                // its own createGen() so the simplifier can constant-fold the
                // ridx-dependent shmem index expressions.
                const UNROLL_THRESHOLD = 32;
                if (reSize <= UNROLL_THRESHOLD) {
                  // Manually unrolled: emit body reSize times
                  for (let ri = 0; ri < reSize; ri++) {
                    const riGen = createGen(
                      bKernel,
                      `${prefix}_r${ri}`,
                      makeBodyResolve(
                        bStepInputInfo,
                        bStep,
                        privateVarNames,
                        fl.shmemBankPad,
                      ),
                      undefined,
                      structuredGidx,
                      AluExp.i32(ri),
                      buildStepRemap(bStepInputInfo),
                      buildStepBankPad(bStep),
                    );
                    emitTileLoopOpen();
                    for (let g = 0; g < gridRank; g++) {
                      emit(
                        `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                      );
                    }
                    const rhs = strip1(riGen(bodyExp));
                    const castRhs = wgslCastReductionRhs(
                      rhs,
                      bodyDtype,
                      reDtype,
                    );
                    emit(`${privName}[_rt_idx] += ${castRhs};`);
                    emitTileLoopClose();
                  }
                } else {
                  // Too large to unroll — use standard loop
                  emit(
                    `for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
                    pushIndent,
                  );
                  emitTileLoopOpen();
                  for (let g = 0; g < gridRank; g++) {
                    emit(
                      `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                    );
                  }
                  const rhs = strip1(gen(bodyExp));
                  const castRhs = wgslCastReductionRhs(rhs, bodyDtype, reDtype);
                  emit(`${privName}[_rt_idx] += ${castRhs};`);
                  emitTileLoopClose();
                  emit(popIndent, `}`); // end ridx
                }
              } else {
                // General case: separate accumulator array
                const accArrName = `${prefix}_acc_arr`;
                emit(`var ${accArrName}: array<${reTy}, ${tileElems}>;`);
                emitTileLoopOpen();
                emit(
                  `${accArrName}[_rt_idx] = ${constToWgsl(reDtype, bRe.identity)};`,
                );
                emitTileLoopClose();

                // O5: Manual unrolling for small reduction sizes
                const UNROLL_THRESHOLD_GEN = 32;
                const emitAccum = (accName: string, castRhs: string) => {
                  emit(
                    wgslReductionAccumStmt(
                      bRe.op,
                      `${accName}[_rt_idx]`,
                      castRhs,
                    ),
                  );
                };

                if (reSize <= UNROLL_THRESHOLD_GEN) {
                  for (let ri = 0; ri < reSize; ri++) {
                    const riGen = createGen(
                      bKernel,
                      `${prefix}_r${ri}`,
                      makeBodyResolve(
                        bStepInputInfo,
                        bStep,
                        privateVarNames,
                        fl.shmemBankPad,
                      ),
                      undefined,
                      structuredGidx,
                      AluExp.i32(ri),
                      buildStepRemap(bStepInputInfo),
                      buildStepBankPad(bStep),
                    );
                    emitTileLoopOpen();
                    for (let g = 0; g < gridRank; g++) {
                      emit(
                        `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                      );
                    }
                    const rhs = strip1(riGen(bodyExp));
                    const castRhs = wgslCastReductionRhs(
                      rhs,
                      bodyDtype,
                      reDtype,
                    );
                    emitAccum(accArrName, castRhs);
                    emitTileLoopClose();
                  }
                } else {
                  emit(
                    `for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
                    pushIndent,
                  );
                  emitTileLoopOpen();
                  for (let g = 0; g < gridRank; g++) {
                    emit(
                      `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                    );
                  }
                  const rhs = strip1(gen(bodyExp));
                  const castRhs = wgslCastReductionRhs(rhs, bodyDtype, reDtype);
                  emitAccum(accArrName, castRhs);
                  emitTileLoopClose();
                  emit(popIndent, `}`); // end ridx
                }

                // Apply epilogue and store results
                emitTileLoopOpen();
                for (let g = 0; g < gridRank; g++) {
                  emit(
                    `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                  );
                }
                emit(`gidx = i32(${flatGidxExpr()});`);

                let finalValue = `${accArrName}[_rt_idx]`;
                if (!isIdentityEpilogue) {
                  const epilogueGen = createGen(
                    bKernel,
                    `${prefix}_ep`,
                    makeBodyResolve(
                      bStepInputInfo,
                      bStep,
                      privateVarNames,
                      fl.shmemBankPad,
                    ),
                    new Map([["acc", `${accArrName}[_rt_idx]`]]),
                    structuredGidx,
                    undefined,
                    buildStepRemap(bStepInputInfo),
                    buildStepBankPad(bStep),
                  );
                  finalValue = strip1(epilogueGen(bRe.epilogue));
                }
                if (privName) {
                  const targetDtype = bodyOutShmemDtype(outId);
                  const targetTy = targetDtype
                    ? dtypeToWgsl(targetDtype)
                    : null;
                  const castFinal =
                    targetTy && targetTy !== dtypeToWgsl(reDtype)
                      ? `${targetTy}(${finalValue})`
                      : finalValue;
                  emit(`${privName}[_rt_idx] = ${castFinal};`);
                }
                emitTileLoopClose();
              }
            } else {
              // Non-reduction private step: standard tile loop
              emitTileLoopOpen();
              // O5: Emit per-axis coordinate variables so the AluExp
              // simplifier's structured gidx maps to real WGSL names.
              for (let g = 0; g < gridRank; g++) {
                emit(
                  `let _rt_c${g}: i32 = i32(tidx_${g} * RT_T${g} + _rt_${g});`,
                );
              }
              emit(`gidx = i32(${flatGidxExpr()});`);
              // Private elementwise
              const gen = createGen(
                bKernel,
                `fl${entry.flIdx}_s${bsi}`,
                makeBodyResolve(
                  bStepInputInfo,
                  bStep,
                  privateVarNames,
                  fl.shmemBankPad,
                ),
                undefined,
                structuredGidx,
                undefined,
                buildStepRemap(bStepInputInfo),
                buildStepBankPad(bStep),
              );
              for (let oi = 0; oi < bKernel.numOutputs; oi++) {
                const outId = bStep.outputs[oi];
                const rhs = strip1(gen(bKernel.outputs[oi].exp));
                const privName = privateVarNames.get(outId);
                if (privName) {
                  const targetDtype = bodyOutShmemDtype(outId);
                  const targetTy = targetDtype
                    ? dtypeToWgsl(targetDtype)
                    : null;
                  const expDtype = bKernel.outputs[oi].exp.dtype;
                  const castRhs =
                    targetTy && targetDtype !== expDtype
                      ? `${targetTy}(${rhs})`
                      : rhs;
                  emit(`${privName}[_rt_idx] = ${castRhs};`);
                }
              }

              emitTileLoopClose();
            }
          } else {
            // Cooperative step — stride loop over all elements
            emit(
              `for (var _li: u32 = tidx; _li < ${bKernelSize}u; _li += ${numThreads}u) {`,
              pushIndent,
            );
            emit(`gidx = i32(_li);`);

            const gen = createGen(
              bKernel,
              `fl${entry.flIdx}_s${bsi}`,
              makeBodyResolve(bStepInputInfo, bStep, null, fl.shmemBankPad),
              undefined,
              undefined,
              undefined,
              buildStepRemap(bStepInputInfo),
              buildStepBankPad(bStep),
            );
            for (let oi = 0; oi < bKernel.numOutputs; oi++) {
              const outId = bStep.outputs[oi];
              const rhs = strip1(gen(bKernel.outputs[oi].exp));
              const targetName = bodyIdToName.get(outId);
              if (targetName) {
                const targetDtype = bodyOutShmemDtype(outId);
                const targetTy = targetDtype ? dtypeToWgsl(targetDtype) : null;
                const expDtype = bKernel.outputs[oi].exp.dtype;
                const castRhs =
                  targetTy && targetDtype !== expDtype
                    ? `${targetTy}(${rhs})`
                    : rhs;
                // O11: bank-padded cooperative write
                const pad = fl.shmemBankPad.get(outId);
                const writeIdx = pad ? `_li + _li / ${pad}u` : "_li";
                emit(`${targetName}[${writeIdx}] = ${castRhs};`);
              }
            }

            emit(popIndent, "}");
          }
        }

        // P1: Wrap-around barrier
        if (fl.needWrapBarrier) {
          emit("workgroupBarrier();");
        }

        emit(popIndent, "}"); // end for loop

        // Restore gidx for subsequent codegen entries
        emit("gidx = i32(tidx);");

        // Write carries to global output
        for (let ci = 0; ci < numCarries; ci++) {
          const parentOutId = fs.outputs[ci];
          const resultIdx = bodyOutputIds.indexOf(parentOutId);
          if (resultIdx < 0 || passThroughOutputs.has(resultIdx)) continue;
          const bodyOutId = fl.bodyOutputIds[ci];

          if (fl.privateShmemIds.has(bodyOutId)) {
            // Private carry → write TM×TN elements per thread
            const varName = privateVarNames.get(bodyOutId)!;
            const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
            const outStrides = outBufStrides[resultIdx];

            emitTileLoopOpen();
            const oTerms: string[] = [];
            for (let g = 0; g < gridRank; g++) {
              const axes = params.outAxes[resultIdx];
              if (axes[g] !== null) {
                const pos = `(tidx_${g} * RT_T${g} + _rt_${g})`;
                const stride = outStrides[axes[g]!];
                oTerms.push(stride !== 1 ? `${pos} * ${stride}u` : pos);
              }
            }
            const oExpr = oTerms.length > 0 ? oTerms.join(" + ") : `_rt_idx`;
            emit(
              `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${oExpr})] = ${resultTy}(${varName}[_rt_idx]);`,
            );
            emitTileLoopClose();
          } else {
            // Standard shmem carry
            const carryName = idToReadName.get(parentOutId)!;
            const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${carryName}[tidx]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${carryName}[tidx]);`,
              );
            }
          }
        }
      } else {
        // ==============================================================
        // Standard 1:1 fori_loop (no register tiling)
        // ==============================================================

        // Initialize carry shmem from init carry values
        for (let ci = 0; ci < numCarries; ci++) {
          const parentInitId = fs.initCarries[ci];
          const parentOutId = fs.outputs[ci];
          const initName =
            idToReadName.get(parentInitId) ?? inputIdToName.get(parentInitId);
          const carryName = idToReadName.get(parentOutId)!;
          if (initName && initName !== carryName) {
            if (
              inputIdToName.has(parentInitId) &&
              !idIsShmem.has(parentInitId)
            ) {
              const parentBodyIdx = bodyInputIds.indexOf(parentInitId);
              const inIdx =
                parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;
              if (inIdx >= 0) {
                const remapped = inRemap(inIdx, "i32(tidx)");
                const readExpr = `${initName}[i32(in_base_${inIdx}) + ${remapped}]`;
                emit(
                  `${carryName}[tidx] = ${hasBoundary ? `select(${dtypeToWgsl(shmemMap.get(parentOutId)?.dtype ?? DType.Float32)}(0), ${readExpr}, valid)` : readExpr};`,
                );
              } else {
                emit(`${carryName}[tidx] = ${initName}[i32(tidx)];`);
              }
            } else {
              emit(`${carryName}[tidx] = ${initName}[tidx];`);
            }
          }
        }
        emit("workgroupBarrier();");

        // Carry snapshot: carry inputs and carry outputs alias the same
        // shmem array. Within one iteration, a body kernel that writes
        // carry output X overwrites shmem before a later kernel reads
        // carry input X (expecting the OLD value). Snapshot each carry
        // into a `let` binding at the start of each iteration so all
        // body kernel reads see pre-iteration values.
        const carrySnapNames: string[] = [];
        const carrySnapDtypes: string[] = [];
        for (let ci = 0; ci < numCarries; ci++) {
          const parentOutId = fs.outputs[ci];
          const dtype = shmemMap.get(parentOutId)?.dtype ?? DType.Float32;
          const ty = dtypeToWgsl(dtype);
          const snapName = `_fl${entry.flIdx}_snap${ci}`;
          carrySnapNames.push(snapName);
          carrySnapDtypes.push(ty);
          // Remap bodyInputInfo so body kernels read from snapshots
          const biIdx = numBodyConsts + 1 + ci;
          if (biIdx < bodyInputInfo.length) {
            bodyInputInfo[biIdx] = {
              ...bodyInputInfo[biIdx],
              name: snapName,
              isScalar: true,
            };
          }
        }

        // Emit WGSL for loop
        emit(
          `for (var ${fl.loopVar}: i32 = ${foriLower}; ${fl.loopVar} < ${foriUpper}; ${fl.loopVar}++) {`,
          pushIndent,
        );

        // Emit carry snapshots at the start of each iteration
        for (let ci = 0; ci < numCarries; ci++) {
          const parentOutId = fs.outputs[ci];
          const shmemName = idToReadName.get(parentOutId)!;
          emit(
            `let ${carrySnapNames[ci]}: ${carrySnapDtypes[ci]} = ${shmemName}[tidx];`,
          );
        }

        // Generate body kernel steps
        for (let bsi = 0; bsi < fl.bodyKernels.length; bsi++) {
          if (fl.bodyBarriers.has(bsi)) {
            emit("workgroupBarrier();");
          }

          const { step: bStep, kernel: bKernel } = fl.bodyKernels[bsi];
          const bStepInputInfo = buildStepInputInfo(bStep);

          const gen = createGen(
            bKernel,
            `fl${entry.flIdx}_s${bsi}`,
            makeBodyResolve(bStepInputInfo, bStep, null),
            undefined,
            undefined,
            undefined,
            buildStepRemap(bStepInputInfo),
          );

          // Check for per-thread contraction (reduction kernel)
          const bRe = bKernel.outputs[0]?.reduction ?? null;
          if (bRe && (bKernel.size as number) === blockSize) {
            const outId = bStep.outputs[0];
            const reDtype = bRe.dtype;
            const reTy = dtypeToWgsl(reDtype, false);
            const reSize = bRe.size as number;
            const prefix = `fl${entry.flIdx}_s${bsi}`;
            const accName = `${prefix}_acc`;
            emit(`{`);
            emit(
              `  var ${accName}: ${reTy} = ${constToWgsl(reDtype, bRe.identity)};`,
            );
            emit(
              `  for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
              pushIndent,
            );
            const rhs = strip1(gen(bKernel.outputs[0].exp));
            const exprDtype = bKernel.outputs[0].exp.dtype;
            const castRhs = exprDtype !== reDtype ? `${reTy}(${rhs})` : rhs;
            if (bRe.op === AluOp.Add) emit(`  ${accName} += ${castRhs};`);
            else if (bRe.op === AluOp.Mul) emit(`  ${accName} *= ${castRhs};`);
            else if (bRe.op === AluOp.Min)
              emit(`  ${accName} = min(${accName}, ${castRhs});`);
            else if (bRe.op === AluOp.Max)
              emit(`  ${accName} = max(${accName}, ${castRhs});`);
            emit(popIndent, `  }`);
            const isIdentityEpilogue =
              bRe.epilogue.op === AluOp.Variable && bRe.epilogue.arg === "acc";
            let finalValue = accName;
            if (!isIdentityEpilogue) {
              const epilogueGen = createGen(
                bKernel,
                `${prefix}_ep`,
                makeBodyResolve(bStepInputInfo, bStep, null),
                new Map([["acc", accName]]),
                undefined,
                undefined,
                buildStepRemap(bStepInputInfo),
              );
              finalValue = strip1(epilogueGen(bRe.epilogue));
            }
            const targetName = bodyIdToName.get(outId);
            if (targetName) {
              const targetDtype = bodyOutShmemDtype(outId);
              const targetTy = targetDtype ? dtypeToWgsl(targetDtype) : null;
              const castFinal =
                targetTy && targetTy !== dtypeToWgsl(reDtype)
                  ? `${targetTy}(${finalValue})`
                  : finalValue;
              emit(`  ${targetName}[tidx] = ${castFinal};`);
            }
            emit(`}`);
          } else {
            const bKernelSize = bKernel.size as number;
            const isPromoted =
              bKernel.numOutputs === 1 &&
              fl.promotedScalars.has(bStep.outputs[0]);

            if (isPromoted) {
              const outId = bStep.outputs[0];
              const rhs = strip1(gen(bKernel.outputs[0].exp));
              const promoted = fl.promotedScalars.get(outId)!;
              const ty = dtypeToWgsl(promoted.dtype);
              emit(`let ${promoted.name}: ${ty} = ${ty}(${rhs});`);
            } else {
              const needSizeGuard = bKernelSize < blockSize;
              if (needSizeGuard)
                emit(`if (tidx < ${bKernelSize}u) {`, pushIndent);
              for (let oi = 0; oi < bKernel.numOutputs; oi++) {
                const outId = bStep.outputs[oi];
                const rhs = strip1(gen(bKernel.outputs[oi].exp));
                const targetName = bodyIdToName.get(outId);
                if (targetName) {
                  const targetDtype = bodyOutShmemDtype(outId);
                  const targetTy = targetDtype
                    ? dtypeToWgsl(targetDtype)
                    : null;
                  const expDtype = bKernel.outputs[oi].exp.dtype;
                  const castRhs =
                    targetTy && targetDtype !== expDtype
                      ? `${targetTy}(${rhs})`
                      : rhs;
                  emit(`${targetName}[tidx] = ${castRhs};`);
                }
              }
              if (needSizeGuard) emit(popIndent, "}");
            }
          }
        }

        // P1: Wrap-around barrier for cross-iteration dependencies
        if (fl.needWrapBarrier) {
          emit("workgroupBarrier();");
        }

        emit(popIndent, "}"); // end for loop

        // Write final carries to parent output (global result buffers)
        for (let ci = 0; ci < numCarries; ci++) {
          const parentOutId = fs.outputs[ci];
          const resultIdx = bodyOutputIds.indexOf(parentOutId);
          if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
            const carryName = idToReadName.get(parentOutId)!;
            const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${carryName}[tidx]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${carryName}[tidx]);`,
              );
            }
          }
        }
      }
    } else if (entry.type === "workgroup_assoc_scan") {
      // --- Workgroup associative scan step (Kogge-Stone) ---
      const was = workgroupAssocScans[entry.wasIdx];
      const ws = was.wasStep;
      const numBodyConsts = was.numConsts;

      // Build mapping from body input JitIds to WGSL read info
      const bodyInputInfo: {
        name: string;
        isGlobal: boolean;
        parentInputIdx: number;
      }[] = [];

      for (let bi = 0; bi < was.bodyInputIds.length; bi++) {
        if (bi < numBodyConsts) {
          // Const input — maps to a parent JitId
          const parentJitId = ws.consts[bi];
          const parentName =
            idToReadName.get(parentJitId) ??
            inputIdToName.get(parentJitId) ??
            `__was_unknown_${parentJitId}`;
          const isGlobalInput =
            inputIdToName.has(parentJitId) && !idIsShmem.has(parentJitId);
          const parentBodyIdx = bodyInputIds.indexOf(parentJitId);
          bodyInputInfo.push({
            name: parentName,
            isGlobal: isGlobalInput,
            parentInputIdx:
              parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1,
          });
        } else {
          // a_elem or b_elem — will be resolved dynamically per round
          // placeholder (overridden per-kernel by custom resolveGlobalIndex)
          bodyInputInfo.push({
            name: "__placeholder",
            isGlobal: false,
            parentInputIdx: -1,
          });
        }
      }

      // Build mapping from body JitIds to WGSL read names (for intermediates)
      const bodyIdToName = new Map<JitId, string>();
      for (let bi = 0; bi < numBodyConsts; bi++) {
        bodyIdToName.set(was.bodyInputIds[bi], bodyInputInfo[bi].name);
      }
      for (const [bodyJitId, info] of was.bodyShmemMap) {
        bodyIdToName.set(bodyJitId, info.name);
      }

      // Load input elements into ping shmem arrays
      for (let e = 0; e < was.numElems; e++) {
        const parentElemId = ws.elems[e];
        const [pingName] = was.pingPongNames[e];
        const elemCount = was.elemCounts[e];
        const parentName =
          idToReadName.get(parentElemId) ?? inputIdToName.get(parentElemId);
        const isGlobalInput =
          inputIdToName.has(parentElemId) && !idIsShmem.has(parentElemId);
        const parentBodyIdx = bodyInputIds.indexOf(parentElemId);
        const inIdx =
          parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;

        if (elemCount > 1) {
          emit(
            `for (var _was_ei: u32 = 0u; _was_ei < ${elemCount}u; _was_ei++) {`,
            pushIndent,
          );
          const idx = `tidx * ${elemCount}u + _was_ei`;
          if (isGlobalInput && inIdx >= 0) {
            const remapped = inRemap(inIdx, `i32(${idx})`);
            const readExpr = `${parentName}[i32(in_base_${inIdx}) + ${remapped}]`;
            emit(
              `${pingName}[${idx}] = ${hasBoundary ? `select(${dtypeToWgsl(was.elemDtypes[e])}(0), ${readExpr}, valid)` : readExpr};`,
            );
          } else if (parentName) {
            emit(`${pingName}[${idx}] = ${parentName}[${idx}];`);
          }
          emit(popIndent, "}");
        } else {
          if (isGlobalInput && inIdx >= 0) {
            const remapped2 = inRemap(inIdx, "i32(tidx)");
            const readExpr = `${parentName}[i32(in_base_${inIdx}) + ${remapped2}]`;
            emit(
              `${pingName}[tidx] = ${hasBoundary ? `select(${dtypeToWgsl(was.elemDtypes[e])}(0), ${readExpr}, valid)` : readExpr};`,
            );
          } else if (parentName) {
            emit(`${pingName}[tidx] = ${parentName}[tidx];`);
          }
        }
      }
      emit("workgroupBarrier();");

      // --- Shared helper: emit body kernels for a Kogge-Stone round ---
      // Both subgroup-register and shmem paths use identical input
      // classification, reduction codegen, and epilogue logic. Only the
      // input resolution and output write differ.
      type WasResolveAB = (elemIdx: number, indexExpr: string) => string;
      type WasWriteLeaf = (
        elemIdx: number,
        ec: number,
        value: string,
        isGidx: boolean,
      ) => void;
      type WasMultiElemResolveAB = (elemIdx: number, eiVar: string) => string;

      const emitWasRoundBody = (
        prefix: string,
        resolveA: WasResolveAB,
        resolveB: WasResolveAB,
        writeLeaf: WasWriteLeaf,
        multiElemResolveA: WasMultiElemResolveAB,
        multiElemResolveB: WasMultiElemResolveAB,
      ) => {
        for (let bsi = 0; bsi < was.bodyKernels.length; bsi++) {
          const { step: bStep, kernel: bKernel } = was.bodyKernels[bsi];

          // Classify each input as const / a / b / shmem
          const bStepInputKind: ("const" | "a" | "b" | "shmem")[] = [];
          const bStepInputElemIdx: number[] = [];
          for (let j = 0; j < bStep.inputs.length; j++) {
            const jitId = bStep.inputs[j];
            const biIdx = was.bodyInputIds.indexOf(jitId);
            if (biIdx >= 0 && biIdx < numBodyConsts) {
              bStepInputKind.push("const");
              bStepInputElemIdx.push(-1);
            } else if (
              biIdx >= numBodyConsts &&
              biIdx < numBodyConsts + was.numElems
            ) {
              bStepInputKind.push("a");
              bStepInputElemIdx.push(biIdx - numBodyConsts);
            } else if (biIdx >= numBodyConsts + was.numElems) {
              bStepInputKind.push("b");
              bStepInputElemIdx.push(biIdx - numBodyConsts - was.numElems);
            } else {
              bStepInputKind.push("shmem");
              bStepInputElemIdx.push(-1);
            }
          }

          // Resolve const inputs (shared between both paths)
          const resolveConst = (
            bufIdx: number,
            indexExpr: string,
            dtype: DType,
          ): string => {
            const biIdx2 = was.bodyInputIds.indexOf(bStep.inputs[bufIdx]);
            const inf = bodyInputInfo[biIdx2];
            if (inf.name === BLOCK_IDX_SENTINEL) return `i32(block_idx)`;
            if (inf.isGlobal) {
              if (inf.parentInputIdx >= 0) {
                const remapped = inRemap(inf.parentInputIdx, indexExpr);
                const readExpr = `${inf.name}[i32(in_base_${inf.parentInputIdx}) + ${remapped}]`;
                return hasBoundary
                  ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
                  : readExpr;
              }
              return `${inf.name}[${indexExpr}]`;
            }
            return `${inf.name}[${indexExpr}]`;
          };

          // Build scalar resolve
          const resolve: ResolveGlobalIndex = (bufIdx, indexExpr, dtype) => {
            const kind = bStepInputKind[bufIdx];
            if (kind === "const") return resolveConst(bufIdx, indexExpr, dtype);
            if (kind === "a")
              return resolveA(bStepInputElemIdx[bufIdx], indexExpr);
            if (kind === "b")
              return resolveB(bStepInputElemIdx[bufIdx], indexExpr);
            const sname = bodyIdToName.get(bStep.inputs[bufIdx]);
            return `${sname ?? "__unknown"}[${indexExpr}]`;
          };

          const gen = createGen(bKernel, `${prefix}_s${bsi}`, resolve);

          for (let oi = 0; oi < bKernel.numOutputs; oi++) {
            const outId = bStep.outputs[oi];

            // --- Reduction kernel: gidx loop + ridx accumulation ---
            const bRe = bKernel.outputs[oi]?.reduction ?? null;
            if (bRe) {
              const reDtype = bRe.dtype;
              const reTy = dtypeToWgsl(reDtype, false);
              const bodyDtype = bKernel.outputs[oi].dtype;
              const reSize = bRe.size as number;
              const bodyExp = bKernel.outputs[oi].exp;
              const kernelSize = bKernel.size as number;
              const rePrefix = `${prefix}_s${bsi}`;
              const isIdentityEpilogue =
                bRe.epilogue.op === AluOp.Variable &&
                bRe.epilogue.arg === "acc";

              emit(
                `for (var gidx: i32 = 0; gidx < ${kernelSize}; gidx++) {`,
                pushIndent,
              );
              const accName = `${rePrefix}_acc`;
              emit(
                `var ${accName}: ${reTy} = ${constToWgsl(reDtype, bRe.identity)};`,
              );
              const UNROLL_THRESHOLD = 8;
              const emitAccum = (rhs: string) => {
                const castRhs = wgslCastReductionRhs(rhs, bodyDtype, reDtype);
                emit(wgslReductionAccumStmt(bRe.op, accName, castRhs));
              };
              if (reSize <= UNROLL_THRESHOLD) {
                for (let ri = 0; ri < reSize; ri++) {
                  const riGen = createGen(
                    bKernel,
                    `${rePrefix}_r${ri}`,
                    resolve,
                    undefined,
                    undefined,
                    AluExp.i32(ri),
                  );
                  emitAccum(strip1(riGen(bodyExp)));
                }
              } else {
                emit(
                  `for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
                  pushIndent,
                );
                emitAccum(strip1(gen(bodyExp)));
                emit(popIndent, "}");
              }
              let finalValue: string;
              if (isIdentityEpilogue) {
                finalValue = accName;
              } else {
                const epGen = createGen(
                  bKernel,
                  `${rePrefix}_ep`,
                  resolve,
                  new Map([["acc", accName]]),
                );
                finalValue = strip1(epGen(bRe.epilogue));
              }

              // Write result to leaf output or intermediate
              const bodyOutIdx = was.bodyOutputIds.indexOf(outId);
              if (bodyOutIdx >= 0) {
                writeLeaf(
                  bodyOutIdx,
                  was.elemCounts[bodyOutIdx],
                  finalValue,
                  true,
                );
              } else {
                const sname = bodyIdToName.get(outId);
                if (sname) emit(`${sname}[gidx] = ${finalValue};`);
              }
              emit(popIndent, "}"); // end gidx loop
              continue;
            }

            // --- Elementwise kernel ---
            const bodyOutIdx = was.bodyOutputIds.indexOf(outId);
            if (bodyOutIdx >= 0) {
              const e = bodyOutIdx;
              const ec = was.elemCounts[e];
              if (ec > 1) {
                emit(
                  `for (var _was_oi: u32 = 0u; _was_oi < ${ec}u; _was_oi++) {`,
                  pushIndent,
                );
                // Multi-element resolve: uses _was_oi as element index
                const eiResolve: ResolveGlobalIndex = (
                  bufIdx2,
                  _indexExpr,
                  dtype2,
                ) => {
                  const kind2 = bStepInputKind[bufIdx2];
                  if (kind2 === "const")
                    return resolveConst(bufIdx2, "i32(_was_oi)", dtype2);
                  if (kind2 === "a")
                    return multiElemResolveA(
                      bStepInputElemIdx[bufIdx2],
                      "_was_oi",
                    );
                  if (kind2 === "b")
                    return multiElemResolveB(
                      bStepInputElemIdx[bufIdx2],
                      "_was_oi",
                    );
                  const sname = bodyIdToName.get(bStep.inputs[bufIdx2]);
                  return `${sname ?? "__unknown"}[i32(_was_oi)]`;
                };
                const genEi = createGen(
                  bKernel,
                  `${prefix}_s${bsi}_ei`,
                  eiResolve,
                );
                const rhsEi = strip1(genEi(bKernel.outputs[oi].exp));
                writeLeaf(e, ec, rhsEi, false);
                emit(popIndent, "}");
              } else {
                const rhs = strip1(gen(bKernel.outputs[oi].exp));
                writeLeaf(e, ec, rhs, false);
              }
            } else {
              // Body intermediate → write to var<private> array.
              // The array has `elemCount` entries (1 for scalar ops, M² for
              // matrix ops). Write at the element-local index, NOT tidx.
              const rhs = strip1(gen(bKernel.outputs[oi].exp));
              const sname = bodyIdToName.get(outId);
              if (sname) {
                const elemCount = was.bodyShmemMap.get(outId)?.elemCount ?? 1;
                if (elemCount === 1) {
                  // Scalar intermediate: always index 0
                  emit(`${sname}[0] = ${rhs};`);
                } else if (bKernel.numOutputs > 1) {
                  // Multi-output kernel: output index within the element
                  emit(`${sname}[${oi}] = ${rhs};`);
                } else {
                  // Single-output, multi-element: need gidx loop
                  emit(
                    `for (var _was_ii: u32 = 0u; _was_ii < ${elemCount}u; _was_ii++) {`,
                    pushIndent,
                  );

                  const iiResolve: ResolveGlobalIndex = (
                    bufIdx2,
                    _indexExpr,
                    dtype2,
                  ) => {
                    const kind2 = bStepInputKind[bufIdx2];
                    if (kind2 === "const")
                      return resolveConst(bufIdx2, "i32(_was_ii)", dtype2);
                    if (kind2 === "a")
                      return multiElemResolveA(
                        bStepInputElemIdx[bufIdx2],
                        "_was_ii",
                      );
                    if (kind2 === "b")
                      return multiElemResolveB(
                        bStepInputElemIdx[bufIdx2],
                        "_was_ii",
                      );
                    const sn2 = bodyIdToName.get(bStep.inputs[bufIdx2]);
                    return `${sn2 ?? "__unknown"}[i32(_was_ii)]`;
                  };
                  const genIi = createGen(
                    bKernel,
                    `${prefix}_s${bsi}_ii`,
                    iiResolve,
                  );
                  const rhsIi = strip1(genIi(bKernel.outputs[oi].exp));
                  emit(`${sname}[_was_ii] = ${rhsIi};`);
                  emit(popIndent, "}");
                }
              }
            }
          }
        }
      };

      // --- Emit shmem rounds for a given range ---
      // Factored out so the inclusive-scan tail, subgroup-path tail, and
      // the pure-shmem fallback can all call it with different starting
      // round numbers.
      const emitShmemRounds = (fromRound: number) => {
        for (let r = fromRound; r < was.numRounds; r++) {
          const stride = 1 << r;
          const readBuf = (e: number) =>
            r % 2 === 0 ? was.pingPongNames[e][0] : was.pingPongNames[e][1];
          const writeBuf = (e: number) =>
            r % 2 === 0 ? was.pingPongNames[e][1] : was.pingPongNames[e][0];

          emit(`if (tidx >= ${stride}u) {`, pushIndent);

          emitWasRoundBody(
            `was${entry.wasIdx}_r${r}`,
            // resolveA: shmem read at (tidx - stride)
            (elemIdx, indexExpr) => {
              const ec = was.elemCounts[elemIdx];
              if (ec > 1)
                return `${readBuf(elemIdx)}[(tidx - ${stride}u) * ${ec}u + u32(${indexExpr})]`;
              return `${readBuf(elemIdx)}[tidx - ${stride}u]`;
            },
            // resolveB: shmem read at tidx
            (elemIdx, indexExpr) => {
              const ec = was.elemCounts[elemIdx];
              if (ec > 1)
                return `${readBuf(elemIdx)}[tidx * ${ec}u + u32(${indexExpr})]`;
              return `${readBuf(elemIdx)}[tidx]`;
            },
            // writeLeaf: write to shmem write buffer
            (elemIdx, ec, value, isGidx) => {
              const cast = dtypeToWgsl(was.elemDtypes[elemIdx]);
              if (ec > 1) {
                emit(
                  `${writeBuf(elemIdx)}[tidx * ${ec}u + ${isGidx ? "u32(gidx)" : "_was_oi"}] = ${cast}(${value});`,
                );
              } else {
                emit(`${writeBuf(elemIdx)}[tidx] = ${value};`);
              }
            },
            // multiElemResolveA
            (elemIdx, eiVar) => {
              return `${readBuf(elemIdx)}[(tidx - ${stride}u) * ${was.elemCounts[elemIdx]}u + ${eiVar}]`;
            },
            // multiElemResolveB
            (elemIdx, eiVar) => {
              return `${readBuf(elemIdx)}[tidx * ${was.elemCounts[elemIdx]}u + ${eiVar}]`;
            },
          );

          emit(popIndent, "} else {", pushIndent);

          // Copy: writeBuf[tidx] = readBuf[tidx]
          for (let e = 0; e < was.numElems; e++) {
            const ec = was.elemCounts[e];
            if (ec > 1) {
              emit(
                `for (var _was_ci: u32 = 0u; _was_ci < ${ec}u; _was_ci++) {`,
                pushIndent,
              );
              emit(
                `${writeBuf(e)}[tidx * ${ec}u + _was_ci] = ${readBuf(e)}[tidx * ${ec}u + _was_ci];`,
              );
              emit(popIndent, "}");
            } else {
              emit(`${writeBuf(e)}[tidx] = ${readBuf(e)}[tidx];`);
            }
          }

          emit(popIndent, "}");
          emit("workgroupBarrier();");
        }
      };

      // --- Subgroup shuffle rounds via subgroupShuffleUp ---
      // Up to 3 rounds (stride ≤ 4) unrolled at codegen time into register
      // ops, avoiding shmem writes + workgroupBarrier() for those rounds.
      const emitSubgroupShuffleSection = () => {
        for (let e = 0; e < was.numElems; e++) {
          const ty = dtypeToWgsl(was.elemDtypes[e], false);
          const ec = was.elemCounts[e];
          const regName = `was_sg_reg_${entry.wasIdx}_${e}`;
          const pingName = was.pingPongNames[e][0];
          if (ec > 1) {
            emit(`var ${regName}: array<${ty}, ${ec}>;`);
            for (let ei = 0; ei < ec; ei++) {
              emit(`${regName}[${ei}u] = ${pingName}[tidx * ${ec}u + ${ei}u];`);
            }
          } else {
            emit(`var ${regName}: ${ty} = ${pingName}[tidx];`);
          }
        }

        for (let r = 0; r < sgShuffleRounds; r++) {
          const stride = 1 << r;

          // Shuffle each leaf's elements via subgroupShuffleUp
          for (let e = 0; e < was.numElems; e++) {
            const ty = dtypeToWgsl(was.elemDtypes[e], false);
            const ec = was.elemCounts[e];
            const regName = `was_sg_reg_${entry.wasIdx}_${e}`;
            const shName = `was_sg_a_${entry.wasIdx}_r${r}_${e}`;
            if (ec > 1) {
              emit(`var ${shName}: array<${ty}, ${ec}>;`);
              for (let ei = 0; ei < ec; ei++) {
                emit(
                  `${shName}[${ei}u] = subgroupShuffleUp(${regName}[${ei}u], ${stride}u);`,
                );
              }
            } else {
              emit(
                `let ${shName}: ${ty} = subgroupShuffleUp(${regName}, ${stride}u);`,
              );
            }
          }

          // Guard: only invocations with sg_inv_id >= stride apply the compose.
          // Invocations below stride keep their register value unchanged.
          emit(`if (sg_inv_id >= ${stride}u) {`, pushIndent);

          emitWasRoundBody(
            `was${entry.wasIdx}_sg${r}`,
            // resolveA: read from shuffled registers
            (elemIdx, indexExpr) => {
              const shName = `was_sg_a_${entry.wasIdx}_r${r}_${elemIdx}`;
              const ec = was.elemCounts[elemIdx];
              return ec > 1 ? `${shName}[u32(${indexExpr})]` : shName;
            },
            // resolveB: read from current registers
            (elemIdx, indexExpr) => {
              const regName = `was_sg_reg_${entry.wasIdx}_${elemIdx}`;
              const ec = was.elemCounts[elemIdx];
              return ec > 1 ? `${regName}[u32(${indexExpr})]` : regName;
            },
            // writeLeaf: write to register
            (elemIdx, ec, value, isGidx) => {
              const regName = `was_sg_reg_${entry.wasIdx}_${elemIdx}`;
              const cast = dtypeToWgsl(was.elemDtypes[elemIdx]);
              if (ec > 1) {
                emit(
                  `${regName}[${isGidx ? "u32(gidx)" : "_was_oi"}] = ${cast}(${value});`,
                );
              } else {
                emit(`${regName} = ${cast}(${value});`);
              }
            },
            (elemIdx, eiVar) => {
              const ec = was.elemCounts[elemIdx];
              const n = `was_sg_a_${entry.wasIdx}_r${r}_${elemIdx}`;
              return ec > 1 ? `${n}[${eiVar}]` : n;
            },
            (elemIdx, eiVar) => {
              const ec = was.elemCounts[elemIdx];
              const n = `was_sg_reg_${entry.wasIdx}_${elemIdx}`;
              return ec > 1 ? `${n}[${eiVar}]` : n;
            },
          );

          emit(popIndent, "}"); // end if (sg_inv_id >= stride)
        }

        // Flush registers to shmem. Target is deterministic by parity.
        for (let e = 0; e < was.numElems; e++) {
          const ec = was.elemCounts[e];
          const regName = `was_sg_reg_${entry.wasIdx}_${e}`;
          const flushTarget =
            sgShuffleRounds % 2 === 0
              ? was.pingPongNames[e][0]
              : was.pingPongNames[e][1];
          if (ec > 1) {
            for (let ei = 0; ei < ec; ei++) {
              emit(
                `${flushTarget}[tidx * ${ec}u + ${ei}u] = ${regName}[${ei}u];`,
              );
            }
          } else {
            emit(`${flushTarget}[tidx] = ${regName};`);
          }
        }
        emit("workgroupBarrier();");
      };

      // --- Subgroup inclusive scan via subgroupInclusiveAdd/Mul ---
      // For scalar Add or Mul bodies, a single hardware instruction
      // replaces all intra-subgroup Kogge-Stone rounds.
      const emitInclusiveScanSection = () => {
        const ty = dtypeToWgsl(was.elemDtypes[0], false);
        const ping = was.pingPongNames[0][0];
        const builtinName =
          was.scalarOp === AluOp.Add
            ? "subgroupInclusiveAdd"
            : "subgroupInclusiveMul";
        const regName = `was_sg_inc_${entry.wasIdx}`;
        emit(`var ${regName}: ${ty} = ${ping}[tidx];`);
        emit(`${regName} = ${builtinName}(${regName});`);
        // After sgInclusiveRounds virtual rounds, the next shmem round reads
        // from: even→ping, odd→pong. Write the result there.
        const target =
          sgInclusiveRounds % 2 === 0
            ? was.pingPongNames[0][0]
            : was.pingPongNames[0][1];
        emit(`${target}[tidx] = ${regName};`);
        emit("workgroupBarrier();");
      };

      // --- Determine subgroup optimization levels ---
      // sgInclusiveRounds: for scalar add/mul, hardware inclusive scan
      //   replaces up to 5 rounds (sg_size ≥ 32).
      // sgShuffleRounds: for general bodies, register shuffles replace
      //   up to 3 rounds (sg_size ≥ 8).
      const sgShuffleRounds = useSubgroupShuffle
        ? Math.min(3, was.numRounds)
        : 0;
      const sgInclusiveRounds =
        was.scalarOp != null && useSubgroupShuffle
          ? Math.min(5, was.numRounds)
          : 0;

      // Emit the appropriate branch structure. All branches keep ping/pong
      // parity deterministic by pairing shmem rounds with a fixed starting
      // round number.
      // subgroupShuffleUp and subgroupInclusiveAdd/Mul operate WITHIN a
      // subgroup only. When blockSize > sg_size, threads at subgroup
      // boundaries (e.g., tidx=32 with sg_size=32) never receive data from
      // the previous subgroup. The shmem rounds that follow build on these
      // incomplete partial prefixes, producing wrong results for all
      // subsequent threads. Safe only when sg_size >= blockSize (all threads
      // in one subgroup). See dlm-js issue: reverse 3-field DLM compose.
      const sgGuard = `${blockSize}u`;
      if (sgInclusiveRounds > 0) {
        if (DEBUG >= 1)
          console.info(
            `block_map fused: workgroup_assoc_scan using subgroupInclusive${was.scalarOp === AluOp.Add ? "Add" : "Mul"} (${sgInclusiveRounds} rounds replaced, ${was.numRounds - sgInclusiveRounds} shmem rounds remaining)`,
          );
        // 3-way branch: inclusive scan → shuffle fallback → pure shmem
        emit(`if (sg_size >= ${sgGuard}) {`, pushIndent);
        emitInclusiveScanSection();
        emitShmemRounds(sgInclusiveRounds);
        if (sgShuffleRounds > 0) {
          emit(popIndent, `} else if (sg_size >= ${sgGuard}) {`, pushIndent);
          emitSubgroupShuffleSection();
          emitShmemRounds(sgShuffleRounds);
        }
        emit(popIndent, "} else {", pushIndent);
        emitShmemRounds(0);
        emit(popIndent, "}");
      } else if (sgShuffleRounds > 0) {
        // 2-way branch: shuffle → pure shmem
        emit(`if (sg_size >= ${sgGuard}) {`, pushIndent);
        emitSubgroupShuffleSection();
        emitShmemRounds(sgShuffleRounds);
        emit(popIndent, "} else {", pushIndent);
        emitShmemRounds(0);
        emit(popIndent, "}");
      } else {
        emitShmemRounds(0);
      }

      // Write output from final buffer to parent output shmem / result buffers
      // After numRounds rounds, final data is in:
      //   numRounds even → ping; numRounds odd → pong
      const finalNames = was.pingPongNames.map(([ping, pong]) =>
        was.numRounds % 2 === 0 ? ping : pong,
      );

      for (let e = 0; e < was.numElems; e++) {
        const parentOutId = ws.outputs[e];
        const resultIdx = bodyOutputIds.indexOf(parentOutId);
        const ec = was.elemCounts[e];

        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          // Output goes to global result buffer
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          if (ec > 1) {
            emit(
              `for (var _was_wi: u32 = 0u; _was_wi < ${ec}u; _was_wi++) {`,
              pushIndent,
            );
            const shmemIdx = `tidx * ${ec}u + _was_wi`;
            // The output offset is the block-axis strided offset for tidx
            // plus the flat sub-element index _was_wi. outOffset("tidx")
            // handles the block→global stride remapping; _was_wi indexes
            // into the non-block element dimensions whose strides match.
            const outOff = `i32(${outOffset(resultIdx)}) + i32(_was_wi)`;
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + ${outOff}] = ${resultTy}(${finalNames[e]}[${shmemIdx}]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + ${outOff}] = ${resultTy}(${finalNames[e]}[${shmemIdx}]);`,
              );
            }
            emit(popIndent, "}");
          } else {
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${finalNames[e]}[tidx]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${resultTy}(${finalNames[e]}[tidx]);`,
              );
            }
          }
        } else {
          // Output goes to a parent shmem buffer (another step reads it)
          const shmemName = idToReadName.get(parentOutId);
          if (shmemName) {
            if (ec > 1) {
              emit(
                `for (var _was_wi: u32 = 0u; _was_wi < ${ec}u; _was_wi++) {`,
                pushIndent,
              );
              emit(
                `${shmemName}[tidx * ${ec}u + _was_wi] = ${finalNames[e]}[tidx * ${ec}u + _was_wi];`,
              );
              emit(popIndent, "}");
            } else {
              emit(`${shmemName}[tidx] = ${finalNames[e]}[tidx];`);
            }
          }
        }
      }
    }
  }

  // --- Pass-through outputs: copy input → output ---
  if (passThroughOutputs.size > 0 && hasBoundary) {
    emit("if (valid) {", pushIndent);
  }
  for (const [resultIdx, info] of passThroughOutputs) {
    if (info.inputIdx >= 0) {
      // Non-const input: apply block offset with stride remap
      const remapped = inRemap(info.inputIdx, "i32(tidx)");
      emit(
        `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${info.inputName}[i32(in_base_${info.inputIdx}) + ${remapped}];`,
      );
    } else {
      // Const input: no block offset
      emit(
        `${resultVar(resultIdx)}[i32(out_base_${resultIdx}) + i32(${outOffset(resultIdx)})] = ${info.inputName}[i32(tidx)];`,
      );
    }
  }
  if (passThroughOutputs.size > 0 && hasBoundary) {
    emit(popIndent, "}");
  }

  emit(popIndent, "}");

  const code = withNanInfHeader(shader.join("\n"));

  return {
    code,
    numInputs: storageInputs,
    numOutputs: numOutputBindings,
    hasUniform: effectiveUniformConsts > 0,
    passes: [{ grid: [gridX, gridY] }],
    sharedMemoryBytes: totalShmemBytes > 0 ? totalShmemBytes : undefined,
    numUniformConsts:
      effectiveUniformConsts > 0 ? effectiveUniformConsts : undefined,
    packedLeafLayout: packedLeafLayout ?? undefined,
  };
}
