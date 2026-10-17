/**
 * @file Optimizations applied to kernels by different backends.
 *
 * The main optimizations (for reductions) are:
 *
 * - "Upcast": Multiple values are computed per thread, along a non-reduction
 *   dimension. If appropriate, this lowers to vector/SIMD instructions. Each
 *   thread computes a chunk of output values, which helps with cache
 *   performance (e.g., matmul tiling).
 *
 * - "Unroll": Similar to Upcast, but along a loop dimension, which translates
 *   to loop unrolling. You increment the loop index by the unroll factor. This
 *   does not use vector/SIMD instructions.
 *
 * - "Group": Multiple threads compute the same value. For example, when summing
 *   up the numbers in a vector, K threads each accumulate 1/K of the vector,
 *   stores in shared memory, and thread 0 accumulates at the end.
 *   - Regular order: 4 threads grouped as [1234123412341234]
 *   - "Top": 4 threads grouped as [1111222233334444]
 *
 * These are inspired by Tinygrad's heuristic optimizations.
 * https://github.com/tinygrad/tinygrad/blob/685d5c46df/tinygrad/codegen/heuristic.py
 */

import { accessorGlobal, AluExp, AluOp, AluVar, DType, Kernel } from "./alu";
import type { BackendCapabilities } from "./backend";
import { ShapeTracker, type SizeExpr, unravelAlu } from "./shape";
import { DEBUG, deepEqual, lexCompare, prod, range, sorted } from "./utils";

export const CONSERVATIVE_WEBGPU_PERF_DEFAULTS = {
  dispatchOverheadUs: 25,
  bandwidthGBs: 50,
  tflops: 1.5,
  rOptWords: 128,
  barrierCostFactor: 1.0,
} as const;

export function resolvePerformanceBelief(caps: Partial<BackendCapabilities>): {
  dispatchOverheadUs: number;
  bandwidthGBs: number;
  tflops: number;
  rOptWords: number;
  barrierCostFactor: number;
  source: "measured" | "conservative-defaults";
} {
  return {
    dispatchOverheadUs:
      caps.dispatchOverheadUs ??
      CONSERVATIVE_WEBGPU_PERF_DEFAULTS.dispatchOverheadUs,
    bandwidthGBs:
      caps.bandwidthGBs ?? CONSERVATIVE_WEBGPU_PERF_DEFAULTS.bandwidthGBs,
    tflops: caps.tflops ?? CONSERVATIVE_WEBGPU_PERF_DEFAULTS.tflops,
    rOptWords: caps.rOptWords ?? CONSERVATIVE_WEBGPU_PERF_DEFAULTS.rOptWords,
    barrierCostFactor:
      caps.barrierCostFactor ??
      CONSERVATIVE_WEBGPU_PERF_DEFAULTS.barrierCostFactor,
    source: caps.calibrated ? "measured" : "conservative-defaults",
  };
}

export interface TuneResult {
  /** New expression with GlobalView ops and gidx/ridx lowered. */
  exp: AluExp;

  /** New reduction epilogue expression, present when the kernel output has a reduction. */
  epilogue?: AluExp;

  /** Expression for indexing the result array, including upcast. */
  outputIdxExp: AluExp;

  /** How many total threads to dispatch in the grid. */
  threadCount: SizeExpr;

  /** Sizes of various dimensions of the kernel. */
  size: {
    /** Number of iterations for the reduce loop, `AluExp.special("ridx")`. */
    reduce: SizeExpr;

    /** Amount to upcast in reduce loop, set via `AluVar.unroll`. */
    unroll?: number;

    /** Amount to upcast in non-reduce dimensions, set via `AluVar.upcast`. */
    upcast?: number;
  };
}

/** WebGPU-specific tune result — adds cooperative threading fields. */
export interface WebGPUTuneResult extends TuneResult {
  size: TuneResult["size"] & {
    /**
     * Number of threads for each group.
     * If greater than 1, group index is available as `AluExp.special("group")`.
     */
    groups?: number;

    /** Workgroup-local dimension size (future: shared memory tiling). */
    local?: number;
  };
}

/** WASM-specific tune result — adds vectorization fields. */
export interface WasmTuneResult extends TuneResult {
  size: TuneResult["size"] & {
    /** SIMD lane width (e.g. 4 for f32x4, 2 for f64x2). */
    simdWidth?: number;
  };
}

/** Stores dimensions of the kernel's applied shape. Globals start at 0. */
class TuneDims {
  st: ShapeTracker; // Shape tracker including reduction axes.
  outputSt: ShapeTracker; // Shape tracker including only output axes.

  local: number; // Local axes start here (maps to local_invocation_id).
  groups: number; // Reductions start here, with groups.
  reduce: number; // Single reduction thread.
  unroll: number; // Upcast along the reduce dimension.
  upcast: number; // Upcast along output dimension.

  get end() {
    return this.st.shape.length;
  }

  constructor(shape: number[]) {
    this.st = ShapeTracker.fromShape(shape);
    this.outputSt = ShapeTracker.fromShape(shape.slice(0, -1));
    this.local = this.st.shape.length - 1;
    this.groups = this.st.shape.length - 1;
    this.reduce = this.st.shape.length - 1;
    this.unroll = this.st.shape.length;
    this.upcast = this.st.shape.length;
  }

  // Place the axis at the end of the shape, so it is part of each workgroup.
  applyLocal(axis: number, amount: number) {
    if (axis >= this.groups) throw new Error("Cannot localize non-global axis");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Localize by ${amount} on axis length ${length}`);

    if (length !== amount) {
      // First split it.
      (this.local++,
        this.groups++,
        this.reduce++,
        this.unroll++,
        this.upcast++);
      this.st = this.st.reshape([
        ...this.st.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.st.shape.slice(axis + 1),
      ]);
      this.outputSt = this.outputSt.reshape([
        ...this.outputSt.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.outputSt.shape.slice(axis + 1),
      ]);
      axis++;
    }

    // Now permute axis to the end of the global axes, before local/groups.
    this.st = this.st.permute([
      ...range(axis),
      ...range(axis + 1, this.groups),
      axis,
      ...range(this.groups, this.st.shape.length),
    ]);
    this.outputSt = this.outputSt.permute([
      ...range(axis),
      ...range(axis + 1, this.groups),
      axis,
      ...range(this.groups, this.outputSt.shape.length),
    ]);
    this.local--;
  }

  applyUpcast(axis: number, amount: number) {
    if (axis >= this.groups)
      throw new Error("Cannot upcast along reduction axis");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Upcast by ${amount} on axis length ${length}`);
    this.st = this.st
      .reshape([
        ...this.st.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.st.shape.slice(axis + 1),
      ])
      .permute([
        ...range(axis + 1),
        ...range(axis + 2, this.st.shape.length + 1),
        axis + 1,
      ]);
    this.outputSt = this.outputSt
      .reshape([
        ...this.outputSt.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.outputSt.shape.slice(axis + 1),
      ])
      .permute([
        ...range(axis + 1),
        ...range(axis + 2, this.outputSt.shape.length + 1),
        axis + 1,
      ]);
  }

  applyUnroll(axis: number, amount: number) {
    if (axis < this.groups) throw new Error("Cannot unroll non-reduce axis");
    if (axis >= this.unroll) throw new Error("Axis already unrolled");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Unroll by ${amount} on axis length ${length}`);
    // We're unrolling away the whole axis.
    if (length === amount) {
      this.st = this.st.permute([
        ...range(axis),
        ...range(axis + 1, this.upcast),
        axis,
        ...range(this.upcast, this.st.shape.length),
      ]);
      if (axis < this.reduce) this.reduce--;
      this.unroll--;
    } else {
      this.st = this.st
        .reshape([
          ...this.st.shape.slice(0, axis),
          length / amount,
          amount,
          ...this.st.shape.slice(axis + 1),
        ])
        .permute([
          ...range(axis + 1),
          ...range(axis + 2, this.upcast + 1), // Move to just before upcast
          axis + 1,
          ...range(this.upcast + 1, this.st.shape.length + 1),
        ]);
      this.upcast++;
    }
  }

  /**
   * Split the first reduce axis into cooperative group threads.
   * Creates [amount, length/amount] where `amount` threads run in parallel
   * and each processes `length/amount` elements sequentially.
   * The results are combined via workgroup shared memory.
   */
  applyGroups(amount: number) {
    if (this.groups !== this.reduce) throw new Error("Groups already applied");
    if (this.reduce >= this.unroll) throw new Error("No reduce axis to group");
    const axis = this.reduce;
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Group by ${amount} on reduce axis length ${length}`);
    this.st = this.st.reshape([
      ...this.st.shape.slice(0, axis),
      amount,
      length / amount,
      ...this.st.shape.slice(axis + 1),
    ]);
    this.reduce++;
    this.unroll++;
    this.upcast++;
  }
}

// ── Continuous Cost Modeling ────────────────────────────────────────────

export interface CostFeatures {
  nDispatch: number;
  nBuffers: number;
  countAlu: number;
  countMem: number;
  depthPriv: number; // Max local var depth (estimated words)
  sizeShmem: number; // bytes
  sizeWgsl: number; // bytes approx

  produceCount?: number; // Elements effectively produced by this evaluated block
  parallelism?: number; // Effective threads working together
}

export function evaluateTotalCost(
  features: CostFeatures,
  caps: Partial<BackendCapabilities>,
): number {
  const belief = resolvePerformanceBelief(caps);

  // 1. Predictors from Capabilities
  const cDispatch = belief.dispatchOverheadUs / 1000;
  const bMem = belief.bandwidthGBs; // GB/s roughly equals bytes/ns = million bytes/ms
  const tAlu = belief.tflops; // TFLOPS

  // Wavefront/Subgroup alignment
  const isIntel = caps.adapterVendor?.toLowerCase().includes("intel") ?? false;
  const fSubgroup = isIntel ? 16 : 32;

  // Per-thread register budget from hardware profile (R_opt_words).
  // Critical: igp=48, mobile=64, discrete-legacy=96, discrete-modern/apple=128.
  const rOpt = belief.rOptWords;
  const sOpt = 16384; // 16KB optimal occupancy
  // We heavily discount cCompile in runtime configuration comparisons to prevent it from squashing throughput traits.
  const cCompile = (8 + (features.sizeWgsl / 1024) * 3) * 0.0001;

  // 2. Base Execution Model
  // countMem bytes / (bMem GB/s) -> ms
  const memCost = features.countMem / (bMem * 1e6);

  // countAlu / (tAlu * 1e9 * fSubgroup) -> ms
  const aluCost = features.countAlu / (tAlu * 1e9 * fSubgroup);

  // If parallelism is tracked, compute algorithmic latency bounds
  // (e.g. latency bound of reduction vs throughput bound of standard elementwise)
  const latencyCost = features.parallelism
    ? (features.countAlu / features.parallelism) * 1e-6
    : 0;
  // If cooperative groups imply barriers, add the cost of synchronizing threads.
  // Hardware barriers are O(1) events (all threads in a workgroup sync
  // simultaneously), not O(n). Use constant cost per barrier — scaling
  // linearly with thread count incorrectly penalizes large cooperative groups
  // that actually improve reduction throughput.
  const bFactor = belief.barrierCostFactor;
  const barrierLatency =
    features.parallelism && features.parallelism > 1 ? bFactor * 0.0001 : 0;

  const costExecution =
    features.nDispatch * cDispatch +
    memCost +
    aluCost +
    latencyCost +
    barrierLatency;

  // 3. Danger Multipliers (Exponential Cliffs)
  // Penalty(x) = exp(max(0, x - 1) * steepness)
  const penalty = (ratio: number, steepness = 4) =>
    Math.exp(Math.max(0, ratio - 1) * steepness);

  const dangerPriv = penalty(features.depthPriv / rOpt, 5); // per-thread register spilling
  const dangerShmem = penalty(features.sizeShmem / sOpt, 2); // gradual occupancy drop
  const dangerBind = penalty(features.nBuffers / 8, 3); // Binding pressure

  // Aggregate workgroup register pressure: total registers across all threads
  // must fit in the hardware GRF. On gen-9 igps (rOpt=48), 256 threads × 58
  // regs = 14,848 words massively overflows the register file → TDR.
  // Budget = rOpt × maxConcurrentThreads. Small-GRF GPUs (igp/mobile) can
  // sustain fewer concurrent threads before spilling catastrophically.
  const threads = features.parallelism ?? 1;
  const aggregateRegs = threads * features.depthPriv;
  const maxConcurrentThreads = rOpt <= 48 ? 64 : rOpt <= 64 ? 128 : 256;
  const aggregateBudget = rOpt * maxConcurrentThreads;
  const dangerAggregate = penalty(aggregateRegs / aggregateBudget, 3);

  const dangerMultiplier =
    dangerPriv * dangerShmem * dangerBind * dangerAggregate;

  let totalCost = (costExecution + cCompile) * dangerMultiplier;

  // Normalization scaling (Reward larger structural spans)
  if (features.produceCount) {
    totalCost /= features.produceCount;
  }
  return totalCost;
}

/** Tuning step that does not apply any optimization. */
export function tuneNullopt(kernel: Kernel): TuneResult {
  const o = kernel.outputs[0];
  let exp = o.exp;
  const vars: Record<string, AluExp> = {};
  // For symbolic kernels with a concrete hint, use the hint for gidx range.
  // This gives gidx a bounded range [0, concreteSize-1] so the simplifier
  // can eliminate modulo ops, producing size-independent expressions.
  const gidxSize = kernel.concreteSizeHint ?? kernel.size;
  vars.gidx = AluExp.special(DType.Int32, "gidx", gidxSize);
  if (o.reduction) {
    // Use concrete hint for ridx bound (for expression simplification),
    // falling back to the actual size (which may be symbolic).
    const ridxBound = o.reduction.concreteHint ?? o.reduction.size;
    vars.ridx = AluExp.special(DType.Int32, "ridx", ridxBound);
    if (exp.dtype !== o.reduction.dtype)
      exp = AluExp.cast(o.reduction.dtype, exp);
  }
  let resultExp = exp.substitute(vars).rewriteGlobalViews().simplify();
  let resultEpilogue = o.reduction?.epilogue
    .substitute({ gidx: vars.gidx })
    .rewriteGlobalViews()
    .simplify();

  // For symbolic kernels, replace GlobalIndex len with INT32_MAX.
  // When compiled with the first call's concrete dims, the simplifier
  // eliminates modulus ops (e.g., gidx % 6400 → gidx when range is bounded).
  // But the GlobalIndex len retains the first call's concrete total, which
  // would cause an incorrect WASM bounds clamp for larger sizes. Setting
  // len to INT32_MAX makes the per-element bounds check a no-op; the actual
  // bound comes from the shader/loop guard (gidx < dynamicParams[0]).
  if (kernel.isSymbolic) {
    resultExp = unlimitGlobalIndexLen(resultExp);
    if (resultEpilogue) {
      resultEpilogue = unlimitGlobalIndexLen(resultEpilogue);
    }
  }

  return {
    exp: resultExp,
    epilogue: resultEpilogue,
    outputIdxExp: vars.gidx,
    threadCount: kernel.size,
    size: {
      reduce: o.reduction ? o.reduction.size : 0,
    },
  };
}

/** Replace all GlobalIndex len values with INT32_MAX (0x7FFFFFFF). */
function unlimitGlobalIndexLen(exp: AluExp): AluExp {
  return exp.rewrite((node) => {
    if (node.op === AluOp.GlobalIndex) {
      const [gid, len] = node.arg as [number, number];
      if (len !== 0x7fffffff) {
        return AluExp.globalIndex(node.dtype, gid, 0x7fffffff, node.src[0]);
      }
    }
  });
}

/** Tuning for WebGPU kernels. */
export function tuneWebgpu(
  kernel: Kernel,
  caps?: BackendCapabilities,
): WebGPUTuneResult {
  const reduction = kernel.outputs[0].reduction;
  if (!reduction) return tuneNullopt(kernel);
  // Symbolic kernels can't use upcast/unroll optimizations (unknown size).
  if (kernel.isSymbolic) return tuneNullopt(kernel);
  // Symbolic reduction axis can't use upcast/unroll (unknown reduction size).
  if (kernel.hasSymbolicReduction) return tuneNullopt(kernel);

  const exp = AluExp.cast(reduction.dtype, kernel.outputs[0].exp);
  const globalIndexes = exp.collect((exp) => exp.op === AluOp.GlobalIndex);
  if (globalIndexes.length > 0) {
    if (DEBUG >= 4)
      console.info("Tuning: Found GlobalIndex ops, skipping opt.");
    return tuneNullopt(kernel);
  }

  // 1. Check that kernel GlobalView ops have consistent src[], where the last
  //    dimension is reduction, and others are gidx.
  const globalViews = exp.collect((exp) => exp.op === AluOp.GlobalView);
  if (globalViews.length === 0) {
    if (DEBUG >= 4) console.info("Tuning: No GlobalView ops found in kernel.");
    // Nullary kernels (0 inputs) come from #realize() on virtual arrays
    // (full, arange, eye). Post-O2, pushLit uses initialData instead.
    return tuneNullopt(kernel);
  }
  const shape: number[] = globalViews[0].arg[1].shape;
  const expectedSrc = [
    ...unravelAlu(shape.slice(0, -1), AluVar.gidx),
    AluVar.ridx,
  ].map((e) => e.simplify());
  for (const gv of globalViews) {
    if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
      if (DEBUG >= 4)
        console.info("Tuning: GlobalView src[] not consistent with reduction.");
      return tuneNullopt(kernel);
    }
  }
  if (shape[shape.length - 1] !== reduction.size)
    throw new Error("Invariant violation: shape doesn't match reduction size.");

  // 2. Collect all shape trackers for kernel GlobalView ops.
  const sts: ShapeTracker[] = globalViews.map((gv) => gv.arg[1]);
  for (const st of sts) {
    if (!deepEqual(st.shape, shape))
      throw new Error("Invariant violation: GlobalView shape mismatch"); // sanity check
  }

  // 3. Apply heuristic optimizations based on the shape trackers.
  const dim = new TuneDims(shape);

  // Try to do upcasts of non-reduce axes for global memory coalescing.
  // Heuristic is based on strides, and borrowed from tinygrad.
  const upcastedAxis = new Set<number>();
  while (prod(dim.st.shape.slice(0, dim.groups)) >= 1024) {
    const choices: number[][] = [];
    const composedSts = sts.map((st) => st.compose(dim.st));
    for (let axis = 0; axis < dim.groups; axis++) {
      for (const amount of [3, 4, 5]) {
        // Axis is not upcasted, divisible, and has a buffer with stride 0 on
        // that axis (mem coalescing) while not already a stride-0 upcast.
        if (
          !upcastedAxis.has(axis) &&
          dim.st.shape[axis] % amount === 0 &&
          composedSts.some(
            (st) =>
              st.lastStrides[axis] === 0 &&
              st.lastStrides.slice(dim.unroll).every((stride) => stride > 0),
          )
        ) {
          let nonzeroStrides = 0;
          let totalStrides = 0;
          for (const st of composedSts) {
            nonzeroStrides += st.lastStrides[axis] > 0 ? 1 : 0;
            totalStrides += st.lastStrides[axis];
          }
          choices.push([nonzeroStrides, totalStrides, axis, amount]);
        }
      }
    }
    if (choices.length > 0) {
      choices.sort(lexCompare);
      dim.applyUpcast(choices[0][2], choices[0][3]);
      upcastedAxis.add(choices[0][2]);
    } else {
      break;
    }
  }

  // Apply cooperative groups for large parallel reductions.
  // Evaluate the best group size via total cost equation.
  if (dim.reduce < dim.unroll) {
    const seqReduce = prod(dim.st.shape.slice(dim.reduce, dim.unroll));
    if (seqReduce >= 2048) {
      const bSize = Math.min(caps?.maxComputeWorkgroupSizeX ?? 256, 256);
      let bestCost = Infinity;
      let bestGroup = 1;

      for (let g = bSize; g >= 1; g >>= 1) {
        if (g > 1 && (seqReduce % g !== 0 || seqReduce / g < 2)) continue;

        // Estimate execution cost
        const features: CostFeatures = {
          nDispatch: 1,
          nBuffers: 2,
          countAlu: seqReduce,
          countMem: seqReduce * 4,
          depthPriv: 8,
          sizeShmem: g * 4, // shared memory bytes
          sizeWgsl: 4096,
          parallelism: g,
          produceCount: 1,
        };
        const cost = evaluateTotalCost(features, caps ?? {});

        if (cost < bestCost) {
          bestCost = cost;
          bestGroup = g;
        }
      }

      if (bestGroup > 1) {
        if (DEBUG >= 4)
          console.info(
            `tuneWebgpu: cooperative groups seqReduce=${seqReduce} → g=${bestGroup}`,
          );
        dim.applyGroups(bestGroup);
      }
    }
  }

  // Try to do loop unrolling on the reduce axis, with an upcast limit.
  // Skip doing this on mobile devices, as it may reduce performance.
  // WebGPU returns maxComputeWorkgroupSizeX=256 for many desktops (Intel Gen-9, Apple Silicon)
  // so we should not use > 256 as a strict mobile check.
  const isMobile =
    typeof navigator !== "undefined" &&
    /Mobi|Android/i.test(navigator.userAgent);
  if (
    !isMobile &&
    dim.reduce < dim.unroll &&
    (prod(dim.st.shape.slice(dim.unroll)) <= 4 ||
      (dim.unroll === dim.upcast && prod(dim.st.shape.slice(dim.upcast)) < 64))
  ) {
    // Fully unroll the reduce axis.
    const s = dim.st.shape[dim.unroll - 1];
    if (0 < s && s <= 32) {
      dim.applyUnroll(dim.reduce, s);
    } else {
      // Partially unroll the reduce axis.
      //
      // Note: Unrolling by 8 previously made this faster in January 2026, but
      // in later versions of Chrome on macOS, it seems to have regressed 40%.
      // Seems like 4 is a more stable choice at the moment.
      for (const splits of [4, 2]) {
        if (s % splits === 0) {
          dim.applyUnroll(dim.unroll - 1, splits);
          break;
        }
      }
    }
  }

  // Skip local tiling when cooperative groups are active — the workgroup
  // is already dedicated to the shared-memory reduction.
  if (dim.groups === dim.reduce) {
    for (const ax of sorted(upcastedAxis)) {
      // After applyLocal permutations, upcastedAxis indices may point at
      // axes that are already in the local range. Skip them to avoid
      // decrementing dim.local past global axes into the local range.
      if (ax >= dim.local && ax < dim.groups) continue;

      const s = dim.st.shape[ax];
      const currentLocal = prod(dim.st.shape.slice(dim.local, dim.groups));

      let bestCost = Infinity;
      let bestAmount = 1;

      // Evaluate amount=8,4,2,1 — pick the cheapest per-element cost.
      for (const amount of [8, 4, 2, 1]) {
        if (s % amount !== 0) continue;

        const localThreads = currentLocal * amount;
        if (caps && localThreads > (caps.maxComputeWorkgroupSizeX ?? 256))
          continue;

        const features: CostFeatures = {
          nDispatch: 1,
          nBuffers: 2,
          countAlu: localThreads * 2,
          countMem: localThreads * 8,
          depthPriv: 10 + amount,
          sizeShmem: 0,
          sizeWgsl: 4096 + amount * 32,
          parallelism: localThreads,
          produceCount: localThreads,
        };
        const cost = evaluateTotalCost(features, caps ?? {});

        if (cost < bestCost) {
          bestCost = cost;
          bestAmount = amount;
        }
      }

      // Apply to this axis and CONTINUE to the next axis — multi-axis local
      // tiling is essential for matmul-like kernels where both output
      // dimensions benefit from workgroup-level parallelism.
      if (bestAmount > 1) {
        if (DEBUG >= 4)
          console.info(
            `tuneWebgpu: local tiling axis=${ax} → amount=${bestAmount}`,
          );
        dim.applyLocal(ax, bestAmount);
      }
    }
  }

  // 4. Return the tuned kernel result.
  const indices: AluExp[] = [];
  const addIndices = (s: number[], exp: AluExp) => {
    if (s.length === 0) return;
    else if (s.length === 1) indices.push(exp);
    else indices.push(...unravelAlu(s, exp));
  };
  if (0 < dim.groups) {
    const s = dim.st.shape.slice(0, dim.groups);
    addIndices(s, AluExp.special(DType.Int32, "gidx", prod(s)));
  }
  if (dim.groups < dim.reduce) {
    const s = dim.st.shape.slice(dim.groups, dim.reduce);
    addIndices(s, AluExp.special(DType.Int32, "group", prod(s)));
  }
  if (dim.reduce <= dim.unroll) {
    const s = dim.st.shape.slice(dim.reduce, dim.unroll);
    addIndices(s, AluExp.special(DType.Int32, "ridx", prod(s)));
  }
  if (dim.unroll < dim.upcast) {
    const s = dim.st.shape.slice(dim.unroll, dim.upcast);
    addIndices(s, AluVar.unroll);
  }
  if (dim.upcast < dim.end) {
    const s = dim.st.shape.slice(dim.upcast);
    addIndices(s, AluVar.upcast);
  }

  // Substitute old values of AluVar.gidx and AluVar.ridx.
  //
  // As an optimization to `.substitute(vars).rewriteGlobalViews()`, we compose
  // dim.st with the ShapeTracker of each GlobalView op, which generates better
  // code due to shape simplifications.
  let newExp = exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const gid: number = exp.arg[0];
      const st: ShapeTracker = exp.arg[1];
      return accessorGlobal(exp.dtype, gid, st.compose(dim.st), indices);
    }
  });
  // Substitute any remaining gidx/ridx variables not in views.
  const [iexpr, vexpr] = dim.st.toAluExp(indices);
  if (vexpr.min !== 1) throw new Error("Invariant violation: vexpr !== true");
  newExp = newExp.substitute({
    gidx: AluExp.idiv(iexpr, AluExp.i32(reduction.size)).simplify(),
    ridx: AluExp.mod(iexpr, AluExp.i32(reduction.size)).simplify(),
  });

  const outputGidx = dim.outputSt.shape.slice(0, dim.groups);
  const outputUpcast = dim.outputSt.shape.slice(dim.groups);
  const outputIndices = [
    ...unravelAlu(
      outputGidx,
      AluExp.special(DType.Int32, "gidx", prod(outputGidx)),
    ),
    ...unravelAlu(outputUpcast, AluVar.upcast), // Needs later substitution.
  ];
  const [outputIdxExp, _] = dim.outputSt.toAluExp(outputIndices);
  const newEpilogue = reduction.epilogue.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const gid: number = exp.arg[0];
      const st: ShapeTracker = exp.arg[1];
      return accessorGlobal(
        exp.dtype,
        gid,
        st.compose(dim.outputSt),
        outputIndices,
      );
    }
  });

  // Sanity-check that reduction size looks correct.
  if (prod(dim.st.shape.slice(dim.groups, dim.upcast)) !== reduction.size) {
    throw new Error(
      `Invariant violation: reduction size ${reduction.size} does not match ` +
        `tuned dims ${JSON.stringify(dim.st.shape.slice(dim.groups, dim.upcast))}`,
    );
  }

  const size = {
    local: prod(dim.st.shape.slice(dim.local, dim.groups)),
    groups: prod(dim.st.shape.slice(dim.groups, dim.reduce)),
    reduce: prod(dim.st.shape.slice(dim.reduce, dim.unroll)),
    unroll: prod(dim.st.shape.slice(dim.unroll, dim.upcast)),
    upcast: prod(dim.st.shape.slice(dim.upcast)),
  };

  return {
    exp: newExp.simplify(),
    epilogue: newEpilogue.simplify(),
    outputIdxExp: outputIdxExp.simplify(),
    threadCount: ((kernel.size as number) / size.upcast) * size.groups,
    size,
  };
}
