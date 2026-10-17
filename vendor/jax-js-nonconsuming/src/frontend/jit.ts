// Handle Jit operations by translating Jaxprs into dispatched Kernels.

import {
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import {
  _emitCodeCapture,
  _isCodeCaptureEnabled,
  Backend,
  Slot,
  staticCapabilityFingerprint,
} from "../backend";
import { type CostFeatures, evaluateTotalCost } from "../tuner";
import { aluCompare, IPendingExecute } from "./array";
import { executeBlockMap } from "./block-map-executor";
import {
  _registerCacheSizeGetter,
  _registerJitCacheDisposer,
} from "./check-leaks";
import {
  pool,
  poolTranspose,
  prepareConv,
  prepareConv1x1,
} from "./convolution";
import type { ConvParams } from "./convolution";
import {
  Primitive,
  PrimitiveParams,
  promoteAvals,
  routinePrimitives,
  ShapedArray,
} from "./core";
import { type Atom, Jaxpr, JaxprEqn, Lit, Var } from "./jaxpr";
import { executeAssociativeScan, executeScan } from "./scan-executor";
import { planAssociativeScan, planScan } from "./scan-plan";
import type { WasmBackend } from "../backend/wasm";
import {
  canCompileToMegaModule,
  compileToMegaModule,
  type WasmMegaModule,
} from "../backend/wasm/mega-module";
import type { WebGPUBackend } from "../backend/webgpu";
import {
  canCompileToCommandTape,
  type WebGPUCommandTape,
} from "../backend/webgpu/command-tape";
import { PPrint } from "../pprint";
import { Routine } from "../routine";
import {
  type Dim,
  dimProduct,
  hasSymbolicDims,
  isSymbolicSize,
  Pair,
  resolveDim,
  resolveShape,
  resolveSizeExpr,
  ShapeTracker,
  type SizeExpr,
  sizeExprKey,
  sizeExprMul,
  unravelAlu,
} from "../shape";
import {
  DEBUG,
  deepEqual,
  FpHash,
  generalBroadcast,
  prod,
  range,
  rep,
} from "../utils";
import type { AssocScanPlan, ScanPlan } from "./scan-plan";
import type { ScanPath } from "../utils";

// ── Conv classification signal (Phase A0) ─────────────────────────────────
// Tracks which classification the last Primitive.Conv JIT rule produced.
// Testable via _lastConvClass() without parsing console output.

/**
 * Conv classification for the most recent Conv JIT compilation.
 * Values like `block-map-3x3` indicate the shape was *detected* as eligible
 * for a future specialized path — actual lowering may still use generic-dot.
 */
export type ConvClass =
  | "generic-dot"
  | "fast-1x1-dot"
  | "fast-1x1-block-map"
  | "block-map-3x3"
  | "block-map-5x5";

let lastConvClass: ConvClass | null = null;
let lastConvRewritten = false;
let convRewriteEnabled = true;

/** Return the conv classification from the last Primitive.Conv JIT rule. */
export function _lastConvClass(): ConvClass | null {
  return lastConvClass;
}

/** Whether the last JIT-compiled Conv was actually rewritten to BlockMap. */
export function _lastConvRewritten(): boolean {
  return lastConvRewritten;
}

/** Enable/disable the Conv→BlockMap rewrite. For benchmarking A/B comparisons. */
export function _setConvRewriteEnabled(enabled: boolean): void {
  convRewriteEnabled = enabled;
}

function classifyConv(params: ConvParams, kernelShape: number[]): ConvClass {
  // kernelShape is the spatial kernel dimensions (after stripping vmapDims and channel dims)
  const allOnes = kernelShape.every((k) => k === 1);
  const allStride1 = params.strides.every((s) => s === 1);
  const noDilation =
    params.lhsDilation.every((d) => d === 1) &&
    params.rhsDilation.every((d) => d === 1);
  const noGroups = params.vmapDims === 0;

  if (allOnes && allStride1 && noDilation && noGroups) {
    return "fast-1x1-dot";
  }

  // Detect common kernel sizes eligible for block_map lowering.
  // For eligible shapes (stride=1, SAME padding, spatial >= 16),
  // rewriteConvToBlockMap() rewrites Conv → BlockMap before dataflow analysis.
  // Non-eligible shapes fall through to generic-dot via the normal Conv lowering.
  if (noDilation && noGroups && kernelShape.length === 2) {
    const [kH, kW] = kernelShape;
    if (kH === 3 && kW === 3) return "block-map-3x3";
    if (kH === 5 && kW === 5) return "block-map-5x5";
  }

  return "generic-dot";
}

/**
 * Rewrite eligible Conv equations to BlockMap equations before dataflow analysis.
 *
 * Eligible: block-map-3x3/5x5 classification, stride=1, SAME-equivalent padding
 * (padTop+padBottom = kH-1), spatial dims >= 16, concrete shapes.
 *
 * The body jaxpr performs a VALID conv on the halo-expanded tile. The body Conv
 * naturally falls through to generic-dot because VALID padding fails the
 * SAME-equivalence guard (recursion guard).
 *
 * Runs on all backends. On WebGPU, the block_map fused shader handles the
 * im2col+dot body pattern (per-element reduction codegen).
 */
function rewriteConvToBlockMap(jaxpr: Jaxpr): Jaxpr {
  if (!convRewriteEnabled) return jaxpr;

  let changed = false;
  const newEqns: JaxprEqn[] = [];

  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive !== Primitive.Conv) {
      newEqns.push(eqn);
      continue;
    }

    const params = eqn.params as ConvParams;
    const v = params.vmapDims;
    // Only handle v=0 (no vmap) for now
    if (v !== 0) {
      newEqns.push(eqn);
      continue;
    }

    const lhsShape = eqn.inputs[0].aval.shape;
    const rhsShape = eqn.inputs[1].aval.shape;
    // Guard: no symbolic dims
    if (hasSymbolicDims(lhsShape) || hasSymbolicDims(rhsShape)) {
      newEqns.push(eqn);
      continue;
    }

    const lhs = lhsShape as number[];
    const rhs = rhsShape as number[];
    const kernelShape = rhs.slice(2); // v=0, skip C_out and C_in
    const cls = classifyConv(params, kernelShape);

    if (cls !== "block-map-3x3" && cls !== "block-map-5x5") {
      newEqns.push(eqn);
      continue;
    }

    // Guard: stride=1
    if (!params.strides.every((s) => s === 1)) {
      newEqns.push(eqn);
      continue;
    }

    // Guard: SAME-equivalent padding (outH = H, outW = W)
    const [kH, kW] = kernelShape;
    const totalPadH = params.padding[0][0] + params.padding[0][1];
    const totalPadW = params.padding[1][0] + params.padding[1][1];
    if (totalPadH !== kH - 1 || totalPadW !== kW - 1) {
      newEqns.push(eqn);
      continue;
    }

    // Guard: minimum spatial size (avoids overhead for tiny inputs)
    const H = lhs[2];
    const W = lhs[3];
    if (H < 16 || W < 16) {
      newEqns.push(eqn);
      continue;
    }

    changed = true;
    lastConvClass = cls;
    lastConvRewritten = true;

    if (DEBUG >= 1) {
      console.info(
        `conv→block_map: ${cls}, x=[${lhs}], w=[${rhs}], ` +
          `padding=[${params.padding.map((p) => `[${p}]`)}]`,
      );
    }

    // Tile size for output spatial dims
    const tH = Math.min(32, H);
    const tW = Math.min(32, W);

    // Halo: extend input tiles to cover kernel footprint
    const [padTop] = params.padding[0];
    const [padLeft] = params.padding[1];
    const halo_h: [number, number] = [padTop, kH - 1 - padTop];
    const halo_w: [number, number] = [padLeft, kW - 1 - padLeft];

    // Body jaxpr: Conv(xTile, w, VALID) → outTile
    const N = lhs[0];
    const C_in = lhs[1];
    const C_out = rhs[0];

    const tileH_in = tH + kH - 1; // tH + halo_h_lo + halo_h_hi
    const tileW_in = tW + kW - 1;
    const xTileAval = new ShapedArray(
      [N, C_in, tileH_in, tileW_in],
      eqn.inputs[0].aval.dtype,
      false,
    );
    const wAval = new ShapedArray(rhs, eqn.inputs[1].aval.dtype, false);
    const outTileAval = new ShapedArray(
      [N, C_out, tH, tW],
      eqn.outBinders[0].aval.dtype,
      false,
    );

    const xTileVar = new Var(xTileAval);
    const wVar = new Var(wAval);
    const outTileVar = new Var(outTileAval);

    const bodyConvEqn = new JaxprEqn(
      Primitive.Conv,
      [xTileVar, wVar],
      {
        vmapDims: 0,
        strides: [1, 1],
        padding: [
          [0, 0],
          [0, 0],
        ],
        lhsDilation: [1, 1],
        rhsDilation: [1, 1],
      } satisfies ConvParams,
      [outTileVar],
    );
    const bodyJaxpr = new Jaxpr([xTileVar, wVar], [bodyConvEqn], [outTileVar]);

    // BlockMap params
    const blockMapEqn = new JaxprEqn(
      Primitive.BlockMap,
      eqn.inputs,
      {
        jaxpr: bodyJaxpr,
        blockShape: [tH, tW],
        inAxes: [
          [2, 3], // x: grid dims map to spatial axes 2,3
          [null, null], // w: broadcast (same kernel for all tiles)
        ] as (number | null)[][],
        outAxes: [
          [2, 3], // output: grid dims map to spatial axes 2,3
        ] as (number | null)[][],
        numConsts: 0,
        numInputs: 2,
        halo: [
          [halo_h, halo_w], // x: halo per grid dim
          [
            [0, 0],
            [0, 0],
          ], // w: no halo (broadcast)
        ] as [number, number][][],
      },
      eqn.outBinders,
    );
    newEqns.push(blockMapEqn);
  }

  if (!changed) return jaxpr;
  return new Jaxpr(jaxpr.inBinders, newEqns, jaxpr.outs);
}

// ── ForiLoop → BlockMap rewrite (WebGPU) ──────────────────────────────────
// On WebGPU, a standalone fori_loop dispatches body kernels per iteration
// (~100µs per dispatch × N iterations). Wrapping in block_map fuses the
// entire loop into a single WGSL shader with a native `for` — 1 dispatch.
// WASM already compiles fori_loop to a native loop in mega-module, so this
// rewrite is WebGPU-only.

/**
 * Primitives that are strictly point-wise and have no shape-modifying
 * parameters (like axes, starts, limits). These are safe to unconditionally
 * retile simply by overriding their input and output avals.
 */
const pointwisePrimitives = new Set<Primitive>([
  Primitive.Add,
  Primitive.Mul,
  Primitive.Idiv,
  Primitive.Mod,
  Primitive.Min,
  Primitive.Max,
  Primitive.Neg,
  Primitive.Reciprocal,
  Primitive.Floor,
  Primitive.Ceil,
  Primitive.StopGradient,
  Primitive.Cast,
  Primitive.Bitcast,
  Primitive.Sin,
  Primitive.Cos,
  Primitive.Asin,
  Primitive.Atan,
  Primitive.Exp,
  Primitive.Log,
  Primitive.Erf,
  Primitive.Erfc,
  Primitive.Sqrt,
  Primitive.Compare,
  Primitive.Where,
]);

/**
 * Check whether a foriLoop body can be retiled to operate on block-sized data.
 * The body must consist entirely of purely point-wise operations, and all
 * intermediates must match the carry shape exactly or be scalars. This ensures
 * that the body can be safely resized along all grid dimensions without breaking
 * broadcasting semantics or primitive parameters.
 */
function isForiBodyRetilable(
  bodyJaxpr: Jaxpr,
  carryShape: readonly number[],
): boolean {
  for (const eqn of bodyJaxpr.eqns) {
    if (!pointwisePrimitives.has(eqn.primitive)) return false;
    for (const ob of eqn.outBinders) {
      const s = ob.aval.shape;
      if (
        s.length > 0 &&
        !(
          s.length === carryShape.length &&
          s.every((d, i) => d === carryShape[i])
        )
      ) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Clone a jaxpr, replacing every occurrence of `origShape` with `tileShape`.
 * Scalars and differently-shaped vars are left unchanged. Only safe for
 * elementwise-only bodies where all array intermediates share a single shape.
 */
function retileJaxpr(
  jaxpr: Jaxpr,
  origShape: readonly number[],
  tileShape: readonly number[],
): Jaxpr {
  const varMap = new Map<Var, Var>();

  const shapeMatches = (shape: readonly (number | Dim)[]): boolean =>
    shape.length === origShape.length &&
    shape.every((d, i) => d === origShape[i]);

  const retileAval = (aval: ShapedArray): ShapedArray =>
    shapeMatches(aval.shape)
      ? new ShapedArray(tileShape as number[], aval.dtype, aval.weakType)
      : aval;

  const mapVar = (v: Var): Var => {
    let nv = varMap.get(v);
    if (!nv) {
      nv = new Var(retileAval(v.aval));
      varMap.set(v, nv);
    }
    return nv;
  };

  const mapAtom = (a: Atom): Atom => (a instanceof Var ? mapVar(a) : a);

  const newInBinders = jaxpr.inBinders.map(mapVar);
  const newEqns = jaxpr.eqns.map((eqn) => {
    const newInputs = eqn.inputs.map(mapAtom);
    const newOuts = eqn.outBinders.map((ob) => {
      const nv = new Var(retileAval(ob.aval));
      varMap.set(ob, nv);
      return nv;
    });
    return new JaxprEqn(eqn.primitive, newInputs, eqn.params, newOuts);
  });
  const newOuts = jaxpr.outs.map(mapAtom);

  return new Jaxpr(newInBinders, newEqns, newOuts);
}

let lastForiRewritten = false;

/** Prevents foriLoop→blockMap rewrite in nested jitCompile calls (body compilations). */
let _skipForiRewrite = false;

/**
 * Whether the last JIT-compiled ForiLoop was rewritten to BlockMap.
 * For testing and benchmarking.
 */
export function _lastForiRewritten(): boolean {
  return lastForiRewritten;
}

/**
 * Rewrite eligible ForiLoop equations to BlockMap wrapping ForiLoop.
 *
 * Eligible: WebGPU backend, elementwise body, all non-scalar inputs share the
 * same concrete shape, shape large enough to benefit from tiling (≥ 4 per dim).
 *
 * The BlockMap body contains the original ForiLoop operating on tile-sized
 * data. On WebGPU, the fused shader emits a native WGSL `for` loop — 1
 * dispatch regardless of iteration count.
 */
function rewriteForiLoopToBlockMap(jaxpr: Jaxpr, backend: Backend): Jaxpr {
  // Only WebGPU benefits — WASM mega-module already compiles fori_loop natively.
  if (backend.type !== "webgpu") return jaxpr;
  // Skip in nested body compilations (blockMap/scan/foriLoop bodies).
  if (_skipForiRewrite) return jaxpr;

  let changed = false;
  const newEqns: JaxprEqn[] = [];

  if (DEBUG >= 1) {
    const foriCount = jaxpr.eqns.filter(
      (e) => e.primitive === Primitive.ForiLoop,
    ).length;
    console.info(
      `rewriteForiLoopToBlockMap: ${jaxpr.eqns.length} eqns, ${foriCount} ForiLoop`,
    );
  }

  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive !== Primitive.ForiLoop) {
      newEqns.push(eqn);
      continue;
    }

    const params = eqn.params as PrimitiveParams<typeof Primitive.ForiLoop>;
    const { jaxpr: bodyJaxpr, numConsts, lower, upper } = params;

    // Guard: concrete bounds
    if (typeof lower !== "number" || typeof upper !== "number") {
      if (DEBUG >= 1) console.info(`  fori rewrite SKIP: non-concrete bounds`);
      newEqns.push(eqn);
      continue;
    }

    // Guard: at least 2 iterations (overhead of block_map setup)
    if (upper - lower < 2) {
      if (DEBUG >= 1) console.info(`  fori rewrite SKIP: <2 iterations`);
      newEqns.push(eqn);
      continue;
    }

    // Guard: all carries must have same concrete shape, rank >= 1
    const carries = eqn.inputs.slice(numConsts); // init carries
    if (carries.length === 0) {
      if (DEBUG >= 1) console.info(`  fori rewrite SKIP: no carries`);
      newEqns.push(eqn);
      continue;
    }
    const carryShape = carries[0].aval.shape;
    if (hasSymbolicDims(carryShape) || carryShape.length === 0) {
      if (DEBUG >= 1)
        console.info(`  fori rewrite SKIP: symbolic/scalar carry`);
      newEqns.push(eqn);
      continue;
    }
    const cShape = carryShape as number[];
    if (
      !carries.every((c) => {
        const s = c.aval.shape as number[];
        return s.length === cShape.length && s.every((d, i) => d === cShape[i]);
      })
    ) {
      newEqns.push(eqn);
      continue;
    }

    // Guard: body must be retilable (all outputs carry-shaped or scalar,
    // no structural ops or routines)
    if (!isForiBodyRetilable(bodyJaxpr, cShape)) {
      if (DEBUG >= 1) {
        for (const e of bodyJaxpr.eqns) {
          if (!pointwisePrimitives.has(e.primitive)) {
            console.info(
              `  fori rewrite SKIP: non-pointwise primitive #${e.primitive}`,
            );
            break;
          }
          for (const ob of e.outBinders) {
            const s = ob.aval.shape;
            if (
              s.length > 0 &&
              !(
                s.length === cShape.length && s.every((d, i) => d === cShape[i])
              )
            ) {
              console.info(
                `  fori rewrite SKIP: shape mismatch [${s}] vs carry [${cShape}]`,
              );
              break;
            }
          }
        }
      }
      newEqns.push(eqn);
      continue;
    }

    // Guard: consts must be either same-shape-as-carry or scalar
    const consts = eqn.inputs.slice(0, numConsts);
    const constIsTiled: boolean[] = [];
    let constGuardOk = true;
    for (const c of consts) {
      const s = c.aval.shape as number[];
      if (s.length === 0 || hasSymbolicDims(c.aval.shape)) {
        constIsTiled.push(false);
        continue;
      }
      if (s.length === cShape.length && s.every((d, i) => d === cShape[i])) {
        constIsTiled.push(true);
        continue;
      }
      constGuardOk = false;
      break;
    }
    if (!constGuardOk) {
      if (DEBUG >= 1) console.info(`  fori rewrite SKIP: const shape mismatch`);
      newEqns.push(eqn);
      continue;
    }

    // Guard: each carry dimension >= 4 (below this, tiling overhead dominates)
    if (cShape.some((d) => d < 4)) {
      newEqns.push(eqn);
      continue;
    }

    const maxWg = backend.capabilities.maxComputeWorkgroupSizeX ?? 256;
    const maxInvocations =
      backend.capabilities.maxComputeInvocationsPerWorkgroup ?? maxWg;
    const maxShmem =
      backend.capabilities.maxComputeWorkgroupStorageSize ?? 16384;

    const estimateShmemBytes = (shape: number[]): number => {
      const blockElements = shape.reduce((a, b) => a * b, 1);
      let shmemBytes = 0;

      for (let ci = 0; ci < numConsts; ci++) {
        if (constIsTiled[ci]) {
          shmemBytes += blockElements * byteWidth(consts[ci].aval.dtype);
        }
      }
      for (const c of carries) {
        shmemBytes += blockElements * byteWidth(c.aval.dtype);
      }
      for (const e of bodyJaxpr.eqns) {
        for (const ob of e.outBinders) {
          if (ob.aval.shape.length > 0) {
            shmemBytes += blockElements * byteWidth(ob.aval.dtype);
          }
        }
      }

      return shmemBytes;
    };

    const isFeasibleBlockShape = (shape: number[]): boolean => {
      const blockElements = shape.reduce((a, b) => a * b, 1);
      return (
        blockElements <= maxWg &&
        blockElements <= maxInvocations &&
        estimateShmemBytes(shape) <= maxShmem
      );
    };

    // Choose blockShape: tile all carry dimensions using CostEquation.
    const rank = cShape.length;
    let blockShape: number[];

    if (rank === 1) {
      let bestCost = Infinity;
      let bestB = 64;
      for (const b of [256, 128, 64, 32]) {
        const candidateB = Math.min(b, cShape[0]);
        if (!isFeasibleBlockShape([candidateB])) continue;
        const features: CostFeatures = {
          nDispatch: 1,
          nBuffers: 2,
          countAlu: candidateB * 4,
          countMem: candidateB * 8, // read + write
          depthPriv: 4, // pointwise body: add/mul/sub/clip — minimal registers
          sizeShmem: candidateB * 4, // 1 array
          sizeWgsl: 4096,
          parallelism: candidateB,
          produceCount: candidateB,
        };
        const cost = evaluateTotalCost(features, backend.capabilities);
        if (cost < bestCost) {
          bestCost = cost;
          bestB = candidateB;
        }
      }
      if (bestCost === Infinity) {
        if (DEBUG >= 1)
          console.info(`  fori rewrite SKIP: no feasible 1D block shape`);
        newEqns.push(eqn);
        continue;
      }
      blockShape = [Math.min(bestB, cShape[0])];
      if (DEBUG >= 4)
        console.info(
          `rewriteForiLoopToBlockMap: 1D blockShape=[${blockShape}]`,
        );
    } else if (rank === 2) {
      // For 2D, pick from candidates via cost equation
      const candidates = [
        [32, 32],
        [16, 16],
        [16, 8],
        [8, 8],
      ];
      let bestCost = Infinity;
      let bestC = [8, 8];
      for (const c of candidates) {
        if (c[0] > cShape[0] || c[1] > cShape[1]) continue;
        if (!isFeasibleBlockShape(c)) continue;

        const bSize = c[0] * c[1];
        const features: CostFeatures = {
          nDispatch: 1,
          nBuffers: 2,
          countAlu: bSize * 4,
          countMem: bSize * 8,
          depthPriv: 4, // pointwise body: add/mul/sub/clip — minimal registers
          sizeShmem: bSize * 4 * 2, // 2D buffers
          sizeWgsl: 4096,
          parallelism: bSize,
          produceCount: bSize,
        };
        const cost = evaluateTotalCost(features, backend.capabilities);
        if (cost < bestCost) {
          bestCost = cost;
          bestC = c;
        }
      }
      if (bestCost === Infinity) {
        if (DEBUG >= 1)
          console.info(`  fori rewrite SKIP: no feasible 2D block shape`);
        newEqns.push(eqn);
        continue;
      }
      blockShape = bestC;
      if (DEBUG >= 4)
        console.info(
          `rewriteForiLoopToBlockMap: 2D blockShape=[${blockShape}]`,
        );
    } else {
      // rank ≥ 3: tile first 2 dims only, rest handled per element
      // Punt for now — complex to get right. Fall through.
      newEqns.push(eqn);
      continue;
    }

    // Safety check: blockShape was filtered for feasibility above.
    const numTiledConsts = constIsTiled.filter(Boolean).length;
    const shmemBytes = estimateShmemBytes(blockShape);
    if (shmemBytes > maxShmem) {
      if (DEBUG >= 1)
        console.info(`  fori rewrite SKIP: shmem ${shmemBytes} > ${maxShmem}`);
      newEqns.push(eqn);
      continue;
    }

    changed = true;
    lastForiRewritten = true;

    if (DEBUG >= 1) {
      console.info(
        `foriLoop→block_map: shape=[${cShape}], block=[${blockShape}], ` +
          `iters=${upper - lower}, consts=${numConsts}(${numTiledConsts} tiled)`,
      );
    }

    // Tile shape for the block_map body
    const tileShape =
      rank === 1 ? [blockShape[0]] : [blockShape[0], blockShape[1]];

    // Retile the foriLoop body jaxpr to operate on tile-sized data
    const tiledBodyJaxpr = retileJaxpr(bodyJaxpr, cShape, tileShape);

    // Build the outer BlockMap body jaxpr.
    // Inputs: [tiled_const0, ..., tiled_carry0, ...]
    // Body: one ForiLoop equation operating on tiles
    // Outputs: [tiled_result0, ...]
    const bmBodyInBinders: Var[] = [];
    const foriInputAtoms: Atom[] = [];

    // Const inputs to the inner foriLoop (tiled or scalar)
    for (let ci = 0; ci < numConsts; ci++) {
      const orig = eqn.inputs[ci];
      const aval = constIsTiled[ci]
        ? new ShapedArray(tileShape, orig.aval.dtype, orig.aval.weakType)
        : orig.aval;
      const v = new Var(aval);
      bmBodyInBinders.push(v);
      foriInputAtoms.push(v);
    }

    // Carry inputs to the inner foriLoop (always tiled)
    for (let ci = 0; ci < carries.length; ci++) {
      const orig = carries[ci];
      const v = new Var(
        new ShapedArray(tileShape, orig.aval.dtype, orig.aval.weakType),
      );
      bmBodyInBinders.push(v);
      foriInputAtoms.push(v);
    }

    // ForiLoop output vars (tile-sized carries)
    const foriOutVars: Var[] = [];
    for (const outVar of eqn.outBinders) {
      foriOutVars.push(
        new Var(
          new ShapedArray(tileShape, outVar.aval.dtype, outVar.aval.weakType),
        ),
      );
    }

    const foriEqn = new JaxprEqn(
      Primitive.ForiLoop,
      foriInputAtoms,
      { jaxpr: tiledBodyJaxpr, numConsts, lower, upper },
      foriOutVars,
    );

    const bmBodyJaxpr = new Jaxpr(bmBodyInBinders, [foriEqn], foriOutVars);

    // Compute inAxes: tiled inputs map to grid axes, scalar inputs broadcast
    const gridAxes: (number | null)[] = range(rank).map((i) => i);
    const nullAxes: (number | null)[] = range(rank).map(() => null);
    const inAxes: (number | null)[][] = [];
    for (let ci = 0; ci < numConsts; ci++) {
      inAxes.push(constIsTiled[ci] ? [...gridAxes] : [...nullAxes]);
    }
    for (let ci = 0; ci < carries.length; ci++) {
      inAxes.push([...gridAxes]);
    }

    // outAxes: all outputs tiled along grid axes
    const outAxes: (number | null)[][] = eqn.outBinders.map(() => [
      ...gridAxes,
    ]);

    const blockMapEqn = new JaxprEqn(
      Primitive.BlockMap,
      eqn.inputs,
      {
        jaxpr: bmBodyJaxpr,
        blockShape,
        inAxes,
        outAxes,
        numConsts: 0, // BlockMap's own consts (none — all passed as inputs)
        numInputs: numConsts + carries.length,
      } satisfies PrimitiveParams<typeof Primitive.BlockMap>,
      eqn.outBinders,
    );
    newEqns.push(blockMapEqn);
  }

  if (!changed) return jaxpr;
  return new Jaxpr(jaxpr.inBinders, newEqns, jaxpr.outs);
}

/**
 * Rewrite a body jaxpr: replace all `Primitive.BlockIndex` equations with a
 * reference to a new synthetic input (appended as the last inBinder).
 * Returns the original jaxpr unchanged if no BlockIndex is present.
 */
function rewriteBlockIndex(jaxpr: Jaxpr): {
  jaxpr: Jaxpr;
  hasBlockIndex: boolean;
} {
  const biEqns: number[] = [];
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    if (jaxpr.eqns[i].primitive === Primitive.BlockIndex) biEqns.push(i);
  }
  if (biEqns.length === 0) return { jaxpr, hasBlockIndex: false };

  // Create synthetic inBinder for the block index
  const biVar = new Var(new ShapedArray([], DType.Int32, false));

  // Map old BlockIndex output Vars → biVar
  const varMap = new Map<Var, Var>();
  for (const idx of biEqns) {
    for (const outBinder of jaxpr.eqns[idx].outBinders) {
      varMap.set(outBinder, biVar);
    }
  }

  // Remap atom references
  const remapAtom = (a: Atom): Atom =>
    a instanceof Var ? (varMap.get(a) ?? a) : a;

  // Filter out BlockIndex eqns and remap Var references in remaining eqns
  const newEqns: JaxprEqn[] = [];
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    if (jaxpr.eqns[i].primitive === Primitive.BlockIndex) continue;
    const eqn = jaxpr.eqns[i];
    const newInputs = eqn.inputs.map(remapAtom);
    const needsRemap = newInputs.some((a, j) => a !== eqn.inputs[j]);
    if (needsRemap) {
      const newEqn = new JaxprEqn(
        eqn.primitive,
        newInputs,
        eqn.params,
        eqn.outBinders,
      );
      newEqn.copyEffectsFrom(eqn);
      newEqns.push(newEqn);
    } else {
      newEqns.push(eqn);
    }
  }

  const newOuts = jaxpr.outs.map(remapAtom);
  const newJaxpr = new Jaxpr([...jaxpr.inBinders, biVar], newEqns, newOuts);
  return { jaxpr: newJaxpr, hasBlockIndex: true };
}

export type JitId = number;

export type JitStep =
  | {
      type: "execute";
      source: Kernel | Routine;
      inputs: JitId[]; // mapped to backend Slot
      outputs: JitId[]; // mapped to backend Slot
    }
  | {
      type: "malloc";
      size: SizeExpr;
      output: JitId;
      /** Pre-computed constant data to fill the buffer with at allocation time.
       *  When present, no kernel dispatch is needed — the backend uses
       *  writeBuffer (WebGPU) or memcpy (WASM) instead. */
      initialData?: Uint8Array;
    }
  | {
      type: "incref";
      input: JitId;
    }
  | {
      type: "free";
      input: JitId;
    }
  | {
      type: "recycle";
      input: JitId;
      output: JitId;
    }
  | {
      type: "scan";
      plan: ScanPlan;
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      length: number | Dim;
      numCarry: number;
      numConsts: number;
      numX: number;
      numY: number;
      reverse: boolean;
      consts: JitId[];
      initCarry: JitId[];
      xs: JitId[];
      xsAvals: ShapedArray[];
      outputs: JitId[];
    }
  | {
      type: "dus";
      dst: JitId; // the destination buffer (mutated via Mutate effect)
      src: JitId; // the source slice to copy
      output: JitId; // the result (same slot as dst if recycled, else separate)
      offsetBytes: number; // per-fiber byte offset into dst where src is written
      sliceBytes: SizeExpr; // byte size of the src slice (axis=0 fast path)
      dstSizeBytes: SizeExpr; // total byte size of dst (= output size)
      outerFibers: number; // product(shape[0:axis]), 1 for axis=0
      srcFiberBytes: number; // bytes per fiber in src (stride between fibers)
      dstFiberBytes: number; // bytes per fiber in dst (stride between fibers)
      srcBaseOffset?: number; // byte offset into src where fiber reads begin (default 0)
      // ND stride info for multi-axis updates where dst fiber stride is non-constant.
      // When present, the executor decomposes fiber index into coordinates via
      // ndOuterExtents and computes per-fiber dst offset from ndDstStrides.
      ndOuterExtents?: number[]; // srcShape[0..rank-2] — extents per outer dim
      ndDstStrides?: number[]; // dst byte strides per outer dim (same length)
    }
  | {
      type: "scatter_add";
      target: JitId;
      indices: JitId;
      updates: JitId;
      output: JitId;
      axis: number;
      targetShape: number[];
      updatesLen: number; // number of updates along the scatter axis
      dtype: DType;
    }
  | {
      type: "reverse";
      input: JitId;
      output: JitId;
      axis: number;
      axisSize: Dim;
      innerBytes: number;
      totalBytes: SizeExpr;
      dtype: DType;
    }
  | {
      type: "assoc_scan";
      plan: AssocScanPlan;
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      numLeaves: number;
      numConsts: number;
      axis: number;
      reverse: boolean;
      consts: JitId[];
      elems: JitId[];
      constAvals: ShapedArray[];
      elemAvals: ShapedArray[];
      outputs: JitId[];
    }
  | {
      type: "block_map";
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      blockShape: number[];
      inAxes: (number | null)[][];
      outAxes: (number | null)[][];
      numConsts: number;
      numInputs: number;
      /** Potentially symbolic — must be resolved via dimBindings at execution. */
      inputShapes: Dim[][];
      /** Potentially symbolic — must be resolved via dimBindings at execution. */
      outputShapes: Dim[][];
      consts: JitId[];
      inputs: JitId[];
      outputs: JitId[];
      threadTile?: number[];
      /** Explicit grid shape from BlockMap params. */
      explicitGridShape?: number[];
      /** Per-input per-grid-axis overlap. */
      halo?: [number, number][][];
      /** Body uses BlockIndex: the compiled body program has one extra input (block index scalar). */
      hasBlockIndex?: boolean;
    }
  | {
      type: "fori_loop";
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      lower: number | Dim;
      upper: number | Dim;
      numConsts: number;
      consts: JitId[];
      initCarries: JitId[];
      outputs: JitId[];
      /** Byte size of each carry buffer (for copy at end of loop). */
      carrySizeBytes: number[];
    }
  | {
      type: "workgroup_assoc_scan";
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      numConsts: number;
      numElems: number;
      consts: JitId[];
      elems: JitId[];
      outputs: JitId[];
      elemAvals: ShapedArray[];
    };

/** Per-type step counts from {@link JitProgram.stepCounts}. */
export interface JitStepCounts {
  execute: number;
  malloc: number;
  free: number;
  recycle: number;
  incref: number;
  scan: number;
  dus: number;
  scatter_add: number;
  reverse: number;
  assoc_scan: number;
  block_map: number;
  fori_loop: number;
  workgroup_assoc_scan: number;
}

/**
 * Structural stats about a command tape's optimization state.
 * @internal — test-only, not part of the public API contract.
 */
export interface CommandTapeStats {
  /** Whether O9c constants slab is active. */
  hasConstSlab: boolean;
  /** Number of entries in the constants slab. */
  constSlabEntries: number;
  /** Number of O9a-v2 colored arena slab GPUBuffers. */
  arenaSlabCount: number;
  /** Total number of table entries allocated in arena slabs. */
  arenaEntryCount: number;
  /** Number of recycle ops in the tape. */
  recycleCount: number;
  /** Number of dispatch ops in the tape. */
  dispatchCount: number;
  /** Total table size (number of buffer slots). */
  tableSize: number;
}

/**
 * Pool hints computed at JIT compile time. Tells the backend which buffer
 * sizes the program will allocate and how many bytes are live at peak, so the
 * pool can evict stale entries and cap retained memory.
 */
export interface PoolHints {
  /** Peak simultaneously-live bytes across all malloc/free/recycle steps. */
  readonly peakBytes: number;
  /** Set of every malloc'd byte size in the program (padded to 4-byte multiples). */
  readonly mallocSizes: ReadonlySet<number>;
}

/** Walk JitSteps to compute peak live bytes and the set of malloc sizes. */
function computePoolHints(steps: JitStep[]): PoolHints {
  const mallocSizes = new Set<number>();
  // Map JitId → padded byte size for live tracking.
  // WebGPU pads all allocations to 4-byte multiples; use the same padding here
  // so the budget and eviction sets match the actual GPUBuffer sizes.
  const sizeOf = new Map<JitId, number>();
  for (const s of steps) {
    if (s.type === "malloc") {
      // Resolve symbolic sizes using the current dim bindings (set during
      // jitCompile). If no bindings are available, skip — can't compute
      // concrete pool hints.
      let concreteSize: number;
      if (typeof s.size === "number") {
        concreteSize = s.size;
      } else if (_currentDimBindings) {
        concreteSize = resolveSizeExpr(s.size, _currentDimBindings);
      } else {
        continue;
      }
      const padded = Math.ceil(concreteSize / 4) * 4;
      sizeOf.set(s.output, padded);
      mallocSizes.add(padded);
    }
  }

  let liveBytes = 0;
  let peakBytes = 0;
  const live = new Map<JitId, number>(); // JitId → padded byte size (currently live)

  for (const s of steps) {
    switch (s.type) {
      case "malloc": {
        const padded = sizeOf.get(s.output)!;
        live.set(s.output, padded);
        liveBytes += padded;
        break;
      }
      case "free":
        liveBytes -= live.get(s.input) ?? 0;
        live.delete(s.input);
        break;
      case "recycle":
        // Size stays the same, just rename.
        live.set(s.output, live.get(s.input) ?? 0);
        live.delete(s.input);
        break;
      // execute, incref, scan don't change the set of live mallocs.
    }
    if (liveBytes > peakBytes) peakBytes = liveBytes;
  }

  return { peakBytes, mallocSizes };
}

/** Compute slotCount from steps when not provided by the builder. */
function computeSlotCount(steps: JitStep[], inputs: JitId[]): number {
  let max = 0;
  for (const id of inputs) if (id >= max) max = id + 1;
  for (const step of steps) {
    if (step.type === "malloc" && step.output >= max) max = step.output + 1;
    if (step.type === "recycle" && step.output >= max) max = step.output + 1;
  }
  return max;
}

/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
export class JitProgram {
  readonly poolHints: PoolHints;
  /** Total number of distinct JitIds (for fast Slot[] scope). */
  readonly slotCount: number;
  /**
   * Human-readable label for this program.
   * E.g. "mandelbrot#a3f1", "closure#7e2b", "scan_body#c0d4".
   * Set by jitCompile. Combines function name with short hash prefix.
   */
  label?: string;
  /** Cached mega-module: undefined = not attempted, null = unsupported. */
  private _megaModule?: WasmMegaModule | null;
  /** M6.2c: worker pool registration state for parallel mega-module dispatch.
   *  undefined = not attempted, false = registering, true = ready. */
  private _megaModulePoolReady?: boolean;
  /** Cached command tape: undefined = not attempted, null = unsupported. */
  private _commandTape?: WebGPUCommandTape | null;

  /**
   * Destroy GPU resources owned by the cached command tape (if any).
   * Called during cache eviction to prevent GPU memory leaks from persistent
   * uniform buffers, constants slab, and arena slabs.
   */
  _disposeCommandTape(): void {
    if (this._commandTape) {
      (this.backend as WebGPUBackend).destroyCommandTapeResources(
        this._commandTape,
      );
      this._commandTape = null;
    }
  }

  constructor(
    readonly backend: Backend,
    readonly steps: JitStep[],
    readonly inputs: JitId[],
    readonly outputs: JitId[],
    slotCount?: number,
  ) {
    this.poolHints = computePoolHints(steps);
    this.slotCount = slotCount ?? computeSlotCount(steps, inputs);
  }

  pprint(): PPrint {
    const steps: PPrint[] = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const inputsNice = step.inputs
            .map((id, i) => `${i}: %${id}`)
            .join(", ");
          const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
          const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
          if (step.source instanceof Kernel) {
            const kLabel = step.source.label
              ? `, kernel "${step.source.label}"`
              : ", kernel";
            return PPrint.pp(`${executeText}${kLabel}`).concat(
              step.source.pprint().indent(2),
            );
          } else if (step.source instanceof Routine) {
            return PPrint.pp(`${executeText}, routine ${step.source.name}`);
          } else {
            step.source satisfies never; // static check
            return PPrint.pp(executeText);
          }
        }
        case "malloc":
          return PPrint.pp(
            `%${step.output} = malloc <${step.size} bytes>` +
              (step.initialData ? " [prefilled]" : ""),
          );
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
        case "recycle":
          return PPrint.pp(`%${step.output} = recycle %${step.input}`);
        case "scan":
          return PPrint.pp(
            `scan [${step.plan.path}] length=${step.length} numCarry=${step.numCarry} ` +
              `numConsts=${step.numConsts} numX=${step.numX} numY=${step.numY}` +
              (step.reverse ? " reverse" : ""),
          );
        case "dus":
          return PPrint.pp(
            `%${step.output} = dus %${step.dst}[${step.offsetBytes}] <- %${step.src} (${step.sliceBytes} bytes)`,
          );
        case "scatter_add":
          return PPrint.pp(
            `%${step.output} = scatter_add %${step.target} %${step.indices} %${step.updates} axis=${step.axis}`,
          );
        case "reverse":
          return PPrint.pp(
            `%${step.output} = reverse %${step.input} axis=${step.axis} axisSize=${step.axisSize}`,
          );
        case "assoc_scan":
          return PPrint.pp(
            `assoc_scan [${step.plan.path}] numLeaves=${step.numLeaves} numConsts=${step.numConsts} axis=${step.axis}` +
              (step.reverse ? " reverse" : ""),
          );
        case "block_map":
          return PPrint.pp(
            `block_map blockShape=[${step.blockShape}] numConsts=${step.numConsts} numInputs=${step.numInputs} inputShapes=[${step.inputShapes.map((s) => `[${s}]`).join(",")}]`,
          );
        case "fori_loop":
          return PPrint.pp(
            `fori_loop lower=${step.lower} upper=${step.upper} numConsts=${step.numConsts}`,
          );
        case "workgroup_assoc_scan":
          return PPrint.pp(
            `workgroup_assoc_scan numElems=${step.numElems} numConsts=${step.numConsts}`,
          );
      }
    });
    const displayParts: PPrint[] = [];
    if (this.label) displayParts.push(PPrint.pp(`label = "${this.label}"`));
    displayParts.push(
      PPrint.pp(`device = ${this.backend.type}`),
      PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"),
      PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"),
      PPrint.pp("steps ="),
      PPrint.prototype.concat(...steps).indent(2),
    );
    const display = PPrint.prototype.concat(...displayParts);
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }

  toString(): string {
    return this.pprint().toString();
  }

  /**
   * Return structural stats about the cached command tape (if any).
   * Useful for verifying that O9c constants slab and O9a-v2 colored arena
   * are active, not silently bypassed.
   * @internal — test-only, not part of the public API contract.
   */
  commandTapeStats(): CommandTapeStats | null {
    if (!this._commandTape) return null;
    const tape = this._commandTape;
    let recycleCount = 0;
    let dispatchCount = 0;
    for (const op of tape.ops) {
      if (op.type === "recycle") recycleCount++;
      if (op.type === "dispatch") dispatchCount++;
    }
    return {
      hasConstSlab: tape.constSlab !== null,
      constSlabEntries: tape.constSlab?.entries.length ?? 0,
      arenaSlabCount: tape.arenaSlabs?.length ?? 0,
      arenaEntryCount:
        tape.arenaSlabs?.reduce((n, s) => n + s.entries.length, 0) ?? 0,
      recycleCount,
      dispatchCount,
      tableSize: tape.tableSize,
    };
  }

  /**
   * Count steps by type. Useful for verifying fusion reduces dispatch count.
   *
   * @example
   * ```ts
   * const counts = program.stepCounts();
   * expect(counts.execute).toBe(1); // verify single dispatch after fusion
   * ```
   */
  stepCounts(): JitStepCounts {
    const counts: JitStepCounts = {
      execute: 0,
      malloc: 0,
      free: 0,
      recycle: 0,
      incref: 0,
      scan: 0,
      dus: 0,
      scatter_add: 0,
      reverse: 0,
      assoc_scan: 0,
      block_map: 0,
      fori_loop: 0,
      workgroup_assoc_scan: 0,
    };
    for (const step of this.steps) {
      counts[step.type]++;
    }
    return counts;
  }

  /**
   * Execute the JitProgram with the given inputs.
   * @param dimBindings Optional map of symbolic dimension names to concrete
   *   values. Required when the program contains symbolic sizes.
   */
  execute(
    inputs: Slot[],
    dimBindings?: ReadonlyMap<string, number>,
  ): { outputs: Slot[]; pending: IPendingExecute[] } {
    // Tell the backend which buffer sizes we'll need and our peak memory,
    // so it can evict stale pool entries and cap retained bytes.
    this.backend.configurePool?.(this.poolHints);

    // Mega-module fast path (WASM only, kernel-only programs):
    // Compiles all steps into a single WASM function, eliminating
    // JS↔WASM boundary crossings between kernel dispatches.
    if (this.backend.type === "wasm") {
      if (this._megaModule === undefined) {
        this._megaModule = canCompileToMegaModule(this.steps)
          ? compileToMegaModule(this.steps, this.inputs, this.outputs)
          : null;
      }
      if (this._megaModule) {
        const wasmBackend = this.backend as WasmBackend;

        // M6.2c: parallel mega-module dispatch for programs with large kernels.
        // First call triggers async worker registration (falls through to
        // monolithic path). Once registered, subsequent calls use the parallel
        // path which fans out large kernels across workers.
        if (this._megaModulePoolReady === true) {
          // Worker pool ready — use parallel step-by-step dispatch
          const outputSlots = wasmBackend.executeMegaModuleParallelSync(
            this._megaModule,
            inputs,
          );
          return { outputs: outputSlots, pending: [] };
        }

        if (
          this._megaModulePoolReady === undefined &&
          wasmBackend.shouldUseParallelMegaModule(this._megaModule)
        ) {
          // First call: kick off async registration, fall through to
          // monolithic path for this invocation.
          this._megaModulePoolReady = false;
          const mm = this._megaModule;
          wasmBackend
            .registerMegaModuleOnPool(mm)
            .then(() => {
              this._megaModulePoolReady = true;
            })
            .catch(() => {
              // Registration failed — stay on monolithic path
              this._megaModulePoolReady = undefined;
            });
        }

        // Monolithic path: orchestrator (M6.2b) or direct execution
        const outputSlots = wasmBackend.executeMegaModule(
          this._megaModule,
          inputs,
        );
        return { outputs: outputSlots, pending: [] };
      }
    }

    // Command tape fast path (WebGPU only, kernel/routine-only programs):
    // Pre-compiles the dispatch sequence into a flat representation with
    // pre-resolved pipelines and buffer table indices, eliminating per-step
    // JS overhead (scope lookups, array alloc, refcounting, pipeline lookups).
    if (this.backend.type === "webgpu") {
      if (this._commandTape === undefined) {
        this._commandTape = canCompileToCommandTape(this.steps)
          ? (this.backend as WebGPUBackend).compileCommandTape(
              this.steps,
              this.inputs,
              this.outputs,
            )
          : null;
      }
      if (this._commandTape) {
        return (this.backend as WebGPUBackend).executeCommandTape(
          this._commandTape,
          inputs,
        );
      }
    }

    const scope: Slot[] = new globalThis.Array(this.slotCount);
    if (inputs.length !== this.inputs.length) {
      throw new TypeError(
        `Expected ${this.inputs.length} inputs, got ${inputs.length}`,
      );
    }
    for (let i = 0; i < this.inputs.length; i++) {
      scope[this.inputs[i]] = inputs[i];
    }

    // Batch direct dispatch: collect execute steps and flush as a single
    // beginBatch/endBatch block.  Avoids IPendingExecute object allocation
    // and Map overhead while preserving ref-count safety.
    type BatchEntry = {
      source: Kernel | Routine;
      inputs: Slot[];
      outputs: Slot[];
      dynamicParams?: number[];
    };
    const batch: BatchEntry[] = [];

    const flushBatch = () => {
      if (batch.length === 0) return;
      this.backend.beginBatch?.();
      try {
        for (let i = 0; i < batch.length; i++) {
          const e = batch[i];
          const exe =
            e.source instanceof Kernel
              ? this.backend.prepareKernelSync(e.source)
              : this.backend.prepareRoutineSync(e.source);
          this.backend.dispatch(exe, e.inputs, e.outputs, e.dynamicParams);
          for (const s of e.inputs) this.backend.decRef(s);
          for (const s of e.outputs) this.backend.decRef(s);
        }
      } finally {
        this.backend.endBatch?.();
      }
      batch.length = 0;
    };

    const flushSubPending = (sub: IPendingExecute[]) => {
      if (sub.length === 0) return;
      flushPendingBatched(sub, this.backend);
    };
    for (const step of this.steps) {
      switch (step.type) {
        case "execute": {
          const si = step.inputs;
          const so = step.outputs;
          const ins: Slot[] = new globalThis.Array(si.length);
          for (let j = 0; j < si.length; j++) ins[j] = scope[si[j]];
          const outs: Slot[] = new globalThis.Array(so.length);
          for (let j = 0; j < so.length; j++) outs[j] = scope[so[j]];

          // Hold refs to prevent premature freeing by interleaved free steps
          for (const s of ins) this.backend.incRef(s);
          for (const s of outs) this.backend.incRef(s);

          // Compute dynamicParams for kernels with symbolic dimensions
          let dynamicParams: number[] | undefined;
          if (
            step.source instanceof Kernel &&
            step.source.needsDynamicParams &&
            dimBindings
          ) {
            const resolvedSize = resolveSizeExpr(step.source.size, dimBindings);
            dynamicParams = [resolvedSize];
            const re = step.source.outputs[0].reduction;
            if (re && isSymbolicSize(re.size)) {
              dynamicParams.push(resolveSizeExpr(re.size, dimBindings));
            }
          }
          batch.push({
            source: step.source,
            inputs: ins,
            outputs: outs,
            dynamicParams,
          });
          break;
        }
        case "malloc": {
          const concreteSize =
            typeof step.size === "number"
              ? step.size
              : resolveSizeExpr(step.size, dimBindings!);
          scope[step.output] = this.backend.malloc(
            concreteSize,
            step.initialData,
          );
          break;
        }
        case "incref": {
          this.backend.incRef(scope[step.input]);
          break;
        }
        case "free": {
          this.backend.decRef(scope[step.input]);
          break;
        }
        case "recycle": {
          scope[step.output] = scope[step.input];
          break;
        }
        case "scan": {
          // Flush batched dispatches before scan — scan needs materialized inputs
          flushBatch();

          // Resolve slots from scope
          const constSlots = step.consts.map((id) => scope[id]);
          const initCarrySlots = step.initCarry.map((id) => scope[id]);
          const xsSlots = step.xs.map((id) => scope[id]);
          const outputSlots = step.outputs.map((id) => scope[id]);

          // IncRef consts and xs — executeScan borrows them
          for (const s of constSlots) this.backend.incRef(s);
          for (const s of xsSlots) this.backend.incRef(s);

          const result = executeScan({
            backend: this.backend,
            plan: step.plan,
            bodyProgram: step.bodyProgram,
            bodyJaxpr: step.bodyJaxpr,
            length:
              typeof step.length === "number"
                ? step.length
                : resolveDim(step.length, dimBindings!),
            numCarry: step.numCarry,
            numConsts: step.numConsts,
            numX: step.numX,
            numY: step.numY,
            reverse: step.reverse,
            constSlots,
            initCarrySlots,
            xsSlots,
            xsAvals: step.xsAvals,
            outputSlots,
            dimBindings,
          });

          // DecRef borrowed consts and xs
          for (const s of constSlots) this.backend.decRef(s);
          for (const s of xsSlots) this.backend.decRef(s);

          // Flush sub-executor pending ops immediately
          flushSubPending(result.pending);

          // Update scope with output slots
          for (let oi = 0; oi < step.outputs.length; oi++) {
            scope[step.outputs[oi]] = result.outputs[oi];
          }
          break;
        }
        case "dus": {
          // Flush batched dispatches — DUS needs materialized inputs
          flushBatch();

          const dstSlot = scope[step.dst];
          const srcSlot = scope[step.src];
          const outSlot = scope[step.output];

          // Zero-copy: if effectDrivenAllocate recycled dst → output,
          // skip the full copy (they share the same buffer).
          const concreteDstSize =
            typeof step.dstSizeBytes === "number"
              ? step.dstSizeBytes
              : resolveSizeExpr(step.dstSizeBytes, dimBindings!);

          // Batch all DUS copies into a single command submission
          this.backend.beginBatch?.();
          if (dstSlot !== outSlot) {
            this.backend.copyBufferToBuffer!(
              dstSlot,
              0,
              outSlot,
              0,
              concreteDstSize,
            );
          }
          // Copy src slice into output — fiber loop for axis > 0
          if (step.outerFibers === 1) {
            // Contiguous fast path (axis=0)
            const concreteSliceSize =
              typeof step.sliceBytes === "number"
                ? step.sliceBytes
                : resolveSizeExpr(step.sliceBytes, dimBindings!);
            this.backend.copyBufferToBuffer!(
              srcSlot,
              step.srcBaseOffset ?? 0,
              outSlot,
              step.offsetBytes,
              concreteSliceSize,
            );
          } else if (step.ndOuterExtents) {
            // ND fiber loop: coordinate decomposition for non-constant dst stride.
            // src is row-major contiguous, so fibers are evenly spaced.
            const extents = step.ndOuterExtents;
            const dstStrides = step.ndDstStrides!;
            const rowBytes = step.srcFiberBytes; // innermost row size
            // Precompute radix divisors: divisors[k] = prod(extents[k+1:])
            const divisors = new Array<number>(extents.length);
            divisors[extents.length - 1] = 1;
            for (let k = extents.length - 2; k >= 0; k--)
              divisors[k] = divisors[k + 1] * extents[k + 1];
            for (let i = 0; i < step.outerFibers; i++) {
              // Decompose fiber index into coordinates → dst offset
              let rem = i;
              let dstOff = step.offsetBytes;
              for (let k = 0; k < extents.length; k++) {
                const coord = Math.floor(rem / divisors[k]);
                rem = rem % divisors[k];
                dstOff += coord * dstStrides[k];
              }
              this.backend.copyBufferToBuffer!(
                srcSlot,
                i * rowBytes,
                outSlot,
                dstOff,
                rowBytes,
              );
            }
          } else {
            // Fiber-by-fiber copy for non-contiguous axis > 0
            const srcBase = step.srcBaseOffset ?? 0;
            for (let i = 0; i < step.outerFibers; i++) {
              this.backend.copyBufferToBuffer!(
                srcSlot,
                srcBase + i * step.srcFiberBytes,
                outSlot,
                i * step.dstFiberBytes + step.offsetBytes,
                step.srcFiberBytes,
              );
            }
          }
          this.backend.endBatch?.();
          break;
        }
        case "scatter_add": {
          // Flush batched dispatches — scatter_add needs materialized inputs
          flushBatch();

          const targetSlot = scope[step.target];
          const indicesSlot = scope[step.indices];
          const updatesSlot = scope[step.updates];
          const outSlot = scope[step.output];

          // Copy target to output if not already recycled
          const targetBytes = prod(step.targetShape) * byteWidth(step.dtype);
          // Batch copy + dispatch into a single command submission
          this.backend.beginBatch?.();
          if (targetSlot !== outSlot) {
            this.backend.copyBufferToBuffer!(
              targetSlot,
              0,
              outSlot,
              0,
              targetBytes,
            );
          }

          // Dispatch scatter_add kernel on the backend
          this.backend.dispatchScatterAdd!(
            outSlot,
            indicesSlot,
            updatesSlot,
            step.axis,
            step.targetShape,
            step.updatesLen,
            step.dtype,
          );
          this.backend.endBatch?.();
          break;
        }
        case "reverse": {
          // Flush batched dispatches — reverse needs materialized input
          flushBatch();

          const inputSlot = scope[step.input];
          const outSlot = scope[step.output];
          const concreteAxisSize =
            typeof step.axisSize === "number"
              ? step.axisSize
              : dimBindings!.get(
                  (step.axisSize as import("../dim").SymDim).name,
                )!;

          if (this.backend.reverseBuffer) {
            this.backend.reverseBuffer(
              inputSlot,
              outSlot,
              concreteAxisSize,
              step.innerBytes,
              step.dtype,
            );
          } else {
            // Generic fallback: copy slices in reverse order
            // Batch all reverse copies into a single command submission
            this.backend.beginBatch?.();
            for (let i = 0; i < concreteAxisSize; i++) {
              this.backend.copyBufferToBuffer(
                inputSlot,
                i * step.innerBytes,
                outSlot,
                (concreteAxisSize - 1 - i) * step.innerBytes,
                step.innerBytes,
              );
            }
            this.backend.endBatch?.();
          }
          break;
        }
        case "assoc_scan": {
          // Flush batched dispatches — assoc_scan needs materialized inputs
          flushBatch();

          const constSlots = step.consts.map((id) => scope[id]);
          const elemSlots = step.elems.map((id) => scope[id]);
          const outputSlots = step.outputs.map((id) => scope[id]);

          const assocResult = executeAssociativeScan({
            backend: this.backend,
            plan: step.plan,
            bodyJaxpr: step.bodyJaxpr,
            numLeaves: step.numLeaves,
            numConsts: step.numConsts,
            axis: step.axis,
            reverse: step.reverse,
            constSlots,
            elemSlots,
            constAvals: step.constAvals,
            elemAvals: step.elemAvals,
            outputSlots,
            dimBindings,
          });

          // Update scope with result slots (may differ from pre-allocated
          // output slots when the fallback path replaces them).
          for (let i = 0; i < step.numLeaves; i++) {
            scope[step.outputs[i]] = assocResult.outputs[i];
          }
          flushSubPending(assocResult.pending);
          break;
        }
        case "block_map": {
          // Flush batched dispatches — block_map needs materialized inputs
          flushBatch();

          const constSlots = step.consts.map((id) => scope[id]);
          const inputSlots = step.inputs.map((id) => scope[id]);
          const outputSlots = step.outputs.map((id) => scope[id]);

          // Resolve potentially-symbolic shapes to concrete numbers.
          const inputShapes = step.inputShapes.map((s) =>
            dimBindings ? resolveShape(s, dimBindings) : (s as number[]),
          );
          const outputShapes = step.outputShapes.map((s) =>
            dimBindings ? resolveShape(s, dimBindings) : (s as number[]),
          );

          // Compute gridShape from explicit params or resolved input shapes.
          const gridRank = step.blockShape.length;
          const gridShape: number[] = step.explicitGridShape
            ? [...step.explicitGridShape]
            : new globalThis.Array(gridRank).fill(0);
          if (!step.explicitGridShape) {
            for (let ii = 0; ii < step.numInputs; ii++) {
              const axes = step.inAxes[ii];
              for (let g = 0; g < gridRank; g++) {
                if (axes[g] !== null) {
                  gridShape[g] = Math.ceil(
                    inputShapes[ii][axes[g]!] / step.blockShape[g],
                  );
                }
              }
            }
          }

          const bmResult = executeBlockMap({
            backend: this.backend,
            bodyProgram: step.bodyProgram,
            bodyJaxpr: step.bodyJaxpr,
            blockShape: step.blockShape,
            inAxes: step.inAxes,
            outAxes: step.outAxes,
            numConsts: step.numConsts,
            numInputs: step.numInputs,
            gridShape,
            inputShapes,
            outputShapes,
            constSlots,
            inputSlots,
            outputSlots,
            threadTile: step.threadTile,
            hasBlockIndex: step.hasBlockIndex,
            halo: step.halo,
          });

          for (let i = 0; i < step.outputs.length; i++) {
            scope[step.outputs[i]] = bmResult.outputs[i];
          }
          flushSubPending(bmResult.pending);

          break;
        }
        case "fori_loop": {
          // Flush batched dispatches — fori_loop needs materialized inputs
          flushBatch();

          const lower =
            typeof step.lower === "number"
              ? step.lower
              : resolveDim(step.lower, dimBindings!);
          const upper =
            typeof step.upper === "number"
              ? step.upper
              : resolveDim(step.upper, dimBindings!);

          const constSlots = step.consts.map((id) => scope[id]);
          let carrySlots = step.initCarries.map((id) => scope[id]);
          let ownsCarry = false; // first iteration uses parent-owned init carries

          for (let i = lower; i < upper; i++) {
            // Create scalar int32 index slot
            const idxData = new Int32Array([i]);
            const idxSlot = this.backend.malloc(
              4,
              new Uint8Array(idxData.buffer),
            );

            // IncRef consts (body borrows them)
            for (const s of constSlots) this.backend.incRef(s);

            const bodyInputs = [...constSlots, idxSlot, ...carrySlots];
            const bodyResult = step.bodyProgram.execute(
              bodyInputs,
              dimBindings,
            );
            flushPendingBatched(bodyResult.pending, this.backend);

            // DecRef consts and index
            for (const s of constSlots) this.backend.decRef(s);
            this.backend.decRef(idxSlot);

            // Release previous carry if we own it (body outputs from prior iterations)
            if (ownsCarry) {
              for (const s of carrySlots) this.backend.decRef(s);
            }
            carrySlots = bodyResult.outputs;
            ownsCarry = true;
          }

          // Write final carries to output slots
          // Batch carry copies into a single command submission
          this.backend.beginBatch?.();
          for (let k = 0; k < step.outputs.length; k++) {
            const outSlot = scope[step.outputs[k]];
            const carrySlot = carrySlots[k];
            this.backend.copyBufferToBuffer(
              carrySlot,
              0,
              outSlot,
              0,
              step.carrySizeBytes[k],
            );
            if (ownsCarry) this.backend.decRef(carrySlot);
          }
          this.backend.endBatch?.();
          break;
        }
        case "workgroup_assoc_scan": {
          // Fallback: run as sequential associative scan on the block data.
          // Flush batched dispatches first.
          flushBatch();

          const constSlots = step.consts.map((id) => scope[id]);
          const elemSlots = step.elems.map((id) => scope[id]);
          const outputSlots = step.outputs.map((id) => scope[id]);

          const N = step.elemAvals[0].shape[0] as number;
          const numElems = step.numElems;

          // Sequential prefix scan: y[0] = x[0], y[i] = body(y[i-1], x[i])
          // We run evalJaxpr for each position, slicing/concatenating.
          // For the fallback path inside block_map, N = blockSize (small).
          const elemBytes = step.elemAvals.map(
            (a) => (a.size / (a.shape[0] as number)) * byteWidth(a.dtype),
          );

          // Read initial elements at position 0 → carry
          let carrySlots: Slot[] = [];
          for (let e = 0; e < numElems; e++) {
            const slotBytes = elemBytes[e];
            const carry = this.backend.malloc(slotBytes);
            this.backend.copyBufferToBuffer(
              elemSlots[e],
              0,
              carry,
              0,
              slotBytes,
            );
            carrySlots.push(carry);
            // Write to output position 0
            this.backend.copyBufferToBuffer(
              carry,
              0,
              outputSlots[e],
              0,
              slotBytes,
            );
          }

          for (let i = 1; i < N; i++) {
            // Slice element at position i
            const bSlots: Slot[] = [];
            for (let e = 0; e < numElems; e++) {
              const slotBytes = elemBytes[e];
              const b = this.backend.malloc(slotBytes);
              this.backend.copyBufferToBuffer(
                elemSlots[e],
                i * slotBytes,
                b,
                0,
                slotBytes,
              );
              bSlots.push(b);
            }

            // IncRef consts for body
            for (const s of constSlots) this.backend.incRef(s);

            // Flush the batch encoder before body execution: the copy commands
            // above are still in the batch encoder. If the inner body uses the
            // command tape fast path, it creates its own GPUCommandEncoder — so
            // the copies must be submitted first or the body reads stale data.
            this.backend.flushBatch?.();

            // Body inputs: [consts, a (carry), b (current)]
            const bodyInputs = [...constSlots, ...carrySlots, ...bSlots];
            const bodyResult = step.bodyProgram.execute(bodyInputs);
            flushPendingBatched(bodyResult.pending, this.backend);

            // DecRef consts
            for (const s of constSlots) this.backend.decRef(s);

            // Release old carry and b slots
            for (const s of carrySlots) this.backend.decRef(s);
            for (const s of bSlots) this.backend.decRef(s);

            // New carry = body outputs
            carrySlots = bodyResult.outputs;

            // Write carry to output position i
            for (let e = 0; e < numElems; e++) {
              const slotBytes = elemBytes[e];
              this.backend.copyBufferToBuffer(
                carrySlots[e],
                0,
                outputSlots[e],
                i * slotBytes,
                slotBytes,
              );
            }
          }

          // Release final carry
          for (const s of carrySlots) this.backend.decRef(s);
          break;
        }
        default:
          step satisfies never;
      }
    }
    flushBatch();
    const outputSlots: Slot[] = new globalThis.Array(this.outputs.length);
    for (let i = 0; i < this.outputs.length; i++) {
      outputSlots[i] = scope[this.outputs[i]];
    }
    return {
      outputs: outputSlots,
      pending: [] as IPendingExecute[],
    };
  }
}

/** Flush pending ops with batched dispatch when the backend supports it. */
function flushPendingBatched(
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

/** Check whether a JitStep references the given JitId as input or output. */
function stepUsesId(step: JitStep, id: JitId): boolean {
  switch (step.type) {
    case "execute":
      return step.inputs.includes(id) || step.outputs.includes(id);
    case "malloc":
      return step.output === id;
    case "scan":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.initCarry.includes(id) ||
        step.xs.includes(id)
      );
    case "dus":
      return step.dst === id || step.src === id || step.output === id;
    case "scatter_add":
      return (
        step.target === id ||
        step.indices === id ||
        step.updates === id ||
        step.output === id
      );
    case "reverse":
      return step.input === id || step.output === id;
    case "assoc_scan":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.elems.includes(id)
      );
    case "block_map":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.inputs.includes(id)
      );
    case "fori_loop":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.initCarries.includes(id)
      );
    case "workgroup_assoc_scan":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.elems.includes(id)
      );
    default:
      return false;
  }
}

class JitProgramBuilder {
  backend: Backend;
  #nextId: number;
  steps: JitStep[];

  /** Number of distinct JitIds allocated (for sizing Slot[] scope). */
  get slotCount(): number {
    return this.#nextId;
  }

  constructor(backend: Backend, nargs: number) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }

  pushLit(lit: Lit): JitId {
    // Compute the constant value as raw bytes and embed it directly in the
    // malloc step. This avoids dispatching a zero-input kernel (GPU shader or
    // WASM routine) just to fill a scalar buffer with a constant.
    const bw = byteWidth(lit.dtype);
    const buf = new ArrayBuffer(bw);
    const view = new DataView(buf);
    switch (lit.dtype) {
      case DType.Float32:
        view.setFloat32(0, lit.value, true);
        break;
      case DType.Int32:
      case DType.Bool:
        view.setInt32(0, lit.value | 0, true);
        break;
      case DType.Uint32:
        view.setUint32(0, lit.value >>> 0, true);
        break;
      case DType.Float16:
        view.setFloat16(0, lit.value, true);
        break;
      case DType.Float64:
        view.setFloat64(0, lit.value, true);
        break;
    }
    return this.pushBuffer(bw, new Uint8Array(buf));
  }

  pushBuffer(size: SizeExpr, initialData?: Uint8Array): JitId {
    const id = this.#nextId++;
    const step: JitStep = {
      type: "malloc",
      size,
      output: id,
    };
    if (initialData) step.initialData = initialData;
    this.steps.push(step);
    return id;
  }

  pushKernel(kernel: Kernel, inputs: JitId[]): JitId {
    const id = this.pushBuffer(kernel.outputs[0].bytes);
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: [id],
    });
    return id;
  }

  /**
   * Push a multi-output kernel. Allocates one buffer per output and emits a
   * single execute step. Returns array of JitIds, one per kernel output.
   */
  pushMultiKernel(kernel: Kernel, inputs: JitId[]): JitId[] {
    const ids: JitId[] = [];
    for (const o of kernel.outputs) {
      ids.push(this.pushBuffer(o.bytes));
    }
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: ids,
    });
    return ids;
  }

  pushRoutine(routine: Routine, inputs: JitId[], outputs: JitId[]): void {
    this.steps.push({
      type: "execute",
      source: routine,
      inputs,
      outputs,
    });
  }

  pushIncref(id: JitId): void {
    this.steps.push({
      type: "incref",
      input: id,
    });
  }

  pushFree(id: JitId): void {
    // Should be paired with the output of pushKernel() when last used.
    this.steps.push({
      type: "free",
      input: id,
    });
  }

  /**
   * Effect-driven buffer lifecycle allocation.
   *
   * Replaces the two-pass `insertFreeSteps` + `recycleBuffers` with a
   * single-pass algorithm that uses liveness analysis for optimal buffer
   * reuse. Key improvement: can recycle buffers across execute/scan step
   * boundaries, catching opportunities the old adjacent-pair scanner missed.
   *
   * Algorithm:
   * 1. Collect all malloc'd JitIds and their byte sizes.
   * 2. Compute last-use step index for each (excluding program outputs).
   * 3. Forward pass: when a JitId dies, add to a free pool (keyed by size).
   *    When a malloc is needed, check the pool → emit recycle or malloc.
   * 4. Emit free steps for any remaining pooled buffers.
   */
  effectDrivenAllocate(outputIds: JitId[]): void {
    // Phase 1: Collect all malloc'd JitIds and their sizes
    const mallocSizes = new Map<JitId, SizeExpr>();
    for (const s of this.steps) {
      if (s.type === "malloc") mallocSizes.set(s.output, s.size);
    }

    // Phase 2: Compute last-use step index for each non-output malloc'd JitId
    const outputSet = new Set(outputIds);
    const lastUse = new Map<JitId, number>();
    for (const [id] of mallocSizes) {
      if (outputSet.has(id)) continue;
      for (let j = this.steps.length - 1; j >= 0; j--) {
        if (stepUsesId(this.steps[j], id)) {
          lastUse.set(id, j);
          break;
        }
      }
    }

    // Pre-compute which JitIds die after each step index
    const pendingFree = new Map<number, JitId[]>();
    for (const [id, stepIdx] of lastUse) {
      let list = pendingFree.get(stepIdx);
      if (!list) {
        list = [];
        pendingFree.set(stepIdx, list);
      }
      list.push(id);
    }

    // Phase 3: Single forward pass with free pool for cross-boundary recycling
    const freePool = new Map<string | number, JitId[]>(); // sizeExprKey → available JitIds
    const newSteps: JitStep[] = [];
    let recycleCount = 0;

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];

      if (step.type === "malloc") {
        // Malloc steps with initialData must always allocate fresh — a recycled
        // buffer would contain stale data. These are tiny (scalar) buffers from
        // pushLit, so the recycling loss is negligible.
        if (step.initialData) {
          newSteps.push(step);
        } else {
          // Check free pool for a same-size buffer to recycle
          const key = sizeExprKey(step.size);
          const pool = freePool.get(key);
          if (pool && pool.length > 0) {
            const reusedId = pool.pop()!;
            newSteps.push({
              type: "recycle",
              input: reusedId,
              output: step.output,
            });
            recycleCount++;
          } else {
            newSteps.push(step);
          }
        }
      } else {
        newSteps.push(step);
      }

      // After this step, release any JitIds whose lifetime ends here
      const dying = pendingFree.get(i);
      if (dying) {
        for (const id of dying) {
          const size = mallocSizes.get(id);
          if (size === undefined) continue;
          const key = sizeExprKey(size);
          let pool = freePool.get(key);
          if (!pool) {
            pool = [];
            freePool.set(key, pool);
          }
          pool.push(id);
        }
      }
    }

    // Phase 4: Emit free steps for any pooled buffers not recycled
    for (const [, pool] of freePool) {
      for (const id of pool) {
        newSteps.push({ type: "free", input: id });
      }
    }

    this.steps = newSteps;
    if (DEBUG >= 1 && recycleCount > 0) {
      console.info(`jit: effect-driven recycled ${recycleCount} buffer(s)`);
    }
  }
}

type JitValue =
  | { type: "imm"; arg: JitId } // Immediate
  | { type: "exp"; exp: AluExp; args: JitId[] } // Expression, lazily fused
  | { type: "red"; exp: AluExp; reduction: Reduction; args: JitId[] }; // Reduction + epilogue

const jitCompileCache = new Map<string, JitProgram>();

/**
 * Clear the internal JIT compilation cache. Called by `_disposeAllJitCaches()`
 * in jaxpr.ts during leak checking.
 *
 * Destroys GPU resources owned by cached command tapes (uniform buffers,
 * constants slabs, arena slabs) before dropping JitProgram references.
 * @internal
 */
export function _clearJitCompileCache(): void {
  for (const prog of jitCompileCache.values()) {
    prog._disposeCommandTape();
  }
  jitCompileCache.clear();
}

// Register with jaxpr.ts so checkLeaks.stop() can flush this cache.
_registerJitCacheDisposer(_clearJitCompileCache);
_registerCacheSizeGetter("jitCompile", () => jitCompileCache.size);

/**
 * Return command tape stats for all cached WebGPU JitPrograms that have a
 * compiled command tape. Useful for verifying that O9c/O9a-v2 optimizations
 * are active in test scenarios.
 * @internal — test-only.
 */
export function _getCommandTapeStats(): CommandTapeStats[] {
  const results: CommandTapeStats[] = [];
  for (const prog of jitCompileCache.values()) {
    const stats = prog.commandTapeStats();
    if (stats) results.push(stats);
  }
  return results;
}

/**
 * Module-level dim bindings for jitRules to use when resolving symbolic shapes
 * to concrete values for ShapeTracker operations. Set by jitCompile().
 * @internal
 */
let _currentDimBindings: ReadonlyMap<string, number> | undefined;

export function jitCompile(
  backend: Backend,
  jaxpr: Jaxpr,
  dimBindings?: ReadonlyMap<string, number>,
  label?: string,
): JitProgram {
  // Cache key includes static device fingerprint + calibrated performance
  // parameters so that tile/block config selections change when the hardware
  // profile is updated by microbenchmark calibration (P10).
  const caps = backend.capabilities;
  let capsKey = "";
  if (caps.calibrated) {
    const r = (v: number | undefined) =>
      v != null ? Number(v.toPrecision(3)) : 0;
    capsKey =
      "," +
      staticCapabilityFingerprint(caps) +
      ",d" +
      r(caps.dispatchOverheadUs) +
      "b" +
      r(caps.bandwidthGBs) +
      "t" +
      r(caps.tflops) +
      "r" +
      (caps.rOptWords ?? 0);
  }
  const cacheKey = backend.type + "," + FpHash.hash(jaxpr) + capsKey;

  const cached = jitCompileCache.get(cacheKey);
  if (cached) {
    if (!cached.label) {
      const hashHex = FpHash.hash(jaxpr).toString(16).slice(0, 4);
      cached.label = `${label || "program"}#${hashHex}`;
    }
    return cached;
  }

  // Save/restore dim bindings for nested jitCompile calls (e.g., grad(foriLoop)
  // creates Scan body that contains ForiLoop, each triggering jitCompile).
  const prevDimBindings = _currentDimBindings;
  const prevSkipFori = _skipForiRewrite;
  _currentDimBindings = dimBindings;

  try {
    const _jitT0 = performance.now();
    if (DEBUG >= 1) {
      console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
    }

    jaxpr = jaxpr.flatten().simplify();
    // Rewrite eligible Conv equations to BlockMap before dataflow analysis.
    // This ensures splitGraphDataflow treats them as special black primitives,
    // materializing their inputs properly.
    lastConvRewritten = false;
    jaxpr = rewriteConvToBlockMap(jaxpr);
    // Rewrite eligible ForiLoop equations to BlockMap on WebGPU.
    // Must run after flatten/simplify so body shapes are concrete.
    if (!_skipForiRewrite) lastForiRewritten = false;
    jaxpr = rewriteForiLoopToBlockMap(jaxpr, backend);
    // Prevent rewrite in nested body compilations (scan, assocScan, blockMap
    // bodies). Block_map steps are rejected by both command tape and
    // mega-module, so rewriting foriLoops inside these bodies would force
    // fallback paths — a performance regression, not improvement.
    _skipForiRewrite = true;
    const nargs = jaxpr.inBinders.length;
    const builder = new JitProgramBuilder(backend, nargs);

    const blackNodes = splitGraphDataflow(backend, jaxpr);

    /** Set concreteSizeHint on a kernel if dimBindings are available. */
    const setConcreteHint = (kernel: Kernel, size: SizeExpr): void => {
      if (_currentDimBindings && isSymbolicSize(size)) {
        kernel.concreteSizeHint = resolveSizeExpr(
          size,
          _currentDimBindings,
        ) as number;
      }
      // Also set concreteHint on any symbolic reductions.
      if (_currentDimBindings) {
        for (const o of kernel.outputs) {
          if (o.reduction && isSymbolicSize(o.reduction.size)) {
            o.reduction.concreteHint = resolveSizeExpr(
              o.reduction.size,
              _currentDimBindings,
            ) as number;
          }
        }
      }
    };

    // Initialize jaxpr inBinders.
    const ctx = new Map<Var, JitValue>();
    for (let i = 0; i < nargs; i++) {
      const v = jaxpr.inBinders[i];
      ctx.set(v, { type: "imm", arg: i }); // JitId i = input #i
    }

    // ---- Pending kernel batching ----
    // Defer black-node kernel dispatch to batch same-size non-reduction kernels
    // into multi-output Kernel steps, reducing dispatch overhead.
    let kernelOrdinal = 0;
    /** Derive a semantic label for a kernel from its root AluOp(s). */
    const labelKernel = (kernel: Kernel): void => {
      const idx = kernelOrdinal++;
      if (kernel.outputs.length === 1) {
        const o = kernel.outputs[0];
        const opName = o.exp.op.toLowerCase();
        if (o.reduction) {
          kernel.label = `k${idx}_reduce_${o.reduction.op.toLowerCase()}`;
        } else {
          kernel.label = `k${idx}_${opName}`;
        }
      } else {
        const ops = kernel.outputs.map((o) => o.exp.op.toLowerCase());
        kernel.label = `k${idx}_multi(${ops.join(",")})`;
      }
    };
    interface PendingKernelEntry {
      outVar: Var;
      exp: AluExp;
      reduction: Reduction | undefined;
      inputArgs: JitId[];
      size: SizeExpr;
    }
    const pendingKernels: PendingKernelEntry[] = [];

    /** Flush all pending kernels into JitSteps. Groups same-size non-reduction
     *  kernels with identical inputArgs into multi-output Kernel steps. */
    const flushPendingKernels = (): void => {
      if (pendingKernels.length === 0) return;

      // If backend doesn't support multi-output, emit all as solo
      if (!backend.capabilities.multiOutputKernel) {
        for (const entry of pendingKernels) {
          const kernel = Kernel.single(
            entry.inputArgs.length,
            entry.size,
            entry.exp,
            entry.reduction,
          );
          labelKernel(kernel);
          setConcreteHint(kernel, entry.size);
          const outId = builder.pushKernel(kernel, entry.inputArgs);
          ctx.set(entry.outVar, { type: "imm", arg: outId });
        }
        pendingKernels.length = 0;
        return;
      }

      // Group by size + inputArgs identity (sorted JitIds as key).
      // Only non-reduction kernels with matching size AND inputArgs can merge.
      const groups = new Map<string, PendingKernelEntry[]>();
      const soloEntries: PendingKernelEntry[] = [];

      for (const entry of pendingKernels) {
        if (entry.reduction) {
          // Kernels with reductions always emit solo
          soloEntries.push(entry);
        } else {
          const key = `${sizeExprKey(entry.size)}:${entry.inputArgs.join(",")}`;
          const group = groups.get(key);
          if (group) group.push(entry);
          else groups.set(key, [entry]);
        }
      }

      // Emit multi-output kernels for groups with > 1 entries
      for (const [, group] of groups) {
        if (group.length === 1) {
          soloEntries.push(group[0]);
        } else {
          // Check maxArgs: nargs + numOutputs <= backend limit
          const maxArgs = backend.maxArgs;
          if (group[0].inputArgs.length + group.length > maxArgs) {
            // Too many buffers — fall back to individual kernels
            for (const e of group) soloEntries.push(e);
          } else {
            const nargs = group[0].inputArgs.length;
            const size = group[0].size;
            const outputDescs = group.map((e) => ({
              exp: e.exp,
              reduction: e.reduction,
            }));
            const kernel = Kernel.multi(nargs, size, outputDescs);
            labelKernel(kernel);
            setConcreteHint(kernel, size);
            const outIds = builder.pushMultiKernel(kernel, group[0].inputArgs);
            for (let j = 0; j < group.length; j++) {
              ctx.set(group[j].outVar, { type: "imm", arg: outIds[j] });
            }
          }
        }
      }

      // Emit solo kernels
      for (const entry of soloEntries) {
        const kernel = Kernel.single(
          entry.inputArgs.length,
          entry.size,
          entry.exp,
          entry.reduction,
        );
        labelKernel(kernel);
        setConcreteHint(kernel, entry.size);
        const outId = builder.pushKernel(kernel, entry.inputArgs);
        ctx.set(entry.outVar, { type: "imm", arg: outId });
      }

      pendingKernels.length = 0;
    };

    // Now run each primitive through a set of rules, mirroring implRules.
    for (let i = 0; i < jaxpr.eqns.length; i++) {
      const eqn = jaxpr.eqns[i];

      // Handle Primitive.Scan specially — it compiles the body jaxpr and
      // emits a single "scan" JitStep with a ScanPlan.
      if (eqn.primitive === Primitive.Scan) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<typeof Primitive.Scan>;
        const {
          jaxpr: bodyJaxpr,
          numCarry,
          numConsts,
          length,
          reverse,
          acceptPath,
        } = params;
        const numX = bodyJaxpr.inBinders.length - numConsts - numCarry;
        const numY = bodyJaxpr.outs.length - numCarry;

        // Resolve input JitIds (all must be "imm" — black nodes)
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: scan primitive input is not imm");
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const initCarryIds = allInputIds.slice(numConsts, numConsts + numCarry);
        const xsIds = allInputIds.slice(numConsts + numCarry);

        // xs avals (actual shapes from the jaxpr, include leading length dim)
        const xsAvals: ShapedArray[] = [];
        const xsInputs = eqn.inputs.slice(numConsts + numCarry);
        for (const input of xsInputs) {
          xsAvals.push(input.aval);
        }

        // Allocate output buffers: [carry_out..., stacked_ys...]
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile body jaxpr
        const bodyProgram = jitCompile(
          backend,
          bodyJaxpr,
          _currentDimBindings,
          "scan_body",
        );

        // Determine scan plan
        const scanPlan = planScan(
          backend,
          bodyProgram,
          bodyJaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          acceptPath as ScanPath | ScanPath[] | undefined,
          _currentDimBindings,
        );

        // Compute per-slice xsAvals (without leading length dimension)
        const xsSliceAvals = xsAvals.map(
          (a) => new ShapedArray(a.shape.slice(1), a.dtype, a.weakType),
        );

        builder.steps.push({
          type: "scan",
          plan: scanPlan,
          bodyProgram,
          bodyJaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          consts: constsIds,
          initCarry: initCarryIds,
          xs: xsIds,
          xsAvals: xsSliceAvals,
          outputs: outputIds,
        });
        continue;
      }

      // DynamicUpdateSlice: compile to a zero-copy DUS step.
      // The Mutate effect on dst allows effectDrivenAllocate to recycle
      // dst → output, avoiding the full buffer copy.
      if (eqn.primitive === Primitive.DynamicUpdateSlice) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.DynamicUpdateSlice
        >;
        const { offset, axis } = params;

        // Resolve input JitIds
        const dstInput = eqn.inputs[0];
        const srcInput = eqn.inputs[1];
        let dstId: JitId;
        let srcId: JitId;

        if (dstInput instanceof Var) {
          const jv = ctx.get(dstInput)!;
          if (jv.type !== "imm") {
            throw new Error("jit: DUS dst input is not imm");
          }
          dstId = jv.arg;
        } else {
          dstId = builder.pushLit(dstInput as Lit);
        }

        if (srcInput instanceof Var) {
          const jv = ctx.get(srcInput)!;
          if (jv.type !== "imm") {
            throw new Error("jit: DUS src input is not imm");
          }
          srcId = jv.arg;
        } else {
          srcId = builder.pushLit(srcInput as Lit);
        }

        const outVar = eqn.outBinders[0];
        const elemBytes = byteWidth(outVar.aval.dtype);
        const innerSize = (outVar.aval.shape as number[])
          .slice(axis + 1)
          .reduce((a, b) => a * b, 1);
        const offsetBytes = offset * innerSize * elemBytes;
        const sliceBytes = sizeExprMul(srcInput.aval.sizeExpr, elemBytes);
        const dstSizeBytes = sizeExprMul(outVar.aval.sizeExpr, elemBytes);

        // Fiber loop data for axis > 0 (non-contiguous slices)
        const outerFibers =
          axis === 0
            ? 1
            : (srcInput.aval.shape as number[])
                .slice(0, axis)
                .reduce((a, b) => a * b, 1);
        const srcFiberBytes =
          axis === 0
            ? 0
            : (srcInput.aval.shape[axis] as number) * innerSize * elemBytes;
        const dstFiberBytes =
          axis === 0
            ? 0
            : (outVar.aval.shape[axis] as number) * innerSize * elemBytes;

        // Allocate output buffer (same size as dst — recycling may reclaim dst)
        const outId = builder.pushBuffer(dstSizeBytes);
        ctx.set(outVar, { type: "imm", arg: outId });

        builder.steps.push({
          type: "dus",
          dst: dstId,
          src: srcId,
          output: outId,
          offsetBytes,
          sliceBytes,
          dstSizeBytes,
          outerFibers,
          srcFiberBytes,
          dstFiberBytes,
        });
        continue;
      }

      // DynamicUpdateSliceGeneral: emit a full dst→output copy, then a single ND-aware
      // DUS step that patches the src region using ndOuterExtents/ndDstStrides for
      // non-constant-stride coordinate decomposition.
      if (eqn.primitive === Primitive.DynamicUpdateSliceGeneral) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.DynamicUpdateSliceGeneral
        >;
        const { startIndices } = params;

        // Resolve src input
        const srcInput = eqn.inputs[1];
        let srcId: JitId;
        if (srcInput instanceof Var) {
          const jv = ctx.get(srcInput)!;
          if (jv.type !== "imm")
            throw new Error("jit: DUS_General src input is not imm");
          srcId = jv.arg;
        } else {
          srcId = builder.pushLit(srcInput as Lit);
        }

        // Resolve dst input
        const dstInput = eqn.inputs[0];
        let currentDstId: JitId;
        if (dstInput instanceof Var) {
          const jv = ctx.get(dstInput)!;
          if (jv.type !== "imm")
            throw new Error("jit: DUS_General dst input is not imm");
          currentDstId = jv.arg;
        } else {
          currentDstId = builder.pushLit(dstInput as Lit);
        }

        const outVar = eqn.outBinders[0];
        const outShape = outVar.aval.shape as number[];
        const srcShape = srcInput.aval.shape as number[];
        const rank = outShape.length;
        const elemBytes = byteWidth(outVar.aval.dtype);

        // Find axes that actually need updates (offset > 0 or src smaller than dst)
        const activeAxes: number[] = [];
        for (let ax = 0; ax < rank; ax++) {
          if (startIndices[ax] !== 0 || srcShape[ax] !== outShape[ax]) {
            activeAxes.push(ax);
          }
        }

        if (activeAxes.length === 0) {
          // src == dst shape and all offsets 0: just a full copy.
          // Emit a single axis=0 DUS with offset 0.
          const dstSizeBytes = sizeExprMul(outVar.aval.sizeExpr, elemBytes);
          const sliceBytes = dstSizeBytes;
          const outId = builder.pushBuffer(dstSizeBytes);
          ctx.set(outVar, { type: "imm", arg: outId });
          builder.steps.push({
            type: "dus",
            dst: currentDstId,
            src: srcId,
            output: outId,
            offsetBytes: 0,
            sliceBytes,
            dstSizeBytes,
            outerFibers: 1,
            srcFiberBytes: 0,
            dstFiberBytes: 0,
          });
        } else if (activeAxes.length === 1) {
          // Single active axis: emit one DUS step directly (fast path).
          const axis = activeAxes[0];
          const offset = startIndices[axis];
          const innerSize = outShape.slice(axis + 1).reduce((a, b) => a * b, 1);
          const offsetBytes = offset * innerSize * elemBytes;
          const sliceBytes = sizeExprMul(srcInput.aval.sizeExpr, elemBytes);
          const dstSizeBytes = sizeExprMul(outVar.aval.sizeExpr, elemBytes);
          const outerFibers =
            axis === 0 ? 1 : srcShape.slice(0, axis).reduce((a, b) => a * b, 1);
          const srcFiberBytes =
            axis === 0 ? 0 : srcShape[axis] * innerSize * elemBytes;
          const dstFiberBytes =
            axis === 0 ? 0 : outShape[axis] * innerSize * elemBytes;

          const outId = builder.pushBuffer(dstSizeBytes);
          ctx.set(outVar, { type: "imm", arg: outId });
          builder.steps.push({
            type: "dus",
            dst: currentDstId,
            src: srcId,
            output: outId,
            offsetBytes,
            sliceBytes,
            dstSizeBytes,
            outerFibers,
            srcFiberBytes,
            dstFiberBytes,
          });
        } else {
          // Multi-axis ND update: copy dst → output, then patch src into output
          // using a single DUS step with ND stride info.
          const dstSizeBytes = sizeExprMul(outVar.aval.sizeExpr, elemBytes);
          const outId = builder.pushBuffer(dstSizeBytes);
          ctx.set(outVar, { type: "imm", arg: outId });

          // Step 1: copy entire dst → output (full buffer copy)
          builder.steps.push({
            type: "dus",
            dst: currentDstId,
            src: currentDstId,
            output: outId,
            offsetBytes: 0,
            sliceBytes: dstSizeBytes,
            dstSizeBytes,
            outerFibers: 1,
            srcFiberBytes: 0,
            dstFiberBytes: 0,
          });

          // Step 2: patch src into output.
          // Precompute dst byte strides (row-major)
          const dstByteStrides: number[] = [];
          {
            dstByteStrides.length = rank;
            dstByteStrides[rank - 1] = elemBytes;
            for (let k = rank - 2; k >= 0; k--) {
              dstByteStrides[k] = dstByteStrides[k + 1] * outShape[k + 1];
            }
          }

          // Compute base offset in dst from startIndices
          let ndOffsetBytes = 0;
          for (let ax = 0; ax < rank; ax++) {
            ndOffsetBytes += startIndices[ax] * dstByteStrides[ax];
          }

          const totalFibers = srcShape
            .slice(0, rank - 1)
            .reduce((a, b) => a * b, 1);
          const rowBytes = srcShape[rank - 1] * elemBytes;

          // Check if all outer axes match (constant stride fast path)
          let singleFiberOk = true;
          for (let k = 0; k < rank - 1; k++) {
            if (srcShape[k] !== outShape[k]) {
              singleFiberOk = false;
              break;
            }
          }

          if (singleFiberOk) {
            // Constant dst stride — use standard fiber loop
            builder.steps.push({
              type: "dus",
              dst: outId,
              src: srcId,
              output: outId,
              offsetBytes: ndOffsetBytes,
              sliceBytes: sizeExprMul(srcInput.aval.sizeExpr, elemBytes),
              dstSizeBytes,
              outerFibers: totalFibers,
              srcFiberBytes: rowBytes,
              dstFiberBytes: outShape[rank - 1] * elemBytes,
            });
          } else {
            // Non-constant dst stride — single step with ND stride info.
            // The executor decomposes fiber indices into coordinates and
            // computes per-fiber dst offsets from ndDstStrides.
            builder.steps.push({
              type: "dus",
              dst: outId,
              src: srcId,
              output: outId,
              offsetBytes: ndOffsetBytes,
              sliceBytes: sizeExprMul(srcInput.aval.sizeExpr, elemBytes),
              dstSizeBytes,
              outerFibers: totalFibers,
              srcFiberBytes: rowBytes,
              dstFiberBytes: 0, // unused when ndOuterExtents present
              ndOuterExtents: srcShape.slice(0, rank - 1),
              ndDstStrides: dstByteStrides.slice(0, rank - 1),
            });
          }
        }
        continue;
      }

      // Handle Primitive.AssociativeScan — creates an assoc_scan JitStep
      // that runs the Kogge-Stone loop at execution time.
      if (eqn.primitive === Primitive.AssociativeScan) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.AssociativeScan
        >;
        const { jaxpr: bodyJaxpr, numLeaves, axis, reverse } = params;
        const numConsts = eqn.inputs.length - numLeaves;

        // Resolve input JitIds
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error(
                "jit: AssociativeScan primitive input is not imm",
              );
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const elemsIds = allInputIds.slice(numConsts);

        const constAvals = eqn.inputs
          .slice(0, numConsts)
          .map((v) => v.aval as ShapedArray);
        const elemAvals = eqn.inputs
          .slice(numConsts)
          .map((v) => v.aval as ShapedArray);

        // Allocate output buffers
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile the body jaxpr → JitProgram (for fallback path)
        const bodyProgram = jitCompile(
          backend,
          bodyJaxpr,
          _currentDimBindings,
          "assoc_scan_body",
        );

        // Plan the assoc scan — try compiled-loop (WASM), fallback otherwise
        const assocPlan = planAssociativeScan(
          backend,
          bodyProgram,
          bodyJaxpr,
          numLeaves,
          numConsts,
          axis,
          reverse,
          _currentDimBindings,
        );

        builder.steps.push({
          type: "assoc_scan",
          plan: assocPlan,
          bodyProgram,
          bodyJaxpr,
          numLeaves,
          numConsts,
          axis,
          reverse,
          consts: constsIds,
          elems: elemsIds,
          constAvals,
          elemAvals,
          outputs: outputIds,
        });
        continue;
      }

      // ScatterAdd: compile to a scatter_add JitStep.
      // The Mutate effect on target allows effectDrivenAllocate to recycle
      // target → output, avoiding a full buffer copy when sizes match.
      if (eqn.primitive === Primitive.ScatterAdd) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.ScatterAdd
        >;
        const { axis } = params;

        // Resolve input JitIds: target, indices, updates
        const resolveInput = (input: Var | Lit): JitId => {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: ScatterAdd input is not imm");
            }
            return jv.arg;
          }
          return builder.pushLit(input as Lit);
        };

        const targetId = resolveInput(eqn.inputs[0]);
        const indicesId = resolveInput(eqn.inputs[1]);
        const updatesId = resolveInput(eqn.inputs[2]);

        const outVar = eqn.outBinders[0];
        const targetShape = eqn.inputs[0].aval.shape as number[];
        const updatesLen = eqn.inputs[2].aval.shape[axis] as number;
        const dtype = outVar.aval.dtype;
        const outSizeBytes = sizeExprMul(
          outVar.aval.sizeExpr,
          byteWidth(dtype),
        );

        // Allocate output buffer (same size as target — recycling may reclaim target)
        const outId = builder.pushBuffer(outSizeBytes);
        ctx.set(outVar, { type: "imm", arg: outId });

        builder.steps.push({
          type: "scatter_add",
          target: targetId,
          indices: indicesId,
          updates: updatesId,
          output: outId,
          axis,
          targetShape,
          updatesLen,
          dtype,
        });
        continue;
      }

      // Handle Primitive.Reverse — compile to a reverse JitStep.
      // Unlike Flip (which uses reshapeJit/ShapeTracker), Reverse materializes
      // the reversal at execution time, supporting polymorphic lengths.
      if (eqn.primitive === Primitive.Reverse) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<typeof Primitive.Reverse>;
        const { axis } = params;

        // Resolve input JitId
        const input = eqn.inputs[0];
        let inputId: JitId;
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type !== "imm") {
            throw new Error("jit: Reverse input is not imm");
          }
          inputId = jv.arg;
        } else {
          inputId = builder.pushLit(input as Lit);
        }

        const outVar = eqn.outBinders[0];
        const dtype = outVar.aval.dtype;
        const shape = input.aval.shape;
        const axisSize = shape[axis]; // may be Dim (symbolic)
        // innerBytes: product of all dimensions after the reversed axis × byteWidth
        const trailingShape = (shape as Dim[]).slice(axis + 1);
        const innerBytes =
          (trailingShape.length > 0
            ? (trailingShape as number[]).reduce((a, b) => a * b, 1)
            : 1) * byteWidth(dtype);
        const totalBytes = sizeExprMul(outVar.aval.sizeExpr, byteWidth(dtype));

        const outId = builder.pushBuffer(totalBytes);
        ctx.set(outVar, { type: "imm", arg: outId });

        builder.steps.push({
          type: "reverse",
          input: inputId,
          output: outId,
          axis,
          axisSize,
          innerBytes,
          totalBytes,
          dtype,
        });
        continue;
      }

      // Handle Primitive.BlockMap — compile the body jaxpr and emit a
      // "block_map" JitStep that iterates over the grid of blocks.
      if (eqn.primitive === Primitive.BlockMap) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<typeof Primitive.BlockMap>;
        const {
          jaxpr: bodyJaxpr,
          blockShape,
          inAxes,
          outAxes,
          numConsts,
          numInputs,
          threadTile,
          gridShape: explicitGridShape,
        } = params;

        // Rewrite body jaxpr: replace BlockIndex equations with extra input.
        const { jaxpr: compiledBodyJaxpr, hasBlockIndex } =
          rewriteBlockIndex(bodyJaxpr);

        // Resolve input JitIds
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: BlockMap primitive input is not imm");
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const inputIds = allInputIds.slice(numConsts, numConsts + numInputs);

        // Store raw (possibly symbolic) input/output shapes.
        // gridShape is computed at execution time from resolved shapes.
        const inputAvals = eqn.inputs
          .slice(numConsts, numConsts + numInputs)
          .map((v) => v.aval);

        // Allocate output buffers
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile body jaxpr (rewritten if BlockIndex was present).
        // Save/restore lastConvClass and lastConvRewritten so body compilation
        // (which may contain a Conv equation e.g. from conv→block_map rewriting)
        // doesn't overwrite the outer conv classification/rewrite flag.
        const savedConvClass = lastConvClass;
        const savedConvRewritten: boolean = lastConvRewritten;
        const bodyProgram = jitCompile(
          backend,
          compiledBodyJaxpr,
          _currentDimBindings,
          "block_map_body",
        );
        lastConvClass = savedConvClass;
        lastConvRewritten = savedConvRewritten;

        const inputShapes: Dim[][] = inputAvals.map((a) => [...a.shape]);
        const outputShapes: Dim[][] = eqn.outBinders.map((v) => [
          ...v.aval.shape,
        ]);

        builder.steps.push({
          type: "block_map",
          bodyProgram,
          bodyJaxpr: compiledBodyJaxpr,
          blockShape,
          inAxes,
          outAxes,
          numConsts,
          numInputs,
          inputShapes,
          outputShapes,
          consts: constsIds,
          inputs: inputIds,
          outputs: outputIds,
          threadTile,
          explicitGridShape,
          halo: params.halo,
          hasBlockIndex,
        });
        continue;
      }

      // Handle Primitive.WorkgroupAssociativeScan — compile the scan body
      // and emit a "workgroup_assoc_scan" JitStep. Inside a block_map body,
      // the fused shader compiler emits inlined Kogge-Stone WGSL. In the
      // fallback path, the executor runs the body program sequentially.
      if (eqn.primitive === Primitive.WorkgroupAssociativeScan) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.WorkgroupAssociativeScan
        >;
        const { jaxpr: bodyJaxpr, numConsts } = params;
        const numElems = eqn.inputs.length - numConsts;

        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error(
                "jit: WorkgroupAssociativeScan primitive input is not imm",
              );
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const elemsIds = allInputIds.slice(numConsts);
        const elemAvals = eqn.inputs
          .slice(numConsts)
          .map((v) => v.aval as ShapedArray);

        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        const bodyProgram = jitCompile(
          backend,
          bodyJaxpr,
          _currentDimBindings,
          "workgroup_assoc_scan_body",
        );

        builder.steps.push({
          type: "workgroup_assoc_scan",
          bodyProgram,
          bodyJaxpr,
          numConsts,
          numElems,
          consts: constsIds,
          elems: elemsIds,
          outputs: outputIds,
          elemAvals,
        });
        continue;
      }

      // Handle Primitive.ForiLoop — compile the body jaxpr and emit a
      // "fori_loop" JitStep that iterates from lower to upper.
      if (eqn.primitive === Primitive.ForiLoop) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<typeof Primitive.ForiLoop>;
        const { jaxpr: bodyJaxpr, numConsts, lower, upper } = params;

        // Resolve input JitIds: [consts..., initCarries...]
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: ForiLoop primitive input is not imm");
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const initCarryIds = allInputIds.slice(numConsts);

        // Allocate output buffers (same shape/dtype as carries)
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile body jaxpr
        const bodyProgram = jitCompile(
          backend,
          bodyJaxpr,
          _currentDimBindings,
          "fori_loop_body",
        );

        const carrySizeBytes = eqn.outBinders.map(
          (v) => (v.aval.size as number) * byteWidth(v.aval.dtype),
        );

        builder.steps.push({
          type: "fori_loop",
          bodyProgram,
          bodyJaxpr,
          lower,
          upper,
          numConsts,
          consts: constsIds,
          initCarries: initCarryIds,
          outputs: outputIds,
          carrySizeBytes,
        });
        continue;
      }

      // If this is a routine, construct and dispatch the routine.
      if (routinePrimitives.has(eqn.primitive)) {
        flushPendingKernels();
        // The rest of the code collaborates to make sure that all inputs to a
        // routine are "imm" (black node, dispatched) and so is itself.
        const routine = new Routine(
          routinePrimitives.get(eqn.primitive)!,
          {
            inputShapes: eqn.inputs.map((x) => x.aval.shape as number[]),
            inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
            outputShapes: eqn.outBinders.map((x) => x.aval.shape as number[]),
            outputDtypes: eqn.outBinders.map((x) => x.aval.dtype),
          },
          eqn.params as any,
        );
        const inputs: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error(
                `jit: routine primitive ${eqn.primitive} input is not imm`,
              );
            }
            inputs.push(jv.arg);
          } else if (input instanceof Lit) {
            inputs.push(builder.pushLit(input));
          }
        }
        const outputs: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputs.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }
        builder.pushRoutine(routine, inputs, outputs);
        continue;
      }

      // If any input references a pending kernel's output, flush first so the
      // output is materialized and available in ctx.
      if (pendingKernels.length > 0) {
        for (const input of eqn.inputs) {
          if (
            input instanceof Var &&
            !ctx.has(input) &&
            pendingKernels.some((pk) => pk.outVar === input)
          ) {
            flushPendingKernels();
            break;
          }
        }
      }

      // Transform each input into an AluExp to start, and normalize any arguments
      // as needed.
      const inputExps: AluExp[] = []; // len(inputs)
      const inputAvals: ShapedArray[] = []; // len(inputs)
      const inputArgs: JitId[] = [];

      let inputReduction: (JitValue & { type: "red" }) | null = null;

      // May need to reindex gids to match order, returns array of new gids.
      const addArgs = (args: JitId[]): number[] => {
        const newGids: number[] = [];
        for (const jitId of args) {
          let newGid = inputArgs.indexOf(jitId);
          if (newGid === -1) {
            newGid = inputArgs.length;
            inputArgs.push(jitId);
          }
          newGids.push(newGid);
        }
        return newGids;
      };

      for (const input of eqn.inputs) {
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type === "exp") {
            const newGids = addArgs(jv.args);
            inputExps.push(jv.exp.reindexGids(newGids));
          } else if (jv.type === "imm") {
            const [gid] = addArgs([jv.arg]);
            // For symbolic shapes, resolve to concrete using dimBindings for
            // ShapeTracker operations. The kernel.size stays symbolic (SizeExpr)
            // and is resolved at execution time via dynamicParams.
            const shape =
              hasSymbolicDims(input.aval.shape) && _currentDimBindings
                ? (resolveShape(
                    input.aval.shape,
                    _currentDimBindings,
                  ) as number[])
                : (input.aval.shape as number[]);
            const st = ShapeTracker.fromShape(shape);
            const indices = unravelAlu(st.shape, AluVar.gidx);
            inputExps.push(
              AluExp.globalView(input.aval.dtype, gid, st, indices),
            );
          } else if (jv.type === "red") {
            // Special case: We are consuming a 'red' JitValue, so we must be in the
            // fused epilogue of a reduction.
            if (inputReduction)
              throw new Error("jit: unexpected, multiple red inputs");
            const newGids = addArgs(jv.args);
            inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
            inputReduction = jv;
          } else {
            jv satisfies never; // static check
          }
          inputAvals.push(input.aval);
        } else if (input instanceof Lit) {
          inputExps.push(AluExp.const(input.dtype, input.value));
          inputAvals.push(input.aval);
        } else {
          throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
        }
      }

      // Produce a new expression and/or reduction for the operation based on the
      // jit() implementation of the primitive.
      const rule = jitRules[eqn.primitive];
      if (!rule)
        throw new TypeError(
          `JIT not implemented for primitive ${eqn.primitive}`,
        );

      let exp: AluExp[];
      let reduction: Reduction | undefined;

      if (inputReduction) {
        // Special case: we are in the fused epilogue of a reduction.
        const jv = inputReduction;
        const newEpilogue = rule(inputExps, inputAvals, eqn.params as any)
          .exp[0];
        exp = [jv.exp.reindexGids(addArgs(jv.args))];
        reduction = new Reduction(
          jv.reduction.dtype,
          jv.reduction.op,
          jv.reduction.size,
          newEpilogue,
        );
      } else {
        const ruleOutput = rule(inputExps, inputAvals, eqn.params as any);
        exp = ruleOutput.exp;
        reduction = ruleOutput.reduction;
      }

      // Then dispatch the kernel, if it is a "black" node as determined from
      // dataflow analysis above. Defer to pending kernel batching for potential
      // multi-output fusion.
      for (let i = 0; i < eqn.outBinders.length; i++) {
        const outVar = eqn.outBinders[i];
        if (blackNodes.has(outVar)) {
          const size = outVar.aval.sizeExpr;
          pendingKernels.push({
            outVar,
            exp: exp[i],
            reduction,
            inputArgs: [...inputArgs],
            size,
          });
        } else if (reduction) {
          // Reduction but not black, means it will have an epilogue.
          ctx.set(outVar, {
            type: "red",
            exp: exp[i],
            reduction,
            args: inputArgs,
          });
        } else {
          // Otherwise, fuse the kernel into the next expression.
          ctx.set(outVar, { type: "exp", exp: exp[i], args: inputArgs });
        }
      }
    }

    // Flush any remaining pending kernels before output collection.
    flushPendingKernels();

    // Finally, loop through the outputs.
    const outputIds: JitId[] = [];
    for (const out of jaxpr.outs) {
      if (out instanceof Var) {
        const jitValue = ctx.get(out)!;
        if (jitValue.type !== "imm")
          throw new Error("internal: Expected imm, since outs are black nodes");
        outputIds.push(jitValue.arg);
      } else if (out instanceof Lit) {
        outputIds.push(builder.pushLit(out));
      } else {
        out satisfies never; // static check
      }
    }

    // Each output should have its own backend reference. If an output slot is
    // returned twice, or if an input is returned directly, insert "incref" steps
    // to balance the books.
    const outputNeedsRef = new Set<JitId>(range(nargs)); // inputs
    for (const outputId of outputIds) {
      if (outputNeedsRef.has(outputId)) {
        builder.pushIncref(outputId);
      } else {
        // If this output is seen again, increment its ref at that point.
        outputNeedsRef.add(outputId);
      }
    }

    // Effect-driven buffer allocation: compute liveness, insert free steps,
    // and recycle same-size buffers across execute/scan boundaries.
    builder.effectDrivenAllocate(outputIds);

    const jp = new JitProgram(
      backend,
      builder.steps,
      range(0, nargs),
      outputIds,
      builder.slotCount,
    );
    // Derive program label: "<name>#<hash4>" where name comes from the
    // caller (e.g. jit(fn) passes fn.name) and hash4 is a 4-char hex prefix
    // of the jaxpr hash for disambiguation.
    const hashHex = FpHash.hash(jaxpr).toString(16).slice(0, 4);
    jp.label = `${label || "program"}#${hashHex}`;

    if (DEBUG >= 4) console.info(jp.toString());
    if (DEBUG >= 1)
      console.info(
        `[jitCompile] ${(performance.now() - _jitT0).toFixed(1)}ms, ${builder.steps.length} steps, ${jaxpr.eqns.length} eqns`,
      );
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: backend.type,
        kind: "program",
        label: jp.label,
        code: jp.toString(),
        metadata: {
          numSteps: jp.steps.length,
          numInputs: jp.inputs.length,
          numOutputs: jp.outputs.length,
          ...jp.stepCounts(),
        },
      });
    }
    jitCompileCache.set(cacheKey, jp);
    return jp;
  } finally {
    _currentDimBindings = prevDimBindings;
    _skipForiRewrite = prevSkipFori;
  }
}

/**
 * Rule for fusing the operation into a JIT expression to the backend.
 *
 * This takes in the expressions of the `src[]` inputs and produces a subsequent
 * expression, as well as optionally a reduction. The expressions use
 * AluVar.gidx (output index) and AluVar.ridx (reduction index).
 */
type JitRule<P extends Primitive> = (
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => {
  exp: AluExp[]; // One expression for each output
  reduction?: Reduction;
};

function reshapeViews(
  exp: AluExp,
  mapping: (st: ShapeTracker) => ShapeTracker | undefined,
  reduceAxis: boolean = false,
): AluExp {
  return exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const [gid, st]: [number, ShapeTracker] = exp.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices = reduceAxis
          ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(
              AluVar.ridx,
            )
          : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    } else if (exp.op === AluOp.GlobalIndex) {
      throw new Error("internal: reshapeViews() called with GlobalIndex op");
    }
  });
}

// JIT handler for a broadcasted operation on at least 1 input.
function broadcastedJit<P extends Primitive>(
  fn: (exps: AluExp[], params: PrimitiveParams<P>) => AluExp,
  opts?: { skipCastIdx?: number[] },
): JitRule<P> {
  return (exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);

    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length) {
      // Skip casting some indices to a shared dtype.
      newDtype = avals
        .filter((_, i) => !skipCastIdx.includes(i))
        .reduce(promoteAvals).dtype;
    }

    // Perform a broadcast on each of the input expressions.
    //
    // Only GlobalView is affected. GlobalIndex is not used here, and neither is
    // AluVar.idx, since those are realized before jit().
    // For symbolic shapes, resolve to concrete using dimBindings for
    // ShapeTracker broadcast operations.
    const concreteNewShape =
      _currentDimBindings && hasSymbolicDims(newShape)
        ? (resolveShape(newShape, _currentDimBindings) as number[])
        : (newShape as number[]);
    exps = exps.map((exp, i) => {
      exp = reshapeViews(exp, (st) => {
        if (!deepEqual(st.shape, concreteNewShape))
          return st.broadcast(
            concreteNewShape,
            range(concreteNewShape.length - st.shape.length),
          );
      });
      if (exp.dtype !== newDtype && !skipCastIdx.includes(i)) {
        exp = AluExp.cast(newDtype, exp);
      }
      return exp;
    });

    // Then, we can call the function to produce a new expression.
    return { exp: [fn(exps, params)] };
  };
}

// Simpler JIT handler, equivalent to broadcastedJit for unary ops.
function unopJit<P extends Primitive>(
  fn: (exp: AluExp, params: PrimitiveParams<P>) => AluExp,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [fn(a, params)] };
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [reshapeViews(a, (st) => fn(st, params))] };
  };
}

function routineNoJit<P extends Primitive>(): JitRule<P> {
  return () => {
    throw new Error("jit: rule is not implemented for routines");
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Mod]: broadcastedJit(([a, b]) => AluExp.mod(a, b)),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.BitCombine]: broadcastedJit(([a, b], { op }) =>
    AluExp.bitCombine(a, b, op),
  ),
  [Primitive.BitShift]: broadcastedJit(([a, b], { op }) =>
    AluExp.bitShift(a, b, op),
  ),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.Floor]: unopJit(AluExp.floor),
  [Primitive.Ceil]: unopJit(AluExp.ceil),
  [Primitive.StopGradient]: unopJit((a) => a), // No-op, just return the input.
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Erf]: unopJit(AluExp.erf),
  [Primitive.Erfc]: unopJit(AluExp.erfc),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Reduce]([a], [as], { op, axis }) {
    // Resolve symbolic shapes to concrete for ShapeTracker operations.
    const shape =
      _currentDimBindings && hasSymbolicDims(as.shape)
        ? (resolveShape(as.shape, _currentDimBindings) as number[])
        : (as.shape as number[]);
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    const newShape: number[] = [];
    for (let i = 0; i < shape.length; i++) {
      if (axis.includes(i)) shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(shape[i]);
      }
    }
    const reductionSize = prod(shiftedAxes.map((ax) => shape[ax]));
    newShape.push(reductionSize);

    // Compute the reduction size as SizeExpr from the original (possibly
    // symbolic) shape so that the Reduction caches correctly under a
    // symbolic key and the dynamic loop bound is resolved at execute time.
    const reductionSizeExpr: SizeExpr = hasSymbolicDims(as.shape)
      ? dimProduct(shiftedAxes.map((ax) => as.shape[ax]))
      : reductionSize;

    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSizeExpr);
    return { exp: [a], reduction };
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) =>
    pool(st, window, strides),
  ),
  [Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(
      ShapeTracker.fromShape(as.shape as number[]),
      inShape,
      window,
      strides,
    );
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]); // Combine all reduce axes.
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(
      a.dtype,
      AluOp.Add,
      stX.shape[stX.shape.length - 1],
    );
    return { exp: [a], reduction };
  },
  [Primitive.Dot]([a, b], [as, bs]) {
    // Dot is just Mul->Reduce in sequence.
    const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
    const [c] = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce]([c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1],
    });
  },
  [Primitive.Conv]([a, b], [as, bs], params) {
    const v = (params as ConvParams).vmapDims;
    const kernelShape = (bs.shape as number[]).slice(v + 2);
    lastConvClass = classifyConv(params as ConvParams, kernelShape);
    // Eligible block-map-3x3/5x5 convs are rewritten to BlockMap equations
    // by rewriteConvToBlockMap() before reaching this rule. Any Conv that
    // reaches here uses the generic im2col→Dot lowering path.
    const prepare =
      lastConvClass === "fast-1x1-dot" ? prepareConv1x1 : prepareConv;
    const [stX, stY] = prepare(
      ShapeTracker.fromShape(as.shape as number[]),
      ShapeTracker.fromShape(bs.shape as number[]),
      params,
    );
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot]([a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(
    ([cond, a, b]) => AluExp.where(cond, a, b),
    { skipCastIdx: [0] },
  ),
  [Primitive.Concatenate](exps, avals, { axis }) {
    const ndim = avals[0].ndim;
    const sizes = avals.map((x) => x.shape[axis] as number);
    const finalSize = sizes.reduce((a, b) => a + b, 0);
    const { dtype: dtypeOut } = avals
      .map((x) => x.scalar())
      .reduce(promoteAvals);
    const makePadAxis = (start: number, end: number): Pair[] =>
      range(ndim).map((i) => (i === axis ? [start, end] : [0, 0]));
    let cum = 0;
    const src: AluExp[] = [];
    for (let i = 0; i < exps.length; i++) {
      const padding = makePadAxis(cum, finalSize - cum - (sizes[i] as number));
      src.push(
        reshapeViews(AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)),
      );
      cum += sizes[i] as number;
    }
    return { exp: [src.reduce(AluExp.add)] };
  },
  [Primitive.Split]([a], [as], { axis, sizes }) {
    const exp: AluExp[] = [];
    let start = 0;
    for (const size of sizes) {
      const slice = range(as.ndim).map<Pair>((d) =>
        d === axis ? [start, start + size] : [0, as.shape[d] as number],
      );
      exp.push(reshapeViews(a, (st) => st.shrink(slice)));
      start += size;
    }
    return { exp };
  },
  [Primitive.RandomBits]: (keys, keyShapes, { shape, mode }) => {
    const keyShape = keyShapes[0].shape;
    const mapping = (st: ShapeTracker): ShapeTracker | undefined => {
      if (!deepEqual(st.shape, shape))
        return st.broadcast(shape, range(st.shape.length, shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.mod(
      AluExp.cast(DType.Uint32, AluVar.gidx),
      // max(..., 1) to avoid mod-by-zero compile error in degenerate case
      AluExp.u32(Math.max(prod(shape.slice(keyShape.length)), 1)),
    );
    const exp = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return { exp: [exp] };
  },
  [Primitive.Gather](
    [x, ...indices],
    [xs, ...indicesShapes],
    { axis, outDim },
  ) {
    const axisSet = new Set(axis);

    // First, broadcast each integer array in `indices`.
    const indexShape = indicesShapes
      .map((c) => c.shape)
      .reduce(generalBroadcast) as number[];
    const finalShape = (xs.shape as number[]).filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const _idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < xs.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }

    for (const [i, iexp] of indices.entries()) {
      // Index iexp by the idxAxis variables, after broadcasting (via GlobalView).
      // [ ... | outDim | ... <iexp> | outDim + indexShape.length | ... ]
      src[axis[i]] = AluExp.cast(
        DType.Int32,
        reshapeViews(iexp, (st) =>
          st.broadcast(finalShape, [
            // Broadcast indices (aligned to the right), plus leading before outDim.
            ...range(outDim + indexShape.length - st.shape.length),
            // Indices to the right of outDim.
            ...range(outDim + indexShape.length, finalShape.length),
          ]),
        ),
      );
    }

    // Finally, index into "x" by replacing its gidx with a flat accessor into
    // the gathered indices.
    const [index, valid] = ShapeTracker.fromShape(
      xs.shape as number[],
    ).toAluExp(src);
    if (!valid.resolve())
      throw new Error("internal: expected full validity mask in Gather");
    return { exp: [x.substitute({ gidx: index })] };
  },
  [Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
  [Primitive.Broadcast]: reshapeJit((st, { shape, axis }) => {
    const concreteShape =
      _currentDimBindings && hasSymbolicDims(shape)
        ? resolveShape(shape, _currentDimBindings)
        : (shape as number[]);
    return st.broadcast(concreteShape, axis);
  }),
  [Primitive.Reshape]: reshapeJit((st, { shape }) => {
    const concreteShape =
      _currentDimBindings && hasSymbolicDims(shape)
        ? resolveShape(shape, _currentDimBindings)
        : (shape as number[]);
    return st.reshape(concreteShape);
  }),
  [Primitive.Flip]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis) arg[ax] = true;
    return st.flip(arg);
  }),
  [Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
  [Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
  [Primitive.Sort]: routineNoJit(),
  [Primitive.Argsort]: routineNoJit(),
  [Primitive.TriangularSolve]: routineNoJit(),
  [Primitive.Cholesky]: routineNoJit(),
  [Primitive.LU]: routineNoJit(),
  [Primitive.QR]: routineNoJit(),
  [Primitive.Jit]() {
    throw new Error(
      "internal: Jit should have been flattened before JIT compilation",
    );
  },
  [Primitive.DynamicUpdateSlice]() {
    throw new Error(
      "internal: DynamicUpdateSlice is handled specially in jitCompile",
    );
  },
  [Primitive.DynamicUpdateSliceGeneral]() {
    throw new Error(
      "internal: DynamicUpdateSliceGeneral is handled specially in jitCompile",
    );
  },
  [Primitive.Scan]() {
    throw new Error(
      "internal: Scan is handled specially in jitCompile, not via jitRules",
    );
  },
  [Primitive.ScatterAdd]() {
    throw new Error("internal: ScatterAdd is handled specially in jitCompile");
  },
  [Primitive.Reverse]() {
    throw new Error("internal: Reverse is handled specially in jitCompile");
  },
  [Primitive.AssociativeScan]() {
    throw new Error(
      "internal: AssociativeScan is handled specially in jitCompile",
    );
  },
  [Primitive.BlockMap]() {
    throw new Error("internal: BlockMap is handled specially in jitCompile");
  },
  [Primitive.BlockIndex]() {
    throw new Error(
      "internal: BlockIndex should be rewritten by rewriteBlockIndex before jitCompile",
    );
  },
  [Primitive.ForiLoop]() {
    throw new Error("internal: ForiLoop is handled specially in jitCompile");
  },
  [Primitive.WorkgroupAssociativeScan]() {
    throw new Error(
      "internal: WorkgroupAssociativeScan is handled specially in jitCompile",
    );
  },
  [Primitive.DynamicSlice](exps, avals, { sliceSizes }) {
    const operandExp = exps[0];
    const startExps = exps.slice(1);
    const operandShape = avals[0].shape as number[];
    const ndim = operandShape.length;

    // Build multi-dim output indices by unraveling gidx over sliceSizes
    const outIndices = unravelAlu(sliceSizes, AluVar.gidx);

    // Clamp each start index to [0, dim_k - sliceSize_k] and add output coord
    const readIndices: AluExp[] = [];
    for (let k = 0; k < ndim; k++) {
      const maxStart = (operandShape[k] as number) - sliceSizes[k];
      let start = AluExp.cast(DType.Int32, startExps[k]);
      start = AluExp.max(start, AluExp.i32(0));
      start = AluExp.min(start, AluExp.i32(maxStart));
      readIndices.push(AluExp.add(start, outIndices[k]));
    }

    // If the operand is a GlobalView, directly replace its indices to avoid
    // an unravel→ravel roundtrip that creates unsimplifiable expressions.
    if (operandExp.op === AluOp.GlobalView) {
      const [gid, st] = operandExp.arg as [number, ShapeTracker];
      return {
        exp: [AluExp.globalView(operandExp.dtype, gid, st, readIndices)],
      };
    }

    // Fallback: convert multi-dim read indices to flat index and substitute
    const [index] = ShapeTracker.fromShape(operandShape).toAluExp(readIndices);
    return { exp: [operandExp.substitute({ gidx: index })] };
  },
  [Primitive.UncheckedDynamicSlice](exps, avals, { sliceSizes }) {
    const operandExp = exps[0];
    const startExps = exps.slice(1);
    const operandShape = avals[0].shape as number[];
    const ndim = operandShape.length;

    const outIndices = unravelAlu(sliceSizes, AluVar.gidx);

    // No clamping — caller guarantees in-bounds
    const readIndices: AluExp[] = [];
    for (let k = 0; k < ndim; k++) {
      const start = AluExp.cast(DType.Int32, startExps[k]);
      readIndices.push(AluExp.add(start, outIndices[k]));
    }

    // If the operand is a GlobalView, directly replace its indices to avoid
    // an unravel→ravel roundtrip that creates unsimplifiable expressions.
    if (operandExp.op === AluOp.GlobalView) {
      const [gid, st] = operandExp.arg as [number, ShapeTracker];
      return {
        exp: [AluExp.globalView(operandExp.dtype, gid, st, readIndices)],
      };
    }

    // Fallback: convert multi-dim read indices to flat index and substitute
    const [index] = ShapeTracker.fromShape(operandShape).toAluExp(readIndices);
    return { exp: [operandExp.substitute({ gidx: index })] };
  },
};

/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
let _splitStats = {
  total: 0,
  reductions: 0,
  heterogeneous: 0,
  routines: 0,
  special: 0,
  cascade: 0,
  diamonds: 0,
  cheapDiamonds: 0,
  cleanShape: 0,
  p2: 0,
};
function splitGraphDataflow(backend: Backend, jaxpr: Jaxpr): Set<Var> {
  if (DEBUG >= 1) {
    _splitStats = {
      total: 0,
      reductions: 0,
      heterogeneous: 0,
      routines: 0,
      special: 0,
      cascade: 0,
      diamonds: 0,
      cheapDiamonds: 0,
      cleanShape: 0,
      p2: 0,
    };
  }
  const varToDefn = new Map<Var, number>(); // Var -> eqn index of definition
  const varToUsages: Map<Var, number[]> = new Map(); // Var -> eqn indices of usages
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) {
      if (v instanceof Var) varToDefn.set(v, i);
    }
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const usages = varToUsages.get(input);
        if (usages) usages.push(i);
        else varToUsages.set(input, [i]);
      }
    }
  }

  // Calculate reduction epilogues.
  //
  // A reduction can be fused with one or more operations that use its output,
  // which are either 1) unary or 2) binary ops with a literal, or an array not
  // larger than it.
  //
  // We also need to make sure we don't fuse two reductions together.
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose,
  ];
  const reductionEpilogueEqns = new Set<number>();
  const reductionEndpointEqns = new Set<number>();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive)) {
      let head = i;
      while (true) {
        reductionEpilogueEqns.add(head);

        // Try moving outVar forward through the graph.
        const outVar = jaxpr.eqns[head].outBinders[0];
        const usages = varToUsages.get(outVar) ?? [];
        if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;

        // Next is already fused into a reduction epilogue, can't fuse again.
        if (reductionEpilogueEqns.has(usages[0])) break;

        const nextEqn = jaxpr.eqns[usages[0]];
        switch (nextEqn.primitive) {
          // We can always fuse unary operations.
          case Primitive.Neg:
          case Primitive.Reciprocal:
          case Primitive.Floor:
          case Primitive.Ceil:
          case Primitive.StopGradient:
          case Primitive.Cast:
          case Primitive.Bitcast:
          case Primitive.Sin:
          case Primitive.Cos:
          case Primitive.Asin:
          case Primitive.Atan:
          case Primitive.Exp:
          case Primitive.Log:
          case Primitive.Erf:
          case Primitive.Erfc:
          case Primitive.Sqrt:
            head = usages[0];
            continue;

          // We can fuse binary operations with a literal, or with an array such
          // that the array doesn't lead to broadcasting thus recomputation.
          case Primitive.Add:
          case Primitive.Mul:
          case Primitive.Idiv:
          case Primitive.Mod:
          case Primitive.Min:
          case Primitive.Max:
          case Primitive.Compare:
          case Primitive.BitCombine:
          case Primitive.BitShift: {
            const otherInput = nextEqn.inputs.find((v) => v !== outVar)!;
            if (
              otherInput instanceof Lit ||
              deepEqual(
                generalBroadcast(otherInput.aval.shape, outVar.aval.shape),
                outVar.aval.shape,
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }

          // Ternary Where: fusable if all non-chain inputs are literals or
          // same-shape arrays (no broadcasting that enlarges the result).
          case Primitive.Where: {
            const otherInputs = nextEqn.inputs.filter((v) => v !== outVar);
            if (
              otherInputs.every(
                (inp) =>
                  inp instanceof Lit ||
                  deepEqual(
                    generalBroadcast(inp.aval.shape, outVar.aval.shape),
                    outVar.aval.shape,
                  ),
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }
        }
        break; // Can't move forward anymore.
      }
      reductionEndpointEqns.add(head);
    }
  }

  // Move backwards through the program and compute "black" endpoints.
  //
  // Black nodes are the endpoints where we dispatch a kernel to the backend
  // rather than producing intermediates. This includes:
  //
  // - Kernel outputs
  // - Reductions, except when fused with epilogue
  // - Gather/RandomBits operations (violates rule that kernels must have
  //   homogeneous GlobalView indices)
  // - Inputs to Pad operations, which need clean inputs
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes. View ops
  // (reshape, shrink, transpose, broadcast, flip, pad, pool, split) are exempt
  // from this diamond rule because they have zero compute cost so duplicating
  // them into multiple kernels is free.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Set<Var>>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, new Set([v]));
    }
  }
  const heterogeneousViewPrimitives = [
    // These primitives generate heterogeneous GlobalView outputs, there are
    // multiple views in the expression with different indexing.
    Primitive.RandomBits,
    Primitive.Gather,
    Primitive.DynamicSlice,
    Primitive.UncheckedDynamicSlice,
  ];
  const needsCleanShapePrimitives = [
    // Concatenate is based on Pad internally.
    Primitive.Concatenate,
    // If Pad is applied to a non-clean input, the reshaped padding would apply
    // to the view _inside_ of the expression. Imagine `GlobalView(...)+1`: if
    // you reshape each view, it adds zeros into the inner expression, so the
    // effect is to pad the intermediate with 1s instead of 0s!
    Primitive.Pad,
  ];
  const specialBlackPrimitives = [
    // These primitives are handled specially in jitCompile and must always
    // be black nodes with materialized inputs.
    Primitive.Scan,
    Primitive.DynamicUpdateSlice,
    Primitive.DynamicUpdateSliceGeneral,
    Primitive.ScatterAdd,
    Primitive.AssociativeScan,
    Primitive.Reverse,
    Primitive.BlockMap,
    Primitive.ForiLoop,
  ];
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      reductionEndpointEqns.has(i) ||
      heterogeneousViewPrimitives.includes(eqn.primitive) ||
      routinePrimitives.has(eqn.primitive) ||
      specialBlackPrimitives.includes(eqn.primitive) ||
      eqn.outBinders.some((v) => blackNodes.has(v))
    ) {
      if (DEBUG >= 1) {
        if (reductionEndpointEqns.has(i)) _splitStats.reductions++;
        else if (heterogeneousViewPrimitives.includes(eqn.primitive))
          _splitStats.heterogeneous++;
        else if (routinePrimitives.has(eqn.primitive)) _splitStats.routines++;
        else if (specialBlackPrimitives.includes(eqn.primitive))
          _splitStats.special++;
        else _splitStats.cascade++;
      }
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, new Set([v]));
      }
      continue;
    }
    const reach = new Set<Var>();
    let needsCleanOutput = false;
    outer: for (const v of eqn.outBinders) {
      for (const j of varToUsages.get(v) ?? []) {
        if (
          needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) ||
          routinePrimitives.has(jaxpr.eqns[j].primitive) ||
          specialBlackPrimitives.includes(jaxpr.eqns[j].primitive)
        ) {
          needsCleanOutput = true;
          break outer;
        }
        for (const o of jaxpr.eqns[j].outBinders) {
          const us = p1NextBlack.get(o);
          if (us) for (const u of us) reach.add(u);
        }
      }
    }
    // View ops: zero-cost index transforms that can be freely duplicated
    // into multiple kernels without any compute overhead.
    const isViewOp =
      eqn.primitive === Primitive.Reshape ||
      eqn.primitive === Primitive.Shrink ||
      eqn.primitive === Primitive.Transpose ||
      eqn.primitive === Primitive.Broadcast ||
      eqn.primitive === Primitive.Flip ||
      eqn.primitive === Primitive.Split;
    // Cheap recompute ops: elementwise operations whose recomputation cost
    // is negligible compared to the per-dispatch overhead (~100µs). Allowing
    // these through the diamond means they get duplicated into each
    // downstream kernel, but the extra ALU cost is minimal.
    // For binary ops, require at least one Lit input to avoid increasing
    // binding count in downstream kernels.
    let isCheapRecompute = false;
    switch (eqn.primitive) {
      case Primitive.Neg:
      case Primitive.Reciprocal:
      case Primitive.Floor:
      case Primitive.Ceil:
      case Primitive.StopGradient:
      case Primitive.Cast:
      case Primitive.Bitcast:
      case Primitive.Sin:
      case Primitive.Cos:
      case Primitive.Asin:
      case Primitive.Atan:
      case Primitive.Exp:
      case Primitive.Log:
      case Primitive.Erf:
      case Primitive.Erfc:
      case Primitive.Sqrt:
        isCheapRecompute = true;
        break;
      case Primitive.Add:
      case Primitive.Mul:
      case Primitive.Idiv:
      case Primitive.Mod:
      case Primitive.Min:
      case Primitive.Max:
      case Primitive.Compare:
        isCheapRecompute = eqn.inputs.some((v) => v instanceof Lit);
        break;
      case Primitive.Where:
        isCheapRecompute =
          eqn.inputs.filter((v) => v instanceof Lit).length >= 2;
        break;
    }
    if (
      needsCleanOutput ||
      (reach.size > 1 && !isViewOp && !isCheapRecompute)
    ) {
      if (DEBUG >= 1) {
        if (reach.size > 1) _splitStats.diamonds++;
        if (needsCleanOutput) _splitStats.cleanShape++;
      }
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, new Set([v]));
      }
    } else if (reach.size >= 1) {
      if (DEBUG >= 1 && reach.size > 1 && isCheapRecompute)
        _splitStats.cheapDiamonds++;
      // Propagate all reachable black nodes through this node. For view ops
      // and cheap recompute ops, this preserves the fan-out so upstream
      // nodes see the diamond correctly.
      for (const v of eqn.outBinders) p1NextBlack.set(v, new Set(reach));
    }
  }

  // Also, mark nodes black if the maximum number of arguments per kernel is
  // exceeded (i.e., maxComputeBuffersPerShaderStage for WebGPU). This needs to
  // be done in a second forward pass over the equations list.
  const p2Deps = new Map<Var, Set<Var>>(); // -> members are Var (black) or inBinders.
  for (const v of jaxpr.inBinders) {
    p2Deps.set(v, new Set([v])); // Each input is a dependency of itself.
  }
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    // Non-kernel black nodes (Scan, Routines) handle their own buffer bindings
    // internally and don't go through the WebGPU kernel compiler, so maxArgs
    // doesn't apply to them. Skip the dep check for these.
    //
    // NOTE: elementwise black nodes (e.g. Jaxpr outputs, multi-use nodes) DO
    // get compiled as WebGPU kernels and DO need the dep count check even when
    // all their outBinders are already in blackNodes. The former "skip all
    // all-black equations" early-continue was a bug: it allowed those kernel
    // endpoints to accumulate too many fused inputs without backtracking.
    const isNonKernelBlack =
      eqn.outBinders.every((v) => blackNodes.has(v)) &&
      (specialBlackPrimitives.includes(eqn.primitive) ||
        routinePrimitives.has(eqn.primitive));
    if (isNonKernelBlack) {
      for (const out of eqn.outBinders) p2Deps.set(out, new Set([out]));
      continue;
    }
    // For all-black kernel endpoints the output is already materialized, so
    // p2Deps should represent it as a single "base" dep rather than propagating
    // its transitive dep set to downstream equations.
    const isAllBlack = eqn.outBinders.every((v) => blackNodes.has(v));
    const deps: Set<Var>[] = [];
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        if (blackNodes.has(input)) deps.push(new Set([input]));
        else deps.push(p2Deps.get(input)!);
      } else {
        deps.push(new Set());
      }
    }
    const depCounter = new Map<Var, number>(); // includes counts
    for (const depSet of deps) {
      for (const dep of depSet) {
        depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
      }
    }
    if (depCounter.size > backend.maxArgs) {
      // We have too many dependencies, so we need to backtrack and mark one of
      // the inputs as black. By heuristic, we'll mark the one with the most
      // unique dependencies.
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0; i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToDefn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i]) {
            if (depCounter.get(dep) === 1) uniqueDeps++;
          }
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1) {
        throw new Error(
          `internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`,
        );
      }
      const assocVar = eqn.inputs[assocInput] as Var;
      p2idx = varToDefn.get(assocVar)!; // backtrack to that equation
      if (DEBUG >= 1) _splitStats.p2++;
      for (const out of jaxpr.eqns[p2idx++].outBinders) {
        blackNodes.add(out);
      }
    } else {
      if (isAllBlack) {
        // Black kernel endpoint: output is materialized, treat as a single base dep.
        for (const out of eqn.outBinders) p2Deps.set(out, new Set([out]));
      } else {
        const s = new Set(depCounter.keys());
        for (const out of eqn.outBinders) p2Deps.set(out, s);
      }
    }
  }

  if (DEBUG >= 1) {
    _splitStats.total = blackNodes.size;
    console.log(
      `splitGraphDataflow: ${blackNodes.size} black / ${jaxpr.eqns.length} eqns` +
        ` (reductions=${_splitStats.reductions} cascade=${_splitStats.cascade}` +
        ` diamonds=${_splitStats.diamonds} cheapDiamonds=${_splitStats.cheapDiamonds}` +
        ` cleanShape=${_splitStats.cleanShape}` +
        ` heterogeneous=${_splitStats.heterogeneous} routines=${_splitStats.routines}` +
        ` special=${_splitStats.special} p2=${_splitStats.p2})`,
    );
  }
  return blackNodes;
}
