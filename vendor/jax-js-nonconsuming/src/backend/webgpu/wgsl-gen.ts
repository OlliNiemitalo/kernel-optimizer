/**
 * @file Shared WGSL expression generator for WebGPU codegen.
 *
 * Extracted from block-map.ts to be reusable by both block_map fused shaders
 * and standalone associative scan shaders. Provides CSE (common subexpression
 * elimination) and AluExp simplification with bounded variable ranges.
 */

import { castSaturateWgsl, constToWgsl, dtypeToWgsl } from "./codegen";
import {
  accessorGlobal,
  AluExp,
  AluGroup,
  AluOp,
  DType,
  isFloatDtype,
  Kernel,
} from "../../alu";
import { isSymbolicSize } from "../../shape";
import { strip1 } from "../../utils";

/**
 * Generate a WGSL reduction accumulation statement.
 *
 * Shared by block-map tiled reductions and workgroup-associative-scan
 * reduction bodies. Returns the WGSL source for accumulating `rhs` into
 * `accExpr` using `op`.
 */
export function wgslReductionAccumStmt(
  op: AluOp,
  accExpr: string,
  rhs: string,
): string {
  if (op === AluOp.Add) return `${accExpr} += ${rhs};`;
  if (op === AluOp.Mul) return `${accExpr} *= ${rhs};`;
  if (op === AluOp.Min) return `${accExpr} = min(${accExpr}, ${rhs});`;
  if (op === AluOp.Max) return `${accExpr} = max(${accExpr}, ${rhs});`;
  throw new Error(`Unsupported reduction op in WGSL: ${op}`);
}

/**
 * Cast a reduction body RHS to the reduction accumulator dtype if needed.
 */
export function wgslCastReductionRhs(
  rhs: string,
  bodyDtype: DType,
  reDtype: DType,
): string {
  const reTy = dtypeToWgsl(reDtype, false);
  return bodyDtype !== reDtype ? `${reTy}(${rhs})` : rhs;
}

/**
 * Info for remapping a GlobalIndex flat index from body-local strides
 * to global buffer strides at the AluExp level (enabling simplify()).
 */
export interface GlobalIndexRemapInfo {
  bodyShape: number[];
  bodyStrides: number[];
  globalStrides: number[];
}

/** Callback to resolve a GlobalIndex read to a WGSL expression. */
export type ResolveGlobalIndex = (
  bufIdx: number,
  indexExpr: string,
  dtype: DType,
  isGloballyStrided?: boolean,
  bankPadApplied?: boolean,
) => string;

/** Emit function signature for WGSL codegen. */
export type EmitFn = (...lines: (string | symbol)[]) => void;

export interface CreateWgslGenOptions {
  /** The kernel whose expressions to translate. */
  kernel: Kernel;
  /** Prefix for gensym'd CSE variable names. */
  prefix: string;
  /** Callback to resolve GlobalIndex buffer reads. */
  resolveGlobalIndex: ResolveGlobalIndex;
  /** Emit function for CSE variable declarations. */
  emit: EmitFn;
  /** Block/workgroup size — fallback for symbolic kernel sizes. */
  blockSize: number;
  /** Override variable names. */
  variableOverrides?: Map<string, string>;
  /** Override gidx with a custom AluExp (e.g., eidx for scan bodies). */
  gidxOverride?: AluExp;
  /** Override ridx with a custom AluExp (e.g., for tree reductions). */
  ridxOverride?: AluExp;
  /** Remap GlobalIndex flat indices from body-local to global strides. */
  globalIndexRemap?: Map<number, GlobalIndexRemapInfo>;
  /** Map from buffer index to inner dimension size for bank-padding. */
  bankPadDims?: Map<number, number>;
}

/**
 * Create a gen() function that translates AluExp trees into WGSL expressions.
 *
 * Features:
 * - CSE: common subexpressions are hoisted to `let` bindings via `emit`
 * - Expression simplification with bounded gidx/ridx ranges
 * - Bank-pad index rewriting (when bankPadDims provided)
 * - GlobalView → Where(valid, GlobalIndex, 0) rewriting
 * - GlobalIndex remap for stride correction (when globalIndexRemap provided)
 */
export function createWgslGen(
  opts: CreateWgslGenOptions,
): (exp: AluExp) => string {
  const {
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
  } = opts;

  let gensymCount = 0;
  const gensym = () => `${prefix}_alu${gensymCount++}`;
  const isGensym = (text: string) =>
    text.startsWith(prefix + "_alu") &&
    /^\d+$/.test(text.slice(prefix.length + 4));

  // Simplify kernel expressions with bounded gidx range.
  // gidx ∈ [0, kernelSize-1] lets the simplifier eliminate redundant
  // mod/div from unravelAlu().
  const kernelSize = isSymbolicSize(kernel.size)
    ? blockSize
    : (kernel.size as number);
  const gidxBound =
    gidxOverride ?? AluExp.special(DType.Int32, "gidx", kernelSize);
  const simplifiedMap = new Map<AluExp, AluExp>();

  // Bank-pad rewrite: transform GlobalIndex flat index from `idx` to
  // `idx + Idiv(idx, innerDim)` for bank-padded shmem buffers.
  const bankPadRewrite = bankPadDims
    ? (exp: AluExp): AluExp | undefined => {
        if (exp.op === AluOp.GlobalIndex) {
          const bufIdx = exp.arg[0] as number;
          const innerDim = bankPadDims.get(bufIdx);
          if (innerDim !== undefined) {
            const flatIdx = exp.src[0];
            const paddedIdx = AluExp.add(
              flatIdx,
              AluExp.idiv(flatIdx, AluExp.i32(innerDim)),
            );
            return AluExp.globalIndex(
              exp.dtype,
              bufIdx,
              exp.arg[1] as number,
              paddedIdx,
            );
          }
        }
      }
    : undefined;

  for (const output of kernel.outputs) {
    const vars: Record<string, AluExp> = { gidx: gidxBound };
    if (output.reduction && !isSymbolicSize(output.reduction.size)) {
      vars.ridx =
        ridxOverride ??
        AluExp.special(DType.Int32, "ridx", output.reduction.size as number);
    }
    let processed = output.exp.substitute(vars).rewriteGlobalViews();
    if (bankPadRewrite) processed = processed.rewrite(bankPadRewrite);
    simplifiedMap.set(output.exp, processed.simplify());
    if (output.reduction) {
      let procEpilogue = output.reduction.epilogue
        .substitute({ gidx: gidxBound })
        .rewriteGlobalViews();
      if (bankPadRewrite) procEpilogue = procEpilogue.rewrite(bankPadRewrite);
      simplifiedMap.set(output.reduction.epilogue, procEpilogue.simplify());
    }
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
  for (const sExp of simplifiedMap.values()) countReferences(sExp);

  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    // Resolve original expressions to their simplified counterparts
    exp = simplifiedMap.get(exp) ?? exp;
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
      const [k0, k1, c0, c1] = src.map((s) => strip1(gen(s)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return variableOverrides?.get(arg as string) ?? (arg as string);
    } else if (op === AluOp.GlobalView) {
      const [gid, st] = arg as [number, import("../../shape").ShapeTracker];
      const rewritten = accessorGlobal(dtype, gid, st, src);
      return gen(rewritten);
    } else if (op === AluOp.GlobalIndex) {
      const bufIdx = arg[0] as number;
      const remap = globalIndexRemap?.get(bufIdx);
      if (remap) {
        const flatIdx = src[0];
        const nd = remap.bodyShape.length;
        let correction: AluExp | null = null;
        for (let d = 0; d < nd; d++) {
          const strideDiff = remap.globalStrides[d] - remap.bodyStrides[d];
          if (strideDiff === 0) continue;
          let coord: AluExp;
          if (d === 0) {
            coord = AluExp.idiv(flatIdx, AluExp.i32(remap.bodyStrides[d]));
          } else {
            coord = AluExp.mod(
              AluExp.idiv(flatIdx, AluExp.i32(remap.bodyStrides[d])),
              AluExp.i32(remap.bodyShape[d]),
            );
          }
          const term = AluExp.mul(coord, AluExp.i32(strideDiff));
          correction = correction ? AluExp.add(correction, term) : term;
        }
        const remapped = correction
          ? AluExp.add(flatIdx, correction).simplify()
          : flatIdx;
        const indexExpr = strip1(gen(remapped));
        source = resolveGlobalIndex(
          bufIdx,
          indexExpr,
          dtype,
          true,
          bankPadDims?.has(bufIdx),
        );
      } else {
        const indexExpr = strip1(gen(src[0]));
        source = resolveGlobalIndex(
          bufIdx,
          indexExpr,
          dtype,
          undefined,
          bankPadDims?.has(bufIdx),
        );
      }
      if (dtype === DType.Bool) source = `(${source} != 0)`;
    }

    if (!source) {
      throw new Error(`WGSL gen: unsupported AluOp ${op}`);
    }

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
  return gen;
}
