/**
 * @file Linear stencil analysis for halo block_map VJP.
 *
 * Analyzes a block_map body jaxpr to determine if it is a linear stencil —
 * a weighted sum of axis-shifted slices of a single input. When eligible,
 * enables an O(1)-dispatch gather-based backward pass instead of the
 * O(numBlocks) unrolled pad+add fallback.
 *
 * A linear stencil body has only {Shrink, Add, Mul(Lit)} equations and
 * produces output = Σ weight_i · Shrink(input, offset_i).
 */

import type { Pair } from "../shape";
import { Primitive } from "./core";
import { type Atom, Jaxpr, Lit, Var } from "./jaxpr";

/** A single term in a linear stencil decomposition. */
export interface StencilTerm {
  /** Per-axis Shrink bounds: [[start0, end0], [start1, end1], ...] */
  slice: Pair[];
  /** Scalar weight (1.0 for unweighted terms). */
  weight: number;
}

/** Result of stencil analysis. null means ineligible. */
export interface StencilDescriptor {
  /** Index into jaxpr.inBinders of the stencil input. */
  inputIdx: number;
  /** Linear combination of Shrink terms that form the output. */
  terms: StencilTerm[];
}

function getTerms(
  atom: Atom,
  varTerms: Map<Var, StencilTerm[] | null>,
): StencilTerm[] | null | undefined {
  if (atom instanceof Lit) return null;
  if (atom instanceof Var) return varTerms.get(atom);
  return undefined;
}

/**
 * Analyze a block_map forward body jaxpr for linear stencil pattern.
 *
 * @param jaxpr        The forward body jaxpr.
 * @param haloInputIdx Index into jaxpr.inBinders of the halo-ed input.
 * @returns StencilDescriptor if the body is a linear combination of
 *          Shrink(input, offsets), null if non-linear or ineligible.
 */
export function analyzeLinearStencil(
  jaxpr: Jaxpr,
  haloInputIdx: number,
): StencilDescriptor | null {
  const stencilInput = jaxpr.inBinders[haloInputIdx];

  // Track each Var's value as a list of stencil terms, or null (non-stencil).
  const varTerms = new Map<Var, StencilTerm[] | null>();

  // All input binders except the stencil input are non-stencil.
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (i !== haloInputIdx) varTerms.set(jaxpr.inBinders[i], null);
  }

  for (const eqn of jaxpr.eqns) {
    if (eqn.outBinders.length !== 1) return null;
    const out = eqn.outBinders[0];

    switch (eqn.primitive) {
      case Primitive.Shrink: {
        const input = eqn.inputs[0];
        if (input === stencilInput) {
          varTerms.set(out, [
            { slice: eqn.params.slice as Pair[], weight: 1.0 },
          ]);
        } else if (input instanceof Var && varTerms.get(input) !== null) {
          // Shrink of a stencil intermediate — not a simple pattern.
          return null;
        } else {
          varTerms.set(out, null);
        }
        break;
      }

      case Primitive.Add: {
        const termsA = getTerms(eqn.inputs[0], varTerms);
        const termsB = getTerms(eqn.inputs[1], varTerms);
        if (termsA === undefined || termsB === undefined) return null;
        if (termsA !== null && termsB !== null) {
          varTerms.set(out, [...termsA, ...termsB]);
        } else if (termsA === null && termsB === null) {
          varTerms.set(out, null);
        } else {
          // Mixed stencil + non-stencil: allow if the non-stencil side is a
          // Lit (additive bias). Bias drops out in the adjoint, so the
          // stencil terms are the only linear contribution.
          if (termsA !== null && eqn.inputs[1] instanceof Lit) {
            varTerms.set(out, termsA);
          } else if (termsB !== null && eqn.inputs[0] instanceof Lit) {
            varTerms.set(out, termsB);
          } else {
            return null;
          }
        }
        break;
      }

      case Primitive.Mul: {
        const [a, b] = eqn.inputs;
        let litValue: number | undefined;
        let terms: StencilTerm[] | null | undefined;

        if (a instanceof Lit && a.aval.shape.length === 0) {
          litValue = a.value;
          terms = getTerms(b, varTerms);
        } else if (b instanceof Lit && b.aval.shape.length === 0) {
          litValue = b.value;
          terms = getTerms(a, varTerms);
        } else {
          // Mul of two non-literal operands: only OK if both non-stencil.
          const tA = getTerms(a, varTerms);
          const tB = getTerms(b, varTerms);
          if (tA === null && tB === null) {
            varTerms.set(out, null);
            break;
          }
          return null; // Mul of stencil values = nonlinear.
        }

        if (terms === undefined) return null;
        if (terms === null) {
          varTerms.set(out, null);
        } else {
          varTerms.set(
            out,
            terms.map((t) => ({
              slice: t.slice,
              weight: t.weight * litValue!,
            })),
          );
        }
        break;
      }

      default:
        return null;
    }
  }

  // Check output: single output, must be stencil-derived.
  if (jaxpr.outs.length !== 1) return null;
  const outTerms = getTerms(jaxpr.outs[0], varTerms);
  if (outTerms === null || outTerms === undefined || outTerms.length === 0)
    return null;

  return { inputIdx: haloInputIdx, terms: outTerms };
}

/**
 * Compute reversed Shrink slice for a stencil term in the backward body.
 *
 * For each grid-mapped axis, the offset s is reversed to (span - s)
 * where span = lo + hi. Non-grid axes are kept unchanged.
 *
 * @param forwardSlice The forward term's Shrink slice.
 * @param inAxes       Per-grid-axis array axis mapping (from block_map params).
 * @param forwardHalo  Per-grid-axis [lo, hi] from the forward pass.
 * @param blockShape   Per-grid-axis block size.
 */
export function reverseStencilSlice(
  forwardSlice: Pair[],
  inAxes: (number | null)[],
  forwardHalo: [number, number][],
  blockShape: number[],
): Pair[] {
  const gridRank = blockShape.length;

  // Build array-axis → grid-axis mapping.
  const arrayAxisToGrid = new Map<number, number>();
  for (let g = 0; g < gridRank; g++) {
    if (inAxes[g] !== null) arrayAxisToGrid.set(inAxes[g]!, g);
  }

  return forwardSlice.map(([start, _end], arrayAxis) => {
    const gridAxis = arrayAxisToGrid.get(arrayAxis);
    if (gridAxis !== undefined) {
      const [lo, hi] = forwardHalo[gridAxis];
      const span = lo + hi;
      const revStart = span - start;
      return [revStart, revStart + blockShape[gridAxis]] as Pair;
    }
    return [start, _end] as Pair;
  });
}
