// lax.blockMap — apply a function independently to each block of inputs.
//
// Partitions inputs along grid axes into blocks of `blockShape`, applies
// the body to each block, and reassembles the results.

import { Array } from "../frontend/array";
import { bind, getAval, Primitive, ShapedArray } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as tree from "../tree";
import type { JsTree } from "../tree";

const JsArray = globalThis.Array;

/**
 * Return the current block index inside a {@link blockMap} body.
 * Only valid when called from within a `blockMap` body function.
 * Produces a scalar int32.
 */
export function blockIndex(): Array {
  return (bind(Primitive.BlockIndex, [], {}) as unknown as Array[])[0];
}

/**
 * Options for {@link blockMap}.
 */
export interface BlockMapOptions {
  /**
   * The block size along each grid axis. `blockShape[g]` is the number of
   * elements processed per block along grid axis `g`.
   */
  blockShape: number[];

  /**
   * Per-input mapping from grid axes to input dimensions.
   *
   * - A single `(number | null)[]` is broadcast to all input leaves.
   * - An array of `(number | null)[][]` provides per-leaf mappings (length
   *   must match the number of flattened input leaves).
   *
   * For each leaf, `inAxes[g]` is the dimension mapped to grid axis `g`,
   * or `null` if that grid axis doesn't slice this input (broadcast).
   *
   * @default `[0, 1, ..., gridRank-1]` for all leaves.
   */
  inAxes?: (number | null)[] | (number | null)[][];

  /**
   * Per-output mapping from grid axes to output dimensions. Same format as
   * `inAxes` but applied to body outputs.
   *
   * @default Same as `inAxes` default.
   */
  outAxes?: (number | null)[] | (number | null)[][];

  /**
   * Explicit grid shape. When provided, overrides grid inference from mapped
   * inputs. Useful when all inputs are broadcast (no mapped axis) but the
   * body should still execute over a grid (e.g., gather/apply stages).
   */
  gridShape?: number[];

  /**
   * Register tiling: each thread computes `threadTile[g]` outputs along
   * grid axis `g`. Workgroup size becomes `blockShape[g] / threadTile[g]`
   * per axis. When set, carries are accumulated in `var<private>` and input
   * tiles are cooperatively loaded to shared memory.
   *
   * Each dimension of `threadTile` must evenly divide the corresponding
   * `blockShape` dimension.
   */
  threadTile?: number[];

  /**
   * Per-input overlap along each mapped grid axis. Each input tile extends
   * beyond the output block range by `[lo, hi]` elements along the mapped
   * dimension. The body receives input shapes of `blockShape[g] + lo + hi`
   * (instead of `blockShape[g]`) along each halo-expanded axis.
   *
   * Format: `halo[i][g] = [lo, hi]` — extra elements before/after the
   * output block's range for input `i` along grid axis `g`.
   * `null` or `[0, 0]` means no halo.
   *
   * A single `([number, number] | null)[]` is broadcast to all inputs.
   * An array of `([number, number] | null)[][]` provides per-input specs.
   *
   * Elements outside the array boundary are zero-padded.
   *
   * @example 3×3 convolution halos
   * ```ts
   * lax.blockMap(stencilBody, { image, kernel }, {
   *   blockShape: [16, 16],
   *   inAxes: { image: [2, 3], kernel: [null, null] },
   *   outAxes: [2, 3],
   *   halo: { image: [[1, 1], [1, 1]], kernel: [null, null] },
   * });
   * // image body shape: [16+1+1, 16+1+1] = [18, 18] per tile
   * // kernel body shape: unchanged (broadcast, no halo)
   * ```
   */
  halo?: ([number, number] | null)[] | ([number, number] | null)[][];
}

/**
 * Apply `f` independently to blocks of `elems`, tiled along a grid defined
 * by `blockShape` and `inAxes`/`outAxes`.
 *
 * @example Elementwise doubling in blocks of 4
 * ```ts
 * const result = lax.blockMap(
 *   (block) => block.mul(np.array(2)),
 *   xs,
 *   { blockShape: [4] },
 * );
 * ```
 */
export function blockMap<I extends JsTree<Array>, O extends JsTree<Array>>(
  f: (block: I) => O,
  elems: I,
  options: BlockMapOptions,
): O {
  const { blockShape } = options;
  const gridRank = blockShape.length;

  // Flatten elems pytree
  const [flatElems, elemsTreedef] = tree.flatten<Array>(elems);
  const numInputs = flatElems.length;
  if (numInputs === 0) {
    throw new Error("blockMap: elems must have at least one leaf array");
  }

  // Resolve inAxes
  const defaultAxes: (number | null)[] = JsArray.from(
    { length: gridRank },
    (_, i) => i,
  );
  const flatInAxes = resolveAxes(
    options.inAxes,
    numInputs,
    defaultAxes,
    "inAxes",
  );

  // Resolve halo: flatHalo[inputIdx][gridAxis] = [lo, hi], default [0,0]
  const flatHalo = resolveHalo(options.halo, numInputs, gridRank);
  // Only store halo in params if any entry is non-zero
  const hasHalo = flatHalo.some((h) =>
    h.some(([lo, hi]) => lo !== 0 || hi !== 0),
  );

  // Compute block-shaped abstract values for tracing
  const blockAvals: ShapedArray[] = flatElems.map((elem, i) => {
    const aval = getAval(elem);
    const shape = [...(aval.shape as number[])];
    for (let g = 0; g < gridRank; g++) {
      if (flatInAxes[i][g] !== null) {
        const [lo, hi] = flatHalo[i][g];
        shape[flatInAxes[i][g]!] = blockShape[g] + lo + hi;
      }
    }
    return new ShapedArray(shape, aval.dtype, aval.weakType);
  });

  // Trace the body function
  let outTreedef: tree.JsTreeDef | undefined;
  const traceFn = (...blockFlat: Array[]): Array[] => {
    const blockPytree = tree.unflatten(elemsTreedef, blockFlat) as I;
    const out = f(blockPytree);
    const [outFlat, td] = tree.flatten<Array>(out as JsTree<Array>);
    outTreedef = td;
    return outFlat;
  };

  const { jaxpr: closedJaxpr } = makeJaxpr(traceFn)(...blockAvals);
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;
  const numConsts = consts.length;

  // Resolve outAxes
  const numOutputs = jaxpr.outs.length;
  const flatOutAxes = resolveAxes(
    options.outAxes,
    numOutputs,
    defaultAxes,
    "outAxes",
  );

  // Call the BlockMap primitive
  const results = bind(Primitive.BlockMap, [...consts, ...flatElems], {
    jaxpr,
    blockShape,
    inAxes: flatInAxes,
    outAxes: flatOutAxes,
    numConsts,
    numInputs,
    gridShape: options.gridShape,
    threadTile: options.threadTile,
    halo: hasHalo ? flatHalo : undefined,
  }) as Array[];

  // Dispose the captured consts from tracing
  closedJaxpr.dispose();

  // Reconstruct output pytree
  return tree.unflatten(outTreedef!, results) as O;
}

/** Resolve a user-provided axes spec into a flat (number|null)[][] array. */
function resolveAxes(
  axes: (number | null)[] | (number | null)[][] | undefined,
  count: number,
  defaultAxes: (number | null)[],
  name: string,
): (number | null)[][] {
  if (axes === undefined) {
    return JsArray.from({ length: count }, () => [...defaultAxes]);
  }
  // Single axes spec (1D array of number|null) → broadcast to all leaves
  if (axes.length > 0 && !JsArray.isArray(axes[0])) {
    const single = axes as (number | null)[];
    return JsArray.from({ length: count }, () => [...single]);
  }
  // Array of per-leaf specs
  const multi = axes as (number | null)[][];
  if (multi.length !== count) {
    throw new Error(
      `blockMap: ${name} has ${multi.length} entries but expected ${count}`,
    );
  }
  return multi;
}

/**
 * Resolve a user-provided halo spec into `flatHalo[inputIdx][gridAxis] = [lo, hi]`.
 * Defaults to `[0, 0]` for all entries.
 */
function resolveHalo(
  halo: ([number, number] | null)[] | ([number, number] | null)[][] | undefined,
  numInputs: number,
  gridRank: number,
): [number, number][][] {
  const zero: [number, number] = [0, 0];
  if (halo === undefined) {
    return JsArray.from({ length: numInputs }, () =>
      JsArray.from({ length: gridRank }, () => [...zero] as [number, number]),
    );
  }
  // Single halo spec (1D array of [lo,hi]|null) → broadcast to all inputs
  if (halo.length > 0 && !JsArray.isArray(halo[0]?.[0])) {
    const single = halo as ([number, number] | null)[];
    if (single.length !== gridRank) {
      throw new Error(
        `blockMap: halo has ${single.length} entries but gridRank is ${gridRank}`,
      );
    }
    const resolved = single.map((h) =>
      h === null
        ? ([...zero] as [number, number])
        : ([...h] as [number, number]),
    );
    return JsArray.from({ length: numInputs }, () =>
      resolved.map((h) => [...h] as [number, number]),
    );
  }
  // Array of per-input specs
  const multi = halo as ([number, number] | null)[][];
  if (multi.length !== numInputs) {
    throw new Error(
      `blockMap: halo has ${multi.length} entries but expected ${numInputs}`,
    );
  }
  return multi.map((perInput) => {
    if (perInput.length !== gridRank) {
      throw new Error(
        `blockMap: halo entry has ${perInput.length} axes but gridRank is ${gridRank}`,
      );
    }
    return perInput.map((h) =>
      h === null
        ? ([...zero] as [number, number])
        : ([...h] as [number, number]),
    );
  });
}
