// lax.scan primitive implementation for jax-js
//
// Scans a function over leading axis of input arrays while carrying state.
// This enables expressing recurrences (like Kalman filters) in a functional style.
//
// This version uses the Scan primitive for efficient execution.

import type { Array } from "../frontend/array";
// eslint-disable-next-line no-restricted-imports -- zeros() is correct here: creates user-visible empty Y arrays for length-0 scan that need anonymous marking.
import { zeros } from "../frontend/array";
import { bind, getAval, Primitive, ShapedArray } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as tree from "../tree";
import type { JsTree } from "../tree";
import type { ScanPath } from "../utils";

/**
 * Options for {@link scan}.
 */
export interface ScanOptions {
  /**
   * Number of iterations. If not provided, inferred from the leading axis of `xs`.
   * Required when `xs` is empty or null.
   */
  length?: number;

  /**
   * If `true`, process `xs` in reverse order (from last to first element).
   * The output `ys` will also be in reverse order.
   * @default false
   */
  reverse?: boolean;

  /**
   * Accept only specific scan implementation paths. Throws an error if the
   * actual path chosen is not in the list.
   *
   * This is primarily useful for testing to ensure optimized code paths are used.
   *
   * Valid paths:
   * - `"compiled-loop"` — entire scan loop compiled to native code (WASM/WebGPU)
   * - `"preencoded-routine"` — pre-encoded GPU dispatches for routine bodies (WebGPU only)
   * - `"fallback"` — JS loop calling body program per iteration
   *
   * @example
   * ```ts
   * // Accept only the compiled-loop (native) scan path
   * lax.scan(f, init, xs, { acceptPath: "compiled-loop" });
   *
   * // Accept any native path (compiled-loop or preencoded-routine)
   * lax.scan(f, init, xs, { acceptPath: ["compiled-loop", "preencoded-routine"] });
   * ```
   */
  acceptPath?: ScanPath | ScanPath[];

  /**
   * Control gradient checkpointing during reverse-mode autodiff.
   *
   * By default, `grad(scan)` uses √N checkpointing: only O(√N) intermediate carry
   * values are stored, and the rest are recomputed from the nearest checkpoint
   * during the backward pass. This trades ~2× computation for O(√N) memory.
   *
   * - `undefined` or `true` (default): use segment size of `ceil(√N)`
   * - A positive integer: use that as the segment size (larger = more memory, less recompute)
   * - `false`: store all N intermediate carries (O(N) memory, no recomputation)
   *
   * @example
   * ```ts
   * // Default: √N checkpointing is used automatically
   * const dxs = grad((xs) => {
   *   const [carry, _] = lax.scan(step, init, xs);
   *   return carry.sum();
   * })(xs);
   *
   * // Opt out: store all carries (faster, more memory)
   * const dxs2 = grad((xs) => {
   *   const [carry, _] = lax.scan(step, init, xs, { checkpoint: false });
   *   return carry.sum();
   * })(xs);
   * ```
   */
  checkpoint?: boolean | number;
}

/**
 * Scan a function over leading array axes while carrying along state.
 *
 * Think of `scan` as a functional `reduce` that also returns all intermediate
 * results. It iterates over the leading axis of `xs`, threading a "carry" value
 * through each step and collecting outputs.
 *
 * ## Type Signature
 *
 * ```ts
 * scan(f, init, xs) → [finalCarry, ys]
 * scan(f, init, null, { length }) → [finalCarry, ys]  // carry-only scan
 *
 * // Where:
 * // f: (carry: C, x: X | null) => [C, Y | null]  -- step function
 * // init: C                               -- initial carry
 * // xs: X[] | null                        -- input array or null for carry-only
 * // finalCarry: C                         -- carry after last iteration
 * // ys: Y[] | null                        -- stacked outputs (null if Y=null)
 * ```
 *
 * ## Semantics
 *
 * The semantics are roughly equivalent to this JavaScript:
 * ```ts
 * function scan(f, init, xs) {
 *   let carry = init;
 *   const ys = [];
 *   for (const x of xs) {
 *     const [newCarry, y] = f(carry, x);
 *     carry = newCarry;
 *     ys.push(y);
 *   }
 *   return [carry, np.stack(ys)];
 * }
 * ```
 *
 * Unlike a plain JavaScript loop:
 * - Both `xs` and `ys` can be arbitrary pytrees (nested objects/arrays)
 * - The scan is compiled to efficient native code (WASM/WebGPU)
 * - Supports autodiff: `grad(f)` works through scan
 * - The carry shape/dtype must be fixed across all iterations
 *
 * ## Ownership
 *
 * **Inputs:** `init` and `xs` are NOT consumed. Use `using` or `.dispose()` to
 * manage their lifetime if needed.
 *
 * **Body function:**
 * - `carry` and `x` are **managed** by scan — do NOT manually dispose them
 * - Operations do NOT consume inputs, so no `.ref` is needed inside the body
 *
 * **Example — simple body:**
 * ```ts
 * const step = (carry, x) => {
 *   const newCarry = np.add(carry, x);
 *   return [newCarry, carry];  // carry used freely in multiple places
 * };
 * ```
 *
 * **Outputs (caller owns):**
 * - `finalCarry` and `ys` are owned by caller — dispose when done
 *
 * @param f - Step function `(carry, x) => [newCarry, y]` where:
 *   - `carry` is the current state (same structure as `init`)
 *   - `x` is a slice of `xs` along axis 0, or `null` if `xs` is null
 *   - `newCarry` is the updated state (same structure/shape as `carry`)
 *   - `y` is the output for this iteration, or `null` to skip output stacking
 * @param init - Initial carry value. Can be a single array or a pytree of arrays.
 * @param xs - Input sequence to scan over, or `null` for carry-only scans.
 *   When an array/pytree, the leading axis is the scan dimension.
 *   When `null`, you must provide `{ length }` in options.
 * @param options - Scan options
 * @returns `[finalCarry, ys]` where:
 *   - `finalCarry` has the same structure as `init`
 *   - `ys` has the same structure as `y` from `f`, with each leaf having
 *     an additional leading axis of size `length`. If `y` is `null`, `ys` is `null`
 *     (no memory allocated for outputs).
 *
 * @example Cumulative sum
 * ```ts
 * import { lax, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
 *
 * const step = (carry, x) => {
 *   const sum = np.add(carry, x);
 *   return [sum, sum];  // same array can be used in both outputs
 * };
 *
 * using init = np.array([0.0]);
 * using xs = np.array([[1], [2], [3], [4], [5]]);
 * const [final, sums] = await lax.scan(step, init, xs);
 *
 * console.log(await final.data());  // [15]
 * console.log(await sums.data());   // [[1], [3], [6], [10], [15]]
 *
 * final.dispose();
 * sums.dispose();
 * ```
 *
 * @example Factorial via scan
 * ```ts
 * // Compute n! for n = 1..5
 * const step = (carry, x) => {
 *   const next = np.multiply(carry, x);
 *   return [next, next];
 * };
 *
 * using init = np.array([1]);
 * using xs = np.array([[1], [2], [3], [4], [5]]);
 * const [final, factorials] = await lax.scan(step, init, xs);
 * // factorials = [[1], [2], [6], [24], [120]]
 *
 * final.dispose();
 * factorials.dispose();
 * ```
 *
 * @example Pytree carry (multiple state variables)
 * ```ts
 * // Track both sum and count
 * using one = np.array([1]);
 * const step = (carry, x) => {
 *   const newSum = np.add(carry.sum, x);
 *   const newCount = np.add(carry.count, one);
 *   return [
 *     { sum: newSum, count: newCount },
 *     { sum: newSum, count: newCount }
 *   ];
 * };
 *
 * const init = { sum: np.array([0]), count: np.array([0]) };
 * using xs = np.array([[10], [20], [30]]);
 * const [final, history] = await lax.scan(step, init, xs);
 * // final.sum = [60], final.count = [3]
 *
 * init.sum.dispose();
 * init.count.dispose();
 * final.sum.dispose();
 * final.count.dispose();
 * history.sum.dispose();
 * history.count.dispose();
 * ```
 *
 * @example Reverse scan
 * ```ts
 * // Process sequence from end to beginning
 * const [final, ys] = await lax.scan(step, init, xs, { reverse: true });
 * ```
 *
 * ## jax-js Extensions
 *
 * These features extend JAX's scan API for TypeScript/JavaScript ergonomics:
 *
 * ### xs=null (carry-only scan)
 *
 * Pass `null` as `xs` with `{ length }` to iterate without input arrays.
 * Useful for generators, RNG sequences, Fibonacci, or any state-only iteration.
 * The body receives `null` as the second argument.
 *
 * ### Y=null (skip output stacking)
 *
 * Return `[newCarry, null]` from the body to skip allocating stacked outputs.
 * Useful when you only need the final carry (e.g., Mandelbrot iteration counts).
 * The returned `ys` will be `null`, saving memory for large iteration counts.
 *
 * @example xs=null: Carry-only scan
 * ```ts
 * // Generate a sequence without allocating input arrays
 * using one = np.array([1.0]);
 * const step = (carry, _x) => {
 *   const next = np.add(carry, one);
 *   return [next, carry];  // output is old carry value
 * };
 *
 * using init = np.array([0.0]);
 * const [final, ys] = await lax.scan(step, init, null, { length: 5 });
 * // ys = [[0], [1], [2], [3], [4]], final = [5]
 *
 * final.dispose();
 * ys.dispose();
 * ```
 *
 * @example Y=null: Skip output stacking
 * ```ts
 * // Only need final carry, not intermediate outputs (saves memory)
 * const step = (carry, x) => {
 *   const Asq = carry.A.mul(carry.A);
 *   const newA = Asq.add(x);
 *   const newCount = carry.count.add(Asq.less(100).astype(np.int32));
 *   return [{ A: newA, count: newCount }, null];  // null skips Y stacking
 * };
 *
 * const init = {
 *   A: np.zeros([100]),
 *   count: np.zeros([100], { dtype: np.int32 }),
 * };
 * using xs = np.ones([10, 100]);
 * const [final, ys] = await lax.scan(step, init, xs);
 * // ys is null — no memory allocated for intermediate outputs
 *
 * init.A.dispose();
 * init.count.dispose();
 * final.A.dispose();
 * final.count.dispose();
 * ```
 *
 * @example jit(scan) - Compile the entire scan loop
 * ```ts
 * import { jit, lax, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
 *
 * // Wrap scan in jit to compile the entire loop into optimized native code.
 * // This is the most common and efficient pattern for production use.
 * const step = (carry, x) => {
 *   const newCarry = np.add(carry, x);
 *   return [newCarry, newCarry];
 * };
 *
 * const scanFn = jit((init, xs) => lax.scan(step, init, xs));
 *
 * using init = np.array([0.0]);
 * using xs = np.array([[1.0], [2.0], [3.0]]);
 * const [final, ys] = await scanFn(init, xs);
 *
 * console.log(await final.data());  // [6]
 * final.dispose();
 * ys.dispose();
 * scanFn.dispose();  // Free compiled program
 * ```
 *
 * @example scan(jit(body)) - JIT-compile only the step function
 * ```ts
 * import { jit, lax, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
 *
 * // JIT-compile just the step function. Each iteration calls compiled code,
 * // but the loop itself runs in JavaScript. Useful when step is expensive
 * // but you want to inspect intermediate values or the scan body is dynamic.
 * const step = jit((carry, x) => {
 *   const newCarry = np.add(carry, x);
 *   return [newCarry, newCarry];
 * });
 *
 * using init = np.array([0.0]);
 * using xs = np.array([[1.0], [2.0], [3.0]]);
 * const [final, ys] = await lax.scan(step, init, xs);
 *
 * console.log(await final.data());  // [6]
 * final.dispose();
 * ys.dispose();
 * step.dispose();  // Free compiled step function
 * ```
 *
 * @example With grad for differentiation
 * ```ts
 * import { grad, lax, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
 *
 * const loss = (init, xs) => {
 *   const [final, ys] = lax.scan(step, init, xs);
 *   using _final = final;
 *   using _ys = ys;
 *   return np.sum(ys);
 * };
 *
 * const gradLoss = grad(loss);
 * using init = np.array([0.0]);
 * using xs = np.array([[1.0], [2.0], [3.0]]);
 * const [dInit, dXs] = await gradLoss(init, xs);
 * dInit.dispose();
 * dXs.dispose();
 * ```
 *
 * **Performance Note (WebGPU):** `scan` involves declarative multi-step execution. While it
 * optimizes heavily within a compiled loop (especially for reductions), bodies with complex
 * cross-element data hazards may fall back to multi-dispatch execution with VRAM roundtrips.
 * If you are executing simple strictly pointwise operations sequentially without tracking
 * history (`Y = null`), consider `lax.foriLoop` instead — when its rewrite guards are met
 * (pointwise-only body, rank 1–2 same-shape carries, concrete bounds, WebGPU backend),
 * it compiles into a single fused GPU dispatch. See `lax.foriLoop` docs for the full
 * list of requirements.
 *
 * @see {@link https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html | JAX lax.scan}
 */
export function scan<
  Carry extends JsTree<Array>,
  X extends JsTree<Array> | null,
  Y extends JsTree<Array> | null,
>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  options?: ScanOptions,
): [Carry, Y] {
  const opts: ScanOptions = options ?? {};
  const { length: lengthOpt, reverse = false, acceptPath, checkpoint } = opts;

  // Handle xs=null case (carry-only scan with no input arrays)
  const xsIsNull = xs === null;

  // Flatten inputs for primitive call
  const [initFlat, initTreedef] = tree.flatten(init);
  const [xsFlat, xsTreedef] = xsIsNull ? [[], null] : tree.flatten(xs);

  // Determine scan length from input
  const n = lengthOpt ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);

  // Length-0 is handled after tracing (we need Y treedef from the trace).

  // Get abstract values for carry and x_slice (xs with leading dim removed)
  const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
  const xSliceAvals = xsFlat.map((arr) => {
    const aval = getAval(arr);
    // Remove leading dimension for x_slice
    return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
  });

  // Create a wrapper function that takes flat arrays and returns flat arrays.
  // yTreedef_ is captured by the closure and set during tracing.
  let yTreedef_: tree.JsTreeDef | undefined;
  const flatF = (
    carryFlat: Array[],
    xSliceFlat: Array[],
  ): [Array[], Array[]] => {
    const carry = tree.unflatten(initTreedef, carryFlat) as Carry;
    // For xs=null, pass null/undefined to body function (matches JAX behavior)
    const xSlice = xsIsNull
      ? xs
      : (tree.unflatten(xsTreedef!, xSliceFlat) as X);
    const [newCarry, y] = f(carry, xSlice as X);
    const [newCarryFlat] = tree.flatten(newCarry);
    // tree.flatten handles null as empty node with no leaves (NodeType.None)
    const [yFlat, yTreedef] = tree.flatten(y as JsTree<Array>);
    yTreedef_ = yTreedef;
    return [newCarryFlat, yFlat];
  };

  // Trace the function to get jaxpr
  const traceFn = (...args: Array[]): Array[] => {
    const numCarry = carryAvals.length;
    const carryFlat = args.slice(0, numCarry);
    const xSliceFlat = args.slice(numCarry);
    const [newCarryFlat, yFlat] = flatF(carryFlat, xSliceFlat);
    return [...newCarryFlat, ...yFlat];
  };

  // Build abstract values for tracing
  const traceAvals = [...carryAvals, ...xSliceAvals];

  // Trace to get jaxpr.
  const { jaxpr: closedJaxpr, treedef: _outTreedef } = makeJaxpr(traceFn)(
    ...traceAvals,
  );
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;

  // Handle length-0 scans: return (init, empty_ys) like JAX.
  if (n === 0) {
    if (xsIsNull && lengthOpt === undefined) {
      throw new Error("scan: length option is required when xs is null");
    }

    // Dispose captured consts (nothing was consumed on this path)
    closedJaxpr.dispose();

    // Final carry is the initial carry.
    const finalCarryFlat = initFlat;

    // Build empty Y leaves based on jaxpr outputs.
    const numCarry = initFlat.length;
    const yOutAtoms = jaxpr.outs.slice(numCarry);

    const yFlatEmpty = yOutAtoms.map((atom) => {
      const aval = atom.aval;
      return zeros([0, ...(aval.shape as number[])], { dtype: aval.dtype });
    });

    // Reconstruct pytrees
    const finalCarry = tree.unflatten(initTreedef, finalCarryFlat) as Carry;
    const ys =
      yTreedef_ === tree.JsTreeDef.none
        ? (null as Y)
        : (tree.unflatten(yTreedef_!, yFlatEmpty) as Y);

    return [finalCarry, ys];
  }

  // Call the Scan primitive
  // Args: [...consts, ...initCarry, ...xs]
  const scanArgs = [...consts, ...initFlat, ...xsFlat];

  const numCarry = initFlat.length;
  const numConsts = consts.length;

  const results = bind(Primitive.Scan, scanArgs, {
    jaxpr,
    numCarry,
    numConsts,
    length: n,
    reverse,
    acceptPath,
    checkpoint,
  });

  // Split results into carry and ys
  const carryOut = results.slice(0, numCarry) as Array[];
  const ysFlat = results.slice(numCarry) as Array[];

  // Dispose the captured constants from tracing.  getOrMakeConstTracer ref'd
  // each const, so ClosedJaxpr.dispose() decrements that ref.  User-held consts
  // survive (they still have the user's own ref).  Anonymous consts created
  // inside the body (e.g. np.array([2,3])) get freed.
  closedJaxpr.dispose();

  // Reconstruct pytrees
  const finalCarry = tree.unflatten(initTreedef, carryOut) as Carry;
  const ys = tree.unflatten(yTreedef_!, ysFlat) as Y;

  return [finalCarry, ys];
}
