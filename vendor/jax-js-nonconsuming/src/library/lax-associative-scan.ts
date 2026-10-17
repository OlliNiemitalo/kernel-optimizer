// lax.associativeScan — parallel prefix scan via an associative binary operator.
//
// Algorithm: Kogge-Stone parallel prefix scan (doubling ladder).
//
// In round k (stride = 2^k), we compute for each element i:
//   result[i] = fn(result[i - stride], result[i])   for i ≥ stride
//   result[i] = result[i]                           for i <  stride
//
// This requires ceil(log₂ N) rounds. Each round is a fully parallel
// elementwise operation across all N positions — perfect for GPU kernel
// fusion via JIT (ceil(log₂ N) dispatch roundtrips instead of N).
//
// Backend performance notes:
//   WebGPU (M7.4 fused): All body kernel steps fused into a single WGSL
//     compute shader per Kogge-Stone round. Leaves are interleaved into
//     ping/pong buffers; JS drives ceil(log₂ N) rounds, each requiring
//     exactly 1 GPU dispatch regardless of body complexity. This is the
//     hardware-imposed floor: Kogge-Stone needs a global barrier between
//     rounds, and WebGPU has no cross-workgroup barrier within a single
//     dispatch. Requires all leaves to share the same dtype (homogeneous).
//   WASM (M7.2 compiled-loop + M7.3 parallel): Full Kogge-Stone ladder
//     compiled into a single WASM module with N as runtime i32 parameter.
//     M7.3 adds parallel inner-round dispatch via WasmWorkerPool for
//     N ≥ 4096. O(N log N) total work vs scan's O(N).
//   CPU: same JS-loop overhead as WASM; lax.scan is faster.
//   WebGL: no compiled-loop for scan (uses JS fallback), so assocScan's
//     O(log N) shader dispatches may be competitive — untested.
//
// Properties:
//  - O(N log N) total work, O(log N) depth (rounds)
//  - Works on any pytree of Arrays
//  - Supports reverse scan and arbitrary axis
//  - Fully differentiable through JAX-style AD (fn is an ordinary function)
//
// Reference: Kogge & Stone (1973), "A Parallel Algorithm for the Efficient
// Solution of a General Class of Recurrence Equations"
//
// Ownership contract:
//  - Caller-provided `elems` leaves are NEVER consumed.
//  - Caller owns all returned leaves and must dispose them.
//  - The function disposes all internal intermediates.

import type { Array } from "../frontend/array";
import * as core from "../frontend/core";
import {
  _registerAssociativeScanCoreImpl,
  bind,
  Primitive,
  ShapedArray,
} from "../frontend/core";
import { evalJaxpr, makeJaxpr } from "../frontend/jaxpr";
import type { Jaxpr } from "../frontend/jaxpr";
import { moveaxis, vmap } from "../frontend/vmap";
import * as tree from "../tree";
import type { JsTree } from "../tree";
import { checkAxis } from "../utils";
import { DEBUG } from "../utils";

/**
 * Options for {@link associativeScan}.
 */
export interface AssociativeScanOptions {
  /**
   * The axis along which to perform the scan.
   * @default 0
   */
  axis?: number;

  /**
   * If `true`, perform the scan in reverse order (right-to-left inclusive
   * prefix scan). The output at position `i` combines elements from `i` to
   * `N-1`.
   * @default false
   */
  reverse?: boolean;
}

function associativeScanCore<T extends JsTree<Array>>(
  fn: (a: T, b: T) => T,
  elems: T,
  { axis = 0, reverse = false }: AssociativeScanOptions = {},
): T {
  // In eager mode, we compile the whole associative scan at the public wrapper.
  // In abstract traces (jit/grad/vmap tracing), avoid creating nested JIT
  // wrappers here so the outer trace can see and optimize the full body.
  const runStep = fn;

  // Slice an Array along `axis` from `start` (inclusive) to `end` (exclusive).
  // Returns a zero-copy ShapeTracker view — no allocation.
  function sliceAxis(
    a: Array,
    axis: number,
    start: number,
    end: number,
  ): Array {
    const slice = a.shape.map<[number, number]>((size, i) =>
      i === axis ? [start, end] : [0, size],
    );
    return core.shrink(a, slice) as Array;
  }

  /**
   * Apply an associative binary operator for a parallel prefix scan.
   *
   * `fn` must be **associative** (need not be commutative). Result satisfies:
   * ```
   * result[0] = elems[0]
   * result[i] = fn(result[i-1], elems[i])
   * ```
   *
   * Implemented via the Kogge-Stone doubling algorithm: O(log N) rounds of
   * parallel pair-wise combination instead of O(N) sequential steps. Each
   * round is a single batched `fn` call that can be kernel-fused by `jit()`.
   *
   * @param fn - Associative binary operator. Takes and returns pytrees with the
   *   same leaf shapes as `elems`. Must not close over mutable state. Any Arrays
   *   created internally by `fn` that are not part of the returned pytree must
   *   be disposed by `fn`.
   * @param elems - The input sequence. A pytree of Arrays whose `axis`-th dim
   *   has length N. All leaves must agree on N.
   * @param options - `{ axis?, reverse? }`
   * @returns A pytree matching `elems` structure and shape, where position i
   *   holds the prefix result up to (and including) position i. Caller owns
   *   all returned leaf Arrays and must dispose them.
   *
   * @example Cumulative sum
   * ```ts
   * using xs = np.array([1, 2, 3, 4]);
   * using ys = lax.associativeScan((a, b) => np.add(a, b), xs);
   * // ys ≈ [1, 3, 6, 10]
   * ```
   *
   * @example Parallel Kalman filter via affine-map composition
   * ```ts
   * // compose((a1,b1), (a2,b2)) = (a2*a1, a2*b1 + b2)
   * const compose = (p, q) => ({ a: q.a.mul(p.a), b: q.a.mul(p.b).add(q.b) });
   * const result = lax.associativeScan(compose, { a: aArr, b: bArr });
   * ```
   */

  // ------------------------------------------------------------------
  // 1. Flatten pytree and validate.
  // ------------------------------------------------------------------
  const [flatElems, treedef] = tree.flatten<Array>(elems);
  if (flatElems.length === 0) {
    throw new Error("associativeScan: elems must have at least one leaf array");
  }
  const ndim = flatElems[0].ndim;
  if (ndim === 0) {
    throw new Error("associativeScan: leaf arrays must be at least 1-D");
  }
  const normAxis = checkAxis(axis, ndim);
  const N = flatElems[0].shape[normAxis];

  // ------------------------------------------------------------------
  // 2. Move scan axis to position 0 for uniform treatment.
  //    `moveaxis` returns the same object when axis == 0 already.
  // ------------------------------------------------------------------
  // Track which moved arrays are fresh allocations (need disposal later).
  const movedArrays: Array[] = flatElems.map(
    (a) => moveaxis(a, normAxis, 0) as Array,
  );

  // ------------------------------------------------------------------
  // 3. Optionally reverse the input.
  // ------------------------------------------------------------------
  // `working` is the array we'll evolve. Each element is either:
  //   - The original flatElems[i] (if axis==0 and no reverse) → don't dispose
  //   - A moved view (if axis≠0) → dispose after use
  //   - A flipped view (if reverse) → dispose after use
  // We track "is this array owned by us (was created by us)?" per element.
  let working: Array[];
  let workingOwned: boolean[]; // true = we must dispose this when done

  if (reverse) {
    working = [];
    workingOwned = [];
    for (let i = 0; i < flatElems.length; i++) {
      const m = movedArrays[i];
      const flipped = core.reverse(m, 0) as Array;
      // Dispose the moved intermediate if it was a fresh allocation (axis≠0).
      if (m !== flatElems[i]) m.dispose();
      working.push(flipped);
      workingOwned.push(true); // flip always creates a new view
    }
  } else {
    working = movedArrays;
    workingOwned = flatElems.map((a, i) => movedArrays[i] !== a); // false if same object (axis==0)
  }

  // ------------------------------------------------------------------
  // 4. Trivial cases (N=0, N=1).
  // ------------------------------------------------------------------
  if (N <= 1) {
    // Move axis back to normAxis and return. Reverse is a no-op for N<=1.
    const result = working.map((a, i) => {
      const back = moveaxis(a, 0, normAxis) as Array;
      if (workingOwned[i] && back !== a) a.dispose();
      return back;
    });
    return tree.unflatten(treedef, result) as T;
  }

  // ------------------------------------------------------------------
  // 5. Kogge-Stone parallel doubling.
  //
  //   For stride in [1, 2, 4, ...]:
  //     left  = current[0 : N-stride]   (view, no alloc)
  //     right = current[stride : N]     (view, no alloc)
  //     combined = fn(left, right)      (fn output: leaves we own)
  //     next  = concat(current[0:stride], combined)  (new alloc per leaf)
  //
  // All "left" and "right" slices are O(1) ShapeTracker views.
  // `fn` is called with pytrees of shape [N-stride, ...rest].
  // `fn`'s output leaves are the only allocations per round.
  // ------------------------------------------------------------------

  // `current[i]` is owned by us iff currentOwned[i] is true.
  let current: Array[] = working;
  let currentOwned: boolean[] = workingOwned;

  for (let stride = 1; stride < N; stride *= 2) {
    // Slice views (no allocation — ShapeTracker shrink).
    const flatLeft = current.map((a) => sliceAxis(a, 0, 0, N - stride));
    const flatRight = current.map((a) => sliceAxis(a, 0, stride, N));

    // Build pytrees for fn call.
    const leftTree = tree.unflatten(treedef, flatLeft) as T;
    const rightTree = tree.unflatten(treedef, flatRight) as T;

    // Call fn — fn may produce intermediates; those are fn's responsibility.
    // fn returns a pytree whose leaves we own.
    const combined = runStep(leftTree, rightTree) as T;

    // Dispose slice views — they are always fresh ShapeTracker views.
    for (const a of flatLeft) a.dispose();
    for (const a of flatRight) a.dispose();

    // Flatten fn output.
    const flatCombined = tree.flatten<Array>(combined)[0];

    // Build next = concat(current[0:stride], combined).
    // The prefix slice `current[0:stride]` is a view (no alloc).
    const next: Array[] = current.map((a, i) => {
      const prefix = sliceAxis(a, 0, 0, stride);
      // concat allocates a new Array — we own the result.
      const cat = core.concatenate([prefix, flatCombined[i]], 0) as Array;
      prefix.dispose();
      return cat;
    });

    // Dispose the fn-output leaves — they've been consumed into `next` via concat.
    for (const a of flatCombined) a.dispose();

    // Dispose the previous `current` arrays if we owned them.
    for (let i = 0; i < current.length; i++) {
      if (currentOwned[i]) current[i].dispose();
    }

    // All next entries are freshly allocated by concat.
    current = next;
    currentOwned = next.map(() => true);
  }

  // ------------------------------------------------------------------
  // 6. Post-process: reverse back if needed, then move axis back.
  // ------------------------------------------------------------------

  // Reverse back.
  let postReverse: Array[];
  let postReverseOwned: boolean[];
  if (reverse) {
    postReverse = current.map((a) => core.reverse(a, 0) as Array);
    // Dispose current (we own them all at this point).
    for (let i = 0; i < current.length; i++) {
      if (currentOwned[i]) current[i].dispose();
    }
    postReverseOwned = postReverse.map(() => true);
  } else {
    postReverse = current;
    postReverseOwned = currentOwned;
  }

  // Move axis from 0 back to normAxis.
  const result: Array[] = postReverse.map((a, i) => {
    const back = moveaxis(a, 0, normAxis) as Array;
    if (postReverseOwned[i] && back !== a) a.dispose();
    return back;
  });

  return tree.unflatten(treedef, result) as T;
}

export function associativeScan<T extends JsTree<Array>>(
  fn: (a: T, b: T) => T,
  elems: T,
  { axis = 0, reverse = false }: AssociativeScanOptions = {},
): T {
  // 1. Flatten and validate
  const [flatElems, treedef] = tree.flatten<Array>(elems);
  if (flatElems.length === 0) {
    throw new Error("associativeScan: elems must have at least one leaf array");
  }
  const ndim = flatElems[0].ndim;
  if (ndim === 0) {
    throw new Error("associativeScan: leaf arrays must be at least 1-D");
  }
  const normAxis = checkAxis(axis, ndim);
  const N = flatElems[0].shape[normAxis];
  const numLeaves = flatElems.length;

  // 2. Move scan axis to position 0
  const movedElems = flatElems.map((a) => moveaxis(a, normAxis, 0) as Array);
  const movedOwned = flatElems.map((a, i) => movedElems[i] !== a);

  // 3. Trivial cases (N ≤ 1): no body application needed
  if (N <= 1) {
    const result = movedElems.map((a, i) => {
      const back = moveaxis(a, 0, normAxis) as Array;
      if (movedOwned[i] && back !== a) a.dispose();
      return back;
    });
    return tree.unflatten(treedef, result) as T;
  }

  // 4. Trace fn into a body jaxpr (per-element shapes, scan axis removed).
  //    If tracing fails, fall back to the direct Kogge-Stone path via
  //    associativeScanCore.  Note: einsum patterns like "nij,njk->nik" work
  //    fine — einsumFastPath lowers them to matmul before subscript parsing.
  const elemAvals = movedElems.map(
    (e) => new ShapedArray(e.shape.slice(1) as number[], e.dtype, false),
  );

  let closedJaxpr;
  try {
    ({ jaxpr: closedJaxpr } = makeJaxpr((...args: any[]) => {
      const a = tree.unflatten(treedef, args.slice(0, numLeaves));
      const b = tree.unflatten(treedef, args.slice(numLeaves));
      const result = fn(a as T, b as T);
      return tree.flatten<Array>(result)[0];
    })(...elemAvals, ...elemAvals));
  } catch {
    // Body can't be traced per-element; fall back to direct Kogge-Stone.
    // This unrolls the algorithm into the current trace (jit/grad), which
    // produces a larger graph but works with compose fns that explicitly
    // reference the batch dimension.
    if (DEBUG >= 1) {
      console.warn(
        "[associativeScan] Per-element body tracing failed — falling back to " +
          "unrolled Kogge-Stone (slower on GPU).",
      );
    }
    for (let i = 0; i < movedElems.length; i++) {
      if (movedOwned[i]) movedElems[i].dispose();
    }
    return associativeScanCore(fn, elems, { axis, reverse }) as T;
  }

  // 4b. Validate body output shapes match input shapes.
  // Constants like reshape(eye, [1,m,m]) can broadcast per-element [m,m]
  // outputs to [1,m,m], which vmap then inflates to [batch,1,m,m] — shape
  // mismatch at the Kogge-Stone concatenation. Fall back when this happens.
  {
    const bodyOuts = closedJaxpr.jaxpr.outs;
    const shapeOk = bodyOuts.every((out, i) => {
      const expected = elemAvals[i % numLeaves].shape;
      const actual = out.aval.shape;
      return (
        actual.length === expected.length &&
        actual.every((d, j) => d === expected[j])
      );
    });
    if (!shapeOk) {
      closedJaxpr.dispose();
      for (let i = 0; i < movedElems.length; i++) {
        if (movedOwned[i]) movedElems[i].dispose();
      }
      return associativeScanCore(fn, elems, { axis, reverse }) as T;
    }
  }

  // 5. Canonicalize reverse: Reverse(AssocScan(Reverse(xs))).
  //    The primitive always sees reverse=false; reverse semantics live in
  //    surrounding Reverse nodes so the executor never needs reverse awareness.
  let scanInputs = movedElems;
  if (reverse) {
    scanInputs = movedElems.map((a) => core.reverse(a, 0) as Array);
  }

  /* eslint-disable jax-js/no-use-after-dispose */
  const rawResults = bind(
    Primitive.AssociativeScan,
    [...closedJaxpr.consts, ...scanInputs],
    {
      jaxpr: closedJaxpr.jaxpr,
      numLeaves,
      axis: 0,
      reverse: false,
    },
  );

  // 6. Cleanup
  closedJaxpr.dispose();
  /* eslint-enable jax-js/no-use-after-dispose */
  if (reverse) {
    for (const s of scanInputs) (s as Array).dispose();
  }
  for (let i = 0; i < movedElems.length; i++) {
    if (movedOwned[i]) movedElems[i].dispose();
  }

  // 6b. Reverse outputs back if needed
  const results = reverse
    ? (rawResults as Array[]).map((r) => {
        const rev = core.reverse(r, 0) as Array;
        (r as Array).dispose();
        return rev;
      })
    : rawResults;

  // 7. Move axis back if needed
  let finalResults = results as Array[];
  if (normAxis !== 0) {
    finalResults = (results as Array[]).map((r) => {
      const back = moveaxis(r, 0, normAxis) as Array;
      if (back !== r) (r as Array).dispose();
      return back;
    });
  }

  return tree.unflatten(treedef, finalResults) as T;
}

// Registered implementation for the AssociativeScan primitive.
// Called by array.ts (eager impl) and jit.ts (JIT execution).
//
// The body jaxpr was traced with per-element shapes (scan axis removed).
// associativeScanCore calls fn with batched sub-arrays (scan axis present).
// We bridge the gap using vmap: it vectorizes the element-level evalJaxpr
// over the batch dimension, so operations correctly handle the extra axis.
function associativeScanFromJaxpr(
  bodyJaxpr: Jaxpr,
  consts: Array[],
  elems: Array[],
  numLeaves: number,
  axis: number,
  reverse: boolean,
): Array[] {
  const inAxes = new globalThis.Array(numLeaves * 2).fill(0);

  const fn = (aLeaves: Array[], bLeaves: Array[]): Array[] => {
    // Vectorize the element-level body jaxpr over axis 0 (batch dim).
    const inner = (...args: Array[]): Array[] => {
      // evalJaxpr is non-consuming — consts and args stay alive.
      return evalJaxpr(bodyJaxpr, [...consts, ...args]) as Array[];
    };
    const allArgs = [...aLeaves, ...bLeaves];
    const vmapped = vmap(inner, inAxes);
    const result = vmapped(...allArgs);
    const resultArr = (
      globalThis.Array.isArray(result) ? result : [result]
    ) as Array[];
    return resultArr;
  };

  return associativeScanCore(fn, elems, { axis, reverse }) as Array[];
}

_registerAssociativeScanCoreImpl(associativeScanFromJaxpr);
