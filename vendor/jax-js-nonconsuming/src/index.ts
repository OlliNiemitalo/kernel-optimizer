import { DType } from "./alu";
import {
  _setCalibrationState,
  calibrateGpu,
  defaultDevice,
  Device,
  devices,
  getBackend,
  getWebGPUDevice,
  init,
  setCodeCapture,
} from "./backend";
import type {
  BackendCapabilities,
  CodeCaptureEntry,
  GpuTimingResult,
} from "./backend";
import { type Dim, hasSymbolicDims, isSymbolicDim, SymDim } from "./dim";
import { Array, ArrayLike } from "./frontend/array";
import { aotLinearize } from "./frontend/artifacts";
import type {
  AotLinearizeOptions,
  AotLinearizeResult,
  PrimalArtifact,
  PullbackArtifact,
  ResidualPack,
} from "./frontend/artifacts";
import { _fusedCacheCompileCount } from "./frontend/block-map-executor";
import {
  type CacheSizes,
  checkLeaks,
  clearCaches,
  getCacheSizes,
  type LeakReport,
  type SnapshotEntry,
} from "./frontend/check-leaks";
import { reverse, scatterAdd } from "./frontend/core";
import * as jaxprModule from "./frontend/jaxpr";
import {
  _setVerifyEffects,
  ClosedJaxpr,
  type EffectVerificationResult,
  Jaxpr,
  MemoryEffect,
  OwnedFunction,
  verifyJaxprEffects,
} from "./frontend/jaxpr";
import type { ConvClass, JitStepCounts } from "./frontend/jit";
import {
  _lastConvClass,
  _lastConvRewritten,
  _lastForiRewritten,
  jitCompile,
  JitProgram,
} from "./frontend/jit";
import * as jvpModule from "./frontend/jvp";
import * as linearizeModule from "./frontend/linearize";
import * as vmapModule from "./frontend/vmap";
import { captureJitReport, formatJitReport } from "./jit-capture";
import type {
  CapturedCode,
  CapturedKernel,
  CapturedProgram,
  CapturedRoutine,
  CapturedSubProgram,
  JitReport,
} from "./jit-capture";
import * as lax from "./library/lax";
import * as nn from "./library/nn";
import * as numpy from "./library/numpy";
import * as random from "./library/random";
import * as scipyLinalg from "./library/scipy-linalg";
import * as scipySpecial from "./library/scipy-special";
import type { PerformanceBeliefState } from "./microbench";
import { startTrace, stopTrace } from "./tracing";
import * as tree from "./tree";
import type { JsTree, JsTreeDef, MapJsTree } from "./tree";
import { setDebug } from "./utils";
import type { ScanPath } from "./utils";

import "./polyfills";

export {
  aotLinearize,
  type AotLinearizeOptions,
  type AotLinearizeResult,
  type BackendCapabilities,
  type CacheSizes,
  calibrateGpu,
  type CapturedCode,
  type CapturedKernel,
  type CapturedProgram,
  type CapturedRoutine,
  type CapturedSubProgram,
  captureJitReport,
  type CodeCaptureEntry,
  _setCalibrationState,
  type ConvClass,
  type Dim,
  type GpuTimingResult,
  init,
  Array,
  type ArrayLike,
  checkLeaks,
  clearCaches,
  ClosedJaxpr,
  defaultDevice,
  type Device,
  devices,
  DType,
  type EffectVerificationResult,
  _fusedCacheCompileCount,
  formatJitReport,
  getBackend,
  getCacheSizes,
  hasSymbolicDims,
  isSymbolicDim,
  Jaxpr,
  jitCompile,
  JitProgram,
  type JitReport,
  type JsTree,
  type JsTreeDef,
  type JitStepCounts,
  _lastConvClass,
  _lastConvRewritten,
  _lastForiRewritten,
  lax,
  type LeakReport,
  MemoryEffect,
  nn,
  numpy,
  type OwnedFunction,
  type PrimalArtifact,
  type PerformanceBeliefState,
  type PullbackArtifact,
  getWebGPUDevice,
  random,
  type ResidualPack,
  reverse,
  scatterAdd,
  setCodeCapture,
  _setVerifyEffects,
  setDebug,
  scipyLinalg,
  scipySpecial,
  type ScanPath,
  type SnapshotEntry,
  SymDim,
  tree,
  verifyJaxprEffects,
};

/** @namespace */
export const profiler = {
  startTrace,
  stopTrace,
};

/**
 * @function
 * Compute the forward-mode Jacobian-vector product for a function.
 */
export const jvp = jvpModule.jvp as <
  F extends (...args: any[]) => JsTree<Array>,
  HA extends boolean = false,
>(
  f: F,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>,
  opts?: { hasAux?: HA },
) => HA extends true
  ? ReturnType<F> extends [infer Out, infer Aux]
    ? [Out, Out, Aux]
    : never
  : [ReturnType<F>, ReturnType<F>];

export const _setJvpRetainedHandoffObserver =
  jvpModule._setJvpRetainedHandoffObserver;

/**
 * @function
 * Vectorize an operation on a batched axis for one or more inputs.
 */
export const vmap = vmapModule.vmap as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  inAxes?: number | (number | null | JsTree<number | null>)[],
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * @function
 * Compute the Jacobian evaluated column-by-column by forward-mode AD.
 */
export const jacfwd = vmapModule.jacfwd as <F extends (x: Array) => Array>(
  f: F,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * @function
 * Construct a Jaxpr by dynamically tracing a function with example inputs.
 */
export const makeJaxpr = jaxprModule.makeJaxpr as unknown as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
) => (...args: Parameters<F>) => {
  jaxpr: ClosedJaxpr;
  treedef: JsTreeDef;
};

/**
 * @function
 * Mark a function for automatic JIT compilation, with operator fusion.
 *
 * The function will be compiled the first time it is called with a set of
 * argument shapes.
 *
 * You can call `.dispose()` on the returned, JIT-compiled function after all
 * calls to free memory associated with array constants.
 *
 * **Options:**
 * - `staticArgnums`: An array of argument indices to treat as static
 *   (compile-time constant). These arguments must be hashable, won't be traced,
 *   and different values will trigger recompilation.
 * - `device`: The device to place the computation on. If not specified, the
 *   computation will be placed on the default device.
 */
export const jit = jaxprModule.jit as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  opts?: jaxprModule.JitOpts,
) => OwnedFunction<
  (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>
>;

/**
 * @function
 * Produce a local linear approximation to a function at a point using jvp() and
 * partial evaluation.
 */
export const linearize = linearizeModule.linearize as <
  F extends (...args: any[]) => JsTree<Array>,
  HA extends boolean = false,
>(
  f: F,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  opts?: { hasAux?: HA },
) => HA extends true
  ? ReturnType<F> extends [infer Out, infer Aux]
    ? [
        Out,
        OwnedFunction<
          (...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => Out
        >,
        Aux,
      ]
    : never
  : [
      ReturnType<F>,
      OwnedFunction<
        (
          ...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>
        ) => ReturnType<F>
      >,
    ];

/**
 * @function
 * Calculate the reverse-mode vector-Jacobian product for a function.
 *
 * The return value is a tuple of `[out, vjpFn]`, where `out` is the output of
 * `f(primals)`, and `vjpFn` is a function that takes in cotangents for each
 * output and returns the cotangents for each input.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return an
 * `[out, aux]` tuple, and `vjp` returns `[out, vjpFn, aux]`.
 *
 * @example
 * ```ts
 * const [y, vjpFn] = vjp(f, [x]);
 *
 * // With hasAux
 * const [y, vjpFn, aux] = vjp(f, [x], { hasAux: true });
 * ```
 */
export const vjp = linearizeModule.vjp as <
  F extends (...args: any[]) => JsTree<Array>,
  const HA extends boolean = false,
>(
  f: F,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  opts?: { hasAux?: HA },
) => HA extends true
  ? ReturnType<F> extends [infer Out, infer Aux]
    ? [
        Out,
        OwnedFunction<
          (
            cotangents: MapJsTree<Out, Array, ArrayLike>,
          ) => MapJsTree<Parameters<F>, ArrayLike, Array>
        >,
        Aux,
      ]
    : never
  : [
      ReturnType<F>,
      OwnedFunction<
        (
          cotangents: MapJsTree<ReturnType<F>, Array, ArrayLike>,
        ) => MapJsTree<Parameters<F>, ArrayLike, Array>
      >,
    ];

/** @inline */
type GradOutputType<I, F extends (...args: any[]) => any> = MapJsTree<
  I extends undefined
    ? Parameters<F>[0]
    : I extends number
      ? Parameters<F>[I]
      : I extends number[]
        ? { [K in keyof I]: I[K] extends number ? Parameters<F>[I[K]] : never }
        : never,
  ArrayLike,
  Array
>;

/**
 * @function
 * Compute the gradient of a scalar-valued function `f` with respect to its
 * first argument.
 *
 * Pass in different `argnums` to differentiate with respect to other
 * arguments. If a tuple is provided, the return value will be a tuple of
 * gradients corresponding to each argument index.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return a
 * `[out, aux]` tuple, and the return value will be `[gradient, aux]`.
 *
 * @example
 * ```ts
 * const gradient = grad(f)(x);
 *
 * // With `argnums`
 * const [gradientX, gradientZ] = grad(f, { argnums: [0, 2] })(x, y, z);
 *
 * // With `hasAux`
 * const [gradient, aux] = grad(f, { hasAux: true })(x);
 * ```
 */
export const grad = linearizeModule.grad as <
  F extends (...args: any[]) => JsTree<Array>,
  const I extends undefined | number | number[] = undefined,
  const HA extends boolean = false,
>(
  f: F,
  opts?: Omit<linearizeModule.GradOpts, "argnums" | "hasAux"> & {
    argnums?: I;
    hasAux?: HA;
  },
) => (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => HA extends true
  ? ReturnType<F> extends [any, infer Aux]
    ? [GradOutputType<I, F>, Aux]
    : never
  : GradOutputType<I, F>;

/**
 * @function
 * Create a function that evaluates both `f` and the gradient of `f`.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return an
 * `[out, aux]` tuple, and the return value will be `[[out, aux], gradient]`.
 *
 * @example
 * ```ts
 * // Without hasAux
 * const [value, gradient] = valueAndGrad(f)(x);
 *
 * // With hasAux
 * const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);
 * ```
 */
export const valueAndGrad = linearizeModule.valueAndGrad as <
  F extends (...args: any[]) => JsTree<Array>,
  const I extends undefined | number | number[] = undefined,
  const HA extends boolean = false,
>(
  f: F,
  opts?: Omit<linearizeModule.GradOpts, "argnums"> & {
    argnums?: I;
    hasAux?: HA;
  },
) => (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [ReturnType<F>, GradOutputType<I, F>];

/**
 * @function
 * Compute the Jacobian evaluated row-by-row by reverse-mode AD.
 */
export const jacrev = linearizeModule.jacrev as typeof jacfwd;

export { jacrev as jacobian };

/**
 * @function
 * Compute the Hessian matrix of a scalar-valued function.
 *
 * The Hessian is the matrix of second-order partial derivatives of a function.
 * This is implemented as `jacfwd(grad(f))`.
 *
 * @example
 * ```ts
 * const f = (x: np.Array) => np.sum(x.mul(x).mul(x)); // x^3
 * using x = np.array([1, 2, 3]);
 * using H = hessian(f)(x);
 * // H[i,j] = d^2f / dx_i dx_j
 * ```
 */
export const hessian = linearizeModule.hessian as <
  F extends (x: Array) => Array,
>(
  f: F,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * Create a function with a custom vector-Jacobian product (VJP) rule.
 *
 * When the returned function is differentiated via {@link vjp}, {@link grad},
 * or {@link valueAndGrad}, the supplied `bwd` function is invoked instead of
 * the standard autodiff backward pass.  This enables implicit
 * differentiation, numerically stable gradients, and memory-efficient
 * backpropagation.
 *
 * @param fwd  Forward function returning `[outputs, residuals]`.
 * @param bwd  Backward function `(residuals, cotangents) => inputCotangents`.
 *
 * @example
 * ```ts
 * const stableLog1pExp = customVjp(
 *   (x) => {
 *     const out = np.log(np.add(np.array(1), np.exp(x)));
 *     return [out, x];
 *   },
 *   (x, g) => np.multiply(g, np.subtract(np.array(1), np.exp(np.negative(x)))),
 * );
 *
 * const dx = grad(stableLog1pExp)(np.array(100.0)); // 1.0, not NaN
 * ```
 */
export const customVjp = linearizeModule.customVjp;

/**
 * Create a function with a custom Jacobian-vector product (JVP) rule.
 *
 * Defines how forward-mode differentiation behaves for a function.
 * Unlike {@link customVjp}, a custom JVP rule automatically provides
 * correct reverse-mode derivatives via transpose.
 *
 * @param fn       The primal function.
 * @param jvpRule  Custom JVP rule `(primals, tangents) => [primalOut, tangentOut]`.
 *
 * @example
 * ```ts
 * const stableLog1pExp = customJvp(
 *   (x) => np.log(np.add(np.array(1), np.exp(x))),
 *   ([x], [dx]) => {
 *     const y = np.log(np.add(np.array(1), np.exp(x)));
 *     const dy = np.multiply(dx, np.subtract(np.array(1), np.exp(np.negative(x))));
 *     return [y, dy];
 *   },
 * );
 * ```
 */
export const customJvp = linearizeModule.customJvp;

/**
 * Solve a linear system `matvec(x) = b` with implicit differentiation.
 *
 * Reverse-mode differentiation uses the transpose solve instead of
 * differentiating through the solver's internal iterations.
 *
 * @param matvec  Linear operator `A`: maps `x → Ax`.
 * @param b       Right-hand side.
 * @param solve   Forward solver `(matvec, b) → x`.
 * @param opts    Optional transpose solve and symmetry settings.
 */
export const customLinearSolve = linearizeModule.customLinearSolve;

export type { CustomLinearSolveOpts } from "./frontend/linearize";

/**
 * Solve a root-finding problem with implicit differentiation.
 *
 * Computes `x* = solve(f, x0)` such that `f(x*) ≈ 0`, and provides an
 * analytically correct VJP via the Implicit Function Theorem that avoids
 * differentiating through the solver's internal iterations.
 *
 * @param f  Optimality function: `f(x*) ≈ 0` at the solution.
 * @param initialGuess  Starting point for the solver.
 * @param solve  Forward solver: `solve(f, x0) → x*`.
 * @param tangentSolve  Linear solver: `tangentSolve(g, y) → x` s.t. `g(x) = y`.
 */
export const customRoot = linearizeModule.customRoot;

/**
 * Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
 *
 * This can be used to wait for the results of an intermediate computation to
 * finish. It's recommended to call this regularly in an iterative computation
 * to avoid queueing up too many pending operations.
 *
 * Does not consume reference to the arrays.
 */
export async function blockUntilReady<T extends JsTree<any>>(x: T): Promise<T> {
  const promises: Promise<Array>[] = [];
  for (const leaf of tree.leaves(x)) {
    if (leaf instanceof Array) {
      promises.push(leaf.blockUntilReady());
    }
  }
  await Promise.all(promises);
  return x;
}

/**
 * Batch GPU dispatches across multiple jit calls into a single
 * `queue.submit()`. Reduces per-submission driver overhead when
 * chaining many jit calls (e.g., multi-layer transformer inference).
 *
 * All jit calls within the callback share one command encoder.
 * The batched commands are submitted when the callback returns.
 * Call `blockUntilReady` **after** `withBatch` to wait for results.
 *
 * No-op on backends that don't support batching (CPU, WebGL).
 */
export function withBatch<T>(fn: () => T): T {
  const backend = getBackend();
  backend.beginBatch?.();
  try {
    return fn();
  } finally {
    backend.endBatch?.();
  }
}

/**
 * Transfer `x` to `device`.
 *
 * `x` may be a nested container of arrays or scalars. The resulting structure
 * is committed to the device.
 *
 * If `device` is not specified, this function behaves as identity if the input
 * is already an `Array`, otherwise it places the scalar uncommitted on the
 * default device.
 */
export async function devicePut<T extends JsTree<any>>(
  x: T,
  device?: Device,
): Promise<MapJsTree<T, number | boolean, Array>> {
  const [xflat, structure] = tree.flatten(x);
  const yflat = await Promise.all(
    xflat.map((leaf) => {
      if (leaf instanceof Array) {
        return device ? leaf._put(getBackend(device)) : Promise.resolve(leaf);
      } else {
        return Promise.resolve(numpy.array(leaf as any, { device }));
      }
    }),
  );
  return tree.unflatten(structure, yflat) as any;
}

/**
 * Profile GPU compute passes executed by `fn`.
 *
 * Requires the WebGPU backend with `timestamp-query` support.
 * Returns the function's result alongside per-pass GPU timing data.
 * Per-pass metadata (grid, workgroupSize, label) is included when available.
 */
export async function profileGpu<T>(
  fn: () => T | Promise<T>,
): Promise<{ result: T; timing: GpuTimingResult }> {
  const backend = getBackend();
  if (!backend.startProfiling || !backend.stopProfiling) {
    throw new Error(
      "profileGpu requires a backend with timestamp-query support (WebGPU)",
    );
  }
  backend.startProfiling();
  let result: T;
  try {
    result = await fn();
  } catch (e) {
    await backend.stopProfiling();
    throw e;
  }
  const timing = await backend.stopProfiling();
  return { result, timing };
}

/**
 * Alias for `profileGpu`. Per-pass metadata (grid dimensions, workgroup size,
 * shader label) is always included in the result when available.
 */
export const profileGpuDetailed = profileGpu;
