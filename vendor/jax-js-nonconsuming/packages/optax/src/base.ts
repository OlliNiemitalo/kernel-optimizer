import { JsTree, numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";

/** The internal state of the optimizer, varies in shape. */
export type OptState = JsTree<np.Array>;

/**
 * A pair of pure functions implementing a gradient transformation.
 *
 * Optimizers are implemented with this interface. They do not contain any
 * internal state. The "optimizer state" `OptState` is initialized, then passed
 * into each update call, which returns a new state.
 *
 * Gradients are transformed during the update call, and they should have the
 * same PyTree shape as the parameters.
 */
export interface GradientTransformation {
  init<Params extends JsTree<np.Array>>(params: Params): OptState;
  update<Params extends JsTree<np.Array>>(
    updates: Params,
    state: OptState,
    params?: Params,
  ): [Params, OptState];
}

/**
 * A learning rate or weight decay schedule function.
 *
 * Takes and returns `np.Array` scalars so the schedule body can be traced
 * by `jit()`. Write schedule logic using `np.*` array operations:
 *
 * ```ts
 * const schedule: Schedule = (count) => np.power(np.array(0.9), count);
 * ```
 *
 * @inline
 */
export type Schedule = (count: np.Array) => np.Array;

/** @inline */
export type ScalarOrSchedule = number | Schedule;

/** Simplest possible state for a transformation. */
export function initEmptyState(_params: JsTree<np.Array>) {
  return [];
}

/** Stateless identity transformation that leaves input gradients untouched. */
export function identity(): GradientTransformation {
  return {
    init: initEmptyState,
    update<Params extends JsTree<np.Array>>(
      updates: Params,
      state: OptState,
      _params?: Params,
    ): [Params, OptState] {
      return [updates, state];
    },
  };
}

/** Stateless transformation that maps input gradients to zero. */
export function setToZero(): GradientTransformation {
  return {
    init: initEmptyState,
    update<Params extends JsTree<np.Array>>(
      updates: Params,
      state: OptState,
      _params?: Params,
    ): [Params, OptState] {
      const zeros = tree.map((g: np.Array) => np.zerosLike(g), updates);
      return [zeros, state];
    },
  };
}

/**
 * Applies an update to the corresponding parameters.
 *
 * This function is provided for convenience, as it just adds the updates to the
 * parameters directly. You can get updates from a `GradientTransformation`.
 */
export function applyUpdates<Params extends JsTree<np.Array>>(
  params: Params,
  updates: Params,
): Params {
  return tree.map((p: np.Array, u: np.Array) => p.add(u), params, updates);
}
