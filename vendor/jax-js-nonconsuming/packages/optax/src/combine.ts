import { tree } from "@hamk-uas/jax-js-nonconsuming";

import { GradientTransformation, OptState } from "./base";

/** Applies a list of chainable update transformations. */
export function chain(
  ...transforms: GradientTransformation[]
): GradientTransformation {
  const initFns = transforms.map((t) => t.init);
  const updateFns = transforms.map((t) => t.update);
  return {
    init(params) {
      const states = initFns.map((fn) => fn(params));
      return states;
    },
    update(updates, state, params) {
      state = state as OptState[];
      if (updateFns.length !== state.length) {
        throw new Error(
          `Expected ${updateFns.length} states, got ${state.length}. Make sure you called init first!`,
        );
      }
      const originalUpdates = updates;
      const newState: OptState[] = [];
      for (let i = 0; i < updateFns.length; i++) {
        const prevUpdates = updates;
        let newS: OptState;
        [updates, newS] = updateFns[i](
          updates,
          state[i],
          params ? params : undefined,
        );
        // Dispose intermediate updates from previous transform,
        // but never the caller's original updates.
        if (prevUpdates !== originalUpdates) {
          tree.dispose(prevUpdates);
        }
        newState.push(newS);
      }
      return [updates, newState];
    },
  };
}
