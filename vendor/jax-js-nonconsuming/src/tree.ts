// Utilities for working with tree-like container data structures ("pytrees").

import type { DataArray } from "./alu";
import type { Array } from "./frontend/array";
import { Tracer } from "./frontend/core";
import { deepEqual, unzip2 } from "./utils";

const JsArray = globalThis.Array;

export enum NodeType {
  Array = "Array",
  Object = "Object",
  Leaf = "Leaf",
  None = "None",
}

/** Analog to the JAX "pytree" object, but for JavaScript. */
export type JsTree<T> = T | JsTree<T>[] | { [key: string]: JsTree<T> };

type Same<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2
    ? true
    : false;

type MappedJsTree<T, A, B> = T extends A
  ? B
  : T extends Array // Special case: Do not recurse into np.Array
    ? T
    : T extends globalThis.Array<infer U>
      ? number extends T["length"]
        ? MapJsTree<U, A, B>[] // plain array
        : { [K in keyof T]: MapJsTree<T[K], A, B> } // tuple: map each slot, keep tuple shape
      : { [K in keyof T]: MapJsTree<T[K], A, B> }; // object: map each slot, keep object shape

/** @ignore Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure. */
export type MapJsTree<T, A, B> =
  Same<A, B> extends true ? T : MappedJsTree<T, A, B>;

/** Represents the structure of a JsTree. */
export class JsTreeDef {
  static leaf = new JsTreeDef(NodeType.Leaf, null, []);
  static none = new JsTreeDef(NodeType.None, null, []);

  constructor(
    readonly nodeType: NodeType,
    readonly nodeMetadata: any, // Must be comparable with deepEqual.
    readonly childTreedefs: JsTreeDef[],
  ) {}

  /** Get the total number of leaves in the tree. */
  get size(): number {
    if (this.nodeType === NodeType.None) return 0;
    return this.nodeType === NodeType.Leaf
      ? 1
      : this.childTreedefs.reduce((a, b) => a + b.size, 0);
  }

  /** Returns a string representation of this tree definition. */
  toString(root = true): string {
    if (root) {
      return "JsTreeDef(" + this.toString(false) + ")";
    }
    switch (this.nodeType) {
      case NodeType.Leaf:
        return "*";
      case NodeType.None:
        return "null";
      case NodeType.Array:
        return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
      case NodeType.Object: {
        const parts: string[] = [];
        for (let i = 0; i < this.childTreedefs.length; i++) {
          parts.push(
            `${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`,
          );
        }
        return `{${parts.join(", ")}}`;
      }
    }
  }

  /** Compare this tree definition with another. */
  equals(other: JsTreeDef): boolean {
    return (
      this.nodeType === other.nodeType &&
      deepEqual(this.nodeMetadata, other.nodeMetadata) &&
      this.childTreedefs.length === other.childTreedefs.length &&
      this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]))
    );
  }
}

function quoteObjectKey(key: string): string {
  // Check if the key is a valid JavaScript identifier
  if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)) {
    return key; // No need to quote
  }
  return JSON.stringify(key);
}

/** Flatten a structured object, returning the tree definition. */
export function flatten<T>(tree: JsTree<T>): [T[], JsTreeDef] {
  const leaves: T[] = [];
  const treedef = _flatten(tree, leaves);
  return [leaves, treedef];
}

function _flatten<T>(tree: JsTree<T>, leaves: T[]): JsTreeDef {
  // Handle null/undefined as empty node (like JAX's None)
  if (tree === null || tree === undefined) {
    return JsTreeDef.none;
  }
  if (JsArray.isArray(tree)) {
    const childTrees = tree.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Array, null, childTrees);
  } else if (
    typeof tree === "object" &&
    tree !== null &&
    tree.constructor === Object // Needed to avoid treating Array as an object.
  ) {
    const [keys, values] = unzip2(Object.entries(tree));
    const childTrees = values.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Object, keys, childTrees);
  } else {
    leaves.push(tree as T);
    return JsTreeDef.leaf;
  }
}

/** Get the leaves of a tree. */
export function leaves<T>(tree: JsTree<T>): T[] {
  return flatten<T>(tree)[0];
}

/** Get the treedef for a tree. */
export function structure<T>(tree: JsTree<T>): JsTreeDef {
  return flatten<T>(tree)[1];
}

/** Reconstruct a structured object from the flattened representation. */
export function unflatten<T>(
  treedef: JsTreeDef,
  leaves: Iterable<T>,
): JsTree<T> {
  return _unflatten(treedef, leaves[Symbol.iterator]());
}

function _unflatten<T>(treedef: JsTreeDef, leaves: Iterator<T>): JsTree<T> {
  switch (treedef.nodeType) {
    case NodeType.None:
      // None node type represents null/undefined - return null
      return null as unknown as JsTree<T>;
    case NodeType.Leaf: {
      const { value, done } = leaves.next();
      if (done) {
        throw new TypeError("Ran out of leaves while unflattening JsTree");
      }
      return value;
    }
    case NodeType.Array:
      return treedef.childTreedefs.map((c) => _unflatten(c, leaves));
    case NodeType.Object: {
      const obj: any = {};
      for (let i = 0; i < treedef.childTreedefs.length; i++) {
        obj[treedef.nodeMetadata[i]] = _unflatten(
          treedef.childTreedefs[i],
          leaves,
        );
      }
      return obj;
    }
  }
}

/** Maps a multi-input function over pytree args to produce a new pytree. */
export function map<T, U, Tree extends JsTree<T>>(
  fn: (...args: T[]) => U,
  tree: Tree,
  ...rest: Tree[]
): MapJsTree<Tree, T, U> {
  const [leaves, treedef] = flatten<T>(tree);
  const restLeaves = rest.map((x) => flatten<T>(x)[0]);
  const resultLeaves: U[] = [];
  for (let i = 0; i < leaves.length; i++) {
    resultLeaves.push(fn(leaves[i], ...restLeaves.map((x) => x[i])));
  }
  return unflatten(treedef, resultLeaves) as MapJsTree<Tree, T, U>;
}

/** Take a reference of every array in a tree. */
export function ref<Tree extends JsTree<any>>(tree: Tree): Tree {
  // jax-js-lint: allow-ref
  return map((x) => (x instanceof Tracer ? x.ref : x), tree) as unknown as Tree;
}

/** Dispose every array in a tree. */
export function dispose<Tree extends JsTree<any>>(
  tree: Tree | null | undefined,
): void {
  if (tree) {
    const seen = new Set<Tracer>();
    for (const x of leaves(tree as JsTree<any>)) {
      try {
        if (!(x instanceof Tracer)) continue;
        if (seen.has(x)) continue;
        seen.add(x);
        const refCount = (x as any).refCount;
        if (typeof refCount === "number" && refCount <= 0) continue;
        x.dispose();
      } catch (err) {
        if (
          !(err instanceof ReferenceError) ||
          !String(err.message).includes("has been disposed")
        ) {
          throw err;
        }
      }
    }
  }
}

/**
 * Make a plain object `Disposable`, so it works with `using`.
 *
 * Calling `[Symbol.dispose]()` on the result disposes every `Array` leaf
 * found in the object's own enumerable values (shallow, one level deep).
 *
 * Useful for scan/jit result pytrees:
 * ```ts
 * using result = tree.makeDisposable(await lax.scan(f, init, xs));
 * // result[0] is the carry, result[1] is stacked ys
 * // both are disposed at block end
 * ```
 */
export function makeDisposable<T extends object>(obj: T): T & Disposable {
  return Object.assign(obj, {
    [Symbol.dispose]() {
      dispose(obj as any);
    },
  });
}

/**
 * Read the data from every `Array` leaf in a pytree, **in parallel**.
 *
 * On WebGPU this is dramatically faster than reading each leaf sequentially
 * because all `GPUBuffer.mapAsync()` calls overlap (one GPU round-trip
 * instead of N).
 *
 * ```ts
 * const result = jit(f)(x);          // { a: Array, b: Array }
 * const data = await tree.data(result);  // { a: Float32Array, b: Float32Array }
 * ```
 */
export function data<Tree extends JsTree<any>>(
  tree: Tree,
): Promise<MapJsTree<Tree, Array, DataArray>> {
  const leafArr = leaves(tree as JsTree<any>);
  const promises = leafArr.map((x) =>
    x instanceof Tracer ? (x as unknown as Array).data() : Promise.resolve(x),
  );
  return Promise.all(promises).then(
    (resolvedLeaves) =>
      unflatten(structure(tree as JsTree<any>), resolvedLeaves) as MapJsTree<
        Tree,
        Array,
        DataArray
      >,
  );
}

/**
 * Read and dispose every `Array` leaf in a pytree, **in parallel**.
 *
 * Equivalent to `tree.data(t)` followed by `tree.dispose(t)` but issues
 * all GPU reads before disposing, maximising overlap.
 *
 * ```ts
 * const result = jit(f)(x);
 * const data = await tree.consumeData(result);
 * // all Array leaves are now disposed
 * ```
 */
export function consumeData<Tree extends JsTree<any>>(
  tree: Tree,
): Promise<MapJsTree<Tree, Array, DataArray>> {
  const leafArr = leaves(tree as JsTree<any>);
  const promises = leafArr.map((x) =>
    x instanceof Tracer ? (x as unknown as Array).data() : Promise.resolve(x),
  );
  return Promise.all(promises).then((resolvedLeaves) => {
    dispose(tree);
    return unflatten(
      structure(tree as JsTree<any>),
      resolvedLeaves,
    ) as MapJsTree<Tree, Array, DataArray>;
  });
}
