import { byteWidth, DType, isFloatDtype } from "../alu";
import { defaultDevice } from "../backend";
import { PPrint } from "../pprint";
import {
  concreteDim,
  type Dim,
  dimEquals,
  type Pair,
  resolveShape,
  SymDim,
} from "../shape";
import {
  JsTreeDef,
  MapJsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import {
  DEBUG,
  deepEqual,
  FpHash,
  FpHashable,
  generalBroadcast,
  runWithCache,
  unzip2,
  zip,
} from "../utils";
import { array, Array, ArrayLike, pureArray } from "./array";
import { _jitFunctionDisposers } from "./check-leaks";
import { checkConvShape, checkPoolShape } from "./convolution";
import {
  _setInMakeJaxprBody,
  AbstractValue,
  bind,
  flattenFun,
  fullRaise,
  getAval,
  inMakeJaxprBody,
  ndim,
  newDynamic,
  newMain,
  Primitive,
  PrimitiveParams,
  promoteAvals,
  propagateCustomDerivativeMetadata,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  UseAfterFreeError,
} from "./core";

/**
 * Memory effects annotating Jaxpr variables and equations.
 *
 * - `Alloc`: Creates a new buffer (output of a primitive).
 * - `Borrow`: Reads a buffer without taking ownership.
 * - `Consume`: Takes ownership; buffer cannot be used again.
 * - `Mutate`: In-place modification (requires exclusive ownership).
 */
export enum MemoryEffect {
  Alloc = "Alloc",
  Borrow = "Borrow",
  Consume = "Consume",
  Mutate = "Mutate",
}

/**
 * Per-primitive input effect overrides.
 *
 * Returns an array of effects for the primitive's inputs, or `undefined` to
 * use the default (all Borrow). Keyed by `Primitive` string value.
 */
const primitiveInputEffects: Partial<
  Record<Primitive, (nInputs: number) => MemoryEffect[]>
> = {
  // DynamicUpdateSlice mutates its first input (dst) in place.
  [Primitive.DynamicUpdateSlice]: (n: number) => {
    const effects = globalThis.Array.from(
      { length: n },
      () => MemoryEffect.Borrow,
    );
    effects[0] = MemoryEffect.Mutate;
    return effects;
  },
  // DynamicUpdateSliceGeneral mutates its first input (dst) in place.
  [Primitive.DynamicUpdateSliceGeneral]: (n: number) => {
    const effects = globalThis.Array.from(
      { length: n },
      () => MemoryEffect.Borrow,
    );
    effects[0] = MemoryEffect.Mutate;
    return effects;
  },
  // ScatterAdd mutates its first input (target) in place; indices + updates are Borrow.
  [Primitive.ScatterAdd]: (n: number) => {
    const effects = globalThis.Array.from(
      { length: n },
      () => MemoryEffect.Borrow,
    );
    effects[0] = MemoryEffect.Mutate;
    return effects;
  },
};

/** Result of verifying memory effects on a Jaxpr. */
export interface EffectVerificationResult {
  /** Whether the Jaxpr is effect-safe. */
  ok: boolean;
  /** Human-readable error messages for each violation. */
  errors: string[];
}

/**
 * Static borrow checker for Jaxpr memory effects.
 *
 * Walks the Jaxpr equations and enforces ownership rules:
 * 1. No `Borrow` or `Mutate` after `Consume` — once consumed, a var is dead.
 * 2. `Mutate` requires exclusive ownership — the same var cannot appear as
 *    both `Mutate` and `Borrow` in the same equation's inputs.
 * 3. All `Alloc` vars must be either `Consume`d, returned as a Jaxpr output,
 *    or unreferenced (dead code, acceptable).
 *
 * Equations without effect annotations are silently skipped.
 */
export function verifyJaxprEffects(jaxpr: Jaxpr): EffectVerificationResult {
  const errors: string[] = [];

  // Set of output Vars — these are "returned" and don't need to be consumed.
  const outputVars = new Set<Var>();
  for (const out of jaxpr.outs) {
    if (out instanceof Var) outputVars.add(out);
  }

  // Track per-Var state: "alive" (usable) or "consumed" (dead).
  // Jaxpr inBinders start alive. Equation outBinders start alive when created.
  const consumed = new Set<Var>();

  // Track which Vars were allocated (Alloc effect on output) and whether they
  // were subsequently consumed or returned.
  const allocatedVars = new Set<Var>();

  for (const eqn of jaxpr.eqns) {
    if (!eqn.inputEffects || !eqn.outputEffects) continue;

    // Collect per-equation Mutate vars to check exclusivity
    const mutateVarsInEqn = new Set<Var>();

    // Check each input
    for (let i = 0; i < eqn.inputs.length; i++) {
      const atom = eqn.inputs[i];
      if (!(atom instanceof Var)) continue;
      const effect = eqn.inputEffects[i];
      if (!effect) continue;

      // Rule 1: No use after Consume
      if (consumed.has(atom)) {
        errors.push(
          `Use-after-consume: var %${atom.id} is used with ${effect} ` +
            `in "${eqn.primitive}" but was already consumed`,
        );
        continue;
      }

      if (effect === MemoryEffect.Consume) {
        consumed.add(atom);
      } else if (effect === MemoryEffect.Mutate) {
        mutateVarsInEqn.add(atom);
      }
    }

    // Rule 2: Mutate exclusivity — check that no Mutate var appears as Borrow
    // in the same equation
    if (mutateVarsInEqn.size > 0) {
      for (let i = 0; i < eqn.inputs.length; i++) {
        const atom = eqn.inputs[i];
        if (!(atom instanceof Var)) continue;
        const effect = eqn.inputEffects[i];
        if (effect === MemoryEffect.Borrow && mutateVarsInEqn.has(atom)) {
          errors.push(
            `Mutate exclusivity violation: var %${atom.id} is both ` +
              `Mutate and Borrow in "${eqn.primitive}"`,
          );
        }
      }
    }

    // Track Alloc outputs
    for (let i = 0; i < eqn.outBinders.length; i++) {
      const effect = eqn.outputEffects[i];
      if (effect === MemoryEffect.Alloc) {
        allocatedVars.add(eqn.outBinders[i]);
      }
    }
  }

  // Rule 3: All Alloc vars must be consumed, returned, or referenced.
  // In the non-consuming model, intermediates are Borrowed (not Consumed)
  // by subsequent equations — this is valid. Only dead allocations (never
  // referenced by any equation and not returned) are flagged.
  for (const v of allocatedVars) {
    if (consumed.has(v) || outputVars.has(v)) continue;
    const isReferenced = jaxpr.eqns.some((e) =>
      e.inputs.some((a) => a instanceof Var && a === v),
    );
    if (!isReferenced) {
      errors.push(
        `Dead allocation: var %${v.id} (${v.aval}) is allocated but ` +
          `never referenced, consumed, or returned`,
      );
    }
  }

  return { ok: errors.length === 0, errors };
}

/** Whether to run effect verification at the end of makeJaxpr. */
let _verifyEffectsEnabled = false;

/**
 * Enable or disable automatic effect verification in `makeJaxpr`.
 * When enabled, `verifyJaxprEffects` runs after every `makeJaxpr` call
 * and throws on violations. Gated behind this flag for performance.
 */
export function _setVerifyEffects(enabled: boolean): void {
  _verifyEffectsEnabled = enabled;
}

/**
 * Function callback with an associated dispose() method.
 *
 * The dispose() method should be called to clean up any tracer resources needed
 * by the function after the last time it is called.
 */
// eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
export type OwnedFunction<F extends Function> = F & {
  dispose: () => void;
  [Symbol.dispose]: () => void;
};

/** Variable in a Jaxpr expression. */
export class Var {
  static #nextId = 1; // For debugging, since JavaScript has no id() function like Python.

  readonly id: number;
  readonly aval: ShapedArray;
  /** Memory effect annotation (populated by M1 tracing). */
  effect?: MemoryEffect;

  constructor(aval: ShapedArray, effect?: MemoryEffect) {
    this.id = Var.#nextId++;
    this.aval = aval;
    this.effect = effect;
  }

  toString(): string {
    return `Var(${this.id}):${this.aval.toString()}`;
  }
}

/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
export class Lit {
  readonly value: number;
  readonly aval: ShapedArray;

  get dtype(): DType {
    return this.aval.dtype;
  }

  constructor(aval: AbstractValue, value: number) {
    if (aval.shape.length !== 0) {
      throw new Error(`internal: Lit must be a scalar`);
    }
    this.value = value;
    this.aval = ShapedArray.fromAval(aval);
  }
}

export type Atom = Var | Lit;

function atomIsLit(
  atom: Atom,
  literal?: number | boolean,
): atom is Lit & boolean {
  return (
    atom instanceof Lit && (literal === undefined || atom.value === literal)
  );
}

class VarPrinter {
  names: Map<Var, string> = new Map();
  #next = "a";

  // a, b, c, ..., z, aa, ab, ..., az, ba, bb, ...
  #advance() {
    const ret = this.#next;
    let lastNonz = this.#next.length - 1;
    while (lastNonz >= 0 && this.#next[lastNonz] === "z") {
      lastNonz--;
    }
    if (lastNonz < 0) {
      this.#next = "a".repeat(this.#next.length + 1);
    } else {
      let result = this.#next.slice(0, lastNonz);
      result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
      result += "a".repeat(this.#next.length - 1 - lastNonz);
      this.#next = result;
    }
    return ret;
  }

  name(v: Var): string {
    if (this.names.has(v)) {
      return this.names.get(v)!;
    }
    const name = this.#advance();
    this.names.set(v, name);
    return name;
  }

  nameType(v: Var): string {
    const effectSuffix = v.effect ? `{${v.effect}}` : "";
    return `${this.name(v)}:${v.aval.toString()}${effectSuffix}`;
  }
}

/** A single statement / binding in a Jaxpr, in ANF form. */
export class JaxprEqn {
  /** Per-input memory effects (same length as `inputs`). Populated by M1 tracing. */
  inputEffects?: MemoryEffect[];
  /** Per-output memory effects (same length as `outBinders`). Populated by M1 tracing. */
  outputEffects?: MemoryEffect[];

  constructor(
    readonly primitive: Primitive,
    readonly inputs: Atom[],
    readonly params: Record<string, any>,
    readonly outBinders: Var[],
  ) {}

  /** Copy effect annotations from another equation (e.g., during simplify/flatten). */
  copyEffectsFrom(other: JaxprEqn): this {
    if (other.inputEffects) this.inputEffects = [...other.inputEffects];
    if (other.outputEffects) this.outputEffects = [...other.outputEffects];
    return this;
  }

  pprint(usedVars?: Set<Var>, vp = new VarPrinter()): PPrint {
    const lhs = PPrint.pp(
      this.outBinders
        .map((v) => (!usedVars || usedVars.has(v) ? vp.nameType(v) : "_"))
        .join(" "),
    );
    let rhs = PPrint.pp(this.primitive);
    // pprint params
    const paramsList = Object.entries(this.params)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => PPrint.pp(`${k}=${v}`));
    if (paramsList.length > 0) {
      rhs = rhs
        .stack(PPrint.pp(" [ "))
        .stack(PPrint.prototype.concat(...paramsList))
        .stack(PPrint.pp(" ] "));
    } else {
      rhs = rhs.stack(PPrint.pp(" "));
    }
    // pprint inputs (vars and literals)
    rhs = rhs.stack(
      PPrint.pp(
        this.inputs
          .map((x) => (x instanceof Var ? vp.name(x) : String(x.value)))
          .join(" "),
      ),
    );
    // pprint effect annotations when present
    if (this.inputEffects || this.outputEffects) {
      const parts: string[] = [];
      if (this.inputEffects) {
        parts.push(`in[${this.inputEffects.join(",")}]`);
      }
      if (this.outputEffects) {
        parts.push(`out[${this.outputEffects.join(",")}]`);
      }
      rhs = rhs.stack(PPrint.pp(` {${parts.join(" ")}}`));
    }
    return lhs.stack(PPrint.pp(" = ")).stack(rhs);
  }

  toString(): string {
    return this.pprint().toString();
  }
}

/**
 * JSON.stringify replacer that substitutes Jaxpr objects with their stable hash.
 * This ensures that `JSON.stringify(eqn.params, _jaxprHashReplacer)` produces
 * deterministic output across different tracings of the same function.
 */
function _jaxprHashReplacer(_key: string, value: unknown): unknown {
  if (value instanceof Jaxpr) return `jaxpr#${value.getHash()}`;
  return value;
}

/** Typed intermediate representation for traced computations. */
export class Jaxpr implements FpHashable {
  #hash?: bigint;

  constructor(
    readonly inBinders: Var[],
    readonly eqns: JaxprEqn[],
    readonly outs: Atom[],
  ) {}

  pprint(): PPrint {
    const vp = new VarPrinter();
    const usedVars = new Set<Var>(
      [...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter(
        (x) => x instanceof Var,
      ),
    );
    const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
    const eqns = PPrint.prototype.concat(
      ...this.eqns.map((e) => e.pprint(usedVars, vp)),
    );
    const outs = this.outs
      .map((x) => (x instanceof Var ? vp.name(x) : x.value))
      .join(", ");
    return PPrint.pp(`{ lambda ${inBinders} .`).concat(
      (this.eqns.length
        ? PPrint.pp("let ")
            .stack(eqns)
            .concat(PPrint.pp(`in ( ${outs} ) }`))
        : PPrint.pp(`( ${outs} ) }`)
      ).indent(2),
    );
  }

  toString(): string {
    return this.pprint().toString();
  }

  /**
   * Gets a hash of this Jaxpr.
   *
   * Var identity is not considered in the hash, so two Jaxprs with the same
   * order of assignments and operators but different variable IDs will resolve
   * to the same hash (and toString representation).
   */
  getHash(): bigint {
    if (this.#hash !== undefined) return this.#hash;
    const hasher = new FpHash();
    const varIds = new Map<Var, bigint>();
    const vi = (v: Var) => {
      if (varIds.has(v)) return varIds.get(v)!;
      const id = varIds.size + 1; // Start from 1, why not?
      varIds.set(
        v,
        FpHash.hash(
          id,
          v.aval.dtype,
          ...v.aval.shape.map((d) =>
            typeof d === "number" ? d : d.toString(),
          ),
        ),
      );
      return id;
    };
    hasher.update(this.inBinders.length);
    for (const x of this.inBinders) hasher.update(vi(x));
    hasher.update(this.eqns.length);
    for (const eqn of this.eqns) {
      hasher.update(eqn.primitive);
      hasher.update(eqn.inputs.length);
      for (const x of eqn.inputs)
        hasher.update(x instanceof Var ? vi(x) : x.value);
      hasher.update(JSON.stringify(eqn.params, _jaxprHashReplacer));
      hasher.update(eqn.outBinders.length);
      for (const x of eqn.outBinders) hasher.update(vi(x));
      // Include effect annotations in hash when present
      if (eqn.inputEffects) {
        for (const e of eqn.inputEffects) hasher.update(e);
      }
      if (eqn.outputEffects) {
        for (const e of eqn.outputEffects) hasher.update(e);
      }
    }
    hasher.update(this.outs.length);
    for (const x of this.outs)
      hasher.update(x instanceof Var ? vi(x) : x.value);
    return (this.#hash = hasher.value);
  }

  hash(state: FpHash): void {
    state.update(this.getHash());
  }

  /**
   * Produce a simplified Jaxpr with basic optimizations applied.
   *  - Trim away unused variables.
   *  - Fold away *1, *0, or +0 operations against literals.
   *  - Remove no-op movement operations.
   */
  simplify(): Jaxpr {
    const context = new Map<Var, Atom>();
    const newEqns: JaxprEqn[] = [];
    for (const e of this.eqns) {
      const inputs = e.inputs.map((x) =>
        x instanceof Var ? (context.get(x) ?? x) : x,
      );
      const eqn = new JaxprEqn(
        e.primitive,
        inputs,
        e.params,
        e.outBinders,
      ).copyEffectsFrom(e);

      if (eqn.primitive === Primitive.Add) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 0)) {
          context.set(c, b);
        } else if (atomIsLit(b, 0)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(
            c,
            new Lit(
              promoteAvals(a.aval, b.aval),
              a.dtype === DType.Bool
                ? Math.min(a.value + b.value, 1) // Special case: Bool ||
                : a.value + b.value,
            ),
          );
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Neg) {
        const [a] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a)) {
          context.set(c, new Lit(a.aval, -a.value));
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Mul) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        // TODO: handle *0 once we have shaped zero arrays
        if (atomIsLit(a, 1)) {
          context.set(c, b);
        } else if (atomIsLit(b, 1)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(
            c,
            new Lit(promoteAvals(a.aval, b.aval), a.value * b.value),
          );
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Idiv) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(b, 1) && !isFloatDtype(a.aval.dtype)) {
          context.set(c, a);
        } else {
          newEqns.push(eqn);
        }
      } else if (
        ((eqn.primitive === Primitive.Broadcast ||
          eqn.primitive === Primitive.Reshape) &&
          deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape)) ||
        (eqn.primitive === Primitive.Transpose &&
          (eqn.params.perm as number[]).every((p, i) => p === i)) ||
        (eqn.primitive === Primitive.Flip && eqn.params.axis.length === 0) ||
        (eqn.primitive === Primitive.Shrink &&
          (eqn.params.slice as Pair[]).every(
            ([s, e], i) => s === 0 && e === eqn.inputs[0].aval.shape[i],
          )) ||
        (eqn.primitive === Primitive.Pad &&
          (eqn.params.width as Pair[]).every(
            ([w0, w1]) => w0 === 0 && w1 === 0,
          ))
      ) {
        // No-op movement operation, just pass through the input.
        context.set(eqn.outBinders[0], eqn.inputs[0]);
      } else {
        newEqns.push(eqn);
      }
    }

    const outs = this.outs.map((x) =>
      x instanceof Var ? (context.get(x) ?? x) : x,
    );

    // Skip unused output variables
    const usedVars = new Set<Var>(outs.filter((x) => x instanceof Var));
    const liveEqns: JaxprEqn[] = [];
    for (let i = newEqns.length - 1; i >= 0; i--) {
      const eqn = newEqns[i];
      if (eqn.outBinders.some((v) => usedVars.has(v))) {
        liveEqns.push(eqn);
        for (const v of eqn.inputs) {
          if (v instanceof Var) {
            usedVars.add(v);
          }
        }
      }
    }

    return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
  }

  /** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
  flatten(): Jaxpr {
    if (!this.eqns.some((eqn) => eqn.primitive === Primitive.Jit)) {
      // Fast path: no Jit to flatten.
      return this;
    }
    // Otherwise, we need to flatten this Jaxpr.
    const newEqns: JaxprEqn[] = [];
    const varMap = new Map<Var, Atom>(); // outBinders from Jit are replaced with new values
    const varMapF = (x: Atom) => (x instanceof Var ? (varMap.get(x) ?? x) : x);
    for (const eqn of this.eqns) {
      if (eqn.primitive === Primitive.Jit) {
        // First, flatten the Jaxpr recursively.
        const jaxpr = (eqn.params.jaxpr as Jaxpr).flatten();
        // Make a mapping of this Jaxpr's variables to translated values.
        const translation = new Map<Var, Atom>();
        const translationF = (x: Atom) =>
          x instanceof Var ? translation.get(x)! : x;
        for (const [v, x] of zip(jaxpr.inBinders, eqn.inputs)) {
          translation.set(v, varMapF(x));
        }
        for (const ieqn of jaxpr.eqns) {
          const inputs = ieqn.inputs.map(translationF);
          const outBinders: Var[] = [];
          for (const v of ieqn.outBinders) {
            const u = new Var(v.aval);
            outBinders.push(u);
            translation.set(v, u);
          }
          newEqns.push(
            new JaxprEqn(
              ieqn.primitive,
              inputs,
              ieqn.params,
              outBinders,
            ).copyEffectsFrom(ieqn),
          );
        }
        // Add the outputs to the mapping.
        for (const [v, x] of zip(eqn.outBinders, jaxpr.outs)) {
          varMap.set(v, translationF(x));
        }
      } else {
        if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) {
          // Replace any input variables if needed.
          newEqns.push(
            new JaxprEqn(
              eqn.primitive,
              eqn.inputs.map(varMapF),
              eqn.params,
              eqn.outBinders,
            ).copyEffectsFrom(eqn),
          );
        } else {
          newEqns.push(eqn);
        }
      }
    }
    // Replace the output variables if needed.
    const newOuts = this.outs.map(varMapF);
    return new Jaxpr(this.inBinders, newEqns, newOuts);
  }

  /**
   * Resolve symbolic dimensions to concrete values.
   * Returns a new Jaxpr with all SymDim instances replaced by numbers.
   */
  resolveDims(bindings: ReadonlyMap<string, number>): Jaxpr {
    const resolveAval = (aval: ShapedArray): ShapedArray =>
      new ShapedArray(
        resolveShape(aval.shape, bindings),
        aval.dtype,
        aval.weakType,
      );

    const varMap = new Map<Var, Var>();
    const resolveVar = (v: Var): Var => {
      let resolved = varMap.get(v);
      if (!resolved) {
        resolved = new Var(resolveAval(v.aval), v.effect);
        varMap.set(v, resolved);
      }
      return resolved;
    };
    const resolveAtom = (a: Atom): Atom =>
      a instanceof Var ? resolveVar(a) : a;

    const newIn = this.inBinders.map(resolveVar);
    const newEqns = this.eqns.map((eqn) =>
      new JaxprEqn(
        eqn.primitive,
        eqn.inputs.map(resolveAtom),
        eqn.params,
        eqn.outBinders.map(resolveVar),
      ).copyEffectsFrom(eqn),
    );
    const newOuts = this.outs.map(resolveAtom);
    return new Jaxpr(newIn, newEqns, newOuts);
  }
}

export class JaxprType {
  constructor(
    readonly inTypes: ShapedArray[],
    readonly outTypes: ShapedArray[],
  ) {}

  toString(): string {
    const inTypes = this.inTypes.map((aval) => aval.toString()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.toString()).join(", ");
    return `(${inTypes}) -> (${outTypes})`;
  }
}

export function typecheckJaxpr(jaxpr: Jaxpr): JaxprType {
  const env = new Set<Var>();

  for (const v of jaxpr.inBinders) {
    if (env.has(v)) {
      throw new TypeError(`Duplicate variable binding: ${v}`);
    }
    env.add(v);
  }

  for (const eqn of jaxpr.eqns) {
    const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
    const rule = abstractEvalRules[eqn.primitive];
    const outTypes = rule(inTypes, eqn.params as any);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes)) {
      if (!outType.equals(outBinder.aval)) {
        throw new TypeError(
          `Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`,
        );
      }
      if (env.has(outBinder)) {
        throw new TypeError(`Duplicate variable binding: ${outBinder}`);
      }
      env.add(outBinder);
    }
  }

  const inTypes = jaxpr.inBinders.map((v) => v.aval);
  const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
  return new JaxprType(inTypes, outTypes);
}

function typecheckAtom(env: Set<Var>, x: Atom): ShapedArray {
  if (x instanceof Var) {
    if (!env.has(x)) {
      throw new Error(`Unknown variable: ${x}`);
    }
    return x.aval;
  } else if (x instanceof Lit) {
    return x.aval;
  } else {
    throw new TypeError(`Invalid atom type: ${x}`);
  }
}

/** Evaluate a Jaxpr on an array of inputs. */
export function evalJaxpr(jaxpr: Jaxpr, args: Tracer[]): Tracer[] {
  const env = new Map<Var, Tracer>();

  // Number of usages of each variable, in an eqn or the output.
  const usageCount = new Map<Var, number>();
  for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) {
    if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
  }

  // Variables that appear in the outputs — don't dispose these early.
  const outputVars = new Set<Var>();
  for (const x of jaxpr.outs) {
    if (x instanceof Var) outputVars.add(x);
  }

  // Input variables (consts + args) — don't dispose at last use.
  // Non-consuming model: caller owns the inputs.  Only intermediates
  // (equation outputs) are auto-disposed by evalJaxpr.
  const inputVars = new Set<Var>(jaxpr.inBinders);

  const remainingRefs = new Map<Var, number>();

  // Track arrays created from Lit values for disposal (they have no owning Var).
  const litArrays: Tracer[] = [];

  // Track how many live Var bindings reference each value by identity.
  // Identity-returning primitives (stopGradient, convert_element_type when
  // types match) cause multiple Vars to share the same Array.  consumeRead
  // must not dispose a value while other Vars still reference it.
  const valueBindCount = new Map<Tracer, number>();

  const read = (x: Atom) => {
    if (x instanceof Var) {
      return env.get(x)!;
    } else {
      // Lit values are materialized as fresh arrays. They are tracked in
      // litArrays and disposed at the end of evalJaxpr (unless they appear
      // in the result set, in which case the caller owns them).
      const arr = array(x.value, { dtype: x.dtype });
      litArrays.push(arr);
      return arr;
    }
  };

  const write = (v: Var, val: Tracer) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    const refCount = usageCount.get(v) ?? 0;
    if (refCount || inputVars.has(v)) {
      env.set(v, val);
      remainingRefs.set(v, refCount);
      valueBindCount.set(val, (valueBindCount.get(val) ?? 0) + 1);
    } else {
      val.dispose(); // If variable not used, dispose immediately.
    }
  };

  // Decrement remaining count for a variable; dispose if last use and not output/input.
  const consumeRead = (x: Atom) => {
    if (x instanceof Var) {
      const left = remainingRefs.get(x)!;
      remainingRefs.set(x, left - 1);
      if (left === 1 && !outputVars.has(x) && !inputVars.has(x)) {
        const val = env.get(x);
        env.delete(x);
        if (val !== undefined) {
          const binds = valueBindCount.get(val)! - 1;
          valueBindCount.set(val, binds);
          // Only dispose when no other Var bindings reference this value.
          if (binds === 0) val.dispose();
        }
      }
    }
  };

  try {
    for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
    for (const eqn of jaxpr.eqns) {
      const inVals = eqn.inputs.map(read);
      const outVals = bind(eqn.primitive, inVals, eqn.params);
      for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
      // Dispose variables after their last use in equations.
      for (const x of eqn.inputs) consumeRead(x);
    }
    const results = jaxpr.outs.map(read);
    // .ref pass-through outputs (outputs that ARE input arrays).
    // Without this, when callers dispose their inputs after evalJaxpr,
    // they'd also kill the output — since it's the same Array object.
    const inputSet = new Set<Tracer>(args);
    for (const r of results) {
      if (inputSet.has(r)) r.ref; // jax-js-lint: allow-ref
    }
    // Dispose Lit-created arrays that aren't in the output.
    const resultSet = new Set(results);
    for (const a of litArrays) {
      if (!resultSet.has(a)) a.dispose();
    }
    return results;
  } catch (error) {
    // Clean up any remaining intermediates on error, to avoid leaking memory.
    // Skip input vars — caller owns those.  Use a disposed set to avoid
    // double-free when identity-sharing makes two Vars reference one value.
    const disposed = new Set<Tracer>();
    for (const [v, val] of env.entries()) {
      if (!inputVars.has(v) && !disposed.has(val)) {
        disposed.add(val);
        val.dispose();
      }
    }
    for (const a of litArrays) {
      if (!disposed.has(a)) a.dispose();
    }
    throw error;
  }
}

/** Convert a Jaxpr to a callable function by evaluating it. */
export function jaxprAsFun(jaxpr: Jaxpr): (...args: Tracer[]) => Tracer[] {
  return (...args: Tracer[]) => evalJaxpr(jaxpr, args);
}

export type ClosedJaxprConstOwnership = "borrowed" | "owned";

/** Jaxpr with a collection of associated, traced constants. */
export class ClosedJaxpr {
  constructor(
    readonly jaxpr: Jaxpr,
    readonly consts: Tracer[],
    readonly constOwnership: ClosedJaxprConstOwnership = "borrowed",
  ) {}

  static borrowed(jaxpr: Jaxpr, consts: Tracer[]): ClosedJaxpr {
    return new ClosedJaxpr(jaxpr, consts, "borrowed");
  }

  static owned(jaxpr: Jaxpr, consts: Tracer[]): ClosedJaxpr {
    return new ClosedJaxpr(jaxpr, consts, "owned");
  }

  /** String representation of this Jaxpr. */
  toString(): string {
    return this.jaxpr.toString();
  }

  /** Apply a function to the underlying Jaxpr. */
  mapJaxpr(f: (jaxpr: Jaxpr) => Jaxpr): ClosedJaxpr {
    return new ClosedJaxpr(f(this.jaxpr), this.consts, this.constOwnership);
  }

  /** Dispose of the constants in this Jaxpr. */
  dispose() {
    if (this.constOwnership === "owned") {
      for (const c of this.consts) c.dispose();
      return;
    }
    for (const c of this.consts) {
      try {
        c.dispose();
      } catch (error) {
        if (!(error instanceof UseAfterFreeError)) throw error;
        // Borrowed ClosedJaxprs are transient tracing/build wrappers, so
        // callers can legitimately hand them external const Arrays that have
        // already been freed elsewhere. Continue releasing later const refs
        // instead of bailing out on the first already-freed value.
      }
    }
  }
}

// Cache-owned const environments are strict: each retained handle here is an
// independent owner and teardown should fail loudly. Borrowed ClosedJaxpr
// wrappers above keep the only remaining tolerant transient teardown path.
export class OwnedConstEnv {
  readonly values: Tracer[];

  private constructor(values: Tracer[]) {
    this.values = values;
  }

  static retain(values: Tracer[]): OwnedConstEnv {
    return new OwnedConstEnv(
      values.map((c) => {
        if (c instanceof Array) return c.retainedHandle;
        // jax-js-lint: allow-ref
        return c.ref;
      }),
    );
  }

  get length(): number {
    return this.values.length;
  }

  dispose(): void {
    for (const c of this.values) c.dispose();
  }
}

export interface CachedJaxprArtifact {
  jaxpr: Jaxpr;
  constEnv: OwnedConstEnv;
}

export function buildCachedJaxprArtifact<T extends { jaxpr: ClosedJaxpr }>(
  traced: T,
): Omit<T, "jaxpr"> & CachedJaxprArtifact {
  const constEnv = OwnedConstEnv.retain(traced.jaxpr.consts);
  const jaxpr = traced.jaxpr.jaxpr;
  traced.jaxpr.dispose();
  const { jaxpr: _closedJaxpr, ...rest } = traced;
  return { ...rest, jaxpr, constEnv };
}

/** Tracer that records its operations to dynamically construct a Jaxpr. */
class JaxprTracer extends Tracer {
  #rc: number;

  constructor(
    trace: Trace,
    readonly aval: ShapedArray,
  ) {
    super(trace);
    this.#rc = 1;
  }

  toString(): string {
    return `JaxprTracer(${this.aval.toString()})`;
  }

  get ref() {
    this.#rc++;
    return this;
  }
  dispose() {
    this.#rc--;
  }
  /** Number of live references the user holds (1 + .ref count − .dispose count). */
  get refCount(): number {
    return this.#rc;
  }

  override isAliveForCleanup(): boolean {
    return this.#rc > 0;
  }

  // JaxprTracer can be created from a constant; if the constant is lifted
  // multiple times we need to increment the reference count each time. We can't
  // use `.ref` for this as that might raise a `UseAfterFreeError` when rc=0.
  trackLiftedConstant() {
    this.#rc++;
  }
}

/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
class JaxprTrace extends Trace {
  /** Register a Jaxpr argument with a given shape and return the tracer. */
  newArg(aval: ShapedArray): JaxprTracer {
    aval = ShapedArray.fromAval(aval);
    const tracer = this.builder.newTracer(this, aval);
    this.builder.addVar(tracer);
    return tracer;
  }

  /** Register a constant / literal in this Jaxpr. */
  getOrMakeConstTracer(val: TracerValue): JaxprTracer {
    // If val is a raw value (number/boolean), pureArray creates a fresh Array
    // that nobody else references. The creation ref must be balanced here
    // because no external owner exists.
    const wasRaw = !(val instanceof Tracer);
    if (wasRaw) {
      val = pureArray(val);
    }
    // After pureArray, val is always a Tracer.
    const tval = val as Tracer;
    let tracer = this.builder.constTracers.get(tval);
    if (tracer === undefined) {
      tracer = this.builder.newTracer(
        this,
        ShapedArray.fromAval(getAval(tval)),
      );
      // Take an independent .ref for this builder.
      tval.ref; // jax-js-lint: allow-ref — builder retained handle
      // For raw values, the creation ref is unowned — balance it now that
      // the builder holds an independent retained handle.
      if (wasRaw) {
        tval.dispose();
        if (tval instanceof Array) tval.markCreationRefBalancedByMakeJaxpr();
      } else if (tval instanceof Array && tval.hasUnownedCreationRef) {
        // Non-raw JaxArray created during makeJaxpr body (e.g., fudgeArray(1),
        // np.array([3]) inside traced function). Its creation ref has no
        // external owner. Track it for balancing by makeJaxpr after build().
        this.builder.constsNeedingCreationRefBalance.add(tval);
      }
      this.builder.addConst(tracer, tval);
    } else {
      // Already captured by this builder. Raw values produce a fresh Array
      // every call, but identity-based cache lookup never hits for them,
      // so this branch is only reached for Tracer inputs.
      tracer.trackLiftedConstant();
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: JaxprTracer[],
    params: PrimitiveParams<P>,
  ): JaxprTracer[] {
    const avalsIn = tracers.map((t) => t.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const outTracers = avalsOut.map((aval) =>
      this.builder.newTracer(this, aval),
    );
    const eqn = new JaxprEqn(
      primitive,
      tracers.map((t) => this.builder.getVar(t)),
      params,
      outTracers.map((t) => this.builder.addVar(t)),
    );
    // Assign memory effects: per-primitive overrides, else default Borrow/Alloc
    eqn.inputEffects =
      primitiveInputEffects[primitive]?.(eqn.inputs.length) ??
      eqn.inputs.map(() => MemoryEffect.Borrow);
    eqn.outputEffects = eqn.outBinders.map(() => MemoryEffect.Alloc);
    // Also annotate output Vars
    for (const v of eqn.outBinders) {
      v.effect = MemoryEffect.Alloc;
    }
    this.builder.addEqn(eqn);
    return outTracers;
  }

  get builder(): JaxprBuilder {
    return this.main.globalData;
  }
}

/** Incrementally constructs a Jaxpr. */
class JaxprBuilder {
  eqns: JaxprEqn[] = [];
  tracerToVar: Map<JaxprTracer, Var> = new Map();
  constTracers: Map<Tracer, JaxprTracer> = new Map(); // already-seen value -> tracer
  constVals: Map<Var, Tracer> = new Map(); // var -> const value
  tracers: JaxprTracer[] = [];

  /**
   * Concrete JaxArrays whose creation ref is unowned after tracing.
   * These were created inside makeJaxpr body, captured as consts via
   * getOrMakeConstTracer (not wasRaw), and need one extra dispose to
   * balance their rc=1 from creation. The builder ref is balanced by
   * ClosedJaxpr.dispose() or _inlineLiterals; this set tracks the
   * creation-ref debt.
   */
  readonly constsNeedingCreationRefBalance: Set<Tracer> = new Set();

  newTracer(trace: JaxprTrace, aval: ShapedArray): JaxprTracer {
    const tracer = new JaxprTracer(trace, aval);
    this.tracers.push(tracer);
    return tracer;
  }

  addEqn(eqn: JaxprEqn) {
    this.eqns.push(eqn);
  }

  addVar(tracer: JaxprTracer): Var {
    if (this.tracerToVar.has(tracer)) {
      throw new Error(`Tracer was added as variable twice: ${tracer}`);
    }
    const v = new Var(tracer.aval);
    this.tracerToVar.set(tracer, v);
    return v;
  }

  getVar(tracer: JaxprTracer): Var {
    const v = this.tracerToVar.get(tracer);
    if (v === undefined) {
      throw new Error(`Could not find variable for tracer: ${tracer}`);
    }
    return v;
  }

  addConst(tracer: JaxprTracer, val: Tracer) {
    const v = this.addVar(tracer);
    this.constTracers.set(val, tracer);
    this.constVals.set(v, val);
    return v;
  }

  build(inTracers: JaxprTracer[], outTracers: JaxprTracer[]): ClosedJaxpr {
    // Initially, concatenate the constants as the first few inputs.
    const [constVars, consts] = unzip2(this.constVals.entries());
    const t2v = this.getVar.bind(this); // Maps tracer to value.
    const inBinders = [...constVars, ...inTracers.map(t2v)];
    const outVars = outTracers.map(t2v);
    const jaxpr = new Jaxpr(inBinders, this.eqns, outVars);

    // Inline any scalar constants as Lit and remove from the input list.
    typecheckJaxpr(jaxpr);
    const cjaxpr = ClosedJaxpr.borrowed(jaxpr, consts);
    return _inlineLiterals(cjaxpr);
  }
}

function _inlineLiterals({
  jaxpr,
  consts,
  constOwnership,
}: ClosedJaxpr): ClosedJaxpr {
  const literals = new Map<Atom, Lit>();
  const constBinders: Var[] = [];
  const newConsts: Tracer[] = [];

  for (let i = 0; i < consts.length; i++) {
    if (ndim(consts[i]) === 0 && consts[i] instanceof Array) {
      const ar = consts[i] as Array;
      let data: number[] | { [index: number]: number };
      try {
        data = ar.dataSync();
      } catch {
        // Sync readback not available (e.g., WebGPU without OffscreenCanvas)
        // — keep as const instead of inlining as Lit
        constBinders.push(jaxpr.inBinders[i]);
        newConsts.push(consts[i]);
        continue;
      }
      literals.set(jaxpr.inBinders[i], new Lit(ar.aval, data[0]));
      // Dispose undoes the .ref from getOrMakeConstTracer.
      ar.dispose();
    } else {
      constBinders.push(jaxpr.inBinders[i]);
      newConsts.push(consts[i]);
    }
  }

  const newEqns: JaxprEqn[] = jaxpr.eqns.map((eqn) =>
    new JaxprEqn(
      eqn.primitive,
      eqn.inputs.map((x) => literals.get(x) ?? x),
      eqn.params,
      eqn.outBinders,
    ).copyEffectsFrom(eqn),
  );
  const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
  const newJaxpr = new Jaxpr(
    [...constBinders, ...jaxpr.inBinders.slice(consts.length)],
    newEqns,
    newOuts,
  );
  typecheckJaxpr(newJaxpr); // Double-check for sanity.
  return new ClosedJaxpr(newJaxpr, newConsts, constOwnership);
}

type AbstractEvalRule<P extends Primitive> = (
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => ShapedArray[];

function binopAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  return [promoteAvals(x, y)];
}

function compareAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("compareAbstractEval expects ShapedArray inputs");
  }
  const aval = promoteAvals(x, y); // Make sure they can be typecast for comparison.
  return [new ShapedArray(aval.shape, DType.Bool, false)];
}

function vectorizedUnopAbstractEval([x]: ShapedArray[]) {
  return [ShapedArray.fromAval(x)];
}

export const abstractEvalRules: { [P in Primitive]: AbstractEvalRule<P> } = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Idiv]: binopAbstractEval,
  [Primitive.Mod]: binopAbstractEval,
  [Primitive.Min]: binopAbstractEval,
  [Primitive.Max]: binopAbstractEval,
  [Primitive.BitCombine]([x, y]: ShapedArray[]) {
    const aval = promoteAvals(x, y);
    if (isFloatDtype(aval.dtype))
      throw new TypeError(
        `bitwise operations require integer or boolean inputs, got ${aval.dtype}`,
      );
    return [aval];
  },
  [Primitive.BitShift]([x, y]: ShapedArray[]) {
    const shape = generalBroadcast(x.shape, y.shape);
    if (
      isFloatDtype(x.dtype) ||
      isFloatDtype(y.dtype) ||
      x.dtype === DType.Bool ||
      y.dtype === DType.Bool
    )
      throw new TypeError(
        `bit shift operations require integer inputs, got ${x} and ${y}`,
      );
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Reciprocal]: vectorizedUnopAbstractEval,
  [Primitive.Floor]: vectorizedUnopAbstractEval,
  [Primitive.Ceil]: vectorizedUnopAbstractEval,
  [Primitive.StopGradient]: vectorizedUnopAbstractEval,
  [Primitive.Cast]([x]: ShapedArray[], { dtype }) {
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.Bitcast]([x]: ShapedArray[], { dtype }) {
    if (x.dtype !== dtype && (x.dtype === DType.Bool || dtype === DType.Bool)) {
      throw new TypeError("Bitcast to/from bool is not allowed");
    }
    if (byteWidth(x.dtype) !== byteWidth(dtype)) {
      throw new TypeError(
        `Bitcast from ${x.dtype} to ${dtype} with different byte width`,
      );
    }
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.Asin]: vectorizedUnopAbstractEval,
  [Primitive.Atan]: vectorizedUnopAbstractEval,
  [Primitive.Exp]: vectorizedUnopAbstractEval,
  [Primitive.Log]: vectorizedUnopAbstractEval,
  [Primitive.Erf]: vectorizedUnopAbstractEval,
  [Primitive.Erfc]: vectorizedUnopAbstractEval,
  [Primitive.Sqrt]: vectorizedUnopAbstractEval,
  [Primitive.Reduce]([x], { axis }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pool]([x], { window, strides }) {
    const shape = checkPoolShape(x.shape as number[], window, strides);
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
    const shape = checkPoolShape(inShape, window, strides);
    if (!deepEqual(shape, x.shape)) {
      throw new TypeError(
        `PoolTranspose shape mismatch: expected ${JSON.stringify(shape)}, got ${JSON.stringify(x.shape)}`,
      );
    }
    return [new ShapedArray(inShape, x.dtype, x.weakType)];
  },
  [Primitive.Dot]([x, y]) {
    if (x.ndim === 0 && y.ndim === 0)
      throw new TypeError("Dot requires at least 1D inputs");
    const { shape, dtype, weakType } = promoteAvals(x, y);
    shape.splice(-1, 1); // Remove the contracted dimension.
    return [new ShapedArray(shape, dtype, weakType)];
  },
  [Primitive.Conv]([lhs, rhs], params) {
    const { dtype, weakType } = promoteAvals(lhs.scalar(), rhs.scalar());
    const shape = checkConvShape(
      lhs.shape as number[],
      rhs.shape as number[],
      params,
    );
    return [new ShapedArray(shape, dtype, weakType)];
  },
  [Primitive.Compare]: compareAbstractEval,
  [Primitive.Where]([cond, x, y]) {
    if (cond.dtype !== DType.Bool)
      throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
    const xy = promoteAvals(x, y);
    const shape = generalBroadcast(cond.shape, xy.shape);
    return [new ShapedArray(shape, xy.dtype, xy.weakType)];
  },
  [Primitive.Concatenate](xs, { axis }) {
    if (xs.length === 0)
      throw new TypeError("Concatenate requires at least one input");
    for (const x of xs) {
      if (
        x.ndim !== xs[0].ndim ||
        !x.shape.every((s, i) => i === axis || dimEquals(s, xs[0].shape[i]))
      )
        throw new TypeError(
          `Concatenate: inputs ${xs[0]} and ${x} must match shapes except on axis ${axis}`,
        );
    }
    const shape: Dim[] = xs[0].shape.slice();
    shape[axis] = xs.reduce(
      (sum, x) => sum + concreteDim(x.shape[axis], "Concatenate"),
      0,
    );
    const { dtype, weakType } = xs.map((x) => x.scalar()).reduce(promoteAvals);
    return [new ShapedArray(shape, dtype, weakType)];
  },
  [Primitive.Split]([x], { axis, sizes }) {
    const totalSize = sizes.reduce((a, b) => a + b, 0);
    if (!dimEquals(x.shape[axis], totalSize)) {
      throw new TypeError(
        `Split: sizes ${sizes} do not sum to dimension ${x.shape[axis]} on axis ${axis}`,
      );
    }
    return sizes.map((size) => {
      return new ShapedArray(
        x.shape.toSpliced(axis, 1, size),
        x.dtype,
        x.weakType,
      );
    });
  },
  [Primitive.RandomBits]([k0, k1]: ShapedArray[], { shape }) {
    if (k0.dtype !== DType.Uint32 || k1.dtype !== DType.Uint32) {
      throw new TypeError(
        `RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`,
      );
    }
    if (!deepEqual(k0.shape, k1.shape)) {
      throw new TypeError(
        `RandomBits: Keys have different shapes ${k0.shape} and ${k1.shape}`,
      );
    }
    if (!deepEqual(shape.slice(0, k0.ndim), k0.shape)) {
      throw new TypeError(
        `RandomBits: generated shape ${shape} must match key shape ${k0.shape}`,
      );
    }
    return [new ShapedArray(shape, DType.Uint32, false)];
  },
  [Primitive.Gather]([x, ...indices], { axis, outDim }) {
    for (const a of indices)
      if (a.dtype !== DType.Int32 && a.dtype !== DType.Uint32)
        throw new TypeError(
          `Gather indices must be Int32 or Uint32, got ${a.dtype}`,
        );
    if (axis.length !== indices.length)
      throw new TypeError(`Gather: ${axis} axes but ${indices.length} indices`);
    if (indices.length === 0)
      throw new TypeError("Gather must have 1+ indices with same shape");
    if (axis.some((a) => a < 0 || a >= x.shape.length))
      throw new TypeError("Gather axis out of bounds");
    if (outDim < 0 || outDim > x.shape.length - axis.length)
      throw new TypeError("Gather outDim out of bounds");
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length)
      throw new TypeError("Gather axes are not unique");
    const gatherShape = indices.reduce<Dim[]>(
      (shape, a) => generalBroadcast(shape, a.shape),
      [],
    );
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    newShape.splice(outDim, 0, ...gatherShape);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Transpose]([x], { perm }) {
    return [
      new ShapedArray(
        perm.map((i) => x.shape[i]),
        x.dtype,
        x.weakType,
      ),
    ];
  },
  [Primitive.Broadcast]([x], { shape }) {
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.Reshape]([x], { shape }) {
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.Flip]([x], _) {
    return [ShapedArray.fromAval(x)];
  },
  [Primitive.Reverse]([x], _) {
    return [ShapedArray.fromAval(x)];
  },
  [Primitive.Shrink]([x], { slice }) {
    const newShape: Dim[] = slice.map((s) => s[1] - s[0]);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pad]([x], { width }) {
    const newShape = x.shape.map(
      (dim, i) => concreteDim(dim, "Pad") + width[i][0] + width[i][1],
    );
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.DynamicUpdateSlice]([dst, src], { offset, axis }) {
    if (!(dst instanceof ShapedArray) || !(src instanceof ShapedArray)) {
      throw new TypeError("dynamicUpdateSlice expects shaped array inputs");
    }
    const dstShape = dst.shape;
    const srcShape = src.shape;
    if (dstShape.length === srcShape.length) {
      for (let i = 0; i < dstShape.length; i++) {
        if (i === axis) continue;
        if (!dimEquals(dstShape[i], srcShape[i]))
          throw new TypeError("dynamicUpdateSlice: shape mismatch");
      }
      if (
        offset + concreteDim(srcShape[axis], "DUS") >
        concreteDim(dstShape[axis], "DUS")
      )
        throw new TypeError("dynamicUpdateSlice: out of bounds");
    } else if (axis === 0 && dstShape.length === srcShape.length + 1) {
      for (let i = 0; i < srcShape.length; i++) {
        if (!dimEquals(dstShape[i + 1], srcShape[i]))
          throw new TypeError("dynamicUpdateSlice: stacked shape mismatch");
      }
      if (offset + 1 > concreteDim(dstShape[0], "DUS"))
        throw new TypeError("dynamicUpdateSlice: stacked out of bounds");
    } else {
      throw new TypeError("dynamicUpdateSlice: unsupported shapes");
    }
    return [new ShapedArray(dst.shape, dst.dtype, dst.weakType)];
  },
  [Primitive.DynamicUpdateSliceGeneral]([dst, src], { startIndices }) {
    if (!(dst instanceof ShapedArray) || !(src instanceof ShapedArray)) {
      throw new TypeError(
        "dynamicUpdateSliceGeneral expects shaped array inputs",
      );
    }
    const dstShape = dst.shape;
    const srcShape = src.shape;
    if (dstShape.length !== srcShape.length) {
      throw new TypeError(
        "dynamicUpdateSliceGeneral: dst and src must have same rank",
      );
    }
    if (startIndices.length !== dstShape.length) {
      throw new TypeError(
        `dynamicUpdateSliceGeneral: expected ${dstShape.length} start indices, got ${startIndices.length}`,
      );
    }
    for (let i = 0; i < dstShape.length; i++) {
      if (
        startIndices[i] + concreteDim(srcShape[i], "DUS_General") >
        concreteDim(dstShape[i], "DUS_General")
      ) {
        throw new TypeError(
          `dynamicUpdateSliceGeneral: out of bounds on axis ${i}`,
        );
      }
    }
    return [new ShapedArray(dst.shape, dst.dtype, dst.weakType)];
  },
  [Primitive.ScatterAdd]([target, indices, updates], { axis: _axis }) {
    if (!(target instanceof ShapedArray))
      throw new TypeError("scatter_add: target must be a shaped array");
    if (!(indices instanceof ShapedArray))
      throw new TypeError("scatter_add: indices must be a shaped array");
    if (!(updates instanceof ShapedArray))
      throw new TypeError("scatter_add: updates must be a shaped array");
    if (indices.dtype !== DType.Int32 && indices.dtype !== DType.Uint32)
      throw new TypeError(
        `scatter_add: indices must be Int32 or Uint32, got ${indices.dtype}`,
      );
    // Output shape = target shape, output dtype = target dtype.
    return [new ShapedArray(target.shape, target.dtype, target.weakType)];
  },
  [Primitive.Sort]([x]) {
    if (x.ndim === 0) throw new TypeError("sort: requires at least 1D input");
    return [ShapedArray.fromAval(x)];
  },
  [Primitive.Argsort]([x]) {
    if (x.ndim === 0)
      throw new TypeError("argsort: requires at least 1D input");
    return [
      ShapedArray.fromAval(x),
      new ShapedArray(x.shape, DType.Int32, false),
    ];
  },
  [Primitive.TriangularSolve]([a, b]) {
    if (a.ndim < 2)
      throw new TypeError(`triangular_solve: a must be at least 2D, got ${a}`);
    if (b.ndim < 2)
      throw new TypeError(`triangular_solve: b must be at least 2D, got ${b}`);
    // Solve a @ x.T = b.T
    // [n, n] @ [batch, n].T -> [batch, n].T
    const [m, n] = a.shape.slice(-2);
    const [_batch, q] = b.shape.slice(-2);
    if (
      !deepEqual(a.shape.slice(0, -2), b.shape.slice(0, -2)) ||
      a.dtype !== b.dtype ||
      !dimEquals(m, n) ||
      !dimEquals(n, q)
    )
      throw new TypeError(`triangular_solve: mismatch ${a} vs ${b}`);
    return [new ShapedArray(b.shape, b.dtype, a.weakType && b.weakType)];
  },
  [Primitive.Cholesky]([a]) {
    if (a.ndim < 2)
      throw new TypeError(`cholesky: requires at least 2D input, got ${a}`);
    if (!dimEquals(a.shape[a.ndim - 2], a.shape[a.ndim - 1]))
      throw new TypeError(`cholesky: must be square, got ${a}`);
    return [ShapedArray.fromAval(a)];
  },
  [Primitive.LU]([a]) {
    if (a.ndim < 2)
      throw new TypeError(`lu: requires at least 2D input, got ${a}`);
    const batch = a.shape.slice(0, -2);
    const [m, n] = a.shape.slice(-2) as number[];
    return [
      ShapedArray.fromAval(a),
      new ShapedArray([...batch, Math.min(m, n)], DType.Int32, false),
      new ShapedArray([...batch, m], DType.Int32, false),
    ];
  },
  [Primitive.QR]([a]) {
    if (a.ndim < 2)
      throw new TypeError(`qr: requires at least 2D input, got ${a}`);
    const batch = a.shape.slice(0, -2);
    const m = a.shape[a.ndim - 2] as number;
    const n = a.shape[a.ndim - 1] as number;
    const k = Math.min(m, n);
    return [
      new ShapedArray([...batch, m, k], a.dtype, a.weakType),
      new ShapedArray([...batch, k, n], a.dtype, a.weakType),
    ];
  },
  [Primitive.Jit](args, { jaxpr }) {
    const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
    if (args.length !== inTypes.length) {
      throw new TypeError(
        `jit expected ${inTypes.length} arguments, got ${args.length}`,
      );
    }
    for (let i = 0; i < inTypes.length; i++) {
      // Use compatible() to allow concrete args to match symbolic inTypes
      if (!args[i].compatible(inTypes[i])) {
        throw new TypeError(
          `jit argument ${i} has type ${args[i]}, expected ${inTypes[i]}`,
        );
      }
    }
    return outTypes;
  },
  [Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse: _ }) {
    // Args: [...consts, ...initCarry, ...xs]
    // jaxpr inputs: [...consts, ...carry, ...x_slice]
    // jaxpr outputs: [...newCarry, ...y_slice]
    // Note: reverse doesn't affect output shapes
    const numX = args.length - numConsts - numCarry;
    const { outTypes } = typecheckJaxpr(jaxpr);

    // Validate input types match jaxpr expectations
    if (jaxpr.inBinders.length !== numConsts + numCarry + numX) {
      throw new TypeError(
        `Scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`,
      );
    }

    // Return types: [...carryOut, ...ys]
    // carryOut shapes match initCarry shapes
    // ys shapes are [length, ...y_slice_shape]
    const carryOutTypes = outTypes.slice(0, numCarry);
    const ySliceTypes = outTypes.slice(numCarry);

    const yTypes = ySliceTypes.map((t) => {
      return new ShapedArray([length, ...t.shape], t.dtype, t.weakType);
    });

    return [...carryOutTypes, ...yTypes];
  },
  [Primitive.AssociativeScan](
    args,
    { jaxpr: bodyJaxpr, numLeaves, axis: _axis },
  ) {
    // Args: [...consts, ...elems_leaves]
    // bodyJaxpr: (a_leaves..., b_leaves...) -> result_leaves...
    // Output: same shape as elems_leaves (prefix scan result)
    const numConsts = args.length - numLeaves;
    const { outTypes } = typecheckJaxpr(bodyJaxpr);

    if (bodyJaxpr.inBinders.length !== numConsts + numLeaves * 2) {
      throw new TypeError(
        `AssociativeScan body jaxpr expects ${bodyJaxpr.inBinders.length} inputs, got ${numConsts + numLeaves * 2}`,
      );
    }
    if (outTypes.length !== numLeaves) {
      throw new TypeError(
        `AssociativeScan body jaxpr returns ${outTypes.length} outputs, expected ${numLeaves}`,
      );
    }

    // Output shapes = input elem shapes (prefix scan preserves shape)
    return args.slice(numConsts);
  },

  [Primitive.BlockIndex]() {
    return [new ShapedArray([], DType.Int32, false)];
  },

  [Primitive.BlockMap](
    args,
    {
      jaxpr: bodyJaxpr,
      blockShape,
      inAxes,
      outAxes,
      numConsts,
      numInputs,
      gridShape: explicitGridShape,
    },
  ) {
    // Args layout: [...consts, ...inputs]
    // bodyJaxpr operates on block-shaped slices; outputs are block-shaped.
    // Full output shapes: restore the original array dimensions along outAxes.
    const { outTypes } = typecheckJaxpr(bodyJaxpr);

    if (bodyJaxpr.inBinders.length !== numConsts + numInputs) {
      throw new TypeError(
        `BlockMap body jaxpr expects ${bodyJaxpr.inBinders.length} inputs, got ${numConsts + numInputs}`,
      );
    }

    // Compute grid shape from inputs + inAxes, or use explicit gridShape
    const inputs = args.slice(numConsts);
    const gridShape: number[] = explicitGridShape
      ? [...explicitGridShape]
      : new globalThis.Array(blockShape.length).fill(0);
    if (!explicitGridShape) {
      for (let i = 0; i < inputs.length; i++) {
        const axes = inAxes[i];
        for (let g = 0; g < blockShape.length; g++) {
          if (axes[g] !== null) {
            const dim = inputs[i].shape[axes[g]!] as number;
            gridShape[g] = Math.ceil(dim / blockShape[g]);
          }
        }
      }
    }

    // Output shapes: body output shapes with mapped dims set to original
    // input dimensions (not padded). The last block may be partial — the
    // executor handles this by only copying the valid portion.
    // Build a map from grid axis → original input dimension.
    const origDims: number[] = new globalThis.Array(blockShape.length).fill(0);
    for (let i = 0; i < inputs.length; i++) {
      const axes = inAxes[i];
      for (let g = 0; g < blockShape.length; g++) {
        if (axes[g] !== null) {
          origDims[g] = inputs[i].shape[axes[g]!] as number;
        }
      }
    }
    // When explicit gridShape is provided and no mapped input contributes
    // the original dimension, use gridShape[g] * blockShape[g].
    if (explicitGridShape) {
      for (let g = 0; g < blockShape.length; g++) {
        if (origDims[g] === 0) {
          origDims[g] = explicitGridShape[g] * blockShape[g];
        }
      }
    }

    return outTypes.map((bodyOutAval, oi) => {
      const axes = outAxes[oi];
      const fullShape = [...bodyOutAval.shape];
      for (let g = 0; g < blockShape.length; g++) {
        if (axes[g] !== null) {
          fullShape[axes[g]!] = origDims[g];
        }
      }
      return new ShapedArray(
        fullShape,
        bodyOutAval.dtype,
        bodyOutAval.weakType,
      );
    });
  },

  [Primitive.ForiLoop](
    args,
    { jaxpr, numConsts, lower: _lower, upper: _upper },
  ) {
    const { outTypes } = typecheckJaxpr(jaxpr);
    // Body signature: (i: int32, carry...) => carry...
    const numCarry = args.length - numConsts;
    if (jaxpr.inBinders.length !== numConsts + 1 + numCarry) {
      throw new TypeError(
        `ForiLoop body expects ${jaxpr.inBinders.length} inputs, got ${numConsts + 1 + numCarry}`,
      );
    }
    if (outTypes.length !== numCarry) {
      throw new TypeError(
        `ForiLoop body returns ${outTypes.length} outputs, expected ${numCarry} carry outputs`,
      );
    }
    // Output is the final carry
    return args.slice(numConsts);
  },

  [Primitive.WorkgroupAssociativeScan](args, { jaxpr, numConsts }) {
    // Args: [...consts, elem]
    // Body jaxpr: (consts..., a_scalar, b_scalar) => result_scalar
    // Output: same shape as elem (prefix scan preserves shape)
    const { outTypes } = typecheckJaxpr(jaxpr);
    const numElems = args.length - numConsts;
    if (jaxpr.inBinders.length !== numConsts + numElems * 2) {
      throw new TypeError(
        `WorkgroupAssociativeScan body expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numElems * 2}`,
      );
    }
    if (outTypes.length !== numElems) {
      throw new TypeError(
        `WorkgroupAssociativeScan body returns ${outTypes.length} outputs, expected ${numElems}`,
      );
    }
    return args.slice(numConsts);
  },

  [Primitive.DynamicSlice](args, { sliceSizes }) {
    const operand = args[0];
    const numIndices = args.length - 1;
    if (numIndices !== operand.shape.length) {
      throw new TypeError(
        `DynamicSlice expected ${operand.shape.length} start indices, got ${numIndices}`,
      );
    }
    for (let i = 1; i < args.length; i++) {
      if (args[i].shape.length !== 0) {
        throw new TypeError(
          `DynamicSlice start indices must be scalars, got shape ${args[i].shape}`,
        );
      }
    }
    return [new ShapedArray(sliceSizes, operand.dtype, operand.weakType)];
  },
  [Primitive.UncheckedDynamicSlice](args, { sliceSizes }) {
    const operand = args[0];
    const numIndices = args.length - 1;
    if (numIndices !== operand.shape.length) {
      throw new TypeError(
        `UncheckedDynamicSlice expected ${operand.shape.length} start indices, got ${numIndices}`,
      );
    }
    for (let i = 1; i < args.length; i++) {
      if (args[i].shape.length !== 0) {
        throw new TypeError(
          `UncheckedDynamicSlice start indices must be scalars, got shape ${args[i].shape}`,
        );
      }
    }
    if (DEBUG >= 2) {
      for (let k = 0; k < operand.shape.length; k++) {
        if ((sliceSizes[k] as number) > (operand.shape[k] as number)) {
          throw new Error(
            `UncheckedDynamicSlice: slice[${k}]=${sliceSizes[k]} > shape[${k}]=${operand.shape[k]}`,
          );
        }
      }
    }
    return [new ShapedArray(sliceSizes, operand.dtype, operand.weakType)];
  },
};

function splitIdx(values: any[], argnums: Set<number>): [any[], any[]] {
  const a: any[] = [];
  const b: any[] = [];
  for (let i = 0; i < values.length; i++) {
    if (argnums.has(i)) a.push(values[i]);
    else b.push(values[i]);
  }
  return [a, b];
}

function joinIdx(n: number, a: any[], b: any[], argnums: Set<number>): any[] {
  const result: any[] = [];
  let ai = 0;
  let bi = 0;
  for (let i = 0; i < n; i++) {
    if (argnums.has(i)) result.push(a[ai++]);
    else result.push(b[bi++]);
  }
  return result;
}

/** @inline */
export type JitOpts = {
  staticArgnums?: number[];
  validateRefs?: boolean;
  /**
   * Map from argument axis index to symbolic dimension name.
   * When set, the specified axes are traced symbolically (as `SymDim`),
   * allowing a single traced Jaxpr to be reused for different sizes
   * on those axes without re-tracing.
   *
   * Example: `{ 0: "T" }` traces axis 0 as symbolic dimension "T".
   */
  dynamic_axes?: Record<number, string>;
};

export function makeJaxpr(
  f: (...args: any[]) => any,
  opts?: JitOpts,
): (...argsIn: any) => { jaxpr: ClosedJaxpr; treedef: JsTreeDef } {
  return (...argsIn) => {
    const staticArgnums = new Set(opts?.staticArgnums ?? []);
    const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);

    const [avalsIn, inTree] = treeFlatten(shapedArgs);
    const [fFlat, outTree] = flattenFun((...dynamicArgs: any[]) => {
      return f(
        ...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums),
      );
    }, inTree);

    const builder = new JaxprBuilder();
    using main = newMain(JaxprTrace, builder);
    main.isAbstract = true;
    using _dynamic = newDynamic(main);

    const trace = new JaxprTrace(main);
    const tracersIn = avalsIn.map((aval) =>
      trace.newArg(typeof aval === "object" ? aval : pureArray(aval)),
    );

    const prevBody = inMakeJaxprBody();
    _setInMakeJaxprBody(true);
    let outs: any;
    try {
      outs = fFlat(...tracersIn);
    } finally {
      _setInMakeJaxprBody(prevBody);
    }
    const tracersOut = outs.map(
      (out: Tracer) => fullRaise(trace, out) as JaxprTracer,
    );

    // Snapshot the builder's creation-ref-debt set BEFORE build(). build()
    // calls _inlineLiterals which removes scalar consts from the ClosedJaxpr
    // and disposes their builder ref — but those arrays still need their
    // creation ref balanced.
    //
    // At pre-build time, each const has exactly 1 builder ref (from .ref in
    // getOrMakeConstTracer). Any additional ref is the creation ref, still
    // outstanding because no user/framework dispose() balanced it during body
    // execution. refCount==1 means the creation ref was already balanced
    // (e.g., jacfwd disposes eyeMatrix explicitly before body returns).
    const constsToBalance: Tracer[] = [];
    for (const c of builder.constsNeedingCreationRefBalance) {
      if (c.refCount > 1) {
        constsToBalance.push(c);
      }
    }

    const jaxpr = builder.build(tracersIn, tracersOut);

    // Balance creation refs of concrete JaxArrays created during this
    // makeJaxpr body execution. These arrays were born with rc=1 (creation
    // ref) and then captured via getOrMakeConstTracer (.ref). The builder
    // refs are balanced by ClosedJaxpr.dispose() and _inlineLiterals.
    // The creation ref has no owner — balance it here.
    //
    // Deliberate jit/eager divergence: in eager mode, body-created arrays
    // with no dispose() leak silently. In JIT mode, makeJaxpr balances
    // them here to prevent GPU slot exhaustion. checkLeaks + ESLint catch
    // both at dev time. See PLAN.md Phase 3d.5 Step 5 for rationale.
    for (const c of constsToBalance) {
      if (c.refCount > 0) {
        c.dispose();
        if (c instanceof Array) c.markCreationRefBalancedByMakeJaxpr();
      }
    }
    if (DEBUG >= 1 && constsToBalance.length > 0) {
      // This diagnostic catches body-created *concrete* arrays captured as
      // consts (e.g. `np.array([2])` without `using`). Traced temporaries
      // like `x.mul(x)` are graph nodes, not real buffers, so they don't
      // need rescuing here — the ESLint jax-js/no-array-chain and
      // jax-js/no-nested-array-leak rules catch those structural patterns.
      const descs = constsToBalance
        .filter((c) => c instanceof Array)
        .map((c) => c.toString());
      console.warn(
        `[jax-js] makeJaxpr rescued ${constsToBalance.length} intermediate(s) ` +
          `that would leak in eager mode: ${descs.join(", ")}. ` +
          `Extract to \`using\` bindings for eager/jit parity.`,
      );
    }

    if (outTree.value === undefined) {
      throw new Error("outTree was not set in makeJaxpr");
    }
    const simplified = jaxpr.mapJaxpr((j) => j.simplify());

    // When effect verification is enabled, validate the simplified Jaxpr.
    if (_verifyEffectsEnabled) {
      const result = verifyJaxprEffects(simplified.jaxpr);
      if (!result.ok) {
        simplified.dispose();
        throw new Error(
          `Effect verification failed:\n${result.errors.join("\n")}`,
        );
      }
    }

    return { jaxpr: simplified, treedef: outTree.value };
  };
}

/**
 * Function-identity registry for jit() deduplication.
 *
 * When `jit(fn)(args)` is called inline (no persistent reference to the
 * wrapper), each call would create a new OwnedFunction with a new cache —
 * accumulating GPU-backed ClosedJaxpr consts indefinitely. The registry
 * deduplicates: same `(fn, opts)` → same OwnedFunction + shared cache.
 *
 * WeakMap allows GC of `fn` when it goes out of scope (though the dispose
 * callback in `_jitFunctionDisposers` retains the cache until `clearCaches()`).
 */
// eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
const _jitRegistry = new WeakMap<Function, Map<string, OwnedFunction<any>>>();

function _serializeJitOpts(opts?: JitOpts): string {
  if (!opts) return "";
  const parts: string[] = [];
  if (opts.staticArgnums?.length) {
    parts.push("s:" + [...opts.staticArgnums].sort((a, b) => a - b).join(","));
  }
  if (opts.dynamic_axes) {
    const entries = Object.entries(opts.dynamic_axes)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}=${v}`);
    parts.push("d:" + entries.join(","));
  }
  return parts.join("|");
}

export function jit<F extends (...args: any[]) => any>(
  f: F,
  opts?: JitOpts,
): OwnedFunction<
  (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>
> {
  // Deduplicate by function identity + opts: inline jit(fn)(args) patterns
  // reuse the same OwnedFunction and shared cache across calls.
  const optsKey = _serializeJitOpts(opts);
  const byOpts = _jitRegistry.get(f);
  if (byOpts) {
    const existing = byOpts.get(optsKey);
    if (existing) return existing;
  }

  type JitCacheEntry = Omit<ReturnType<ReturnType<typeof makeJaxpr>>, "jaxpr"> &
    CachedJaxprArtifact;
  const cache = new Map<string, JitCacheEntry>();
  const staticArgnums = new Set(opts?.staticArgnums ?? []);

  const dynamicAxes = opts?.dynamic_axes;

  const result = ((...args) => {
    const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);

    const [argsFlat, inTree] = treeFlatten(dynamicArgs);
    const avalsInFlat = argsFlat.map((x) => ShapedArray.fromAval(getAval(x)));

    // When dynamic_axes is set, replace specified axes with SymDim instances
    // so that the traced Jaxpr is polymorphic over those dimensions.
    let avalsForCache: ShapedArray[];
    if (dynamicAxes) {
      avalsForCache = avalsInFlat.map((aval) => {
        const newShape: Dim[] = aval.shape.map((d, i) =>
          i in dynamicAxes ? new SymDim(dynamicAxes[i]) : d,
        );
        return new ShapedArray(newShape, aval.dtype, aval.weakType);
      });
    } else {
      avalsForCache = avalsInFlat;
    }

    const avalsIn = treeUnflatten(inTree, avalsForCache) as any[];
    const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
    // Include backend type: cached const environments are concrete arrays living on
    // whichever device was active at first trace. Cross-device reuse would
    // read stale data from the wrong backend.
    const {
      jaxpr,
      constEnv,
      treedef: outTree,
    } = runWithCache(cache, [defaultDevice(), ...jaxprArgs], () => {
      return buildCachedJaxprArtifact(makeJaxpr(f, opts)(...jaxprArgs));
    });

    const outs = bind(Primitive.Jit, [...constEnv.values, ...argsFlat], {
      name: f.name || "closure",
      jaxpr,
      numConsts: constEnv.length,
      dynamicAxes,
    });
    return treeUnflatten(outTree, outs);
  }) as OwnedFunction<F>;

  result.dispose = () => {
    for (const { constEnv } of cache.values()) {
      constEnv.dispose();
    }
    cache.clear();
  };
  result[Symbol.dispose] = result.dispose;

  // Register for bulk disposal during leak detection. The dispose callback
  // frees ClosedJaxpr consts and clears the cache — the next call will
  // re-trace and create fresh consts.
  _jitFunctionDisposers.add(result.dispose);

  propagateCustomDerivativeMetadata(f, result);

  // Store in registry for dedup of subsequent jit(f, opts) calls.
  let registry = byOpts;
  if (!registry) {
    registry = new Map();
    _jitRegistry.set(f, registry);
  }
  registry.set(optsKey, result);

  return result;
}
