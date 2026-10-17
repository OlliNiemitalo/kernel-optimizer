/**
 * @file Symbolic dimension types for polymorphic shapes (M4.1).
 *
 * This module is dependency-free (no imports from other src/ modules) to avoid
 * circular imports — both shape.ts and utils.ts depend on it.
 */

/**
 * A symbolic dimension variable used for polymorphic shapes.
 * During tracing with `jit({ dynamic_axes })`, dynamic axes are
 * represented as SymDim instances instead of concrete numbers.
 * SymDim values are resolved to concrete numbers before compilation.
 */
export class SymDim {
  constructor(readonly name: string) {}
  toString(): string {
    return this.name;
  }
  toJSON(): string {
    return `__sym__${this.name}`;
  }
}

/**
 * A dimension is either a concrete number or a symbolic variable.
 * ShapedArray.shape uses Dim[] to support polymorphic shapes.
 */
export type Dim = number | SymDim;

/** Check if a dimension is symbolic. */
export function isSymbolicDim(d: Dim): d is SymDim {
  return d instanceof SymDim;
}

/** Returns true if any dimension in the shape is symbolic. */
export function hasSymbolicDims(shape: readonly Dim[]): boolean {
  return shape.some(isSymbolicDim);
}

/**
 * Subtract two dimensions: `a - b`.
 * Returns a concrete number when both are concrete, the `Dim` `a` when
 * `b === 0`, or `0` when `a` and `b` are the same symbolic dim.
 * Throws if the subtraction cannot be resolved symbolically.
 */
export function dimSub(a: Dim, b: Dim): Dim {
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof b === "number" && b === 0) return a;
  if (a instanceof SymDim && b instanceof SymDim && a.name === b.name) return 0;
  throw new Error(
    `Cannot compute symbolic dimension subtraction: ${a} - ${b}. ` +
      `foriLoop transpose requires lower=0 or both bounds concrete.`,
  );
}

/**
 * Assert that a dimension is concrete, returning the number.
 * Throws if the dimension is symbolic.
 */
export function concreteDim(d: Dim, context?: string): number {
  if (typeof d === "number") return d;
  throw new Error(
    `Expected concrete dimension but got symbolic dim '${d.name}'${context ? ` in ${context}` : ""}`,
  );
}

/**
 * Resolve a potentially-symbolic dimension using bindings.
 * Returns the concrete number value.
 */
export function resolveDim(
  d: Dim,
  bindings: ReadonlyMap<string, number>,
): number {
  if (typeof d === "number") return d;
  const val = bindings.get(d.name);
  if (val === undefined) {
    throw new Error(
      `Unresolved symbolic dimension '${d.name}' — not in bindings`,
    );
  }
  return val;
}

/**
 * Assert that all dimensions in a shape are concrete.
 * Returns the shape typed as number[]. Throws if any dimension is symbolic.
 */
export function concreteShape(
  shape: readonly Dim[],
  context?: string,
): number[] {
  for (const d of shape) {
    if (typeof d !== "number") {
      throw new Error(
        `Expected concrete shape but got symbolic dim '${(d as SymDim).name}' in [${shape}]${context ? ` (${context})` : ""}`,
      );
    }
  }
  return shape as number[];
}

/**
 * Compare two dimensions for equality. Two SymDims are equal if they have
 * the same name.
 */
export function dimEquals(a: Dim, b: Dim): boolean {
  if (typeof a === "number" && typeof b === "number") return a === b;
  if (a instanceof SymDim && b instanceof SymDim) return a.name === b.name;
  return false;
}

/**
 * Check if dimension `a` is compatible with pattern `b`.
 * A concrete dim is compatible with a symbolic dim (any value satisfies it).
 * Otherwise behaves like dimEquals.
 */
export function dimCompatible(a: Dim, b: Dim): boolean {
  if (dimEquals(a, b)) return true;
  // Concrete value satisfies a symbolic pattern
  if (typeof a === "number" && b instanceof SymDim) return true;
  if (a instanceof SymDim && typeof b === "number") return true;
  return false;
}

/**
 * Resolve a shape by substituting symbolic dimensions with concrete values.
 */
export function resolveShape(
  shape: readonly Dim[],
  bindings: ReadonlyMap<string, number>,
): number[] {
  return shape.map((d) => {
    if (typeof d === "number") return d;
    const val = bindings.get(d.name);
    if (val === undefined) {
      throw new Error(
        `Unresolved symbolic dimension '${d.name}' — not in bindings`,
      );
    }
    return val;
  });
}

// ---------------------------------------------------------------------------
// SizeExpr — symbolic size expressions for parameterized codegen (M4.2)
// ---------------------------------------------------------------------------

/**
 * A symbolic size: `factor × prod(syms)`.
 * Used when Kernel.size or malloc byte count depends on symbolic dims.
 */
export class SymbolicSize {
  constructor(
    readonly factor: number,
    readonly syms: readonly string[],
  ) {}

  resolve(bindings: ReadonlyMap<string, number>): number {
    let n = this.factor;
    for (const s of this.syms) {
      const v = bindings.get(s);
      if (v === undefined)
        throw new Error(`Unresolved symbolic dim '${s}' in SizeExpr`);
      n *= v;
    }
    return n;
  }

  /** Canonical string key for pool/recycling matching. */
  key(): string {
    return `${[...this.syms].sort().join("*")}*${this.factor}`;
  }

  toString(): string {
    return this.syms.length > 0
      ? `${this.syms.join("×")}×${this.factor}`
      : String(this.factor);
  }

  /** Multiply this symbolic size by a concrete factor. */
  mul(n: number): SymbolicSize {
    return new SymbolicSize(this.factor * n, this.syms);
  }
}

/**
 * A size expression: either a concrete number or a symbolic product.
 * `Kernel.size`, `KernelOutput.bytes`, and `JitStep.malloc.size` use this type.
 */
export type SizeExpr = number | SymbolicSize;

/** Check if a size expression is symbolic. */
export function isSymbolicSize(expr: SizeExpr): expr is SymbolicSize {
  return expr instanceof SymbolicSize;
}

/** Resolve a SizeExpr to a concrete number. */
export function resolveSizeExpr(
  expr: SizeExpr,
  bindings: ReadonlyMap<string, number>,
): number {
  return typeof expr === "number" ? expr : expr.resolve(bindings);
}

/**
 * Return a canonical key for a SizeExpr, suitable for pool/recycling matching.
 * Concrete numbers return themselves; symbolic sizes return a canonical string.
 */
export function sizeExprKey(expr: SizeExpr): string | number {
  return typeof expr === "number" ? expr : expr.key();
}

/**
 * Compute `prod(shape) × multiplier` as a SizeExpr.
 * If any dimension is symbolic, returns a SymbolicSize; otherwise a number.
 */
export function dimProduct(
  shape: readonly Dim[],
  multiplier: number = 1,
): SizeExpr {
  const syms: string[] = [];
  let factor = multiplier;
  for (const d of shape) {
    if (isSymbolicDim(d)) syms.push(d.name);
    else factor *= d;
  }
  return syms.length === 0 ? factor : new SymbolicSize(factor, syms);
}

/**
 * Multiply a SizeExpr by a concrete number.
 */
export function sizeExprMul(expr: SizeExpr, n: number): SizeExpr {
  if (typeof expr === "number") return expr * n;
  return expr.mul(n);
}
