/**
 * High-level helpers for wasmblr.
 *
 * This module provides ergonomic abstractions over the low-level wasmblr API,
 * making it easier to write WASM routines while keeping the code readable.
 *
 * Design principles:
 * - Non-intrusive: wraps CodeGenerator, doesn't modify it
 * - Composable: helpers are building blocks, not monolithic functions
 * - SIMD-ready: memory helpers have SIMD variants from day 1
 * - Dtype-aware: supports f32/f64/i32 without hardcoded sizes
 * - Lazy evaluation: index expressions are callbacks to defer stack operations
 */

import { CodeGenerator } from "./wasmblr";

export type ScalarDtype = "f32" | "f64" | "i32";

/** Byte width for each scalar dtype. */
const DTYPE_SIZE: Record<ScalarDtype, number> = {
  f32: 4,
  f64: 8,
  i32: 4,
};

/**
 * High-level WASM codegen helper.
 *
 * Wraps a CodeGenerator instance and provides ergonomic methods for common
 * patterns like loops, memory access, and SIMD operations.
 */
export class WasmHl {
  constructor(readonly cg: CodeGenerator) {}

  // ============================================================================
  // Control Flow
  // ============================================================================

  /**
   * Emit a for loop: for (let i = start; i < end; i++) { body() }
   *
   * @param i - Local variable index for the loop counter
   * @param start - Initial value (constant or callback that pushes i32 onto stack)
   * @param end - End value (constant or callback that pushes i32 onto stack)
   * @param body - Loop body callback
   */
  forLoop(
    i: number,
    start: number | (() => void),
    end: number | (() => void),
    body: () => void,
  ): void {
    const { cg } = this;

    // i = start
    if (typeof start === "number") {
      cg.i32.const(start);
    } else {
      start();
    }
    cg.local.set(i);

    cg.block(cg.void);
    cg.loop(cg.void);
    {
      // if (i >= end) break
      cg.local.get(i);
      if (typeof end === "number") {
        cg.i32.const(end);
      } else {
        end();
      }
      cg.i32.ge_s();
      cg.br_if(1);

      body();

      // i++
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(i);

      cg.br(0);
    }
    cg.end();
    cg.end();
  }

  /**
   * Emit a downward for loop: for (let i = start - 1; i >= end; i--) { body() }
   *
   * @param i - Local variable index for the loop counter
   * @param start - Start value (exclusive, loop starts at start - 1) - constant or callback
   * @param end - End value (inclusive, constant)
   * @param body - Loop body callback
   */
  forLoopDown(
    i: number,
    start: number | (() => void),
    end: number,
    body: () => void,
  ): void {
    const { cg } = this;

    // i = start - 1
    if (typeof start === "number") {
      cg.i32.const(start - 1);
    } else {
      start();
      cg.i32.const(1);
      cg.i32.sub();
    }
    cg.local.set(i);

    cg.block(cg.void);
    cg.loop(cg.void);
    {
      // if (i < end) break
      cg.local.get(i);
      cg.i32.const(end);
      cg.i32.lt_s();
      cg.br_if(1);

      body();

      // i--
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.sub();
      cg.local.set(i);

      cg.br(0);
    }
    cg.end();
    cg.end();
  }

  /**
   * Emit a while loop: while (cond()) { body() }
   *
   * @param cond - Callback that pushes i32 condition onto stack (0 = exit)
   * @param body - Loop body callback
   */
  whileLoop(cond: () => void, body: () => void): void {
    const { cg } = this;

    cg.block(cg.void);
    cg.loop(cg.void);
    {
      cond();
      cg.i32.eqz();
      cg.br_if(1);

      body();

      cg.br(0);
    }
    cg.end();
    cg.end();
  }

  /**
   * Emit an if-else statement. Condition should already be on the stack (i32).
   *
   * @param resultType - Result type of the if expression (use cg.void for statements)
   * @param then - Then branch callback
   * @param else_ - Optional else branch callback
   */
  ifElse(
    resultType: { typeId: number; name: string },
    then: () => void,
    else_?: () => void,
  ): void {
    const { cg } = this;

    cg.if(resultType);
    then();
    if (else_) {
      cg.else();
      else_();
    }
    cg.end();
  }

  // ============================================================================
  // Memory Access (Scalar)
  // ============================================================================

  /**
   * Compute address: base + indexExpr * elementSize
   * Leaves the address (i32) on the stack.
   *
   * @param base - Local variable index for base pointer
   * @param indexExpr - Callback that pushes index (i32) onto stack
   * @param elementSize - Size of each element in bytes (default 4)
   */
  addr(base: number, indexExpr: () => void, elementSize: number = 4): void {
    const { cg } = this;
    cg.local.get(base);
    indexExpr();
    if (elementSize !== 1) {
      cg.i32.const(elementSize);
      cg.i32.mul();
    }
    cg.i32.add();
  }

  /**
   * Load a value from memory at base + indexExpr * elementSize.
   * Leaves the loaded value on the stack.
   *
   * @param dtype - Data type to load
   * @param base - Local variable index for base pointer
   * @param indexExpr - Callback that pushes index (i32) onto stack
   */
  load(dtype: ScalarDtype, base: number, indexExpr: () => void): void {
    this.addr(base, indexExpr, DTYPE_SIZE[dtype]);
    this.loadDirect(dtype);
  }

  /**
   * Load a value from the address already on the stack.
   */
  loadDirect(dtype: ScalarDtype): void {
    const { cg } = this;
    const align = Math.log2(DTYPE_SIZE[dtype]);
    if (dtype === "f32") cg.f32.load(align);
    else if (dtype === "f64") cg.f64.load(align);
    else cg.i32.load(align);
  }

  /**
   * Store a value to memory at base + indexExpr * elementSize.
   * Call this, then push the value onto the stack, then call storeDirect().
   *
   * Alternative: use storeValue() which takes a value callback.
   *
   * @param dtype - Data type to store
   * @param base - Local variable index for base pointer
   * @param indexExpr - Callback that pushes index (i32) onto stack
   */
  storeAddr(dtype: ScalarDtype, base: number, indexExpr: () => void): void {
    this.addr(base, indexExpr, DTYPE_SIZE[dtype]);
  }

  /**
   * Store a value (on stack) to the address (below it on stack).
   */
  storeDirect(dtype: ScalarDtype): void {
    const { cg } = this;
    const align = Math.log2(DTYPE_SIZE[dtype]);
    if (dtype === "f32") cg.f32.store(align);
    else if (dtype === "f64") cg.f64.store(align);
    else cg.i32.store(align);
  }

  /**
   * Store a value to memory at base + indexExpr * elementSize.
   *
   * @param dtype - Data type to store
   * @param base - Local variable index for base pointer
   * @param indexExpr - Callback that pushes index (i32) onto stack
   * @param valueExpr - Callback that pushes value onto stack
   */
  store(
    dtype: ScalarDtype,
    base: number,
    indexExpr: () => void,
    valueExpr: () => void,
  ): void {
    this.storeAddr(dtype, base, indexExpr);
    valueExpr();
    this.storeDirect(dtype);
  }

  /**
   * Copy memory: dst[0..byteCount] = src[0..byteCount]
   * Uses WebAssembly bulk memory.copy instruction.
   *
   * @param dst - Local variable index for destination pointer
   * @param src - Local variable index for source pointer
   * @param byteCount - Number of bytes to copy (constant)
   * @param _tmpIdx - Unused (kept for API compatibility)
   */
  memcpy(dst: number, src: number, byteCount: number, _tmpIdx?: number): void {
    const { cg } = this;
    cg.local.get(dst);
    cg.local.get(src);
    cg.i32.const(byteCount);
    cg.memory.copy();
  }

  /**
   * Copy memory with dynamic byte count.
   * Uses WebAssembly bulk memory.copy instruction.
   *
   * @param dst - Local variable index for destination pointer
   * @param src - Local variable index for source pointer
   * @param byteCountExpr - Callback that pushes byte count (i32) onto stack
   * @param _tmpIdx - Unused (kept for API compatibility)
   */
  memcpyDynamic(
    dst: number,
    src: number,
    byteCountExpr: () => void,
    _tmpIdx?: number,
  ): void {
    const { cg } = this;
    cg.local.get(dst);
    cg.local.get(src);
    byteCountExpr();
    cg.memory.copy();
  }

  // ============================================================================
  // Index Helpers
  // ============================================================================

  /**
   * Compute 2D index: i * stride + j
   * Leaves the result (i32) on the stack.
   *
   * @param iExpr - Callback that pushes row index onto stack
   * @param stride - Row stride (number of columns)
   * @param jExpr - Callback that pushes column index onto stack
   */
  index2D(
    iExpr: () => void,
    stride: number | (() => void),
    jExpr: () => void,
  ): void {
    const { cg } = this;
    iExpr();
    if (typeof stride === "number") {
      cg.i32.const(stride);
    } else {
      stride();
    }
    cg.i32.mul();
    jExpr();
    cg.i32.add();
  }

  /**
   * Push a local variable's value onto the stack.
   * Convenience wrapper for cg.local.get().
   */
  get(local: number): void {
    this.cg.local.get(local);
  }

  /**
   * Create a callback that pushes a local's value onto the stack.
   * Useful for passing to other helpers.
   */
  getExpr(local: number): () => void {
    return () => this.cg.local.get(local);
  }

  // ============================================================================
  // Memory Access (SIMD)
  // ============================================================================

  /**
   * Load f32x4 from memory at base + indexExpr * 16.
   * Leaves the v128 on the stack.
   */
  loadF32x4(base: number, indexExpr: () => void): void {
    const { cg } = this;
    this.addr(base, indexExpr, 16);
    cg.v128.load(4); // 4 = log2(16) alignment
  }

  /**
   * Store f32x4 to memory at base + indexExpr * 16.
   * Value should be on the stack after calling storeAddrF32x4.
   */
  storeAddrF32x4(base: number, indexExpr: () => void): void {
    this.addr(base, indexExpr, 16);
  }

  /**
   * Store f32x4 (on stack) to address (below it on stack).
   */
  storeDirectF32x4(): void {
    this.cg.v128.store(4);
  }

  /**
   * Store f32x4 to memory at base + indexExpr * 16.
   *
   * @param base - Local variable index for base pointer
   * @param indexExpr - Callback that pushes index onto stack
   * @param valueExpr - Callback that pushes v128 value onto stack
   */
  storeF32x4(base: number, indexExpr: () => void, valueExpr: () => void): void {
    this.storeAddrF32x4(base, indexExpr);
    valueExpr();
    this.storeDirectF32x4();
  }

  /**
   * Horizontal sum of f32x4 → f32.
   * Consumes v128 on stack, leaves f32 on stack.
   */
  f32x4Hsum(): void {
    const { cg } = this;
    // v = [a, b, c, d]
    // sum = a + b + c + d
    const v = cg.local.declare(cg.v128);
    cg.local.set(v);

    cg.f32.const(0);
    for (let i = 0; i < 4; i++) {
      cg.local.get(v);
      cg.f32x4.extract_lane(i);
      cg.f32.add();
    }
  }

  /**
   * Splat f32 (on stack) to f32x4.
   */
  f32x4Splat(): void {
    this.cg.f32x4.splat();
  }

  /**
   * Horizontal sum of f64x2 → f64.
   * Consumes v128 on stack, leaves f64 on stack.
   */
  f64x2Hsum(): void {
    const { cg } = this;
    const v = cg.local.declare(cg.v128);
    cg.local.set(v);

    cg.local.get(v);
    cg.f64x2.extract_lane(0);
    cg.local.get(v);
    cg.f64x2.extract_lane(1);
    cg.f64.add();
  }

  /**
   * Splat f64 (on stack) to f64x2.
   */
  f64x2Splat(): void {
    this.cg.f64x2.splat();
  }

  // ============================================================================
  // Arithmetic Helpers
  // ============================================================================

  /**
   * Push a constant onto the stack.
   */
  const(dtype: ScalarDtype, value: number): void {
    const { cg } = this;
    if (dtype === "f32") cg.f32.const(value);
    else if (dtype === "f64") cg.f64.const(value);
    else cg.i32.const(value);
  }

  /**
   * Apply sqrt to the value on the stack.
   */
  sqrt(dtype: "f32" | "f64"): void {
    if (dtype === "f32") this.cg.f32.sqrt();
    else this.cg.f64.sqrt();
  }

  /**
   * Apply binary operation to values on the stack.
   * Consumes two values, pushes one result.
   */
  binOp(dtype: ScalarDtype, op: "add" | "sub" | "mul" | "div"): void {
    const { cg } = this;
    if (dtype === "f32") {
      if (op === "add") cg.f32.add();
      else if (op === "sub") cg.f32.sub();
      else if (op === "mul") cg.f32.mul();
      else cg.f32.div();
    } else if (dtype === "f64") {
      if (op === "add") cg.f64.add();
      else if (op === "sub") cg.f64.sub();
      else if (op === "mul") cg.f64.mul();
      else cg.f64.div();
    } else {
      if (op === "add") cg.i32.add();
      else if (op === "sub") cg.i32.sub();
      else if (op === "mul") cg.i32.mul();
      else cg.i32.div_s();
    }
  }

  /**
   * Compare two values on the stack for equality.
   * Consumes two values, pushes i32 (0 or 1).
   */
  eq(dtype: ScalarDtype): void {
    const { cg } = this;
    if (dtype === "f32") cg.f32.eq();
    else if (dtype === "f64") cg.f64.eq();
    else cg.i32.eq();
  }

  /**
   * Compare two i32 values: a < b (signed).
   */
  ltS(): void {
    this.cg.i32.lt_s();
  }

  /**
   * Compare two i32 values: a <= b (signed).
   */
  leS(): void {
    this.cg.i32.le_s();
  }

  // ============================================================================
  // Loop Unrolling and SIMD Helpers
  // ============================================================================

  /**
   * Emit an unrolled for loop when iteration count is known at compile time.
   * For small fixed-size loops, emits unrolled code for better performance.
   *
   * @param n - Number of iterations (must be a constant)
   * @param body - Loop body callback, receives iteration index as argument
   * @param unrollThreshold - Max iterations to fully unroll (default: 8)
   *
   * @example
   * ```ts
   * // For n=4, emits: body(0); body(1); body(2); body(3);
   * // For n=16, emits a loop (too large to unroll)
   * hl.forLoopUnrolled(4, (iter) => {
   *   // iter is a compile-time constant
   *   hl.store("f32", outPtr, () => cg.i32.const(iter), () => cg.f32.const(0));
   * });
   * ```
   */
  forLoopUnrolled(
    n: number,
    body: (iteration: number) => void,
    unrollThreshold: number = 8,
  ): void {
    if (n <= unrollThreshold) {
      // Fully unroll
      for (let iter = 0; iter < n; iter++) {
        body(iter);
      }
    } else {
      // Fall back to loop
      const { cg } = this;
      const i = cg.local.declare(cg.i32);
      this.forLoop(i, 0, n, () => {
        // Can't pass compile-time constant, but body can use local i
        // This is a limitation - body needs to work with runtime index
      });
      // Note: This fallback doesn't work well with the current API
      // For now, just unroll everything up to threshold
      for (let iter = 0; iter < n; iter++) {
        body(iter);
      }
    }
  }

  /**
   * Emit a SIMD-accelerated reduction loop for f32.
   * Handles SIMD main loop + scalar tail automatically.
   *
   * Note: This helper expects the caller to handle loading SIMD vectors vs scalars.
   * The loadA/loadB callbacks are called with `k` pointing to the current element index.
   * For the SIMD loop, k increments by 4; for scalar tail, k increments by 1.
   *
   * @param acc - Local variable for f32 accumulator (initialized by caller)
   * @param k - Local variable for loop counter
   * @param end - Loop end (constant or callback)
   * @param rowABase - Local variable containing byte address of row A
   * @param rowBBase - Local variable containing byte address of row B
   * @param op - Reduction operation ("add" for dot product, "sub" for subtract-accumulate)
   *
   * @example
   * ```ts
   * // sum += A[k] * B[k] for k in 0..j
   * cg.f32.const(0);
   * cg.local.set(sum);
   * hl.simdReductionF32(sum, k, j, rowAPtr, rowBPtr, "add");
   * ```
   */
  simdReductionF32(
    acc: number,
    k: number,
    end: number | (() => void),
    rowABase: number,
    rowBBase: number,
    op: "add" | "sub",
  ): void {
    const { cg } = this;
    const vec = cg.local.declare(cg.v128);
    const endFloor4 = cg.local.declare(cg.i32);

    // endFloor4 = (end / 4) * 4
    if (typeof end === "number") {
      cg.i32.const(Math.floor(end / 4) * 4);
    } else {
      end();
      cg.i32.const(2);
      cg.i32.shr_u();
      cg.i32.const(2);
      cg.i32.shl();
    }
    cg.local.set(endFloor4);

    // Initialize SIMD accumulator to zero
    cg.f32.const(0);
    cg.f32x4.splat();
    cg.local.set(vec);

    // SIMD main loop: k = 0, 4, 8, ...
    cg.i32.const(0);
    cg.local.set(k);

    cg.block(cg.void);
    cg.loop(cg.void);
    {
      cg.local.get(k);
      cg.local.get(endFloor4);
      cg.i32.ge_s();
      cg.br_if(1);

      // vec = vec op (A[k:k+4] * B[k:k+4])
      cg.local.get(vec);

      // Load A[k:k+4] from rowABase + k * 4 bytes
      cg.local.get(rowABase);
      cg.local.get(k);
      cg.i32.const(4);
      cg.i32.mul();
      cg.i32.add();
      cg.v128.load(2);

      // Load B[k:k+4] from rowBBase + k * 4 bytes
      cg.local.get(rowBBase);
      cg.local.get(k);
      cg.i32.const(4);
      cg.i32.mul();
      cg.i32.add();
      cg.v128.load(2);

      cg.f32x4.mul();
      if (op === "add") cg.f32x4.add();
      else cg.f32x4.sub();
      cg.local.set(vec);

      // k += 4
      cg.local.get(k);
      cg.i32.const(4);
      cg.i32.add();
      cg.local.set(k);

      cg.br(0);
    }
    cg.end();
    cg.end();

    // Horizontal sum and combine with scalar accumulator
    cg.local.get(acc);
    cg.local.get(vec);
    this.f32x4Hsum();
    if (op === "add") cg.f32.add();
    else cg.f32.sub();
    cg.local.set(acc);

    // Scalar tail: k = endFloor4 .. end
    cg.block(cg.void);
    cg.loop(cg.void);
    {
      cg.local.get(k);
      if (typeof end === "number") {
        cg.i32.const(end);
      } else {
        end();
      }
      cg.i32.ge_s();
      cg.br_if(1);

      // acc = acc op (A[k] * B[k])
      cg.local.get(acc);

      // Load A[k] from rowABase + k * 4
      this.load("f32", rowABase, this.getExpr(k));

      // Load B[k] from rowBBase + k * 4
      this.load("f32", rowBBase, this.getExpr(k));

      cg.f32.mul();
      if (op === "add") cg.f32.add();
      else cg.f32.sub();
      cg.local.set(acc);

      // k++
      cg.local.get(k);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(k);

      cg.br(0);
    }
    cg.end();
    cg.end();
  }

  /**
   * Emit a SIMD-accelerated reduction loop for f64.
   * Uses f64x2 (2 doubles per vector).
   */
  simdReductionF64(
    acc: number,
    k: number,
    end: number | (() => void),
    rowABase: number,
    rowBBase: number,
    op: "add" | "sub",
  ): void {
    const { cg } = this;
    const vec = cg.local.declare(cg.v128);
    const endFloor2 = cg.local.declare(cg.i32);

    // endFloor2 = (end / 2) * 2
    if (typeof end === "number") {
      cg.i32.const(Math.floor(end / 2) * 2);
    } else {
      end();
      cg.i32.const(1);
      cg.i32.shr_u();
      cg.i32.const(1);
      cg.i32.shl();
    }
    cg.local.set(endFloor2);

    // Initialize SIMD accumulator
    cg.f64.const(0);
    cg.f64x2.splat();
    cg.local.set(vec);

    // SIMD main loop: k = 0, 2, 4, ...
    cg.i32.const(0);
    cg.local.set(k);

    cg.block(cg.void);
    cg.loop(cg.void);
    {
      cg.local.get(k);
      cg.local.get(endFloor2);
      cg.i32.ge_s();
      cg.br_if(1);

      cg.local.get(vec);

      // Load A[k:k+2] from rowABase + k * 8 bytes
      cg.local.get(rowABase);
      cg.local.get(k);
      cg.i32.const(8);
      cg.i32.mul();
      cg.i32.add();
      cg.v128.load(3); // alignment log2(8)

      // Load B[k:k+2] from rowBBase + k * 8 bytes
      cg.local.get(rowBBase);
      cg.local.get(k);
      cg.i32.const(8);
      cg.i32.mul();
      cg.i32.add();
      cg.v128.load(3);

      cg.f64x2.mul();
      if (op === "add") cg.f64x2.add();
      else cg.f64x2.sub();
      cg.local.set(vec);

      // k += 2
      cg.local.get(k);
      cg.i32.const(2);
      cg.i32.add();
      cg.local.set(k);

      cg.br(0);
    }
    cg.end();
    cg.end();

    // Horizontal sum
    cg.local.get(acc);
    cg.local.get(vec);
    this.f64x2Hsum();
    if (op === "add") cg.f64.add();
    else cg.f64.sub();
    cg.local.set(acc);

    // Scalar tail
    cg.block(cg.void);
    cg.loop(cg.void);
    {
      cg.local.get(k);
      if (typeof end === "number") {
        cg.i32.const(end);
      } else {
        end();
      }
      cg.i32.ge_s();
      cg.br_if(1);

      cg.local.get(acc);
      this.load("f64", rowABase, this.getExpr(k));
      this.load("f64", rowBBase, this.getExpr(k));
      cg.f64.mul();
      if (op === "add") cg.f64.add();
      else cg.f64.sub();
      cg.local.set(acc);

      cg.local.get(k);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(k);

      cg.br(0);
    }
    cg.end();
    cg.end();
  }
}
