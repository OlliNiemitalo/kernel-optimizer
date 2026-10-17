/**
 * Triangular solve using wasmblr with size specialization.
 *
 * Solves A @ X = B for X where A is triangular.
 * Size specialization bakes in n, batchRows, unitDiagonal, and lower at compile time.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate size-specialized upper-triangular solve for single vector.
 * Back-substitution: solve from bottom to top.
 */
function genSolveUpperSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  unitDiagonal: boolean,
): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // solve_upper(aPtr, bPtr, xPtr)
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const bPtr = 1;
    const xPtr = 2;

    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ty);

    // for i = n-1 down to 0
    hl.forLoopDown(i, n, 0, () => {
      // sum = b[i]
      hl.load(dtype, bPtr, hl.getExpr(i));
      cg.local.set(sum);

      // for j = i+1 to n
      hl.forLoop(
        j,
        () => {
          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
        },
        n,
        () => {
          // sum -= A[i,j] * x[j]
          cg.local.get(sum);
          // A[i,j] = a[i * n + j]
          hl.load(dtype, aPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
          });
          hl.load(dtype, xPtr, hl.getExpr(j));
          hl.binOp(dtype, "mul");
          hl.binOp(dtype, "sub");
          cg.local.set(sum);
        },
      );

      // x[i] = sum or sum / A[i,i] depending on unitDiagonal
      if (unitDiagonal) {
        hl.store(dtype, xPtr, hl.getExpr(i), () => cg.local.get(sum));
      } else {
        hl.store(dtype, xPtr, hl.getExpr(i), () => {
          cg.local.get(sum);
          // A[i,i] = a[i * n + i]
          hl.load(dtype, aPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(i);
            cg.i32.add();
          });
          hl.binOp(dtype, "div");
        });
      }
    });
  });
}

/**
 * Generate size-specialized lower-triangular solve for single vector.
 * Forward-substitution: solve from top to bottom.
 */
function genSolveLowerSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  unitDiagonal: boolean,
): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // solve_lower(aPtr, bPtr, xPtr)
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const bPtr = 1;
    const xPtr = 2;

    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ty);

    // for i = 0 to n
    hl.forLoop(i, 0, n, () => {
      // sum = b[i]
      hl.load(dtype, bPtr, hl.getExpr(i));
      cg.local.set(sum);

      // for j = 0 to i
      hl.forLoop(j, 0, hl.getExpr(i), () => {
        // sum -= A[i,j] * x[j]
        cg.local.get(sum);
        // A[i,j] = a[i * n + j]
        hl.load(dtype, aPtr, () => {
          cg.local.get(i);
          cg.i32.const(n);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.add();
        });
        hl.load(dtype, xPtr, hl.getExpr(j));
        hl.binOp(dtype, "mul");
        hl.binOp(dtype, "sub");
        cg.local.set(sum);
      });

      // x[i] = sum or sum / A[i,i] depending on unitDiagonal
      if (unitDiagonal) {
        hl.store(dtype, xPtr, hl.getExpr(i), () => cg.local.get(sum));
      } else {
        hl.store(dtype, xPtr, hl.getExpr(i), () => {
          cg.local.get(sum);
          // A[i,i] = a[i * n + i]
          hl.load(dtype, aPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(i);
            cg.i32.add();
          });
          hl.binOp(dtype, "div");
        });
      }
    });
  });
}

/**
 * Generate size-specialized batched triangular solve.
 */
function genTriangularSolveBatchedSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  batchRows: number,
  dtype: "f32" | "f64",
  solveFunc: number,
): number {
  const elemSize = dtype === "f32" ? 4 : 8;
  const matrixBytes = n * n * elemSize;
  const vectorBytes = n * elemSize;

  // triangular_solve_batched(aPtr, bPtr, xPtr, numBatches)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const bPtr = 1;
    const xPtr = 2;
    const numBatches = 3;

    const batch = cg.local.declare(cg.i32);
    const row = cg.local.declare(cg.i32);
    const idx = cg.local.declare(cg.i32);

    hl.forLoop(batch, 0, hl.getExpr(numBatches), () => {
      hl.forLoop(row, 0, batchRows, () => {
        // idx = batch * batchRows + row
        cg.local.get(batch);
        cg.i32.const(batchRows);
        cg.i32.mul();
        cg.local.get(row);
        cg.i32.add();
        cg.local.set(idx);

        // Call solve: solve(aPtr + batch * matrixBytes, bPtr + idx * vectorBytes, xPtr + idx * vectorBytes)
        cg.local.get(aPtr);
        cg.local.get(batch);
        cg.i32.const(matrixBytes);
        cg.i32.mul();
        cg.i32.add();

        cg.local.get(bPtr);
        cg.local.get(idx);
        cg.i32.const(vectorBytes);
        cg.i32.mul();
        cg.i32.add();

        cg.local.get(xPtr);
        cg.local.get(idx);
        cg.i32.const(vectorBytes);
        cg.i32.mul();
        cg.i32.add();

        cg.call(solveFunc);
      });
    });
  });
}

/**
 * Build a size-specialized triangular solve WASM module.
 *
 * @param n - Matrix size
 * @param batchRows - Number of rows in B matrix
 * @param dtype - f32 or f64
 * @param unitDiagonal - Whether diagonal is unit (1s)
 * @param lower - Whether lower triangular (else upper)
 *
 * Exports:
 * - triangular_solve(aPtr, bPtr, xPtr) - single solve
 * - triangular_solve_batched(aPtr, bPtr, xPtr, numBatches) - batched solve
 */
export function buildTriangularSolveModuleSized(
  n: number,
  batchRows: number,
  dtype: "f32" | "f64",
  unitDiagonal: boolean,
  lower: boolean,
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  configureMemoryImport(cg);

  const solveFunc = lower
    ? genSolveLowerSized(cg, hl, n, dtype, unitDiagonal)
    : genSolveUpperSized(cg, hl, n, dtype, unitDiagonal);

  const batchedFunc = genTriangularSolveBatchedSized(
    cg,
    hl,
    n,
    batchRows,
    dtype,
    solveFunc,
  );

  cg.export(solveFunc, "triangular_solve");
  cg.export(batchedFunc, "triangular_solve_batched");

  return cg.finish();
}
