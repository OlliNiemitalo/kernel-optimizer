/**
 * Cholesky decomposition using wasmblr with size specialization.
 *
 * This implements the Cholesky-Banachiewicz algorithm: A = L @ L^T
 * Size specialization allows the compiler to:
 * - Use constants for loop bounds (better branch prediction)
 * - Potentially unroll inner loops for small matrices
 * - Eliminate runtime size parameter overhead
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate size-specialized Cholesky decomposition function.
 *
 * @param cg - CodeGenerator instance
 * @param n - Matrix size (compile-time constant)
 * @param dtype - f32 or f64
 * @returns Function index
 */
function genCholeskySized(
  cg: CodeGenerator,
  n: number,
  dtype: "f32" | "f64",
): number {
  const hl = new WasmHl(cg);
  const ty = dtype === "f32" ? cg.f32 : cg.f64;
  const nn = n * n;

  // cholesky(inPtr, outPtr)
  return cg.function([cg.i32, cg.i32], [], () => {
    const inPtr = 0;
    const outPtr = 1;

    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ty);
    const idx = cg.local.declare(cg.i32);

    // Zero output matrix (upper triangle won't be written by algorithm)
    hl.forLoop(idx, 0, nn, () => {
      hl.store(dtype, outPtr, hl.getExpr(idx), () => hl.const(dtype, 0));
    });

    // Cholesky-Banachiewicz algorithm
    // https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation
    hl.forLoop(i, 0, n, () => {
      // for j in 0..=i
      hl.forLoop(
        j,
        0,
        () => {
          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
        },
        () => {
          // sum = A[i, j]
          // index = i * n + j
          hl.load(dtype, inPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
          });
          cg.local.set(sum);

          // sum -= sum(L[i,k] * L[j,k] for k in 0..j)
          hl.forLoop(k, 0, hl.getExpr(j), () => {
            cg.local.get(sum);
            // L[i, k] = out[i * n + k]
            hl.load(dtype, outPtr, () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(k);
              cg.i32.add();
            });
            // L[j, k] = out[j * n + k]
            hl.load(dtype, outPtr, () => {
              cg.local.get(j);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(k);
              cg.i32.add();
            });
            hl.binOp(dtype, "mul");
            hl.binOp(dtype, "sub");
            cg.local.set(sum);
          });

          // if (i == j) L[i,j] = sqrt(sum) else L[i,j] = sum / L[j,j]
          cg.local.get(i);
          cg.local.get(j);
          cg.i32.eq();
          hl.ifElse(
            cg.void,
            () => {
              // Diagonal: L[i,i] = sqrt(sum)
              hl.store(
                dtype,
                outPtr,
                () => {
                  cg.local.get(i);
                  cg.i32.const(n);
                  cg.i32.mul();
                  cg.local.get(j);
                  cg.i32.add();
                },
                () => {
                  cg.local.get(sum);
                  hl.sqrt(dtype);
                },
              );
            },
            () => {
              // Off-diagonal: L[i,j] = sum / L[j,j]
              hl.store(
                dtype,
                outPtr,
                () => {
                  cg.local.get(i);
                  cg.i32.const(n);
                  cg.i32.mul();
                  cg.local.get(j);
                  cg.i32.add();
                },
                () => {
                  cg.local.get(sum);
                  // L[j, j] = out[j * n + j]
                  hl.load(dtype, outPtr, () => {
                    cg.local.get(j);
                    cg.i32.const(n);
                    cg.i32.mul();
                    cg.local.get(j);
                    cg.i32.add();
                  });
                  hl.binOp(dtype, "div");
                },
              );
            },
          );
        },
      );
    });
  });
}

/**
 * Generate size-specialized batched Cholesky function.
 *
 * @param cg - CodeGenerator instance
 * @param n - Matrix size (compile-time constant)
 * @param dtype - f32 or f64
 * @param singleFunc - Function index of single-matrix cholesky
 * @returns Function index
 */
function genCholeskyBatchedSized(
  cg: CodeGenerator,
  n: number,
  dtype: "f32" | "f64",
  singleFunc: number,
): number {
  const hl = new WasmHl(cg);
  const elemSize = dtype === "f32" ? 4 : 8;
  const matrixBytes = n * n * elemSize;

  // cholesky_batched(inPtr, outPtr, batch)
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const inPtr = 0;
    const outPtr = 1;
    const batch = 2;

    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batch), () => {
      // Call single cholesky: cholesky(inPtr + b * matrixBytes, outPtr + b * matrixBytes)
      cg.local.get(inPtr);
      cg.local.get(b);
      cg.i32.const(matrixBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(outPtr);
      cg.local.get(b);
      cg.i32.const(matrixBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.call(singleFunc);
    });
  });
}

/**
 * Build a size-specialized Cholesky WASM module.
 *
 * @param n - Matrix size
 * @param dtype - f32 or f64
 *
 * Exports:
 * - cholesky(inPtr, outPtr) - single matrix
 * - cholesky_batched(inPtr, outPtr, batch) - multiple matrices
 */
export function buildCholeskyModuleSized(
  n: number,
  dtype: "f32" | "f64",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  configureMemoryImport(cg);

  const singleFunc = genCholeskySized(cg, n, dtype);
  const batchedFunc = genCholeskyBatchedSized(cg, n, dtype, singleFunc);

  cg.export(singleFunc, "cholesky");
  cg.export(batchedFunc, "cholesky_batched");

  return cg.finish();
}
