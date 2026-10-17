/**
 * Cholesky decomposition using wasmblr with SIMD (f32x4) optimization.
 *
 * This implements the Cholesky-Banachiewicz algorithm: A = L @ L^T
 * Uses SIMD for the inner dot product loop when dimension >= 4.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate SIMD-optimized Cholesky for f32.
 * Uses f32x4 for the inner dot product loop.
 */
function genCholeskySimd(cg: CodeGenerator, n: number): number {
  const hl = new WasmHl(cg);
  const nn = n * n;

  // cholesky(inPtr, outPtr)
  return cg.function([cg.i32, cg.i32], [], () => {
    const inPtr = 0;
    const outPtr = 1;

    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const sum = cg.local.declare(cg.f32);
    const idx = cg.local.declare(cg.i32);
    const rowI = cg.local.declare(cg.i32); // outPtr + i * n * 4
    const rowJ = cg.local.declare(cg.i32); // outPtr + j * n * 4
    const jFloor4 = cg.local.declare(cg.i32);
    const vec = cg.local.declare(cg.v128);

    // Zero output matrix
    hl.forLoop(idx, 0, nn, () => {
      hl.store("f32", outPtr, hl.getExpr(idx), () => hl.const("f32", 0));
    });

    // Cholesky-Banachiewicz algorithm
    hl.forLoop(i, 0, n, () => {
      // rowI = outPtr + i * n * 4 (byte offset to row i)
      cg.local.get(outPtr);
      cg.local.get(i);
      cg.i32.const(n * 4);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(rowI);

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
          // rowJ = outPtr + j * n * 4
          cg.local.get(outPtr);
          cg.local.get(j);
          cg.i32.const(n * 4);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(rowJ);

          // sum = A[i, j]
          hl.load("f32", inPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
          });
          cg.local.set(sum);

          // jFloor4 = (j / 4) * 4
          cg.local.get(j);
          cg.i32.const(2);
          cg.i32.shr_u(); // j >> 2 = j / 4
          cg.i32.const(2);
          cg.i32.shl(); // * 4
          cg.local.set(jFloor4);

          // SIMD loop: k = 0, 4, 8, ... < jFloor4
          // Zero the accumulator
          cg.f32.const(0);
          cg.f32x4.splat();
          cg.local.set(vec);

          // Manual while loop stepping by 4
          cg.i32.const(0);
          cg.local.set(k);

          cg.block(cg.void);
          cg.loop(cg.void);
          {
            // if (k >= jFloor4) break
            cg.local.get(k);
            cg.local.get(jFloor4);
            cg.i32.ge_s();
            cg.br_if(1);

            // vec += L[i, k:k+4] * L[j, k:k+4]
            cg.local.get(vec);

            // Load L[i, k:k+4] from rowI + k * 4
            cg.local.get(rowI);
            cg.local.get(k);
            cg.i32.const(4);
            cg.i32.mul();
            cg.i32.add();
            cg.v128.load(2); // alignment log2(4)

            // Load L[j, k:k+4] from rowJ + k * 4
            cg.local.get(rowJ);
            cg.local.get(k);
            cg.i32.const(4);
            cg.i32.mul();
            cg.i32.add();
            cg.v128.load(2);

            // Multiply
            cg.f32x4.mul();

            // Add to accumulator
            cg.f32x4.add();
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

          // Horizontal sum of vec and subtract from sum
          cg.local.get(sum);
          cg.local.get(vec);
          hl.f32x4Hsum();
          cg.f32.sub();
          cg.local.set(sum);

          // Scalar tail: k = jFloor4 .. j
          hl.forLoop(k, hl.getExpr(jFloor4), hl.getExpr(j), () => {
            cg.local.get(sum);
            // L[i, k] - rowI is byte address, k is element index
            hl.load("f32", rowI, hl.getExpr(k));
            // L[j, k]
            hl.load("f32", rowJ, hl.getExpr(k));
            cg.f32.mul();
            cg.f32.sub();
            cg.local.set(sum);
          });

          // Store result
          cg.local.get(i);
          cg.local.get(j);
          cg.i32.eq();
          hl.ifElse(
            cg.void,
            () => {
              // Diagonal: L[i,i] = sqrt(sum)
              hl.store(
                "f32",
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
                  cg.f32.sqrt();
                },
              );
            },
            () => {
              // Off-diagonal: L[i,j] = sum / L[j,j]
              hl.store(
                "f32",
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
                  // L[j, j]
                  hl.load("f32", outPtr, () => {
                    cg.local.get(j);
                    cg.i32.const(n);
                    cg.i32.mul();
                    cg.local.get(j);
                    cg.i32.add();
                  });
                  cg.f32.div();
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
 * Generate batched wrapper.
 */
function genCholeskyBatchedSized(
  cg: CodeGenerator,
  n: number,
  singleFunc: number,
): number {
  const hl = new WasmHl(cg);
  const matrixBytes = n * n * 4;

  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const inPtr = 0;
    const outPtr = 1;
    const batch = 2;
    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batch), () => {
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
 * Build SIMD-optimized Cholesky module (f32 only).
 */
export function buildCholeskySimdModule(n: number): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  configureMemoryImport(cg);

  const singleFunc = genCholeskySimd(cg, n);
  const batchedFunc = genCholeskyBatchedSized(cg, n, singleFunc);

  cg.export(singleFunc, "cholesky");
  cg.export(batchedFunc, "cholesky_batched");

  return cg.finish();
}
