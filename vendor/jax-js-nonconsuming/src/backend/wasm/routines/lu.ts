/**
 * LU decomposition using wasmblr with size specialization.
 *
 * Computes LU factorization with partial pivoting: P @ A = L @ U
 * where L is lower-triangular with unit diagonal, U is upper-triangular.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate size-specialized LU decomposition for single matrix.
 */
function genLUSized(
  cg: CodeGenerator,
  hl: WasmHl,
  m: number,
  n: number,
  dtype: "f32" | "f64",
): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;
  const r = Math.min(m, n);

  // lu(aPtr, luPtr, pivPtr, permPtr)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const luPtr = 1;
    const pivPtr = 2;
    const permPtr = 3;

    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const col = cg.local.declare(cg.i32);
    const maxVal = cg.local.declare(ty);
    const maxRow = cg.local.declare(cg.i32);
    const val = cg.local.declare(ty);
    const tmp = cg.local.declare(ty);
    const tmpP = cg.local.declare(cg.i32);
    const diag = cg.local.declare(ty);
    const factor = cg.local.declare(ty);

    // Copy A to LU
    hl.forLoop(j, 0, m * n, () => {
      hl.store(dtype, luPtr, hl.getExpr(j), () => {
        hl.load(dtype, aPtr, hl.getExpr(j));
      });
    });

    // Initialize permutation
    hl.forLoop(i, 0, m, () => {
      hl.store("i32", permPtr, hl.getExpr(i), () => cg.local.get(i));
    });

    // Main loop
    hl.forLoop(j, 0, r, () => {
      // Find pivot: max |LU[i,j]| for i >= j
      // LU[j,j] = lu[j * n + j]
      hl.load(dtype, luPtr, () => {
        cg.local.get(j);
        cg.i32.const(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
      });
      if (dtype === "f32") cg.f32.abs();
      else cg.f64.abs();
      cg.local.set(maxVal);
      cg.local.get(j);
      cg.local.set(maxRow);

      hl.forLoop(
        i,
        () => {
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
        },
        m,
        () => {
          // LU[i,j] = lu[i * n + j]
          hl.load(dtype, luPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
          });
          if (dtype === "f32") cg.f32.abs();
          else cg.f64.abs();
          cg.local.set(val);

          // if val > maxVal
          cg.local.get(val);
          cg.local.get(maxVal);
          if (dtype === "f32") cg.f32.gt();
          else cg.f64.gt();
          hl.ifElse(cg.void, () => {
            cg.local.get(val);
            cg.local.set(maxVal);
            cg.local.get(i);
            cg.local.set(maxRow);
          });
        },
      );

      // Store pivot index
      hl.store("i32", pivPtr, hl.getExpr(j), () => cg.local.get(maxRow));

      // Swap rows if needed
      cg.local.get(maxRow);
      cg.local.get(j);
      cg.i32.ne();
      hl.ifElse(cg.void, () => {
        // Swap LU rows
        hl.forLoop(col, 0, n, () => {
          // tmp = LU[j, col] = lu[j * n + col]
          hl.load(dtype, luPtr, () => {
            cg.local.get(j);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(col);
            cg.i32.add();
          });
          cg.local.set(tmp);

          // LU[j, col] = LU[maxRow, col]
          hl.store(
            dtype,
            luPtr,
            () => {
              cg.local.get(j);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            },
            () => {
              // LU[maxRow, col] = lu[maxRow * n + col]
              hl.load(dtype, luPtr, () => {
                cg.local.get(maxRow);
                cg.i32.const(n);
                cg.i32.mul();
                cg.local.get(col);
                cg.i32.add();
              });
            },
          );

          // LU[maxRow, col] = tmp
          hl.store(
            dtype,
            luPtr,
            () => {
              cg.local.get(maxRow);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            },
            () => cg.local.get(tmp),
          );
        });

        // Swap permutation
        hl.load("i32", permPtr, hl.getExpr(j));
        cg.local.set(tmpP);
        hl.store("i32", permPtr, hl.getExpr(j), () => {
          hl.load("i32", permPtr, hl.getExpr(maxRow));
        });
        hl.store("i32", permPtr, hl.getExpr(maxRow), () => cg.local.get(tmpP));
      });

      // Update L and U
      // diag = LU[j,j]
      hl.load(dtype, luPtr, () => {
        cg.local.get(j);
        cg.i32.const(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
      });
      cg.local.set(diag);

      // if diag != 0
      cg.local.get(diag);
      hl.const(dtype, 0);
      if (dtype === "f32") cg.f32.ne();
      else cg.f64.ne();
      hl.ifElse(cg.void, () => {
        hl.forLoop(
          i,
          () => {
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
          },
          m,
          () => {
            // factor = LU[i,j] / diag
            hl.load(dtype, luPtr, () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(j);
              cg.i32.add();
            });
            cg.local.get(diag);
            hl.binOp(dtype, "div");
            cg.local.set(factor);

            // LU[i,j] = factor (store L)
            hl.store(
              dtype,
              luPtr,
              () => {
                cg.local.get(i);
                cg.i32.const(n);
                cg.i32.mul();
                cg.local.get(j);
                cg.i32.add();
              },
              () => cg.local.get(factor),
            );

            // Update U: LU[i,col] -= factor * LU[j,col]
            hl.forLoop(
              col,
              () => {
                cg.local.get(j);
                cg.i32.const(1);
                cg.i32.add();
              },
              n,
              () => {
                hl.store(
                  dtype,
                  luPtr,
                  () => {
                    cg.local.get(i);
                    cg.i32.const(n);
                    cg.i32.mul();
                    cg.local.get(col);
                    cg.i32.add();
                  },
                  () => {
                    hl.load(dtype, luPtr, () => {
                      cg.local.get(i);
                      cg.i32.const(n);
                      cg.i32.mul();
                      cg.local.get(col);
                      cg.i32.add();
                    });
                    cg.local.get(factor);
                    hl.load(dtype, luPtr, () => {
                      cg.local.get(j);
                      cg.i32.const(n);
                      cg.i32.mul();
                      cg.local.get(col);
                      cg.i32.add();
                    });
                    hl.binOp(dtype, "mul");
                    hl.binOp(dtype, "sub");
                  },
                );
              },
            );
          },
        );
      });
    });
  });
}

/**
 * Generate size-specialized batched LU decomposition.
 */
function genLUBatchedSized(
  cg: CodeGenerator,
  hl: WasmHl,
  m: number,
  n: number,
  dtype: "f32" | "f64",
  singleFunc: number,
): number {
  const elemSize = dtype === "f32" ? 4 : 8;
  const r = Math.min(m, n);
  const matrixBytes = m * n * elemSize;
  const pivBytes = r * 4;
  const permBytes = m * 4;

  // lu_batched(aPtr, luPtr, pivPtr, permPtr, batchSize)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const luPtr = 1;
    const pivPtr = 2;
    const permPtr = 3;
    const batchSize = 4;

    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batchSize), () => {
      // Call single LU
      cg.local.get(aPtr);
      cg.local.get(b);
      cg.i32.const(matrixBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(luPtr);
      cg.local.get(b);
      cg.i32.const(matrixBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(pivPtr);
      cg.local.get(b);
      cg.i32.const(pivBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(permPtr);
      cg.local.get(b);
      cg.i32.const(permBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.call(singleFunc);
    });
  });
}

/**
 * Build a size-specialized LU WASM module.
 *
 * @param m - Number of rows
 * @param n - Number of columns
 * @param dtype - f32 or f64
 *
 * Exports:
 * - lu(aPtr, luPtr, pivPtr, permPtr) - single matrix
 * - lu_batched(aPtr, luPtr, pivPtr, permPtr, batchSize) - multiple matrices
 */
export function buildLUModuleSized(
  m: number,
  n: number,
  dtype: "f32" | "f64",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  configureMemoryImport(cg);

  const luFunc = genLUSized(cg, hl, m, n, dtype);
  const luBatchedFunc = genLUBatchedSized(cg, hl, m, n, dtype, luFunc);

  cg.export(luFunc, "lu");
  cg.export(luBatchedFunc, "lu_batched");

  return cg.finish();
}
