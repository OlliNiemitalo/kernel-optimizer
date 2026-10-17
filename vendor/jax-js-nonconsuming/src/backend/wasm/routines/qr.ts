/**
 * QR decomposition using wasmblr with size specialization.
 *
 * Computes thin QR factorization via Householder reflections: A = Q @ R
 * where Q is m×k orthogonal columns and R is k×n upper triangular, k = min(m,n).
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate size-specialized QR decomposition for single matrix.
 * Takes 4 arguments: (aPtr, qPtr, rPtr, workPtr)
 * workPtr must point to m×n scratch space.
 */
function genQRSizedWithWork(
  cg: CodeGenerator,
  hl: WasmHl,
  m: number,
  n: number,
  dtype: "f32" | "f64",
): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;
  const k = Math.min(m, n);

  // qr(aPtr, qPtr, rPtr, workPtr)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const qPtr = 1;
    const rPtr = 2;
    const workPtr = 3;

    const idx = cg.local.declare(cg.i32);
    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const col = cg.local.declare(cg.i32);
    const norm = cg.local.declare(ty);
    const alpha = cg.local.declare(ty);
    const tau = cg.local.declare(ty);
    const dotVar = cg.local.declare(ty);
    const vNormSq = cg.local.declare(ty);
    const v0 = cg.local.declare(ty);
    const val = cg.local.declare(ty);

    // Copy A to work buffer (m × n)
    hl.forLoop(idx, 0, m * n, () => {
      hl.store(dtype, workPtr, hl.getExpr(idx), () => {
        hl.load(dtype, aPtr, hl.getExpr(idx));
      });
    });

    // Initialize Q as identity (m × k)
    // Zero Q first
    hl.forLoop(idx, 0, m * k, () => {
      hl.store(dtype, qPtr, hl.getExpr(idx), () => hl.const(dtype, 0));
    });
    // Set diagonal to 1
    hl.forLoop(idx, 0, Math.min(m, k), () => {
      // Q[idx, idx] = 1: index = idx * k + idx
      hl.store(
        dtype,
        qPtr,
        () => {
          cg.local.get(idx);
          cg.i32.const(k);
          cg.i32.mul();
          cg.local.get(idx);
          cg.i32.add();
        },
        () => hl.const(dtype, 1),
      );
    });

    // Householder QR
    hl.forLoop(j, 0, k, () => {
      // Compute norm of work[j:m, j]
      hl.const(dtype, 0);
      cg.local.set(norm);
      hl.forLoop(i, hl.getExpr(j), m, () => {
        // val = work[i * n + j]
        hl.load(dtype, workPtr, () => {
          cg.local.get(i);
          cg.i32.const(n);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.add();
        });
        cg.local.set(val);
        // norm += val * val
        cg.local.get(norm);
        cg.local.get(val);
        cg.local.get(val);
        hl.binOp(dtype, "mul");
        hl.binOp(dtype, "add");
        cg.local.set(norm);
      });
      // norm = sqrt(norm)
      cg.local.get(norm);
      hl.sqrt(dtype);
      cg.local.set(norm);

      // Skip if norm == 0
      cg.local.get(norm);
      hl.const(dtype, 0);
      if (dtype === "f32") cg.f32.eq();
      else cg.f64.eq();
      cg.if(cg.void);
      // norm == 0, skip this column
      cg.else();

      // sign = work[j, j] >= 0 ? 1 : -1
      // alpha = -sign * norm
      // We compute: if work[j,j] >= 0, alpha = -norm, else alpha = norm
      hl.load(dtype, workPtr, () => {
        cg.local.get(j);
        cg.i32.const(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
      });
      hl.const(dtype, 0);
      if (dtype === "f32") cg.f32.ge();
      else cg.f64.ge();
      hl.ifElse(
        ty,
        () => {
          // sign >= 0: alpha = -norm
          cg.local.get(norm);
          if (dtype === "f32") cg.f32.neg();
          else cg.f64.neg();
        },
        () => {
          // sign < 0: alpha = norm
          cg.local.get(norm);
        },
      );
      cg.local.set(alpha);

      // v[0] = work[j,j] - alpha (store in work[j,j] temporarily)
      hl.load(dtype, workPtr, () => {
        cg.local.get(j);
        cg.i32.const(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
      });
      cg.local.get(alpha);
      hl.binOp(dtype, "sub");
      cg.local.set(v0);

      // v[i-j] for i > j is work[i, j] (already in place)

      // Compute vNormSq = v0*v0 + sum(work[i,j]^2 for i > j)
      cg.local.get(v0);
      cg.local.get(v0);
      hl.binOp(dtype, "mul");
      cg.local.set(vNormSq);
      hl.forLoop(
        i,
        () => {
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
        },
        m,
        () => {
          hl.load(dtype, workPtr, () => {
            cg.local.get(i);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
          });
          cg.local.set(val);
          cg.local.get(vNormSq);
          cg.local.get(val);
          cg.local.get(val);
          hl.binOp(dtype, "mul");
          hl.binOp(dtype, "add");
          cg.local.set(vNormSq);
        },
      );

      // Skip if vNormSq == 0
      cg.local.get(vNormSq);
      hl.const(dtype, 0);
      if (dtype === "f32") cg.f32.eq();
      else cg.f64.eq();
      cg.if(cg.void);
      // vNormSq == 0, skip
      cg.else();

      // tau = 2 / vNormSq
      hl.const(dtype, 2);
      cg.local.get(vNormSq);
      hl.binOp(dtype, "div");
      cg.local.set(tau);

      // Apply Householder to work[j:m, j:n]
      // For each col:
      //   dot = v0 * work[j,col] + sum(work[i,col] * work[i,j] for i > j)
      //   work[j,col] -= tau * v0 * dot
      //   work[i,col] -= tau * work[i,j] * dot  for i > j
      hl.forLoop(col, hl.getExpr(j), n, () => {
        // dot = v0 * work[j, col]
        cg.local.get(v0);
        hl.load(dtype, workPtr, () => {
          cg.local.get(j);
          cg.i32.const(n);
          cg.i32.mul();
          cg.local.get(col);
          cg.i32.add();
        });
        hl.binOp(dtype, "mul");
        cg.local.set(dotVar);

        // dot += sum(work[i,j] * work[i,col] for i in j+1..m)
        hl.forLoop(
          i,
          () => {
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
          },
          m,
          () => {
            cg.local.get(dotVar);
            hl.load(dtype, workPtr, () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(j);
              cg.i32.add();
            });
            hl.load(dtype, workPtr, () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            });
            hl.binOp(dtype, "mul");
            hl.binOp(dtype, "add");
            cg.local.set(dotVar);
          },
        );

        // work[j, col] -= tau * v0 * dot
        hl.store(
          dtype,
          workPtr,
          () => {
            cg.local.get(j);
            cg.i32.const(n);
            cg.i32.mul();
            cg.local.get(col);
            cg.i32.add();
          },
          () => {
            hl.load(dtype, workPtr, () => {
              cg.local.get(j);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            });
            cg.local.get(tau);
            cg.local.get(v0);
            hl.binOp(dtype, "mul");
            cg.local.get(dotVar);
            hl.binOp(dtype, "mul");
            hl.binOp(dtype, "sub");
          },
        );

        // work[i, col] -= tau * work[i,j] * dot  for i in j+1..m
        hl.forLoop(
          i,
          () => {
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
          },
          m,
          () => {
            hl.store(
              dtype,
              workPtr,
              () => {
                cg.local.get(i);
                cg.i32.const(n);
                cg.i32.mul();
                cg.local.get(col);
                cg.i32.add();
              },
              () => {
                hl.load(dtype, workPtr, () => {
                  cg.local.get(i);
                  cg.i32.const(n);
                  cg.i32.mul();
                  cg.local.get(col);
                  cg.i32.add();
                });
                cg.local.get(tau);
                hl.load(dtype, workPtr, () => {
                  cg.local.get(i);
                  cg.i32.const(n);
                  cg.i32.mul();
                  cg.local.get(j);
                  cg.i32.add();
                });
                hl.binOp(dtype, "mul");
                cg.local.get(dotVar);
                hl.binOp(dtype, "mul");
                hl.binOp(dtype, "sub");
              },
            );
          },
        );
      });

      // Apply Householder to Q[j:m, 0:k]
      hl.forLoop(col, 0, k, () => {
        // dot = v0 * Q[j, col]
        cg.local.get(v0);
        hl.load(dtype, qPtr, () => {
          cg.local.get(j);
          cg.i32.const(k);
          cg.i32.mul();
          cg.local.get(col);
          cg.i32.add();
        });
        hl.binOp(dtype, "mul");
        cg.local.set(dotVar);

        // dot += sum(work[i,j] * Q[i,col] for i in j+1..m)
        hl.forLoop(
          i,
          () => {
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
          },
          m,
          () => {
            cg.local.get(dotVar);
            hl.load(dtype, workPtr, () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(j);
              cg.i32.add();
            });
            hl.load(dtype, qPtr, () => {
              cg.local.get(i);
              cg.i32.const(k);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            });
            hl.binOp(dtype, "mul");
            hl.binOp(dtype, "add");
            cg.local.set(dotVar);
          },
        );

        // Q[j, col] -= tau * v0 * dot
        hl.store(
          dtype,
          qPtr,
          () => {
            cg.local.get(j);
            cg.i32.const(k);
            cg.i32.mul();
            cg.local.get(col);
            cg.i32.add();
          },
          () => {
            hl.load(dtype, qPtr, () => {
              cg.local.get(j);
              cg.i32.const(k);
              cg.i32.mul();
              cg.local.get(col);
              cg.i32.add();
            });
            cg.local.get(tau);
            cg.local.get(v0);
            hl.binOp(dtype, "mul");
            cg.local.get(dotVar);
            hl.binOp(dtype, "mul");
            hl.binOp(dtype, "sub");
          },
        );

        // Q[i, col] -= tau * work[i,j] * dot  for i in j+1..m
        hl.forLoop(
          i,
          () => {
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
          },
          m,
          () => {
            hl.store(
              dtype,
              qPtr,
              () => {
                cg.local.get(i);
                cg.i32.const(k);
                cg.i32.mul();
                cg.local.get(col);
                cg.i32.add();
              },
              () => {
                hl.load(dtype, qPtr, () => {
                  cg.local.get(i);
                  cg.i32.const(k);
                  cg.i32.mul();
                  cg.local.get(col);
                  cg.i32.add();
                });
                cg.local.get(tau);
                hl.load(dtype, workPtr, () => {
                  cg.local.get(i);
                  cg.i32.const(n);
                  cg.i32.mul();
                  cg.local.get(j);
                  cg.i32.add();
                });
                hl.binOp(dtype, "mul");
                cg.local.get(dotVar);
                hl.binOp(dtype, "mul");
                hl.binOp(dtype, "sub");
              },
            );
          },
        );
      });

      // After applying H, set work[j,j] to alpha (the diagonal of R)
      hl.store(
        dtype,
        workPtr,
        () => {
          cg.local.get(j);
          cg.i32.const(n);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.add();
        },
        () => cg.local.get(alpha),
      );

      // Zero out work[i,j] for i > j (below diagonal — these held the v components)
      hl.forLoop(
        i,
        () => {
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
        },
        m,
        () => {
          hl.store(
            dtype,
            workPtr,
            () => {
              cg.local.get(i);
              cg.i32.const(n);
              cg.i32.mul();
              cg.local.get(j);
              cg.i32.add();
            },
            () => hl.const(dtype, 0),
          );
        },
      );

      cg.end(); // end vNormSq != 0
      cg.end(); // end norm != 0
    });

    // Extract R: first k rows of work (k × n)
    hl.forLoop(idx, 0, k * n, () => {
      hl.store(dtype, rPtr, hl.getExpr(idx), () => {
        hl.load(dtype, workPtr, hl.getExpr(idx));
      });
    });
  });
}

/**
 * Generate size-specialized batched QR function.
 */
function genQRBatchedSized(
  cg: CodeGenerator,
  hl: WasmHl,
  m: number,
  n: number,
  dtype: "f32" | "f64",
  singleFunc: number,
): number {
  const elemSize = dtype === "f32" ? 4 : 8;
  const k = Math.min(m, n);
  const aBytes = m * n * elemSize;
  const qBytes = m * k * elemSize;
  const rBytes = k * n * elemSize;
  const _workBytes = m * n * elemSize;

  // qr_batched(aPtr, qPtr, rPtr, workPtr, batchSize)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const aPtr = 0;
    const qPtr = 1;
    const rPtr = 2;
    const workPtr = 3;
    const batchSize = 4;

    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batchSize), () => {
      // Call single QR
      cg.local.get(aPtr);
      cg.local.get(b);
      cg.i32.const(aBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(qPtr);
      cg.local.get(b);
      cg.i32.const(qBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(rPtr);
      cg.local.get(b);
      cg.i32.const(rBytes);
      cg.i32.mul();
      cg.i32.add();

      // workPtr is shared scratch — each batch item uses the same area
      // (sequential batching, no parallelism)
      cg.local.get(workPtr);

      cg.call(singleFunc);
    });
  });
}

/**
 * Build a size-specialized QR WASM module.
 *
 * @param m - Number of rows
 * @param n - Number of columns
 * @param dtype - f32 or f64
 *
 * Exports:
 * - qr(aPtr, qPtr, rPtr, workPtr) - single matrix
 * - qr_batched(aPtr, qPtr, rPtr, workPtr, batchSize) - multiple matrices
 */
export function buildQRModuleSized(
  m: number,
  n: number,
  dtype: "f32" | "f64",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  configureMemoryImport(cg);

  const qrFunc = genQRSizedWithWork(cg, hl, m, n, dtype);
  const qrBatchedFunc = genQRBatchedSized(cg, hl, m, n, dtype, qrFunc);

  cg.export(qrFunc, "qr");
  cg.export(qrBatchedFunc, "qr_batched");

  return cg.finish();
}
