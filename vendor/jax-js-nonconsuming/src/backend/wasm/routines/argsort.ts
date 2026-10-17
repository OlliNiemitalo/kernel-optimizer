/**
 * Argsort routine using wasmblr with size specialization.
 *
 * Returns indices that would sort the array.
 * Uses merge sort on indices for stability.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate merge function for argsort (compares by data values).
 */
function genMergeIdx(
  cg: CodeGenerator,
  hl: WasmHl,
  dtype: "f32" | "f64",
): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // merge_idx(dataPtr, idxPtr, auxPtr, left, mid, right)
  return cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const dataPtr = 0;
      const idxPtr = 1;
      const auxPtr = 2;
      const left = 3;
      const mid = 4;
      const right = 5;

      const idx = cg.local.declare(cg.i32);
      const i = cg.local.declare(cg.i32);
      const j = cg.local.declare(cg.i32);
      const k = cg.local.declare(cg.i32);
      const idxI = cg.local.declare(cg.i32);
      const idxJ = cg.local.declare(cg.i32);
      const ai = cg.local.declare(ty);
      const aj = cg.local.declare(ty);

      // Copy indices to aux
      cg.local.get(left);
      cg.local.set(idx);
      hl.whileLoop(
        () => {
          cg.local.get(idx);
          cg.local.get(right);
          cg.i32.le_s();
        },
        () => {
          hl.store("i32", auxPtr, hl.getExpr(idx), () => {
            hl.load("i32", idxPtr, hl.getExpr(idx));
          });
          cg.local.get(idx);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(idx);
        },
      );

      // Merge
      cg.local.get(left);
      cg.local.set(i);
      cg.local.get(mid);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(j);
      cg.local.get(left);
      cg.local.set(k);

      hl.whileLoop(
        () => {
          cg.local.get(i);
          cg.local.get(mid);
          cg.i32.le_s();
          cg.local.get(j);
          cg.local.get(right);
          cg.i32.le_s();
          cg.i32.and();
        },
        () => {
          // Get indices
          hl.load("i32", auxPtr, hl.getExpr(i));
          cg.local.set(idxI);
          hl.load("i32", auxPtr, hl.getExpr(j));
          cg.local.set(idxJ);

          // Get values by index
          hl.load(dtype, dataPtr, hl.getExpr(idxI));
          cg.local.set(ai);
          hl.load(dtype, dataPtr, hl.getExpr(idxJ));
          cg.local.set(aj);

          // Stable comparison: (ai < aj) || (ai == aj && i <= j) || isNaN(aj)
          cg.local.get(ai);
          cg.local.get(aj);
          if (dtype === "f32") cg.f32.lt();
          else cg.f64.lt();

          cg.local.get(ai);
          cg.local.get(aj);
          if (dtype === "f32") cg.f32.eq();
          else cg.f64.eq();
          cg.local.get(i);
          cg.local.get(j);
          cg.i32.le_s();
          cg.i32.and();
          cg.i32.or();

          cg.local.get(aj);
          cg.local.get(aj);
          if (dtype === "f32") cg.f32.ne();
          else cg.f64.ne(); // isNaN(aj)
          cg.i32.or();

          hl.ifElse(
            cg.void,
            () => {
              hl.store("i32", idxPtr, hl.getExpr(k), () => cg.local.get(idxI));
              cg.local.get(i);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(i);
            },
            () => {
              hl.store("i32", idxPtr, hl.getExpr(k), () => cg.local.get(idxJ));
              cg.local.get(j);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(j);
            },
          );
          cg.local.get(k);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(k);
        },
      );

      // Copy remaining
      hl.whileLoop(
        () => {
          cg.local.get(i);
          cg.local.get(mid);
          cg.i32.le_s();
        },
        () => {
          hl.store("i32", idxPtr, hl.getExpr(k), () => {
            hl.load("i32", auxPtr, hl.getExpr(i));
          });
          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(i);
          cg.local.get(k);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(k);
        },
      );

      hl.whileLoop(
        () => {
          cg.local.get(j);
          cg.local.get(right);
          cg.i32.le_s();
        },
        () => {
          hl.store("i32", idxPtr, hl.getExpr(k), () => {
            hl.load("i32", auxPtr, hl.getExpr(j));
          });
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(j);
          cg.local.get(k);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(k);
        },
      );
    },
  );
}

/**
 * Generate size-specialized argsort function.
 */
function genArgsortSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  mergeFunc: number,
): number {
  // argsort(dataPtr, outPtr, idxPtr, auxPtr)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const dataPtr = 0;
    const outPtr = 1;
    const idxPtr = 2;
    const auxPtr = 3;

    const i = cg.local.declare(cg.i32);
    const width = cg.local.declare(cg.i32);
    const left = cg.local.declare(cg.i32);
    const mid = cg.local.declare(cg.i32);
    const right = cg.local.declare(cg.i32);
    const idx = cg.local.declare(cg.i32);

    // Initialize indices: idxPtr[i] = i
    hl.forLoop(i, 0, n, () => {
      hl.store("i32", idxPtr, hl.getExpr(i), () => cg.local.get(i));
    });

    // Bottom-up merge sort on indices
    cg.i32.const(1);
    cg.local.set(width);

    hl.whileLoop(
      () => {
        cg.local.get(width);
        cg.i32.const(n);
        cg.i32.lt_s();
      },
      () => {
        cg.i32.const(0);
        cg.local.set(left);

        hl.whileLoop(
          () => {
            cg.local.get(left);
            cg.i32.const(n);
            cg.local.get(width);
            cg.i32.sub();
            cg.i32.lt_s();
          },
          () => {
            cg.local.get(left);
            cg.local.get(width);
            cg.i32.add();
            cg.i32.const(1);
            cg.i32.sub();
            cg.local.set(mid);

            cg.local.get(left);
            cg.local.get(width);
            cg.i32.const(2);
            cg.i32.mul();
            cg.i32.add();
            cg.i32.const(1);
            cg.i32.sub();
            cg.local.set(right);

            cg.local.get(right);
            cg.i32.const(n - 1);
            cg.i32.ge_s();
            hl.ifElse(cg.void, () => {
              cg.i32.const(n - 1);
              cg.local.set(right);
            });

            cg.local.get(dataPtr);
            cg.local.get(idxPtr);
            cg.local.get(auxPtr);
            cg.local.get(left);
            cg.local.get(mid);
            cg.local.get(right);
            cg.call(mergeFunc);

            cg.local.get(left);
            cg.local.get(width);
            cg.i32.const(2);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(left);
          },
        );

        cg.local.get(width);
        cg.i32.const(2);
        cg.i32.mul();
        cg.local.set(width);
      },
    );

    // Write sorted values: outPtr[i] = dataPtr[idxPtr[i]]
    hl.forLoop(i, 0, n, () => {
      hl.load("i32", idxPtr, hl.getExpr(i));
      cg.local.set(idx);
      hl.store(dtype, outPtr, hl.getExpr(i), () => {
        hl.load(dtype, dataPtr, hl.getExpr(idx));
      });
    });
  });
}

/**
 * Generate size-specialized batched argsort.
 */
function genArgsortBatchedSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  argsortFunc: number,
): number {
  const elemSize = dtype === "f32" ? 4 : 8;
  const dataRowBytes = n * elemSize;
  const idxRowBytes = n * 4;

  // argsort_batched(dataPtr, outPtr, idxPtr, auxPtr, batchSize)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const dataPtr = 0;
    const outPtr = 1;
    const idxPtr = 2;
    const auxPtr = 3;
    const batchSize = 4;

    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batchSize), () => {
      // dataPtr + b * dataRowBytes
      cg.local.get(dataPtr);
      cg.local.get(b);
      cg.i32.const(dataRowBytes);
      cg.i32.mul();
      cg.i32.add();

      // outPtr + b * dataRowBytes
      cg.local.get(outPtr);
      cg.local.get(b);
      cg.i32.const(dataRowBytes);
      cg.i32.mul();
      cg.i32.add();

      // idxPtr + b * idxRowBytes
      cg.local.get(idxPtr);
      cg.local.get(b);
      cg.i32.const(idxRowBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(auxPtr);
      cg.call(argsortFunc);
    });
  });
}

/**
 * Build a size-specialized argsort WASM module.
 *
 * @param n - Array size
 * @param dtype - f32 or f64
 *
 * Exports:
 * - argsort(dataPtr, outPtr, idxPtr, auxPtr) - single array
 * - argsort_batched(dataPtr, outPtr, idxPtr, auxPtr, batchSize) - multiple arrays
 */
export function buildArgsortModuleSized(
  n: number,
  dtype: "f32" | "f64",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  configureMemoryImport(cg);

  const mergeFunc = genMergeIdx(cg, hl, dtype);
  const argsortFunc = genArgsortSized(cg, hl, n, dtype, mergeFunc);
  const argsortBatchedFunc = genArgsortBatchedSized(
    cg,
    hl,
    n,
    dtype,
    argsortFunc,
  );

  cg.export(argsortFunc, "argsort");
  cg.export(argsortBatchedFunc, "argsort_batched");

  return cg.finish();
}
