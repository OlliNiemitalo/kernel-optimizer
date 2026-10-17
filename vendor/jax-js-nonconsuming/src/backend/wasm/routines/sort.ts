/**
 * Sort routine using wasmblr.
 *
 * In-place merge sort for f32/f64 arrays.
 * Uses bottom-up merge sort which is O(n log n) and stable.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Generate merge function for sort.
 */
function genMerge(cg: CodeGenerator, hl: WasmHl, dtype: "f32" | "f64"): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // merge(dataPtr, auxPtr, left, mid, right)
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const dataPtr = 0;
    const auxPtr = 1;
    const left = 2;
    const mid = 3;
    const right = 4;

    const idx = cg.local.declare(cg.i32);
    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const ai = cg.local.declare(ty);
    const aj = cg.local.declare(ty);

    // Copy data[left..right] to aux
    cg.local.get(left);
    cg.local.set(idx);
    hl.whileLoop(
      () => {
        cg.local.get(idx);
        cg.local.get(right);
        cg.i32.le_s();
      },
      () => {
        hl.store(dtype, auxPtr, hl.getExpr(idx), () => {
          hl.load(dtype, dataPtr, hl.getExpr(idx));
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

    // while i <= mid && j <= right
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
        hl.load(dtype, auxPtr, hl.getExpr(i));
        cg.local.set(ai);
        hl.load(dtype, auxPtr, hl.getExpr(j));
        cg.local.set(aj);

        // NaN-aware: ai <= aj || isNaN(aj)
        // In WASM: (ai <= aj) || (aj != aj)
        cg.local.get(ai);
        cg.local.get(aj);
        if (dtype === "f32") cg.f32.le();
        else cg.f64.le();
        cg.local.get(aj);
        cg.local.get(aj);
        if (dtype === "f32") cg.f32.ne();
        else cg.f64.ne(); // isNaN(aj)
        cg.i32.or();
        hl.ifElse(
          cg.void,
          () => {
            hl.store(dtype, dataPtr, hl.getExpr(k), () => cg.local.get(ai));
            cg.local.get(i);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(i);
          },
          () => {
            hl.store(dtype, dataPtr, hl.getExpr(k), () => cg.local.get(aj));
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

    // Copy remaining from left half
    hl.whileLoop(
      () => {
        cg.local.get(i);
        cg.local.get(mid);
        cg.i32.le_s();
      },
      () => {
        hl.store(dtype, dataPtr, hl.getExpr(k), () => {
          hl.load(dtype, auxPtr, hl.getExpr(i));
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

    // Copy remaining from right half
    hl.whileLoop(
      () => {
        cg.local.get(j);
        cg.local.get(right);
        cg.i32.le_s();
      },
      () => {
        hl.store(dtype, dataPtr, hl.getExpr(k), () => {
          hl.load(dtype, auxPtr, hl.getExpr(j));
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
  });
}

/**
 * Generate size-specialized sort function (bottom-up merge sort).
 */
function genSortSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  mergeFunc: number,
): number {
  // sort(dataPtr, auxPtr)
  return cg.function([cg.i32, cg.i32], [], () => {
    const dataPtr = 0;
    const auxPtr = 1;

    const width = cg.local.declare(cg.i32);
    const left = cg.local.declare(cg.i32);
    const mid = cg.local.declare(cg.i32);
    const right = cg.local.declare(cg.i32);

    cg.i32.const(1);
    cg.local.set(width);

    // while width < n
    hl.whileLoop(
      () => {
        cg.local.get(width);
        cg.i32.const(n);
        cg.i32.lt_s();
      },
      () => {
        cg.i32.const(0);
        cg.local.set(left);

        // while left < n - width
        hl.whileLoop(
          () => {
            cg.local.get(left);
            cg.i32.const(n);
            cg.local.get(width);
            cg.i32.sub();
            cg.i32.lt_s();
          },
          () => {
            // mid = left + width - 1
            cg.local.get(left);
            cg.local.get(width);
            cg.i32.add();
            cg.i32.const(1);
            cg.i32.sub();
            cg.local.set(mid);

            // right = min(left + 2*width - 1, n - 1)
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

            // Call merge
            cg.local.get(dataPtr);
            cg.local.get(auxPtr);
            cg.local.get(left);
            cg.local.get(mid);
            cg.local.get(right);
            cg.call(mergeFunc);

            // left += 2 * width
            cg.local.get(left);
            cg.local.get(width);
            cg.i32.const(2);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(left);
          },
        );

        // width *= 2
        cg.local.get(width);
        cg.i32.const(2);
        cg.i32.mul();
        cg.local.set(width);
      },
    );
  });
}

/**
 * Generate size-specialized batched sort.
 */
function genSortBatchedSized(
  cg: CodeGenerator,
  hl: WasmHl,
  n: number,
  dtype: "f32" | "f64",
  sortFunc: number,
): number {
  const elemSize = dtype === "f32" ? 4 : 8;
  const rowBytes = n * elemSize;

  // sort_batched(dataPtr, auxPtr, batchSize)
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const dataPtr = 0;
    const auxPtr = 1;
    const batchSize = 2;

    const b = cg.local.declare(cg.i32);

    hl.forLoop(b, 0, hl.getExpr(batchSize), () => {
      cg.local.get(dataPtr);
      cg.local.get(b);
      cg.i32.const(rowBytes);
      cg.i32.mul();
      cg.i32.add();

      cg.local.get(auxPtr);
      cg.call(sortFunc);
    });
  });
}

/**
 * Build a size-specialized sort WASM module.
 *
 * @param n - Array size
 * @param dtype - f32 or f64
 *
 * Exports:
 * - sort(dataPtr, auxPtr) - single array
 * - sort_batched(dataPtr, auxPtr, batchSize) - multiple arrays
 */
export function buildSortModuleSized(
  n: number,
  dtype: "f32" | "f64",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  configureMemoryImport(cg);

  const mergeFunc = genMerge(cg, hl, dtype);
  const sortFunc = genSortSized(cg, hl, n, dtype, mergeFunc);
  const sortBatchedFunc = genSortBatchedSized(cg, hl, n, dtype, sortFunc);

  cg.export(sortFunc, "sort");
  cg.export(sortBatchedFunc, "sort_batched");

  return cg.finish();
}
