/**
 * Scatter-add routine using wasmblr.
 *
 * Generates a WASM module that performs:
 *   output[indices[j]] += updates[j]  (along a given axis)
 *
 * All loop bounds are baked as compile-time constants for optimal performance.
 */

import { configureMemoryImport } from "../shared-memory-config";
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

/**
 * Build a size-specialized scatter-add WASM module.
 *
 * Exported function: scatter_add(outPtr, idxPtr, updPtr)
 *
 * @param outerSize   - product of dims before the scatter axis
 * @param updatesLen  - number of update indices
 * @param innerSize   - product of dims after the scatter axis
 * @param axisSize    - size of the target axis
 * @param dtype       - element type ("f32" | "f64" | "i32")
 * @returns Compiled WebAssembly.Module bytes (Uint8Array)
 */
export function buildScatterAddModule(
  outerSize: number,
  updatesLen: number,
  innerSize: number,
  axisSize: number,
  dtype: "f32" | "f64" | "i32",
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  configureMemoryImport(cg);
  const hl = new WasmHl(cg);

  const targetInnerStride = axisSize * innerSize;

  // scatter_add(outPtr: i32, idxPtr: i32, updPtr: i32)
  const fnIdx = cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const outPtr = 0;
    const idxPtr = 1;
    const updPtr = 2;

    const o = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const targetAxisIdx = cg.local.declare(cg.i32);
    const outIdx = cg.local.declare(cg.i32);
    const updIdx = cg.local.declare(cg.i32);

    hl.forLoop(o, 0, outerSize, () => {
      hl.forLoop(j, 0, updatesLen, () => {
        // targetAxisIdx = indices[j] (i32 load)
        hl.load("i32", idxPtr, () => {
          cg.local.get(j);
        });
        cg.local.set(targetAxisIdx);

        // Bounds check: skip if targetAxisIdx < 0 || >= axisSize
        cg.local.get(targetAxisIdx);
        cg.i32.const(0);
        cg.i32.lt_s();
        cg.local.get(targetAxisIdx);
        cg.i32.const(axisSize);
        cg.i32.ge_s();
        cg.i32.or();
        cg.if(cg.void);
        // skip this j — continue
        cg.else();

        hl.forLoop(k, 0, innerSize, () => {
          // outIdx = o * targetInnerStride + targetAxisIdx * innerSize + k
          cg.local.get(o);
          cg.i32.const(targetInnerStride);
          cg.i32.mul();
          cg.local.get(targetAxisIdx);
          cg.i32.const(innerSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(k);
          cg.i32.add();
          cg.local.set(outIdx);

          // updIdx = o * updatesLen * innerSize + j * innerSize + k
          cg.local.get(o);
          cg.i32.const(updatesLen * innerSize);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.const(innerSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(k);
          cg.i32.add();
          cg.local.set(updIdx);

          // output[outIdx] += updates[updIdx]
          hl.store(dtype, outPtr, hl.getExpr(outIdx), () => {
            hl.load(dtype, outPtr, hl.getExpr(outIdx));
            hl.load(dtype, updPtr, hl.getExpr(updIdx));
            if (dtype === "f64") {
              cg.f64.add();
            } else if (dtype === "f32") {
              cg.f32.add();
            } else {
              cg.i32.add();
            }
          });
        });

        cg.end(); // end else
      });
    });
  });

  cg.export(fnIdx, "scatter_add");
  return cg.finish();
}
