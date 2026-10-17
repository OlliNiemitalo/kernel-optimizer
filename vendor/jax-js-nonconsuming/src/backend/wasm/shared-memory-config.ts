/**
 * Shared memory configuration for WASM codegen modules.
 *
 * This module is imported by both `wasm.ts` (which sets the flag)
 * and `routines/*.ts` (which read it via `configureMemoryImport`).
 * Kept in a separate file to avoid circular dependencies.
 */

import type { CodeGenerator } from "./wasmblr";

/** Maximum pages for shared WebAssembly.Memory (4 GiB = 65536 × 64 KiB, WASM32 ceiling). */
export const MAX_SHARED_PAGES = 65536;

/**
 * Module-level flag: whether the WASM backend uses shared memory.
 * Set by WasmBackend constructor via `setUseSharedMemory()`.
 * Used by codegen paths to declare memory imports as shared
 * (required by the WebAssembly threads proposal when the provided
 * memory is a SharedArrayBuffer).
 */
let _useSharedMemory = false;

export function setUseSharedMemory(value: boolean): void {
  _useSharedMemory = value;
}

export function getUseSharedMemory(): boolean {
  return _useSharedMemory;
}

/**
 * Configure the standard memory import for a codegen unit.
 * If shared memory is active, declares the import as shared with max pages
 * so that the module can be instantiated with a SharedArrayBuffer-backed memory.
 */
export function configureMemoryImport(cg: CodeGenerator): void {
  const mem = cg.memory.import("env", "memory");
  if (_useSharedMemory) {
    mem.shared(true).pages(0, MAX_SHARED_PAGES);
  }
}
