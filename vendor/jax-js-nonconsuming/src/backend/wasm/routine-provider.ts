/**
 * Runtime WASM routine provider using wasmblr with size specialization.
 *
 * This module provides size-specialized WASM routines that are compiled
 * on-demand and cached with LRU eviction. For ML workloads with fixed-size
 * models, this approach:
 * - Eliminates runtime size parameter passing
 * - Enables loop unrolling for small matrices
 * - Allows better optimization through known bounds
 *
 * The LRU cache ensures memory is bounded while keeping frequently-used
 * sizes (typical for repeated model inference) hot.
 */
import {
  buildArgsortModuleSized,
  buildCholeskyModuleSized,
  buildCholeskySimdModule,
  buildLUModuleSized,
  buildQRModuleSized,
  buildScatterAddModule,
  buildSortModuleSized,
  buildTriangularSolveModuleSized,
} from "./routines/index";
import { _emitCodeCapture, _isCodeCaptureEnabled } from "../../backend";

// ============================================================================
// LRU Cache for size-specialized modules
// ============================================================================

/** Default max cache entries (tunable based on memory constraints) */
const DEFAULT_MAX_CACHE_SIZE = 64;

/** Cache entry with access tracking */
interface CacheEntry {
  module: WebAssembly.Module;
  lastAccess: number;
}

/** LRU cache for compiled modules */
class ModuleLRUCache {
  #cache = new Map<string, CacheEntry>();
  #maxSize: number;
  #accessCounter = 0;

  constructor(maxSize = DEFAULT_MAX_CACHE_SIZE) {
    this.#maxSize = maxSize;
  }

  get(key: string): WebAssembly.Module | undefined {
    const entry = this.#cache.get(key);
    if (entry) {
      entry.lastAccess = ++this.#accessCounter;
      return entry.module;
    }
    return undefined;
  }

  set(key: string, module: WebAssembly.Module): void {
    // Evict LRU entry if at capacity
    if (this.#cache.size >= this.#maxSize && !this.#cache.has(key)) {
      this.#evictLRU();
    }
    this.#cache.set(key, {
      module,
      lastAccess: ++this.#accessCounter,
    });
  }

  #evictLRU(): void {
    let lruKey: string | null = null;
    let lruAccess = Infinity;
    for (const [key, entry] of this.#cache) {
      if (entry.lastAccess < lruAccess) {
        lruAccess = entry.lastAccess;
        lruKey = key;
      }
    }
    if (lruKey) {
      this.#cache.delete(lruKey);
    }
  }

  /** Clear the cache (useful for testing or memory pressure) */
  clear(): void {
    this.#cache.clear();
  }

  /** Get current cache size */
  get size(): number {
    return this.#cache.size;
  }
}

/** Global module cache */
const moduleCache = new ModuleLRUCache();

// ============================================================================
// Size-specialized routine parameters
// ============================================================================

/** Parameters for Cholesky size specialization */
export interface CholeskyParams {
  n: number;
  dtype: "f32" | "f64";
}

/** Parameters for TriangularSolve size specialization */
export interface TriangularSolveParams {
  n: number;
  batchRows: number;
  dtype: "f32" | "f64";
  unitDiagonal: boolean;
  lower: boolean;
}

/** Parameters for LU size specialization */
export interface LUParams {
  m: number;
  n: number;
  dtype: "f32" | "f64";
}

/** Parameters for Sort size specialization */
export interface SortParams {
  n: number;
  dtype: "f32" | "f64";
}

/** Parameters for Argsort size specialization */
export interface ArgsortParams {
  n: number;
  dtype: "f32" | "f64";
}

/** Parameters for QR size specialization */
export interface QRParams {
  m: number;
  n: number;
  dtype: "f32" | "f64";
}

/** Parameters for ScatterAdd size specialization */
export interface ScatterAddParams {
  outerSize: number;
  updatesLen: number;
  innerSize: number;
  axisSize: number;
  dtype: "f32" | "f64" | "i32";
}

// ============================================================================
// Cache key generation
// ============================================================================

function choleskyKey(params: CholeskyParams): string {
  return `cholesky:${params.dtype}:${params.n}`;
}

function triangularSolveKey(params: TriangularSolveParams): string {
  return `trisolve:${params.dtype}:${params.n}:${params.batchRows}:${params.unitDiagonal ? 1 : 0}:${params.lower ? 1 : 0}`;
}

function luKey(params: LUParams): string {
  return `lu:${params.dtype}:${params.m}:${params.n}`;
}

function sortKey(params: SortParams): string {
  return `sort:${params.dtype}:${params.n}`;
}

function argsortKey(params: ArgsortParams): string {
  return `argsort:${params.dtype}:${params.n}`;
}

function qrKey(params: QRParams): string {
  return `qr:${params.dtype}:${params.m}:${params.n}`;
}

function scatterAddKey(params: ScatterAddParams): string {
  return `scatter_add:${params.dtype}:${params.outerSize}:${params.updatesLen}:${params.innerSize}:${params.axisSize}`;
}

// ============================================================================
// Size-specialized module getters
// ============================================================================

/**
 * Get a size-specialized Cholesky module.
 * Uses SIMD (f32x4) for f32 matrices with n >= 32 for ~2-4x speedup.
 * Exports: cholesky(inPtr, outPtr), cholesky_batched(inPtr, outPtr, batch)
 */
export function getCholeskyModule(params: CholeskyParams): WebAssembly.Module {
  const key = choleskyKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    // Use SIMD for f32 with n >= 32 (crossover point where SIMD overhead is worth it)
    const useSIMD = params.dtype === "f32" && params.n >= 32;
    const bytes = useSIMD
      ? buildCholeskySimdModule(params.n)
      : buildCholeskyModuleSized(params.n, params.dtype);
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "cholesky",
        metadata: {
          n: params.n,
          dtype: params.dtype,
          simd: useSIMD,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized TriangularSolve module.
 * Exports: triangular_solve(aPtr, bPtr, xPtr), triangular_solve_batched(aPtr, bPtr, xPtr, numBatches)
 */
export function getTriangularSolveModule(
  params: TriangularSolveParams,
): WebAssembly.Module {
  const key = triangularSolveKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildTriangularSolveModuleSized(
      params.n,
      params.batchRows,
      params.dtype,
      params.unitDiagonal,
      params.lower,
    );
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "triangular_solve",
        metadata: {
          n: params.n,
          dtype: params.dtype,
          lower: params.lower,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized LU module.
 * Exports: lu(aPtr, luPtr, pivPtr, permPtr), lu_batched(aPtr, luPtr, pivPtr, permPtr, batch)
 */
export function getLUModule(params: LUParams): WebAssembly.Module {
  const key = luKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildLUModuleSized(params.m, params.n, params.dtype);
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "lu",
        metadata: {
          m: params.m,
          n: params.n,
          dtype: params.dtype,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized Sort module.
 * Exports: sort(dataPtr, auxPtr), sort_batched(dataPtr, auxPtr, batch)
 */
export function getSortModule(params: SortParams): WebAssembly.Module {
  const key = sortKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildSortModuleSized(params.n, params.dtype);
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "sort",
        metadata: {
          n: params.n,
          dtype: params.dtype,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized Argsort module.
 * Exports: argsort(dataPtr, outPtr, idxPtr, auxPtr), argsort_batched(...)
 */
export function getArgsortModule(params: ArgsortParams): WebAssembly.Module {
  const key = argsortKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildArgsortModuleSized(params.n, params.dtype);
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "argsort",
        metadata: {
          n: params.n,
          dtype: params.dtype,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized QR module.
 * Exports: qr(aPtr, qPtr, rPtr, workPtr), qr_batched(aPtr, qPtr, rPtr, workPtr, batch)
 */
export function getQRModule(params: QRParams): WebAssembly.Module {
  const key = qrKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildQRModuleSized(params.m, params.n, params.dtype);
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "qr",
        metadata: {
          m: params.m,
          n: params.n,
          dtype: params.dtype,
          byteLength: bytes.byteLength,
        },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

/**
 * Get a size-specialized ScatterAdd module.
 * Exports: scatter_add(outPtr, idxPtr, updPtr)
 */
export function getScatterAddModule(
  params: ScatterAddParams,
): WebAssembly.Module {
  const key = scatterAddKey(params);
  let module = moduleCache.get(key);
  if (!module) {
    const bytes = buildScatterAddModule(
      params.outerSize,
      params.updatesLen,
      params.innerSize,
      params.axisSize,
      params.dtype,
    );
    if (_isCodeCaptureEnabled()) {
      _emitCodeCapture({
        backend: "wasm",
        kind: "routine",
        label: "scatter_add",
        metadata: { dtype: params.dtype, byteLength: bytes.byteLength },
      });
    }
    module = new WebAssembly.Module(bytes);
    moduleCache.set(key, module);
  }
  return module;
}

// ============================================================================
// Cache management utilities
// ============================================================================

/** Clear the module cache (useful for testing or memory management) */
export function clearRoutineCache(): void {
  moduleCache.clear();
}

/** Get current cache size */
export function getRoutineCacheSize(): number {
  return moduleCache.size;
}
