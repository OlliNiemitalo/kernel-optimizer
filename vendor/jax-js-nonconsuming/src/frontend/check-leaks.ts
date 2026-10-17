/**
 * Array memory leak detection for jax-js applications.
 *
 * jax-js uses a non-consuming ownership model: operations leave their inputs
 * alive, and callers must explicitly `.dispose()` arrays when done. If you
 * create an array and never call `.dispose()`, the underlying backend memory
 * leaks. `checkLeaks` helps you find these leaks in tests or during development.
 *
 * ## Quick start
 *
 * ```ts
 * import * as jax from "@hamk-uas/jax-js-nonconsuming";
 *
 * jax.checkLeaks.start();
 *
 * // ... your code ...
 *
 * const report = jax.checkLeaks.stop();
 * if (report.leaked > 0) {
 *   console.error(report.summary);
 * }
 * ```
 *
 * ## Example output
 *
 * ```
 * 5 slot(s) leaked (2 tracked array(s)):
 *
 *   test/numpy.test.ts:389:27 — 1× Array:float32[] rc=1  (via onesLike → fullLike)
 *   src/frontend/jvp.ts:107:39 — 1× Array:float32[] rc=1  (@hamk-uas/jax-js-nonconsuming)
 *
 * 1 from user code, 1 in @hamk-uas/jax-js-nonconsuming.
 *
 * 3 slot(s) not in tracked list (created before start() or by internal buffers).
 * rc=1 → never disposed. Use `using` for local temporaries, and call .dispose() at end-of-life
 * (including early .dispose() when a value is no longer needed to reduce peak memory).
 * ```
 *
 * ## How it works
 *
 * `start()` records the current backend slot count as a baseline.
 * While active, every `Array` creation and `.dispose()` call is tracked in
 * a lightweight map. `stop()` flushes JIT caches, diffs the slot count, and
 * — if any slots leaked — resolves stack traces to pinpoint exactly where
 * the leaking arrays were created.
 *
 * **Hot path** (every Array create/dispose): a boolean guard + `Map.set`/`delete`.
 * `new Error()` captures V8 frame pointers cheaply; the `.stack` string is
 * only formatted on first access inside `stop()` (cold path).
 *
 * **No concurrent tests** — `checkLeaks` uses global state (one active
 * session at a time). Do not use `test.concurrent()`. Sequential test
 * execution is the default in vitest.
 *
 * @module
 */

import { type Device, devices, getBackend } from "../backend";

// ── JIT cache disposal registry ──
// jit.ts registers its cache clearer at module load time.
// This avoids circular deps: check-leaks ← jit (safe direction).
const _jitCacheDisposers: (() => void)[] = [];

/** Register a callback that clears a JIT-related cache. Called by jit.ts. */
export function _registerJitCacheDisposer(fn: () => void): void {
  _jitCacheDisposers.push(fn);
}

// ── Cache size getters ──
// Each cache module registers a named size getter so `getCacheSizes()` can
// report current cache occupancy without creating import cycles.
const _cacheSizeGetters = new Map<string, () => number>();

/** Register a named cache size getter. Called by jit.ts, jvp.ts, etc. */
export function _registerCacheSizeGetter(
  name: string,
  getter: () => number,
): void {
  _cacheSizeGetters.set(name, getter);
}

/** Per-cache entry counts for diagnostics and testing. */
export interface CacheSizes {
  /** `jitCompileCache` — compiled JitPrograms keyed by jaxpr signature. */
  jitCompile: number;
  /** `jvpJaxprCache` — forward-mode differentiated body jaxprs. */
  jvpJaxpr: number;
  /** `transposeJaxprCache` — transposed body jaxprs (total entries across all inner maps). */
  transposeJaxpr: number;
  /** `vmapJaxprCache` — vectorized body jaxprs (total entries across all inner maps). */
  vmapJaxpr: number;
  /** Number of registered jit function disposers (≈ unique jit wrappers created). */
  jitFunctions: number;
}

/**
 * Return current cache sizes for all JIT-related caches.
 *
 * Useful for testing that caches grow and shrink as expected, and for
 * verifying deduplication behaviour (e.g., calling `jit(fn)` twice with
 * the same function reference should not increase `jitFunctions`).
 *
 * @example
 * ```ts
 * import { getCacheSizes, clearCaches } from "@hamk-uas/jax-js-nonconsuming";
 *
 * const before = getCacheSizes();
 * // ... run some jit-compiled code ...
 * const after = getCacheSizes();
 * console.log("new JIT entries:", after.jitCompile - before.jitCompile);
 * clearCaches();
 * console.log("after clear:", getCacheSizes().jitCompile); // 0
 * ```
 */
export function getCacheSizes(): CacheSizes {
  return {
    jitCompile: _cacheSizeGetters.get("jitCompile")?.() ?? 0,
    jvpJaxpr: _cacheSizeGetters.get("jvpJaxpr")?.() ?? 0,
    transposeJaxpr: _cacheSizeGetters.get("transposeJaxpr")?.() ?? 0,
    vmapJaxpr: _cacheSizeGetters.get("vmapJaxpr")?.() ?? 0,
    jitFunctions: _jitFunctionDisposers.size,
  };
}

/**
 * Per-function jit() cache disposers. Each jit() call registers a callback
 * that disposes its ClosedJaxpr consts and clears its cache. This allows
 * `_disposeAllJitCaches()` to clean up module-level jit functions whose consts
 * were created during a test session.
 */
export const _jitFunctionDisposers = new Set<() => void>();

function _disposeAllJitCaches(): void {
  for (const fn of _jitCacheDisposers) fn();
  for (const fn of _jitFunctionDisposers) fn();
  // Don't clear _jitFunctionDisposers — the registrations persist
  // (they're registered once when the jit function is created).
}

/**
 * Flush all internal JIT compilation caches, freeing GPU/WASM memory held
 * by cached `ClosedJaxpr` constants.
 *
 * Call this when you create many short-lived `jit()` closures (e.g., in an
 * optimization loop where each iteration builds a new loss function). Without
 * clearing, each closure's cached constants accumulate until GC runs — which
 * may never happen for GPU buffers.
 *
 * After calling `clearCaches()`, existing `jit()` functions remain usable —
 * they will re-trace and re-compile on their next call.
 *
 * @example
 * ```ts
 * import { clearCaches, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
 *
 * for (let i = 0; i < 100; i++) {
 *   const f = jit((x) => x.mul(np.array([i])).sum());
 *   const result = f(np.array([1, 2, 3]));
 *   result.dispose();
 *   f.dispose(); // free this closure's cache
 * }
 * // Or clear everything at once:
 * clearCaches();
 * ```
 */
export function clearCaches(): void {
  _disposeAllJitCaches();
}

// ── Tracking state (checked on the hot path) ──

/** True only between start() and stop(). Checked in Array ctor/dispose. */
export let _leakTrackingEnabled = false;

/**
 * Live tracked Arrays → Error captured at construction time.
 * `new Error()` records V8 frame pointers cheaply; the `.stack` string
 * is only formatted on first access (cold path, inside stop()).
 */
export const _leakTrackingMap = new Map<object, Error>();

/**
 * When `trackRefs` is enabled, records the last `.ref` call site per array.
 * Only populated when `start({ trackRefs: true })` — checked in array.ts.
 */
export let _trackRefsEnabled = false;
export const _lastRefMap = new Map<object, Error>();

// ── Snapshot state ──

const _startSlots = new Map<Device, number>();
let _active = false;

// ── Cold-path helpers (only called when leaks are detected) ──

/**
 * Library source URL prefix, derived from this module's own URL.
 * Any stack frame whose URL starts with this is library-internal.
 * Computed once at load time (not on the hot path).
 */
const _libSrcPrefix = (() => {
  try {
    // This file is <root>/src/frontend/check-leaks.{ts,js}
    const url = import.meta.url;
    const i = url.lastIndexOf("/src/frontend/check-leaks.");
    if (i >= 0) return url.slice(0, i + 1) + "src/";
  } catch {
    // import.meta.url unavailable (e.g. CJS context)
  }
  return null;
})();

/**
 * Detect which package a raw frame URL belongs to.
 * Returns the package name (e.g. "@hamk-uas/jax-js-nonconsuming", "some-lib") or null for user code.
 */
function detectPackage(rawUrl: string): string | null {
  // node_modules/<scope>/<pkg> or node_modules/<pkg>
  const nm = rawUrl.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/);
  if (nm) return nm[1];
  // jax-js library src
  if (_libSrcPrefix !== null && rawUrl.startsWith(_libSrcPrefix))
    return "@hamk-uas/jax-js-nonconsuming";
  return null;
}

/** Convert a Vite/V8 stack location to a workspace-relative path. */
function toRelativePath(raw: string): string {
  let loc = raw.replace(/\?[^:]*(?=:\d+:\d+)/, ""); // strip Vite query params
  loc = loc.replace(/^https?:\/\/[^/]+\//, ""); // strip URL origin
  if (loc.startsWith("@fs/")) {
    loc = loc.slice(4);
    if (!loc.startsWith("/")) loc = "/" + loc;
  }
  loc = loc.replace(/^file:\/\//, "");
  for (const m of ["/src/", "/test/", "/packages/", "/bench/"]) {
    const i = loc.indexOf(m);
    if (i >= 0) return loc.slice(i + 1);
  }
  return loc;
}

/** Boilerplate frames to skip (Array constructor internals). */
const SKIP_FUNCS = ["new Array", "new PendingExecute", "Array2", "new Array2"];

/** Dep functions that appear in every creation stack — filtered from via chain. */
const VIA_SKIP = new Set([
  "bind",
  "bind1",
  "pureArray",
  "fullLower",
  "fullRaise",
]);

interface CreationSite {
  location: string;
  via: string | null;
  /** Package name if the creation site is inside a dependency, or null for user code. */
  pkg: string | null;
}

/**
 * Walk the Error's stack to find the creation site.
 * Accessing err.stack here triggers V8's lazy string formatting.
 */
function findCreationSite(err: Error): CreationSite {
  const stack = err.stack;
  if (!stack) return { location: "(unknown)", via: null, pkg: "(unknown)" };

  let lastDepLoc: string | null = null;
  let lastDepPkg: string | null = null;
  let firstUserSite: CreationSite | null = null;
  /** Dep function names (innermost-first) for building via chain. */
  const depChain: string[] = [];
  let prevClean: string | null = null;

  for (const line of stack.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("at ")) continue;

    // Parse "at FuncName (location)" or "at location"
    let rawLoc: string;
    let funcName: string | undefined;
    const m = trimmed.match(/^at\s+(.+?)\s+\((.+)\)$/);
    if (m) {
      funcName = m[1];
      rawLoc = m[2];
    } else {
      rawLoc = trimmed.slice(3);
    }

    const location = toRelativePath(rawLoc);
    const pathOnly = location.replace(/:\d+:\d+$/, "");

    // Skip non-file frames
    if (
      pathOnly === "<anonymous>" ||
      pathOnly === "native" ||
      pathOnly.startsWith("eval ")
    )
      continue;
    // Skip constructor boilerplate
    if (funcName && SKIP_FUNCS.some((p) => funcName!.startsWith(p))) continue;

    // Dependency frame if URL matches a package, but test files are always user code
    const pkg = detectPackage(rawLoc);
    if (pkg && !/\.test\.[jt]sx?$/.test(pathOnly)) {
      lastDepLoc = location;
      lastDepPkg = pkg;
      // Collect unique, non-generic function names for the via chain
      const clean = funcName?.replace(/^(Module|Object)\./, "") ?? null;
      if (clean && clean !== prevClean && !VIA_SKIP.has(clean)) {
        depChain.push(clean);
        prevClean = clean;
      }
      continue;
    }

    // Found a user-code frame — build via chain (outermost → innermost call order)
    const chain = depChain.reverse().slice(0, 2);
    const via = chain.length > 0 ? chain.join(" → ") : null;
    const site = { location, via, pkg: null };

    // In browser REPLs, bundled runtime frames may appear before the virtual
    // source-mapped user frame. Prefer explicit virtual module frames when found.
    if (/(?:^|\/)(?:index|main)\.ts:\d+:\d+$/.test(location)) {
      return site;
    }

    if (firstUserSite === null) {
      firstUserSite = site;
    }
  }

  if (firstUserSite !== null) {
    return firstUserSite;
  }

  return {
    location: lastDepLoc ?? "(unknown)",
    via: null,
    pkg: lastDepPkg ?? "(unknown)",
  };
}

function safeDesc(arr: object): string {
  try {
    return (arr as { toString(): string }).toString();
  } catch {
    return "(unknown)";
  }
}

function safeRefCountNum(arr: object): number {
  try {
    const rc = (arr as { refCount: number }).refCount;
    return typeof rc === "number" ? rc : 0;
  } catch {
    return 0;
  }
}

function safeRefCount(arr: object): string {
  const rc = safeRefCountNum(arr);
  return rc > 0 ? ` rc=${rc}` : "";
}

// ── Public API ──

/** Result returned by {@link checkLeaks.stop}. */
export interface LeakReport {
  /** Number of unreleased backend slots (0 = no leaks). */
  leaked: number;
  /** Tracked leaks originating from user code (not inside a library package). */
  userLeaked: number;
  /** Tracked leaks originating from inside @hamk-uas/jax-js-nonconsuming. */
  internalLeaked: number;
  /** Array descriptions with creation locations (best-effort from tracking map). */
  details: string[];
  /** Human-readable summary with source-mapped file:line:col locations. */
  summary: string;
}

/** A single entry from {@link checkLeaks.snapshot}. */
export interface SnapshotEntry {
  /** Array description (dtype, shape). */
  desc: string;
  /** Current reference count. */
  rc: number;
  /** Creation location (workspace-relative path). */
  location: string;
  /** Package name, or null for user code. */
  pkg: string | null;
  /** Last .ref call site (only when trackRefs enabled). */
  lastRef: string | null;
}

export const checkLeaks = {
  /**
   * Begin tracking array allocations.
   *
   * Records the current backend slot count as a baseline and enables
   * per-allocation tracking. Call this before the code you want to check,
   * then call {@link stop} afterwards to get a leak report.
   *
   * @param opts.trackRefs - When `true`, also records each `.ref` call site
   *   so the leak report can show `↳ last .ref at <location>` for rc≥2 leaks.
   *   Adds one `Map.set` per `.ref` call. Off by default.
   */
  start(opts?: { trackRefs?: boolean }): void {
    // If already active, reset cleanly — supports befoteEach re-entrance.
    _leakTrackingMap.clear();
    _lastRefMap.clear();
    _trackRefsEnabled = opts?.trackRefs ?? false;

    // Record baseline slot counts for ALL initialized backends.
    _startSlots.clear();
    for (const d of devices) {
      try {
        _startSlots.set(d, getBackend(d).slotCount());
      } catch {
        /* not initialized */
      }
    }
    _leakTrackingEnabled = true;
    _active = true;
  },

  /**
   * Return a snapshot of all currently-tracked arrays without stopping.
   *
   * Useful for mid-session debugging — you can inspect which arrays are
   * still alive at any point between `start()` and `stop()`.
   */
  snapshot(): SnapshotEntry[] {
    const entries: SnapshotEntry[] = [];
    for (const [arr, err] of _leakTrackingMap) {
      const site = findCreationSite(err);
      const refErr = _lastRefMap.get(arr);
      let lastRef: string | null = null;
      if (refErr) {
        const refSite = findCreationSite(refErr);
        lastRef = refSite.location;
      }
      entries.push({
        desc: safeDesc(arr),
        rc: safeRefCountNum(arr),
        location: site.location,
        pkg: site.pkg,
        lastRef,
      });
    }
    return entries;
  },

  /**
   * Stop tracking and return a report of leaked backend slots.
   *
   * Flushes all JIT caches (freeing their cached constants),
   * computes the backend slot delta since `start()`, and — when
   * leaks are found — builds a human-readable summary with creation
   * sites, reference counts, and diagnostic tips.
   */
  stop(): LeakReport {
    if (!_active) {
      return {
        leaked: 0,
        userLeaked: 0,
        internalLeaked: 0,
        details: [],
        summary: "No leaks detected.",
      };
    }

    _leakTrackingEnabled = false;
    _trackRefsEnabled = false;
    _active = false;

    // Flush all jit caches so their const arrays are freed before counting.
    _disposeAllJitCaches();

    // Sum slot deltas across ALL currently-initialized backends.
    // Backends initialized between start() and stop() (e.g. WebGPU init
    // inside a test body) use baseline 0 — any surviving slots are leaks.
    let leaked = 0;
    for (const d of devices) {
      try {
        const baseline = _startSlots.get(d) ?? 0;
        leaked += getBackend(d).slotCount() - baseline;
      } catch {
        /* backend not initialized or gone */
      }
    }
    _startSlots.clear();

    if (leaked === 0) {
      _leakTrackingMap.clear();
      _lastRefMap.clear();
      return {
        leaked: 0,
        userLeaked: 0,
        internalLeaked: 0,
        details: [],
        summary: "No leaks detected.",
      };
    }

    // ── Cold path: build leak report ──

    const details: string[] = [];
    const groups = new Map<
      string,
      {
        count: number;
        descs: string[];
        via: string | null;
        pkg: string | null;
        maxRc: number;
        lastRefLocs: string[];
      }
    >();
    /** Leak counts by package (null key = user code). */
    const pkgCounts = new Map<string | null, number>();

    for (const [arr, err] of _leakTrackingMap) {
      const site = findCreationSite(err);
      const desc = safeDesc(arr) + safeRefCount(arr);
      details.push(`${desc} created at ${site.location}`);

      let g = groups.get(site.location);
      if (!g) {
        g = {
          count: 0,
          descs: [],
          via: site.via,
          pkg: site.pkg,
          maxRc: 0,
          lastRefLocs: [],
        };
        groups.set(site.location, g);
      }
      g.count++;
      g.maxRc = Math.max(g.maxRc, safeRefCountNum(arr));
      if (g.descs.length < 4) g.descs.push(desc);

      // Collect last .ref sites for rc≥2 arrays
      const refErr = _lastRefMap.get(arr);
      if (refErr && safeRefCountNum(arr) >= 2) {
        const refSite = findCreationSite(refErr);
        if (
          g.lastRefLocs.length < 2 &&
          !g.lastRefLocs.includes(refSite.location)
        ) {
          g.lastRefLocs.push(refSite.location);
        }
      }
      pkgCounts.set(site.pkg, (pkgCounts.get(site.pkg) ?? 0) + 1);
    }

    _leakTrackingMap.clear();
    _lastRefMap.clear();

    // User-code sites first (most actionable), then dependency, by count desc
    const sorted = [...groups.entries()].sort(
      (a, b) =>
        Number(a[1].pkg !== null) - Number(b[1].pkg !== null) ||
        b[1].count - a[1].count,
    );

    const total = [...pkgCounts.values()].reduce((a, b) => a + b, 0);
    const lines: string[] = [
      `${leaked} slot(s) leaked (${total} tracked array(s)):\n`,
    ];

    for (const [loc, g] of sorted) {
      const descs =
        g.descs.join(", ") + (g.count > g.descs.length ? ", …" : "");
      const tagParts: string[] = [];
      if (g.pkg) tagParts.push(g.pkg);
      if (g.via) tagParts.push(`via ${g.via}`);
      const tag = tagParts.length > 0 ? `  (${tagParts.join(", ")})` : "";
      lines.push(`  ${loc} — ${g.count}× ${descs}${tag}`);
      // Show last .ref sites for rc≥2 leaks
      for (const refLoc of g.lastRefLocs) {
        lines.push(`    ↳ last .ref at ${refLoc}`);
      }
    }

    const summaryParts: string[] = [];
    for (const [pkg, count] of pkgCounts) {
      summaryParts.push(pkg ? `${count} in ${pkg}` : `${count} from user code`);
    }
    if (summaryParts.length > 0) {
      lines.push("", summaryParts.join(", ") + ".");
    }

    // Diagnostic tips based on observed patterns
    const tips: string[] = [];
    if (leaked > total) {
      tips.push(
        `${leaked - total} slot(s) not in tracked list (created before start() or by internal buffers).`,
      );
    }
    if (total > leaked) {
      tips.push(
        `Some tracked arrays share backend storage (${total} tracked, ${leaked} slots).`,
      );
    }
    if (sorted.some(([, g]) => !g.pkg && g.maxRc === 1)) {
      tips.push(
        "rc=1 → never disposed. Use `using` for local temporaries, and call .dispose() at end-of-life (including early .dispose() when no longer needed to reduce peak memory).",
      );
    }
    if (sorted.some(([, g]) => g.maxRc >= 2)) {
      tips.push("rc≥2 → extra .ref without matching .dispose().");
    }
    if (pkgCounts.size > (pkgCounts.has(null) ? 1 : 0)) {
      tips.push("Package-tagged leaks are library bugs, not user error.");
    }
    if (tips.length === 0) {
      tips.push(
        "Prefer `using` for local temporaries; call .dispose() for values kept across scopes.",
      );
    }
    lines.push("", ...tips);

    const summary = lines.join("\n");
    const userLeaked = pkgCounts.get(null) ?? 0;
    const internalLeaked = total - userLeaked;
    return { leaked, userLeaked, internalLeaked, details, summary };
  },

  /** Whether leak tracking is currently active. */
  get active(): boolean {
    return _active;
  },
};
