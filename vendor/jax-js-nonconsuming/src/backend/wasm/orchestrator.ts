/**
 * Orchestrator worker for off-main-thread mega-module execution (M6.2b).
 *
 * Moves `mega_execute` to a dedicated Web Worker so the main thread is
 * free (or spin-waiting) and the orchestrator can use `Atomics.wait`
 * (required for M6.2c parallel kernel dispatch).
 *
 * **Alloc/free proxy:** The mega-module's `env.alloc`/`env.free` imports
 * proxy back to the main thread's `WasmAllocator` via a shared control
 * buffer + `Atomics.wait`/`Atomics.notify`. The main thread spin-waits
 * (matching `WasmWorkerPool`'s pattern) during mega_execute, servicing
 * alloc/free requests.
 *
 * **Protocol:**
 *
 * ```
 * Main Thread                      Orchestrator Worker
 *     │                                  │
 *     ├─ write params to controlBuf      │
 *     ├─ postMessage("dispatch")────────→│
 *     ├─ spin-loop:                      ├─ mega_execute(...)
 *     │    ALLOC_REQ → malloc, respond   │    call $alloc → ALLOC_REQ → wait
 *     │    FREE_REQ  → free, respond     │    call $free  → FREE_REQ  → wait
 *     │    DONE      → break             │    ← done
 *     │←────────────────────────────────←│
 * ```
 */

import { DEBUG } from "../../utils";

// ---------------------------------------------------------------------------
// Control buffer layout (Int32 slots on a SharedArrayBuffer)
// ---------------------------------------------------------------------------

/** Orchestrator idle, waiting for dispatch message. */
const STATE_IDLE = 0;
/** Params written, orchestrator should run mega_execute. */
const STATE_DISPATCHED = 1;
/** Orchestrator requests alloc; size at AUX, result expected at AUX_RESULT. */
const STATE_ALLOC_REQ = 2;
/** Orchestrator requests free; ptr at AUX. */
const STATE_FREE_REQ = 3;
/** Orchestrator finished mega_execute successfully. */
const STATE_DONE = 4;
/** Orchestrator hit an error during mega_execute. */
const STATE_ERROR = 5;
/** Module registration confirmed by worker. */
const STATE_REGISTERED = 6;

// Control buffer field offsets (Int32 indices):
const CTRL_STATE = 0;
const CTRL_MODULE_ID = 1;
const CTRL_NUM_INPUTS = 2;
const CTRL_RESULT_PTR = 3;
/** Alloc size (for ALLOC_REQ) or free ptr (for FREE_REQ). */
const CTRL_AUX = 4;
/** Alloc result pointer (written by main thread). */
const CTRL_AUX_RESULT = 5;
/** Start of input pointer array. */
const CTRL_INPUTS = 6;
/** Maximum number of mega-module inputs. */
const MAX_INPUTS = 64;
/** Total Int32 slots in the control buffer. */
const CTRL_TOTAL = CTRL_INPUTS + MAX_INPUTS; // 70

// ---------------------------------------------------------------------------
// Inline worker source
// ---------------------------------------------------------------------------

function buildOrchestratorCode(): string {
  // Self-contained JS — no imports, constants baked in.
  return `
"use strict";
var STATE = ${CTRL_STATE};
var MODULE_ID = ${CTRL_MODULE_ID};
var NUM_INPUTS = ${CTRL_NUM_INPUTS};
var RESULT_PTR = ${CTRL_RESULT_PTR};
var AUX = ${CTRL_AUX};
var AUX_RESULT = ${CTRL_AUX_RESULT};
var INPUTS = ${CTRL_INPUTS};

var IDLE = ${STATE_IDLE};
var DISPATCHED = ${STATE_DISPATCHED};
var ALLOC_REQ = ${STATE_ALLOC_REQ};
var FREE_REQ = ${STATE_FREE_REQ};
var DONE = ${STATE_DONE};
var ERROR = ${STATE_ERROR};
var REGISTERED = ${STATE_REGISTERED};

var memory = null;
var control = null;
var instances = Object.create(null); // moduleId -> WebAssembly.Instance

// Alloc proxy — blocks the worker until the main thread responds.
function proxyAlloc(size) {
  Atomics.store(control, AUX, size);
  Atomics.store(control, STATE, ALLOC_REQ);
  Atomics.notify(control, STATE); // wake main spin-loop (harmless if already spinning)
  Atomics.wait(control, STATE, ALLOC_REQ); // blocks until main changes state
  return Atomics.load(control, AUX_RESULT);
}

// Free proxy — blocks until main acknowledges.
function proxyFree(ptr) {
  Atomics.store(control, AUX, ptr);
  Atomics.store(control, STATE, FREE_REQ);
  Atomics.notify(control, STATE);
  Atomics.wait(control, STATE, FREE_REQ);
}

function runDispatch() {
  var moduleId = Atomics.load(control, MODULE_ID);
  var numInputs = Atomics.load(control, NUM_INPUTS);
  var resultPtr = Atomics.load(control, RESULT_PTR);
  var args = new Array(numInputs + 1);
  for (var i = 0; i < numInputs; i++) {
    args[i] = Atomics.load(control, INPUTS + i);
  }
  args[numInputs] = resultPtr;
  var inst = instances[moduleId];
  inst.exports.mega_execute.apply(null, args);
}

self.onmessage = function (e) {
  var msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    control = new Int32Array(msg.controlBuffer);
  } else if (msg.type === "register") {
    try {
      var inst = new WebAssembly.Instance(msg.module, {
        env: {
          memory: memory,
          alloc: proxyAlloc,
          free: proxyFree,
        },
      });
      instances[msg.moduleId] = inst;
      // Signal registration via control buffer + postMessage.
      Atomics.store(control, STATE, REGISTERED);
      Atomics.notify(control, STATE);
      self.postMessage({ type: "registered", moduleId: msg.moduleId });
    } catch (err) {
      Atomics.store(control, STATE, ERROR);
      Atomics.notify(control, STATE);
      self.postMessage({ type: "error", moduleId: msg.moduleId, error: String(err) });
    }
  } else if (msg.type === "dispatch") {
    try {
      runDispatch();
      Atomics.store(control, STATE, DONE);
    } catch (err) {
      console.error("Orchestrator error:", err);
      Atomics.store(control, STATE, ERROR);
    }
    Atomics.notify(control, STATE);
  } else if (msg.type === "destroy") {
    self.close();
  }
};
`;
}

// ---------------------------------------------------------------------------
// OrchestratorWorker class (main-thread side)
// ---------------------------------------------------------------------------

export class OrchestratorWorker {
  #worker: Worker;
  #controlBuf: SharedArrayBuffer;
  #control: Int32Array;
  #destroyed = false;

  // Module registration tracking
  #nextModuleId = 0;
  #moduleIds = new WeakMap<WebAssembly.Module, number>();
  #registeredModules = new Set<number>();

  constructor(memory: WebAssembly.Memory) {
    // Create shared control buffer
    this.#controlBuf = new SharedArrayBuffer(CTRL_TOTAL * 4);
    this.#control = new Int32Array(this.#controlBuf);
    Atomics.store(this.#control, CTRL_STATE, STATE_IDLE);

    // Create worker from inline Blob URL
    const code = buildOrchestratorCode();
    const blob = new Blob([code], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    this.#worker = new Worker(url, { type: "module" });
    // Defer revocation — module workers may not have loaded the blob
    // URL synchronously by the time revokeObjectURL is called.
    setTimeout(() => URL.revokeObjectURL(url), 0);

    // Send init message with shared memory and control buffer
    this.#worker.postMessage({
      type: "init",
      memory,
      controlBuffer: this.#controlBuf,
    });
  }

  /**
   * Register a WASM mega-module on the orchestrator.
   *
   * Uses `postMessage` to send the module (structured-cloneable), then
   * spin-waits on the control buffer for the worker's confirmation.
   * Called synchronously from `executeMegaModule`.
   *
   * @returns The module ID for use in `dispatch()`.
   */
  registerModuleSync(module: WebAssembly.Module): number {
    const existing = this.#moduleIds.get(module);
    if (existing !== undefined) return existing;

    const id = this.#nextModuleId++;
    this.#moduleIds.set(module, id);

    // Send module to worker
    this.#worker.postMessage({ type: "register", moduleId: id, module });

    // Spin-wait for confirmation via control buffer.
    // The worker sets STATE → REGISTERED after instantiation.
    while (true) {
      const state = Atomics.load(this.#control, CTRL_STATE);
      if (state === STATE_REGISTERED) {
        Atomics.store(this.#control, CTRL_STATE, STATE_IDLE);
        break;
      }
      if (state === STATE_ERROR) {
        Atomics.store(this.#control, CTRL_STATE, STATE_IDLE);
        throw new Error(
          `Orchestrator: failed to register mega-module (id=${id})`,
        );
      }
      // Still IDLE — worker hasn't processed the message yet; keep spinning.
    }

    this.#registeredModules.add(id);
    if (DEBUG >= 1) {
      console.info(`orchestrator: registered module id=${id}`);
    }
    return id;
  }

  /**
   * Dispatch mega_execute to the orchestrator worker.
   *
   * Writes input pointers to the shared control buffer, sends a "dispatch"
   * message, then spin-waits while servicing alloc/free proxy requests.
   *
   * @param moduleId    Module ID from `registerModuleSync()`
   * @param inputPtrs   Raw memory pointers for mega-module inputs
   * @param resultBufPtr Pointer to the pre-allocated result buffer
   * @param malloc      Main-thread allocator's malloc function
   * @param free        Main-thread allocator's free function
   */
  dispatch(
    moduleId: number,
    inputPtrs: number[],
    resultBufPtr: number,
    malloc: (size: number) => number,
    free: (ptr: number) => void,
  ): void {
    if (this.#destroyed) throw new Error("OrchestratorWorker destroyed");
    if (!this.#registeredModules.has(moduleId)) {
      throw new Error(`Orchestrator: module ${moduleId} not registered`);
    }
    if (inputPtrs.length > MAX_INPUTS) {
      throw new Error(
        `Orchestrator: too many inputs (${inputPtrs.length} > ${MAX_INPUTS})`,
      );
    }

    // Write dispatch params to control buffer
    Atomics.store(this.#control, CTRL_MODULE_ID, moduleId);
    Atomics.store(this.#control, CTRL_NUM_INPUTS, inputPtrs.length);
    Atomics.store(this.#control, CTRL_RESULT_PTR, resultBufPtr);
    for (let i = 0; i < inputPtrs.length; i++) {
      Atomics.store(this.#control, CTRL_INPUTS + i, inputPtrs[i]);
    }
    Atomics.store(this.#control, CTRL_STATE, STATE_DISPATCHED);

    // Send dispatch trigger via postMessage (worker's onmessage handler).
    this.#worker.postMessage({ type: "dispatch" });

    // Spin-wait servicing alloc/free proxy requests until done.
    while (true) {
      const state = Atomics.load(this.#control, CTRL_STATE);

      if (state === STATE_ALLOC_REQ) {
        // Service alloc request
        const size = Atomics.load(this.#control, CTRL_AUX);
        const ptr = malloc(size);
        Atomics.store(this.#control, CTRL_AUX_RESULT, ptr);
        Atomics.store(this.#control, CTRL_STATE, STATE_DISPATCHED);
        Atomics.notify(this.#control, CTRL_STATE);
      } else if (state === STATE_FREE_REQ) {
        // Service free request
        const ptr = Atomics.load(this.#control, CTRL_AUX);
        free(ptr);
        Atomics.store(this.#control, CTRL_STATE, STATE_DISPATCHED);
        Atomics.notify(this.#control, CTRL_STATE);
      } else if (state === STATE_DONE) {
        // Mega-execute completed successfully
        Atomics.store(this.#control, CTRL_STATE, STATE_IDLE);
        return;
      } else if (state === STATE_ERROR) {
        // Mega-execute threw an error
        Atomics.store(this.#control, CTRL_STATE, STATE_IDLE);
        throw new Error("Orchestrator: mega_execute failed on worker");
      }
      // STATE_DISPATCHED → worker is running; keep spinning.
    }
  }

  /** Terminate the orchestrator worker and release resources. */
  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    this.#worker.postMessage({ type: "destroy" });
    this.#worker.terminate();
  }
}
