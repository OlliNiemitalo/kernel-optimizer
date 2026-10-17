// Global tracing state for the profiler API.

import { Kernel } from "./alu";
import { isSymbolicSize, type SizeExpr } from "./dim";
import { Routine } from "./routine";

let traceEnabled = false;
const flushCallbacks: (() => void)[] = [];

/**
 * Start collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
export function startTrace(): void {
  traceEnabled = true;
}

/**
 * Stop collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
export function stopTrace(): void {
  traceEnabled = false;
  const cbs = flushCallbacks.splice(0);
  for (const cb of cbs) cb();
}

/** Check if tracing is currently enabled. */
export function isTracing(): boolean {
  return traceEnabled;
}

/** Register a callback to flush pending trace data when tracing stops. */
export function onFlushTrace(cb: () => void): void {
  flushCallbacks.push(cb);
}

export interface TraceInfo {
  label: string;
  color: string;
  properties: [string, string][];
}

function humanSize(n: SizeExpr): string {
  if (isSymbolicSize(n)) return `${n}`;
  if (n >= 1e9) return `${(n / 1e9).toPrecision(3)}B`;
  if (n >= 1e6) return `${(n / 1e6).toPrecision(3)}M`;
  if (n >= 1e3) return `${(n / 1e3).toPrecision(3)}K`;
  return `${n}`;
}

/** Build a trace label, properties, and color from a kernel or routine source. */
export function traceSourceInfo(source: Kernel | Routine): TraceInfo {
  const properties: [string, string][] = [];
  let label: string;
  let color: string;
  if (source instanceof Kernel) {
    label = `Kernel[${humanSize(source.size)}]`;
    properties.push(["exp", `${source.outputs[0].exp}`]);
    properties.push(["size", `${source.size}`]);
    properties.push(["nargs", `${source.nargs}`]);
    if (source.isMultiOutput) {
      properties.push(["outputs", `${source.numOutputs}`]);
    }
    if (!source.hasReduction) {
      color = "primary";
    } else {
      color = "secondary";
      const red = source.outputs[0].reduction;
      if (red) {
        properties.push(["reduction", `${red.op}:${red.size}`]);
      }
    }
  } else {
    color = "tertiary";
    label = source.name;
    properties.push([
      "inputShapes",
      source.type.inputShapes.map((s) => `[${s}]`).join(", "),
    ]);
    properties.push([
      "outputShapes",
      source.type.outputShapes.map((s) => `[${s}]`).join(", "),
    ]);
    properties.push(["dtype", source.type.inputDtypes.join(", ")]);
  }
  return { label, color, properties };
}

/** Emit a trace entry as a `performance.measure` with devtools metadata. */
export function emitTrace(
  track: string,
  info: TraceInfo,
  start: number,
  end: number,
): void {
  performance.measure(info.label, {
    detail: {
      devtools: {
        trackGroup: "JAX Profiler",
        track,
        color: info.color,
        properties: info.properties,
      },
    },
    start,
    end,
  });
}
