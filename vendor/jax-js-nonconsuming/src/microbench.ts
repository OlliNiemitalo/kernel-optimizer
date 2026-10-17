// Microbenchmark-driven auto-tuning (P10).
//
// Isolated GPU microbenchmarks that measure fundamental hardware characteristics
// WITHOUT benchmarking compiler choices. All benchmarks run on raw WebGPU API
// to avoid self-reinforcing bias from our own JIT decisions.
//
// Measured variables:
//   C_dispatch  — Fixed dispatch overhead (JS + queue.submit)
//   BW_global   — VRAM streaming bandwidth (GB/s)
//   TFLOPS      — Compute throughput (TFLOPS)
//   C_barrier   — Workgroup barrier cost
//   R_opt       — Effective register budget (words before spill)

import { DEBUG } from "./utils";

/** Frozen performance parameters from microbenchmarks. */
export interface PerformanceBeliefState {
  /** Fixed dispatch overhead in microseconds. */
  readonly dispatchOverheadUs: number;
  /** Global memory bandwidth in GB/s. */
  readonly bandwidthGBs: number;
  /** Compute throughput in TFLOPS. */
  readonly tflops: number;
  /** Workgroup barrier cost relative to ALU (1.0 = free). */
  readonly barrierCostFactor: number;
  /** Effective per-thread register budget (words before spill). */
  readonly rOptWords: number;
}

/**
 * Hash a PerformanceBeliefState to a stable string for use in cache keys.
 * Rounds to 3 significant figures to prevent trivial floating-point noise
 * from busting the cache.
 */
export function hashBeliefState(state: PerformanceBeliefState): string {
  const r = (v: number) => Number(v.toPrecision(3));
  return `d${r(state.dispatchOverheadUs)}b${r(state.bandwidthGBs)}t${r(state.tflops)}w${r(state.barrierCostFactor)}r${state.rOptWords}`;
}

// ── Microbenchmark kernels ───────────────────────────────────────────────

const WARMUP_ITERATIONS = 3;
const MEASURE_ITERATIONS = 5;

/** Run the full microbenchmark suite on a GPUDevice. */
export async function runMicrobenchmarks(
  device: GPUDevice,
): Promise<PerformanceBeliefState> {
  if (DEBUG >= 1) console.log("[microbench] starting calibration");
  const t0 = performance.now();

  const dispatchOverheadUs = await measureDispatchOverhead(device);

  const [bandwidthGBs, tflops, barrierCostFactor, rOptWords] =
    await Promise.all([
      measureBandwidth(device, dispatchOverheadUs),
      measureTflops(device, dispatchOverheadUs),
      measureBarrierCost(device),
      measureRopt(device),
    ]);

  const state: PerformanceBeliefState = {
    dispatchOverheadUs,
    bandwidthGBs,
    tflops,
    barrierCostFactor,
    rOptWords,
  };

  if (DEBUG >= 1) {
    const elapsed = (performance.now() - t0).toFixed(1);
    console.log(
      `[microbench] calibration done in ${elapsed}ms: ` +
        `dispatch=${dispatchOverheadUs.toFixed(1)}µs bw=${bandwidthGBs.toFixed(1)}GB/s ` +
        `tflops=${tflops.toFixed(3)} barrier=${barrierCostFactor.toFixed(2)} rOpt=${rOptWords}`,
    );
  }
  return state;
}

// ── Helpers ──────────────────────────────────────────────────────────────

/** Create a compute pipeline from inline WGSL. */
function createPipeline(
  device: GPUDevice,
  wgsl: string,
  label: string,
): GPUComputePipeline {
  const module = device.createShaderModule({ code: wgsl, label });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label,
  });
}

/** Time `fn` over warmup+measure iterations, return median milliseconds. */
async function timedGpuMs(
  device: GPUDevice,
  fn: (encoder: GPUCommandEncoder) => void,
  iterations: number = MEASURE_ITERATIONS,
): Promise<number> {
  // Warmup
  for (let i = 0; i < WARMUP_ITERATIONS; i++) {
    const enc = device.createCommandEncoder();
    fn(enc);
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const enc = device.createCommandEncoder();
    fn(enc);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

// ── 1. Dispatch Overhead (C_dispatch) ────────────────────────────────────
// Measures fixed JS + queue.submit cost via grouped no-op dispatches.

const NOOP_SHADER = /* wgsl */ `
@compute @workgroup_size(1)
fn main() {}
`;

async function measureDispatchOverhead(device: GPUDevice): Promise<number> {
  const pipeline = createPipeline(device, NOOP_SHADER, "microbench-noop");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [],
  });

  // Measure N dispatches in a single submit, then compute per-dispatch cost.
  const counts = [1, 8, 32, 64];
  const results: { count: number; ms: number }[] = [];

  for (const count of counts) {
    const ms = await timedGpuMs(device, (enc) => {
      for (let i = 0; i < count; i++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
      }
    });
    results.push({ count, ms });
  }

  // Linear regression: ms = base + count * C_dispatch_ms
  // Use least-squares on the (count, ms) pairs.
  const n = results.length;
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0;
  for (const { count, ms } of results) {
    sumX += count;
    sumY += ms;
    sumXY += count * ms;
    sumXX += count * count;
  }
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const dispatchUs = Math.max(1, slope * 1000); // ms → µs, floor at 1µs

  return dispatchUs;
}

// ── 2. Bandwidth (BW_global) ─────────────────────────────────────────────
// Streams a large contiguous buffer through a trivial copy kernel.

const COPY_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx < arrayLength(&src)) {
    dst[idx] = src[idx];
  }
}
`;

async function measureBandwidth(
  device: GPUDevice,
  cDispatchUs: number,
): Promise<number> {
  // Use a buffer larger than typical L2 caches (e.g., RTX 4070 Ti has 48MB L2).
  // 4M vec4s = 64MB buffer -> 128MB total footprint per pass.
  // We cap the allocation to device.limits.maxBufferSize / 2 if needed.
  const limitMaxTokens = Math.floor(
    (device.limits.maxBufferSize ?? 268435456) / 2 / 16,
  );
  const numVec4 = Math.min(4 * 1024 * 1024, limitMaxTokens); // up to 64MB per buffer
  const byteSize = numVec4 * 16;
  const workgroups = Math.ceil(numVec4 / 256);

  const srcBuf = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE,
  });
  const dstBuf = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE,
  });

  const pipeline = createPipeline(device, COPY_SHADER, "microbench-bw");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: srcBuf } },
      { binding: 1, resource: { buffer: dstBuf } },
    ],
  });

  // dispatch 10 times to accumulate enough time (up to 1.28 GB total moved per measurement)
  const ms = await timedGpuMs(device, (enc) => {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < 10; i++) {
      pass.dispatchWorkgroups(workgroups);
    }
    pass.end();
  });

  srcBuf.destroy();
  dstBuf.destroy();

  // Subtract base dispatch overhead to uncover true throughput.
  const effectiveMs = Math.max(0.1, ms - cDispatchUs / 1000);

  // byteSize read + byteSize written; convert ms → seconds
  const bwGBs = (2 * byteSize * 10) / (effectiveMs / 1000) / 1e9;
  return Math.max(1, bwGBs); // floor at 1 GB/s
}

// ── 3. TFLOPS Compute Throughput ─────────────────────────────────────────
// FMA-heavy kernel with minimal memory traffic.

const FMA_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  var a = f32(gid.x) * 0.001;
  var b = a + 1.0;
  var c = a + 2.0;
  // 4096 FMAs per thread (4 per iteration × 1024 iterations)
  for (var i = 0u; i < 1024u; i++) {
    a = a * b + c; b = b * a + c; c = c * b + a; a = a * c + b;
  }
  out[gid.x] = a + b + c;
}
`;

async function measureTflops(
  device: GPUDevice,
  cDispatchUs: number,
): Promise<number> {
  // Launch 2M threads. Each does 4096 FMAs.
  // 2M * 4096 * 2 = 17.1 GFLOPs per dispatch.
  const numThreads = 2048 * 1024;
  const workgroups = numThreads / 256;
  const fmaPerThread = 4096; // 4 FMA/iteration × 1024 iterations
  const totalFlops = numThreads * fmaPerThread * 2; // FMA = 2 FLOPs

  const outBuf = device.createBuffer({
    size: numThreads * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const pipeline = createPipeline(device, FMA_SHADER, "microbench-tflops");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: outBuf } }],
  });

  // dispatch 10 times to accumulate ~171 GFLOPs.
  // Intel @ 1 TFLOPS ≈ 171ms. RTX 4070Ti @ 22 TFLOPS ≈ 7.7ms.
  const ms = await timedGpuMs(device, (enc) => {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < 10; i++) {
      pass.dispatchWorkgroups(workgroups);
    }
    pass.end();
  });

  outBuf.destroy();

  const effectiveMs = Math.max(0.1, ms - cDispatchUs / 1000);
  const tflops = (totalFlops * 10) / (effectiveMs / 1000) / 1e12;
  return Math.max(0.01, tflops); // floor at 0.01 TFLOPS
}

// ── 4. Barrier Cost ──────────────────────────────────────────────────────
// Measures workgroupBarrier() cost relative to ALU by comparing barrier-heavy
// vs barrier-free workgroup reductions.

const BARRIER_SHADER = /* wgsl */ `
var<workgroup> shmem: array<f32, 256>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  shmem[lid.x] = f32(lid.x);
  workgroupBarrier();
  var s = shmem[lid.x];
  // 8 barrier-interleaved reductions
  for (var stride = 128u; stride >= 1u; stride >>= 1u) {
    if (lid.x < stride) { shmem[lid.x] = shmem[lid.x] + shmem[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { out[wid.x] = shmem[0]; }
}
`;

const NO_BARRIER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  var s = f32(gid.x);
  // Same number of ALU ops but no barriers or shared memory
  for (var i = 0u; i < 8u; i++) {
    s = s + s * 0.5;
  }
  out[gid.x] = s;
}
`;

async function measureBarrierCost(device: GPUDevice): Promise<number> {
  const numWorkgroups = 1024;
  const numThreads = numWorkgroups * 256;

  const outBuf = device.createBuffer({
    size: numWorkgroups * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const outBuf2 = device.createBuffer({
    size: numThreads * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const barrierPipeline = createPipeline(
    device,
    BARRIER_SHADER,
    "microbench-barrier",
  );
  const noBarrierPipeline = createPipeline(
    device,
    NO_BARRIER_SHADER,
    "microbench-no-barrier",
  );

  const barrierBG = device.createBindGroup({
    layout: barrierPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: outBuf } }],
  });
  const noBarrierBG = device.createBindGroup({
    layout: noBarrierPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: outBuf2 } }],
  });

  const barrierMs = await timedGpuMs(device, (enc) => {
    const pass = enc.beginComputePass();
    pass.setPipeline(barrierPipeline);
    pass.setBindGroup(0, barrierBG);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
  });

  const noBarrierMs = await timedGpuMs(device, (enc) => {
    const pass = enc.beginComputePass();
    pass.setPipeline(noBarrierPipeline);
    pass.setBindGroup(0, noBarrierBG);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
  });

  outBuf.destroy();
  outBuf2.destroy();

  // Ratio: > 1.0 means barriers are expensive, 1.0 means free
  const factor = noBarrierMs > 0 ? Math.max(1.0, barrierMs / noBarrierMs) : 1.0;
  return factor;
}

// ── 5. Effective Register Budget (R_opt) ─────────────────────────────────
// Launches kernels with escalating private variable counts to find the
// throughput cliff where register spilling tanks performance.

function makeRegShader(numVars: number): string {
  let body = "";
  // Declare private variables
  for (let i = 0; i < numVars; i++) {
    body += `  var r${i} = f32(gid.x) * ${(0.001 * (i + 1)).toFixed(4)};\n`;
  }
  // Chain FMA ops across all variables to prevent dead-code elimination
  for (let round = 0; round < 4; round++) {
    for (let i = 0; i < numVars; i++) {
      const j = (i + 1) % numVars;
      body += `  r${i} = r${i} * r${j} + r${(i + 2) % numVars};\n`;
    }
  }
  // Reduce to output
  let sum = "r0";
  for (let i = 1; i < numVars; i++) sum += ` + r${i}`;

  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
${body}
  out[gid.x] = ${sum};
}
`;
}

async function measureRopt(device: GPUDevice): Promise<number> {
  const numThreads = 64 * 1024;
  const workgroups = numThreads / 256;

  const outBuf = device.createBuffer({
    size: numThreads * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Test register counts: 16, 32, 48, 64, 96, 128, 192
  const regCounts = [16, 32, 48, 64, 96, 128, 192];
  const throughputs: { regs: number; tput: number }[] = [];

  for (const regs of regCounts) {
    let pipeline: GPUComputePipeline;
    try {
      pipeline = createPipeline(
        device,
        makeRegShader(regs),
        `microbench-reg-${regs}`,
      );
    } catch {
      // Shader compilation failure at high register counts — stop probing.
      break;
    }

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: outBuf } }],
    });

    const ms = await timedGpuMs(
      device,
      (enc) => {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
      },
      3,
    ); // fewer iterations since we have many probes

    // Throughput: FMA ops / ms (higher is better)
    const fmaOps = numThreads * regs * 4; // 4 rounds of FMAs
    const tput = fmaOps / ms;
    throughputs.push({ regs, tput });
  }

  if (throughputs.length < 2) {
    outBuf.destroy();
    return 64; // conservative default
  }

  // Find the register count where throughput drops to <60% of peak.
  // This indicates register spilling has begun.
  const peakTput = Math.max(...throughputs.map((t) => t.tput));
  let rOpt = throughputs[0].regs;
  for (const { regs, tput } of throughputs) {
    if (tput >= peakTput * 0.6) {
      rOpt = regs;
    }
  }

  outBuf.destroy();
  return rOpt;
}
