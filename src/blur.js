// Separable circular blur with phased Gaussian components, as plain WebGPU compute shaders.
//
// The kernel is a sum of separable complex channels n with 1-D kernels h_n (horizontal) and
// v_n (vertical), each of the form (k0 + k2 (t/R)²) exp((-b + i c) (t/R)²), t = -K..K taps,
// R = disc radius in pixels. The 2-D kernel is Σ_n Re[h_n(x) v_n(y)]. A single component is one
// channel with h = v; a merged pair (repeated rate) is two channels (see separableChannels).
//   Pass 1 (horizontal): mid_n = I *_x h_n           (complex, per color channel)
//   Pass 2 (vertical):   out = Σ_n Re[mid_n *_y v_n] (only the real part is ever formed)
// Each 1-D pass costs O(taps) per pixel, independent of the 2-D kernel area.

import { separableChannels } from "./problem.js";

export const MAX_CHANNELS = 16;
const ROWS = 8;

function computeShader(f16) {
  const MID = f16 ? "vec2<f16>" : "vec2f";
  return /* wgsl */ `
${f16 ? "enable f16;" : ""}

const ROWS = 8u; // output rows per thread in the vertical pass

struct Params {
  width: u32,
  height: u32,
  halfTaps: i32,   // K: taps run over -K..K
  taps: u32,       // 2K + 1
  chans: u32,      // separable complex channels
  invNorm: f32,    // 1 / (sum of the 2-D kernel), preserves brightness
  stride: u32,     // per-channel length of a tap table: taps plus ROWS - 1 zeros on each side
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;        // linear RGB
@group(0) @binding(2) var<storage, read> kern: array<vec2f>;       // [h | v][chan][stride] complex taps
@group(0) @binding(3) var<storage, read_write> mid: array<${MID}>; // [chan][color][y][x]
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;  // linear RGB

const PAD = ROWS - 1u;

@compute @workgroup_size(64, 1)
fn horizontal(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.width || id.y >= P.height) { return; }
  let row = id.y * P.width;
  let plane = P.width * P.height;
  // Channels in the outer loop keep the accumulators in registers; repeated
  // source reads hit the cache.
  for (var n = 0u; n < P.chans; n++) {
    var re = vec3f(0.0);
    var im = vec3f(0.0);
    for (var t = 0u; t < P.taps; t++) {
      let sx = clamp(i32(id.x) + i32(t) - P.halfTaps, 0, i32(P.width) - 1); // clamp to edge
      let p = src[row + u32(sx)].rgb;
      let h = kern[n * P.stride + PAD + t];
      re += p * h.x;
      im += p * h.y;
    }
    let i = n * 3u * plane + row + id.x;
    mid[i] = ${MID}(vec2f(re.r, im.r));
    mid[i + plane] = ${MID}(vec2f(re.g, im.g));
    mid[i + 2u * plane] = ${MID}(vec2f(re.b, im.b));
  }
}

@compute @workgroup_size(64, 1)
fn vertical(@builtin(global_invocation_id) id: vec3u) {
  let y0 = i32(id.y * ROWS);
  if (id.x >= P.width || y0 >= i32(P.height)) { return; }
  let plane = P.width * P.height;
  // Each thread produces ROWS consecutive output rows, so every input row it reads
  // (once) feeds up to ROWS outputs: tap t = s - (y0 + j) + K for output row y0 + j.
  var acc: array<vec3f, ROWS>;
  for (var n = 0u; n < P.chans; n++) {
    for (var s = y0 - P.halfTaps; s < y0 + i32(ROWS) + P.halfTaps; s++) {
      let i = n * 3u * plane + u32(clamp(s, 0, i32(P.height) - 1)) * P.width + id.x;
      let r = vec2f(mid[i]);
      let g = vec2f(mid[i + plane]);
      let b = vec2f(mid[i + 2u * plane]);
      let re = vec3f(r.x, g.x, b.x);
      let im = vec3f(r.y, g.y, b.y);
      // Tap for output row y0 + j is s - y0 - j + K; out-of-range taps read the zero padding.
      let base = (P.chans + n) * P.stride + PAD + u32(s - y0 + P.halfTaps); // vertical table
      for (var j = 0u; j < ROWS; j++) {
        let h = kern[base - j];
        acc[j] += re * h.x - im * h.y; // Re[(re + i im)(h.x + i h.y)]
      }
    }
  }
  for (var j = 0u; j < ROWS; j++) {
    let y = u32(y0) + j;
    if (y < P.height) {
      dst[y * P.width + id.x] = vec4f(acc[j] * P.invNorm, 1.0);
    }
  }
}
`;
}

const displayShader = /* wgsl */ `
struct View {
  width: u32,
  height: u32,
  exposure: f32,
  tonemap: u32,   // 0 = clip, 1 = Reinhard
  showSource: u32,
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> blurred: array<vec4f>;
@group(0) @binding(2) var<storage, read> source: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let i = u32(pos.y) * V.width + u32(pos.x);
  var c = select(blurred[i].rgb, source[i].rgb, V.showSource == 1u);
  c = max(c * V.exposure, vec3f(0.0)); // ringing can go slightly negative
  if (V.tonemap == 1u) { c = c / (1.0 + c); }
  return vec4f(toSrgb(min(c, vec3f(1.0))), 1.0);
}
`;

export class BlurRenderer {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const supportsF16 = adapter.features.has("shader-f16");
    const supportsTimestamps = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: [supportsF16 && "shader-f16", supportsTimestamps && "timestamp-query"].filter(Boolean),
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    return new BlurRenderer(canvas, device, supportsF16, supportsTimestamps);
  }

  constructor(canvas, device, supportsF16, supportsTimestamps) {
    this.canvas = canvas;
    this.device = device;
    this.supportsF16 = supportsF16;
    if (supportsTimestamps) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 4 });
      this.queryResolve = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.queryRead = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.view = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pipelines = {};
    const module = device.createShaderModule({ code: displayShader });
    this.displayPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
    });
    this.precision = "f32";
    const storage = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("storage"),
        storage("storage"),
      ].map((e, binding) => ({ binding, ...e })),
    });
  }

  computePipelines(f16) {
    const key = f16 ? "f16" : "f32";
    if (!this.pipelines[key]) {
      const module = this.device.createShaderModule({ code: computeShader(f16) });
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
      const make = (entryPoint) => this.device.createComputePipeline({ layout, compute: { module, entryPoint } });
      this.pipelines[key] = { horizontal: make("horizontal"), vertical: make("vertical") };
    }
    return this.pipelines[key];
  }

  setImage(rgba, width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.source?.destroy();
    this.dst?.destroy();
    this.source = this.device.createBuffer({ size: rgba.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.source, 0, rgba);
    this.dst = this.device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.mid?.destroy();
    this.mid = null;
  }

  setKernel(params, radius, transition) {
    const channels = separableChannels(params);
    if (channels.length > MAX_CHANNELS) throw new Error(`At most ${MAX_CHANNELS} separable channels.`);
    const K = Math.max(1, Math.ceil(radius * (1 + transition)));
    const taps = 2 * K + 1;
    const pad = ROWS - 1;
    const stride = taps + 2 * pad;
    const n = channels.length;
    const data = new Float32Array(2 * n * stride * 2);
    // Writes (k0 + k2 s) e^{(-b + i c) s} taps into table slot `slot`; returns their sum.
    const write = (slot, { decay, phase }, [k0r, k0i], [k2r, k2i]) => {
      let sumRe = 0;
      let sumIm = 0;
      for (let t = 0; t < taps; t++) {
        const s = ((t - K) / radius) ** 2;
        const e = Math.exp(-decay * s);
        const er = e * Math.cos(phase * s);
        const ei = e * Math.sin(phase * s);
        const wr = k0r + k2r * s;
        const wi = k0i + k2i * s;
        const o = (slot * stride + pad + t) * 2;
        data[o] = wr * er - wi * ei;
        data[o + 1] = wr * ei + wi * er;
        sumRe += data[o];
        sumIm += data[o + 1];
      }
      return [sumRe, sumIm];
    };
    let total = 0;
    channels.forEach((ch, k) => {
      const [hr, hi] = write(k, ch, ch.h0, ch.h2);
      const [vr, vi] = write(n + k, ch, ch.v0, ch.v2);
      total += hr * vr - hi * vi; // Re[(Σ h)(Σ v)]: this channel's 2-D sum
    });
    this.kernel?.destroy();
    this.kernel = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.kernel, 0, data);
    this.channels = n;
    this.K = K;
    this.taps = taps;
    this.stride = stride;
    this.invNorm = 1 / total;
  }

  // Runs both passes and draws; resolves with the wall time until the GPU finished.
  async render({ exposure = 1, tonemap = 1, precision = "f32", showSource = false } = {}) {
    const f16 = precision === "f16" && this.supportsF16;
    const { width, height, device } = this;
    const bytesPerValue = f16 ? 4 : 8;
    const midBytes = this.channels * 3 * width * height * bytesPerValue;
    if (!this.mid || this.mid.size < midBytes) {
      this.mid?.destroy();
      this.mid = device.createBuffer({ size: midBytes, usage: GPUBufferUsage.STORAGE });
    }
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Int32Array(u, 8, 1)[0] = this.K;
    new Uint32Array(u, 12, 2).set([this.taps, this.channels]);
    new Float32Array(u, 20, 1)[0] = this.invNorm;
    new Uint32Array(u, 24, 1)[0] = this.stride;
    device.queue.writeBuffer(this.params, 0, u);
    const v = new ArrayBuffer(32);
    new Uint32Array(v, 0, 2).set([width, height]);
    new Float32Array(v, 8, 1)[0] = exposure;
    new Uint32Array(v, 12, 2).set([tonemap, showSource ? 1 : 0]);
    device.queue.writeBuffer(this.view, 0, v);

    const { horizontal, vertical } = this.computePipelines(f16);
    const bindGroup = device.createBindGroup({
      layout: this.computeLayout,
      entries: [this.params, this.source, this.kernel, this.mid, this.dst].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    const groups = { horizontal: [Math.ceil(width / 64), height], vertical: [Math.ceil(width / 64), Math.ceil(height / ROWS)] };
    for (const [k, [name, pipeline]] of [["horizontal", horizontal], ["vertical", vertical]].entries()) {
      const pass = encoder.beginComputePass(
        this.querySet
          ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } }
          : {},
      );
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...groups[name]);
      pass.end();
    }
    const render = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    render.setPipeline(this.displayPipeline);
    render.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [this.view, this.dst, this.source].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    render.draw(3);
    render.end();
    const timed = this.querySet && this.queryRead.mapState === "unmapped";
    if (timed) {
      encoder.resolveQuerySet(this.querySet, 0, 4, this.queryResolve, 0);
      encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 32);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const result = { ms: performance.now() - t0, taps: this.taps, f16, midBytes };
    if (timed) {
      await this.queryRead.mapAsync(GPUMapMode.READ);
      const ns = new BigUint64Array(this.queryRead.getMappedRange().slice(0));
      this.queryRead.unmap();
      result.horizontalMs = Number(ns[1] - ns[0]) / 1e6;
      result.verticalMs = Number(ns[3] - ns[2]) / 1e6;
    }
    return result;
  }
}
