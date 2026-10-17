// Separable circular blur with phased Gaussian components, as plain WebGPU compute shaders.
//
// The kernel is a sum of separable complex channels n with 1-D kernels h_n (horizontal) and
// v_n (vertical), each of the form (k0 + k2 (t/R)²) exp((-b + i c) (t/R)²), t = -K..K taps,
// R = disc radius in pixels. The 2-D kernel is Σ_n Re[h_n(x) v_n(y)]; a single component is
// one channel, a merged pair (repeated rate) two (see separableChannels).
//
// Memory: channels are processed one at a time, so only ONE channel's complex intermediate
// exists, independent of the channel count:
//   for each channel n:  mid  = I *_x h_n                 (complex; lanes r, g, b, w)
//                        acc += Re[mid *_y v_n]            (real; acc = 0 before channel 0)
//   display:             out  = acc.rgb / acc.w
// Lane w filters a constant 1 alongside the color, so acc.w is the sum of the 2-D kernel
// weights that actually reached the pixel. Dividing by it normalizes brightness exactly for a
// uniform radius (also at clamped edges) and stays brightness-preserving when the radius varies.
//
// Spatially varying radius: per-pixel gather. Each pixel reads its radius from a map, picks the
// nearest of a geometric ladder of radii (tap tables precomputed per level; level 0 is an exact
// identity for in-focus pixels), and convolves with that level's taps in both passes. Across rows
// of different radius the 2-D kernel becomes a blend of neighboring levels' kernels, which suits
// smoothly varying radius (e.g. depth of field).
//
// Uniform radius uses a faster register-blocked path: each thread computes ROWS consecutive
// outputs along the pass, every input sample and tap is read once per thread (the ROWS taps in
// flight slide through registers), so loads per output drop by ~ROWS.

import { separableChannels } from "./problem.js";

export const MAX_CHANNELS = 16;
const ROWS = 8;
const PAD = ROWS - 1; // zero taps around each table so the sliding window needs no bounds checks
const LEVELS_PER_OCTAVE = 6;
const RADIUS_MIN = 1; // smallest blurring level, in pixels; smaller radii map to the identity level
const QUERIES_PER_CHANNEL = 4;

const J = Array.from({ length: ROWS }, (_, j) => j);
const each = (f) => J.map(f).join("\n    ");

function computeShader(packed) {
  // One channel's complex intermediate per pixel, lanes (r, g, b, w):
  //   f32:    two vec4f (re, im)                      32 bytes per pixel
  //   packed: vec4u, lane = pack2x16float(re, im)     16 bytes per pixel
  const midDecl = packed ? "array<vec4u>" : "array<vec4f>";
  const midAccess = packed
    ? /* wgsl */ `
fn storeMid(p: u32, re: vec4f, im: vec4f) {
  mid[p] = vec4u(pack2x16float(vec2f(re.x, im.x)), pack2x16float(vec2f(re.y, im.y)),
                 pack2x16float(vec2f(re.z, im.z)), pack2x16float(vec2f(re.w, im.w)));
}
fn loadMid(p: u32) -> Complex4 {
  let q = mid[p];
  let a = unpack2x16float(q.x);
  let b = unpack2x16float(q.y);
  let c = unpack2x16float(q.z);
  let d = unpack2x16float(q.w);
  return Complex4(vec4f(a.x, b.x, c.x, d.x), vec4f(a.y, b.y, c.y, d.y));
}`
    : /* wgsl */ `
fn storeMid(p: u32, re: vec4f, im: vec4f) {
  mid[2u * p] = re;
  mid[2u * p + 1u] = im;
}
fn loadMid(p: u32) -> Complex4 {
  return Complex4(mid[2u * p], mid[2u * p + 1u]);
}`;
  return /* wgsl */ `
const ROWS = ${ROWS};
const PAD = ${PAD}u;

struct Params {
  width: u32,
  height: u32,
  halfTaps: i32,          // uniform radius: K, taps run over -K..K
  channel: u32,           // channel processed by this dispatch
  channels: u32,
  stride: u32,            // length of one tap table: max taps + 2 PAD
  levels: u32,            // number of radius levels (1 for uniform radius)
  first: u32,             // 1 for channel 0: overwrite the accumulator
  radiusMin: f32,         // radius of level 1 (level 0 is the identity), in pixels
  levelsPerOctave: f32,
}

struct Complex4 { re: vec4f, im: vec4f }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;          // linear RGB
@group(0) @binding(2) var<storage, read> kern: array<vec2f>;         // [level][h|v][channel][stride]
@group(0) @binding(3) var<storage, read> levelHalf: array<i32>;      // K of each level
@group(0) @binding(4) var<storage, read> radius: array<f32>;         // per-pixel radius (varying)
@group(0) @binding(5) var<storage, read_write> mid: ${midDecl};
@group(0) @binding(6) var<storage, read_write> acc: array<vec4f>;    // rgb + kernel weight sum
${midAccess}

// Index of tap 0 of a table: dir 0 = horizontal, 1 = vertical.
fn tableStart(level: u32, dir: u32) -> u32 {
  return ((level * 2u + dir) * P.channels + P.channel) * P.stride + PAD;
}

fn accumulate(i: u32, value: vec4f) {
  acc[i] = select(acc[i], vec4f(0.0), P.first == 1u) + value;
}

fn pixel(x: i32, y: u32) -> vec4f {
  return vec4f(src[y * P.width + u32(clamp(x, 0, i32(P.width) - 1))].rgb, 1.0); // clamp to edge
}

// ---- uniform radius: register-blocked, ROWS outputs per thread -------------------------------

@compute @workgroup_size(64, 1)
fn horizontal(@builtin(global_invocation_id) id: vec3u) {
  let x0 = i32(id.x) * ROWS;
  if (x0 >= i32(P.width) || id.y >= P.height) { return; }
  var k = tableStart(0u, 0u);
  ${each((j) => `var re${j} = vec4f(0.0); var im${j} = vec4f(0.0); var h${j} = vec2f(0.0);`)}
  // Input x0 - K + i feeds output x0 + j through tap i - j: h_j = kern[k - j], zero-padded.
  for (var u = x0 - P.halfTaps; u < x0 + ROWS + P.halfTaps; u++) {
    ${J.slice(1).reverse().map((j) => `h${j} = h${j - 1};`).join(" ")}
    h0 = kern[k];
    k++;
    let p = pixel(u, id.y);
    ${each((j) => `re${j} += p * h${j}.x; im${j} += p * h${j}.y;`)}
  }
  let row = id.y * P.width;
  ${each((j) => `if (x0 + ${j} < i32(P.width)) { storeMid(row + u32(x0 + ${j}), re${j}, im${j}); }`)}
}

@compute @workgroup_size(64, 1)
fn vertical(@builtin(global_invocation_id) id: vec3u) {
  let y0 = i32(id.y) * ROWS;
  if (id.x >= P.width || y0 >= i32(P.height)) { return; }
  var k = tableStart(0u, 1u);
  ${each((j) => `var a${j} = vec4f(0.0); var h${j} = vec2f(0.0);`)}
  for (var s = y0 - P.halfTaps; s < y0 + ROWS + P.halfTaps; s++) {
    ${J.slice(1).reverse().map((j) => `h${j} = h${j - 1};`).join(" ")}
    h0 = kern[k];
    k++;
    let m = loadMid(u32(clamp(s, 0, i32(P.height) - 1)) * P.width + id.x);
    ${each((j) => `a${j} += m.re * h${j}.x - m.im * h${j}.y;`)}
  }
  ${each((j) => `if (y0 + ${j} < i32(P.height)) { accumulate(u32(y0 + ${j}) * P.width + id.x, a${j}); }`)}
}

// ---- spatially varying radius: per-pixel gather -----------------------------------------------

fn levelAt(i: u32) -> u32 {
  let r = radius[i];
  if (r < P.radiusMin * exp2(-0.5 / P.levelsPerOctave)) { return 0u; }
  return min(u32(round(log2(r / P.radiusMin) * P.levelsPerOctave)) + 1u, P.levels - 1u);
}

@compute @workgroup_size(64, 1)
fn horizontalVarying(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.width || id.y >= P.height) { return; }
  let i = id.y * P.width + id.x;
  let level = levelAt(i);
  let K = levelHalf[level];
  let k = tableStart(level, 0u);
  var re = vec4f(0.0);
  var im = vec4f(0.0);
  for (var t = 0; t <= 2 * K; t++) {
    let p = pixel(i32(id.x) + t - K, id.y);
    let h = kern[k + u32(t)];
    re += p * h.x;
    im += p * h.y;
  }
  storeMid(i, re, im);
}

@compute @workgroup_size(64, 1)
fn verticalVarying(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.width || id.y >= P.height) { return; }
  let i = id.y * P.width + id.x;
  let level = levelAt(i);
  let K = levelHalf[level];
  let k = tableStart(level, 1u);
  var a = vec4f(0.0);
  for (var t = 0; t <= 2 * K; t++) {
    let m = loadMid(u32(clamp(i32(id.y) + t - K, 0, i32(P.height) - 1)) * P.width + id.x);
    let h = kern[k + u32(t)];
    a += m.re * h.x - m.im * h.y;
  }
  accumulate(i, a);
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
  let b = blurred[i];
  var c = select(b.rgb / b.w, source[i].rgb, V.showSource == 1u);
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
    const supportsTimestamps = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: supportsTimestamps ? ["timestamp-query"] : [],
    });
    return new BlurRenderer(canvas, device, supportsTimestamps);
  }

  constructor(canvas, device, supportsTimestamps) {
    this.canvas = canvas;
    this.device = device;
    if (supportsTimestamps) {
      const count = QUERIES_PER_CHANNEL * MAX_CHANNELS;
      this.querySet = device.createQuerySet({ type: "timestamp", count });
      this.queryResolve = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.queryRead = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    // One 256-byte-aligned Params block per channel, selected with a dynamic offset.
    this.params = device.createBuffer({ size: 256 * MAX_CHANNELS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.view = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noRadius = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    this.pipelines = {};
    const module = device.createShaderModule({ code: displayShader });
    this.displayPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
    });
    const storage = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 48 } },
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("storage"),
        storage("storage"),
      ].map((e, binding) => ({ binding, ...e })),
    });
    this.radiusMap = null;
  }

  computePipelines(packed) {
    const key = packed ? "packed" : "f32";
    if (!this.pipelines[key]) {
      const module = this.device.createShaderModule({ code: computeShader(packed) });
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
      const make = (entryPoint) => this.device.createComputePipeline({ layout, compute: { module, entryPoint } });
      this.pipelines[key] = {
        horizontal: make("horizontal"),
        vertical: make("vertical"),
        horizontalVarying: make("horizontalVarying"),
        verticalVarying: make("verticalVarying"),
      };
    }
    return this.pipelines[key];
  }

  setImage(rgba, width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    for (const b of [this.source, this.acc, this.mid, this.radiusBuffer]) b?.destroy();
    this.source = this.device.createBuffer({ size: rgba.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.source, 0, rgba);
    this.acc = this.device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.mid = null;
    this.radiusBuffer = null;
    this.radiusMap = null;
    this.tablesDirty = true;
  }

  // Per-pixel blur radius in pixels (Float32Array, width × height), or null for a uniform radius.
  setRadiusMap(map) {
    this.radiusBuffer?.destroy();
    this.radiusBuffer = null;
    this.radiusMap = map;
    if (map) {
      this.radiusBuffer = this.device.createBuffer({ size: map.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.radiusBuffer, 0, map);
      let max = RADIUS_MIN;
      for (const r of map) if (r > max) max = r;
      this.maxMapRadius = max;
    }
    this.tablesDirty = true;
  }

  setKernel(params, radius, transition) {
    const channels = separableChannels(params);
    if (channels.length > MAX_CHANNELS) throw new Error(`At most ${MAX_CHANNELS} separable channels.`);
    this.kernelSpec = { channels, radius, transition };
    this.tablesDirty = true;
  }

  // Tap tables, computed in double precision: one level at the exact radius, or an identity
  // level followed by a geometric ladder of radii from RADIUS_MIN up to the map's maximum.
  buildTables() {
    const { channels, radius, transition } = this.kernelSpec;
    const radii = [];
    if (this.radiusMap) {
      radii.push(null);
      const levels = Math.max(1, Math.ceil(Math.log2(this.maxMapRadius / RADIUS_MIN) * LEVELS_PER_OCTAVE) + 1);
      for (let l = 0; l < levels; l++) radii.push(RADIUS_MIN * Math.pow(2, l / LEVELS_PER_OCTAVE));
    } else {
      radii.push(radius);
    }
    const halves = radii.map((r) => (r === null ? 0 : Math.max(1, Math.ceil(r * (1 + transition)))));
    const stride = 2 * Math.max(...halves) + 1 + 2 * PAD;
    const n = channels.length;
    const data = new Float32Array(radii.length * 2 * n * stride * 2);
    radii.forEach((R, level) => {
      const K = halves[level];
      if (R === null) {
        // Identity: a unit center tap in channel 0, both directions; Re[1 · 1] = 1.
        data[((level * 2 + 0) * n * stride + PAD) * 2] = 1;
        data[((level * 2 + 1) * n * stride + PAD) * 2] = 1;
        return;
      }
      channels.forEach(({ decay, phase, h0, h2, v0, v2 }, ch) => {
        [[h0, h2], [v0, v2]].forEach(([[k0r, k0i], [k2r, k2i]], dir) => {
          const start = ((level * 2 + dir) * n + ch) * stride + PAD;
          for (let t = 0; t <= 2 * K; t++) {
            // (k0 + k2 s) e^{(-b + i c) s} at s = ((t - K) / R)²
            const s = ((t - K) / R) ** 2;
            const e = Math.exp(-decay * s);
            const er = e * Math.cos(phase * s);
            const ei = e * Math.sin(phase * s);
            const wr = k0r + k2r * s;
            const wi = k0i + k2i * s;
            data[(start + t) * 2] = wr * er - wi * ei;
            data[(start + t) * 2 + 1] = wr * ei + wi * er;
          }
        });
      });
    });
    for (const b of [this.kernel, this.levelHalf]) b?.destroy();
    this.kernel = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.kernel, 0, data);
    this.levelHalf = this.device.createBuffer({ size: halves.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.levelHalf, 0, Int32Array.from(halves));
    this.channels = n;
    this.K = halves[0];
    this.taps = 2 * Math.max(...halves) + 1;
    this.levels = radii.length;
    this.stride = stride;
    this.tablesDirty = false;
  }

  // Bytes of GPU memory held for the blur, by buffer.
  memory(packed) {
    const px = this.width * this.height;
    return {
      source: px * 16,
      accumulator: px * 16,
      intermediate: px * (packed ? 16 : 32),
      radiusMap: this.radiusMap ? px * 4 : 0,
      tables: this.kernel?.size ?? 0,
    };
  }

  // Runs the passes and draws; resolves with timings once the GPU has finished.
  async render({ exposure = 1, tonemap = 1, precision = "f32", showSource = false } = {}) {
    if (this.tablesDirty) this.buildTables();
    const packed = precision === "f16";
    const { width, height, device } = this;
    const midBytes = width * height * (packed ? 16 : 32);
    if (!this.mid || this.mid.size !== midBytes) {
      this.mid?.destroy();
      this.mid = device.createBuffer({ size: midBytes, usage: GPUBufferUsage.STORAGE });
    }
    const varying = !!this.radiusMap;
    const block = new ArrayBuffer(256 * this.channels);
    for (let ch = 0; ch < this.channels; ch++) {
      const o = 256 * ch;
      new Uint32Array(block, o, 2).set([width, height]);
      new Int32Array(block, o + 8, 1)[0] = this.K;
      new Uint32Array(block, o + 12, 5).set([ch, this.channels, this.stride, this.levels, ch === 0 ? 1 : 0]);
      new Float32Array(block, o + 32, 2).set([RADIUS_MIN, LEVELS_PER_OCTAVE]);
    }
    device.queue.writeBuffer(this.params, 0, block);
    const v = new ArrayBuffer(32);
    new Uint32Array(v, 0, 2).set([width, height]);
    new Float32Array(v, 8, 1)[0] = exposure;
    new Uint32Array(v, 12, 2).set([tonemap, showSource ? 1 : 0]);
    device.queue.writeBuffer(this.view, 0, v);

    const p = this.computePipelines(packed);
    const [horizontal, vertical] = varying ? [p.horizontalVarying, p.verticalVarying] : [p.horizontal, p.vertical];
    const groups = varying
      ? { h: [Math.ceil(width / 64), height], v: [Math.ceil(width / 64), height] }
      : { h: [Math.ceil(Math.ceil(width / ROWS) / 64), height], v: [Math.ceil(width / 64), Math.ceil(height / ROWS)] };
    const bindGroup = device.createBindGroup({
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.params, size: 48 } },
        ...[this.source, this.kernel, this.levelHalf, this.radiusBuffer ?? this.noRadius, this.mid, this.acc].map((buffer, k) => ({
          binding: k + 1,
          resource: { buffer },
        })),
      ],
    });
    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    const timed = this.querySet && this.queryRead.mapState === "unmapped";
    let query = 0;
    const pass = (pipeline, offset, workgroups) => {
      const writes = timed
        ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: query++, endOfPassWriteIndex: query++ } }
        : {};
      const cp = encoder.beginComputePass(writes);
      cp.setPipeline(pipeline);
      cp.setBindGroup(0, bindGroup, [offset]);
      cp.dispatchWorkgroups(...workgroups);
      cp.end();
    };
    for (let ch = 0; ch < this.channels; ch++) {
      pass(horizontal, 256 * ch, groups.h);
      pass(vertical, 256 * ch, groups.v);
    }
    const render = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    render.setPipeline(this.displayPipeline);
    render.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [this.view, this.acc, this.source].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    render.draw(3);
    render.end();
    if (timed) {
      encoder.resolveQuerySet(this.querySet, 0, query, this.queryResolve, 0);
      encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, query * 8);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const result = { ms: performance.now() - t0, taps: this.taps, levels: this.levels, varying, packed, memory: this.memory(packed) };
    if (timed) {
      await this.queryRead.mapAsync(GPUMapMode.READ, 0, query * 8);
      const ns = new BigUint64Array(this.queryRead.getMappedRange(0, query * 8).slice(0));
      this.queryRead.unmap();
      result.horizontalMs = 0;
      result.verticalMs = 0;
      for (let k = 0; k < query; k += 4) {
        result.horizontalMs += Number(ns[k + 1] - ns[k]) / 1e6;
        result.verticalMs += Number(ns[k + 3] - ns[k + 2]) / 1e6;
      }
    }
    return result;
  }
}

// Tilt-shift style depth of field: sharp band around `focus` (fraction of height), blur growing
// linearly to `maxRadius` over `ramp` of the height on both sides.
export function tiltShiftRadiusMap(width, height, maxRadius, focus = 0.55, band = 0.06, ramp = 0.3) {
  const map = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const r = maxRadius * Math.min(1, Math.max(0, (Math.abs((y + 0.5) / height - focus) - band) / ramp));
    map.fill(r, y * width, (y + 1) * width);
  }
  return map;
}
