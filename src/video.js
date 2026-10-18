// RGB-D video as an animated source for the blur, in one of three layouts:
// - stacked: one video, color on top and depth below as grayscale inverse depth (white = near);
// - pair: a color video and a separate depth video coded with matplotlib's "inferno" colormap
//   (black = far, yellow = near), as Video Depth Anything's visualizations are;
// - side: one video, color on the left and inferno-coded depth on the right (Video Depth
//   Anything's long-video results).
// Depth is relative inverse depth v in [0, 1], used as 1/z = invDepthScale · v per meter: 2 for
// the stacked layout (v = 1 at 0.5 m), 1 for relative depth from an estimator (v = 1 at 1 m).
// A pair can have a second element of the same depth video one depth frame ahead; the shader then
// interpolates depth in time to the color frame, for depth at a lower frame rate than the color.
// The elements of a pair play independently and drift apart by a frame or more, so each presented
// frame is converted into a small cache, and the color frame is rendered with its own depth frames.
// Writes linear color and the signed circle of
// confusion, computed as for the 3D world, at a resolution of at most maxWidth pixels across.

const decodeShader = /* wgsl */ `
fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn infernoV(rgb: vec3f) -> f32 {
  let q = min(vec3u(clamp(rgb, vec3f(0.0), vec3f(1.0)) * 32.0), vec3u(31u));
  return inferno[(q.r * 32u + q.g) * 32u + q.b];
}

// Colormap colors don't blend into colormap colors, so decode texels first, then interpolate.
// The depth occupies the region [lo, lo + extent) of the frame, in texture coordinates.
fn infernoDepth(frame: texture_external, uv: vec2f, lo: vec2f, extent: vec2f) -> f32 {
  let size = vec2f(textureDimensions(frame));
  let p = (lo + uv * extent) * size - 0.5;
  let first = vec2i(lo * size);
  let last = vec2i((lo + extent) * size) - 1;
  let p0 = vec2i(floor(p));
  let f = p - floor(p);
  let a = clamp(p0, first, last);
  let b = clamp(p0 + 1, first, last);
  let top = mix(infernoV(textureLoad(frame, a).rgb), infernoV(textureLoad(frame, vec2i(b.x, a.y)).rgb), f.x);
  let bottom = mix(infernoV(textureLoad(frame, vec2i(a.x, b.y)).rgb), infernoV(textureLoad(frame, b).rgb), f.x);
  return mix(top, bottom, f.y);
}
`;

const paramsShader = /* wgsl */ `
const Z_REF = 2.0;

struct Params {
  width: u32,        // output size
  height: u32,
  aperture: f32,     // circle-of-confusion radius at 2 m with focus at infinity, in pixels
  invFocus: f32,     // 1 / focus distance, 1/m
  dof: u32,          // 1: write the circle of confusion
  packing: u32,      // 0: stacked, 1: pair, 2: side by side
  invDepthScale: f32, // 1/m at v = 1
  depthMix: f32,     // pair: weight of the following depth frame
}

@group(0) @binding(0) var<uniform> V: Params;
@group(0) @binding(4) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> coc: array<f32>;

fn writeDepth(i: u32, v: f32) {
  color[i].w = v * V.invDepthScale; // 1 / depth, for focus picking
  coc[i] = clamp(V.aperture * Z_REF * (V.invFocus - v * V.invDepthScale), -V.aperture, V.aperture);
}
`;

// Stacked and side-by-side layouts: color and depth come from the same video frame.
const videoShader = decodeShader + paramsShader + /* wgsl */ `
@group(0) @binding(1) var colorFrame: texture_external;
@group(0) @binding(2) var depthFrame: texture_external;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(6) var<storage, read> inferno: array<f32>; // 32^3 RGB cells -> v

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= V.width || id.y >= V.height) { return; }
  let i = id.y * V.width + id.x;
  let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(V.width), f32(V.height));
  let region = vec2f(select(1.0, 0.5, V.packing == 2u), select(1.0, 0.5, V.packing == 0u));
  // Keep bilinear color taps out of the depth half.
  let texel = 0.5 / vec2f(textureDimensions(colorFrame));
  let colorUv = clamp(uv * region, texel, region - texel);
  let c = toLinear(clamp(textureSampleBaseClampToEdge(colorFrame, linearSampler, colorUv).rgb, vec3f(0.0), vec3f(1.0)));
  color[i] = vec4f(c, 0.0);
  if (V.dof == 1u) {
    if (V.packing == 2u) {
      writeDepth(i, infernoDepth(colorFrame, uv, vec2f(0.5, 0.0), vec2f(0.5, 1.0)));
    } else {
      writeDepth(i, textureSampleBaseClampToEdge(depthFrame, linearSampler, vec2f(uv.x, 0.5 + 0.5 * uv.y)).r);
    }
  }
}
`;

// Pair layout: each presented frame of either video is converted at the output size: linear color
// as f16, depth v as f32.
const ingestShader = decodeShader + /* wgsl */ `
struct Size { width: u32, height: u32 }

@group(0) @binding(0) var<uniform> S: Size;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<storage, read> inferno: array<f32>;
@group(0) @binding(4) var<storage, read_write> colorOut: array<vec2u>;
@group(0) @binding(5) var<storage, read_write> depthOut: array<f32>;

fn pixelUv(id: vec3u) -> vec2f {
  return (vec2f(id.xy) + 0.5) / vec2f(f32(S.width), f32(S.height));
}

@compute @workgroup_size(8, 8)
fn ingestColor(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= S.width || id.y >= S.height) { return; }
  let texel = 0.5 / vec2f(textureDimensions(frame));
  let uv = clamp(pixelUv(id), texel, 1.0 - texel);
  let c = toLinear(clamp(textureSampleBaseClampToEdge(frame, linearSampler, uv).rgb, vec3f(0.0), vec3f(1.0)));
  colorOut[id.y * S.width + id.x] = vec2u(pack2x16float(c.rg), pack2x16float(vec2f(c.b, 0.0)));
}

@compute @workgroup_size(8, 8)
fn ingestDepth(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= S.width || id.y >= S.height) { return; }
  depthOut[id.y * S.width + id.x] = infernoDepth(frame, pixelUv(id), vec2f(0.0), vec2f(1.0));
}
`;

// Pair layout: one cached color frame with its depth (interpolated between two depth frames).
const combineShader = paramsShader + /* wgsl */ `
@group(0) @binding(1) var<storage, read> colorIn: array<vec2u>;
@group(0) @binding(2) var<storage, read> depthIn: array<f32>;
@group(0) @binding(3) var<storage, read> depthNextIn: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= V.width || id.y >= V.height) { return; }
  let i = id.y * V.width + id.x;
  let p = colorIn[i];
  color[i] = vec4f(unpack2x16float(p.x), unpack2x16float(p.y).x, 0.0);
  if (V.dof == 1u) {
    var v = depthIn[i];
    if (V.depthMix > 0.0) { v = mix(v, depthNextIn[i], V.depthMix); }
    writeDepth(i, v);
  }
}
`;

// matplotlib's inferno colormap, 256 entries.
const INFERNO_HEX =
  "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4";

// For each cell of a 32^3 RGB grid, the position in [0, 1] of the nearest inferno color.
function infernoInverse() {
  const map = Array.from({ length: 256 }, (_, k) => [0, 2, 4].map((j) => parseInt(INFERNO_HEX.substr(6 * k + j, 2), 16)));
  const out = new Float32Array(32 ** 3);
  for (let r = 0; r < 32; r++) {
    for (let g = 0; g < 32; g++) {
      for (let b = 0; b < 32; b++) {
        const [cr, cg, cb] = [r, g, b].map((q) => (q + 0.5) * 8);
        let best = 0;
        let bestD = Infinity;
        for (let k = 0; k < 256; k++) {
          const d = (map[k][0] - cr) ** 2 + (map[k][1] - cg) ** 2 + (map[k][2] - cb) ** 2;
          if (d < bestD) [best, bestD] = [k, d];
        }
        out[(r * 32 + g) * 32 + b] = best / 255;
      }
    }
  }
  return out;
}

let infernoTable = null;

// Converted frames held per video of a pair: more than the drift syncDepthVideo allows while playing.
const HELD_FRAMES = { color: 6, depth: 8 };

export class VideoSource {
  // color: an HTMLVideoElement; depth: an inferno-coded depth video in sync with it, or null for
  // the stacked or side layout. depthNext: the same depth video one frame ahead (at depthFps), or null.
  // fps: the color frame rate; onFrame: called when a pair has converted a newly presented frame.
  constructor(device, color, depth = null, {
    layout = depth ? "pair" : "stacked", invDepthScale = layout === "stacked" ? 2 : 1, maxWidth = 960,
    depthNext = null, depthFps = 0, fps = 30, onFrame = null,
  } = {}) {
    this.device = device;
    this.color = color;
    this.depth = depth;
    this.depthNext = depthNext;
    this.depthFps = depthFps || fps;
    this.fps = fps;
    this.onFrame = onFrame;
    this.layout = ["stacked", "pair", "side"].indexOf(layout);
    this.invDepthScale = invDepthScale;
    const fullWidth = layout === "side" ? Math.floor(color.videoWidth / 2) : color.videoWidth;
    const fullHeight = layout === "stacked" ? Math.floor(color.videoHeight / 2) : color.videoHeight;
    const scale = Math.min(1, maxWidth / fullWidth);
    this.width = Math.round(fullWidth * scale);
    this.height = Math.round(fullHeight * scale);
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: videoShader }), entryPoint: "main" },
    });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noCoc = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    infernoTable ??= infernoInverse();
    this.inferno = device.createBuffer({ size: infernoTable.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.inferno, 0, infernoTable);
    if (this.layout === 1) this.holdFrames();
  }

  // Converts every frame the pair's videos present, keyed by frame number.
  holdFrames() {
    const { device } = this;
    const ingest = device.createShaderModule({ code: ingestShader });
    this.ingestPipelines = {
      color: device.createComputePipeline({ layout: "auto", compute: { module: ingest, entryPoint: "ingestColor" } }),
      depth: device.createComputePipeline({ layout: "auto", compute: { module: ingest, entryPoint: "ingestDepth" } }),
    };
    this.combinePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: combineShader }), entryPoint: "main" },
    });
    this.size = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.size, 0, new Uint32Array([this.width, this.height]));
    this.frames = { color: [], depth: [] }; // { key, buffer }, the most recently presented last
    for (const [element, kind, fps] of [
      [this.color, "color", this.fps],
      [this.depth, "depth", this.depthFps],
      [this.depthNext, "depth", this.depthFps],
    ]) {
      if (!element) continue;
      this.ingest(element, kind, Math.round(element.currentTime * fps));
      const presented = (now, meta) => {
        if (this.destroyed) return;
        this.ingest(element, kind, Math.round(meta.mediaTime * fps));
        this.onFrame?.();
        element.requestVideoFrameCallback(presented);
      };
      element.requestVideoFrameCallback(presented);
    }
  }

  ingest(element, kind, key) {
    const list = this.frames[kind];
    const held = list.findIndex((f) => f.key === key);
    if (held >= 0) {
      list.push(...list.splice(held, 1));
      return;
    }
    const bytes = this.width * this.height * (kind === "color" ? 8 : 4);
    const frame = list.length >= HELD_FRAMES[kind] ? list.shift()
      : { buffer: this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE }) };
    frame.key = key;
    list.push(frame);
    const pipeline = this.ingestPipelines[kind];
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.size } },
        { binding: 1, resource: this.device.importExternalTexture({ source: element }) },
        ...(kind === "color"
          ? [{ binding: 2, resource: this.sampler }, { binding: 4, resource: { buffer: frame.buffer } }]
          : [{ binding: 3, resource: { buffer: this.inferno } }, { binding: 5, resource: { buffer: frame.buffer } }]),
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // The most recently presented color frame whose depth frames are held, with the weight of the
  // following depth frame; the newest of each if the videos have drifted too far apart.
  heldFrames() {
    const { color, depth } = this.frames;
    const find = (key) => depth.find((f) => f.key === key);
    for (let i = color.length - 1; i >= 0; i--) {
      const t = (color[i].key / this.fps) * this.depthFps;
      const k = Math.floor(t + 1e-3);
      const mix = t - k > 1e-3 ? t - k : 0;
      const first = find(k);
      const next = mix ? find(k + 1) : first;
      if (first && next) return { color: color[i], depth: first, depthNext: next, mix };
    }
    return { color: color.at(-1), depth: depth.at(-1), depthNext: depth.at(-1), mix: 0 };
  }

  // Bytes of GPU memory held for converted frames.
  get heldBytes() {
    return this.frames ? [...this.frames.color, ...this.frames.depth].reduce((sum, f) => sum + f.buffer.size, 0) : 0;
  }

  destroy() {
    this.destroyed = true;
    for (const f of this.frames ? [...this.frames.color, ...this.frames.depth] : []) f.buffer.destroy();
  }

  encode(encoder, { color, coc, width, height, aperture, invFocus = 0, timestampWrites }) {
    if (this.destroyed) return;
    const held = this.frames && this.heldFrames();
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 2).set([aperture, invFocus]);
    new Uint32Array(u, 16, 2).set([coc ? 1 : 0, this.layout]);
    new Float32Array(u, 24, 2).set([this.invDepthScale, held?.mix ?? 0]);
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
    if (held) {
      pass.setPipeline(this.combinePipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.combinePipeline.getBindGroupLayout(0),
        entries: [this.uniform, held.color.buffer, held.depth.buffer, held.depthNext.buffer, color, coc ?? this.noCoc]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      }));
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      return;
    }
    // External textures are valid only until the end of this task, so they are imported per frame.
    const colorFrame = this.device.importExternalTexture({ source: this.color });
    const depthFrame = this.depth ? this.device.importExternalTexture({ source: this.depth }) : colorFrame;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: colorFrame },
          { binding: 2, resource: depthFrame },
          { binding: 3, resource: this.sampler },
          { binding: 4, resource: { buffer: color } },
          { binding: 5, resource: { buffer: coc ?? this.noCoc } },
          { binding: 6, resource: { buffer: this.inferno } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
  }
}
