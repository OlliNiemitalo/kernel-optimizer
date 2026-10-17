// RGB-D video as an animated source for the blur, in one of two layouts:
// - stacked: one video, color on top and depth below as grayscale inverse depth (white = near);
// - pair: a color video and a separate depth video coded with matplotlib's "inferno" colormap
//   (black = far, yellow = near), as Video Depth Anything's visualizations are.
// Depth is relative inverse depth v in [0, 1], used as 1/z = invDepthScale · v per meter: 2 for
// the stacked layout (v = 1 at 0.5 m), 1 for relative depth from an estimator (v = 1 at 1 m).
// Writes linear color (with the same highlight boost as still images) and the signed circle of
// confusion, computed as for the 3D world, at a resolution of at most maxWidth pixels across.

const videoShader = /* wgsl */ `
struct Params {
  width: u32,        // output size
  height: u32,
  aperture: f32,     // circle-of-confusion radius at 2 m with focus at infinity, in pixels
  invFocus: f32,     // 1 / focus distance, 1/m
  boost: f32,        // highlight boost, as for still images
  dof: u32,          // 1: write the circle of confusion
  pair: u32,         // 1: separate inferno-coded depth video; 0: gray depth stacked below color
  invDepthScale: f32, // 1/m at v = 1
}

@group(0) @binding(0) var<uniform> V: Params;
@group(0) @binding(1) var colorFrame: texture_external;
@group(0) @binding(2) var depthFrame: texture_external;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> coc: array<f32>;
@group(0) @binding(6) var<storage, read> inferno: array<f32>; // 32^3 RGB cells -> v

const Z_REF = 2.0;

fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= V.width || id.y >= V.height) { return; }
  let i = id.y * V.width + id.x;
  let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(V.width), f32(V.height));
  let colorUv = select(vec2f(uv.x, 0.5 * uv.y), uv, V.pair == 1u);
  var c = toLinear(clamp(textureSampleBaseClampToEdge(colorFrame, linearSampler, colorUv).rgb, vec3f(0.0), vec3f(1.0)));
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c *= 1.0 + V.boost * pow(min(lum, 1.0), 8.0);
  color[i] = vec4f(c, 1.0);
  if (V.dof == 1u) {
    var v: f32;
    if (V.pair == 1u) {
      let q = min(vec3u(clamp(textureSampleBaseClampToEdge(depthFrame, linearSampler, uv).rgb, vec3f(0.0), vec3f(1.0)) * 32.0), vec3u(31u));
      v = inferno[(q.r * 32u + q.g) * 32u + q.b];
    } else {
      v = textureSampleBaseClampToEdge(depthFrame, linearSampler, vec2f(uv.x, 0.5 + 0.5 * uv.y)).r;
    }
    coc[i] = clamp(V.aperture * Z_REF * (V.invFocus - v * V.invDepthScale), -V.aperture, V.aperture);
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

export class VideoSource {
  // color: an HTMLVideoElement; depth: an inferno-coded depth video in sync with it, or null for
  // the stacked layout.
  constructor(device, color, depth = null, { invDepthScale = depth ? 1 : 2, maxWidth = 960 } = {}) {
    this.device = device;
    this.color = color;
    this.depth = depth;
    this.invDepthScale = invDepthScale;
    const fullWidth = color.videoWidth;
    const fullHeight = depth ? color.videoHeight : Math.floor(color.videoHeight / 2);
    const scale = Math.min(1, maxWidth / fullWidth);
    this.width = Math.round(fullWidth * scale);
    this.height = Math.round(fullHeight * scale);
    this.boost = 8;
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
  }

  encode(encoder, { color, coc, width, height, aperture, invFocus = 0, timestampWrites }) {
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 3).set([aperture, invFocus, this.boost]);
    new Uint32Array(u, 20, 2).set([coc ? 1 : 0, this.depth ? 1 : 0]);
    new Float32Array(u, 28, 1)[0] = this.invDepthScale;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    // External textures are valid only until the end of this task, so they are imported per frame.
    const colorFrame = this.device.importExternalTexture({ source: this.color });
    const depthFrame = this.depth ? this.device.importExternalTexture({ source: this.depth }) : colorFrame;
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
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
