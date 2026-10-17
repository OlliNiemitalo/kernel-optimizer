// RGB-D video as an animated source for the blur: each frame holds the color image on top and its
// depth below, as grayscale inverse depth v = 0.5 m / z (white = 0.5 m, black = infinitely far).
// Writes linear color (with the same highlight boost as still images) and the signed circle of
// confusion, computed as for the 3D world.

const videoShader = /* wgsl */ `
struct Params {
  width: u32,        // of one half
  height: u32,
  aperture: f32,     // circle-of-confusion radius at 2 m with focus at infinity, in pixels
  invFocus: f32,     // 1 / focus distance, 1/m
  boost: f32,        // highlight boost, as for still images
  dof: u32,          // 1: write the circle of confusion
}

@group(0) @binding(0) var<uniform> V: Params;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> coc: array<f32>;

const Z_REF = 2.0;
const INV_DEPTH_WHITE = 2.0; // 1/m at v = 1, i.e. 0.5 m

fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= V.width || id.y >= V.height) { return; }
  let i = id.y * V.width + id.x;
  var c = toLinear(clamp(textureLoad(frame, id.xy).rgb, vec3f(0.0), vec3f(1.0)));
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c *= 1.0 + V.boost * pow(min(lum, 1.0), 8.0);
  color[i] = vec4f(c, 1.0);
  if (V.dof == 1u) {
    let invDepth = textureLoad(frame, vec2u(id.x, id.y + V.height)).r * INV_DEPTH_WHITE;
    coc[i] = clamp(V.aperture * Z_REF * (V.invFocus - invDepth), -V.aperture, V.aperture);
  }
}
`;

export class VideoSource {
  // video: a playing or paused HTMLVideoElement with color above depth, both width × height.
  constructor(device, video) {
    this.device = device;
    this.video = video;
    this.width = video.videoWidth;
    this.height = Math.floor(video.videoHeight / 2);
    this.boost = 8;
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: videoShader }), entryPoint: "main" },
    });
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noCoc = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  }

  encode(encoder, { color, coc, width, height, aperture, invFocus = 0, timestampWrites }) {
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 3).set([aperture, invFocus, this.boost]);
    new Uint32Array(u, 20, 1)[0] = coc ? 1 : 0;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          // Valid only until the end of this task, so it is imported for every frame.
          { binding: 1, resource: this.device.importExternalTexture({ source: this.video }) },
          { binding: 2, resource: { buffer: color } },
          { binding: 3, resource: { buffer: coc ?? this.noCoc } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
  }
}
