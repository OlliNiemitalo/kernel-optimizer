// Test scene rendered on the GPU: three tetrahedra of different colors in a row against a starry
// black background at infinite distance, seen from a camera orbiting their center (yaw, pitch).
// `scale` multiplies all coordinates, the camera's included: the picture stays the same, but the
// depths, and so the blur, change. Writes linear HDR color with 1 / depth in alpha and, for depth
// of field, the signed circle of confusion as in world.js.

const shapesShader = /* wgsl */ `
struct Shapes {
  width: u32,
  height: u32,
  yaw: f32,
  pitch: f32,
  aperture: f32,
  dof: u32,
  invFocus: f32,
  distance: f32,
  scale: f32,
}

@group(0) @binding(0) var<uniform> S: Shapes;
@group(0) @binding(1) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> coc: array<f32>;

const FAR = 1e4;
const Z_REF = 2.0;
const PIXEL = 0.9; // 2 tan(half vertical field of view)
const SPACING = 1.6;
const R = 0.6; // circumradius of the tetrahedra
const LIGHT = vec3f(0.48, 0.8, -0.36);

struct Camera { pos: vec3f, fwd: vec3f, right: vec3f, up: vec3f }

fn camera() -> Camera {
  let back = vec3f(sin(S.yaw) * cos(S.pitch), sin(S.pitch), -cos(S.yaw) * cos(S.pitch));
  let fwd = -back;
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  return Camera(S.distance * back, fwd, right, cross(fwd, right));
}

// Vertex k of a regular tetrahedron centered at the origin with a vertex up, turned by angle a.
fn vertex(k: u32, a: f32) -> vec3f {
  if (k == 0u) { return vec3f(0.0, R, 0.0); }
  let phi = a + f32(k) * 2.0943951;
  let r = R * 0.94280904; // sqrt(8) / 3
  return vec3f(r * cos(phi), -R / 3.0, r * sin(phi));
}

struct Hit { t: f32, normal: vec3f, albedo: vec3f }

// The face opposite vertex v has outward normal -v / R and lies at distance R / 3 from the center.
fn tetrahedron(o: vec3f, d: vec3f, center: vec3f, a: f32, albedo: vec3f, best: Hit) -> Hit {
  var enter = 0.0;
  var exit = FAR;
  var normal = vec3f(0.0);
  let p = o - center;
  for (var k = 0u; k < 4u; k++) {
    let n = -vertex(k, a) / R;
    let dn = dot(d, n);
    let dist = R / 3.0 - dot(p, n);
    if (abs(dn) < 1e-8) {
      if (dist < 0.0) { return best; }
      continue;
    }
    let t = dist / dn;
    if (dn < 0.0) {
      if (t > enter) { enter = t; normal = n; }
    } else {
      exit = min(exit, t);
    }
  }
  if (enter > exit || enter <= 0.0 || enter >= best.t) { return best; }
  return Hit(enter, normal, albedo);
}

fn hash(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

// At infinity: black, with small stars at random positions in the cells of a cube map (STAR_CELLS
// per face side, a fraction STAR_DENSITY of them lit). Each star is a Gaussian of angular width
// STAR_SIZE widened by the sample footprint foot with its energy kept, so it does not alias.
const STAR_CELLS = 64.0;
const STAR_DENSITY = 0.08;
const STAR_SIZE = 0.0004; // radians
const STAR_ENERGY = 6e-5; // radiance times solid angle; 30 over one pixel

fn background(d: vec3f, foot: f32) -> vec3f {
  let a = abs(d);
  var uv: vec2f;
  var face: f32;
  if (a.x >= a.y && a.x >= a.z) { uv = d.yz / a.x; face = select(0.0, 1.0, d.x < 0.0); }
  else if (a.y >= a.z) { uv = d.xz / a.y; face = select(2.0, 3.0, d.y < 0.0); }
  else { uv = d.xy / a.z; face = select(4.0, 5.0, d.z < 0.0); }
  let g = (0.5 * uv + 0.5) * STAR_CELLS;
  let cell = floor(g) + vec2f(0.0, 97.0 * face);
  if (hash(cell) > STAR_DENSITY) { return vec3f(0.0); }
  let pos = 0.2 + 0.6 * vec2f(hash(cell + 3.1), hash(cell + 7.7));
  // Angle per unit of uv shrinks away from the face center.
  let delta = (fract(g) - pos) * (2.0 / STAR_CELLS) / (1.0 + dot(uv, uv));
  let s2 = STAR_SIZE * STAR_SIZE + foot * foot;
  let energy = STAR_ENERGY * (0.15 + 3.0 * pow(hash(cell + 11.3), 4.0));
  let tint = mix(vec3f(1.0, 0.75, 0.45), vec3f(0.6, 0.75, 1.0), hash(cell + 17.0));
  return tint * energy / (6.2831853 * s2) * exp(-0.5 * dot(delta, delta) / s2);
}

// Rays are traced in the unscaled scene; only the depth is scaled.
fn shade(cam: Camera, pos: vec2f) -> vec4f {
  let u = (pos.x - 0.5 * f32(S.width)) * PIXEL / f32(S.height);
  let v = (0.5 * f32(S.height) - pos.y) * PIXEL / f32(S.height);
  let d = normalize(cam.fwd + u * cam.right + v * cam.up);
  var hit = Hit(FAR, vec3f(0.0), vec3f(0.0));
  hit = tetrahedron(cam.pos, d, vec3f(-SPACING, 0.0, 0.0), 0.3, vec3f(0.8, 0.12, 0.08), hit);
  hit = tetrahedron(cam.pos, d, vec3f(0.0, 0.0, 0.0), 1.1, vec3f(0.15, 0.7, 0.2), hit);
  hit = tetrahedron(cam.pos, d, vec3f(SPACING, 0.0, 0.0), 2.0, vec3f(0.15, 0.3, 0.9), hit);
  if (hit.t >= FAR) { return vec4f(background(d, 0.5 * PIXEL / f32(S.height)), 0.0); }
  let key = max(dot(hit.normal, normalize(LIGHT)), 0.0);
  let fill = 0.12 + 0.08 * hit.normal.y;
  let rim = pow(1.0 - max(dot(hit.normal, -d), 0.0), 4.0);
  return vec4f(hit.albedo * (1.6 * key + fill) + 0.3 * rim, 1.0 / (S.scale * hit.t * dot(d, cam.fwd)));
}

// 4 samples per pixel on a rotated grid; alpha averages 1 / depth, so the circle of confusion
// (linear in it) averages the samples' blur radii.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= S.width || id.y >= S.height) { return; }
  let cam = camera();
  let p = vec2f(f32(id.x) + 0.5, f32(id.y) + 0.5);
  let s = 0.25 * (shade(cam, p + vec2f(0.375, 0.125)) + shade(cam, p + vec2f(-0.125, 0.375)) +
    shade(cam, p + vec2f(-0.375, -0.375)) + shade(cam, p + vec2f(0.125, -0.125)));
  let i = id.y * S.width + id.x;
  color[i] = s;
  if (S.dof == 1u) { coc[i] = clamp(S.aperture * Z_REF * (S.invFocus - s.w), -S.aperture, S.aperture); }
}
`;

export class Shapes {
  constructor(device) {
    this.device = device;
    this.yaw = 0.35;
    this.pitch = 0.2;
    this.distance = 5;
    this.scale = 1;
    const module = device.createShaderModule({ code: shapesShader });
    const storage = { visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } };
    this.layout = device.createBindGroupLayout({
      entries: [{ visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, storage, storage].map((e, binding) => ({ binding, ...e })),
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: "main" },
    });
    this.uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noCoc = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  }

  // Turns the view by a drag of (dx, dy) pixels, as if dragging the scene around its center.
  rotate(dx, dy) {
    this.yaw -= dx * 0.008;
    this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + dy * 0.008));
  }

  // Multiplies all coordinates by `factor`; returns the factor applied after clamping.
  zoom(factor) {
    const scale = Math.max(1 / 16, Math.min(16, this.scale * factor));
    const applied = scale / this.scale;
    this.scale = scale;
    return applied;
  }

  // Same interface as World.encode; `time` and `scratch` are unused.
  encode(encoder, { color, coc, width, height, aperture, invFocus = 0, timestampWrites }) {
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 3).set([this.yaw, this.pitch, aperture]);
    new Uint32Array(u, 20, 1)[0] = coc ? 1 : 0;
    new Float32Array(u, 24, 3).set([invFocus, this.distance, this.scale]);
    this.encodedScale = this.scale;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [this.uniform, color, coc ?? this.noCoc].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
  }
}
