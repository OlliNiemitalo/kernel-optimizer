// Animated "game" scene rendered on the GPU: a night street ray-cast analytically per pixel. Writes
// linear HDR color and a z-buffer, then (for depth of field) autofocuses and turns depth into a
// signed circle of confusion in pixels: negative in front of the focus distance, positive behind.

export const worldShader = /* wgsl */ `
struct World {
  width: u32,
  height: u32,
  time: f32,
  aperture: f32,   // circle-of-confusion radius at infinity, in pixels
  focusX: f32,     // autofocus point, as fractions of the image size
  focusY: f32,
  dof: u32,        // 1: write depth and circle of confusion
}

@group(0) @binding(0) var<uniform> W: World;
@group(0) @binding(1) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> depth: array<f32>;  // view depth, then signed CoC
@group(0) @binding(3) var<storage, read_write> focus: array<f32>;  // [0]: smoothed 1 / focus distance

const FAR = 1e4;
const LAMP_SPACING = 15.0;
const LAMP_LIGHT = vec3f(30.0, 21.0, 11.0);
const BLOCK = 20.0;
const STREET = 8.0;    // facade distance from the street center
const SPEED = 4.0;     // camera speed, m/s

fn hash(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn wrap(x: f32, span: f32) -> f32 {
  return x - span * floor(x / span);
}

struct Camera { pos: vec3f, fwd: vec3f, right: vec3f, up: vec3f }

fn camera(time: f32) -> Camera {
  let pos = vec3f(1.2 * sin(time * 0.31), 1.7 + 0.03 * sin(time * 8.0), time * SPEED);
  let yaw = 0.22 * sin(time * 0.23);
  let pitch = 0.05 + 0.04 * sin(time * 0.17);
  let fwd = vec3f(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  return Camera(pos, fwd, right, cross(fwd, right));
}

struct Hit { t: f32, albedo: vec3f, emission: vec3f, normal: vec3f }

fn sphere(o: vec3f, d: vec3f, c: vec3f, r: f32) -> f32 {
  let oc = o - c;
  let b = dot(oc, d);
  let h = b * b - dot(oc, oc) + r * r;
  if (h < 0.0) { return FAR; }
  let t = -b - sqrt(h);
  return select(FAR, t, t > 0.0);
}

// Vertical cylinder from y = 0 to y = top.
fn pole(o: vec3f, d: vec3f, cx: f32, cz: f32, r: f32, top: f32) -> f32 {
  let ox = o.x - cx;
  let oz = o.z - cz;
  let a = d.x * d.x + d.z * d.z;
  let b = ox * d.x + oz * d.z;
  let disc = b * b - a * (ox * ox + oz * oz - r * r);
  if (disc < 0.0 || a < 1e-8) { return FAR; }
  let t = (-b - sqrt(disc)) / a;
  let y = o.y + t * d.y;
  return select(FAR, t, t > 0.0 && y > 0.0 && y < top);
}

// Axis-aligned box: (t, outward normal).
fn box(o: vec3f, d: vec3f, c: vec3f, h: vec3f) -> vec4f {
  let inv = 1.0 / select(d, vec3f(1e-6), abs(d) < vec3f(1e-6));
  let t0 = (c - h - o) * inv;
  let t1 = (c + h - o) * inv;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tn = max(max(lo.x, lo.y), lo.z);
  let tf = min(min(hi.x, hi.y), hi.z);
  if (tn > tf || tn <= 0.0) { return vec4f(FAR, 0.0, 0.0, 0.0); }
  let n = -sign(d) * step(lo.yzx, lo) * step(lo.zxy, lo);
  return vec4f(tn, n);
}

fn lampPos(k: f32, side: f32) -> vec3f {
  return vec3f(side * 5.2, 5.0, (k + select(0.0, 0.5, side > 0.0)) * LAMP_SPACING);
}

fn trace(o: vec3f, d: vec3f, time: f32) -> Hit {
  var hit = Hit(FAR, vec3f(0.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0));

  // Street, curbs, lane markings, and sidewalks.
  if (d.y < 0.0) {
    let t = -o.y / d.y;
    let p = o + t * d;
    let ax = abs(p.x);
    var albedo = vec3f(0.035);
    if (ax > 5.0) {
      albedo = select(vec3f(0.09), vec3f(0.05), fract(p.z * 0.5) < 0.04 || fract(ax * 0.5) < 0.04);
    } else if (ax > 4.75 && ax < 4.9) {
      albedo = vec3f(0.5);
    } else if (ax < 0.08 && fract(p.z / 6.0) < 0.5) {
      albedo = vec3f(0.55, 0.45, 0.15);
    }
    hit = Hit(t, albedo, vec3f(0.0), vec3f(0.0, 1.0, 0.0));
  }

  // Building facades with lit windows and neon shop signs.
  if (abs(d.x) > 1e-4) {
    let side = sign(d.x);
    let t = (side * STREET - o.x) / d.x;
    let p = o + t * d;
    let block = floor(p.z / BLOCK);
    let s = select(0.0, 1.0, side > 0.0);
    let local = p.z - block * BLOCK;
    if (t > 0.0 && t < hit.t && p.y < 9.0 + 28.0 * hash(vec2f(block, s))) {
      var albedo = mix(vec3f(0.07, 0.06, 0.055), vec3f(0.05, 0.06, 0.08), hash(vec2f(block, s + 7.0)));
      var emission = vec3f(0.0);
      if (local > 16.0) {
        albedo = vec3f(0.01); // alley
      } else {
        let cell = floor(vec2f(local / 2.5, p.y / 3.2));
        let f = fract(vec2f(local / 2.5, p.y / 3.2));
        if (p.y > 3.5 && f.x > 0.2 && f.x < 0.8 && f.y > 0.3 && f.y < 0.8) {
          let r = hash(cell + vec2f(block * 13.0, s * 71.0));
          let warm = hash(cell + vec2f(5.0, block)) < 0.75;
          let on = r < 0.35 && fract(time * 0.05 + r * 37.0) > 0.03;
          albedo = vec3f(0.02);
          emission = select(vec3f(0.0), select(vec3f(0.6, 0.8, 1.2), vec3f(1.4, 0.9, 0.45), warm) * (1.0 + 12.0 * r), on);
        }
        let shop = hash(vec2f(block, s + 3.0));
        if (p.y > 2.6 && p.y < 2.85 && local > 3.0 && local < 13.0 && shop < 0.6) {
          let hue = shop * 10.0;
          emission = (0.5 + 0.5 * cos(hue + vec3f(0.0, 2.1, 4.2))) * 6.0 * (0.8 + 0.2 * sin(time * 9.0 + hue));
        }
      }
      hit = Hit(t, albedo, emission, vec3f(-side, 0.0, 0.0));
    }
  }

  // Street lamps on both sides.
  let k0 = floor(o.z / LAMP_SPACING) - 1.0;
  for (var k = 0; k < 24; k++) {
    for (var s = -1.0; s <= 1.0; s += 2.0) {
      let c = lampPos(k0 + f32(k), s);
      let tl = sphere(o, d, c, 0.22);
      if (tl < hit.t) { hit = Hit(tl, vec3f(0.0), LAMP_LIGHT, vec3f(0.0, 1.0, 0.0)); }
      let tp = pole(o, d, s * 5.7, c.z, 0.07, 5.2);
      if (tp < hit.t) {
        let p = o + tp * d;
        hit = Hit(tp, vec3f(0.15), vec3f(0.0), normalize(vec3f(p.x - s * 5.7, 0.0, p.z - c.z)));
      }
    }
  }

  // Traffic: odd cars oncoming, even cars in the camera's direction.
  for (var i = 0; i < 8; i++) {
    let fi = f32(i);
    let dir = select(1.0, -1.0, (i & 1) == 1);
    let speed = select(7.0, -11.0, (i & 1) == 1);
    let span = 180.0;
    let z = time * SPEED - 20.0 + wrap(hash(vec2f(fi, 3.0)) * span + (speed - SPEED) * time, span);
    let lane = -2.4 * dir;
    let paint = 0.5 + 0.5 * cos(hash(vec2f(fi, 9.0)) * 6.28 + vec3f(0.0, 2.1, 4.2));
    let body = box(o, d, vec3f(lane, 0.7, z), vec3f(0.95, 0.45, 2.1));
    if (body.x < hit.t) { hit = Hit(body.x, paint * 0.4, vec3f(0.0), body.yzw); }
    let cabin = box(o, d, vec3f(lane, 1.4, z - 0.2 * dir), vec3f(0.85, 0.3, 1.1));
    if (cabin.x < hit.t) { hit = Hit(cabin.x, vec3f(0.02), vec3f(0.0), cabin.yzw); }
    for (var s = -1.0; s <= 1.0; s += 2.0) {
      let th = sphere(o, d, vec3f(lane + 0.6 * s, 0.75, z + 2.1 * dir), 0.13);
      if (th < hit.t) { hit = Hit(th, vec3f(0.0), vec3f(60.0, 55.0, 45.0), vec3f(0.0, 0.0, dir)); }
      let tt = sphere(o, d, vec3f(lane + 0.65 * s, 0.8, z - 2.1 * dir), 0.1);
      if (tt < hit.t) { hit = Hit(tt, vec3f(0.0), vec3f(25.0, 1.0, 0.5), vec3f(0.0, 0.0, -dir)); }
    }
  }

  // Glowing orbs drifting past the camera: near-field blur.
  for (var i = 0; i < 7; i++) {
    let fi = f32(i);
    let span = 30.0;
    let zr = 0.8 + wrap(hash(vec2f(fi, 1.0)) * span - time * 1.5, span);
    let c = vec3f(
      (hash(vec2f(fi, 2.0)) * 2.0 - 1.0) * 3.0 + 0.5 * sin(time * 0.7 + fi),
      1.0 + 1.4 * hash(vec2f(fi, 4.0)) + 0.3 * sin(time * 1.3 + fi * 2.0),
      time * SPEED + zr,
    );
    let to = sphere(o, d, c, 0.05);
    if (to < hit.t) {
      hit = Hit(to, vec3f(0.0), (0.55 + 0.45 * cos(fi * 2.4 + vec3f(0.0, 2.1, 4.2))) * 30.0, vec3f(0.0, 1.0, 0.0));
    }
  }
  return hit;
}

fn skyColor(d: vec3f) -> vec3f {
  var sky = mix(vec3f(0.03, 0.025, 0.05), vec3f(0.005, 0.008, 0.025), sqrt(max(d.y, 0.0)));
  let cell = floor(vec2f(atan2(d.z, d.x), asin(clamp(d.y, -1.0, 1.0))) * 500.0);
  if (d.y > 0.02 && hash(cell) < 0.004) { sky += vec3f(1.0, 0.95, 0.85) * (2.0 + 30.0 * hash(cell + 1.7)); }
  let moon = dot(d, normalize(vec3f(-0.4, 0.45, 1.0)));
  sky += select(vec3f(0.0), vec3f(6.0, 5.8, 5.0), moon > 0.99965) + vec3f(0.2, 0.2, 0.25) * pow(max(moon, 0.0), 400.0);
  return sky;
}

@compute @workgroup_size(8, 8)
fn scene(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= W.width || id.y >= W.height) { return; }
  let cam = camera(W.time);
  let tanHalf = 0.45;
  let u = (2.0 * (f32(id.x) + 0.5) / f32(W.width) - 1.0) * tanHalf * f32(W.width) / f32(W.height);
  let v = (1.0 - 2.0 * (f32(id.y) + 0.5) / f32(W.height)) * tanHalf;
  let d = normalize(cam.fwd + u * cam.right + v * cam.up);
  let hit = trace(cam.pos, d, W.time);
  var c: vec3f;
  var z = FAR;
  if (hit.t >= FAR) {
    c = skyColor(d);
  } else {
    let p = cam.pos + hit.t * d;
    var light = vec3f(0.015, 0.018, 0.03);
    let kc = floor(p.z / LAMP_SPACING);
    for (var k = -2.0; k <= 2.0; k += 1.0) {
      for (var s = -1.0; s <= 1.0; s += 2.0) {
        let l = lampPos(kc + k, s) - p;
        let r2 = dot(l, l);
        light += LAMP_LIGHT * 0.6 * max(dot(hit.normal, l * inverseSqrt(r2)), 0.0) / r2;
      }
    }
    c = mix(vec3f(0.02, 0.02, 0.035), hit.albedo * light + hit.emission, exp(-hit.t * 0.012));
    z = hit.t * dot(d, cam.fwd);
  }
  let i = id.y * W.width + id.x;
  color[i] = vec4f(c, 1.0);
  if (W.dof == 1u) { depth[i] = z; }
}

// Mean inverse depth around the focus point, smoothed over frames like a camera's focus pull.
@compute @workgroup_size(1)
fn autofocus() {
  let cx = i32(W.focusX * f32(W.width));
  let cy = i32(W.focusY * f32(W.height));
  var sum = 0.0;
  for (var dy = -3; dy <= 3; dy++) {
    for (var dx = -3; dx <= 3; dx++) {
      let x = clamp(cx + dx * 2, 0, i32(W.width) - 1);
      let y = clamp(cy + dy * 2, 0, i32(W.height) - 1);
      sum += 1.0 / depth[u32(y) * W.width + u32(x)];
    }
  }
  let aim = sum / 49.0;
  focus[0] = select(mix(focus[0], aim, 0.15), aim, focus[0] <= 0.0);
}

@compute @workgroup_size(64)
fn coc(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= W.width * W.height) { return; }
  let z = depth[id.x];
  depth[id.x] = clamp(W.aperture * (1.0 - 1.0 / max(z * focus[0], 1e-6)), -W.aperture, W.aperture);
}
`;

export class World {
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ code: worldShader });
    const make = (entryPoint) => device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
    this.scene = make("scene");
    this.autofocus = make("autofocus");
    this.coc = make("coc");
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.focus = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    this.noDepth = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  }

  // Encodes one frame into `color` (vec4f per pixel) and, if `depth` is given, the signed CoC.
  encode(encoder, { color, depth, width, height, time, aperture, focusX = 0.5, focusY = 0.75, timestampWrites }) {
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 4).set([time, aperture, focusX, focusY]);
    new Uint32Array(u, 24, 1)[0] = depth ? 1 : 0;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const bind = (pipeline) =>
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [this.uniform, color, depth ?? this.noDepth, this.focus]
          .map((buffer, binding) => ({ binding, resource: { buffer } }))
          .filter(({ binding }) => pipeline === this.scene ? binding < 3 : binding !== 1),
      });
    // Separate passes: dispatches within one pass were not reliably ordered on Intel Gen9.
    const steps = [[this.scene, [Math.ceil(width / 8), Math.ceil(height / 8)]]];
    if (depth) steps.push([this.autofocus, [1]], [this.coc, [Math.ceil((width * height) / 64)]]);
    steps.forEach(([pipeline, workgroups], k) => {
      const first = k === 0;
      const last = k === steps.length - 1;
      const writes = timestampWrites && (first || last) && {
        querySet: timestampWrites.querySet,
        ...(first && { beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex }),
        ...(last && { endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex }),
      };
      const pass = encoder.beginComputePass(writes ? { timestampWrites: writes } : {});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind(pipeline));
      pass.dispatchWorkgroups(...workgroups);
      pass.end();
    });
  }
}
