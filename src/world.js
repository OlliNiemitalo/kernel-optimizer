// Animated "game" scene rendered on the GPU: a night street ray-cast analytically per pixel. Writes
// linear HDR color and, for depth of field, the signed circle of confusion from the z-buffer:
// A Z_REF (1/f - 1/z) pixels for focus distance f (A = blur at Z_REF with focus at infinity),
// capped at ±A; negative in front of the focus.

export const worldShader = /* wgsl */ `
struct World {
  width: u32,
  height: u32,
  time: f32,
  aperture: f32,   // circle-of-confusion radius at Z_REF and the largest radius, in pixels
  dof: u32,        // 1: write the circle of confusion
  invFocus: f32,   // 1 / focus distance, 1/m (0 = infinity)
}

@group(0) @binding(0) var<uniform> W: World;
@group(0) @binding(1) var<storage, read_write> color: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> coc: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<vec4f>;

const FAR = 1e4;
const Z_REF = 2.0;
const LAMP_SPACING = 15.0;
const LAMP_LIGHT = vec3f(30.0, 21.0, 11.0);
const BLOCK = 20.0;
const STREET = 8.0;    // facade distance from the street center
const BACK = 30.0;     // back of the buildings, and the next street's facades
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

// spec: Fresnel reflectance at normal incidence of a glossy coat (0 = matte), gloss: its Phong
// exponent for lamp highlights, mirror: weight of a traced mirror reflection (puddles).
struct Hit { t: f32, albedo: vec3f, emission: vec3f, normal: vec3f, spec: f32, gloss: f32, mirror: f32 }

fn matte(t: f32, albedo: vec3f, emission: vec3f, normal: vec3f) -> Hit {
  return Hit(t, albedo, emission, normal, 0.0, 1.0, 0.0);
}

// Value noise with period 256, so hash inputs stay small however far the camera travels.
fn lattice(i: vec2f) -> f32 {
  return hash(i - 256.0 * floor(i / 256.0));
}

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(lattice(i), lattice(i + vec2f(1.0, 0.0)), u.x), mix(lattice(i + vec2f(0.0, 1.0)), lattice(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// Four octaves; octaves finer than the footprint (in units of p) fade to their mean, like mip levels.
fn fbm(p: vec2f, foot: f32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var freq = 1.0;
  for (var k = 0; k < 4; k++) {
    sum += amp * mix(vnoise(p * freq + f32(k) * 17.0), 0.5, smoothstep(0.25, 0.5, foot * freq));
    amp *= 0.5;
    freq *= 2.0;
  }
  return sum / 0.9375;
}

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
  // One axis even on edges and corners: a two-axis normal has length > 1 and overflows glints.
  var n = vec3f(0.0, 0.0, -sign(d.z));
  if (tn == lo.y) { n = vec3f(0.0, -sign(d.y), 0.0); }
  if (tn == lo.x) { n = vec3f(-sign(d.x), 0.0, 0.0); }
  return vec4f(tn, n);
}

fn lampPos(k: f32, side: f32) -> vec3f {
  return vec3f(side * 5.2, 5.0, (k + select(0.0, 0.5, side > 0.0)) * LAMP_SPACING);
}

// Procedural patterns are box-filtered over the pixel footprint w (the procedural counterpart of
// mip-mapping), so distant and grazing-angle detail averages out instead of aliasing.

// Fraction of [x - w/2, x + w/2] where fract(x) < duty.
fn stripes(x: f32, w: f32, duty: f32) -> f32 {
  let h = 0.5 * max(w, 1e-4);
  let c = fract(x); // keeps the differences below small despite large x
  let a = c - h;
  let b = c + h;
  let ia = floor(a) * duty + min(fract(a), duty);
  let ib = floor(b) * duty + min(fract(b), duty);
  return (ib - ia) / (2.0 * h);
}

// Fraction of [x - w/2, x + w/2] inside [lo, hi].
fn inside(x: f32, w: f32, lo: f32, hi: f32) -> f32 {
  let h = 0.5 * max(w, 1e-4);
  return clamp((min(x + h, hi) - max(x - h, lo)) / (2.0 * h), 0.0, 1.0);
}

// Narrows the ray-parameter interval span to where o + t d lies within [lo, hi] on one axis.
fn slab(o: f32, d: f32, lo: f32, hi: f32, span: vec2f) -> vec2f {
  if (abs(d) < 1e-6) { return select(vec2f(1.0, 0.0), span, o >= lo && o <= hi); }
  let ta = (lo - o) / d;
  let tb = (hi - o) / d;
  return vec2f(max(span.x, min(ta, tb)), min(span.y, max(ta, tb)));
}

// Lamp or pole indices k whose objects can meet the ray within span.
fn lampRange(o: vec3f, d: vec3f, span: vec2f, side: f32) -> vec2f {
  let za = o.z + span.x * d.z;
  let zb = o.z + span.y * d.z;
  let off = select(0.0, 0.5, side > 0.0);
  return vec2f(floor(min(za, zb) / LAMP_SPACING - off - 0.02), ceil(max(za, zb) / LAMP_SPACING - off + 0.02));
}

// A building face at ray parameter t with normal n: wall with lit windows, and neon shop signs on
// the street side. u runs along the face; px is the pixel footprint in radians.
fn building(o: vec3f, d: vec3f, t: f32, n: vec3f, block: f32, s: f32, time: f32, px: f32) -> Hit {
  let p = o + t * d;
  let street = abs(n.x) > 0.5;
  let u = select(abs(p.x) - STREET, p.z - block * BLOCK, street);
  let footU = t * px / max(abs(select(d.z, d.x, street)), 1e-3);
  let footY = t * px;
  let stain = fbm(vec2f(u, p.y) * 0.4 + vec2f(block * 7.3, s * 3.1), max(footU, footY) * 0.4);
  let floors = stripes(p.y / 3.2, footY / 3.2, 0.05); // darker floor slabs
  let wall = mix(vec3f(0.07, 0.06, 0.055), vec3f(0.05, 0.06, 0.08), hash(vec2f(block, s + 7.0))) *
    (0.6 + 0.8 * stain) * (1.0 - 0.4 * floors);
  var albedo = wall;
  var emission = vec3f(0.0);
  var glass = 0.0;
  let cell = floor(vec2f(u / 2.5, p.y / 3.2)) + select(vec2f(50.0, 0.0), vec2f(0.0), street);
  let f = fract(vec2f(u / 2.5, p.y / 3.2));
  if (p.y > 3.5 && f.x > 0.2 && f.x < 0.8 && f.y > 0.3 && f.y < 0.8) {
    let r = hash(cell + vec2f(block * 13.0, s * 71.0));
    let warm = hash(cell + vec2f(5.0, block)) < 0.75;
    albedo = vec3f(0.02);
    emission = select(vec3f(0.0), select(vec3f(0.6, 0.8, 1.2), vec3f(1.4, 0.9, 0.45), warm) * (1.0 + 12.0 * r), r < 0.35);
    glass = 1.0;
  }
  // Beyond about one window per pixel, fade to the windows' average (30% glass, 34% lit).
  let blend = smoothstep(0.3, 1.0, max(footU / 2.5, footY / 3.2)) * inside(p.y, footY, 3.5, 1e9);
  albedo = mix(albedo, mix(wall, vec3f(0.02), 0.3), blend);
  emission = mix(emission, vec3f(0.379, 0.276, 0.201), blend);
  let shop = hash(vec2f(block, s + 3.0));
  if (street && shop < 0.6) {
    let hue = shop * 10.0;
    let neon = (0.5 + 0.5 * cos(hue + vec3f(0.0, 2.1, 4.2))) * 6.0;
    emission += neon * inside(p.y, footY, 2.6, 2.85) * inside(u, footU, 3.0, 13.0);
  }
  return Hit(t, albedo, emission, n, 0.04 * glass * (1.0 - blend), 3000.0, 0.0);
}

// px: pixel footprint in radians, for prefiltering.
fn trace(o: vec3f, d: vec3f, time: f32, px: f32) -> Hit {
  var hit = matte(FAR, vec3f(0.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0));

  // Street, curbs, lane markings, sidewalks, and puddles.
  if (d.y < 0.0) {
    let t = -o.y / d.y;
    let p = o + t * d;
    let ax = abs(p.x);
    let footX = t * px;
    let footZ = t * px / max(-d.y, 1e-3); // stretched along the view direction at grazing angles
    let foot = max(footX, footZ);
    let grain = fbm(p.xz * 2.0, foot * 2.0);
    let tile = 0.85 + 0.3 * mix(lattice(floor(p.xz * 0.5)), 0.5, smoothstep(0.5, 1.0, foot * 0.5));
    let tiles = 1.0 - (1.0 - stripes(p.z * 0.5, footZ * 0.5, 0.04)) * (1.0 - stripes(ax * 0.5, footX * 0.5, 0.04));
    var albedo = vec3f(0.035) * (0.8 + 0.4 * vnoise(p.xz * 0.15 + 17.0));
    albedo = mix(albedo, mix(vec3f(0.09) * tile, vec3f(0.05), tiles), inside(ax, footX, 5.0, 1e9));
    albedo = mix(albedo, vec3f(0.5), inside(ax, footX, 4.75, 4.9));
    albedo = mix(albedo, vec3f(0.55, 0.45, 0.15), inside(p.x, footX, -0.08, 0.08) * stripes(p.z / 6.0, footZ / 6.0, 0.5));
    albedo *= 0.7 + 0.6 * grain;
    // Puddles in the road's low spots, with edges widened to the footprint.
    let low = fbm(p.xz * 0.12 + vec2f(31.0, 7.0), foot * 0.12);
    let edge = 0.02 + foot * 0.1;
    let wet = smoothstep(0.66 - edge, 0.66 + edge, low) * inside(ax, footX, 0.4, 4.5);
    hit = Hit(t, albedo * (1.0 - 0.6 * wet), vec3f(0.0), vec3f(0.0, 1.0, 0.0), 0.0, 1.0, wet);
  }

  // Buildings: one box per block and side, 16 m along the street with a 4 m alley to the next,
  // walls with lit windows on every face, neon shop signs on the street faces.
  for (var side = -1.0; side <= 1.0; side += 2.0) {
    let s = select(0.0, 1.0, side > 0.0);
    let span = slab(o.x, d.x, min(side * STREET, side * BACK), max(side * STREET, side * BACK), vec2f(0.0, hit.t));
    if (span.x > span.y) { continue; }
    let za = o.z + span.x * d.z;
    let zb = o.z + span.y * d.z;
    let first = floor(min(za, zb) / BLOCK);
    let last = floor(max(za, zb) / BLOCK);
    let forward = d.z >= 0.0;
    var block = select(last, first, forward);
    for (var n = 0; n < 48; n++) {
      if (block < first || block > last) { break; }
      let h = 9.0 + 28.0 * hash(vec2f(block, s));
      let b = box(o, d, vec3f(side * 0.5 * (STREET + BACK), 0.5 * h, block * BLOCK + 8.0), vec3f(0.5 * (BACK - STREET), 0.5 * h, 8.0));
      if (b.x < hit.t) {
        hit = building(o, d, b.x, b.yzw, block, s, time, px);
        break;
      }
      block += select(-1.0, 1.0, forward);
    }
    // The next street over, seen through the alleys.
    if (abs(d.x) > 1e-4) {
      let t = (side * BACK - o.x) / d.x;
      if (t > 0.0 && t < hit.t && o.y + t * d.y < 14.0) { hit = matte(t, vec3f(0.012), vec3f(0.0), vec3f(-side, 0.0, 0.0)); }
    }
  }

  // Street lamps and poles: only those near where the ray crosses their row.
  for (var s = -1.0; s <= 1.0; s += 2.0) {
    let lamps = slab(o.y, d.y, 4.78, 5.22, slab(o.x, d.x, s * 5.2 - 0.22, s * 5.2 + 0.22, vec2f(0.0, hit.t)));
    if (lamps.x <= lamps.y) {
      let ks = lampRange(o, d, lamps, s);
      for (var k = ks.x; k <= min(ks.y, ks.x + 32.0); k += 1.0) {
        let tl = sphere(o, d, lampPos(k, s), 0.22);
        if (tl < hit.t) { hit = matte(tl, vec3f(0.0), LAMP_LIGHT, vec3f(0.0, 1.0, 0.0)); }
      }
    }
    let poles = slab(o.y, d.y, 0.0, 5.2, slab(o.x, d.x, s * 5.7 - 0.07, s * 5.7 + 0.07, vec2f(0.0, hit.t)));
    if (poles.x <= poles.y) {
      let ks = lampRange(o, d, poles, s);
      for (var k = ks.x; k <= min(ks.y, ks.x + 32.0); k += 1.0) {
        let cz = lampPos(k, s).z;
        let tp = pole(o, d, s * 5.7, cz, 0.07, 5.2);
        if (tp < hit.t) {
          let p = o + tp * d;
          hit = Hit(tp, vec3f(0.12), vec3f(0.0), normalize(vec3f(p.x - s * 5.7, 0.0, p.z - cz)), 0.08, 300.0, 0.0);
        }
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
    if (body.x < hit.t) { hit = Hit(body.x, paint * 0.4, vec3f(0.0), body.yzw, 0.04, 800.0, 0.0); }
    let cabin = box(o, d, vec3f(lane, 1.4, z - 0.2 * dir), vec3f(0.85, 0.3, 1.1));
    if (cabin.x < hit.t) { hit = Hit(cabin.x, vec3f(0.02), vec3f(0.0), cabin.yzw, 0.05, 3000.0, 0.0); }
    for (var s = -1.0; s <= 1.0; s += 2.0) {
      let th = sphere(o, d, vec3f(lane + 0.6 * s, 0.75, z + 2.1 * dir), 0.13);
      if (th < hit.t) { hit = matte(th, vec3f(0.0), vec3f(60.0, 55.0, 45.0), vec3f(0.0, 0.0, dir)); }
      let tt = sphere(o, d, vec3f(lane + 0.65 * s, 0.8, z - 2.1 * dir), 0.1);
      if (tt < hit.t) { hit = matte(tt, vec3f(0.0), vec3f(25.0, 1.0, 0.5), vec3f(0.0, 0.0, -dir)); }
    }
  }
  return hit;
}

fn skyColor(d: vec3f) -> vec3f {
  var sky = mix(vec3f(0.03, 0.025, 0.05), vec3f(0.005, 0.008, 0.025), sqrt(max(d.y, 0.0)));
  let moon = dot(d, normalize(vec3f(-0.4, 0.45, 1.0)));
  sky += select(vec3f(0.0), vec3f(6.0, 5.8, 5.0), moon > 0.99965) + vec3f(0.2, 0.2, 0.25) * pow(max(moon, 0.0), 400.0);
  return sky;
}

// Light from the nearest street lamps: diffuse, plus Fresnel-weighted normalized Phong highlights
// on glossy coats (small bright glints that show off the blur).
fn direct(hit: Hit, p: vec3f, d: vec3f) -> vec3f {
  var light = vec3f(0.015, 0.018, 0.03);
  var glint = vec3f(0.0);
  let r = reflect(d, hit.normal);
  let kc = floor(p.z / LAMP_SPACING);
  for (var k = -2.0; k <= 2.0; k += 1.0) {
    for (var s = -1.0; s <= 1.0; s += 2.0) {
      let l = lampPos(kc + k, s) - p;
      let r2 = dot(l, l);
      let ld = l * inverseSqrt(r2);
      let e = LAMP_LIGHT * 0.6 * max(dot(hit.normal, ld), 0.0) / r2;
      light += e;
      glint += e * (0.5 * hit.gloss + 1.0) * pow(clamp(dot(r, ld), 0.0, 1.0), hit.gloss);
    }
  }
  let fresnel = hit.spec + (1.0 - hit.spec) * pow(1.0 - clamp(dot(-d, hit.normal), 0.0, 1.0), 5.0);
  return hit.albedo * light + hit.emission + select(vec3f(0.0), fresnel * glint, hit.spec > 0.0);
}

fn fogged(c: vec3f, t: f32) -> vec3f {
  return mix(vec3f(0.02, 0.02, 0.035), c, exp(-t * 0.012));
}

// Linear radiance and inverse view depth through image position pos (pixels).
fn shade(cam: Camera, pos: vec2f, px: f32) -> vec4f {
  let u = (pos.x - 0.5 * f32(W.width)) * PIXEL / f32(W.height);
  let v = (0.5 * f32(W.height) - pos.y) * PIXEL / f32(W.height);
  let d = normalize(cam.fwd + u * cam.right + v * cam.up);
  let hit = trace(cam.pos, d, W.time, px);
  if (hit.t >= FAR) { return vec4f(skyColor(d), 0.0); }
  let p = cam.pos + hit.t * d;
  var c = direct(hit, p, d);
  var invDepth = 1.0 / (hit.t * dot(d, cam.fwd));
  if (hit.mirror > 0.0) {
    // Puddle: a traced mirror reflection with the Fresnel reflectance of water. The reflection is
    // the bright content, and its virtual image lies behind the surface by the reflected distance.
    let rd = reflect(d, hit.normal);
    let o2 = p + 0.001 * hit.normal;
    let h2 = trace(o2, rd, W.time, px);
    var rc = skyColor(rd);
    if (h2.t < FAR) { rc = fogged(direct(h2, o2 + h2.t * rd, rd), h2.t); }
    let f = hit.mirror * (0.02 + 0.98 * pow(1.0 - max(-d.y, 0.0), 5.0));
    c = mix(c, rc, f);
    invDepth = mix(invDepth, 1.0 / ((hit.t + h2.t) * dot(d, cam.fwd)), hit.mirror);
  }
  return vec4f(fogged(c, hit.t), invDepth);
}

const PIXEL = 0.9; // 2 tan(half vertical field of view)

fn center(id: vec3u) -> vec2f {
  return vec2f(f32(id.x) + 0.5, f32(id.y) + 0.5);
}

// Adaptive antialiasing in three passes. 1: one sample per pixel, (radiance, 1 / depth) into color.
@compute @workgroup_size(8, 8)
fn primary(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= W.width || id.y >= W.height) { return; }
  color[id.y * W.width + id.x] = shade(camera(W.time), center(id), PIXEL / f32(W.height));
}

fn differs(a: vec4f, b: vec4f) -> bool {
  let la = dot(a.rgb, vec3f(0.2126, 0.7152, 0.0722));
  let lb = dot(b.rgb, vec3f(0.2126, 0.7152, 0.0722));
  return abs(la - lb) > 0.1 * (la + lb) + 0.005 || abs(a.w - b.w) > 0.05 * max(a.w, b.w) + 1e-4;
}

// 2: where a pixel differs from a 4-neighbor (an edge or a small light), 3 more samples on a
// rotated grid with half the footprint; result into scratch, as neighbors still read color.
@compute @workgroup_size(8, 8)
fn refine(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= W.width || id.y >= W.height) { return; }
  let i = id.y * W.width + id.x;
  let c = color[i];
  let edge =
    (id.x > 0u && differs(c, color[i - 1u])) || (id.x + 1u < W.width && differs(c, color[i + 1u])) ||
    (id.y > 0u && differs(c, color[i - W.width])) || (id.y + 1u < W.height && differs(c, color[i + W.width]));
  var sum = c;
  if (edge) {
    let cam = camera(W.time);
    let px = 0.5 * PIXEL / f32(W.height);
    sum += shade(cam, center(id) + vec2f(0.375, 0.125), px);
    sum += shade(cam, center(id) + vec2f(-0.125, 0.375), px);
    sum += shade(cam, center(id) + vec2f(-0.375, -0.375), px);
    sum *= 0.25;
  }
  scratch[i] = sum;
}

// 3: final color, and the circle of confusion (linear in 1 / depth, so averaging 1 / depth
// averages the samples' blur radii).
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= W.width * W.height) { return; }
  let s = scratch[id.x];
  color[id.x] = s; // alpha: 1 / depth, for focus picking
  if (W.dof == 1u) { coc[id.x] = clamp(W.aperture * Z_REF * (W.invFocus - s.w), -W.aperture, W.aperture); }
}
`;

export class World {
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ code: worldShader });
    const storage = { visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } };
    this.layout = device.createBindGroupLayout({
      entries: [{ visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, storage, storage, storage].map((e, binding) => ({ binding, ...e })),
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const make = (entryPoint) => device.createComputePipeline({ layout, compute: { module, entryPoint } });
    this.pipelines = ["primary", "refine", "resolve"].map(make);
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noCoc = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  }

  // Encodes one frame into `color` (vec4f per pixel) and, if `coc` is given, the circle of
  // confusion. `scratch` (16 bytes per pixel) is overwritten.
  encode(encoder, { color, coc, scratch, width, height, time, aperture, invFocus = 0, timestampWrites }) {
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 2).set([width, height]);
    new Float32Array(u, 8, 2).set([time, aperture]);
    new Uint32Array(u, 16, 1)[0] = coc ? 1 : 0;
    new Float32Array(u, 20, 1)[0] = invFocus;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [this.uniform, color, coc ?? this.noCoc, scratch].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const groups = [
      [Math.ceil(width / 8), Math.ceil(height / 8)],
      [Math.ceil(width / 8), Math.ceil(height / 8)],
      [Math.ceil((width * height) / 64)],
    ];
    // Separate passes: dispatches within one pass were not reliably ordered on Intel Gen9.
    this.pipelines.forEach((pipeline, k) => {
      const writes = timestampWrites && k !== 1 && {
        querySet: timestampWrites.querySet,
        ...(k === 0 ? { beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } : { endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex }),
      };
      const pass = encoder.beginComputePass(writes ? { timestampWrites: writes } : {});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...groups[k]);
      pass.end();
    });
  }
}
