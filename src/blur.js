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
// Spatially varying radius: scatter-as-gather. A prepass quantizes each pixel's radius to the
// nearest of a geometric ladder of radii (tap tables precomputed and normalized per level; level 0
// is an exact identity) into a byte-per-pixel level map. Every tap then uses the level of the
// SOURCE pixel it reads, so each source spreads its own disc: a blurry highlight next to sharper
// pixels keeps its full disc. Pixels gather over a per-tile reach (the largest disc that can touch
// the tile). The vertical pass uses a per-pixel vertical level: the average level of the row
// sources that reached the pixel in the horizontal pass, weighted by brightness and disc coverage.
// With a signed radius (negative = in front of the focus, as from a z-buffer), an in-focus or
// background destination hides sources more than an octave of blur behind it, beyond its own blur.
//
// Depth layers (z-buffer radius): sources are split into up to MAX_LAYERS layers by depth, with
// splits chosen per tile where a blurrier group lies in front of a sharper one, and memberships
// interpolated between tiles. The layers run through the same passes one after another, nearest
// first, and are composited front to back; the last one's holes are filled by pull-push.
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
const TILE = 16; // pixels per side of the tiles used to bound the gather reach
export const MAX_LAYERS = 4;
const HALVES = 128; // kern entries before the tap tables, holding each level's K (levels <= 127)
const HIST = 128; // signed-level histogram bins per tile, for choosing the layer splits
const MAX_SPLITS = MAX_LAYERS - 1;
const FILL_LEVELS = 16; // hole-filling pyramid levels, enough for 65536 pixels
const FILL_BASE = MAX_LAYERS * MAX_CHANNELS; // Params blocks: [layer][channel], then one per pyramid level
const BLOCKS = FILL_BASE + FILL_LEVELS;
const MAX_QUERIES = 2 * (3 + MAX_LAYERS * (5 + 2 * MAX_CHANNELS) + 2 * FILL_LEVELS);

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

const TILE = ${TILE}u;
const HALVES = ${HALVES}u;
const HIST = ${HIST}u;
const MAX_SPLITS = ${MAX_SPLITS}u;
const MIN_SEPARATION = ${LEVELS_PER_OCTAVE}.0; // a nearer layer must be blurrier by an octave, in levels
const SOFT = 1.5; // half-width of the soft step between layers, in levels
const FILL_COVERAGE = 0.02; // last-layer coverage below which hole filling blends in

struct Params {
  width: u32,
  height: u32,
  halfTaps: i32,          // largest K over the levels; tables are centered on tap K
  channel: u32,           // channel processed by this dispatch
  channels: u32,
  stride: u32,            // length of one tap table: 2 halfTaps + 1 + 2 PAD
  levels: u32,            // number of radius levels (1 for uniform radius)
  first: u32,             // 1 for channel 0: overwrite the accumulator
  radiusMin: f32,         // radius of level 1 (level 0 is the identity), in pixels
  levelsPerOctave: f32,
  rowWords: u32,          // u32 words per row of a level map (4 levels per word)
  tilesX: u32,
  tilesY: u32,
  occlusion: i32,         // > 0: hide sources this many signed levels behind the destination
  sizeBlur: u32,          // 1: vertical level = weighted average over the row ("blur the blur size")
  sizeBrightness: f32,    // exponent of the brightness weight in that average
  layered: u32,           // 1: depth layers (layerCount >= 2)
  layer: u32,             // layer processed by this dispatch, 0 = nearest
  layerCount: u32,
  level: u32,             // pyramid level of a hole-filling dispatch
}

struct Complex4 { re: vec4f, im: vec4f }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;          // linear RGB
@group(0) @binding(2) var<storage, read> kern: array<vec2f>;         // K per level in .x, then [level][h|v][channel][stride]
@group(0) @binding(3) var<storage, read> radius: array<f32>;         // signed per-pixel radius (varying)
@group(0) @binding(4) var<storage, read_write> mid: ${midDecl};
@group(0) @binding(5) var<storage, read_write> acc: array<vec4f>;    // rgb + kernel weight sum
@group(0) @binding(6) var<storage, read_write> lv: array<u32>;       // level map, vertical level map, layer weight map
@group(0) @binding(7) var<storage, read_write> tiles: array<atomic<u32>>; // max K, max vertical K, reach x, reach y, histograms, splits
@group(0) @binding(8) var<storage, read_write> layer: array<vec4f>;  // nearer layers composited (rgb, transmittance), then the fill pyramid
${midAccess}

// Index of tap -K of a table (K = halfTaps): dir 0 = horizontal, 1 = vertical.
fn tableStart(level: u32, dir: u32) -> u32 {
  return HALVES + ((level * 2u + dir) * P.channels + P.channel) * P.stride + PAD;
}

fn halfAt(level: u32) -> i32 {
  return i32(kern[level].x);
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

// ---- spatially varying radius: scatter-as-gather ----------------------------------------------
// Level map byte: bits 0-6 level, bit 7 set in front of the focus (negative radius).

fn levelOf(r: f32) -> u32 {
  if (r < P.radiusMin * exp2(-0.5 / P.levelsPerOctave)) { return 0u; }
  return min(u32(round(log2(r / P.radiusMin) * P.levelsPerOctave)) + 1u, P.levels - 1u);
}

fn levelByte(map: u32, x: u32, y: u32) -> u32 {
  return (lv[map + y * P.rowWords + (x >> 2u)] >> ((x & 3u) * 8u)) & 0xffu;
}

fn signedLevel(b: u32) -> i32 {
  return select(i32(b & 127u), -i32(b & 127u), (b & 128u) != 0u);
}

// A source well behind an in-focus or background destination is hidden by it, except within the
// destination's own blur (destHalf = its K): sources above the destination's limit are hidden.
// Foreground (blurry, in front of the focus) destinations hide nothing: they cover only part of
// the aperture, so with layers, farther light keeps flowing behind them.
fn occlusionLimit(dest: i32, x: u32, y: u32) -> i32 {
  if (P.occlusion <= 0 || dest < 0) { return 1000; }
  return dest + P.occlusion;
}

fn hidden(source: i32, limit: i32, t: i32, destHalf: i32) -> bool {
  return source > limit && abs(t) > destHalf;
}

fn halfOf(level: i32) -> i32 {
  return halfAt(u32(abs(level)));
}

fn tileOf(x: u32, y: u32) -> u32 {
  return (y / TILE) * P.tilesX + x / TILE;
}

fn verticalMap() -> u32 {
  return P.height * P.rowWords;
}

fn weightMap() -> u32 {
  return 2u * P.height * P.rowWords;
}

// Membership of a pixel in the current layer, 0..255.
fn weightByte(x: u32, y: u32) -> u32 {
  return levelByte(weightMap(), x, y);
}

fn sourceWeight(x: u32, y: u32) -> f32 {
  if (P.layered == 0u) { return 1.0; }
  return f32(weightByte(x, y)) / 255.0;
}

fn tileCount() -> u32 {
  return P.tilesX * P.tilesY;
}

fn histBase() -> u32 {
  return 4u * tileCount();
}

fn splitBase() -> u32 {
  return (4u + HIST) * tileCount();
}

// Prepass 1: level map, and the largest K per tile. One thread per 4 pixels of a row. With
// layers, also a histogram of signed levels per tile.
@compute @workgroup_size(64, 1)
fn levels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.rowWords || id.y >= P.height) { return; }
  var word = 0u;
  var kmax = 0;
  for (var j = 0u; j < 4u; j++) {
    let x = id.x * 4u + j;
    if (x >= P.width) { break; }
    let r = radius[id.y * P.width + x];
    let level = levelOf(abs(r));
    word |= (level | select(0u, 128u, r < 0.0)) << (j * 8u);
    kmax = max(kmax, halfAt(level));
    if (P.layered == 1u) {
      let bin = select(i32(level), -i32(level), r < 0.0) + i32(HIST / 2u);
      atomicAdd(&tiles[histBase() + tileOf(x, id.y) * HIST + u32(clamp(bin, 0, i32(HIST) - 1))], 1u);
    }
  }
  lv[id.y * P.rowWords + id.x] = word;
  atomicMax(&tiles[tileOf(id.x * 4u, id.y)], u32(kmax));
}

// Layers prepass, one thread per tile: up to layerCount - 1 depth splits (in signed levels, near
// to far) from the histogram of the surrounding 5 x 5 tiles. Greedy Otsu: repeatedly split the
// bin range with the largest between-class variance, at the middle of the empty bins around the
// best threshold. A split is only made where it is needed: where the nearer part lies in front of
// the focus and is blurrier than the farther part by MIN_SEPARATION, so that it must spread over
// it (a sharper part in front is handled by occlusion). Unused splits come first, at -infinity, so
// the nearest layers are empty where fewer layers are needed.
@compute @workgroup_size(64, 1)
fn splits(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= tileCount()) { return; }
  let tx = i32(id.x % P.tilesX);
  let ty = i32(id.x / P.tilesX);
  var h: array<f32, ${HIST}>;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let sx = tx + dx;
      let sy = ty + dy;
      if (sx < 0 || sy < 0 || sx >= i32(P.tilesX) || sy >= i32(P.tilesY)) { continue; }
      let base = histBase() + (u32(sy) * P.tilesX + u32(sx)) * HIST;
      for (var b = 0u; b < HIST; b++) { h[b] += f32(atomicLoad(&tiles[base + b])); }
    }
  }
  var lo: array<u32, ${MAX_LAYERS}>;
  var hi: array<u32, ${MAX_LAYERS}>;
  var cuts: array<f32, ${MAX_SPLITS}>;
  lo[0] = 0u;
  hi[0] = HIST - 1u;
  var segments = 1u;
  var found = 0u;
  for (var k = 0u; k + 1u < P.layerCount; k++) {
    var bestGain = 0.0;
    var bestSegment = ${MAX_LAYERS}u;
    var bestCut = 0u;
    for (var s = 0u; s < segments; s++) {
      var total = 0.0;
      var moment = 0.0;
      for (var b = lo[s]; b <= hi[s]; b++) {
        total += h[b];
        moment += h[b] * f32(b);
      }
      var w0 = 0.0;
      var m0 = 0.0;
      for (var b = lo[s]; b < hi[s]; b++) {
        w0 += h[b];
        m0 += h[b] * f32(b);
        let w1 = total - w0;
        if (w0 == 0.0 || w1 == 0.0) { continue; }
        let near = m0 / w0 - f32(HIST / 2u);
        let far = (moment - m0) / w1 - f32(HIST / 2u);
        let gap = far - near;
        let gain = w0 * w1 * gap * gap;
        if (-near - abs(far) >= MIN_SEPARATION && gain > bestGain) {
          bestGain = gain;
          bestSegment = s;
          bestCut = b;
        }
      }
    }
    if (bestSegment == ${MAX_LAYERS}u) { break; }
    var left = bestCut;
    while (left > lo[bestSegment] && h[left] == 0.0) { left--; }
    var right = bestCut + 1u;
    while (right < hi[bestSegment] && h[right] == 0.0) { right++; }
    lo[segments] = bestCut + 1u;
    hi[segments] = hi[bestSegment];
    hi[bestSegment] = bestCut;
    segments++;
    cuts[found] = 0.5 * (f32(left) + f32(right)) - f32(HIST / 2u);
    found++;
  }
  for (var i = 1u; i < found; i++) {
    for (var j = i; j > 0u && cuts[j - 1u] > cuts[j]; j--) {
      let t = cuts[j];
      cuts[j] = cuts[j - 1u];
      cuts[j - 1u] = t;
    }
  }
  let used = P.layerCount - 1u;
  let unused = used - found;
  for (var i = 0u; i < used; i++) {
    var value = -1.0e6;
    if (i >= unused) { value = cuts[i - unused]; }
    atomicStore(&tiles[splitBase() + id.x * MAX_SPLITS + i], bitcast<u32>(value));
  }
}

fn splitOf(tile: u32, i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&tiles[splitBase() + tile * MAX_SPLITS + i]));
}

// Membership of signed level s in the current layer, with one tile's splits: between split
// layer - 1 and split layer (soft steps, -/+ infinity at the ends).
fn tileWeight(s: f32, tile: u32) -> f32 {
  var lower = 1.0;
  if (P.layer > 0u) { lower = smoothstep(-SOFT, SOFT, s - splitOf(tile, P.layer - 1u)); }
  var upper = 0.0;
  if (P.layer + 1u < P.layerCount) { upper = smoothstep(-SOFT, SOFT, s - splitOf(tile, P.layer)); }
  return lower - upper;
}

// Layers prepass per layer: each pixel's membership, bilinear between the four nearest tile
// centers so the splits vary smoothly, into the weight map; and the largest K per tile among the
// pixels in the layer. One thread per 4 pixels of a row.
@compute @workgroup_size(64, 1)
fn layerWeights(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.rowWords || id.y >= P.height) { return; }
  let last = vec2i(i32(P.tilesX) - 1, i32(P.tilesY) - 1);
  var word = 0u;
  var kmax = 0;
  for (var j = 0u; j < 4u; j++) {
    let x = id.x * 4u + j;
    if (x >= P.width) { break; }
    let b = levelByte(0u, x, id.y);
    let s = f32(signedLevel(b));
    let f = (vec2f(f32(x), f32(id.y)) + 0.5) / f32(TILE) - 0.5;
    let i0 = vec2i(floor(f));
    let a = f - floor(f);
    var w = 0.0;
    for (var c = 0; c < 4; c++) {
      let o = vec2i(c & 1, c >> 1);
      let t = clamp(i0 + o, vec2i(0), last);
      let bw = select(1.0 - a, a, o == vec2i(1));
      w += bw.x * bw.y * tileWeight(s, u32(t.y) * P.tilesX + u32(t.x));
    }
    let q = u32(round(clamp(w, 0.0, 1.0) * 255.0));
    word |= q << (j * 8u);
    if (q > 0u) { kmax = max(kmax, halfAt(b & 127u)); }
  }
  lv[weightMap() + id.y * P.rowWords + id.x] = word;
  atomicMax(&tiles[tileOf(id.x * 4u, id.y)], u32(kmax));
}

// After each layer but the last: composite it under the nearer ones, front to back. acc holds the
// layer's color premultiplied by its coverage acc.w.
@compute @workgroup_size(64, 1)
fn composite(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.width * P.height) { return; }
  let a = acc[id.x];
  let c = select(a.rgb, a.rgb / a.w, a.w > 1.0);
  let alpha = clamp(a.w, 0.0, 1.0);
  let before = select(layer[id.x], vec4f(0.0, 0.0, 0.0, 1.0), P.layer == 0u);
  layer[id.x] = vec4f(before.rgb + before.w * c, before.w * (1.0 - alpha));
}

// Hole filling of the last layer by pull-push: where nearer objects removed its sources, its
// coverage is below 1, and it is completed from a pyramid of coverage-weighted averages.
fn levelSize(l: u32) -> vec2u {
  return vec2u((P.width + (1u << l) - 1u) >> l, (P.height + (1u << l) - 1u) >> l);
}

fn levelOffset(l: u32) -> u32 {
  var o = P.width * P.height;
  for (var j = 1u; j < l; j++) {
    let s = levelSize(j);
    o += s.x * s.y;
  }
  return o;
}

// Premultiplied color and confidence (at most 1) of a pyramid texel. Level 0 is acc: its own
// normalized color wherever the layer has some coverage (partial coverage is the same surface
// shared with a nearer layer, or background beside a removed object); filling takes over only as
// the coverage approaches 0.
fn texel(l: u32, x: u32, y: u32) -> vec4f {
  if (l > 0u) { return layer[levelOffset(l) + y * levelSize(l).x + x]; }
  let a = acc[y * P.width + x];
  if (a.w <= 1e-6) { return vec4f(0.0); }
  let confidence = smoothstep(0.0, FILL_COVERAGE, a.w);
  return vec4f(a.rgb / a.w * confidence, confidence);
}

@compute @workgroup_size(8, 8)
fn pull(@builtin(global_invocation_id) id: vec3u) {
  let s = levelSize(P.level);
  if (id.x >= s.x || id.y >= s.y) { return; }
  let below = levelSize(P.level - 1u);
  var sum = vec4f(0.0);
  for (var c = 0u; c < 4u; c++) {
    sum += texel(P.level - 1u, min(2u * id.x + (c & 1u), below.x - 1u), min(2u * id.y + (c >> 1u), below.y - 1u));
  }
  if (sum.w > 1.0) { sum = vec4f(sum.rgb / sum.w, 1.0); }
  layer[levelOffset(P.level) + id.y * s.x + id.x] = sum;
}

fn filledColor(l: u32, x: u32, y: u32) -> vec3f {
  let t = texel(l, x, y);
  return select(vec3f(0.0), t.rgb / t.w, t.w > 0.0);
}

// Coarse to fine: complete each texel's missing coverage with the bilinearly upsampled level above.
@compute @workgroup_size(8, 8)
fn push(@builtin(global_invocation_id) id: vec3u) {
  let s = levelSize(P.level);
  if (id.x >= s.x || id.y >= s.y) { return; }
  let above = levelSize(P.level + 1u);
  let p = (vec2f(f32(id.x), f32(id.y)) + 0.5) * 0.5 - 0.5;
  let i0 = vec2i(floor(p));
  let a = p - floor(p);
  var coarse = vec3f(0.0);
  for (var c = 0; c < 4; c++) {
    let o = vec2i(c & 1, c >> 1);
    let t = clamp(i0 + o, vec2i(0), vec2i(above) - 1);
    let bw = select(1.0 - a, a, o == vec2i(1));
    coarse += bw.x * bw.y * filledColor(P.level + 1u, u32(t.x), u32(t.y));
  }
  let own = texel(P.level, id.x, id.y);
  let filled = vec4f(own.rgb + (1.0 - own.w) * coarse, 1.0);
  if (P.level == 0u) { acc[id.y * P.width + id.x] = filled; }
  else { layer[levelOffset(P.level) + id.y * s.x + id.x] = filled; }
}

// Prepasses 2 and 4: reach of each tile = the largest K among tiles whose discs can touch it,
// along x (from max K) or along y (from max vertical K).
fn reach(tile: u32, alongX: bool) {
  let n = P.tilesX * P.tilesY;
  let tx = i32(tile % P.tilesX);
  let ty = i32(tile / P.tilesX);
  let span = (P.halfTaps + i32(TILE) - 1) / i32(TILE);
  var best = 0u;
  for (var d = -span; d <= span; d++) {
    let sx = select(tx, tx + d, alongX);
    let sy = select(ty + d, ty, alongX);
    if (sx < 0 || sy < 0 || sx >= i32(P.tilesX) || sy >= i32(P.tilesY)) { continue; }
    let k = atomicLoad(&tiles[select(n, 0u, alongX) + u32(sy) * P.tilesX + u32(sx)]);
    if (i32(k) > (abs(d) - 1) * i32(TILE)) { best = max(best, k); }
  }
  atomicStore(&tiles[select(3u, 2u, alongX) * n + tile], best);
}

@compute @workgroup_size(64, 1)
fn reachX(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.tilesX * P.tilesY) { reach(id.x, true); }
}

@compute @workgroup_size(64, 1)
fn reachY(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.tilesX * P.tilesY) { reach(id.x, false); }
}

// Horizontal marginal of a level's unit-sum disc at offset t ("how much of the disc lies in this
// column"); the identity level is a unit impulse.
fn marginal(level: u32, t: i32) -> f32 {
  if (level == 0u) { return select(0.0, 1.0, t == 0); }
  let r = P.radiusMin * exp2(f32(level - 1u) / P.levelsPerOctave);
  return 2.0 * sqrt(max(r * r - f32(t * t), 0.0)) / (3.14159265 * r * r);
}

// Prepass 3: vertical level = average level of the unoccluded row sources reaching the pixel,
// weighted by their brightness and disc marginal, so whichever light dominates the horizontal
// result also decides the vertical kernel (levels are logarithmic in radius).
@compute @workgroup_size(64, 1)
fn verticalLevels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.rowWords || id.y >= P.height) { return; }
  let n = P.tilesX * P.tilesY;
  let r = i32(atomicLoad(&tiles[2u * n + tileOf(id.x * 4u, id.y)]));
  let row = id.y * P.width;
  var word = 0u;
  var kmax = 0;
  for (var j = 0u; j < 4u; j++) {
    let x = id.x * 4u + j;
    if (x >= P.width) { break; }
    let me = levelByte(0u, x, id.y);
    let ld = signedLevel(me);
    let kd = halfOf(ld);
    let limit = occlusionLimit(ld, x, id.y);
    var sum = 0.0;
    var weight = 0.0;
    let span = select(-1, r, P.sizeBlur == 1u); // -1: no sources, keep the own level
    for (var t = -span; t <= span; t++) {
      let xs = u32(clamp(i32(x) + t, 0, i32(P.width) - 1));
      let s = levelByte(0u, xs, id.y);
      let level = s & 127u;
      if (abs(t) > halfAt(level) || hidden(signedLevel(s), limit, t, kd)) { continue; }
      let w = sourceWeight(xs, id.y) * pow(dot(src[row + xs].rgb, vec3f(0.2126, 0.7152, 0.0722)) + 0.01, P.sizeBrightness) * marginal(level, t);
      sum += w * f32(level);
      weight += w;
    }
    let v = select(me & 127u, min(u32(round(sum / weight)), P.levels - 1u), weight > 0.0);
    word |= v << (j * 8u);
    kmax = max(kmax, halfAt(v));
  }
  lv[verticalMap() + id.y * P.rowWords + id.x] = word;
  atomicMax(&tiles[n + tileOf(id.x * 4u, id.y)], u32(kmax));
}

// ROWS outputs per thread, as in the uniform path. A source u reaches output j through tap
// t = u - (x0 + j), read from the source's own level table; the PAD zeros cover |t| <= K + PAD.
@compute @workgroup_size(64, 1)
fn horizontalVarying(@builtin(global_invocation_id) id: vec3u) {
  let x0 = i32(id.x) * ROWS;
  if (x0 >= i32(P.width) || id.y >= P.height) { return; }
  let r = i32(atomicLoad(&tiles[2u * P.tilesX * P.tilesY + tileOf(u32(x0), id.y)]));
  let last = i32(P.width) - 1;
  ${each((j) => `let px${j} = u32(min(x0 + ${j}, last)); let d${j} = signedLevel(levelByte(0u, px${j}, id.y)); let k${j} = halfOf(d${j}); let o${j} = occlusionLimit(d${j}, px${j}, id.y); var re${j} = vec4f(0.0); var im${j} = vec4f(0.0);`)}
  let row = id.y * P.width;
  for (var u = x0 - r; u < x0 + ROWS + r; u++) {
    let x = u32(clamp(u, 0, last));
    let s = levelByte(0u, x, id.y);
    let ls = signedLevel(s);
    let base = tableStart(s & 127u, 0u) + u32(P.halfTaps + u - x0);
    let p = vec4f(src[row + x].rgb, 1.0) * sourceWeight(x, id.y);
    ${each((j) => `{ let h = select(kern[base - ${j}u], vec2f(0.0), hidden(ls, o${j}, u - x0 - ${j}, k${j})); re${j} += p * h.x; im${j} += p * h.y; }`)}
  }
  ${each((j) => `if (x0 + ${j} <= last) { storeMid(row + u32(x0 + ${j}), re${j}, im${j}); }`)}
}

@compute @workgroup_size(64, 1)
fn verticalVarying(@builtin(global_invocation_id) id: vec3u) {
  let y0 = i32(id.y) * ROWS;
  if (id.x >= P.width || y0 >= i32(P.height)) { return; }
  let r = i32(atomicLoad(&tiles[3u * P.tilesX * P.tilesY + tileOf(id.x, u32(y0))]));
  let last = i32(P.height) - 1;
  ${each((j) => `let py${j} = u32(min(y0 + ${j}, last)); let d${j} = signedLevel(levelByte(0u, id.x, py${j})); let k${j} = halfOf(d${j}); let o${j} = occlusionLimit(d${j}, id.x, py${j}); var a${j} = vec4f(0.0);`)}
  for (var u = y0 - r; u < y0 + ROWS + r; u++) {
    let y = u32(clamp(u, 0, last));
    let ls = signedLevel(levelByte(0u, id.x, y));
    let base = tableStart(levelByte(verticalMap(), id.x, y), 1u) + u32(P.halfTaps + u - y0);
    let m = loadMid(y * P.width + id.x);
    ${each((j) => `{ let h = select(kern[base - ${j}u], vec2f(0.0), hidden(ls, o${j}, u - y0 - ${j}, k${j})); a${j} += m.re * h.x - m.im * h.y; }`)}
  }
  ${each((j) => `if (y0 + ${j} <= last) { accumulate(u32(y0 + ${j}) * P.width + id.x, a${j}); }`)}
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
  layered: u32,   // 1: blurred is the hole-filled last layer, under the composited nearer layers
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> blurred: array<vec4f>;
@group(0) @binding(2) var<storage, read> source: array<vec4f>;
@group(0) @binding(3) var<storage, read> nearer: array<vec4f>; // (rgb, transmittance) of the nearer layers

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
  var c = b.rgb / b.w;
  if (V.layered == 1u) {
    let n = nearer[i];
    c = n.rgb + n.w * c;
  }
  if (V.showSource == 1u) { c = source[i].rgb; }
  c = max(c * V.exposure, vec3f(0.0)); // ringing can go slightly negative
  if (V.tonemap == 1u) { c = c / (1.0 + c); }
  return vec4f(toSrgb(min(c, vec3f(1.0))), 1.0);
}
`;

// Standalone WGSL with the host-side protocol, for use outside this app.
export function shaderSource(kind, packed) {
  if (kind === "display") {
    return `// Display pass of the separable circular blur: draws acc.rgb / acc.w as one fullscreen triangle
// (draw(3), no vertex buffers). View (32 bytes): width u32 @0, height u32 @4, exposure f32 @8,
// tonemap u32 @12, showSource u32 @16, layered u32 @20. blurred = the compute passes' acc buffer,
// nearer = their layer buffer (any 16-byte buffer without layers).
${displayShader}`;
  }
  return `// Separable circular blur from complex Gaussian channels: compute passes,
// ${packed ? "packed 16-bit float" : "f32"} intermediate.
//
// Per frame (Params block n at a 256-byte dynamic uniform offset; all blocks equal except channel
// and first):
//   varying radius only, with block 0: clear tiles to 0, then one compute pass each, in order:
//     levels          ceil(rowWords / 64) x height workgroups
//     reachX          ceil(tilesX tilesY / 64) workgroups
//     verticalLevels  ceil(rowWords / 64) x height workgroups
//     reachY          ceil(tilesX tilesY / 64) workgroups
//   for each separable channel n = 0 .. channels - 1:
//     uniform radius:  horizontal  ceil(ceil(width / ${ROWS}) / 64) x height workgroups
//                      vertical    ceil(width / 64) x ceil(height / ${ROWS}) workgroups
//     varying radius:  horizontalVarying, verticalVarying, same workgroup counts
// then draw with the display shader (acc.rgb / acc.w).
//
// Depth layers (z-buffer radius, layerCount L >= 2; Params block (layer, channel) at index
// layer * ${MAX_CHANNELS} + channel, pyramid level l at ${FILL_BASE} + l): clear tiles, then
//     levels (also fills the per-tile histograms), splits (ceil(tilesX tilesY / 64) workgroups),
//     for each layer k = 0 (nearest) .. L - 1, with block (k, 0) for the prepasses:
//       clear tiles words [0, 4 tilesX tilesY), layerWeights (as levels), reachX, verticalLevels,
//       reachY, the channel passes with blocks (k, n), and for k < L - 1: composite
//       (ceil(width height / 64) workgroups);
//     pull for l = 1 .. top, then push for l = top - 1 .. 0 (ceil(size_l / 8)^2 workgroups,
//     size_l = ceil(size / 2^l), top = first level of 1 x 1);
// then draw with layered = 1: layer.rgb + layer.w * acc.rgb.
//
// Params (80 bytes): width u32 @0, height u32 @4, halfTaps i32 @8 (largest K), channel u32 @12,
// channels u32 @16, stride u32 @20, levels u32 @24, first u32 @28 (1 for channel 0, else 0),
// radiusMin f32 @32, levelsPerOctave f32 @36, rowWords u32 @40 (= ceil(width / 4)),
// tilesX u32 @44, tilesY u32 @48 (= ceil(size / ${TILE})), occlusion i32 @52 (threshold in levels for a
// z-buffer radius, ${LEVELS_PER_OCTAVE} per octave; 0 = off), sizeBlur u32 @56 (0: vertical level = own level),
// sizeBrightness f32 @60, layered u32 @64, layer u32 @68, layerCount u32 @72, level u32 @76.
//
// Buffers (one entry per pixel, row-major): src vec4f linear RGB (alpha unused by the blur; the demo
// keeps 1 / depth there for focus picking); acc vec4f;
// mid ${packed ? "16" : "32"} bytes; radius f32, signed: negative in front of the focus (varying only).
// Varying only (else any 4-byte buffers): lv u32[3 height rowWords],
// tiles u32[(4 + ${HIST}) tilesX tilesY + ${MAX_SPLITS} tilesX tilesY]. Layers only (else any 16-byte buffer):
// layer vec4f[width height + pyramid levels 1 .. top].
// kern: vec2f; entries 0 .. ${HALVES - 1} hold K of each level in .x (taps -K..K); then the tap tables
// (re, im), layout [level][dir][channel][stride],
// dir 0 = horizontal, 1 = vertical, stride = 2 halfTaps + 1 + ${2 * PAD}. Entry ${PAD} + halfTaps + t,
// |t| <= K, holds g (k0 + k2 s) exp((-b + i c) s),  s = (t / R)^2,  K = ceil(R (1 + transition
// bandwidth)), with b, c, k0, k2 from the kernel's separable channels (b = -a; k0, k2 = h0, h2
// horizontally and v0, v2 vertically; single components have k2 = 0 and v0 = h0); all other
// entries are 0. g = 1 / sqrt(S) normalizes the level's 2-D kernel: S = sum over channels of
// Re[(sum of h taps)(sum of v taps)].
// Uniform radius: one level, R = disc radius. Varying radius: level 0 is the identity (K = 0,
// tap 1 + 0i in channel 0, both directions), level l >= 1 has R = radiusMin 2^((l - 1) / levelsPerOctave)
// up to the largest radius.
${computeShader(packed)}`;
}

export class BlurRenderer {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const supportsTimestamps = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: supportsTimestamps ? ["timestamp-query"] : [],
    });
    const blur = new BlurRenderer(canvas, device, supportsTimestamps);
    const { vendor, architecture, device: name, description } = adapter.info ?? {};
    blur.adapterInfo = { vendor, architecture, device: name, description };
    return blur;
  }

  constructor(canvas, device, supportsTimestamps) {
    this.canvas = canvas;
    this.device = device;
    if (supportsTimestamps) {
      const count = MAX_QUERIES;
      this.querySet = device.createQuerySet({ type: "timestamp", count });
      this.queryResolve = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.queryRead = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    // One 256-byte-aligned Params block per (layer, channel) and per pyramid level, selected with a dynamic offset.
    this.params = device.createBuffer({ size: 256 * BLOCKS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.view = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.noRadius = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    this.noLevels = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    this.noTiles = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    this.noLayer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
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
        { visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 80 } },
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("read-only-storage"),
        storage("storage"),
        storage("storage"),
        storage("storage"),
        storage("storage"),
        storage("storage"),
      ].map((e, binding) => ({ binding, ...e })),
    });
    this.varying = null; // null, { kind: "map", maxRadius } or { kind: "depth" }
    this.animated = null;
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
        levels: make("levels"),
        verticalLevels: make("verticalLevels"),
        reachX: make("reachX"),
        reachY: make("reachY"),
        splits: make("splits"),
        layerWeights: make("layerWeights"),
        composite: make("composite"),
        pull: make("pull"),
        push: make("push"),
      };
    }
    return this.pipelines[key];
  }

  setImage(rgba, width, height) {
    this.resize(width, height);
    this.device.queue.writeBuffer(this.source, 0, rgba);
  }

  // Renders an animated source (World in world.js, VideoSource in video.js) into the source image
  // every frame; it also supplies the depth for depth of field.
  setAnimated(source, width, height) {
    this.resize(width, height);
    this.animated = source;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    for (const b of [this.source, this.acc, this.mid, this.radiusBuffer, this.levelMaps, this.tiles, this.layerBuffer]) b?.destroy();
    this.source = this.device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.acc = this.device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.rowWords = Math.ceil(width / 4);
    this.tilesX = Math.ceil(width / TILE);
    this.tilesY = Math.ceil(height / TILE);
    this.mid = null;
    this.radiusBuffer = null;
    this.levelMaps = null;
    this.tiles = null;
    this.layerBuffer = null;
    this.varying = null;
    this.animated = null;
    this.tablesDirty = true;
  }

  ensureVaryingBuffers() {
    const { width, height, device } = this;
    this.radiusBuffer ??= device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.levelMaps ??= device.createBuffer({ size: 3 * height * this.rowWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.tiles ??= device.createBuffer({ size: (4 + HIST + MAX_SPLITS) * this.tilesX * this.tilesY * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  }

  // Pyramid levels 1 .. top of the hole filling: sizes and the first 1 x 1 level.
  fillLevels() {
    const sizes = [];
    for (let l = 1; l < FILL_LEVELS; l++) {
      const size = [Math.ceil(this.width / 2 ** l), Math.ceil(this.height / 2 ** l)];
      sizes.push(size);
      if (size[0] === 1 && size[1] === 1) break;
    }
    return sizes;
  }

  ensureLayerBuffer() {
    const texels = this.width * this.height + this.fillLevels().reduce((n, [w, h]) => n + w * h, 0);
    this.layerBuffer ??= this.device.createBuffer({ size: texels * 16, usage: GPUBufferUsage.STORAGE });
  }

  releaseVaryingBuffers() {
    for (const b of [this.radiusBuffer, this.levelMaps, this.tiles, this.layerBuffer]) b?.destroy();
    this.radiusBuffer = this.levelMaps = this.tiles = this.layerBuffer = null;
  }

  // Per-pixel blur radius in pixels (Float32Array, width × height), or null for a uniform radius.
  setRadiusMap(map) {
    if (map) {
      this.ensureVaryingBuffers();
      this.device.queue.writeBuffer(this.radiusBuffer, 0, map);
      let max = RADIUS_MIN;
      for (const r of map) max = Math.max(max, Math.abs(r));
      this.varying = { kind: "map", maxRadius: max };
    } else {
      this.releaseVaryingBuffers();
      this.varying = null;
    }
    this.tablesDirty = true;
  }

  // Radius from the animated source's depth: the disc radius is the blur at 2 m with focus at infinity.
  setDepthOfField() {
    if (!this.animated) throw new Error("Depth of field needs the 3D world or an RGB-D video.");
    this.ensureVaryingBuffers();
    this.varying = { kind: "depth" };
    this.tablesDirty = true;
  }

  setKernel(params, radius, transition) {
    const channels = separableChannels(params);
    if (channels.length > MAX_CHANNELS) throw new Error(`At most ${MAX_CHANNELS} separable channels.`);
    this.kernelSpec = { channels, radius, transition };
    this.tablesDirty = true;
  }

  // Tap tables, computed in double precision: one level at the exact radius, or an identity
  // level followed by a geometric ladder of radii from RADIUS_MIN up to the largest radius.
  // Each level's 2-D kernel is normalized to unit sum, so levels can mix.
  buildTables() {
    const { channels, radius, transition } = this.kernelSpec;
    const radii = [];
    if (this.varying) {
      const maxRadius = this.varying.kind === "map" ? this.varying.maxRadius : Math.max(RADIUS_MIN, radius);
      radii.push(null);
      const levels = Math.min(126, Math.max(1, Math.ceil(Math.log2(maxRadius / RADIUS_MIN) * LEVELS_PER_OCTAVE) + 1));
      for (let l = 0; l < levels; l++) radii.push(RADIUS_MIN * Math.pow(2, l / LEVELS_PER_OCTAVE));
    } else {
      radii.push(radius);
    }
    const halves = radii.map((r) => (r === null ? 0 : Math.max(1, Math.ceil(r * (1 + transition)))));
    const Kmax = Math.max(...halves);
    const stride = 2 * Kmax + 1 + 2 * PAD;
    const n = channels.length;
    const data = new Float32Array((HALVES + radii.length * 2 * n * stride) * 2);
    halves.forEach((K, level) => (data[2 * level] = K));
    radii.forEach((R, level) => {
      const K = halves[level];
      const start = (dir, ch) => HALVES + ((level * 2 + dir) * n + ch) * stride + PAD + Kmax;
      if (R === null) {
        // Identity: a unit center tap in channel 0, both directions; Re[1 · 1] = 1.
        data[start(0, 0) * 2] = 1;
        data[start(1, 0) * 2] = 1;
        return;
      }
      let S = 0;
      channels.forEach(({ decay, phase, h0, h2, v0, v2 }, ch) => {
        const sums = [[h0, h2], [v0, v2]].map(([[k0r, k0i], [k2r, k2i]], dir) => {
          let sr = 0;
          let si = 0;
          for (let t = -K; t <= K; t++) {
            // (k0 + k2 s) e^{(-b + i c) s} at s = (t / R)²
            const s = (t / R) ** 2;
            const e = Math.exp(-decay * s);
            const er = e * Math.cos(phase * s);
            const ei = e * Math.sin(phase * s);
            const wr = k0r + k2r * s;
            const wi = k0i + k2i * s;
            const o = (start(dir, ch) + t) * 2;
            data[o] = wr * er - wi * ei;
            data[o + 1] = wr * ei + wi * er;
            sr += data[o];
            si += data[o + 1];
          }
          return [sr, si];
        });
        S += sums[0][0] * sums[1][0] - sums[0][1] * sums[1][1];
      });
      const g = S > 0 ? 1 / Math.sqrt(S) : 1;
      for (let dir = 0; dir < 2; dir++) {
        for (let ch = 0; ch < n; ch++) {
          for (let t = -K; t <= K; t++) {
            const o = (start(dir, ch) + t) * 2;
            data[o] *= g;
            data[o + 1] *= g;
          }
        }
      }
    });
    this.kernel?.destroy();
    this.kernel = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.kernel, 0, data);
    this.channels = n;
    this.K = Kmax;
    this.taps = 2 * Kmax + 1;
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
      radiusMap: this.radiusBuffer?.size ?? 0,
      levelMaps: (this.levelMaps?.size ?? 0) + (this.tiles?.size ?? 0),
      layers: this.layerBuffer?.size ?? 0,
      tables: this.kernel?.size ?? 0,
    };
  }

  // Runs the passes and draws; resolves with timings once the GPU has finished. `time` and
  // `invFocus` (1 / focus distance in 1/m) drive the animated source. `capture` returns the frame as a
  // PNG data URL. Varying radius: `sizeBlur` is the brightness exponent of the blur-size smoothing
  // (null = off); `occlusion` is the depth-of-field occlusion threshold in octaves (0 = off).
  // Depth of field: `layers` (1 .. MAX_LAYERS) depth layers with locally chosen splits.
  async render({
    exposure = 1, tonemap = 1, precision = "f32", showSource = false, time = 0, invFocus = 0, capture = false,
    sizeBlur = 1, occlusion = 1, layers = 1,
  } = {}) {
    if (this.tablesDirty) this.buildTables();
    const packed = precision === "f16";
    const { width, height, device } = this;
    const midBytes = width * height * (packed ? 16 : 32);
    if (!this.mid || this.mid.size !== midBytes) {
      this.mid?.destroy();
      this.mid = device.createBuffer({ size: midBytes, usage: GPUBufferUsage.STORAGE });
    }
    const varying = !!this.varying;
    const layerCount = this.varying?.kind === "depth" && !showSource ? Math.max(1, Math.min(MAX_LAYERS, Math.round(layers))) : 1;
    const layered = layerCount > 1;
    if (layered) this.ensureLayerBuffer();
    else if (this.layerBuffer) {
      this.layerBuffer.destroy();
      this.layerBuffer = null;
    }
    const fillSizes = layered ? this.fillLevels() : [];
    const block = new ArrayBuffer(256 * BLOCKS);
    const writeBlock = (index, { ch = 0, layer = 0, level = 0 }) => {
      const o = 256 * index;
      new Uint32Array(block, o, 2).set([width, height]);
      new Int32Array(block, o + 8, 1)[0] = this.K;
      new Uint32Array(block, o + 12, 5).set([ch, this.channels, this.stride, this.levels, ch === 0 ? 1 : 0]);
      new Float32Array(block, o + 32, 2).set([RADIUS_MIN, LEVELS_PER_OCTAVE]);
      new Uint32Array(block, o + 40, 3).set([this.rowWords, this.tilesX, this.tilesY]);
      new Int32Array(block, o + 52, 1)[0] =
        this.varying?.kind === "depth" && occlusion > 0 ? Math.max(1, Math.round(occlusion * LEVELS_PER_OCTAVE)) : 0;
      new Uint32Array(block, o + 56, 1)[0] = sizeBlur === null ? 0 : 1;
      new Float32Array(block, o + 60, 1)[0] = sizeBlur ?? 1;
      new Uint32Array(block, o + 64, 4).set([layered ? 1 : 0, layer, layerCount, level]);
    };
    for (let layer = 0; layer < layerCount; layer++) {
      for (let ch = 0; ch < this.channels; ch++) writeBlock(layer * MAX_CHANNELS + ch, { ch, layer });
    }
    for (let level = 0; level <= fillSizes.length; level++) writeBlock(FILL_BASE + level, { level });
    device.queue.writeBuffer(this.params, 0, block);
    const v = new ArrayBuffer(32);
    new Uint32Array(v, 0, 2).set([width, height]);
    new Float32Array(v, 8, 1)[0] = exposure;
    new Uint32Array(v, 12, 3).set([tonemap, showSource ? 1 : 0, layered ? 1 : 0]);
    device.queue.writeBuffer(this.view, 0, v);

    const p = this.computePipelines(packed);
    const [horizontal, vertical] = varying ? [p.horizontalVarying, p.verticalVarying] : [p.horizontal, p.vertical];
    const groups = { h: [Math.ceil(Math.ceil(width / ROWS) / 64), height], v: [Math.ceil(width / 64), Math.ceil(height / ROWS)] };
    const bindGroup = device.createBindGroup({
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.params, size: 80 } },
        ...[
          this.source, this.kernel, this.radiusBuffer ?? this.noRadius, this.mid, this.acc,
          this.levelMaps ?? this.noLevels, this.tiles ?? this.noTiles, (layered && this.layerBuffer) || this.noLayer,
        ].map((buffer, k) => ({ binding: k + 1, resource: { buffer } })),
      ],
    });
    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    const timed = this.querySet && this.queryRead.mapState === "unmapped";
    const labels = [];
    const timestampWrites = (label) => {
      if (!timed) return undefined;
      labels.push(label);
      const k = 2 * (labels.length - 1);
      return { querySet: this.querySet, beginningOfPassWriteIndex: k, endOfPassWriteIndex: k + 1 };
    };
    if (this.animated) {
      this.animated.encode(encoder, {
        color: this.source,
        coc: this.varying?.kind === "depth" ? this.radiusBuffer : null,
        scratch: this.acc, // free until the blur passes overwrite it
        width,
        height,
        time,
        aperture: this.kernelSpec.radius,
        invFocus,
        timestampWrites: timestampWrites("scene"),
      });
    }
    // One pass per dispatch: dispatches within a single pass were not reliably ordered on Intel Gen9.
    const pass = (pipeline, blockIndex, workgroups, label) => {
      const writes = timestampWrites(label);
      const cp = encoder.beginComputePass(writes ? { timestampWrites: writes } : {});
      cp.setPipeline(pipeline);
      cp.setBindGroup(0, bindGroup, [256 * blockIndex]);
      cp.dispatchWorkgroups(...workgroups);
      cp.end();
    };
    const rowGroups = [Math.ceil(this.rowWords / 64), height];
    const tileGroups = [Math.ceil((this.tilesX * this.tilesY) / 64)];
    if (varying && !showSource) {
      encoder.clearBuffer(this.tiles);
      pass(p.levels, 0, rowGroups, "prepare");
      if (layered) pass(p.splits, 0, tileGroups, "layers");
      for (let layer = 0; layer < layerCount; layer++) {
        const first = layer * MAX_CHANNELS;
        if (layered) {
          encoder.clearBuffer(this.tiles, 0, 4 * this.tilesX * this.tilesY * 4);
          pass(p.layerWeights, first, rowGroups, "layers");
        }
        pass(p.reachX, first, tileGroups, "prepare");
        pass(p.verticalLevels, first, rowGroups, "prepare");
        pass(p.reachY, first, tileGroups, "prepare");
        for (let ch = 0; ch < this.channels; ch++) {
          pass(horizontal, first + ch, groups.h, "horizontal");
          pass(vertical, first + ch, groups.v, "vertical");
        }
        if (layered && layer < layerCount - 1) pass(p.composite, first, [Math.ceil((width * height) / 64)], "layers");
      }
      fillSizes.forEach(([w, h], i) => pass(p.pull, FILL_BASE + i + 1, [Math.ceil(w / 8), Math.ceil(h / 8)], "fill"));
      for (let level = fillSizes.length - 1; level >= 0; level--) {
        const [w, h] = level === 0 ? [width, height] : fillSizes[level - 1];
        pass(p.push, FILL_BASE + level, [Math.ceil(w / 8), Math.ceil(h / 8)], "fill");
      }
    } else if (!showSource) {
      for (let ch = 0; ch < this.channels; ch++) {
        pass(horizontal, ch, groups.h, "horizontal");
        pass(vertical, ch, groups.v, "vertical");
      }
    }
    const render = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    render.setPipeline(this.displayPipeline);
    render.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [this.view, this.acc, this.source, (layered && this.layerBuffer) || this.noLayer].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    render.draw(3);
    render.end();
    const queries = 2 * labels.length;
    if (timed) {
      encoder.resolveQuerySet(this.querySet, 0, queries, this.queryResolve, 0);
      encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, queries * 8);
    }
    device.queue.submit([encoder.finish()]);
    // Must be read in the same task: the canvas texture is discarded once presented.
    const png = capture ? this.canvas.toDataURL("image/png") : undefined;
    await device.queue.onSubmittedWorkDone();
    const result = { ms: performance.now() - t0, taps: this.taps, levels: this.levels, varying, packed, memory: this.memory(packed), png };
    if (timed) {
      await this.queryRead.mapAsync(GPUMapMode.READ, 0, queries * 8);
      const ns = new BigUint64Array(this.queryRead.getMappedRange(0, queries * 8).slice(0));
      this.queryRead.unmap();
      result.gpuMs = {};
      labels.forEach((label, k) => {
        result.gpuMs[label] = (result.gpuMs[label] ?? 0) + Number(ns[2 * k + 1] - ns[2 * k]) / 1e6;
      });
    }
    return result;
  }

  // Nearest depth around a pixel of the displayed frame: the largest 1 / depth (1/m) within
  // `radius` pixels of (x, y), from the source alpha that the animated source writes. Null without one.
  async pickInverseDepth(x, y, radius) {
    // Serialized: the readback buffer can be mapped by one pick at a time.
    const run = () => this.pickNow(x, y, radius);
    this.picking = (this.picking ?? Promise.resolve()).then(run, run);
    return this.picking;
  }

  async pickNow(x, y, radius) {
    if (!this.animated) return null;
    const { width, height, device } = this;
    x = Math.min(width - 1, Math.max(0, x));
    y = Math.min(height - 1, Math.max(0, y));
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    const bytes = (y1 - y0 + 1) * width * 16;
    if (!this.pickBuffer || this.pickBuffer.size < bytes) {
      this.pickBuffer?.destroy();
      this.pickBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.source, y0 * width * 16, this.pickBuffer, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await this.pickBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
    const data = new Float32Array(this.pickBuffer.getMappedRange(0, bytes).slice(0));
    this.pickBuffer.unmap();
    let best = null;
    for (let v = y0; v <= y1; v++) {
      for (let u = Math.max(0, x - radius); u <= Math.min(width - 1, x + radius); u++) {
        if ((u - x) ** 2 + (v - y) ** 2 > radius * radius) continue;
        const inv = data[((v - y0) * width + u) * 4 + 3];
        if (best === null || inv > best) best = inv;
      }
    }
    return best;
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
