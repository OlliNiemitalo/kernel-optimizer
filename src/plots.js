// Canvas 2-D plotting for profiles, errors, convergence, and the 2-D disc.
import {
  PARAMS_PER_COMPONENT, channelSide, componentChannels, componentProfile, multiplicities, profile, separableChannels,
} from "./problem.js";

const COLORS = ["#4cc9f0", "#f72585", "#90be6d", "#f9c74f", "#b5179e", "#43aa8b", "#f3722c", "#577590"];

// Stable, evenly spaced rows: re-scoring unchanged members must not reshuffle the preview.
export function populationPreview(population, dimensions) {
  const size = population.length / dimensions;
  const count = Math.min(24, size);
  return Array.from({ length: count }, (_, i) => {
    const row = Math.floor(i * size / count);
    return Array.from(population.subarray(row * dimensions, (row + 1) * dimensions));
  });
}

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function niceStep(range, target) {
  const raw = range / target;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * pow) return m * pow;
  return 10 * pow;
}

// Draws axes and returns data→pixel mappings.
function frame(ctx, w, h, [x0, x1], [y0, y1], { title, xlabel, logY = false } = {}) {
  const m = { l: 52, r: 12, t: 22, b: 30 };
  const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (w - m.l - m.r);
  const Y = (y) => h - m.b - ((y - y0) / (y1 - y0)) * (h - m.t - m.b);
  ctx.font = "11px system-ui, sans-serif";
  ctx.strokeStyle = "#2a2f3a";
  ctx.fillStyle = "#8b93a7";
  ctx.lineWidth = 1;
  ctx.textAlign = "center";
  const xs = niceStep(x1 - x0, 8);
  for (let x = Math.ceil(x0 / xs) * xs; x <= x1 + 1e-9; x += xs) {
    ctx.beginPath();
    ctx.moveTo(X(x), m.t);
    ctx.lineTo(X(x), h - m.b);
    ctx.stroke();
    ctx.fillText(+x.toFixed(6) + "", X(x), h - m.b + 14);
  }
  ctx.textAlign = "right";
  if (logY) {
    for (let e = Math.ceil(y0); e <= y1; e++) {
      ctx.beginPath();
      ctx.moveTo(m.l, Y(e));
      ctx.lineTo(w - m.r, Y(e));
      ctx.stroke();
      ctx.fillText(`1e${e}`, m.l - 4, Y(e) + 4);
    }
  } else {
    const ys = niceStep(y1 - y0, 6);
    for (let y = Math.ceil(y0 / ys) * ys; y <= y1 + 1e-9; y += ys) {
      ctx.beginPath();
      ctx.moveTo(m.l, Y(y));
      ctx.lineTo(w - m.r, Y(y));
      ctx.stroke();
      ctx.fillText(+y.toPrecision(4) + "", m.l - 4, Y(y) + 4);
    }
  }
  ctx.textAlign = "left";
  ctx.fillStyle = "#c9d1e3";
  if (title) ctx.fillText(title, m.l, 14);
  ctx.textAlign = "right";
  ctx.fillStyle = "#8b93a7";
  if (xlabel) ctx.fillText(xlabel, w - m.r, 14);
  return { X, Y, m };
}

function line(ctx, X, Y, xs, ys, color, width = 1.5, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  xs.forEach((x, k) => (k ? ctx.lineTo(X(x), Y(ys[k])) : ctx.moveTo(X(x), Y(ys[k]))));
  ctx.stroke();
  ctx.restore();
}

function clipToPlot(ctx, w, h, m) {
  ctx.beginPath();
  ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
  ctx.clip();
}

// Radial cross-section of the best kernel, with faint population members and the spec.
export function drawProfile(canvas, { best, label, population = [], transition, ripple, xmax, showComponents }) {
  const { ctx, w, h } = setupCanvas(canvas);
  const n = 600;
  const xs = Array.from({ length: n }, (_, k) => (k / (n - 1)) * xmax);
  const bestY = best ? xs.map((x) => profile(best, x)) : [];
  const nc = best ? best.length / PARAMS_PER_COMPONENT : 0;
  const componentYs = best && showComponents
    ? Array.from({ length: nc }, (_, u) => xs.map((x) => componentProfile(best, u, x)))
    : [];
  // Components can be much larger than the sum, so the y range grows to show them in full.
  const all = bestY.concat(...componentYs);
  const lo = Math.min(-0.15, ...all) * (componentYs.length ? 1.05 : 1);
  const hi = Math.max(1.15, ...all) * (componentYs.length ? 1.05 : 1);
  const { X, Y, m } = frame(ctx, w, h, [0, xmax], [lo, hi], {
    title: `Radial cross-section K(r)${label ? ` — ${label}` : ""}`,
    xlabel: "r (disc radius = 1)",
  });
  ctx.save();
  clipToPlot(ctx, w, h, m);
  ctx.fillStyle = "rgba(255,255,255,0.05)";
  ctx.fillRect(X(1), m.t, X(1 + transition) - X(1), h - m.t - m.b);
  const tol = ripple ?? 0;
  ctx.fillStyle = "rgba(144,190,109,0.15)";
  ctx.fillRect(X(0), Y(1 + tol), X(1) - X(0), Y(1 - tol) - Y(1 + tol));
  ctx.fillRect(X(1 + transition), Y(tol), X(xmax) - X(1 + transition), Y(-tol) - Y(tol));
  line(ctx, X, Y, [0, 1, 1, 1 + transition, 1 + transition, xmax], [1, 1, 0, 0, 0, 0], "#5c677d", 1);
  for (const p of population) line(ctx, X, Y, xs, xs.map((x) => profile(p, x)), "#8b93a7", 1, 0.25);
  componentYs.forEach((ys, u) => line(ctx, X, Y, xs, ys, COLORS[u % COLORS.length], 1, 0.8));
  if (best) line(ctx, X, Y, xs, bestY, "#ffffff", 2);
  ctx.restore();
}

// Absolute error over pass and stop bands.
export function drawError(canvas, { best, transition, ripple, xmax }) {
  const { ctx, w, h } = setupCanvas(canvas);
  if (!best) return;
  const n = 800;
  const xs = [];
  const es = [];
  for (let k = 0; k < n; k++) {
    const x = (k / (n - 1)) * xmax;
    if (x > 1 && x < 1 + transition) continue;
    xs.push(x);
    es.push(profile(best, x) - (x <= 1 ? 1 : 0));
  }
  const lim = Math.max(ripple * 1.3, 1e-6);
  const { X, Y, m } = frame(ctx, w, h, [0, xmax], [-lim, lim], { title: "Error K(r) − target", xlabel: "r" });
  ctx.save();
  clipToPlot(ctx, w, h, m);
  line(ctx, X, Y, [0, xmax], [ripple, ripple], "#f9c74f", 1, 0.7);
  line(ctx, X, Y, [0, xmax], [-ripple, -ripple], "#f9c74f", 1, 0.7);
  const split = xs.findIndex((x) => x > 1);
  line(ctx, X, Y, xs.slice(0, split), es.slice(0, split), "#4cc9f0", 1.5);
  line(ctx, X, Y, xs.slice(split), es.slice(split), "#f72585", 1.5);
  ctx.restore();
}

export function drawConvergence(canvas, history) {
  const { ctx, w, h } = setupCanvas(canvas);
  if (history.length === 0) return;
  const gens = history.map((e) => e.gen);
  const logs = history.flatMap((e) => [e.best, e.mean]).filter((v) => v > 0).map(Math.log10);
  const y0 = Math.floor(Math.min(...logs));
  const y1 = Math.max(Math.ceil(Math.max(...logs)), y0 + 1);
  const g0 = gens[0];
  const g1 = Math.max(gens[gens.length - 1], g0 + 1);
  const { X, Y, m } = frame(ctx, w, h, [g0, g1], [y0, y1], {
    title: "Convergence: best (white) and mean (grey) cost",
    xlabel: "generation",
    logY: true,
  });
  ctx.save();
  clipToPlot(ctx, w, h, m);
  line(ctx, X, Y, gens, history.map((e) => Math.log10(e.mean)), "#8b93a7", 1.5);
  line(ctx, X, Y, gens, history.map((e) => Math.log10(e.best)), "#ffffff", 2);
  ctx.restore();
}

// 2-D kernel K(√(x²+y²)) as light, normalized to its peak: positive values gray, negative values
// red, both with brightness |K| / max |K| sRGB-encoded.
export function drawDisc(canvas, best, xmax) {
  const size = canvas.width;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  const srgb = (v) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  if (best) {
    const lutN = 2048;
    const rmax = xmax * Math.SQRT2;
    const lut = Float64Array.from({ length: lutN }, (_, k) => profile(best, (k / (lutN - 1)) * rmax));
    const peak = lut.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-12);
    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const x = ((px + 0.5) / size - 0.5) * 2 * xmax;
        const y = ((py + 0.5) / size - 0.5) * 2 * xmax;
        const v = lut[Math.min(lutN - 1, Math.round((Math.hypot(x, y) / rmax) * (lutN - 1)))] / peak;
        const o = (py * size + px) * 4;
        const level = srgb(Math.min(Math.abs(v), 1));
        img.data[o] = level;
        img.data[o + 1] = v >= 0 ? level : 0;
        img.data[o + 2] = v >= 0 ? level : 0;
        img.data[o + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

const srgbLevel = (v) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);

// Values (row-major w × h) as drawDisc colors, normalized to their own peak, which is returned.
function valueImage(ctx, values, w, h) {
  const peak = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
  const img = ctx.createImageData(w, h);
  values.forEach((value, i) => {
    const v = value / peak;
    const level = srgbLevel(Math.min(Math.abs(v), 1));
    img.data.set([level, v >= 0 ? level : 0, v >= 0 ? level : 0, 255], 4 * i);
  });
  return { img, peak };
}

// How the kernel is formed, as real separable terms h(x) v(y) (pairs of which are complex channels
// Re[h v]) summed into the kernel: "components" uses each component's own separable form, its channels
// summed into the component; "channels" uses the renderer's channels, summed as the shader accumulates
// them. Each image is normalized to its own peak, printed under it.
export function drawDecomposition(canvas, best, transition, mode) {
  const T = 56, gap = 46, rowH = T + 20, strip = 7, top = 8, left = 8;
  const dpr = window.devicePixelRatio || 1;
  const n = Math.round(T * dpr);
  const S = 1 + transition;
  const xs = Array.from({ length: n }, (_, i) => ((i + 0.5) / n * 2 - 1) * S);
  const mult = best ? multiplicities(best) : [];
  const channels = !best ? [] : mode === "components" ? componentChannels(best) : separableChannels(best, S);
  const leaves = [];
  const groups = channels.map((ch, j) => {
    const caption = mode === "channels" ? `ch${j}`
      : mult[ch.component] === 1 ? `u${ch.component}`
      : `u${ch.component}.${j - channels.findIndex((c) => c.component === ch.component)}`;
    const h = xs.map((x) => channelSide(ch.h, x * x));
    const v = xs.map((y) => channelSide(ch.v, y * y));
    const group = { caption, component: ch.component, leaves: [] };
    // Re[h v] = h_re v_re + h_im (−v_im)
    for (const [hs, vs] of [[h.map((z) => z[0]), v.map((z) => z[0])], [h.map((z) => z[1]), v.map((z) => -z[1])]]) {
      if (!hs.some((a) => a) || !vs.some((a) => a)) continue;
      const tile = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) tile[y * n + x] = hs[x] * vs[y];
      group.leaves.push(leaves.length);
      leaves.push({ tile, h: hs, v: vs });
    }
    if (group.leaves.length === 1) leaves[group.leaves[0]].caption = caption;
    return group;
  });
  const tileX = left + T + 20 + strip + 24;
  const colX = (col) => tileX + col * (T + gap);
  // Terms into channels, channels into components (component form), then one sum into the kernel.
  const ops = [];
  const leafNodes = leaves.map((leaf, i) => ({ tile: leaf.tile, y: top + i * rowH + T / 2, col: 0 }));
  const combine = (children, caption = "") => {
    if (children.length === 1) return children[0];
    const tile = new Float64Array(n * n);
    for (const c of children) for (let i = 0; i < tile.length; i++) tile[i] += c.tile[i];
    const node = { tile, y: (children[0].y + children.at(-1).y) / 2, col: Math.max(...children.map((c) => c.col)) + 1 };
    ops.push({ node, children, caption });
    return node;
  };
  let nodes = groups.map((g) => ({ ...g, node: combine(g.leaves.map((i) => leafNodes[i]), g.caption) }));
  nodes = mode === "components"
    ? mult.map((_, u) => combine(nodes.filter((g) => g.component === u).map((g) => g.node), `u${u}`))
    : nodes.map((g) => g.node);
  if (nodes.length) combine(nodes);
  const columns = Math.max(0, ...ops.map((o) => o.node.col)) + 1;
  canvas.style.width = `${colX(columns) - gap + 8}px`;
  canvas.style.height = `${top + Math.max(1, leaves.length) * rowH}px`;
  const { ctx } = setupCanvas(canvas);
  const put = (values, w, h, x, y) => {
    const { img, peak } = valueImage(ctx, values, w, h);
    ctx.putImageData(img, Math.round(x * dpr), Math.round(y * dpr));
    return peak;
  };
  const label = (text, x, y, size = 10) => {
    ctx.font = `${size}px system-ui, sans-serif`;
    ctx.fillStyle = size > 10 ? "#c9d1e3" : "#8b93a7";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  };
  const drawTile = (node, caption) => {
    const peak = put(node.tile, n, n, colX(node.col), node.y - T / 2);
    label(`${caption ? `${caption} ` : ""}±${peak.toPrecision(2)}`, colX(node.col) + T / 2, node.y + T / 2 + 8);
  };
  const ds = Math.round(strip * dpr);
  leaves.forEach((leaf, i) => {
    const node = { tile: leaf.tile, y: top + i * rowH + T / 2, col: 0 };
    if (leaf.h) {
      put(Array.from({ length: ds }, () => leaf.h).flat(), n, ds, left, node.y - strip / 2);
      label("*", left + T + 10, node.y, 18);
      put(leaf.v.flatMap((a) => Array(ds).fill(a)), ds, n, left + T + 20, node.y - T / 2);
      label("=", left + T + 20 + strip + 12, node.y, 18);
    }
    drawTile(node, leaf.caption);
  });
  for (const { node, children, caption } of ops) {
    const plusX = colX(Math.max(...children.map((c) => c.col))) + T + gap / 2 - 10;
    for (let i = 1; i < children.length; i++) label("+", plusX, (children[i - 1].y + children[i].y) / 2, 18);
    label("=", colX(node.col) - gap / 2 + 10, node.y, 18);
    drawTile(node, ops.at(-1).node === node ? "K" : caption);
  }
}
