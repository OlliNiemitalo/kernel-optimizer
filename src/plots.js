// Canvas 2-D plotting for profiles, errors, convergence, and the 2-D disc.
import { componentProfile, profile } from "./problem.js";

const COLORS = ["#4cc9f0", "#f72585", "#90be6d", "#f9c74f", "#b5179e", "#43aa8b", "#f3722c", "#577590"];

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
export function drawProfile(canvas, { best, population = [], transition, ripple, xmax, showComponents }) {
  const { ctx, w, h } = setupCanvas(canvas);
  const n = 600;
  const xs = Array.from({ length: n }, (_, k) => (k / (n - 1)) * xmax);
  const bestY = best ? xs.map((x) => profile(best, x)) : [];
  const lo = Math.min(-0.15, ...bestY);
  const hi = Math.max(1.15, ...bestY);
  const { X, Y, m } = frame(ctx, w, h, [0, xmax], [lo, hi], {
    title: "Radial cross-section K(r)",
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
  if (best && showComponents) {
    const nc = best.length / 4;
    for (let u = 0; u < nc; u++) {
      line(ctx, X, Y, xs, xs.map((x) => componentProfile(best, u, x)), COLORS[u % COLORS.length], 1, 0.8);
    }
  }
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

// 2-D kernel K(√(x²+y²)); positive values warm, negative values blue with extra gain.
export function drawDisc(canvas, best, xmax, negativeGain = 10) {
  const size = canvas.width;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  if (best) {
    const lutN = 2048;
    const rmax = xmax * Math.SQRT2;
    const lut = Float64Array.from({ length: lutN }, (_, k) => profile(best, (k / (lutN - 1)) * rmax));
    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const x = ((px + 0.5) / size - 0.5) * 2 * xmax;
        const y = ((py + 0.5) / size - 0.5) * 2 * xmax;
        const v = lut[Math.min(lutN - 1, Math.round((Math.hypot(x, y) / rmax) * (lutN - 1)))];
        const o = (py * size + px) * 4;
        if (v >= 0) {
          const t = Math.min(v / 1.1, 1);
          img.data[o] = 255 * Math.min(1, t * 1.6);
          img.data[o + 1] = 255 * Math.pow(t, 1.4);
          img.data[o + 2] = 255 * Math.pow(t, 3);
        } else {
          const t = Math.min(-v * negativeGain, 1);
          img.data[o] = 40 * t;
          img.data[o + 1] = 110 * t;
          img.data[o + 2] = 255 * t;
        }
        img.data[o + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
