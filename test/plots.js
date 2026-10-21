import assert from "node:assert/strict";
import { componentTerms, MAX_MULTIPLICITY, PARAMS_PER_COMPONENT, presetParams, profile } from "../src/problem.js";
import { drawDecomposition, drawDisc, drawProfile } from "../src/plots.js";

const pad = (values) => [...values, ...Array(PARAMS_PER_COMPONENT - values.length).fill(0)];
const reported = pad([5.3932719936984945, 2.1675026283783136, 0.978182557971403, 671.4559782532165,
  -1447.0864723543666, -215.61780912155626, 430.4142726210623, -885.7463643834903]);

function canvasRecorder() {
  const images = [], labels = [], strokes = [];
  let image, points;
  const ctx = {
    setTransform() {}, clearRect() {}, save() {}, restore() {}, rect() {}, clip() {}, fillRect() {},
    beginPath() { points = []; },
    moveTo(x, y) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); points.push([x, y]); },
    lineTo(x, y) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); points.push([x, y]); },
    stroke() { strokes.push({ color: this.strokeStyle, points }); },
    fillText(text, x, y) {
      assert.ok(Number.isFinite(x) && Number.isFinite(y));
      labels.push({ text, image });
    },
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData(img, x, y) {
      assert.ok(Number.isFinite(x) && Number.isFinite(y));
      image = img;
      images.push(img);
    },
  };
  const canvas = {
    width: 640, height: 320,
    get clientWidth() { return parseFloat(this.style.width) || 640; },
    get clientHeight() { return parseFloat(this.style.height) || 320; },
    style: {}, getContext: () => ctx,
  };
  return { canvas, images, labels, strokes };
}

function checkImage(image, params) {
  const n = image.width;
  const values = Array.from({ length: n * n }, (_, i) => {
    const x = ((i % n + 0.5) / n * 2 - 1) * 1.5;
    const y = ((Math.floor(i / n) + 0.5) / n * 2 - 1) * 1.5;
    return profile(params, Math.hypot(x, y));
  });
  const peak = Math.max(...values.map(Math.abs)) || 1;
  values.forEach((value, i) => {
    const v = Math.min(Math.abs(value / peak), 1);
    const level = Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
    const expected = [level, value >= 0 ? level : 0, value >= 0 ? level : 0, 255];
    expected.forEach((byte, channel) => assert.ok(Math.abs(image.data[4 * i + channel] - byte) <= 1));
  });
}

export function testPlots() {
  for (const params of [pad([3, 0, 0, 0]), pad([0.001, 0, 1, 0]), pad([0.001, 0, -1, 0]), presetParams(1)]) {
    const { canvas, images } = canvasRecorder();
    canvas.width = 32;
    canvas.height = 32;
    drawDisc(canvas, params, 2.5);
    const { data } = images[0];
    const lut = Array.from({ length: 2048 }, (_, i) => profile(params, i / 2047 * 2.5 * Math.SQRT2));
    const peak = Math.max(1e-12, ...lut.map(Math.abs));
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
      const radius = Math.hypot(((x + 0.5) / 32 - 0.5) * 5, ((y + 0.5) / 32 - 0.5) * 5);
      const v = lut[Math.min(2047, Math.round(radius / (2.5 * Math.SQRT2) * 2047))] / peak;
      [0.002, 0.005, 0.02].forEach((background, c) => {
        const linear = Math.max(0, Math.min(1, background + v));
        const expected = Math.round(255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055));
        assert.ok(Math.abs(data[(y * 32 + x) * 4 + c] - expected) <= 1);
      });
      assert.equal(data[(y * 32 + x) * 4 + 3], 255);
    }
    if (params[2] === 0) assert.deepEqual(Array.from(data.slice(0, 4)), [7, 16, 39, 255]);
    if (params[2] === -1) assert.deepEqual(Array.from(data.slice(0, 4)), [0, 0, 0, 255]);
    if (params[2] === 1) assert.deepEqual(Array.from(data.slice(0, 4)), [254, 254, 255, 255]);
  }
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  try {
    for (const dpr of [1, 1.25, 2]) {
      Object.defineProperty(globalThis, "window", { configurable: true, value: { devicePixelRatio: dpr } });
      for (const params of [
        reported, presetParams(2), [...presetParams(1), ...reported],
        pad([3, 0, 0, 1, 0, 0, 2, 0]), pad([3, 0, 0, 0]),
        pad([3, 2, ...Array.from({ length: 2 * MAX_MULTIPLICITY }, (_, i) => 0.1 ** Math.floor(i / 2))]),
      ]) {
        const terms = componentTerms(params);
        for (const x of [0, 0.2, 0.7, 1, 1.5, 2.5]) {
          const sum = terms.reduce((s, term) => s + profile(term.params, x), 0);
          assert.ok(Math.abs(sum - profile(params, x)) < 1e-10);
        }
        const plot = canvasRecorder();
        drawProfile(plot.canvas, { best: params, transition: 0.5, xmax: 2.5, showComponents: true });
        const curves = plot.strokes.filter((stroke) => !["#2a2f3a", "#5c677d", "#ffffff"].includes(stroke.color));
        assert.equal(curves.length, terms.length);
        assert.equal(plot.strokes.at(-1), curves.at(-1));
        const xs = Array.from({ length: 600 }, (_, i) => i / 599 * 2.5);
        const all = xs.map((x) => profile(params, x)).concat(...terms.map((term) => xs.map((x) => profile(term.params, x))));
        const lo = Math.min(-0.15, ...all) * 1.05, hi = Math.max(1.15, ...all) * 1.05;
        curves.forEach((curve, i) => curve.points.forEach(([, y], j) => {
          const expected = 290 - (profile(terms[i].params, xs[j]) - lo) / (hi - lo) * 268;
          assert.ok(Math.abs(y - expected) < 1e-10);
        }));
        const diagram = canvasRecorder();
        drawDecomposition(diagram.canvas, params, 0.5, "components");
        for (const term of terms) {
          const label = diagram.labels.find(({ text }) => text.startsWith(`u${term.component} x^${2 * term.power} `));
          if (label) checkImage(label.image, term.params);
          else assert.equal(terms.filter((t) => t.component === term.component).length, 1);
        }
        checkImage(diagram.labels.find(({ text }) => text.startsWith("K ")).image, params);
      }
      const diagram = canvasRecorder();
      drawDecomposition(diagram.canvas, reported, 0.5, "channels");
      checkImage(diagram.labels.find(({ text }) => text.startsWith("K ")).image, reported);
      drawDecomposition(canvasRecorder().canvas, null, 0.5, "components");
    }
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else delete globalThis.window;
  }
}
