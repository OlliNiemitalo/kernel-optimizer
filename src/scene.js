// Procedural HDR test scene for showing lens-blur discs (bokeh), plus image loading.

function srgbToLinear(v) {
  const c = v / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function canvasToLinear(ctx, width, height) {
  const { data } = ctx.getImageData(0, 0, width, height);
  const out = new Float32Array(width * height * 4);
  const lut = Float32Array.from({ length: 256 }, (_, v) => srgbToLinear(v));
  for (let i = 0; i < width * height; i++) {
    out[i * 4] = lut[data[i * 4]];
    out[i * 4 + 1] = lut[data[i * 4 + 1]];
    out[i * 4 + 2] = lut[data[i * 4 + 2]];
    out[i * 4 + 3] = 1;
  }
  return out;
}

// Adds an HDR point light of the given linear color, as a small anti-aliased disc.
function addLight(img, width, height, cx, cy, radius, color) {
  const r = Math.ceil(radius + 1);
  for (let y = Math.floor(cy - r); y <= cy + r; y++) {
    if (y < 0 || y >= height) continue;
    for (let x = Math.floor(cx - r); x <= cx + r; x++) {
      if (x < 0 || x >= width) continue;
      const cover = Math.min(1, Math.max(0, radius + 0.5 - Math.hypot(x + 0.5 - cx, y + 0.5 - cy)));
      if (cover <= 0) continue;
      const o = (y * width + x) * 4;
      img[o] += color[0] * cover;
      img[o + 1] += color[1] * cover;
      img[o + 2] += color[2] * cover;
    }
  }
}

export function proceduralScene(width = 960, height = 600, seed = 3) {
  const rand = mulberry32(seed);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#0b1026");
  sky.addColorStop(0.65, "#1b2147");
  sky.addColorStop(1, "#2a1a2e");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  // Skyline silhouettes with lit windows.
  const windows = [];
  for (let x = 0; x < width; ) {
    const w = 40 + rand() * 90;
    const h = height * (0.25 + rand() * 0.4);
    ctx.fillStyle = `hsl(230, 25%, ${8 + rand() * 6}%)`;
    ctx.fillRect(x, height - h, w, h);
    for (let wy = height - h + 10; wy < height - 20; wy += 14) {
      for (let wx = x + 6; wx < x + w - 8; wx += 11) {
        if (rand() < 0.22) windows.push([wx + 3, wy + 4]);
      }
    }
    x += w + 4;
  }

  // Sharp details: checkerboard and text, to compare against the blur.
  const cell = 12;
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 12; i++) {
      ctx.fillStyle = (i + j) % 2 ? "#d8d8d8" : "#202020";
      ctx.fillRect(40 + i * cell, 40 + j * cell, cell, cell);
    }
  }
  ctx.fillStyle = "#e8e8f0";
  ctx.font = "bold 56px system-ui, sans-serif";
  ctx.fillText("Bokeh", width - 260, 90);
  ctx.font = "16px system-ui, sans-serif";
  ctx.fillText("complex Gaussian disc blur", width - 260, 116);

  const img = canvasToLinear(ctx, width, height);

  // Warm and cool lit windows.
  for (const [x, y] of windows) {
    const warm = rand() < 0.7;
    const g = 1.5 + rand() * 3;
    addLight(img, width, height, x, y, 1.6, warm ? [g, g * 0.7, g * 0.3] : [g * 0.5, g * 0.7, g]);
  }
  // Scattered bright point lights of different colors and strengths.
  for (let k = 0; k < 70; k++) {
    const hue = rand() * 6;
    const c = [Math.abs(hue - 3) - 1, 2 - Math.abs(hue - 2), 2 - Math.abs(hue - 4)].map((v) => Math.min(1, Math.max(0.15, v)));
    const g = 8 + rand() * 60;
    addLight(img, width, height, rand() * width, rand() * height * 0.75, 1 + rand() * 1.5, c.map((v) => v * g));
  }
  // A string of fairy lights along a curve.
  for (let k = 0; k < 40; k++) {
    const t = k / 39;
    const x = 80 + t * (width - 160);
    const y = height * 0.55 + Math.sin(t * Math.PI) * -90 + Math.sin(t * 9) * 6;
    addLight(img, width, height, x, y, 1.5, [40, 30, 12]);
  }
  // A thin neon line.
  for (let x = width * 0.35; x < width * 0.65; x += 0.5) {
    addLight(img, width, height, x, height * 0.2, 0.8, [3, 0.4, 6]);
  }
  return { rgba: img, width, height };
}

// Loads a user image; boost lifts near-white pixels into HDR so highlights bloom into discs.
export async function loadImage(file, boost = 8, maxSize = 1280) {
  const bitmap = await createImageBitmap(file);
  const s = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * s));
  const height = Math.max(1, Math.round(bitmap.height * s));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  const img = canvasToLinear(ctx, width, height);
  for (let i = 0; i < width * height; i++) {
    const lum = 0.2126 * img[i * 4] + 0.7152 * img[i * 4 + 1] + 0.0722 * img[i * 4 + 2];
    const gain = 1 + boost * Math.pow(Math.min(1, lum), 8);
    img[i * 4] *= gain;
    img[i * 4 + 1] *= gain;
    img[i * 4 + 2] *= gain;
  }
  return { rgba: img, width, height };
}
