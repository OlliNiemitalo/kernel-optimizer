// Population initialization shared by the GPU and CPU (projected) differential evolution.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Centered initialization uses a symmetric neighborhood, clipped in width (not shifted) at
// search bounds. Optionally keep the exact center as member zero. Global initialization is unchanged.
export function initialPopulation({ bounds, populationSize, dimensions, seed, center = null, includeCenter = true }) {
  const D = dimensions, P = populationSize;
  if (center && (center.length !== D || Array.from(center).some((v, j) =>
    !Number.isFinite(v) || v < bounds.foldLo[j] || v > bounds.foldHi[j]))) {
    throw new Error("Current kernel must match the component structure and search bounds. Check frequency bands and pinned parameters.");
  }
  const rand = mulberry32(seed);
  const init = new Float32Array(P * D);
  for (let k = 0; k < P; k++) {
    for (let j = 0; j < D; j++) {
      if (center) {
        const radius = Math.min(0.05 * (bounds.initHi[j] - bounds.initLo[j]),
          center[j] - bounds.foldLo[j], bounds.foldHi[j] - center[j]);
        init[k * D + j] = center[j] + (2 * rand() - 1) * radius;
      } else {
        init[k * D + j] = bounds.initLo[j] + (bounds.initHi[j] - bounds.initLo[j]) * rand();
      }
    }
  }
  if (center && includeCenter) init.set(center, 0);
  return init;
}
