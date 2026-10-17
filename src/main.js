import { _setCalibrationState, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { BlurRenderer } from "./blur.js";
import { DifferentialEvolution } from "./de.js";
import { FineTuner } from "./finetune.js";
import { drawConvergence, drawDisc, drawError, drawProfile } from "./plots.js";
import {
  PARAMS_PER_COMPONENT, bounds, centerFrequency, channelCount, formatParams, maxError, multiplicities, parseParams,
  presetParams, sampleGrid, sortByFrequency,
} from "./problem.js";
import { loadImage, proceduralScene } from "./scene.js";

const $ = (id) => document.getElementById(id);
const num = (id) => Number($(id).value);

const state = {
  backend: null,
  de: null,
  running: false,
  busy: false,
  history: [],
  populationSample: [],
  current: null, // { params, source, error, denseError }
  blur: null,
  imageFile: null,
};

// Multiplicity per component (1 = single, 2 = merged pair), padded or truncated to the component count.
function readMultiplicity(components) {
  const given = ($("multiplicity").value.match(/\d+/g) ?? []).map((v) => (Number(v) === 2 ? 2 : 1));
  return Array.from({ length: components }, (_, u) => given[u] ?? 1);
}

function setMultiplicity(mult) {
  $("multiplicity").value = mult.join(", ");
}

function readProblem() {
  const components = Math.min(8, Math.max(1, Math.round(num("components"))));
  return {
    components,
    multiplicity: readMultiplicity(components),
    transition: num("transition"),
    passSamples: Math.max(2, Math.round(num("passSamples"))),
    stopSamples: Math.max(2, Math.round(num("stopSamples"))),
    frequencyBands: $("frequencyBands").checked,
  };
}

function denseGrid(problem) {
  const n = Math.max(10, Math.round(num("denseSamples")));
  return sampleGrid(problem, n, n);
}

function setStatus(text) {
  $("status").textContent = text;
}

// Animation frames stop in hidden tabs; the timer keeps optimization going there.
function nextFrame() {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 100);
    requestAnimationFrame(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function describeRanges(problem) {
  const n = problem.components;
  const b = bounds(problem);
  const bands = problem.frequencyBands
    ? Array.from({ length: n }, (_, u) => `${(centerFrequency(u) - 0.4 * Math.PI).toFixed(2)}–${(centerFrequency(u) + 0.4 * Math.PI).toFixed(2)}`).join(", ")
    : `${b.initLo[1].toFixed(2)}–${b.initHi[1].toFixed(2)}`;
  const m = channelCount(problem.multiplicity);
  $("ranges").textContent =
    `${m} complex channels per pass. Initial ranges: decay b ∈ [${b.initLo[0]}, ${b.initHi[0]}], frequency c ∈ ${bands}, ` +
    `weights r, i (and r₁, i₁ of merged pairs) ∈ ±${b.initHi[2].toPrecision(3)} ` +
    `(grow 2.5× per channel, as in the article's solutions).`;
  // Larger sets have larger, more strongly cancelling weights, so take smaller relative steps.
  if ($("ftLr").dataset.auto) $("ftLr").value = Math.min(1e-3, 1e-3 / Math.pow(3, m - 2)).toPrecision(2);
}

// ---- current kernel -------------------------------------------------------------------

function setCurrent(params, source) {
  const problem = readProblem();
  state.current = {
    params: Array.from(params),
    source,
    error: maxError(params, sampleGrid(problem)),
    denseError: maxError(params, denseGrid(problem)),
  };
  $("output").value =
    formatParams(state.current.params, problem, state.current.error) +
    `\n\n// Source: ${source}; ripple on the dense grid: ±${state.current.denseError.toFixed(6)}`;
  if ($("follow").checked) requestBlur(true);
}

function drawAll() {
  const problem = readProblem();
  const xmax = num("xmax");
  const best = state.current?.params ?? null;
  const ripple = state.current?.error ?? 0;
  drawProfile($("profile"), {
    best,
    label: state.current && `${state.current.source}, ripple ±${state.current.error.toFixed(5)}`,
    population: $("showPopulation").checked ? state.populationSample : [],
    transition: problem.transition,
    ripple,
    xmax,
    showComponents: $("showComponents").checked,
  });
  drawError($("error"), { best, transition: problem.transition, ripple, xmax });
  drawConvergence($("convergence"), state.history);
  drawDisc($("disc"), best, xmax, num("negGain"));
}

// ---- differential evolution ---------------------------------------------------------------

function resetDE({ showBest = false, include = null } = {}) {
  state.de?.dispose();
  state.de = null;
  const problem = readProblem();
  describeRanges(problem);
  const D = problem.components * PARAMS_PER_COMPONENT;
  const candidate = include ?? ($("seedWithCurrent").checked ? state.current?.params : null);
  const seedWith = candidate?.length === D ? candidate : null;
  try {
    state.de = new DifferentialEvolution({
      components: problem.components,
      grid: sampleGrid(problem),
      bounds: bounds(problem),
      populationSize: Math.max(8, Math.round(num("population"))),
      F: num("F"),
      CR: num("CR"),
      seed: Math.round(num("seed")) >>> 0,
      startingPoint: seedWith && Float32Array.from(seedWith),
    });
  } catch (e) {
    setStatus(`Could not create optimizer: ${e.message}`);
    return;
  }
  state.history = [];
  state.populationSample = [];
  observe(performance.now(), 0, showBest);
}

async function observe(startTime, generationsRun, showBest = false) {
  const de = state.de;
  let snap;
  try {
    snap = await de.snapshot();
  } catch (e) {
    if (de === state.de) setStatus(`Optimizer failed: ${e.message}`);
    return;
  }
  if (de !== state.de) return;
  const D = de.D;
  state.history.push({ gen: de.generation, best: snap.bestCost, mean: snap.meanCost });
  if (state.history.length > 4000) state.history = state.history.filter((_, k) => k % 2 === 0);
  const picks = Math.min(24, de.P);
  state.populationSample = Array.from({ length: picks }, () => {
    const k = Math.floor(Math.random() * de.P);
    return Array.from(snap.population.subarray(k * D, (k + 1) * D));
  });
  const keepCurrent =
    state.current && state.current.source !== "differential evolution" &&
    state.current.params.length === D && state.current.error <= snap.bestCost;
  // Before the first generation, keep showing a loaded preset or pasted kernel.
  if (showBest || generationsRun > 0 || !keepCurrent) setCurrent(snap.best, "differential evolution");
  const now = performance.now();
  const rate = generationsRun > 0 ? (generationsRun * de.P) / ((now - startTime) / 1000) : 0;
  setStatus(
    `Optimizer on ${state.backend} · generation ${de.generation} · ` +
      (rate ? `${Math.round(rate).toLocaleString()} evaluations/s · ` : "") +
      `population best ±${snap.bestCost.toFixed(6)}, mean ${snap.meanCost.toFixed(5)}`,
  );
  drawAll();
}

async function runLoop() {
  while (state.running && state.de) {
    const t0 = performance.now();
    const gens = Math.max(1, Math.round(num("gensPerFrame")));
    state.de.step(gens);
    await observe(t0, gens);
    await nextFrame();
  }
}

function setRunning(running) {
  state.running = running;
  $("run").textContent = running ? "Pause" : "Start";
  if (running) runLoop();
}

// ---- gradient fine-tuning -------------------------------------------------------------------

async function fineTune() {
  if (!state.current || state.busy) return;
  const wasRunning = state.running;
  setRunning(false);
  state.busy = true;
  $("finetune").disabled = true;
  const problem = readProblem();
  const grid = denseGrid(problem);
  const components = state.current.params.length / PARAMS_PER_COMPONENT;
  const steps = Math.max(100, Math.round(num("ftSteps")));
  const roundSteps = 100;
  const rounds = Math.ceil(steps / roundSteps);
  const tuner = new FineTuner({
    params: state.current.params,
    components,
    grid,
    learningRate: num("ftLr"),
    halfLife: steps / 3,
  });
  let best = { params: state.current.params, error: state.current.denseError };
  const start = { ...best };
  try {
    for (let k = 0; k < rounds; k++) {
      const sharpness = num("ftSharpness") * Math.pow(10, k / Math.max(1, rounds - 1));
      const r = await tuner.run(roundSteps, sharpness, best.error);
      if (r.error < best.error) {
        best = r;
        setCurrent(r.params, "gradient fine-tuning");
        drawAll();
      }
      setStatus(
        `Fine-tuning on ${state.backend} · step ${tuner.step}/${rounds * roundSteps} · ` +
          `dense-grid ripple ±${best.error.toFixed(6)} (started ±${start.error.toFixed(6)})`,
      );
      await nextFrame();
    }
  } catch (e) {
    setStatus(`Fine-tuning failed: ${e.message}`);
  } finally {
    tuner.dispose();
    // So that resuming the optimizer can't lose the fine-tuned kernel.
    if (best.params !== start.params && state.de?.D === best.params.length) state.de.inject(best.params);
    state.busy = false;
    $("finetune").disabled = false;
    if (wasRunning) setRunning(true);
  }
}

// ---- WebGPU blur demo ---------------------------------------------------------------------

let blurPending = false;
let blurKernelDirty = true;
let blurRendering = false;
let lastBlurTime = 0;

function requestBlur(kernelChanged = false) {
  blurKernelDirty ||= kernelChanged;
  if (blurPending || !state.blur) return;
  blurPending = true;
  const wait = state.running ? Math.max(0, 150 - (performance.now() - lastBlurTime)) : 0;
  setTimeout(() => nextFrame().then(renderBlur), wait);
}

async function renderBlur() {
  blurPending = false;
  if (blurRendering) {
    requestBlur();
    return;
  }
  if (!state.blur || !state.current) return;
  blurRendering = true;
  try {
    const radius = num("radius");
    if (blurKernelDirty) {
      state.blur.setKernel(state.current.params, radius, readProblem().transition);
      blurKernelDirty = false;
    }
    const r = await state.blur.render({
      exposure: Math.pow(2, num("exposure")),
      tonemap: Number($("tonemap").value),
      precision: $("precision").value,
      showSource: $("showSource").checked,
    });
    lastBlurTime = performance.now();
    const gpu =
      r.horizontalMs !== undefined
        ? `GPU time: horizontal ${r.horizontalMs.toFixed(2)} ms + vertical ${r.verticalMs.toFixed(2)} ms`
        : `${r.ms.toFixed(1)} ms round trip`;
    $("blurStats").textContent =
      `${state.blur.width}×${state.blur.height} px · ${state.blur.channels} separable channels × ${r.taps} taps per pass · ` +
      `${(2 * state.blur.channels * r.taps).toLocaleString()} complex MACs per pixel per color · ` +
      `intermediates ${r.f16 ? "f16" : "f32"} (${(r.midBytes / 2 ** 20).toFixed(1)} MiB) · ${gpu}`;
  } catch (e) {
    $("blurStats").textContent = `Blur failed: ${e.message}`;
  } finally {
    blurRendering = false;
  }
}

async function useImage(image) {
  state.blur.setImage(image.rgba, image.width, image.height);
  requestBlur(true);
}

async function initBlur() {
  try {
    state.blur = await BlurRenderer.create($("blurCanvas"));
    if (!state.blur.supportsF16) {
      $("precision").querySelector('[value="f16"]').disabled = true;
    }
    await useImage(proceduralScene());
  } catch (e) {
    $("blurStats").textContent = `WebGPU blur demo unavailable: ${e.message}`;
  }
}

// ---- wiring ---------------------------------------------------------------------------------

function wire() {
  $("run").onclick = () => {
    if (!state.de) resetDE();
    setRunning(!state.running);
  };
  $("reset").onclick = () => {
    const wasRunning = state.running;
    setRunning(false);
    $("seed").value = (Math.round(num("seed")) + 1) >>> 0;
    resetDE({ showBest: true });
    if (wasRunning) setRunning(true);
  };
  for (const id of ["components", "multiplicity", "transition", "passSamples", "stopSamples", "frequencyBands", "population", "F", "CR", "seed"]) {
    $(id).addEventListener("change", () => {
      setRunning(false);
      if (id === "multiplicity") {
        const count = ($("multiplicity").value.match(/\d+/g) ?? []).length;
        if (count >= 1 && count <= 8) $("components").value = count;
      }
      setMultiplicity(readProblem().multiplicity);
      if (state.current) setCurrent(state.current.params, state.current.source);
      resetDE();
    });
  }
  $("denseSamples").addEventListener("change", () => state.current && setCurrent(state.current.params, state.current.source));
  $("preset").onchange = () => {
    const n = Number($("preset").value);
    if (!n) return;
    setRunning(false);
    $("components").value = n;
    setMultiplicity(Array(n).fill(1));
    $("transition").value = 0.2;
    setCurrent(sortByFrequency(presetParams(n)), `article preset (${n} components)`);
    resetDE({ include: state.current.params });
    $("preset").value = "";
  };
  $("finetune").onclick = fineTune;
  $("ftLr").addEventListener("input", () => delete $("ftLr").dataset.auto);
  $("copy").onclick = () => navigator.clipboard.writeText($("output").value);
  $("applyPasted").onclick = () => {
    const box = $("pasted");
    if (box.hidden) {
      box.hidden = false;
      box.focus();
      return;
    }
    try {
      const params = sortByFrequency(parseParams(box.value));
      setRunning(false);
      $("components").value = params.length / PARAMS_PER_COMPONENT;
      setMultiplicity(multiplicities(params));
      setCurrent(params, "pasted parameters");
      resetDE({ include: state.current.params });
      box.hidden = true;
    } catch (e) {
      setStatus(e.message);
    }
  };
  for (const id of ["xmax", "showPopulation", "showComponents", "negGain"]) $(id).addEventListener("input", drawAll);
  $("radius").addEventListener("input", () => {
    $("radiusOut").textContent = $("radius").value;
    requestBlur(true);
  });
  $("exposure").addEventListener("input", () => {
    $("exposureOut").textContent = `${Math.pow(2, num("exposure")).toFixed(2)}×`;
    requestBlur();
  });
  for (const id of ["tonemap", "precision", "showSource"]) $(id).addEventListener("change", () => requestBlur());
  $("follow").addEventListener("change", () => requestBlur(true));
  $("imageFile").addEventListener("change", async () => {
    state.imageFile = $("imageFile").files[0] ?? null;
    if (state.imageFile && state.blur) await useImage(await loadImage(state.imageFile, num("boost")));
  });
  $("boost").addEventListener("change", async () => {
    if (state.imageFile && state.blur) await useImage(await loadImage(state.imageFile, num("boost")));
  });
  $("sceneButton").onclick = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useImage(proceduralScene());
  };
  window.addEventListener("resize", drawAll);
}

async function main() {
  const repo = import.meta.env.VITE_REPO_URL;
  if (repo) {
    $("mathLink").href = `${repo}#how-it-works`;
    $("sourceLink").href = repo;
    $("repoLinks").hidden = false;
  }
  wire();
  describeRanges(readProblem());
  initBlur();
  // jax-js otherwise benchmarks the GPU at startup (~128 MB of buffers plus saturating
  // compute kernels), which can stall the page; its built-in conservative profile suffices.
  const calibrate = new URLSearchParams(location.search).has("calibrate");
  if (!calibrate) _setCalibrationState("off");
  setStatus(`Initializing jax-js${calibrate ? " (WebGPU calibration can take several seconds)" : ""}…`);
  try {
    const t0 = performance.now();
    const devices = await init("webgpu", "wasm");
    console.info(`jax-js init ${calibrate ? "with" : "without"} calibration: ${(performance.now() - t0).toFixed(0)} ms`);
    state.backend = devices.includes("webgpu") ? "webgpu" : "wasm";
    defaultDevice(state.backend);
  } catch (e) {
    setStatus(`jax-js failed to initialize: ${e.message}`);
    return;
  }
  setCurrent(sortByFrequency(presetParams(3)), "article preset (3 components)");
  resetDE();
}

main();
