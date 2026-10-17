import { _setCalibrationState, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { BlurRenderer, shaderSource, tiltShiftRadiusMap } from "./blur.js";
import { World } from "./world.js";
import { VideoSource } from "./video.js";

const SAMPLE_VIDEO = { url: `${import.meta.env.BASE_URL}tum-sitting-rgbd.mp4`, name: "TUM RGB-D fr3/sitting_static" };
import { DifferentialEvolution } from "./de.js";
import { FineTuner } from "./finetune.js";
import { drawConvergence, drawDisc, drawError, drawProfile } from "./plots.js";
import {
  PARAMS_PER_COMPONENT, bounds, centerFrequency, channelCount, formatParams, maxError, multiplicities, parseParams,
  presetParams, nonnegativePresetParams, sampleGrid, sortByFrequency,
} from "./problem.js";
import { loadImage, proceduralScene } from "./scene.js";

const $ = (id) => document.getElementById(id);
const num = (id) => Number($(id).value);
// With this weight the article's 1-component non-negative set is about equiripple (its negative
// lobe × 100 ≈ its ripple), so further optimization keeps the presets' character.
const NONNEGATIVE_WEIGHT = 100;

const state = {
  backend: null,
  de: null,
  running: false,
  busy: false,
  history: [],
  populationSample: [],
  include: null, // kernel to include once in the next population
  current: null, // { params, source, error, denseError }
  blur: null,
  world: null,
  source: null, // "world", "video", "image" or "procedural"
  video: null, // { element, source, name }
  sceneTime: 0,
  playing: true,
  captureNext: false,
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
    negativeWeight: Math.max(1, num("negWeight") || 1),
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

// Drops the population; the next Start or New population creates one.
function discardDE({ include = null } = {}) {
  state.de?.dispose();
  state.de = null;
  if (include) state.include = include;
  state.history = [];
  state.populationSample = [];
  describeRanges(readProblem());
  if (state.current) {
    setStatus(
      `Showing ${state.current.source}, ripple ±${state.current.error.toFixed(6)}. ` +
        `Start creates a population of ${readProblem().components}-component kernels and optimizes it.`,
    );
  }
  drawAll();
}

function createDE() {
  state.de?.dispose();
  state.de = null;
  const problem = readProblem();
  describeRanges(problem);
  const D = problem.components * PARAMS_PER_COMPONENT;
  const candidate = state.include ?? ($("seedWithCurrent").checked ? state.current?.params : null);
  const seedWith = candidate?.length === D ? candidate : null;
  state.include = null;
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
    return false;
  }
  state.history = [];
  state.populationSample = [];
  return true;
}

async function observe(startTime, generationsRun) {
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
  setCurrent(snap.best, "differential evolution");
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
    if (best.params !== start.params) {
      if (state.de?.D === best.params.length) state.de.inject(best.params);
      else if (!state.de) state.include = best.params;
    }
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

const FRAME = 1 / 30; // seconds per step
const hasDepth = () => state.source === "world" || state.source === "video";
const animating = () => hasDepth() && state.playing && !document.hidden;

function showSceneTime() {
  if (document.activeElement !== $("sceneTime")) $("sceneTime").value = state.sceneTime.toFixed(4);
}

// Video time is the video element's; the 3D world's advances with the wall clock while playing.
function seekScene(time) {
  state.sceneTime = Math.max(0, time);
  if (state.source === "video") state.video.element.currentTime = state.sceneTime; // renders on "seeked"
  else requestBlur();
  showSceneTime();
}

function setPlaying(playing) {
  state.playing = playing;
  $("playPause").textContent = playing ? "Pause" : "Play";
  if (state.source === "video") {
    if (playing) state.video.element.play().catch(() => {});
    else state.video.element.pause();
  }
  lastBlurTime = 0;
  requestBlur();
}

function stepScene(frames) {
  setPlaying(false);
  // Video frames are addressed at their middle, away from the boundaries between frames.
  const offset = state.source === "video" ? 0.5 : 0;
  seekScene((Math.round(state.sceneTime / FRAME - offset + frames) + offset) * FRAME);
}

function setFocusSlider(value) {
  $("focus").value = value;
  $("focus").dispatchEvent(new Event("input"));
}

function download(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
}

const VIEW_CONTROLS = ["radius", "blurMode", "focus", "exposure", "tonemap", "precision", "transition"];

// Everything needed to re-render the displayed frame, plus context for bug reports.
function viewState() {
  const controls = Object.fromEntries(VIEW_CONTROLS.map((id) => [id, $(id).type === "checkbox" ? $(id).checked : $(id).value]));
  return {
    scene:
      state.source === "video" ? `video: ${state.video.name}`
      : state.source === "image" ? `image: ${state.imageFile?.name}`
      : state.source,
    time: state.sceneTime,
    playing: state.playing,
    controls,
    kernel: state.current?.params,
    size: state.blur && [state.blur.width, state.blur.height],
    gpu: state.blur?.adapterInfo,
    stats: $("blurStats").textContent,
    userAgent: navigator.userAgent,
  };
}

async function applyViewState(view) {
  if (view.scene === "world" && state.source !== "world") useWorld();
  if (view.scene === "procedural") useImage(proceduralScene(), "procedural");
  if (view.scene === `video: ${SAMPLE_VIDEO.name}` && state.video?.name !== SAMPLE_VIDEO.name) await useVideo(SAMPLE_VIDEO.url, SAMPLE_VIDEO.name);
  for (const [id, value] of Object.entries(view.controls ?? {})) {
    if (!VIEW_CONTROLS.includes(id)) continue;
    if ($(id).type === "checkbox") $(id).checked = value;
    else $(id).value = value;
    $(id).dispatchEvent(new Event("input"));
  }
  if (Array.isArray(view.kernel) && view.kernel.length % PARAMS_PER_COMPONENT === 0) {
    setRunning(false);
    $("components").value = view.kernel.length / PARAMS_PER_COMPONENT;
    setMultiplicity(multiplicities(view.kernel));
    setCurrent(view.kernel, "view state");
    discardDE({ include: state.current.params });
  }
  setPlaying(!!view.playing);
  seekScene(Number(view.time) || 0);
  lastBlurTime = 0;
  requestBlur(true);
}

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
      const { width, height } = state.blur;
      const mode = $("blurMode").value;
      if (mode === "depth" && hasDepth()) state.blur.setDepthOfField();
      else state.blur.setRadiusMap(mode === "tiltShift" ? tiltShiftRadiusMap(width, height, radius) : null);
      blurKernelDirty = false;
    }
    const now = performance.now();
    if (state.source === "video") state.sceneTime = state.video.element.currentTime;
    else if (animating() && lastBlurTime) state.sceneTime += Math.min(0.1, (now - lastBlurTime) / 1000);
    showSceneTime();
    const capture = state.captureNext;
    state.captureNext = false;
    const r = await state.blur.render({
      exposure: Math.pow(2, num("exposure")),
      tonemap: Number($("tonemap").value),
      precision: $("precision").value,
      time: state.sceneTime,
      invFocus: num("focus") ** 2,
      capture,
    });
    const prefix = { world: "street", video: "video" }[state.source];
    if (r.png) download(r.png, prefix ? `${prefix}-t${state.sceneTime.toFixed(4)}s.png` : "blur.png");
    lastBlurTime = now;
    const gpu = r.gpuMs
      ? "GPU time: " + Object.entries(r.gpuMs).map(([k, ms]) => `${k} ${ms.toFixed(2)} ms`).join(" + ")
      : `${r.ms.toFixed(1)} ms round trip`;
    const mib = (b) => (b / 2 ** 20).toFixed(1);
    const m = r.memory;
    const total = Object.values(m).reduce((a, b) => a + b, 0);
    $("blurStats").textContent =
      `${state.blur.width}×${state.blur.height} px · ${state.blur.channels} separable channels × ` +
      (r.varying ? `up to ${r.taps} taps (${r.levels} radius levels)` : `${r.taps} taps`) + ` per pass · ` +
      `GPU memory ${mib(total)} MiB (intermediate ${mib(m.intermediate)}, accumulator ${mib(m.accumulator)}, ` +
      `source ${mib(m.source)}` +
      (m.radiusMap ? `, radius ${mib(m.radiusMap)}, level maps ${mib(m.levelMaps)}` : "") +
      `) · ${gpu}`;
  } catch (e) {
    $("blurStats").textContent = `Blur failed: ${e.message}`;
  } finally {
    blurRendering = false;
    if (animating()) requestBlur();
  }
}

function stopVideo() {
  if (!state.video) return;
  state.video.element.pause();
  state.video.element.removeAttribute("src");
  state.video.element.load();
  state.video = null;
  $("videoCredit").hidden = true;
}

async function useImage(image, source = "image") {
  stopVideo();
  state.source = source;
  state.blur.setImage(image.rgba, image.width, image.height);
  if ($("blurMode").value === "depth") $("blurMode").value = "uniform";
  requestBlur(true);
}

function useWorld() {
  stopVideo();
  state.world ??= new World(state.blur.device);
  state.blur.setAnimated(state.world, 960, 600);
  state.source = "world";
  $("blurMode").value = "depth";
  setFocusSlider(Math.sqrt(0.1)); // 10 m
  requestBlur(true);
}

// An RGB-D video: color on top, inverse depth below (see video.js).
async function useVideo(url, name) {
  stopVideo();
  const element = document.createElement("video");
  Object.assign(element, { muted: true, loop: true, playsInline: true, src: url });
  await new Promise((resolve, reject) => {
    element.onloadeddata = resolve;
    element.onerror = () => reject(new Error(`Could not load the video ${name}.`));
  });
  element.addEventListener("seeked", () => requestBlur());
  const source = new VideoSource(state.blur.device, element);
  source.boost = num("boost");
  state.video = { element, source, name };
  $("videoCredit").hidden = name !== SAMPLE_VIDEO.name;
  state.blur.setAnimated(source, source.width, source.height);
  state.source = "video";
  state.sceneTime = 0;
  $("blurMode").value = "depth";
  setFocusSlider(0.75); // 1.8 m: the people in the sample clip
  setPlaying(state.playing);
  requestBlur(true);
}

async function initBlur() {
  try {
    state.blur = await BlurRenderer.create($("blurCanvas"));
    useWorld();
  } catch (e) {
    $("blurStats").textContent = `WebGPU blur demo unavailable: ${e.message}`;
  }
}

// ---- wiring ---------------------------------------------------------------------------------

const shaderPacked = () => $("precision").value === "f16";

function shaderFileName() {
  return $("shaderKind").value === "display" ? "blur-display.wgsl" : `blur-compute-${shaderPacked() ? "packed" : "f32"}.wgsl`;
}

function showShader() {
  $("shaderCode").value = shaderSource($("shaderKind").value, shaderPacked());
}

function wire() {
  $("run").onclick = () => {
    if (!state.running && !state.de && !createDE()) return;
    setRunning(!state.running);
  };
  $("reset").onclick = () => {
    const wasRunning = state.running;
    setRunning(false);
    $("seed").value = (Math.round(num("seed")) + 1) >>> 0;
    if (!createDE()) return;
    observe(performance.now(), 0);
    if (wasRunning) setRunning(true);
  };
  for (const id of ["components", "multiplicity", "transition", "passSamples", "stopSamples", "frequencyBands", "negWeight", "population", "F", "CR", "seed"]) {
    $(id).addEventListener("change", () => {
      setRunning(false);
      if (id === "multiplicity") {
        const count = ($("multiplicity").value.match(/\d+/g) ?? []).length;
        if (count >= 1 && count <= 8) $("components").value = count;
      }
      setMultiplicity(readProblem().multiplicity);
      if (state.current) setCurrent(state.current.params, state.current.source);
      discardDE();
    });
  }
  $("denseSamples").addEventListener("change", () => state.current && setCurrent(state.current.params, state.current.source));
  $("preset").onchange = () => {
    const value = $("preset").value;
    if (!value) return;
    const nonnegative = value.startsWith("nn");
    const n = Number(value.replace("nn", ""));
    setRunning(false);
    $("components").value = n;
    setMultiplicity(Array(n).fill(1));
    $("transition").value = 0.2;
    $("negWeight").value = nonnegative ? NONNEGATIVE_WEIGHT : 1;
    // The non-negative sets don't follow the frequency-band heuristic (one is almost a plain Gaussian).
    $("frequencyBands").checked = !nonnegative;
    const params = nonnegative ? nonnegativePresetParams(n) : presetParams(n);
    setCurrent(sortByFrequency(params), `article ${nonnegative ? "non-negative " : ""}preset (${n} components)`);
    discardDE({ include: state.current.params });
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
      discardDE({ include: state.current.params });
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
  // Slider position squared is 1 / distance: finer steps at long distances.
  $("focus").addEventListener("input", () => {
    const inv = num("focus") ** 2;
    $("focusOut").textContent = inv > 0 ? `${(1 / inv).toPrecision(2)} m` : "∞";
    requestBlur();
  });
  $("exposure").addEventListener("input", () => {
    $("exposureOut").textContent = `${Math.pow(2, num("exposure")).toFixed(2)}×`;
    requestBlur();
  });
  for (const id of ["tonemap", "precision"]) $(id).addEventListener("change", () => requestBlur());
  for (const id of ["precision", "shaderKind"]) $(id).addEventListener("change", showShader);
  $("copyShader").onclick = () => navigator.clipboard.writeText($("shaderCode").value);
  $("downloadShader").onclick = () => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([$("shaderCode").value], { type: "text/plain" }));
    link.download = shaderFileName();
    link.click();
    URL.revokeObjectURL(link.href);
  };
  $("blurMode").addEventListener("change", () => requestBlur(true));
  $("follow").addEventListener("change", () => requestBlur(true));
  $("imageFile").addEventListener("change", async () => {
    const file = $("imageFile").files[0];
    if (!file || !state.blur) return;
    try {
      if (file.type.startsWith("video/")) {
        state.imageFile = null;
        await useVideo(URL.createObjectURL(file), file.name);
      } else {
        state.imageFile = file;
        await useImage(await loadImage(file, num("boost")));
      }
    } catch (e) {
      $("blurStats").textContent = e.message;
    }
  });
  $("boost").addEventListener("change", async () => {
    if (state.video) {
      state.video.source.boost = num("boost");
      requestBlur();
    }
    if (state.imageFile && state.source === "image" && state.blur) await useImage(await loadImage(state.imageFile, num("boost")));
  });
  $("sceneButton").onclick = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useImage(proceduralScene(), "procedural");
  };
  $("worldButton").onclick = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useWorld();
  };
  $("videoButton").onclick = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useVideo(SAMPLE_VIDEO.url, SAMPLE_VIDEO.name).catch((e) => ($("blurStats").textContent = e.message));
  };
  $("playPause").onclick = () => setPlaying(!state.playing);
  $("stepBack").onclick = () => stepScene(-1);
  $("stepForward").onclick = () => stepScene(1);
  $("sceneTime").addEventListener("change", () => seekScene(num("sceneTime") || 0));
  $("saveFrame").onclick = () => {
    state.captureNext = true;
    requestBlur();
  };
  $("copyView").onclick = () => {
    navigator.clipboard.writeText(JSON.stringify(viewState(), null, 1));
    $("copyView").textContent = "Copied";
    setTimeout(() => ($("copyView").textContent = "Copy view state"), 1500);
  };
  $("applyView").onclick = async () => {
    const box = $("viewState");
    if (box.hidden) {
      box.hidden = false;
      box.focus();
      return;
    }
    try {
      await applyViewState(JSON.parse(box.value));
      box.hidden = true;
    } catch (e) {
      setStatus(`Could not apply view state: ${e.message}`);
    }
  };
  document.addEventListener("visibilitychange", () => {
    lastBlurTime = 0;
    if (animating()) requestBlur();
  });
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
  showShader();
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
  discardDE();
}

main();
