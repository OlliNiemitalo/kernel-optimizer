import { _setCalibrationState, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { BlurRenderer, shaderSource, tiltShiftRadiusMap } from "./blur.js";
import { World } from "./world.js";
import { VideoSource } from "./video.js";

// RGB-D video samples. Video Depth Anything's comparison clips are fetched from its project page
// (served with CORS): color in origin/, inferno-coded relative depth in ours/.
const VDA = "https://videodepthanything.github.io/videos/comparison/";
const VDA_CREDIT =
  'Video and depth: <a href="https://videodepthanything.github.io/">Video Depth Anything</a> project page ' +
  "(S. Chen et al., CVPR 2025), loaded from its site; depth is estimated and relative.";
const TUM_CREDIT =
  'RGB-D video: TUM RGB-D benchmark, sequence fr3/sitting_static (J. Sturm et al., IROS 2012), ' +
  '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; depth holes filled, converted to inverse depth, re-encoded.';
const VIDEO_SAMPLES = [
  // [color file, depth file if named differently, focus slider position, label]
  ["4511004-uhd_3840_2160_24fps_rgb.mp4", null, 0.675, "woman in a field"], // focused on her: v ≈ 0.455
  ["1753029-hd_1920_1080_30fps.mp4"],
  ["4158877-uhd_3840_2160_30fps_rgb.mp4"],
  ["Balloons-32132.mp4"],
  ["Istanbul-26920.mp4"],
  ["Tokyo-Walk_rgb.mp4"],
  ["davis_burnout.mp4"],
  ["davis_rollercoaster.mp4"],
  ["example_5473765-l.mp4"],
  ["face_3.mp4"],
  ["obj_1.mp4"],
  ["sheep_cut1.mp4"],
  ["vdw_000423.mp4"],
  ["2Dto3D_Test2.mp4"],
  ["hulisongputao-cut1-1-1_input.mp4", "hulisongputao-cut1-1-1_vis.mp4"],
  ["yubangxiangzheng-cut3-1_input.mp4", "yubangxiangzheng-cut3-1_vis.mp4"],
]
  .map(([file, depth, focus = 0.7, description]) => {
    const name = file.replace(/\.mp4$/, "").replace(/(-u?hd_\d+_\d+_\d+fps)?(_rgb|_input)?$/, "");
    return {
      label: `Video Depth Anything: ${description ? `${name}, ${description}` : name}`,
      color: VDA + "origin/" + file,
      depth: VDA + "ours/" + (depth ?? file),
      fps: Number(/(\d+)fps/.exec(file)?.[1]) || 30,
      focus,
      credit: VDA_CREDIT,
    };
  })
  .concat([{
    label: "TUM RGB-D: two people talking (measured depth)",
    color: `${import.meta.env.BASE_URL}tum-sitting-rgbd.mp4`,
    depth: null,
    fps: 30,
    focus: 0.75, // 1.8 m: the people
    credit: TUM_CREDIT,
  }]);
import { DifferentialEvolution } from "./de.js";
import { FineTuner } from "./finetune.js";
import { drawConvergence, drawDisc, drawError, drawProfile } from "./plots.js";
import {
  PARAMS_PER_COMPONENT, bounds, centerFrequency, channelCount, formatParams, maxError, multiplicities, parseParams,
  presetParams, nonnegativePresetParams, PRESETS, sampleGrid, sortByFrequency,
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
    negativeWeight: Math.max(1, num("negGain") || 1),
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
  drawDisc($("disc"), best, xmax);
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
  if (state.source === "video") {
    state.video.element.currentTime = state.sceneTime; // renders on "seeked"
    if (state.video.depth) state.video.depth.currentTime = state.sceneTime;
  }
  else requestBlur();
  showSceneTime();
}

function setPlaying(playing) {
  state.playing = playing;
  $("playPause").textContent = playing ? "Pause" : "Play";
  if (state.source === "video") {
    for (const element of [state.video.element, state.video.depth]) {
      if (!element) continue;
      if (playing) element.play().catch(() => {});
      else element.pause();
    }
  }
  lastBlurTime = 0;
  requestBlur();
}

function stepScene(frames) {
  setPlaying(false);
  // Video frames are addressed at their middle, away from the boundaries between frames.
  const offset = state.source === "video" ? 0.5 : 0;
  const frame = state.source === "video" ? 1 / state.video.fps : FRAME;
  seekScene((Math.round(state.sceneTime / frame - offset + frames) + offset) * frame);
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
  const sample = VIDEO_SAMPLES.find((s) => view.scene === `video: ${s.label}`);
  if (sample && state.video?.name !== sample.label) await useVideo(sample);
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
    if (state.source === "video") {
      syncDepthVideo();
      state.sceneTime = state.video.element.currentTime;
    }
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
  for (const element of [state.video.element, state.video.depth]) {
    if (!element) continue;
    element.pause();
    element.removeAttribute("src");
    element.load();
  }
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

async function loadVideo(url) {
  const element = document.createElement("video");
  Object.assign(element, { crossOrigin: "anonymous", muted: true, loop: true, playsInline: true, preload: "auto", src: url });
  await new Promise((resolve, reject) => {
    element.onloadeddata = resolve;
    element.onerror = () => reject(new Error(`Could not load the video ${url}.`));
  });
  element.addEventListener("seeked", () => requestBlur());
  return element;
}

// An RGB-D video sample (see VIDEO_SAMPLES and video.js): stacked color over depth, or a color
// video with a separate depth video.
async function useVideo(sample) {
  stopVideo();
  $("blurStats").textContent = `Loading ${sample.label}…`;
  const [element, depth] = await Promise.all([loadVideo(sample.color), sample.depth ? loadVideo(sample.depth) : null]);
  const source = new VideoSource(state.blur.device, element, depth);
  source.boost = num("boost");
  state.video = { element, depth, source, name: sample.label, fps: sample.fps ?? 30 };
  $("videoCredit").innerHTML = sample.credit ?? "";
  $("videoCredit").hidden = !sample.credit;
  state.blur.setAnimated(source, source.width, source.height);
  state.source = "video";
  state.sceneTime = 0;
  $("blurMode").value = "depth";
  if (sample.focus !== undefined) setFocusSlider(sample.focus);
  setPlaying(state.playing);
  requestBlur(true);
}

// The depth video follows the color video; while playing they drift apart by up to a frame or so.
function syncDepthVideo() {
  const { element, depth, fps } = state.video;
  if (depth && Math.abs(depth.currentTime - element.currentTime) > (element.paused ? 0.25 : 3) / fps) {
    depth.currentTime = element.currentTime;
  }
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
  for (const id of ["components", "multiplicity", "transition", "passSamples", "stopSamples", "frequencyBands", "negGain", "population", "F", "CR", "seed"]) {
    // The component count reacts while typing or stepping, not only when the field loses focus.
    $(id).addEventListener(id === "components" ? "input" : "change", () => {
      if (id === "components" && !(num("components") >= 1)) return;
      setRunning(false);
      if (id === "multiplicity") {
        const count = ($("multiplicity").value.match(/\d+/g) ?? []).length;
        if (count >= 1 && count <= 8) $("components").value = count;
      }
      const problem = readProblem();
      setMultiplicity(problem.multiplicity);
      const shapeChanged =
        state.current &&
        (state.current.params.length !== problem.components * PARAMS_PER_COMPONENT ||
          multiplicities(state.current.params).join() !== problem.multiplicity.join());
      // A new structure shows the article's kernel for it when there is one, so the plots follow.
      if (shapeChanged && problem.multiplicity.every((m) => m === 1) && PRESETS[problem.components]) {
        setCurrent(sortByFrequency(presetParams(problem.components)), `article preset (${problem.components} components)`);
        discardDE({ include: state.current.params });
        return;
      }
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
    $("negGain").value = nonnegative ? NONNEGATIVE_WEIGHT : 1;
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
  for (const id of ["xmax", "showPopulation", "showComponents"]) $(id).addEventListener("input", drawAll);
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
        await useVideo({ label: file.name, color: URL.createObjectURL(file), depth: null });
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
  for (const [k, sample] of VIDEO_SAMPLES.entries()) $("videoSample").add(new Option(sample.label, k));
  const loadSample = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useVideo(VIDEO_SAMPLES[num("videoSample")]).catch((e) => ($("blurStats").textContent = e.message));
  };
  $("videoButton").onclick = loadSample;
  $("videoSample").addEventListener("change", loadSample);
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
