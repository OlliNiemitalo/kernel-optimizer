import { _setCalibrationState, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { BlurRenderer, shaderSource, tiltShiftRadiusMap } from "./blur.js";
import { viewControls } from "./view-controls.js";
import { World } from "./world.js";
import { Shapes } from "./shapes.js";
import { VideoSource } from "./video.js";

// RGB-D video samples. Video Depth Anything's comparison clips are fetched from its project page
// (served with CORS): color in origin/, inferno-coded relative depth in ours/. Its long-video
// results are single videos with color on the left and depth on the right.
const VDA = "https://videodepthanything.github.io/videos/comparison/";
const VDA_LONG = "https://videodepthanything.github.io/videos/long_videos/";
const VDA_CREDIT =
  'Video and depth: <a href="https://videodepthanything.github.io/">Video Depth Anything</a> project page ' +
  "(S. Chen et al., CVPR 2025), loaded from its site; depth is estimated and relative.";
const TUM_CREDIT =
  'RGB-D video: TUM RGB-D benchmark, sequence fr3/sitting_static (J. Sturm et al., IROS 2012), ' +
  '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; depth holes filled, converted to inverse depth, re-encoded.';
const WOMAN_CREDIT =
  'Video: Yaroslav Shuraev, <a href="https://www.pexels.com/video/woman-standing-in-the-middle-of-green-grass-4511004/">' +
  "Woman standing in the middle of green grass</a> (Pexels license), first 13 s at 1920×1080. Depth: " +
  '<a href="https://videodepthanything.github.io/">Video Depth Anything</a> (S. Chen et al., CVPR 2025) at 15 fps, ' +
  "interpolated to 24 fps in the shader; estimated and relative.";
const VIDEO_SAMPLES = [
  // [color file, depth file if named differently, focus slider position, label]
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
  .concat([
    ["bicycle_wild_1080_depth-720_x3.mp4", "bicycle in the wild, 3× speed", 30],
    ["ParkourFPV-cut_depth-576.mp4", "parkour FPV", 25],
    ["Single-TakeFPVOff-road-1080-cut_depth_x3.mp4", "off-road FPV, 3× speed", 30],
  ].map(([file, name, fps]) => ({
    label: `Video Depth Anything long videos: ${name}`,
    color: VDA_LONG + file,
    depth: null,
    layout: "side",
    fps,
    focus: 0.7,
    credit: VDA_CREDIT,
  })));
// Default: the original footage of Video Depth Anything's 4511004 clip (which is 1280×720 at 15 fps),
// with its depth video also loaded one depth frame ahead, for interpolation.
VIDEO_SAMPLES.unshift({
  label: "Pexels: woman in a field, 1080p 24 fps (Video Depth Anything depth, interpolated)",
  color: `${import.meta.env.BASE_URL}woman-4511004-1080p.mp4`,
  depth: `${import.meta.env.BASE_URL}woman-4511004-vda-depth.mp4`,
  depthFps: 15,
  fps: 24,
  focus: Math.sqrt(1 / 2.4), // 2.4 m in its relative depth units
  credit: WOMAN_CREDIT,
});
VIDEO_SAMPLES.push({
  label: "TUM RGB-D: two people talking (measured depth)",
  color: `${import.meta.env.BASE_URL}tum-sitting-rgbd.mp4`,
  depth: null,
  fps: 30,
  focus: 0.75, // 1.8 m: the people
  credit: TUM_CREDIT,
});
import { DifferentialEvolution } from "./de.js";
import { ProjectedEvolution, workerSolvers } from "./projected-de.js";
import { FineTuner } from "./finetune.js";
import { remezSteps } from "./remez.js";
import { drawConvergence, drawDisc, drawError, drawProfile, populationPreview } from "./plots.js";
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
  copyCapture: null,
  copyBusy: false,
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
  $("centerPopulation").disabled = state.busy;
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
  // The preset menu shows which preset is current, if any.
  const preset = /^article (non-negative )?preset \((\d) components\)$/.exec(source);
  $("preset").value = preset ? `${preset[1] ? "nn" : ""}${preset[2]}` : "";
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
  updateRunButton();
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

function createDE({ centered = false, autoStart = false } = {}) {
  const problem = readProblem();
  describeRanges(problem);
  const D = problem.components * PARAMS_PER_COMPONENT;
  // An implicit Start can keep a loaded/fine-tuned kernel; explicit population buttons
  // always honor their checkbox instead.
  const candidate = (autoStart ? state.include : null) ?? ($("seedWithCurrent").checked ? state.current?.params : null);
  const seedWith = candidate?.length === D ? candidate : null;
  let solvers = null;
  try {
    const options = {
      components: problem.components,
      grid: sampleGrid(problem),
      bounds: bounds(problem),
      populationSize: Math.max(8, Math.round(num("population"))),
      F: num("F"),
      CR: num("CR"),
      seed: Math.round(num("seed")) >>> 0,
      populationCenter: centered ? state.current?.params : null,
      includeCenter: $("seedWithCurrent").checked,
    };
    if ($("projected").checked) solvers = workerSolvers(Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)));
    const de = solvers
      ? new ProjectedEvolution({ ...options, multiplicity: problem.multiplicity, startingPoint: seedWith, solvers })
      : new DifferentialEvolution({ ...options, startingPoint: seedWith && Float32Array.from(seedWith) });
    state.de?.dispose();
    state.de = de;
    state.include = null;
    const sameAsCurrent = (p) => p?.length === state.current?.params.length && p.every((v, j) => v === state.current.params[j]);
    const includesCurrent = centered ? $("seedWithCurrent").checked : sameAsCurrent(seedWith);
    state.refinedIn = includesCurrent && REFINED_SOURCES.has(state.current?.source) ? de : null;
    updateRunButton();
  } catch (e) {
    solvers?.dispose();
    setStatus(`Could not create optimizer: ${e.message}`);
    return false;
  }
  state.history = [];
  state.populationSample = [];
  return true;
}

async function observe(startTime, generationsRun, { rescored = false } = {}) {
  const de = state.de;
  const revision = de.revision;
  let snap;
  try {
    snap = await de.snapshot();
  } catch (e) {
    if (de === state.de && revision === de.revision) setStatus(`Optimizer failed: ${e.message}`);
    return;
  }
  if (de !== state.de || revision !== de.revision) return;
  state.history.push({ gen: de.generation, best: snap.bestCost, mean: snap.meanCost });
  if (state.history.length > 4000) state.history = state.history.filter((_, k) => k % 2 === 0);
  state.populationSample = populationPreview(snap.population, de.D);
  // DE minimizes on the sparser DE grid; a refined kernel in this population stays current until
  // the population beats it on the dense grid.
  const refined = state.refinedIn === de && REFINED_SOURCES.has(state.current?.source)
    ? { ...state.current, rival: maxError(snap.best, denseGrid(readProblem())) } : null;
  const keep = refined && refined.denseError <= refined.rival;
  if (!keep) setCurrent(snap.best, "differential evolution");
  const now = performance.now();
  const rate = generationsRun > 0 ? (generationsRun * de.P) / ((now - startTime) / 1000) : 0;
  setStatus(
    `Optimizer on ${de.solvers ? `CPU (${de.solvers.threads} workers)` : state.backend} · generation ${de.generation} · ` +
      (rate ? `${Math.round(rate).toLocaleString()} evaluations/s · ` : "") +
      `population best ±${snap.bestCost.toFixed(6)}, mean ${snap.meanCost.toFixed(5)}` +
      (keep && refined.rival > refined.denseError ? ` · Showing the ${refined.source} kernel until the population beats its dense-grid ripple ±${refined.denseError.toFixed(6)} (population best there ±${refined.rival.toFixed(6)})` : "") +
      (rescored ? " · Same population re-scored; convergence history restarted for the new objective." : ""),
  );
  drawAll();
}

let runVersion = 0;
const REFINED_SOURCES = new Set(["Remez exchange", "gradient fine-tuning"]);
async function runLoop(version) {
  while (state.running && state.de && version === runVersion) {
    const t0 = performance.now();
    const gens = Math.max(1, Math.round(num("gensPerFrame")));
    state.de.step(gens);
    await observe(t0, gens);
    await nextFrame();
  }
}

function updateRunButton() {
  $("run").textContent = state.running ? "Pause" : state.de?.generation > 0 ? "Continue" : "Start";
}

function setRunning(running) {
  state.running = running;
  updateRunButton();
  const version = ++runVersion;
  if (running) runLoop(version);
}

// ---- gradient fine-tuning -------------------------------------------------------------------

async function fineTune() {
  if (!state.current || state.busy) return;
  const wasRunning = state.running;
  setRunning(false);
  state.busy = true;
  $("finetune").disabled = true;
  $("remez").disabled = true;
  $("centerPopulation").disabled = true;
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
      if (state.de?.D === best.params.length) {
        state.de.inject(best.params);
        state.refinedIn = state.de;
      } else if (!state.de) state.include = best.params;
    }
    state.busy = false;
    $("finetune").disabled = false;
    $("remez").disabled = false;
    $("centerPopulation").disabled = !state.current;
    if (wasRunning) setRunning(true);
  }
}

// ---- Remez exchange refinement ----------------------------------------------------------------

async function remezRefine() {
  if (!state.current || state.busy) return;
  const wasRunning = state.running;
  setRunning(false);
  state.busy = true;
  for (const id of ["remez", "finetune", "centerPopulation"]) $(id).disabled = true;
  const grid = denseGrid(readProblem());
  const start = { params: state.current.params, error: state.current.denseError };
  let best = start;
  const t0 = performance.now();
  try {
    for (const step of remezSteps(start.params, grid, { iterations: Math.max(1, Math.round(num("remezIterations"))) })) {
      if (step.error < best.error) {
        best = step;
        setCurrent(step.params, "Remez exchange");
        drawAll();
      }
      setStatus(
        `Remez exchange · iteration ${step.iteration}` + (step.method ? ` (${step.method})` : "") +
          ` · ${step.exchanges} exchanges · ${((performance.now() - t0) / 1000).toFixed(2)} s · ` +
          `dense-grid ripple ±${best.error.toFixed(6)} (started ±${start.error.toFixed(6)})`,
      );
      await nextFrame();
    }
  } catch (e) {
    setStatus(`Remez exchange failed: ${e.message}`);
  } finally {
    if (best !== start) {
      if (state.de?.D === best.params.length) {
        state.de.inject(best.params);
        state.refinedIn = state.de;
      } else if (!state.de) state.include = best.params;
    }
    state.busy = false;
    $("remez").disabled = $("finetune").disabled = false;
    $("centerPopulation").disabled = !state.current;
    if (wasRunning) setRunning(true);
  }
}

// ---- WebGPU blur demo ---------------------------------------------------------------------

let blurPending = false;
let blurKernelDirty = true;
let blurRendering = false;
let lastBlurTime = 0;

const FRAME = 1 / 30; // seconds per step
const hasDepth = () => state.source === "world" || state.source === "video" || state.source === "shapes";
const animating = () => ["world", "video"].includes(state.source) && state.playing && !document.hidden;

function showSceneTime() {
  if (document.activeElement !== $("sceneTime")) $("sceneTime").value = state.sceneTime.toFixed(4);
  const duration = state.source === "video" ? state.video.element.duration : NaN;
  $("durationOut").textContent = Number.isFinite(duration) ? `/ ${duration.toFixed(2)} s` : "s";
  if (Number.isFinite(duration)) $("seek").max = duration;
  if (!state.seeking) $("seek").value = Number.isFinite(duration) ? state.sceneTime : 0;
}

// Video time is the video element's; the 3D world's advances with the wall clock while playing.
function seekScene(time) {
  state.sceneTime = Math.max(0, time);
  if (state.source === "video") {
    state.video.element.currentTime = state.sceneTime; // renders on "seeked"
    syncDepthVideo(true);
  }
  else requestBlur();
  showSceneTime();
}

function setPlaying(playing) {
  state.playing = playing;
  $("playPause").textContent = playing ? "⏸" : "▶";
  $("playPause").title = playing ? "Pause" : "Play";
  $("playPause").setAttribute("aria-label", $("playPause").title);
  if (state.source === "video") {
    for (const element of videoElements()) {
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

function applyVideoPlayback() {
  for (const element of videoElements()) {
    element.playbackRate = num("playbackRate");
    element.loop = $("loop").checked;
  }
}

// Full screen scales the picture up to fit the screen above the player bar.
function fitFullscreen() {
  const canvas = $("blurCanvas");
  if (document.fullscreenElement !== $("viewport")) {
    canvas.style.width = canvas.style.height = "";
    $("fullscreen").title = "Full screen";
  } else {
    const scale = Math.min((innerWidth - 24) / canvas.width, (innerHeight - 36 - $("player").offsetHeight) / canvas.height);
    canvas.style.width = `${canvas.width * scale}px`;
    canvas.style.height = `${canvas.height * scale}px`;
    $("fullscreen").title = "Exit full screen";
  }
  $("fullscreen").setAttribute("aria-label", $("fullscreen").title);
}

function setFocusSlider(value) {
  $("focus").value = value;
  $("focus").dispatchEvent(new Event("input"));
}

// Hovering (or holding a touch) on the canvas eases the focus to the nearest depth around the
// pointer, and restores it afterwards; a click or tap keeps the picked focus.
const PICK_RADIUS = 8; // image pixels
const FOCUS_EASE = 0.12; // time constant, s
const hover = { active: false, x: 0, y: 0, saved: 0, target: 0, value: 0, picking: false, easing: false };

const canPick = () => state.blur && $("blurOn").checked && $("blurMode").value === "depth" && $("showLayers").value !== "6" && hasDepth();

function canvasPixel(e) {
  const rect = $("blurCanvas").getBoundingClientRect();
  return [
    Math.floor(((e.clientX - rect.left) / rect.width) * state.blur.width),
    Math.floor(((e.clientY - rect.top) / rect.height) * state.blur.height),
  ];
}

// Focus slider position (its square is 1 / distance) for the nearest depth around a pixel.
async function pickFocus(x, y) {
  const inv = await state.blur.pickInverseDepth(x, y, PICK_RADIUS);
  return inv === null ? null : Math.sqrt(Math.min(1, Math.max(0, inv)));
}

async function easeFocus() {
  if (hover.easing) return;
  hover.easing = true;
  hover.value = num("focus");
  let last = performance.now();
  while (hover.active || hover.value !== hover.target) {
    await nextFrame();
    if (hover.active && !hover.picking) {
      hover.picking = true;
      pickFocus(hover.x, hover.y)
        .then((v) => { if (v !== null && hover.active) hover.target = v; })
        .finally(() => (hover.picking = false));
    }
    const now = performance.now();
    const step = 1 - Math.exp(-(now - last) / 1000 / FOCUS_EASE);
    last = now;
    const next = Math.abs(hover.target - hover.value) < 1e-4 ? hover.target : hover.value + (hover.target - hover.value) * step;
    if (next !== hover.value) setFocusSlider((hover.value = next));
  }
  hover.easing = false;
}

function startHover(e) {
  if (!canPick() || (drag.down && drag.moved)) return;
  [hover.x, hover.y] = canvasPixel(e);
  if (!hover.active) {
    hover.active = true;
    hover.saved = hover.target = num("focus");
  }
  easeFocus();
}

function endHover() {
  if (!hover.active) return;
  hover.active = false;
  hover.target = hover.saved;
  easeFocus();
}

// Dragging on the tetra scene turns the view; a drag is neither a focus hover nor a click.
const DRAG_START = 4; // CSS pixels of movement before a press becomes a drag
const drag = { down: false, moved: false, x: 0, y: 0 };

function startDrag(e) {
  Object.assign(drag, { down: state.source === "shapes" && e.button === 0, moved: false, x: e.clientX, y: e.clientY });
}

function moveDrag(e) {
  if (!drag.down) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < DRAG_START) return;
  if (!drag.moved) {
    drag.moved = true;
    $("blurCanvas").setPointerCapture(e.pointerId);
    endHover();
  }
  drag.x = e.clientX;
  drag.y = e.clientY;
  state.shapes.rotate(dx, dy);
  requestBlur();
}

function endDrag() {
  drag.down = false;
}

// The Scale slider (log2) and the wheel scale the tetrahedra about their center.
function scaleShapes(factor) {
  state.shapes.zoom(factor);
  showShapesScale();
  requestBlur();
}

function showShapesScale() {
  const scale = state.shapes?.scale ?? 1;
  if (document.activeElement !== $("shapesScale")) $("shapesScale").value = Math.log2(scale);
  $("shapesScaleOut").textContent = `${scale.toPrecision(3)}×`;
}

function wheelScale(e) {
  if (state.source !== "shapes") return;
  e.preventDefault();
  scaleShapes(Math.exp(e.deltaY * 0.001));
}

async function clickFocus(e) {
  if (!canPick() || drag.moved) return;
  const v = await pickFocus(...canvasPixel(e));
  if (v === null) return;
  hover.saved = hover.target = v;
  easeFocus();
}

function download(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
}

const VIEW_CONTROLS = [
  "blurOn", "radius", "blurMode", "focus", "layers", "sizeBlur", "sizeBrightness", "occlusion", "occlusionLevels",
  "exposure", "tonemap", "precision", "transition", "radiusInterpolation", "showLayers", "playbackRate", "loop",
];

// Everything needed to re-render the displayed frame, plus context for bug reports.
function viewState() {
  const controls = Object.fromEntries(VIEW_CONTROLS.map((id) => [id, $(id).type === "checkbox" ? $(id).checked : $(id).value]));
  return {
    scene:
      state.source === "video" ? `video: ${state.video.name}`
      : state.source === "image" ? `image: ${state.imageFile?.name}`
      : state.source,
    ...(state.source === "shapes" && { camera: { yaw: state.shapes.yaw, pitch: state.shapes.pitch, scale: state.shapes.scale } }),
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
  if (view.scene === "shapes" && state.source !== "shapes") useShapes();
  if (view.scene === "shapes" && view.camera) {
    const { yaw, pitch, scale } = view.camera;
    Object.assign(state.shapes, { yaw: Number(yaw) || 0, pitch: Number(pitch) || 0, scale: Number(scale) || 1 });
  }
  if (view.scene === "procedural") useImage(proceduralScene(), "procedural");
  const sample = VIDEO_SAMPLES.find((s) => view.scene === `video: ${s.label}`);
  if (sample && state.video?.name !== sample.label) await useVideo(sample);
  for (const [id, value] of Object.entries(view.controls ?? {})) {
    if (!VIEW_CONTROLS.includes(id) || id === "showLayers") continue;
    // Older view states have a layer count.
    if (id === "layers" && typeof value !== "boolean") $(id).checked = Number(value) > 1;
    else if ($(id).type === "checkbox") $(id).checked = value;
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
  // Validate diagnostics only after their prerequisites, regardless of JSON property order.
  if (view.controls?.showLayers !== undefined) $("showLayers").value = view.controls.showLayers;
  updateVaryingControls();
  seekScene(Number(view.time) || 0);
  lastBlurTime = 0;
  requestBlur(true);
}

function updateVaryingControls() {
  $("sizeBrightnessOut").textContent = num("sizeBrightness").toFixed(1);
  $("occlusionOut").textContent = `${(num("occlusionLevels") / 6).toFixed(2)} oct`;
  const controls = viewControls({
    ready: !!state.blur && !!state.source, source: state.source,
    blur: $("blurOn").checked, mode: $("blurMode").value,
    layers: $("layers").checked, sizeBlur: $("sizeBlur").checked,
    occlusion: $("occlusion").checked, showLayers: $("showLayers").value,
  });
  for (const [id, disabled] of Object.entries(controls.disabled)) $(id).disabled = disabled;
  $("playerBlur").checked = $("blurOn").checked;
  $("playerBlur").disabled = $("blurOn").disabled;
  const clipboardSupported = !!navigator.clipboard?.write && typeof ClipboardItem !== "undefined";
  $("copyImage").disabled = !state.blur || !state.source || !clipboardSupported || state.copyBusy;
  $("copyImage").title = clipboardSupported
    ? "Copy a freshly rendered PNG of the current view to the clipboard"
    : "Image clipboard is unavailable in this browser or context; use Save PNG";
  $("blurMode").querySelector('option[value="depth"]').disabled = !controls.depthAvailable;
  for (const option of $("showLayers").options) option.disabled = !controls.layerViews.includes(option.value);
  $("showLayers").value = controls.view;
}

function requestBlur(kernelChanged = false) {
  blurKernelDirty ||= kernelChanged;
  updateVaryingControls();
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
  if (!state.blur || !state.current || !state.source) return;
  blurRendering = true;
  const copyCapture = state.copyCapture;
  state.copyCapture = null;
  try {
    const radius = num("radius");
    if (blurKernelDirty) {
      state.blur.setKernel(state.current.params, radius, readProblem().transition);
      const { width, height } = state.blur;
      const mode = $("blurMode").value;
      const depth = mode === "depth" && hasDepth();
      if (depth) state.blur.setDepthOfField();
      else state.blur.setRadiusMap(mode === "tiltShift" ? tiltShiftRadiusMap(width, height, radius) : null);
      showShapesScale();
      updateVaryingControls();
      fitFullscreen();
      blurKernelDirty = false;
    }
    const now = performance.now();
    if (state.source === "video") {
      syncDepthVideo();
      state.sceneTime = state.video.element.currentTime;
    }
    else if (animating() && lastBlurTime) state.sceneTime += Math.min(0.1, (now - lastBlurTime) / 1000) * num("playbackRate");
    showSceneTime();
    const saveCapture = state.captureNext;
    state.captureNext = false;
    const capture = saveCapture || !!copyCapture;
    const r = await state.blur.render({
      exposure: Math.pow(2, num("exposure")),
      tonemap: Number($("tonemap").value),
      precision: $("precision").value,
      interpolate: $("radiusInterpolation").checked,
      showSource: !$("blurOn").checked,
      time: state.sceneTime,
      invFocus: num("focus") ** 2,
      sizeBlur: $("sizeBlur").checked ? num("sizeBrightness") : null,
      occlusion: $("occlusion").checked ? num("occlusionLevels") / 6 : 0,
      layers: $("layers").checked ? 2 : 1,
      showLayers: Number($("showLayers").value),
      capture,
    });
    const prefix = { world: "street", video: "video", shapes: "tetras" }[state.source];
    if (r.png && saveCapture) download(r.png, prefix ? `${prefix}-t${state.sceneTime.toFixed(4)}s.png` : "blur.png");
    if (copyCapture) {
      if (!r.png) throw new Error("The renderer did not produce an image.");
      copyCapture.resolve(await (await fetch(r.png)).blob());
    }
    lastBlurTime = now;
    const gpu = r.gpuMs
      ? "GPU time: " + Object.entries(r.gpuMs).map(([k, ms]) => `${k} ${ms.toFixed(2)} ms`).join(" + ")
      : `${r.ms.toFixed(1)} ms round trip`;
    const mib = (b) => (b / 2 ** 20).toFixed(1);
    const m = r.memory;
    const total = Object.values(m).reduce((a, b) => a + b, 0);
    $("blurStats").textContent =
      ($("blurOn").checked ? "" : "Blur off · ") +
      `${state.blur.width}×${state.blur.height} px · ${state.blur.channels} separable channels × ` +
      (r.varying ? `up to ${r.taps} taps (${$("radiusInterpolation").checked ? "interpolated radii" : "nearest radius"})` : `${r.taps} taps`) + ` per pass · ` +
      `GPU memory ${mib(total)} MiB (intermediate ${mib(m.intermediate)}, accumulator ${mib(m.accumulator)}, ` +
      `source ${mib(m.source)}` +
      (m.radiusMap ? `, radius ${mib(m.radiusMap)}, working maps ${mib(m.levelMaps)}` : "") +
      `, kernel tables ${mib(m.tables)}` +
      (m.layers ? `, layers ${mib(m.layers)}` : "") +
      `) · ${gpu}`;
  } catch (e) {
    copyCapture?.reject(e);
    console.error(e);
    $("blurStats").textContent = `Blur failed: ${e.message}`;
  } finally {
    blurRendering = false;
    if (animating()) requestBlur();
  }
}

function stopVideo() {
  state.scene = (state.scene ?? 0) + 1; // invalidates video loads in progress
  if (!state.video) return;
  for (const element of videoElements()) {
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

function useShapes() {
  stopVideo();
  state.shapes ??= new Shapes(state.blur.device);
  state.blur.setAnimated(state.shapes, 960, 600);
  state.source = "shapes";
  $("blurMode").value = "depth";
  setFocusSlider(Math.sqrt(1 / state.shapes.distance)); // the middle tetra
  requestBlur(true);
}

async function loadVideo(url) {
  const element = document.createElement("video");
  Object.assign(element, { crossOrigin: "anonymous", muted: true, loop: true, playsInline: true, preload: "auto", src: url });
  await new Promise((resolve, reject) => {
    element.onloadeddata = () => (element.videoWidth ? resolve() : reject(new Error(`The video ${url} has no picture.`)));
    element.onerror = () => reject(new Error(`Could not load the video ${url}.`));
  });
  element.addEventListener("seeked", () => requestBlur());
  element.addEventListener("ended", () => {
    if (state.video?.element === element) setPlaying(false);
  });
  return element;
}

// An RGB-D video sample (see VIDEO_SAMPLES and video.js): stacked color over depth, or a color
// video with a separate depth video.
async function useVideo(sample) {
  stopVideo();
  const scene = state.scene;
  $("blurStats").textContent = `Loading ${sample.label}…`;
  const [element, depth, depthNext] = await Promise.all([
    loadVideo(sample.color),
    sample.depth ? loadVideo(sample.depth) : null,
    sample.depthFps ? loadVideo(sample.depth) : null,
  ]);
  if (scene !== state.scene) {
    for (const e of [element, depth, depthNext]) e?.removeAttribute("src");
    return;
  }
  const source = new VideoSource(state.blur.device, element, depth, { depthNext, depthFps: sample.depthFps, layout: sample.layout });
  source.boost = num("boost");
  state.video = { element, depth, depthNext, source, name: sample.label, fps: sample.fps ?? 30, depthFps: sample.depthFps };
  applyVideoPlayback();
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

const videoElements = () => [state.video?.element, state.video?.depth, state.video?.depthNext].filter(Boolean);

// The depth video follows the color video; while playing they drift apart by up to a frame or so.
// A second depth element (for interpolation) runs one depth frame ahead. force: seek exactly.
function syncDepthVideo(force = false) {
  const { element, depth, depthNext, fps, depthFps } = state.video;
  const tolerance = (element.paused ? 0.25 : 3) / fps;
  for (const [e, offset] of [[depth, 0], [depthNext, 1 / depthFps]]) {
    if (!e) continue;
    const end = e.duration || Infinity;
    const target = element.loop ? (element.currentTime + offset) % end : Math.min(element.currentTime + offset, end);
    if (force || Math.abs(e.currentTime - target) > tolerance) e.currentTime = target;
  }
}

async function initBlur() {
  try {
    state.blur = await BlurRenderer.create($("blurCanvas"));
  } catch (e) {
    $("blurStats").textContent = `WebGPU blur demo unavailable: ${e.message}`;
    return;
  }
  // The default video is loaded from another site; without it, start with the 3D world.
  await useVideo(VIDEO_SAMPLES[0]).catch(() => state.source || useWorld());
}

// ---- wiring ---------------------------------------------------------------------------------

const shaderPacked = () => $("precision").value === "f16";

function shaderFileName() {
  return $("shaderKind").value === "display" ? "blur-display.wgsl" : `blur-compute-${shaderPacked() ? "packed" : "f32"}.wgsl`;
}

function showShader() {
  $("shaderCode").value = shaderSource($("shaderKind").value, shaderPacked(), $("radiusInterpolation").checked);
}

function wire() {
  $("run").onclick = () => {
    if (!state.running && !state.de && !createDE({ autoStart: true })) return;
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
  $("centerPopulation").onclick = async () => {
    if (!state.current || state.busy || $("centerPopulation").disabled) return;
    const wasRunning = state.running;
    setRunning(false);
    $("centerPopulation").disabled = true;
    try {
      if (!createDE({ centered: true })) return;
      await observe(performance.now(), 0);
      if (wasRunning) setRunning(true);
    } finally {
      $("centerPopulation").disabled = state.busy || !state.current;
    }
  };
  for (const id of ["transition", "passSamples", "stopSamples", "frequencyBands", "negGain"]) {
    $(id).addEventListener("change", async () => {
      setRunning(false);
      const problem = readProblem();
      describeRanges(problem);
      if (state.current) setCurrent(state.current.params, state.current.source);
      if (id === "transition") requestBlur(true);
      if (state.de) {
        try {
          state.de.updateProblem(sampleGrid(problem), bounds(problem));
          // Old costs are not comparable after changing the objective.
          state.history = [];
          await observe(performance.now(), 0, { rescored: true });
        } catch (e) {
          setStatus(`Could not update optimizer: ${e.message}`);
        }
      } else drawAll();
    });
  }
  for (const id of ["components", "multiplicity", "population", "F", "CR", "seed", "projected"]) {
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
  };
  $("finetune").onclick = fineTune;
  $("remez").onclick = remezRefine;
  $("ftLr").addEventListener("input", () => delete $("ftLr").dataset.auto);
  // Space plays or pauses, unless a control that uses the key has focus.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat || e.ctrlKey || e.metaKey || e.altKey || $("playPause").disabled) return;
    if (e.target.closest?.("input, textarea, select, button, summary, [contenteditable]")) return;
    e.preventDefault();
    setPlaying(!state.playing);
  });
  // Exact weights need far fewer, costlier members than the GPU search, unless the user chose a size.
  $("population").addEventListener("input", () => delete $("population").dataset.auto);
  $("projected").addEventListener("change", () => {
    if ($("population").dataset.auto) $("population").value = $("projected").checked ? 32 : 512;
  });
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
  for (const type of ["pointerdown", "pointermove"]) $("blurCanvas").addEventListener(type, startHover);
  for (const type of ["pointerleave", "pointercancel"]) $("blurCanvas").addEventListener(type, endHover);
  $("blurCanvas").addEventListener("pointerdown", startDrag);
  $("blurCanvas").addEventListener("pointermove", moveDrag);
  for (const type of ["pointerup", "pointercancel"]) $("blurCanvas").addEventListener(type, endDrag);
  $("blurCanvas").addEventListener("wheel", wheelScale, { passive: false });
  $("shapesScale").addEventListener("input", () => {
    if (state.source === "shapes") scaleShapes(2 ** num("shapesScale") / state.shapes.scale);
  });
  $("blurCanvas").addEventListener("click", clickFocus);
  $("exposure").addEventListener("input", () => {
    $("exposureOut").textContent = `${Math.pow(2, num("exposure")).toFixed(2)}×`;
    requestBlur();
  });
  for (const id of ["blurOn", "layers", "showLayers", "sizeBlur", "sizeBrightness", "occlusion", "occlusionLevels", "radiusInterpolation"]) {
    $(id).addEventListener("input", () => {
      updateVaryingControls();
      requestBlur();
    });
  }
  $("playerBlur").addEventListener("input", () => {
    $("blurOn").checked = $("playerBlur").checked;
    $("blurOn").dispatchEvent(new Event("input"));
  });
  for (const id of ["tonemap", "precision"]) $(id).addEventListener("change", () => requestBlur());
  for (const id of ["precision", "shaderKind"]) $(id).addEventListener("change", showShader);
  $("radiusInterpolation").addEventListener("input", showShader);
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
  $("shapesButton").onclick = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useShapes();
  };
  const groups = {};
  for (const [k, sample] of VIDEO_SAMPLES.entries()) {
    const [, group, name] = sample.label.match(/^(.*?): (.*)$/);
    if (!groups[group]) $("videoSample").append((groups[group] = Object.assign(document.createElement("optgroup"), { label: group })));
    groups[group].append(new Option(name, k));
  }
  const loadSample = () => {
    state.imageFile = null;
    $("imageFile").value = "";
    if (state.blur) useVideo(VIDEO_SAMPLES[num("videoSample")]).catch((e) => ($("blurStats").textContent = e.message));
  };
  $("videoButton").onclick = loadSample;
  $("videoSample").addEventListener("change", loadSample);
  $("playPause").onclick = () => setPlaying(!state.playing);
  $("toStart").onclick = () => seekScene(0);
  $("seek").addEventListener("pointerdown", () => (state.seeking = true));
  $("seek").addEventListener("pointerup", () => (state.seeking = false));
  $("seek").addEventListener("change", () => (state.seeking = false));
  $("seek").addEventListener("input", () => seekScene(num("seek")));
  $("playbackRate").addEventListener("input", applyVideoPlayback);
  $("loop").addEventListener("input", applyVideoPlayback);
  const toggleFullscreen = () =>
    document.fullscreenElement ? document.exitFullscreen() : $("viewport").requestFullscreen().catch(() => {});
  $("fullscreen").onclick = toggleFullscreen;
  $("blurCanvas").addEventListener("dblclick", toggleFullscreen);
  document.addEventListener("fullscreenchange", fitFullscreen);
  addEventListener("resize", fitFullscreen);
  $("stepBack").onclick = () => stepScene(-1);
  $("stepForward").onclick = () => stepScene(1);
  $("sceneTime").addEventListener("change", () => seekScene(num("sceneTime") || 0));
  $("saveFrame").onclick = () => {
    state.captureNext = true;
    requestBlur();
  };
  $("copyImage").onclick = async () => {
    if (state.copyBusy || $("copyImage").disabled) return;
    state.copyBusy = true;
    $("captureStatus").textContent = "Copying image…";
    let capture;
    const png = new Promise((resolve, reject) => { capture = { resolve, reject }; });
    // Some browsers require clipboard.write during the click's user activation. Supply a
    // promise for the freshly rendered PNG rather than waiting before requesting clipboard access.
    png.catch(() => {});
    state.copyCapture = capture;
    try {
      const writing = navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
      requestBlur();
      await writing;
      $("captureStatus").textContent = "Image copied.";
    } catch (error) {
      if (state.copyCapture === capture) state.copyCapture = null;
      capture.reject(error);
      $("captureStatus").textContent = "Could not copy image. Allow clipboard access or use Save PNG.";
    } finally {
      state.copyBusy = false;
      updateVaryingControls();
    }
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
  if (repo) for (const a of document.querySelectorAll("a[data-readme]")) a.href = repo + a.dataset.readme;
  wire();
  updateVaryingControls();
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
