import assert from "node:assert/strict";
import { test } from "node:test";
import { loadVideo, mediaError, playVideo } from "../src/video-playback.js";
import { VideoSource } from "../src/video.js";

class FakeVideo extends EventTarget {
  videoWidth = 640;
  videoHeight = 360;
  readyState = 2;
  networkState = 1;
  pauses = 0;
  loads = 0;
  pause() { this.pauses++; }
  load() { this.loads++; }
  removeAttribute(name) { delete this[name]; }
}

function setup(t) {
  const videos = [];
  const original = globalThis.document;
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, "video");
      const video = new FakeVideo();
      videos.push(video);
      return video;
    },
  };
  t.after(() => {
    if (original === undefined) delete globalThis.document;
    else globalThis.document = original;
  });
  return videos;
}

test("loaded videos keep runtime errors visible and release resources on abort", async (t) => {
  const videos = setup(t);
  const controller = new AbortController();
  const errors = [];
  let seeks = 0;
  const promise = loadVideo("color.mp4", {
    signal: controller.signal,
    onError: (error) => errors.push(error),
    onSeek: () => seeks++,
  });
  const video = videos[0];
  assert.equal(video.crossOrigin, "anonymous");
  assert.equal(video.muted, true);
  assert.equal(video.playsInline, true);
  video.dispatchEvent(new Event("loadeddata"));
  assert.equal(await promise, video);
  video.dispatchEvent(new Event("seeked"));
  assert.equal(seeks, 1);
  video.error = { code: 3, message: "Decoder exhausted" };
  video.dispatchEvent(new Event("error"));
  assert.match(errors[0].message, /MEDIA_ERR_DECODE.*Decoder exhausted.*color\.mp4/);
  controller.abort();
  assert.equal(video.src, undefined);
  assert.equal(video.pauses, 1);
  video.dispatchEvent(new Event("error"));
  video.dispatchEvent(new Event("seeked"));
  assert.equal(errors.length, 1);
  assert.equal(seeks, 1);
});

test("media errors reject initial loading with browser code, details, and URL", async (t) => {
  const videos = setup(t);
  for (const [code, name] of [[1, "ABORTED"], [2, "NETWORK"], [3, "DECODE"], [4, "SRC_NOT_SUPPORTED"]]) {
    const promise = loadVideo(`depth-${code}.mp4`);
    const video = videos.at(-1);
    video.error = { code, message: "Browser details" };
    video.dispatchEvent(new Event("error"));
    await assert.rejects(promise, new RegExp(`MEDIA_ERR_${name}.*Browser details.*depth-${code}\\.mp4`));
    assert.equal(video.src, undefined);
    assert.equal(video.pauses, 1);
  }
});

test("a stalled load times out rather than waiting forever", async (t) => {
  const videos = setup(t);
  await assert.rejects(loadVideo("stalled.mp4", { timeoutMs: 5 }), /Timed out.*stalled\.mp4.*readyState/);
  assert.equal(videos[0].src, undefined);
  assert.equal(videos[0].pauses, 1);
});

test("changing scenes cancels pending and already loaded members of a video pair", async (t) => {
  const videos = setup(t);
  const controller = new AbortController();
  const color = loadVideo("color.mp4", { signal: controller.signal });
  const depth = loadVideo("depth.mp4", { signal: controller.signal });
  videos[0].dispatchEvent(new Event("loadeddata"));
  await color;
  const cancelled = assert.rejects(depth, { name: "AbortError" });
  controller.abort();
  await cancelled;
  assert.ok(videos.every((video) => !video.src && video.pauses === 1));
});

test("already cancelled loads do not start a request", async (t) => {
  const videos = setup(t);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(loadVideo("unused.mp4", { signal: controller.signal }), { name: "AbortError" });
  assert.equal(videos[0].src, undefined);
});

test("a decoded frame without a picture is rejected", async (t) => {
  const videos = setup(t);
  const promise = loadVideo("empty.mp4");
  videos[0].videoWidth = 0;
  videos[0].dispatchEvent(new Event("loadeddata"));
  await assert.rejects(promise, /no picture.*empty\.mp4/);
  assert.equal(videos[0].src, undefined);
});

test("unknown browser media errors still include useful state", () => {
  assert.match(mediaError(new FakeVideo(), "unknown.mp4").message, /Unknown media error.*unknown\.mp4.*640x360.*readyState: 2/);
});

test("playback policy and decoder errors are reported, intentional interruptions are not", async () => {
  const errors = [];
  for (const name of ["NotAllowedError", "NotSupportedError", "AbortError"]) {
    await playVideo({ play: () => Promise.reject(new DOMException("Cannot play", name)) }, (error) => errors.push(error));
  }
  assert.deepEqual(errors.map((error) => error.name), ["NotAllowedError", "NotSupportedError"]);
  await playVideo({ play: () => Promise.resolve() }, (error) => errors.push(error));
  assert.equal(errors.length, 2);
});

test("frame callback failures reach the UI handler and cancel all frame callbacks", (t) => {
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  });
  const element = () => ({
    currentTime: 0,
    requestVideoFrameCallback(callback) { this.callback = callback; return 42; },
    cancelVideoFrameCallback(id) { assert.equal(id, 42); this.cancelled = true; },
  });
  const errors = [];
  const size = { destroy() { this.destroyed = true; } };
  const source = Object.assign(Object.create(VideoSource.prototype), {
    device: {
      createShaderModule: () => ({}),
      createComputePipeline: () => ({}),
      createBuffer: () => size,
      queue: { writeBuffer() {} },
    },
    color: element(),
    depth: element(),
    fps: 30,
    depthFps: 30,
    width: 640,
    height: 360,
    frameCallbacks: new Map(),
    onError: (error) => errors.push(error),
    ingest() {},
  });
  source.holdFrames();
  source.ingest = () => { throw new Error("External texture import failed"); };
  source.color.callback(0, { mediaTime: 1 });
  assert.equal(source.destroyed, true);
  assert.equal(source.color.cancelled, true);
  assert.equal(source.depth.cancelled, true);
  assert.equal(size.destroyed, true);
  assert.equal(source.frameCallbacks.size, 0);
  assert.match(errors[0].message, /External texture import failed/);
  source.depth.callback(0, { mediaTime: 1 });
  assert.equal(errors.length, 1);
  source.destroy();
});

test("paired videos import only presented frames and wait for both caches before rendering", (t) => {
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  });
  const imported = [];
  const source = Object.assign(Object.create(VideoSource.prototype), {
    device: {
      createShaderModule: () => ({}),
      createComputePipeline: () => ({}),
      createBuffer: () => ({ destroy() {} }),
      queue: { writeBuffer() {} },
    },
    color: {
      paused: true,
      time: 0,
      seeks: 0,
      get currentTime() { return this.time; },
      set currentTime(value) { this.time = value; this.seeks++; },
      requestVideoFrameCallback(callback) { this.callback = callback; return 1; },
      cancelVideoFrameCallback() {},
    },
    depth: {
      paused: false,
      currentTime: 0,
      requestVideoFrameCallback(callback) { this.callback = callback; return 2; },
      cancelVideoFrameCallback() {},
    },
    fps: 24,
    depthFps: 15,
    width: 640,
    height: 360,
    frameCallbacks: new Map(),
    ingest(element, kind, key) {
      imported.push({ element, kind, key });
      this.frames[kind].push({ key, buffer: { destroy() {} } });
    },
  });
  source.holdFrames();
  assert.deepEqual(imported, [], "loadeddata is not a presented-frame notification");
  assert.equal(source.color.seeks, 1, "paused videos request a same-time seek for initial presentation");
  assert.equal(source.ready, false);
  assert.equal(source.heldFrames(), null);
  source.encode(null, {});
  source.color.callback(0, { mediaTime: 0.5 });
  assert.equal(source.ready, false);
  assert.equal(source.heldFrames(), null);
  source.encode(null, {});
  source.depth.callback(0, { mediaTime: 0.5 });
  assert.equal(source.ready, true);
  assert.deepEqual(imported.map(({ kind, key }) => [kind, key]), [["color", 12], ["depth", 8]]);
  assert.equal(source.heldFrames().color.key, 12);
  source.destroy();
  assert.equal(source.ready, false);
});

test("external import errors identify the source, presented frame, and playback state", () => {
  const cause = new DOMException("Failed to execute 'importExternalTexture' on 'GPUDevice'.", "OperationError");
  const source = Object.assign(Object.create(VideoSource.prototype), {
    width: 640,
    height: 360,
    frames: { color: [], depth: [{ key: 0, buffer: { destroy() {} } }] },
    device: {
      createBuffer: () => ({ destroy() {} }),
      importExternalTexture() { throw cause; },
    },
  });
  const video = { currentSrc: "https://example.com/color.mp4", videoWidth: 640, videoHeight: 360,
    readyState: 2, paused: false, seeking: true, currentTime: 0.5 };
  const previousUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1 };
  try {
    assert.throws(() => source.ingest(video, "color", 12), (error) => {
      assert.equal(error.cause, cause);
      assert.match(error.message, /presented color frame 12.*OperationError/);
      assert.match(error.message, /https:\/\/example\.com\/color\.mp4/);
      assert.match(error.message, /640x360; readyState: 2; paused: false; seeking: true; time: 0.5/);
      return true;
    });
  } finally {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
  }
});
