const MEDIA_ERRORS = {
  1: "MEDIA_ERR_ABORTED: loading was aborted",
  2: "MEDIA_ERR_NETWORK: a network error interrupted loading",
  3: "MEDIA_ERR_DECODE: the browser could not decode the video",
  4: "MEDIA_ERR_SRC_NOT_SUPPORTED: the format/codec is unsupported, or the request was blocked",
};

export function mediaError(element, url) {
  const error = element.error;
  const reason = MEDIA_ERRORS[error?.code] ?? "Unknown media error";
  return new Error(`${reason}${error?.message ? ` (${error.message})` : ""}. ` +
    `Source: ${url}. Picture: ${element.videoWidth}x${element.videoHeight}; readyState: ${element.readyState}. ` +
    "Check codec/profile/level support on this device, the connection, and the server's CORS policy.");
}

export function releaseVideo(element) {
  element.pause();
  element.removeAttribute("src");
  element.load();
}

export function loadVideo(url, { signal, onError, onSeek, onEnded, timeoutMs = 30000 } = {}) {
  const element = document.createElement("video");
  Object.assign(element, { crossOrigin: "anonymous", muted: true, loop: true, playsInline: true, preload: "auto" });
  return new Promise((resolve, reject) => {
    let loaded = false;
    let timer;
    const cleanup = () => {
      clearTimeout(timer);
      element.removeEventListener("loadeddata", ready);
    };
    const fail = (error) => {
      cleanup();
      if (loaded) onError?.(error);
      else {
        signal?.removeEventListener("abort", abort);
        element.removeEventListener("error", failed);
        releaseVideo(element);
        reject(error);
      }
    };
    let retries = 2;
    const failed = () => {
      // Chrome's HTTP cache intermittently fails concurrent range requests for the same URL
      // (net::ERR_CACHE_OPERATION_NOT_SUPPORTED), surfacing as MEDIA_ERR_SRC_NOT_SUPPORTED.
      if (!loaded && retries > 0 && element.error?.code !== 3) {
        retries--;
        element.src = url;
        element.load();
        return;
      }
      fail(mediaError(element, url));
    };
    const abort = () => {
      cleanup();
      element.removeEventListener("error", failed);
      element.removeEventListener("seeked", onSeek);
      element.removeEventListener("ended", onEnded);
      releaseVideo(element);
      if (!loaded) reject(new DOMException("Video loading cancelled.", "AbortError"));
    };
    const ready = () => {
      if (!element.videoWidth) {
        fail(new Error(`The video has no picture. Source: ${url}.`));
        return;
      }
      loaded = true;
      cleanup();
      resolve(element);
    };
    element.addEventListener("loadeddata", ready);
    element.addEventListener("error", failed);
    if (onSeek) element.addEventListener("seeked", onSeek);
    if (onEnded) element.addEventListener("ended", onEnded);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) {
      abort();
      return;
    }
    timer = setTimeout(() => fail(new Error(
      `Timed out waiting for a decoded video frame after ${timeoutMs / 1000} s. Source: ${url}. ` +
      `readyState: ${element.readyState}; networkState: ${element.networkState}.`,
    )), timeoutMs);
    element.src = url;
    element.load();
  });
}

export async function playVideo(element, onError) {
  try {
    await element.play();
  } catch (error) {
    // Pausing, seeking, or changing scenes can interrupt a pending play().
    if (error.name !== "AbortError") onError(error);
  }
}
