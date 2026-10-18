# Circular blur kernel optimizer

Interactive optimizer and WebGPU reference implementation for **circularly symmetric blur kernels
built from complex Gaussians**: disc ("bokeh") blurs computed with a horizontal and a vertical 1-D
pass, at a cost proportional to the blur radius rather than its area.

- **Optimizer**: differential evolution (global search) and gradient fine-tuning, both running on
  [jax-js-nonconsuming](https://github.com/hamk-uas/jax-js-nonconsuming) (WebGPU, with a WebAssembly
  fallback).
- **Live visualizations**: radial cross-section with the evolving population, error curve,
  convergence, and the 2-D kernel.
- **Blur demo**: hand-written WGSL compute shaders processing one channel at a time, so memory
  doesn't grow with the channel count. Uniform or spatially varying disc radius: tilt-shift, or
  depth of field from the z-buffer of an animated 3D night street ray-cast on the GPU, or from an
  RGB-D video: Video Depth Anything's comparison clips (loaded from its project page) or two people
  talking at a desk with measured depth (included). Hovering over the picture (or holding a touch)
  eases the focus to the nearest depth within 8 pixels of the pointer, and restores it afterwards;
  a click or tap keeps it. f32 or
  packed 16-bit intermediates, and GPU timings and memory use per
  buffer. The "Shader code (WGSL)" panel shows the shaders, with copy and download buttons. Their
  header comments document the buffers, tap tables, and dispatch order, so you can use them
  elsewhere.

Based on Olli Niemitalo,
[Circularly symmetric convolution and lens blur](https://yehar.com/blog/?p=1495) (2010), extended
with merged pairs (repeated rates).

## Running

```sh
npm ci
npm run dev       # development server at http://localhost:5173/
npm run build     # static site in dist/
npm run preview   # serve dist/
npm run smoke     # Node test of the math, optimizer and fine-tuner (Wasm backend)
```

The browser needs WebGPU for the blur demo (current Chrome, Edge, and Safari; Firefox on Windows).
Without WebGPU, the optimizer still runs on WebAssembly.

URL options: `?calibrate` enables jax-js's GPU microbenchmark at startup. By default it is skipped,
because it allocates about 128 MB and can stall the page for several seconds. Skipping it made no
measurable difference to optimizer speed.

## Publishing on GitHub Pages

1. Push this repository to GitHub.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.

Every push to `main` then runs [`.github/workflows/pages.yml`](.github/workflows/pages.yml): it
installs dependencies, runs the smoke test, builds, and deploys `dist/`. The page links to sections
of this README on [GitHub](https://github.com/OlliNiemitalo/kernel-optimizer); in a fork, the build
points those links at the fork instead.

GitHub Pages cannot send the cross-origin isolation headers (`Cross-Origin-Opener-Policy`,
`Cross-Origin-Embedder-Policy`) that jax-js needs for `SharedArrayBuffer`, which its multithreaded
WebAssembly backend uses. The page therefore includes
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker): on the first visit it registers
a service worker that adds those headers to every response, then reloads the page once. Where the
server already sends the headers, as the development server does, it does nothing. It needs a
secure context (HTTPS or localhost), and private browsing modes may block service workers; the
app still works without cross-origin isolation, just without multithreaded WebAssembly.

## How it works

Distances are in units of the disc radius: the target kernel is 1 for $r \le 1$ and 0 for
$r \ge 1 + \tau$, where $\tau$ is the transition bandwidth. Write $s = r^2 = x^2 + y^2$.

### 1. Why complex Gaussians separate

For any complex rate $\lambda = -b + ic$ (with $b > 0$ for decay),

$$
e^{\lambda s} = e^{\lambda x^2}\, e^{\lambda y^2},
$$

because the exponent is additive. So $e^{\lambda s}$ is both circularly symmetric and separable: a
2-D convolution with it is a horizontal 1-D convolution followed by a vertical one. A real Gaussian
($c = 0$) can only make Gaussian blurs; the imaginary part $c$ adds oscillation in $r^2$. Weighted
sums of real parts of such terms can approximate other circular shapes, such as a disc.

### 2. Kernel model

A **single component** has four parameters $(b, c, r_w, i_w)$:

$$
K(s) = \bigl(r_w \cos(cs) + i_w \sin(cs)\bigr)\, e^{-bs}
     = \mathrm{Re}\,\bigl[B\, e^{\lambda s}\bigr], \qquad B = r_w - i\, i_w .
$$

This matches the article's format `(cos(x*x*c) * r + sin(x*x*c) * i) * exp(-b*x*x)`.

A **merged pair** adds two parameters $(r_1, i_1)$ and a factor of $s$:

$$
K(s) = \mathrm{Re}\,\bigl[(B + A s)\, e^{\lambda s}\bigr], \qquad A = r_1 - i\, i_1 .
$$

It is the limit of two components whose rates coincide:

$$
(B + A s)\, e^{\lambda s}
= \lim_{\delta \to 0} \Bigl[ B\, e^{\lambda s} + \tfrac{A}{\delta}\bigl(e^{(\lambda+\delta)s} - e^{\lambda s}\bigr) \Bigr].
$$

As two nearly equal rates, the same shape would need weights of size $A/\delta$ that almost cancel,
losing precision. The direct form above has no such cancellation. In this app a merged pair has
one shared rate and costs two channels, the same as two single components.

The full kernel is a sum of components. Parameters are stored as six numbers per component
$[b, c, r_w, i_w, r_1, i_1]$, with $r_1 = i_1 = 0$ pinned for single components.

### 3. From components to separable channels

The renderer works with **channels**. A channel $n$ is a pair of 1-D complex kernels,
horizontal $h_n$ and vertical $v_n$, each of the form

$$
h(t) = (h_0 + h_2 t^2)\, e^{\lambda t^2},
$$

and the 2-D kernel is

$$
K(x, y) = \sum_n \mathrm{Re}\,\bigl[h_n(x)\, v_n(y)\bigr].
$$

The real part is taken only at the very end, so per channel the output is
$\mathrm{Re}\,[hv] = h_\mathrm{re} v_\mathrm{re} - h_\mathrm{im} v_\mathrm{im}$.

- **Single component → one channel.** Absorb the weight as a square root, as in the article's
  update: $h_n(t) = v_n(t) = \sqrt{B}\, e^{\lambda t^2}$. Then
  $h_n(x) v_n(y) = B\, e^{\lambda s}$.
- **Merged pair → two channels.** Using $s = x^2 + y^2$,

$$
(B + A x^2 + A y^2)\, e^{\lambda x^2} e^{\lambda y^2}
= \underbrace{\bigl[\sigma e^{\lambda x^2}\bigr]\bigl[\tfrac{B + A y^2}{\sigma} e^{\lambda y^2}\bigr]}_{\text{channel 1}}
+ \underbrace{\bigl[\sigma x^2 e^{\lambda x^2}\bigr]\bigl[\tfrac{A}{\sigma} e^{\lambda y^2}\bigr]}_{\text{channel 2}} .
$$

Any $\sigma \ne 0$ works; the code uses $\sigma = \sqrt{\max(|A|, |B|)}$ to balance magnitudes between
the passes. See `separableChannels` in [`src/problem.js`](src/problem.js); the smoke test checks
that the channels reproduce the 2-D kernel to $10^{-12}$.

The "Current kernel" panel exports these channel coefficients, ready to use in other code.

### 4. Discrete implementation (WebGPU)

See [`src/blur.js`](src/blur.js). For a disc radius of $R$ pixels:

1. **Taps.** Use $t = -K \dots K$ with $K = \lceil R(1+\tau) \rceil$, evaluating each 1-D kernel at
   $(t/R)^2$. Compute tap values in double precision on the CPU and upload them as complex f32
   pairs (separate tables for $h$ and $v$).
2. **One channel at a time.** For each channel $n$, run a horizontal pass, then a vertical pass
   that adds its result into an accumulator:
   $$m(x, y) = \sum_t I(x + t, y)\, h_n(t), \qquad A(x, y) \mathrel{+}= \sum_t \mathrm{Re}\,\bigl[m(x, y + t)\, v_n(t)\bigr].$$
   Only one channel's complex intermediate $m$ exists at a time, so memory does not grow with
   the number of channels. The horizontal pass costs two real multiply-adds per tap (the input is
   real); the vertical pass forms only the real part.
3. **Normalization in the shader.** The passes filter a fourth lane holding a constant 1 together
   with the color, so $A_w(x, y)$ is the sum of the 2-D kernel weights that reached the pixel. The
   output is $A_\mathrm{rgb} / A_w$. For a uniform radius this equals dividing by the kernel's sum,
   also at image edges. It needs no precomputed constant, and it stays brightness-preserving
   when the radius varies across the image.
4. **Register blocking.** Each thread computes 8 consecutive outputs along the pass direction.
   As the loop walks the input, the 8 taps in use slide through registers, so every input sample
   and tap is read once per thread instead of once per output. The tap tables are zero-padded,
   so the sliding window needs no bounds checks.
5. **Edges.** Out-of-image reads clamp to the nearest edge pixel.

The cost per pixel, per color, per pass is (channels × taps) complex multiply-adds, so it grows
linearly with the radius.

#### Memory

Besides the source image and the 4-lane accumulator (16 bytes per pixel each), the
only working buffer is one channel's intermediate: 4 complex lanes, 32 bytes per pixel in f32. The
"16-bit float (packed)" option stores each complex lane as two half floats in one `u32`
(`pack2x16float`): 16 bytes per pixel, with no optional GPU feature needed. At 960×600 the
intermediate is 17.6 MiB, or 8.8 MiB packed, whatever the number of channels. Keeping all
channels at once would take channels × that.

#### Spatially varying radius

The radius comes from a per-pixel map: tilt-shift, or the 3D world's
z-buffer turned into a signed circle of confusion (negative in front of the focus). For focus
distance $f$ it is proportional to $1/f - 1/z$: the demo uses $2R\,\text{m}\,(1/f - 1/z)$, capped at
$\pm R$, so $R$ is the blur at 2 m with focus at infinity. An RGB-D video works the same way, with
relative inverse depth $v \in [0, 1]$ in one of two layouts:

- **Stacked** (the included TUM clip, and the file picker): one video, color on top, depth below
  as grayscale, used as $1/z = 2v$ per meter (white = 0.5 m, black = infinitely far).
- **Pair** (the [Video Depth Anything](https://videodepthanything.github.io/) samples): a color
  video and a separate depth video coded with matplotlib's "inferno" colormap, as that project
  visualizes its estimates. A 32³ lookup table maps each color to the position of the nearest
  colormap entry. Their depth is estimated and relative, so it is used as $1/z = v$ per meter, and
  the focus distance is in these relative units. The depth video is kept within a few frames of
  the color video while playing, and on the same frame when paused or stepping.

Inverse depth suits 8-bit video, since the blur radius is linear in it: one gray level is a fixed
fraction of $R$. Videos wider than 960 pixels are downscaled when read. Varying radii use
**precomputed radius-node tap tables**: nodes at 1/64 pixel spacing through 4 pixels, then
64 per octave. **Interpolate blur radii** in the Varying blur controls is off by default:
the faster nearest-node shader fetches one tap, using that node's
support. Enable it for continuous radius changes: the per-pixel node index and fraction are
prepared once; the inner loops read two complex taps at integer offsets and blend them, without
rescaling a profile or normalizing each tap. The option is included in copied view states and
shader exports; it has no effect on uniform blur. The tables are built from a dense
complex reference profile (256 points per disc radius), with a continuous zero endpoint. Each tap
is the integral of the scaled profile over the pixel interval, not its value at the pixel center.
Radius zero is identity; positive subpixel radii use the same integration as larger radii.
The uniform path keeps its previous point-sampled direct tables.

A plain gather, where each output uses its own radius, cuts the discs of blurry highlights next to
sharper pixels: those pixels neither look far enough nor use the highlight's kernel. So the passes
*scatter as gather*:

1. A prepass retains a coarse byte-per-pixel level map for depth classification (sign bit = in
  front of focus), but computes support bounds from the selected node (or the union of both nodes
  when interpolating). Dilating
  the largest kernel half-width per 16×16 tile over the
   neighbouring tiles gives each tile's **reach**: how far its pixels must look.
2. Every tap uses the radius table(s) selected by the **source** pixel it reads, so each source spreads
  its own disc. Profile samples outside its support are zero. The interpolating horizontal path
  processes four outputs per thread; the nearest, vertical and uniform paths process eight.
3. After the horizontal pass, a pixel mixes light from sources with different radii, but the
   vertical pass can apply only one kernel per pixel. The prepass therefore "blurs the blur size":
  the **vertical radius** is derived from the continuous log-radius average of row sources that reach the pixel, weighted by
   their brightness and by how much of their disc falls in the column. Bright highlights keep
   round discs; sharp detail next to them stays sharp. With a radius that depends only on the row
   (tilt-shift), every disc is exact. In the "Varying blur" controls, this can be turned off (each
  pixel then uses its own radius, cutting the discs of blurry highlights next to sharper pixels),
   and the brightness weight is raised to an adjustable power: 0 counts all sources equally,
   1 (the default) in proportion to brightness, larger values let highlights dominate.
4. **Occlusion** (z-buffer only, optional): a destination hides sources farther than an adjustable
  signed-level threshold behind it (default six levels), beyond its own kernel support. This
  depth order applies on **both sides of focus**; a slightly defocused foreground surface must
  still block distant highlights. Within its support, farther light is allowed, approximating
  partial aperture coverage. A destination with zero membership in the current layer does not
  occlude it: inter-layer visibility belongs to compositing, so far light remains available
  behind the near layer. This separable visibility test is an approximation, not aperture ray tracing.

Working maps contain five packed byte maps and four float planes (smoothed source radius,
smoothing scratch reused as source table coordinate, vertical radius, vertical table coordinate),
in addition to the original signed radius map. Tap tables are included in displayed memory;
the integrated varying-radius path needs no DC correction matrix.
Each prepass step runs in its own compute pass: on the tested Intel GPU, dependent dispatches within
one compute pass were not reliably ordered, which made the output vary from frame to frame.

#### Depth layers

With one layer, a blurry object in front of the focus (a person, with the focus behind them) keeps
a sharp silhouette: in-focus background pixels have the identity kernel, so nothing spreads into
the object, and the normalization rescales the object's half discs at its edge back to full
strength. With **Layers (experimental)** (z-buffer only), a near layer is blurred separately and
composited over the far one, so the silhouette spreads over the background, and the background
shows through the object's semi-transparent edge, as in a real lens.

- **Local split, at depth edges.** Which depths need separating varies across the image, so the
  split is chosen per 16×16 tile from the depth edges of the surrounding 5×5 tiles: pairs of
  pixels 4 px apart whose nearer pixel lies in front of the focus and is at least an octave
  blurrier. Each pair supports every split between its two blur levels; a tile splits in the
  middle of the range supported by at least half the peak (and 16 pairs). A connected surface,
  however steep, has no such edges and is not split.
- **Spread to tiles without a split.** A tile without a split of its own takes the split of the
  nearest tiles that have one, up to 8 tiles (128 px) away. Otherwise the interior of a large
  object, more than two tiles from its silhouette, would fall into the far layer: its blurred
  light, normalized by the far layer's low coverage there, came through the near layer's
  semi-transparent edge as a bright seam (a car's tail light leaking along the car's side).
  Farther away, the near layer is empty, and the passes skip its empty tiles.
- **Membership depends only on the source.** Each pixel's membership in the near layer comes from
  the four nearest tile centers, bilinearly interpolated, with a soft step as wide as the
  supported range (a surface that continues across the split, such as feet on the ground, fades
  between the layers). The mask is then closed morphologically (4 px), so thin details whose
  estimated depth fell behind (a zipper on a coat) stay with their surface. Because membership is
  a function of the source pixel alone, each layer is a linear image and the separable passes
  stay exact.
- **Smooth blur levels within a layer.** Estimated depth is noisy, and with scatter-as-gather,
  neighbors with different disc sizes do not sum to unit weight. The source levels are smoothed
  by a median (a sorting network over the window) and then a mean over ±6 px of the pixels on the
  same side of the split, never across it, so silhouettes keep their depth jumps. One smoothed
  floating-point map serves both layers, without rounding the smoothed radii back to integer levels.
  The median/mean neighborhood rules are unchanged; their depth-boundary artifacts are a separate
  issue from radius quantization and are not fixed by profile interpolation.
- **Back to front.** The far layer runs first; where its coverage is below ½ (behind the near
  layer's sources), it is completed by pull-push (a pyramid of coverage-weighted averages built by
  halving, then filled from coarse to fine with bilinear upsampling). The near layer, its color
  normalized by its coverage $w$, then goes over it with opacity
  $\alpha = \min(\max(\operatorname{close}(w), m f),\ 3w)$: the coverage after a 3 px
  morphological closing, which fills narrow dips from mixed disc sizes but keeps the smooth ramp
  around a silhouette, and, on the layer's own pixels (membership $m$), the fraction $f$ of the
  layer's pixels within the pixel's own blur radius (a square window), which makes the layer
  opaque deeper inside than its blur reaches. The cap $3w$ keeps the noisy color of the kernels'
  fading tails, a ratio of two tiny sums, from showing as a sharp, speckled fringe. The layers
  take two extra buffers (the composite, and debug data; 16 bytes per pixel each) plus the
  pyramid, about a third more, and two more byte maps.
- **Cost.** Per-tile reach is stored as K + 1, with 0 for tiles that no source of the layer
  reaches, so the passes skip them; the near layer is cheap where it is empty. The radius-table
  interpolant avoids the former rescaled-profile path's per-tap division, coordinate arithmetic,
  identity blend, and normalization factor. Pixel integration removes the mixed-radius correction
  lookup, multiply, reciprocal, fade and branch entirely. With that register state gone, nearest
  horizontal filtering shares each source across eight outputs again (four when interpolating).
  The one-layer pipeline specializes away layer membership and map-selection branches.
  Older nearest-node timings are not directly
  comparable; the benchmark page alternates frozen-scene renders on the same device.
- **Empty-layer identity.** The final mask pass records whether the near layer contains any
  members. When it is empty, source-level smoothing and far-layer hole filling are bypassed;
  adding an empty layer must not change the one-layer image. The occupancy flag is reset each frame.

GPU regression tests are available at `/test/gpu.html` on the development server. They read linear
buffers and check visibility on either side of focus, both spatial axes and intermediate precisions,
foreground spread, empty-layer identity, reused frame state, and the tetra scene at two scales.
Radius tests compare GPU impulse responses and mixed-radius ramps with the CPU model in both
nearest and interpolated modes, including the last node and both intermediate precisions. They
check fractional vertical radii, default-off behavior, nearest-node plateaus, and continuity when
interpolation is enabled.

The **Layer view** selector shows the composite, the near layer's opacity (tinted, or as gray at
half scale), the near layer alone, the hole-filled far layer alone, or, with one layer, the summed
kernel weight per pixel.

**Pixel integration and mixing radii.** Point-sampled small kernels do not preserve their complex
channel sums under scaling. Instead, for a continuously scaled profile, build each tap as

$$
h_{n,r}[t]=\int_{t-1/2}^{t+1/2}\frac{1}{r}h_n(x/r)\,dx.
$$

The pixel intervals partition the line, so their sums are radius-independent complex constants
$A_n=\int h_n(x)\,dx$ and $B_n=\int v_n(y)\,dy$. The common profile gain makes
$\sum_n\mathrm{Re}[A_nB_n]=1$. Thus every fixed pair of radii has unit DC, even when the horizontal
and vertical radii differ. Linear interpolation between nodes preserves those channel sums too.
**No mixed-radius correction is necessary in either lookup mode.** Channels are not independently
rescaled or divided by their sums, so zero/near-zero channel integrals and merged polynomial
channels do not introduce singularities.

The implementation analytically integrates each piecewise-linear segment of the dense profile
using a cumulative integral. That is exact for the reference interpolant, not for the underlying
analytic Gaussian. Its primitive saturates outside the compact support. The support bounds
include the extra half-pixel footprint and, when interpolating, both neighboring node supports.
At radius zero each channel is its integral times a delta. If a positive-radius kernel fits wholly
inside the central pixel, it also produces identity; there is no artificial identity-to-radius-one
blend. This changes the subpixel and small-radius PSF intentionally.

For fixed radii this is box filtering the separable 2-D kernel over the destination pixel before
sampling. It is **kernel antialiasing**, not source-image mipmapping, subpixel geometry/depth
antialiasing, or exact variable-radius convolution. The square pixel footprint also breaks exact
radial symmetry at pixel scale. Spatially varying radii, occlusion and boundaries still require
the RGB/weight handling; layer split, smoothing and compositing rules are unchanged.

Tests check every stored node's complex sums, f32 upload error, independent clipped-segment
quadrature, radius interpolation error versus direct pixel integration, subpixel support onset,
GPU raw coverage, impulse responses and mixed-radius ramps in both lookup modes. The old
point-sampled reference is retained for historical tests, not used as the new PSF's correctness
target. Sampled CPU mixed DC error is about $10^{-13}$; sampled interpolation L1 kernel error is
0.00151. These are measured errors, not universal bounds.

`/test/benchmark.html` provides an opt-in alternating GPU-timestamp comparison against captured
baseline source. It warms both pipelines, shares one device, freezes the tetra camera, and reports
per-pass medians and raw samples at radii 15 and 69 in f32 and f16. Keep other GPU workloads idle.
The scene selector also offers the default woman-in-field video from Video Depth Anything: both
RGB and depth are paused at the middle of the first frame, and shared by both renderers. Network
loading and shader warmup are outside the samples. Capture the baseline before editing sampler
revisions; select the same interpolation mode for the comparison.

**Current pixel-integrated tables versus `5372168`.** Intel Gen9, preset 3, one layer,
size blending and occlusion enabled; 17 alternating samples after two warmups. The default
woman video is frozen at 960×540, focus control 0.675. Times are GPU medians in milliseconds;
totals include scene rendering and preparation. Each total is measured independently, rather
than obtained by adding the per-pass medians.

| Lookup | Radius | Intermediate | Horizontal before → after | Total before → after | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Nearest | 15 | f32 | 23.9 → 17.1 | 47.2 → 40.2 | 1.17× |
| Nearest | 15 | packed f16 | 23.0 → 14.4 | 44.2 → 34.0 | 1.30× |
| Nearest | 69 | f32 | 64.9 → 54.5 | 145.8 → 135.7 | 1.07× |
| Nearest | 69 | packed f16 | 93.5 → 76.3 | 197.8 → 179.6 | 1.10× |
| Interpolated | 15 | f32 | 69.9 → 33.2 | 113.0 → 75.1 | 1.50× |
| Interpolated | 15 | packed f16 | 66.5 → 31.5 | 108.7 → 67.9 | 1.60× |
| Interpolated | 69 | f32 | 142.5 → 121.0 | 293.0 → 268.0 | 1.09× |
| Interpolated | 69 | packed f16 | 143.3 → 122.6 | 273.0 → 251.3 | 1.09× |

GPU clocks drifted during the sequence (including within the nearest suite), so compare paired
before/after results, not absolute times across rows or historical runs. The same run's frozen
960×600 tetra scene improved total nearest-mode time by 1.10–1.11× at radius 15 and
1.17× at radius 69. These measurements do not establish parity with published main.

Removing the quadratic DC matrix reduces preset-3 table storage from 1,651,776 to 478,336 bytes
at radius 15 (71% less), and from 4,464,736 to 2,276,576 bytes at radius 69 (49% less),
with no added working maps. Both lookup modes share those tables. The smoke suite and all
692 GPU checks pass, including high-channel and merged-polynomial flat-field tests.

Historical 32-step interpolated-table measurements (`2cbb7fc`) on Intel Gen9 at 960×600,
preset 3, one layer, size blending and occlusion enabled:
nine alternating samples after two warmups, compared with the rescaled-profile sampler (`b6c1470`).
Totals include scene rendering and preparation, not just convolution.

| Radius | Intermediate | Previous (ms) | Lookup (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 15 | f32 | 99.4 | 76.6 | 1.30× |
| 15 | packed f16 | 101.9 | 72.6 | 1.40× |
| 69 | f32 | 397.1 | 298.1 | 1.33× |
| 69 | packed f16 | 399.4 | 276.9 | 1.44× |

For that 32-step version, kernel tables shrink from 1.12 to 0.52 MiB at radius 15 and from 2.08 to 1.63 MiB at radius 69.
Caching the vertical table coordinate adds one float per pixel (2.20 MiB here), so total GPU
memory increases despite the smaller tables. In the sampled preset/radius quality sweep, the
largest integrated absolute kernel difference was 0.00557, with a localized truncation-tail
peak difference of 0.0311 after multiplying by radius squared. These are sampled results,
not universal performance or error bounds.

The historical point-sampled 64-step grid (`25bb7b3`) doubled the radius-node density. An 11-sample alternating comparison
against the 32-step grid under the same conditions found up to 2% slowdown with interpolation,
and up to 5% with nearest lookup. This modest cost was retained for the finer radius spacing.
Its nearest-mode totals were about 56/53 ms at radius 15 and 208/198 ms at radius 69 (f32/f16).
Table storage grows to 1.58/4.26 MiB respectively: taps grow roughly 2×, but the node-pair DC matrix
grows roughly 4×. The sampled interpolated-kernel integrated error falls to 0.00111, with scaled
peak error 0.00395. Nearest lookup deliberately retains small radius steps; these interpolation
error measurements do not describe nearest-mode error.

The historical nearest reciprocal/fade lookup (`5372168`), cached-key reuse, one-layer specialization, and short-kernel
two-output blocking were compared against `25bb7b3` on that frozen 960×540 video (21 alternating
samples, radius 15, default focus 0.675, preset 3, one layer):

| Intermediate | Horizontal before → after (ms) | Total GPU before → after (ms) |
| --- | ---: | ---: |
| f32 | 26.7 → 23.1 | 50.5 → 46.8 |
| packed f16 | 25.9 → 23.0 | 47.6 → 44.6 |

Reciprocal precomputation alone gave only a small improvement, not the whole gain. At radius 69,
the combined total improvement was about 1–2%; on tetrahedra it ranged from about 1% slower at
radius 15 to 2–3% faster at radius 69. Interpolated video timings were essentially unchanged.
That did **not** restore published-main performance: the older renderer has no mixed-radius
correction, far fewer radius nodes, and smaller working maps. Simply removing the correction
from point-sampled taps is not a correctness-preserving optimization; the current integrated
construction instead changes the PSF to make the correction unnecessary.

#### Truncation and corners

Truncating each 1-D kernel at $|t| \le K$ makes the 2-D support a
square. Its corners reach $r = \sqrt{2}\,(1 + \tau)$, so the kernel must already be near zero
there. The optimizer's stop band extends far beyond that, to about $1 + \tau + \ln N_\text{stop}$.

### 5. Precision

Disc kernels with several channels have large weights that cancel. The fitted weights grow
roughly 2.5× per channel, reaching about 100 at 5–6 channels, so small relative errors in the
terms get amplified.

- **Accumulate in f32.** Measured with an impulse input against the exact kernel (4-component
  preset), the f32 path has $2.5 \times 10^{-6}$ relative error, and the packed 16-bit
  intermediate about $9 \times 10^{-3}$.
- **Don't trust GPU `sin`/`cos` for kernel coefficients.** On the tested GPU they had about
  $6 \times 10^{-5}$ absolute error. That is negligible alone, but multiplied by weights near 100
  it exceeds a ripple of $\pm 0.004$. This app computes tap tables on the CPU, and the optimizer
  uses its own reduced-range polynomial sine and cosine ([`src/trig.js`](src/trig.js), about
  $7 \times 10^{-8}$ error).

### 6. Optimization

**Cost.** The cost is the minimax error, $\max_j |K(x_j) - \mathrm{target}_j|$, the ripple shown
in the plots. The pass band is sampled uniformly on $[0, 1]$. Following the article, stop-band
samples are $x_t = 1 + \tau - \ln\frac{N - t}{N}$ for $t = 0 \dots N-1$: dense near the edge,
sparse far out.

**Negative values.** Negative kernel values look like cheap sharpening. The "Negative gain" $w$
(a problem parameter) multiplies the error wherever the kernel is negative:
$\max_j |K(x_j) - \mathrm{target}_j| \cdot (w \text{ if } K(x_j) < 0 \text{ else } 1)$, in the
differential evolution, the fine-tuning loss, and the displayed ripple alike. The presets include
the article's two "almost no negative values" sets, which it gives in the separable form
$(c + d i)^2 e^{(a + bi)s}$; in this app's parameters that is $b \leftarrow -a$, $c \leftarrow b$,
$r = c^2 - d^2$, $i = -2cd$. They load with $w = 100$, at which the 1-component set is about
equiripple, and without the per-slot frequency bands, which they don't follow (one of the
2-component set's rates is almost real).

**Differential evolution** ([`src/de.js`](src/de.js)) is generation-synchronous, so that each
generation is one jitted GPU call with no data read back:

1. Draw a random permutation $\sigma$ of the population (an argsort of uniform random numbers).
   Target $\sigma_k$ gets donors $\sigma_{k+1}, \sigma_{k+2}, \sigma_{k+3}$ (indices mod $P$).
   All four are distinct, and every member plays each role once per generation. This is the
   population management of `logistics_sim`'s
   [`ga.h`](https://github.com/hamk-uas/logistics_sim/blob/main/ga.h), adapted to DE.
2. Mutant: $a + F(b - c)$.
3. Exponential crossover: a cyclic run of mutant genes starting at a random position, of length
   $L = \min\bigl(D,\ 1 + \lfloor \ln(1-u) / \ln \mathrm{CR} \rfloor\bigr)$ for uniform $u$; the
   other genes come from the target.
4. Bounds: reflect once at each bound, then clamp. Pinned parameters have both bounds at zero.
5. The child replaces its target only if its cost is strictly lower.

Search heuristics, mostly taken from the article's solutions:

- Channel slot $k$ oscillates within $(1.14k + 0.5)\pi \pm 0.4\pi$; a merged pair spans the bands of
  the two slots it occupies.
- Initial decay $b \in [0.3,\ 1 + N_\text{ch}]$.
- Initial weights within $\pm 2 \cdot 2.5^{N_\text{ch} - 1}$.

**Gradient fine-tuning** ([`src/finetune.js`](src/finetune.js)) minimizes a smooth maximum of the
absolute error $a_j = \sqrt{e_j^2 + 10^{-12}}$ on a dense grid (2000 + 2000 samples):

$$
L_\beta = m + \frac{1}{\beta} \ln \sum_j e^{\beta (a_j - m)}, \qquad m = \max_j a_j \ \text{(gradient stopped)}.
$$

$\beta$ is set relative to the current ripple, as sharpness / ripple, and the sharpness rises 10×
over the run. Parameters are scaled to relative units $\theta = p / \max(|p|, 0.05)$, so one
learning rate suits both decays and large weights. The optimizer is Adam (optax), with initial
learning rate $10^{-3} / 3^{N_\text{ch} - 2}$ halving every third of the run; decays are kept at
$b \ge 0.05$. The best kernel by true maximum error is kept. Gradients come from jax-js autodiff.

### 7. Reference results

All ripples below are measured on the dense grid ($\tau = 0.2$):

| Kernel | Ripple |
|---|---|
| 2 components, found by DE | ±0.07595 |
| Article's 2-component set | ±0.07729 |
| Article's 3-component set | ±0.02745 |
| Article's 3-component set, perturbed by ±1% and then fine-tuned | ±0.02666 |
| Article's 5-component set | ±0.004116 |
| Article's 5-component set, fine-tuned | ±0.004084 |

The article's own ripple figures were measured on its sparser 200 + 200 grid, which is why they
look slightly better.

On an Intel UHD (Gen9) laptop GPU, differential evolution evaluates roughly 100,000–250,000
candidates per second.

## Project layout

| Path | Contents |
|---|---|
| `src/problem.js` | Kernel model, sample grids, bounds and heuristics, separable channels, export and import |
| `src/model.js`, `src/trig.js` | Traced (jax-js) kernel profile and accurate sine and cosine |
| `src/de.js`, `src/finetune.js` | Differential evolution and gradient fine-tuning |
| `src/blur.js` | WebGPU blur: WGSL compute and display shaders |
| `src/world.js` | Animated 3D night street ray-cast on the GPU: procedural textures, puddles with traced reflections, glossy highlights; adaptive 4× antialiasing and procedural patterns box-filtered over the pixel footprint; the circle of confusion from the z-buffer |
| `src/shapes.js` | Test scene ("Tetras"): three tetrahedra in a row against a black, starry background at infinity, ray-cast with 4× antialiasing; drag the picture to turn the view; the Scale slider (or the wheel over the picture) scales the tetrahedra about the middle one |
| `src/video.js` | RGB-D video as a depth-of-field source: color over grayscale depth, or a color video with an inferno-coded depth video |
| `tools/tum_rgbd_video.py` | Converts a [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) sequence to that format (fills depth holes, H.264) |
| `public/tum-sitting-rgbd.mp4` | 15 s of TUM RGB-D fr3/sitting_static, converted with the tool above |
| `src/plots.js`, `src/scene.js`, `src/main.js` | Plots, procedural HDR test scene, user interface |
| `test/smoke.js` | Checks the math and runs the optimizer and fine-tuner in Node |
| `test/merged-de.js` | Experiment: DE convergence with and without merged pairs |
| `vendor/jax-js-nonconsuming/` | jax-js-nonconsuming v0.12.14 TypeScript sources (MIT), compiled by Vite |
| `public/coi-serviceworker.js` | coi-serviceworker v0.1.7 (MIT), served unbundled at the site root; modified to let cross-origin requests bypass it, as proxied video range requests were aborted |
| `public/favicon.png`, `public/apple-touch-icon.png` | Center of the article's [5-component square-wave kernel image](https://yehar.com/blog/wp-content/uploads/2010/09/squarewave5componentreal.png) |

jax-js-nonconsuming is vendored because it is not published on npm, and its Git install
does not build on Windows.

## License

MIT for this project ([LICENSE](LICENSE)). The vendored jax-js-nonconsuming sources are MIT
licensed; see [vendor/jax-js-nonconsuming/LICENSE](vendor/jax-js-nonconsuming/LICENSE).
`public/coi-serviceworker.js` is by Guido Zuidhof and contributors, MIT licensed (see its header).
`public/tum-sitting-rgbd.mp4` is derived from the TUM RGB-D benchmark sequence fr3/sitting_static
(J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers, "A Benchmark for the Evaluation of
RGB-D SLAM Systems", IROS 2012), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): trimmed to 15 s, depth holes filled,
depth converted to 8-bit inverse depth, and re-encoded as H.264. The Video Depth Anything samples
are not part of this repository: the page loads them from
[videodepthanything.github.io](https://videodepthanything.github.io/) (S. Chen et al., "Video Depth
Anything: Consistent Depth Estimation for Super-Long Videos", CVPR 2025).
