# Circular blur kernel optimizer

Interactive optimizer and WebGPU reference implementation for **circularly symmetric blur kernels
built from complex Gaussians**: disc ("bokeh") blurs computed with a horizontal and a vertical 1-D
pass, at a cost proportional to the blur radius rather than its area.

- **Optimizer**: differential evolution (global search) with exact minimax weights solved by
  Remez exchange on CPU workers, or over all parameters on the GPU; gradient fine-tuning on
  [jax-js-nonconsuming](https://github.com/hamk-uas/jax-js-nonconsuming) (WebGPU, with a WebAssembly
  fallback); and a Remez exchange refinement (float64 on the CPU) that converges to a local minimax
  optimum in a few iterations.
- **Live visualizations**: radial cross-section with the evolving population, error curve,
  convergence, and the 2-D kernel.
- **Blur demo**: hand-written WGSL compute shaders processing one channel at a time, so memory
  doesn't grow with the channel count. Uniform or spatially varying disc radius: tilt-shift, or
  depth of field from GPU-rendered Tetras or an animated 3D night street, or from an
  RGB-D video: by default a woman in a field (1080p, 24 fps, with Video Depth Anything's 15 fps
  depth interpolated in time; included), Video Depth Anything's comparison clips and long-video
  results (loaded from its project page), or two people talking at a desk with measured depth (included). Hovering over the picture (or holding a touch)
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

The blur demo requires a browser and graphics device with WebGPU enabled.
Without WebGPU, the optimizer still runs on WebAssembly.

URL options: `?calibrate` enables jax-js's GPU microbenchmark at startup. By default it is skipped,
because it allocates about 128 MB and can stall the page for several seconds. It is not required
for the optimizer.

**Control availability.** Blur-specific controls are disabled when blur is off. The player bar under
the picture (go to start, previous frame, play/pause, next frame, time, speed) is available only for
videos and the animated world; the seek bar, duration and loop only for videos, and full screen for
every scene (also by double-clicking the picture, again to exit). The bar repeats the Blur on/off switch, so it is available in full screen. The space bar plays and pauses, unless a control that uses it has focus.
Focus, layers, blur-size blending and occlusion require depth mode with a depth source (world,
video or Tetras). Brightness weight additionally requires blur-size blending, and the occlusion
threshold requires occlusion. Radius interpolation is available only for varying blur.
Tilt-shift has one radius per row, so blur-size blending and brightness weighting have no effect and are
disabled. The general **View** selector is in **Varying blur**: with blur enabled it offers the composite
and, for one layer, summed kernel weight; near/far diagnostics are available only with depth
layers enabled. Disabled settings are retained, while an unavailable diagnostic resets to the
composite. **Depth map** is available for all depth sources in depth-of-field mode, with or without
layers. It shows the decoded inverse depth used for focus picking and blur-radius generation,
not the layer-smoothed radius. Brighter means nearer;
inverse depth $d$ maps to display gray $d/(1+d)$ (black at infinity, half gray at $d=1$).
This fixed diagnostic scale bypasses exposure. Controls that only affect blur or
color are disabled in this view. The selector remains usable in uniform mode for kernel-weight
diagnostics even though the varying-blur settings are disabled.

**Image capture.** **Save PNG** downloads the rendered view; **Copy image** writes a freshly
rendered PNG to the system clipboard for pasting into other applications. Copying requires a
secure context (HTTPS or localhost), browser image-clipboard support and clipboard permission.
The button is disabled when the API is unavailable; a denied write offers Save PNG as a fallback.
**Copy view state** copies reproducible settings as JSON, not image pixels.

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

See [`src/blur.js`](src/blur.js). The **uniform-radius path**, for a disc radius of $R$ pixels,
uses point-sampled taps and sliding-register reuse as follows. The varying-radius path instead
uses pixel-integrated tables, with eight horizontal outputs per thread for nearest-node lookup,
four for interpolation, and eight vertical outputs; its source-dependent taps do not use the
uniform path's sliding tap window.

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
  also at image edges. The varying path also filters RGB and weight together to preserve
  constant color where coverage is nonzero; this is not exact variable-radius convolution.
4. **Register blocking.** Each thread computes 8 consecutive outputs along the pass direction.
   As the loop walks the input, the 8 taps in use slide through registers, so every input sample
   and tap is read once per thread instead of once per output. The tap tables are zero-padded,
   so the sliding window needs no bounds checks.
5. **Edges.** Out-of-image reads clamp to the nearest edge pixel.

The cost per pixel, per color, per pass is (channels × taps) complex multiply-adds, so it grows
linearly with the radius.

#### Memory

For uniform blur, besides the source image, tap tables and the 4-lane accumulator (source and
accumulator are 16 bytes per pixel each), the main working buffer is one channel's intermediate:
4 complex lanes, 32 bytes per pixel in f32. The "16-bit float (packed)" option stores each
complex lane as two half floats in one `u32`
(`pack2x16float`): 16 bytes per pixel, with no optional GPU feature needed. At 960×600 the
intermediate is 17.6 MiB, or 8.8 MiB packed, whatever the number of channels. Keeping all
channels at once would take channels × that.

#### Spatially varying radius

The radius comes from a per-pixel map. Tilt-shift uses a radius that varies by row. In depth mode,
the world or Tetras z-buffer is turned into a signed circle of confusion (negative in front of the focus). For focus
distance $f$ it is proportional to $1/f - 1/z$: the demo uses $2R\,\text{m}\,(1/f - 1/z)$, capped at
$\pm R$, so $R$ is the blur at 2 m with focus at infinity. An RGB-D video works the same way, with
relative inverse depth $v \in [0, 1]$ in one of three layouts:

- **Stacked** (the included TUM clip, and the file picker): one video, color on top, depth below
  as grayscale, used as $1/z = 2v$ per meter (white = 0.5 m, black = infinitely far).
- **Pair** (the [Video Depth Anything](https://videodepthanything.github.io/) samples): a color
  video and a separate depth video coded with matplotlib's "inferno" colormap, as that project
  visualizes its estimates. A 32³ lookup table maps each color to the position of the nearest
  colormap entry. Colormap colors don't blend into colormap colors, so the shader decodes the
  four nearest depth texels and then interpolates bilinearly. Their depth is estimated and
  relative, so it is used as $1/z = v$ per meter, and
  the focus distance is in these relative units. The depth video is kept within a few frames of
  the color video while playing, and on the same frame when paused or stepping.
- **Pair with temporal interpolation** (the default woman-in-field video): the color is the
  original footage, 1920×1080 at 24 fps, while Video Depth Anything's depth for it is 1280×720 at
  15 fps. The depth video plays in two elements, one depth frame apart. If they show consecutive
  frames $k$ and $k+1$, the shader blends their decoded depths with weight $15t - k$ at color time
  $t$; otherwise (while they resynchronize) it uses frame $k$ alone. Both files are vendored
  unmodified apart from trimming the color to the 13 s that the depth covers. Its focus starts at
  2.4 relative units, on the woman.
- **Side by side** (Video Depth Anything's long-video results): one video, color on the left and
  inferno-coded depth on the right, decoded as for a pair. The "Play Speed x3" caption burnt into
  the depth half of two of them reads as a small patch of wrong depth.

Inverse depth suits 8-bit video, since the blur radius is linear in it: one gray level is a fixed
fraction of $R$. Videos wider than 960 pixels are downscaled when read.
Varying radii use
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
The uniform path uses point-sampled direct tables.

A plain gather, where each output uses its own radius, cuts the discs of blurry highlights next to
sharper pixels: those pixels neither look far enough nor use the highlight's kernel. So the passes
*scatter as gather*:

1. A prepass prepares a coarse byte-per-pixel level map for depth classification (sign bit = in
  front of focus), but computes support bounds from the selected node (or the union of both nodes
  when interpolating). Dilating the largest kernel half-width per 16×16 tile over the
   neighbouring tiles gives each tile's **reach**: how far its pixels must look.
2. Every tap uses the radius table(s) selected by the **source** pixel it reads, so each source spreads
  its own disc. Profile samples outside its support are zero. The interpolating horizontal path
  processes four outputs per thread; the nearest, vertical and uniform paths process eight.
3. After the horizontal pass, a pixel mixes light from sources with different radii, but the
   vertical pass can apply only one kernel per pixel. The prepass therefore "blurs the blur size":
  the **vertical radius** is derived from the continuous log-radius average of unoccluded row
  sources that reach the pixel. Each source contributes weight
  $$w = m\,(L + 0.01)^b\,d,$$
  where $m$ is its membership in the current layer (1 without layers),
  $L = 0.2126 I_r + 0.7152 I_g + 0.0722 I_b$ is linear luminance, $b$ is the brightness-weight
  exponent, and $d$ is the disc marginal estimating how much of its disc falls in the column.
  The default $b=1$ is a useful highlight-priority heuristic: it helps bright highlights
  retain their shape when radii mix horizontally, but is neither kernel normalization nor exact
  variable-radius convolution. At $b=0$ only luminance weighting disappears; membership and disc
  overlap still apply. Larger exponents favor highlights more. Turning off **Blur the blur size**
  uses each pixel's own radius instead. For tilt-shift, all sources in a row already share one
  radius, so neither this averaging nor the exponent changes it; the separable passes reproduce
  that row-constant source-radius kernel, subject to the chosen tap lookup and precision.
4. **Occlusion** (z-buffer only, optional): a destination hides sources farther than an adjustable
  signed-level threshold behind it (default six levels), beyond its own kernel support. This
  depth order applies on **both sides of focus**; a slightly defocused foreground surface must
  still block distant highlights. Within its support, farther light is allowed, approximating
  partial aperture coverage. A destination with zero membership in the current layer does not
  occlude it: inter-layer visibility belongs to compositing, so far light remains available
  behind the near layer. This separable visibility test is an approximation, not aperture ray tracing.

Working maps contain five packed byte maps (level, vertical level, smoothed level, scratch and
near-layer membership) and four float planes (smoothed source radius,
smoothing scratch reused as source table coordinate, vertical radius, vertical table coordinate),
in addition to the original signed radius map. Tap tables are included in displayed memory;
the integrated varying-radius path needs no DC correction matrix.
Each dependent prepass step runs in its own compute pass to enforce ordering.

#### Depth layers

With one layer, a blurry object in front of the focus (a person, with the focus behind them) keeps
a sharp silhouette: in-focus background pixels have the identity kernel, so nothing spreads into
the object, and the normalization rescales the object's half discs at its edge back to full
strength. With **Layers (experimental)** (z-buffer only), a near layer is blurred separately and
composited over the far one, so the silhouette spreads over the background, and the background
shows through the object's semi-transparent edge. This approximates lens visibility; it does not
reconstruct surfaces hidden in the input.

- **Local split, at depth edges.** Which depths need separating varies across the image, so the
  split is chosen per 16×16 tile from the depth edges of the surrounding 5×5 tiles: pairs of
  pixels 4 px apart whose nearer pixel lies in front of the focus and is at least an octave
  blurrier. Each pair supports every split between its two blur levels; a tile splits in the
  middle of the range supported by at least half the peak (and 16 pairs). This is a local
  depth-difference heuristic, not a connectivity test: a steep connected surface or noisy depth
  can also trigger a split.
- **Spread to tiles without a split.** A tile without a split of its own takes the split of the
  nearest tiles that have one, up to 8 tiles (128 px) away. This extends near-layer membership
  into large objects beyond the depth-edge neighborhood, limiting far-layer leakage through
  their semi-transparent silhouettes.
  Farther away, the near layer is empty, and the passes skip its empty tiles.
- **Membership depends only on the source.** Each pixel's membership in the near layer comes from
  the four nearest tile centers, bilinearly interpolated, with a soft step as wide as the
  supported range (a surface that continues across the split, such as feet on the ground, fades
  between the layers). The mask is then closed morphologically (4 px), so thin details whose
  estimated depth fell behind stay with their surface. Source-only membership gives a consistent
  masked source image for both passes; it does not make arbitrary varying-radius filtering,
  adaptive radius selection or occlusion exactly separable.
- **Depth-guided source-radius smoothing.** Estimated depth is noisy, and with scatter-as-gather,
  neighbors with different disc sizes do not sum to unit weight. A weighted median along x/y,
  followed by a weighted mean along x/y, denoises signed log-radius over ±6 px. The median removes
  thin depth streaks that averaging alone can preserve as underblurred lines. For center and
  neighbor memberships $m,n$, the soft affinity $mn+(1-m)(1-n)$ favors the same layer without a
  hard 50% boundary. Membership alone cannot identify a surface: sharp edges and distant
  background may both be in the far layer. Each phase therefore also uses compact depth weights
  $(\max(0,1-d^2))^2$, where $d$ is the maximum signed-level difference of the original and filtered
  neighbor from the original center, divided by twelve (two octaves outside the subpixel region).
  The mean averages the cleaned values, without reintroducing original outliers; an empty weighted
  neighborhood falls back to the original center. Radii of magnitude at most 0.5 px are untouched;
  smoothing strength rises continuously to full strength at 2 px. One floating-point map serves
  both layers. This suppresses moderate depth errors, but is not reconstruction of missing or
  arbitrarily incorrect depth; the weighted median can still change discretely as ranks change.
- **Back to front.** The far layer runs first; where its coverage is below ½ (behind the near
  layer's sources), it is completed by pull-push (a pyramid of coverage-weighted averages built by
  halving, then filled from coarse to fine with bilinear upsampling). The near layer, its color
  normalized by its coverage $w$, then goes over it with opacity
  $\alpha = \min(\max(\operatorname{close}(w), m f),\ 3w)$: the coverage after a 3 px
  morphological closing, which fills narrow dips from mixed disc sizes but keeps the smooth ramp
  around a silhouette, and, on the layer's own pixels (membership $m$), the fraction $f$ of the
  layer's pixels within the pixel's own blur radius (a square window), which makes the layer
  opaque deeper inside than its blur reaches. The cap $3w$ keeps the noisy color of the kernels'
  fading tails, a ratio of two tiny sums, from showing as a sharp, speckled fringe. Layers allocate
  one `layerBuffer` containing two full-resolution arrays (composite and diagnostic data,
  16 bytes per pixel each) plus the pull-push pyramid (about one-third of one full-resolution
  array). The scratch and membership byte maps are already included in the five varying-radius
  maps above; enabling layers does not allocate two additional byte maps.
- **Cost.** Per-tile reach is stored as K + 1, with 0 for tiles that no source of the layer
  reaches, so the passes skip them; the near layer is cheap where it is empty. Pixel-integrated
  tables need no per-tap mixed-radius DC correction. Horizontal filtering shares each source
  across eight outputs for nearest lookup or four for interpolation; vertical filtering uses
  eight. The one-layer pipeline specializes away layer membership and map-selection branches.
  Depth-guided smoothing uses four dispatches (median x/y, then mean x/y).
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

Layer-smoothing regressions include an in-focus narrow triangle against a blurred background
with a separate active foreground layer, both transposed axes, both lookup modes and precisions,
subpixel radius preservation, and foreground noise reduction. A 960×600 tetra fixture checks
sharp vertices and radius continuity across soft-membership boundaries. A thin depth-streak
regression checks that an erroneous 8 px radius inside a 16 px foreground returns
to within $10^{-4}$ px of its neighbors, while a truly focused thin feature remains focused.
These are fixture-specific checks, not universal error bounds.

The **View** selector in **Varying blur** shows the composite, the depth map, the near layer's opacity (tinted,
or as gray at half scale), the near layer alone, the hole-filled far layer alone, or, with one
layer, the summed kernel weight per pixel.

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
inside the central pixel, it also produces identity. Subpixel and small-radius PSFs follow the
pixel integral.

For fixed radii this is box filtering the separable 2-D kernel over the destination pixel before
sampling. It is **kernel antialiasing**, not source-image mipmapping, subpixel geometry/depth
antialiasing, or exact variable-radius convolution. The square pixel footprint also breaks exact
radial symmetry at pixel scale. Spatially varying radii, occlusion and boundaries still require
the RGB/weight handling and the layer split, smoothing and compositing heuristics described above.

Tests check every stored node's complex sums, f32 upload error, independent clipped-segment
quadrature, radius interpolation error versus direct pixel integration, subpixel support onset,
GPU raw coverage, impulse responses and mixed-radius ramps in both lookup modes, including
high-channel and merged-polynomial flat fields. The correctness target for varying-radius taps
is direct pixel integration, not point sampling. Sampled CPU mixed DC error is about $10^{-13}$;
sampled interpolation L1 kernel error is 0.00151. These are measured errors, not universal bounds.

`/test/benchmark.html` provides an opt-in alternating GPU-timestamp comparison against captured
baseline source. It warms both pipelines, shares one device, freezes the tetra camera, and reports
per-pass medians and raw samples at radii 15 and 69 in f32 and f16. Keep other GPU workloads idle.
The scene selector also offers the default woman-in-field video from Video Depth Anything: both
RGB and depth are paused at the middle of the first frame, and shared by both renderers. Network
loading and shader warmup are outside the samples. Capture the baseline before editing sampler
revisions; select the same interpolation mode for the comparison.

Measure total GPU time independently of per-pass medians; totals include scene rendering and
preparation. Compare paired runs with the same scene, settings and device, since GPU clock drift
can make absolute timings across runs misleading.

For preset 3, pixel-integrated table storage is **478,336 bytes at radius 15** and
**2,276,576 bytes at radius 69**. Nearest and interpolated lookup share the same tables.

#### Truncation and corners

Truncating each 1-D profile at normalized distance $1 + \tau$ makes the nominal 2-D support a
square. Its corners reach $r = \sqrt{2}\,(1 + \tau)$, so the kernel must already be near zero
there. Discrete tap bounds also account for rounding and, in the varying path, the half-pixel
integration footprint. The optimizer's stop band extends far beyond the nominal corners, to about
$1 + \tau + \ln N_\text{stop}$.

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

**Population around current** replaces the population with a local search neighborhood around
the current kernel. Members get independent uniform
perturbations of up to ±5% of each parameter's normal initialization range. The width shrinks
symmetrically near search bounds; pinned parameters remain fixed. It uses the displayed **Seed**
for reproducibility and resumes only if optimization was already running. A kernel outside the
current search bounds or with incompatible pinned parameters is rejected rather than silently
changed; check the frequency-band and component settings. **New population** instead samples
the global initialization ranges. **Include current kernel in a new population**, grouped with
these two buttons, retains the exact current kernel as member zero in either mode when checked;
when unchecked, every member is sampled. The local neighborhood limits initialization only;
subsequent DE generations still use the full search bounds.
An implicit Start after loading or fine-tuning a kernel can retain it automatically; the
checkbox governs the two explicit population-initialization buttons.

**Start / Pause / Continue.** Start creates a population if none exists. Pause keeps it, and
Continue resumes it. A newly initialized population shows Start until it has run a generation.
Changing **Target and error** settings pauses and re-scores the existing
population, preserving its members, generation count and random state. This allows manual
negative-gain ramps: pause, change gain, then continue. The faint population curves use a stable
subset of members and do not reshuffle when re-scored. The highlighted best member can change
under the new objective. Convergence history is cleared because
costs under different objectives are not comparable. The frequency-band option (under
**Differential evolution**; Remez refinement and fine-tuning ignore it) is handled the same way: it
constrains future trial candidates without replacing existing members. Changes to component structure, population
size, DE parameters, search mode or seed discard the population, so the button returns to Start.

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

**Exact weights (variable projection, the default).** With **Solve weights exactly (CPU
workers)** checked, differential evolution ([`src/projected-de.js`](src/projected-de.js)) searches
only the decays and frequencies, $2N$ dimensions instead of $4N$ (or $6N$ with merged pairs). Each
candidate's weights are the minimax optimum for its rates, solved by the exchange algorithm of
the Remez refinement (below) on the DE grid, so its cost is
$\delta(\theta) = \min_w \max_j W_j |t_j - K(x_j; \theta, w)|$. The weights are thereby always exactly
balanced, and no search effort goes into the many large, cancelling weights. The same
three steps above apply to the rate genes. A trial starts the exchange from its target's
reference (the members' extremal points are similar), the first evaluation from the previous
solve's, so a solve takes about 10–20 exchanges. Solves run in float64 in a pool of Web Workers
([`src/weight-worker.js`](src/weight-worker.js)), one fewer than the logical cores (at most 8).
A candidate whose basis is singular (coinciding rates) gets infinite cost. A re-scored population
keeps its rates and re-solves its weights. A kernel included in a new population, or injected
after Remez refinement or fine-tuning, keeps its own weights: those were tuned on the dense grid,
and re-solving them on the sparser DE grid would lower the DE cost but raise the dense-grid
ripple (5 components: ±0.003592 → ±0.003619). Its weights are re-solved only if folding into the
search bounds moves its rates. In either search mode, a refined kernel that is in the population
(injected, or included in a new one) stays the current kernel until the population's best beats
its dense-grid ripple. A new population without it shows the population's own best.

This is far slower per candidate than the GPU search (roughly 1 ms in Node on the test laptop
versus 4–10 µs), but each candidate is worth much more. From random 5-component populations,
on the 200 + 200 grid, with 40,000 evaluations (3 seeds each): population 32 reached
±0.00358–±0.00360, population 64 ±0.0037–±0.0039, population 128 ±0.0055–±0.0059. Hence the population defaults to 32 in this
mode (512 for the GPU search) unless you set it. In the browser, 90 s from a random population
reached ±0.0066 with exact weights, against ±1.7 for the GPU search (both in a background tab,
which throttles both).

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

**Remez exchange refinement** ([`src/remez.js`](src/remez.js)) solves the same minimax problem on
the dense grid, in float64 on the CPU, using its structure: the profile is linear in the weights
$w = (r, i, r_1, i_1)$ and nonlinear only in the rates $\theta = (b, c)$:

$$
K(x) = \sum_k w_k \varphi_k(s; \theta), \qquad \varphi \in \{e^{-bs}\cos cs,\ e^{-bs}\sin cs,\ s e^{-bs}\cos cs,\ s e^{-bs}\sin cs\},\quad s = x^2.
$$

*1. Optimal weights for fixed rates* are a discrete linear Chebyshev problem,
$\min_w \max_j W_j |t_j - \Phi_j w|$, with the negative gain as a sign-dependent weight
($W_j = w_-$ where $e_j = t_j - K_j > 0$ in the stop band, i.e. where $K < 0$; else 1). Its LP dual is

$$
\delta = \max_\lambda \sum_j \lambda_j t_j \quad \text{s.t.} \quad \sum_j \lambda_j \Phi_j = 0,\quad \sum_j |\lambda_j| / W_j(\operatorname{sign}\lambda_j) = 1,
$$

and a basic solution has $m + 1$ nonzero $\lambda_j$ ($m$ weights): the *reference* $R$. On $R$, $\lambda$
is the left null vector $v$ of $\Phi_R$ (from a QR factorization), oriented so that $\sum v t \ge 0$;
the level is $\delta = \sum_R v_k t_k / \sum_R |v_k| / W_k$, the levelled errors are
$e_k = \delta \operatorname{sign}(v_k) / W_k$, and the weights solve $\Phi_R w = t_R - e_R$. If some grid
point $j^*$ has weighted error above $\delta$, it enters: with $\Phi_R^\mathsf{T} \nu = -\Phi_{j^*}$ and
$s^* = \operatorname{sign} e_{j^*}$, the dual vectors $s^*(\nu, 1) + \alpha v$ on $R \cup \{j^*\}$ keep the
constraints, and lowering $\alpha$ until a reference multiplier reaches zero picks the leaving
point. The new level is
$\delta' = \delta + (|e_{j^*}| - \delta / W_{j^*}) / \sum |\lambda'| / W > \delta$, so the level rises
monotonically to the optimum, where it equals the grid maximum (the certificate). This is Stiefel's
exchange, i.e. the dual simplex method. The classical Remez rule (keep alternating signs) needs a
Haar system, which complex Gaussians are not; here the signs come from the dual vector instead.
From any start, 9–23 exchanges (tens of milliseconds for 4000 grid points) solve the weights of
2–6 components.

*2. Gauss–Newton (Osborne–Watson)*: linearize in the rates around the current kernel; the unknowns
are the new weights and the rate increments, the columns are $\varphi_k$ and $\partial K / \partial \theta$.
The subproblem is again linear minimax and is solved by the same exchange, warm-started from the
previous reference. Extra rows $|\mu \tilde{\Delta\theta}_i| \le \delta$ (in units of the column's
maximum kernel change) form a trust region, with $\mu$ adapted from the ratio of actual to
predicted reduction.

*3. Second-order Remez steps.* Gauss–Newton converges quadratically only when the error has
$\dim(w, \theta) + 1$ extremal points at the ripple level. Starting from the article's 5-component set
there are 17–19 of 21, and Gauss–Newton creeps along a curved valley (4000 iterations took it
from ±0.00408 only to ±0.00376). So each iteration also tries Newton steps on the KKT system of
$\min \delta$ subject to equal ripple on the near-active extrema $k$ (band-local maxima within 5% of
the maximum):

$$
c_k(p) = \sigma_k W_k \bigl(t_k - K(s_k(p); p)\bigr) = \delta,\qquad \sum_k \lambda_k \nabla c_k = 0,\qquad \sum_k \lambda_k = 1,
$$

where $s_k(p)$ follows the continuous local extremum ($K_s = 0$) unless it is a band end, giving the
reduced Hessian $\nabla^2 c_k = -\sigma_k W_k (K_{pp} - K_{ps} K_{ps}^\mathsf{T} / K_{ss})$. With
$\dim + 1$ points this is the classical Remez second algorithm (the multipliers are then
determined by the constraints). Points that get negative multipliers are dropped, Levenberg damping
keeps steps safe, and a second-order correction (minimum-norm projections back onto
$c_k = \delta$) is essential: the equal-ripple manifold is strongly curved because large weights
cancel. A step is kept only if it lowers the grid maximum. The run stops when nothing improves.

From the article's sets (dense 2000 + 2000 grid; re-checked on 40,000 + 40,000 points, within
$10^{-6}$):

| Components | Article | Optimal weights for the article's rates | Remez refined | Iterations, time (Node) |
|---|---|---|---|---|
| 2 | ±0.07729 | ±0.07693 | ±0.07593 | 4, 0.4 s |
| 3 | ±0.02745 | ±0.02714 | ±0.02653 | 5, 0.3 s |
| 4 | ±0.01093 | ±0.01090 | ±0.009646 | 6, 0.4 s |
| 5 | ±0.004116 | ±0.004082 | ±0.003592 | 5, 0.6 s |
| 6 | ±0.001966 | ±0.001941 | ±0.001359 | 12, 2.4 s |

In the browser, preset 5 took 6.4 s (with the blur re-rendering every improvement), where 3000
gradient fine-tuning steps took 75 s to reach ±0.004084. The refined kernels also have smaller
weights (5 components: $\sum |r| + |i|$ 81 instead of 190). Refinement is local and ignores the
frequency bands. From far-off starts it can be slow or end in a worse optimum: the 6-component set
with 1% parameter noise (ripple ±1.5) reached only ±0.00197 in 50 iterations. Use differential
evolution to get near a good basin first.

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
| Article's 5-component set, Remez refined | ±0.003592 |
| Article's 6-component set, Remez refined | ±0.001359 |

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
| `src/projected-de.js`, `src/weight-worker.js`, `src/population.js` | Differential evolution with exact weights (variable projection) on CPU workers; shared population initialization |
| `src/remez.js` | Exchange (dual simplex) linear minimax solver and Remez refinement of all parameters |
| `src/blur.js` | WebGPU blur: WGSL compute and display shaders |
| `src/world.js` | Animated 3D night street ray-cast on the GPU: procedural textures, puddles with traced reflections, glossy highlights; adaptive 4× antialiasing and procedural patterns box-filtered over the pixel footprint; the circle of confusion from the z-buffer |
| `src/shapes.js` | Test scene ("Tetras"): three tetrahedra in a row against a black, starry background at infinity, ray-cast with 4× antialiasing; drag the picture to turn the view; the Scale slider (or the wheel over the picture) scales the tetrahedra about the middle one |
| `src/video.js` | RGB-D video as a depth-of-field source: color over grayscale depth, or a color video with an inferno-coded depth video |
| `tools/tum_rgbd_video.py` | Converts a [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) sequence to that format (fills depth holes, H.264) |
| `public/tum-sitting-rgbd.mp4` | 15 s of TUM RGB-D fr3/sitting_static, converted with the tool above |
| `public/woman-4511004-1080p.mp4`, `public/woman-4511004-vda-depth.mp4` | Default video: the first 13 s of Pexels video 4511004 at 1920×1080, 24 fps (H.264, CRF 18), and Video Depth Anything's inferno-coded depth for the same 13 s (1280×720, 15 fps, unmodified) |
| `src/plots.js`, `src/scene.js`, `src/main.js` | Plots, procedural HDR test scene, user interface |
| `test/smoke.js` | Checks the math and runs the optimizer, fine-tuner and Remez refinement in Node |
| `test/merged-de.js` | Experiment: DE convergence with and without merged pairs |
| `test/projected-sizes.js` | Experiment: exact-weight DE results by population size at a fixed evaluation budget |
| `vendor/jax-js-nonconsuming/` | jax-js-nonconsuming v0.12.14 TypeScript sources (MIT), compiled by Vite |
| `public/coi-serviceworker.js` | coi-serviceworker v0.1.7 (MIT), served unbundled at the site root; cross-origin requests bypass it to support video range requests |
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
depth converted to 8-bit inverse depth, and re-encoded as H.264.
`public/woman-4511004-1080p.mp4` is from Yaroslav Shuraev's
[Woman standing in the middle of green grass](https://www.pexels.com/video/woman-standing-in-the-middle-of-green-grass-4511004/)
(free under the [Pexels license](https://www.pexels.com/license/)): the first 13 s, scaled from
3840×2160 to 1920×1080 and re-encoded. `public/woman-4511004-vda-depth.mp4` is Video Depth Anything's
depth estimate for that clip, copied from its project page. The other Video Depth Anything samples
are not part of this repository: the page loads them from
[videodepthanything.github.io](https://videodepthanything.github.io/) (S. Chen et al., "Video Depth
Anything: Consistent Depth Estimation for Super-Long Videos", CVPR 2025).
