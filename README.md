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
  doesn't grow with the channel count. Uniform or spatially varying (tilt-shift) disc radius,
  f32 or packed 16-bit intermediates, and GPU timings and memory use per buffer.

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
installs dependencies, runs the smoke test, builds, and deploys `dist/`. The build also links the
page header to this README.

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

**Memory.** Besides the source image and the 4-lane accumulator (16 bytes per pixel each), the
only working buffer is one channel's intermediate: 4 complex lanes, 32 bytes per pixel in f32. The
"16-bit float (packed)" option stores each complex lane as two half floats in one `u32`
(`pack2x16float`): 16 bytes per pixel, with no optional GPU feature needed. At 960×600 the
intermediate is 17.6 MiB, or 8.8 MiB packed, whatever the number of channels. Keeping all
channels at once would take channels × that.

**Spatially varying radius.** With a per-pixel radius map (the "Tilt-shift" option), each pixel
gathers with taps for its own radius, in both passes. Tap tables are precomputed for a geometric
ladder of radii, six per octave from 1 pixel up to the map's maximum, and each pixel uses the
nearest level. Level 0 is an exact identity, so in-focus pixels stay perfectly sharp. When the
radius changes between rows, the vertical pass combines intermediates computed with
neighbouring levels' horizontal kernels. The resulting 2-D kernel is then a smooth blend rather
than an exact disc, which suits smoothly varying blur such as depth of field. This path does not
use register blocking, because neighbouring pixels can use different tap counts.

**Truncation and corners.** Truncating each 1-D kernel at $|t| \le K$ makes the 2-D support a
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
| `src/plots.js`, `src/scene.js`, `src/main.js` | Plots, procedural HDR test scene, user interface |
| `test/smoke.js` | Checks the math and runs the optimizer and fine-tuner in Node |
| `test/merged-de.js` | Experiment: DE convergence with and without merged pairs |
| `vendor/jax-js-nonconsuming/` | jax-js-nonconsuming v0.12.14 TypeScript sources (MIT), compiled by Vite |
| `public/coi-serviceworker.js` | coi-serviceworker v0.1.7 (MIT), served unbundled at the site root |

jax-js-nonconsuming is vendored because it is not published on npm, and its Git install
does not build on Windows.

## License

MIT for this project ([LICENSE](LICENSE)). The vendored jax-js-nonconsuming sources are MIT
licensed; see [vendor/jax-js-nonconsuming/LICENSE](vendor/jax-js-nonconsuming/LICENSE).
`public/coi-serviceworker.js` is by Guido Zuidhof and contributors, MIT licensed (see its header).
