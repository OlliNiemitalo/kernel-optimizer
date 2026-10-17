// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { DType, isFloatDtype } from "../alu";
import {
  absolute,
  add,
  arange,
  Array,
  ArrayLike,
  clip,
  einsum,
  exp,
  expandDims,
  expm1,
  less,
  log,
  max,
  maximum,
  negative,
  onesLike,
  reciprocal,
  sqrt,
  square,
  squeeze,
  tanh,
  tile,
  where,
  zerosLike,
} from "./numpy";
import { eye, fudgeArray, tri } from "../frontend/array";
import {
  type Axis,
  erfc,
  type ReduceOpts,
  shrink,
  stopGradient,
} from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { Pair } from "../shape";
import { checkAxis, deepEqual, normalizeAxis } from "../utils";

/**
 * Rectified Linear Unit (ReLU) activation function:
 * `relu(x) = max(x, 0)`.
 */
export function relu(x: ArrayLike): Array {
  return maximum(x, 0);
}

/**
 * Rectified Linear Unit 6 (ReLU6) activation function:
 * `relu6(x) = min(max(x, 0), 6)`.
 */
export function relu6(x: ArrayLike): Array {
  return clip(x, 0, 6);
}

/**
 * Sigmoid activation function, computed element-wise:
 * `sigmoid(x) = 1 / (1 + exp(-x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Sigmoid_function
 */
export function sigmoid(x: ArrayLike): Array {
  using neg = negative(x);
  using e = exp(neg);
  using sum = e.add(1);
  return reciprocal(sum);
}

/**
 * Softplus activation function:
 * `softplus(x) = log(1 + exp(x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Softplus
 */
export function softplus(x: ArrayLike): Array {
  using e = exp(x);
  using sum = e.add(1);
  return log(sum);
}

/**
 * @function
 * Sparse plus function:
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x+1)**2 / 4`
 * - When `x >= 1`: `x`
 */
export const sparsePlus = jit((x: Array): Array => {
  using leNeg1 = x.lessEqual(-1);
  using lt1 = x.less(1);
  using shifted = x.add(1);
  using sq = square(shifted);
  using scaled = sq.mul(0.25);
  using inner = where(lt1, scaled, x);
  return where(leNeg1, 0, inner);
});

/**
 * @function
 * Sparse sigmoid activation function.
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x + 1) / 2`
 * - When `x >= 1`: `1`
 */
export const sparseSigmoid = jit((x: Array): Array => {
  using shifted = x.add(1);
  using scaled = shifted.mul(0.5);
  return clip(scaled, 0, 1);
});

/**
 * Soft-sign activation function, computed element-wise:
 * `softsign(x) = x / (|x| + 1)`.
 */
export function softSign(x: ArrayLike): Array {
  x = fudgeArray(x);
  using absX = absolute(x);
  using denom = absX.add(1);
  return x.div(denom);
}

/**
 * @function
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export const silu = jit(function silu(x: Array) {
  using sig = sigmoid(x);
  return x.mul(sig);
});

export { silu as swish };

/**
 * Log-sigmoid activation function, computed element-wise:
 * `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
 */
export function logSigmoid(x: ArrayLike): Array {
  using neg = negative(x);
  using sp = softplus(neg);
  return negative(sp);
}

/**
 * @function
 * Identity activation function. Returns the argument unmodified.
 */
export const identity = fudgeArray;

/** Leaky rectified linear (ReLU) activation function */
export function leakyRelu(
  x: ArrayLike,
  negativeSlope: ArrayLike = 0.01,
): Array {
  x = fudgeArray(x);
  using cond = less(x, 0);
  using scaled = x.mul(negativeSlope);
  return where(cond, scaled, x);
}

/** Hard sigmoid activation function: `relu6(x+3)/6`. */
export function hardSigmoid(x: ArrayLike): Array {
  using sum = add(x, 3);
  using r = relu6(sum);
  return r.mul(1 / 6);
}

/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
export function hardSilu(x: ArrayLike): Array {
  x = fudgeArray(x);
  using hs = hardSigmoid(x);
  return x.mul(hs);
}

export { hardSilu as hardSwish };

/** Hard tanh activation function: `clip(x, -1, 1)`. */
export function hardTanh(x: ArrayLike): Array {
  return clip(x, -1, 1);
}

/**
 * Exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
 */
export function elu(x: ArrayLike, alpha: ArrayLike = 1.0): Array {
  x = fudgeArray(x);
  using cond = less(x, 0);
  using e = exp(x);
  using em1 = e.sub(1);
  using scaled = em1.mul(alpha);
  return where(cond, scaled, x);
}

/**
 * Continuously-differentiable exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
 */
export function celu(x: ArrayLike, alpha: ArrayLike = 1.0): Array {
  x = fudgeArray(x);
  using cond = less(x, 0);
  using ratio = x.div(alpha);
  using e = exp(ratio);
  using em1 = e.sub(1);
  using scaled = em1.mul(alpha);
  return where(cond, scaled, x);
}

/**
 * @function
 * Scaled exponential linear unit activation.
 *
 * Computes the element-wise function:
 * `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
 *
 * Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
 */
export const selu = jit(function selu(x: Array) {
  const alpha = 1.6732632423543772;
  const lambda = 1.0507009873554805;
  using lt = x.less(0);
  using em1 = expm1(x);
  using branch = em1.mul(alpha);
  using selected = where(lt, branch, x);
  return selected.mul(lambda);
});

/**
 * @function
 * Gaussion error linear unit (GELU) activation function.
 *
 * This is computed element-wise. There are two variants depending on whether
 * `approximate` is set (default true):
 *
 * - Approximate: `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
 * - Exact: `gelu(x) = x * 0.5 * erfc(-x / sqrt(2))`
 *
 * Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
 */
export const gelu = jit(
  function gelu(x: Array, opts?: { approximate?: boolean }): Array {
    if (opts?.approximate ?? true) {
      const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
      using x2 = x.mul(x);
      using quadTerm = x2.mul(0.044715);
      using poly = quadTerm.add(1);
      using polyScaled = x.mul(poly);
      using tanhArg = polyScaled.mul(SQRT_2_OVER_PI);
      using tanhVal = tanh(tanhArg);
      using gate = tanhVal.add(1);
      using halfX = x.mul(0.5);
      return halfX.mul(gate);
    } else {
      using scaled = x.mul(Math.SQRT1_2);
      using negScaled = negative(scaled);
      using erfComp = erfc(negScaled);
      using halfX = x.mul(0.5);
      return halfX.mul(erfComp);
    }
  },
  { staticArgnums: [1] },
);

/**
 * Gated linear unit (GLU) activation function.
 *
 * Splits the `axis` dimension of the input into two halves, a and b, then
 * computes `a * sigmoid(b)`.
 */
export function glu(x: ArrayLike, axis: number = -1): Array {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const size = x.shape[axis];
  if (size % 2 !== 0) {
    throw new Error(
      `glu: axis ${axis} of shape (${x.shape}) does not have even length`,
    );
  }
  const slice = x.shape.map<Pair>((a) => [0, a]);
  using a = shrink(x, slice.toSpliced(axis, 1, [0, size / 2])) as Array;
  using b = shrink(x, slice.toSpliced(axis, 1, [size / 2, size])) as Array;
  using sig = sigmoid(b);
  return a.mul(sig);
}

/**
 * Squareplus activation function.
 *
 * Computes the element-wise function:
 * `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
 */
export function squareplus(x: ArrayLike, b: ArrayLike = 4.0): Array {
  x = fudgeArray(x);
  using sq = square(x);
  using sumSq = sq.add(b);
  using sr = sqrt(sumSq);
  using sum = x.add(sr);
  return sum.mul(0.5);
}

/**
 * Mish activation function.
 *
 * Computes the element-wise function:
 * `mish(x) = x * tanh(softplus(x))`
 */
export function mish(x: ArrayLike): Array {
  x = fudgeArray(x);
  using sp = softplus(x);
  using t = tanh(sp);
  return x.mul(t);
}

/**
 * Softmax function. Computes the function which rescales elements to the range
 * [0, 1] such that the elements along `axis` sum to 1.
 *
 * If `axis` is not specified, it defaults to the last axis.
 *
 * Reference: https://en.wikipedia.org/wiki/Softmax_function
 */
export function softmax(x: ArrayLike, axis: Axis = -1): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) {
    return onesLike(x); // scalar case, return ones
  }

  using xMax = max(x, axis, { keepdims: true });
  using sg = stopGradient(xMax);
  using shifted = x.sub(sg);
  using unnormalized = exp(shifted);
  using denom = unnormalized.sum(axis, { keepdims: true });
  return unnormalized.div(denom);
}

/**
 * Log-Softmax function.
 *
 * Computes the logarithm of the `softmax` function, which rescales elements to
 * the range [-infinity, 0).
 *
 * If `axis` is not specified, it defaults to the last axis.
 */
export function logSoftmax(x: ArrayLike, axis: Axis = -1): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) {
    return zerosLike(x); // scalar case, return log(1)
  }

  using xMax = max(x, axis, { keepdims: true }); // keep dims
  using sg = stopGradient(xMax);
  using shifted = x.sub(sg);
  using expShifted = exp(shifted);
  using sumExp = expShifted.sum(axis, { keepdims: true });
  using shiftedLogsumexp = log(sumExp);
  return shifted.sub(shiftedLogsumexp);
}

/**
 * Log-sum-exp reduction. Also a multivariate version of `softplus`.
 *
 * If no axis is specified, the reduction is performed over all elements. This
 * convention differs from `jax.nn.logSoftmax()`.
 *
 * Reference: https://en.wikipedia.org/wiki/LogSumExp
 */
export function logsumexp(
  x: ArrayLike,
  axis: Axis = null,
  opts?: ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;

  using rawMax = max(x, axis, { keepdims: true });
  using xMax = stopGradient(rawMax) as Array;
  using shifted = x.sub(xMax);
  using expShifted = exp(shifted);
  using sumExp = expShifted.sum(axis, { keepdims: true });
  using logSum = log(sumExp);
  if (opts?.keepdims) return xMax.add(logSum);
  using resultToSqueeze = xMax.add(logSum);
  return squeeze(resultToSqueeze, axis);
}

/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
export function logmeanexp(
  x: ArrayLike,
  axis: Axis = null,
  opts?: ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  using lse = logsumexp(x, axis, opts);
  return lse.sub(Math.log(n));
}

/**
 * Standardizes input to zero mean and unit variance.
 *
 * By default, this is computed over the last axis. You can pass in a different
 * axis, or `null` to standardize over all elements.
 *
 * Epsilon is added to denominator, it defaults to `1e-5` for stability.
 */
export function standardize(
  x: ArrayLike,
  axis: Axis = -1,
  opts: {
    mean?: ArrayLike;
    variance?: ArrayLike;
    epsilon?: ArrayLike;
  } = {},
) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;

  using d = new DisposableStack();
  const mu =
    opts.mean !== undefined
      ? fudgeArray(opts.mean)
      : (() => {
          const m = x.mean(axis, { keepdims: true });
          d.use(m);
          return m;
        })();

  // Like JAX, we'll use the Var[X] = E[X^2] - (E[X])^2 formula for this one.
  // It's supposed to be better in the case of neural network activations.
  let sigma2: Array;
  if (opts.variance !== undefined) {
    sigma2 = fudgeArray(opts.variance);
  } else {
    const sq = d.use(square(x));
    const sqMean = d.use(sq.mean(axis, { keepdims: true }));
    const muSq = d.use(square(mu));
    sigma2 = d.use(sqMean.sub(muSq));
  }

  const centered = d.use(x.sub(mu));
  const denom = d.use(sigma2.add(opts.epsilon ?? 1e-5));
  const sqrtDenom = d.use(sqrt(denom));
  return centered.div(sqrtDenom);
}

/**
 * One-hot encodes the given indices.
 *
 * Each index in the integer input `x` is encoded as a vector of zeros of length
 * `numClasses`, with a 1 at the index position specified by its value.
 *
 * ```js
 * import { nn, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
 *
 * using x = np.array([1, 1, 2], { dtype: np.int32 });
 * using y = nn.oneHot(x, 3);
 * // Output:
 * // [[0, 1, 0],
 * //  [0, 1, 0],
 * //  [0, 0, 1]]
 * ```
 */
export function oneHot(x: Array, numClasses: number): Array {
  if (isFloatDtype(x.dtype)) {
    throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
  }
  return eye(numClasses, undefined, { device: x.device }).slice(x);
}

/**
 * Scaled dot product attention (SDPA).
 *
 * Computes `softmax((Q @ K^T) / sqrt(d) + bias) @ V`, where `Q` is the query,
 * `K` is the key, `V` is the value, and `d` is the dimensionality of each key
 * and query vector.
 *
 * Multi-query attention is applied when input `key` and `value` tensors have
 * fewer heads than `query`.
 *
 * We use the following uppercase letters to denote array shapes:
 * - `B` = batch size
 * - `S` = length of key/value sequences (source)
 * - `L` = length of query sequences
 * - `N` = number of attention heads
 * - `H` = dimensionality of each attention head
 * - `K` = number of key/value heads (for grouped-query attention)
 *
 * The batch size `B` may be omitted, which is equivalent to `B = 1`. In this
 * case it must be omitted from all inputs.
 *
 * @param query - Query array; shape `[B, L, N, H]`
 * @param key - Key array; shape `[B, S, K, H]`
 * @param value - Value array; same shape as `key`
 * @param opts.bias - Optional bias to add to the attention logits; shape
 *   `[B, N, L, S]` or broadcastable to it.
 * @param opts.mask - Optional mask to apply to the attention logits; should be
 *   a boolean array broadcastable to `[B, N, L, S]`, where `true` indicates
 *   the element should take part in attention.
 * @param opts.scale - Scaling factor override, default is `1 / sqrt(H)`.
 * @param opts.isCausal - If true, applies a casual mask.
 * @param opts.querySeqLengths - Optional sequence lengths for the queries;
 *   shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.keyValueSeqLengths - Optional sequence lengths for the keys and
 *   values; shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.localWindowSize - If specified, applies a local attention window
 *   of the given size. Can be a single number or a tuple `[left, right]`.
 *
 * @returns The result of the attention operation; shape is the same as query
 *   `[B, L, N, H]`, or `[L, N, H]` if `B` is omitted.
 */
export function dotProductAttention(
  query: ArrayLike,
  key: ArrayLike,
  value: ArrayLike,
  opts: {
    bias?: ArrayLike;
    mask?: ArrayLike;
    scale?: number;
    isCausal?: boolean;
    querySeqLengths?: ArrayLike;
    keyValueSeqLengths?: ArrayLike;
    localWindowSize?: number | [number, number];
  } = {},
): Array {
  query = fudgeArray(query);
  key = fudgeArray(key);
  value = fudgeArray(value);

  if (
    (query.ndim !== 3 && query.ndim !== 4) ||
    query.ndim !== key.ndim ||
    query.ndim !== value.ndim
  )
    throw new Error(
      `dotProductAttention: expected all tensors to have rank 3 or 4, ` +
        `got Q=${query.aval}, K=${key.aval}, V=${value.aval}`,
    );
  if (!deepEqual(key.shape, value.shape))
    throw new Error(
      `dotProductAttention: key and value shapes must match, ` +
        `got K=${key.shape}, V=${value.shape}`,
    );

  const isRank3 = query.ndim === 3;
  using d = new DisposableStack();
  if (isRank3) {
    query = d.use(expandDims(query, 0));
    key = d.use(expandDims(key, 0));
    value = d.use(expandDims(value, 0));
  }

  const [B, L, N, H] = query.shape;
  if (key.shape[0] !== B || key.shape[3] !== H)
    throw new Error(
      `dotProductAttention: query and key shapes mismatch, ` +
        `got Q=${query.aval}, K=${key.aval}`,
    );

  const S = key.shape[1];
  const K = key.shape[2];

  if (N < K || (N != K && N % K !== 0))
    throw new Error(
      `dotProductAttention: number of query heads N=${N} must be ` +
        `divisible by number of key/value heads K=${K} for GQA`,
    );
  const G = N / K; // number of query groups
  key = d.use(tile(key, [1, 1, G, 1]));
  value = d.use(tile(value, [1, 1, G, 1]));

  const scale = opts.scale ?? 1 / Math.sqrt(H);
  const rawScores = d.use(einsum("BLNH,BSNH->BNLS", query, key));
  let scores = rawScores.mul(scale);
  if (opts.bias !== undefined) {
    d.use(scores);
    scores = scores.add(opts.bias);
  }
  if (opts.mask !== undefined) {
    d.use(scores);
    scores = where(opts.mask, scores, -Infinity);
  }
  if (opts.isCausal) {
    const causalMask = d.use(tri(L, S, 0, { dtype: DType.Bool }));
    d.use(scores);
    scores = where(causalMask, scores, -Infinity);
  }
  if (opts.localWindowSize !== undefined) {
    const [before, after] =
      typeof opts.localWindowSize === "number"
        ? [opts.localWindowSize, opts.localWindowSize]
        : opts.localWindowSize;
    if (
      before < 0 ||
      after < 0 ||
      !Number.isInteger(before) ||
      !Number.isInteger(after)
    ) {
      throw new Error(
        `dotProductAttention: localWindowSize values must be non-negative, ` +
          `got ${opts.localWindowSize}`,
      );
    }
    const triAfter = d.use(tri(L, S, after, { dtype: DType.Bool }));
    const triBefore = d.use(tri(L, S, -before - 1, { dtype: DType.Bool }));
    const triBeforeNot = d.use(triBefore.notEqual(true));
    const localMask = d.use(triAfter.mul(triBeforeNot));
    d.use(scores);
    scores = where(localMask, scores, -Infinity);
  }
  if (opts.querySeqLengths !== undefined) {
    const sl = d.use(expandDims(opts.querySeqLengths, [-1, -2, -3]));
    const ar = d.use(arange(L));
    const arR = d.use(ar.reshape([1, 1, L, 1]));
    const cond = d.use(arR.less(sl));
    d.use(scores);
    scores = where(cond, scores, -Infinity);
  }
  if (opts.keyValueSeqLengths !== undefined) {
    const sl = d.use(expandDims(opts.keyValueSeqLengths, [-1, -2, -3]));
    const ar = d.use(arange(S));
    const arR = d.use(ar.reshape([1, 1, 1, S]));
    const cond = d.use(arR.less(sl));
    d.use(scores);
    scores = where(cond, scores, -Infinity);
  }
  d.use(scores);
  const attn = d.use(softmax(scores, -1)); // BNLS
  const out = einsum("BNLS,BSNH->BLNH", attn, value);
  if (isRank3) {
    d.use(out);
    return out.reshape([L, N, H]);
  }
  return out;
}
