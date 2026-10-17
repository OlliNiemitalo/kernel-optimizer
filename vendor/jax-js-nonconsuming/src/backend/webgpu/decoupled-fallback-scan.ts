/**
 * @file Decoupled Fallback single-dispatch prefix scan shader generator.
 *
 * Implements the Decoupled Fallback algorithm (Smith, Levien, & Owens 2024)
 * for WebGPU. Achieves O(N) work in a single dispatch without requiring
 * Forward Progress Guarantee (FPG), making it portable across all WebGPU
 * backends (NVIDIA, AMD, Intel, Apple Silicon).
 *
 * Phase 1 targets scalar binary ops: add, mul, min, max on f32.
 *
 * Descriptor packing: status (2 bits) + value (30 bits) in a single
 * atomic<u32>. For f32, the 2 LSBs of the mantissa are truncated (~4 ULPs
 * precision loss per lookback step — acceptable for f32 prefix sums).
 * u32/i32 excluded: 30-bit packing silently truncates values that exceed
 * the representable range, producing incorrect results.
 *
 * References:
 * - Smith, Levien, & Owens: "Decoupled Fallback" (2024)
 * - Thomas Smith's GPUPrefixSums (MIT): WGSL reference implementation
 */

import { AluOp, DType } from "../../alu";

export type DFScanOp = AluOp.Add | AluOp.Mul | AluOp.Min | AluOp.Max;
export type DFScanDtype = DType.Float32;

interface OpConfig {
  wgslOp: string; // e.g. "a + b"
  identity: string; // e.g. "0.0"
  /** Pack a typed value into 30 bits (stored in low 30 bits of u32). */
  packValue: string; // expression using `v` (the value)
  /** Unpack 30-bit value back to the typed value. */
  unpackValue: string; // expression using `bits` (the 30-bit u32)
}

function getOpConfig(op: DFScanOp, _dtype: DFScanDtype): OpConfig {
  // Operation expression
  let wgslOp: string;
  switch (op) {
    case AluOp.Add:
      wgslOp = "a + b";
      break;
    case AluOp.Mul:
      wgslOp = "a * b";
      break;
    case AluOp.Min:
      wgslOp = "min(a, b)";
      break;
    case AluOp.Max:
      wgslOp = "max(a, b)";
      break;
  }

  // Identity value (f32 only)
  let identity: string;
  switch (op) {
    case AluOp.Add:
      identity = "0.0";
      break;
    case AluOp.Mul:
      identity = "1.0";
      break;
    case AluOp.Min:
      identity = "bitcast<f32>(0x7F7FFFFFu)"; // MAX_FLOAT
      break;
    case AluOp.Max:
      identity = "bitcast<f32>(0xFF7FFFFFu)"; // -MAX_FLOAT
      break;
  }

  // Packing/unpacking: truncate 2 LSBs of f32 mantissa (~4 ULPs precision)
  const packValue = "(bitcast<u32>(v) >> 2u)";
  const unpackValue = "bitcast<f32>(bits << 2u)";

  return { wgslOp, identity, packValue, unpackValue };
}

/**
 * Generate a Decoupled Fallback prefix scan WGSL shader.
 *
 * Bindings:
 * - @group(0) @binding(0): input array (read-only storage)
 * - @group(0) @binding(1): output array (read-write storage)
 * - @group(0) @binding(2): descriptors array<atomic<u32>> (read-write, M entries)
 * - @group(0) @binding(3): uniform { N: u32 }
 */
export function generateDecoupledFallbackScanShader(
  op: DFScanOp,
  dtype: DFScanDtype,
  blockSize: number,
): string {
  const wgslType = "f32"; // DFScanDtype is f32-only (Phase 1)
  const storageType = wgslType; // atomic type is always u32 for descriptors
  const config = getOpConfig(op, dtype);
  const SPIN_LIMIT = 128;

  return `// Decoupled Fallback prefix scan — ${op} ${dtype} B=${blockSize}
// Single-dispatch O(N) scan without Forward Progress Guarantee.

struct Params {
  N: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${storageType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${storageType}>;
@group(0) @binding(2) var<storage, read_write> descriptors: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = ${blockSize}u;
const SPIN_LIMIT: u32 = ${SPIN_LIMIT}u;

// Descriptor flags (packed in top 2 bits)
const FLAG_X: u32 = 0u;  // Not ready
const FLAG_A: u32 = 1u;  // Aggregate ready
const FLAG_P: u32 = 2u;  // Inclusive prefix ready

var<workgroup> shmem: array<${wgslType}, ${blockSize}>;
var<workgroup> shared_prefix: ${wgslType};

fn pack_descriptor(flag: u32, v: ${wgslType}) -> u32 {
  return (flag << 30u) | ${config.packValue};
}

fn unpack_flag(packed: u32) -> u32 {
  return packed >> 30u;
}

fn unpack_value(packed: u32) -> ${wgslType} {
  let bits = packed & 0x3FFFFFFFu;
  return ${config.unpackValue};
}

fn combine(a: ${wgslType}, b: ${wgslType}) -> ${wgslType} {
  return ${config.wgslOp};
}

@compute @workgroup_size(${blockSize})
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let tidx = lid.x;
  let blockIdx = wid.x;
  let gidx = blockIdx * BLOCK_SIZE + tidx;
  let N = params.N;

  // ---- Phase 1: Load into shared memory ----
  var val: ${wgslType} = ${config.identity};
  if (gidx < N) {
    val = input[gidx];
  }
  shmem[tidx] = val;
  workgroupBarrier();

  // ---- Phase 2: Inclusive Hillis-Steele scan within workgroup ----
${generateHillisSteele(blockSize, wgslType, config.identity)}

  val = shmem[tidx];
  let block_aggregate = shmem[BLOCK_SIZE - 1u];

  // ---- Phase 3: Inter-block prefix computation ----

  // Block 0: its inclusive prefix IS its aggregate. Publish and write output.
  if (blockIdx == 0u) {
    if (tidx == 0u) {
      atomicStore(&descriptors[0], pack_descriptor(FLAG_P, block_aggregate));
    }
    if (gidx < N) {
      output[gidx] = val;
    }
    return;
  }

  // Non-zero blocks: publish aggregate, then look back.
  if (tidx == 0u) {
    atomicStore(&descriptors[blockIdx], pack_descriptor(FLAG_A, block_aggregate));
  }

  // ---- Phase 4: Decoupled Lookback with bounded spin + fallback ----
  if (tidx == 0u) {
    var prefix: ${wgslType} = ${config.identity};
    var lookback_idx = blockIdx - 1u;

    loop {
      // Bounded spin: poll descriptor until ready or timeout
      var desc: u32 = 0u;
      for (var spin: u32 = 0u; spin < SPIN_LIMIT; spin = spin + 1u) {
        desc = atomicLoad(&descriptors[lookback_idx]);
        if (unpack_flag(desc) != FLAG_X) { break; }
      }

      let flag = unpack_flag(desc);
      let value = unpack_value(desc);

      if (flag == FLAG_P) {
        // Found inclusive prefix — combine and done.
        prefix = combine(value, prefix);
        break;
      } else if (flag == FLAG_A) {
        // Found aggregate only — combine and continue looking back.
        prefix = combine(value, prefix);
        if (lookback_idx == 0u) { break; }
        lookback_idx = lookback_idx - 1u;
      } else {
        // FLAG_X after spin limit: FALLBACK — compute this block's aggregate
        // from raw input data (work-stealing).
        var fallback_agg: ${wgslType} = ${config.identity};
        for (var i: u32 = 0u; i < BLOCK_SIZE; i = i + 1u) {
          let idx = lookback_idx * BLOCK_SIZE + i;
          if (idx < N) {
            fallback_agg = combine(fallback_agg, input[idx]);
          }
        }
        // Try to claim via CAS (the descriptor might still be FLAG_X = 0)
        let packed_fallback = pack_descriptor(FLAG_A, fallback_agg);
        let cas = atomicCompareExchangeWeak(&descriptors[lookback_idx], 0u, packed_fallback);
        if (!cas.exchanged) {
          // Someone else published — use their value instead.
          let actual_flag = unpack_flag(cas.old_value);
          let actual_value = unpack_value(cas.old_value);
          prefix = combine(actual_value, prefix);
          if (actual_flag == FLAG_P) { break; }
        } else {
          // We claimed it — use our computed aggregate.
          prefix = combine(fallback_agg, prefix);
        }
        if (lookback_idx == 0u) { break; }
        lookback_idx = lookback_idx - 1u;
      }
    }

    // Publish our inclusive prefix
    let my_prefix = combine(prefix, block_aggregate);
    atomicStore(&descriptors[blockIdx], pack_descriptor(FLAG_P, my_prefix));

    shared_prefix = prefix;
  }

  // ---- Phase 5: Broadcast prefix and apply to local scan results ----
  workgroupBarrier();

  if (gidx < N) {
    output[gidx] = combine(shared_prefix, val);
  }
}
`;
}

function generateHillisSteele(
  blockSize: number,
  wgslType: string,
  identity: string,
): string {
  const rounds = Math.ceil(Math.log2(blockSize));
  let code = "";
  for (let r = 0; r < rounds; r++) {
    const stride = 1 << r;
    code += `  // Hillis-Steele round ${r + 1}/${rounds} (stride=${stride})
  {
    var temp: ${wgslType} = ${identity};
    if (tidx >= ${stride}u) {
      temp = shmem[tidx - ${stride}u];
    }
    workgroupBarrier();
    if (tidx >= ${stride}u) {
      shmem[tidx] = combine(temp, shmem[tidx]);
    }
    workgroupBarrier();
  }
`;
  }
  return code;
}
