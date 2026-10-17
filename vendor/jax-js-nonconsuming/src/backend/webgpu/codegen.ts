// Shared codegen helpers for generating WGSL.

import { DType } from "../../alu";

/**
 * Packed leaf buffer layout — used when the number of per-leaf storage
 * bindings would exceed maxStorageBuffersPerShaderStage.  All non-const
 * inputs are packed into a single `array<T>` storage buffer (and similarly
 * for outputs), with compile-time element offsets baked into the WGSL shader.
 */
export interface PackedLeafLayout {
  /** Per-non-const-input element offset in the packed input buffer. */
  inputOffsets: number[];
  /** Total element count of the packed input buffer. */
  totalInputElems: number;
  /** Per-output element offset in the packed output buffer. */
  outputOffsets: number[];
  /** Total element count of the packed output buffer. */
  totalOutputElems: number;
  /** Common dtype for all packed arrays. */
  dtype: DType;
}

export interface ShaderInfo {
  code: string; // WGSL shader source code.
  /** Human-readable label for this shader when it corresponds to a Kernel. */
  label?: string;
  numInputs: number;
  numOutputs: number;
  hasUniform: boolean;
  passes: {
    grid: [number, number]; // Grid size (number of workgroups) in x and y.
    uniform?: Uint8Array<ArrayBuffer>; // Optional uniform value.
  }[];
  /**
   * When true, the shader is parameterized by a runtime total_size value.
   * The grid and guard check are computed from `dynamicParams[0]` at dispatch.
   */
  isSymbolic?: boolean;
  /** Workgroup size for dynamic grid computation (only set when isSymbolic).
   *  Number for 1-D, tuple for multi-dimensional.
   */
  workgroupSize?: number | [number, number?, number?];
  /**
   * When true, the shader's reduction loop uses a runtime reduce_size value
   * from `dynamicParams[1]`. The uniform buffer includes `reduce_size: u32`
   * after `total_size` (if also symbolic) or as the sole field.
   */
  hasSymbolicReduction?: boolean;
  /** Bytes of workgroup shared memory required by the shader (0 = none). */
  sharedMemoryBytes?: number;
  /** Number of constant inputs moved to @group(1) var<uniform> bindings. */
  numUniformConsts?: number;
  /** Packed leaf layout for 6+ tuple assocScan (Phase 2 leaf packing). */
  packedLeafLayout?: PackedLeafLayout;
}

const nanInfDefs = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();

/**
 * Prepend nan()/inf() helper definitions only if the shader body uses them.
 * Matches function-call syntax `nan(` / `inf(` to avoid false positives on
 * variable names like `a_nan`.
 */
export function withNanInfHeader(code: string): string {
  if (code.includes("nan(") || code.includes("inf(")) {
    // Insert after any `enable` directives at the top.
    const lines = code.split("\n");
    let insertIdx = 0;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].startsWith("enable ")) insertIdx = i + 1;
      else if (lines[i].trim() !== "") break;
    }
    lines.splice(insertIdx, 0, nanInfDefs);
    return lines.join("\n");
  }
  return code;
}

export function dtypeToWgsl(dtype: DType, storage: boolean = false): string {
  switch (dtype) {
    case DType.Bool:
      return storage ? "i32" : "bool"; // WebGPU does not support bools in buffers.
    case DType.Int32:
      return "i32";
    case DType.Uint32:
      return "u32"; // WebGPU supports uint32 in buffers.
    case DType.Float32:
      return "f32";
    case DType.Float16:
      return "f16";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function maxValueWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "1"; // Using i32 representation.
    case DType.Int32:
      return "2147483647"; // 2^31 - 1
    case DType.Uint32:
      return "4294967295u"; // 2^32 - 1
    case DType.Float32:
      return "inf()";
    case DType.Float16:
      return "f16(inf())";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Uint32) return value.toString() + "u"; // WebGPU uses 'u' suffix for uint32.
  if (dtype === DType.Float32) {
    if (Number.isNaN(value)) return "nan()";
    if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
    return "f32(" + value.toString() + ")";
  }
  if (dtype === DType.Float16) {
    if (Number.isNaN(value)) return "f16(nan())";
    if (!Number.isFinite(value))
      return value > 0 ? "f16(inf())" : "f16(-inf())";
    return "f16(" + value.toString() + ")";
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

/**
 * Emit a WGSL saturating cast from float→int, matching WASM trunc_sat semantics.
 *
 * WGSL `i32(float)` is implementation-defined for out-of-range values.
 * WASM `i32.trunc_sat_f32_s` saturates to INT32_MIN/MAX. We match that.
 *
 * For in-range float→int or any int→int/float→float, returns a plain cast.
 */
export function castSaturateWgsl(
  value: string,
  fromDtype: DType,
  toDtype: DType,
): string {
  const isFromFloat =
    fromDtype === DType.Float32 || fromDtype === DType.Float16;
  const target = dtypeToWgsl(toDtype);

  if (isFromFloat && toDtype === DType.Int32) {
    // f32 can't represent 2147483647; 2147483648.0 is exact (2^31).
    // select(falseVal, trueVal, cond): cond ? trueVal : falseVal
    return `select(select(${target}(${value}), 2147483647, ${value} >= 2147483648.0), -2147483648, ${value} < -2147483648.0)`;
  }

  if (isFromFloat && toDtype === DType.Uint32) {
    // 4294967296.0 is exact (2^32). Negative → 0.
    return `select(select(${target}(${value}), 4294967295u, ${value} >= 4294967296.0), 0u, ${value} < 0.0)`;
  }

  return `${target}(${value})`;
}

export const gridOffsetY = 16384;

export function calculateGrid(gridSize: number): [number, number] {
  let gridX = gridSize;
  let gridY = 1;
  // https://web3dsurvey.com/webgpu/limits/maxComputeWorkgroupsPerDimension
  // device.limits.maxComputeWorkgroupsPerDimension = 65535
  if (gridSize > 65535) {
    gridX = gridOffsetY;
    gridY = Math.ceil(gridSize / gridOffsetY);
  }
  return [gridX, gridY];
}
