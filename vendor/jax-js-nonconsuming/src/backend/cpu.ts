import { AluOp, DType, dtypedArray, Kernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import type { BackendCapabilities } from "../backend";
import { Routine, runCpuRoutine } from "../routine";
import { tuneNullopt } from "../tuner";

/** Most basic implementation of `Backend` for testing. */
export class CpuBackend implements Backend {
  readonly type: Device = "cpu";
  readonly maxArgs = Infinity;
  readonly capabilities: BackendCapabilities = {
    atomicF32Add: false,
    shaderF16: false,
    sharedMemory: false,
    multiOutputKernel: true,
  };

  #buffers: Map<Slot, { ref: number; buffer: Uint8Array<ArrayBuffer> }>;
  #nextSlot: number;

  constructor() {
    this.#buffers = new Map();
    this.#nextSlot = 1;
  }

  slotCount(): number {
    return this.#buffers.size;
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    const buffer = new Uint8Array(size);
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      buffer.set(initialData);
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { buffer, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.#buffers.delete(slot);
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    return this.readSync(slot, start, count);
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }

  copyBufferToBuffer(
    src: Slot,
    srcOffset: number,
    dst: Slot,
    dstOffset: number,
    size: number,
  ): void {
    const srcBuf = this.#getBuffer(src);
    const dstBuf = this.#getBuffer(dst);
    const srcView = new Uint8Array(
      srcBuf.buffer,
      srcBuf.byteOffset + srcOffset,
      size,
    );
    const dstView = new Uint8Array(
      dstBuf.buffer,
      dstBuf.byteOffset + dstOffset,
      size,
    );
    dstView.set(srcView);
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<void>> {
    return this.prepareKernelSync(kernel);
  }

  prepareKernelSync(kernel: Kernel): Executable<void> {
    return new Executable(kernel, undefined);
  }

  async prepareRoutine(routine: Routine): Promise<Executable> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable {
    return new Executable(routine, undefined);
  }

  dispatch(
    exe: Executable<void>,
    inputs: Slot[],
    outputs: Slot[],
    _dynamicParams?: number[],
  ): void {
    if (exe.source instanceof Routine) {
      return runCpuRoutine(
        exe.source,
        inputs.map((slot) => this.#getBuffer(slot)),
        outputs.map((slot) => this.#getBuffer(slot)),
      );
    }

    const kernel = exe.source as Kernel;
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));

    // Tune each output independently
    const tunes = kernel.outputs.map((o) => {
      const tmp = Kernel.single(kernel.nargs, kernel.size, o.exp, o.reduction);
      return tuneNullopt(tmp);
    });

    // Collect used args across all outputs
    const usedArgs = new Map<number, DType>();
    for (const tune of tunes) {
      for (const exp of tune.exp.collect(
        (exp) => exp.op === AluOp.GlobalIndex,
      )) {
        usedArgs.set(exp.arg[0] as number, exp.dtype);
      }
      if (tune.epilogue) {
        for (const exp of tune.epilogue.collect(
          (exp) => exp.op === AluOp.GlobalIndex,
        )) {
          usedArgs.set(exp.arg[0] as number, exp.dtype);
        }
      }
    }

    const inputArrays = inputBuffers.map((buf, i) => {
      const dtype = usedArgs.get(i);
      if (!dtype) return null!; // This arg is unused, so we just blank it out.
      return dtypedArray(dtype, buf);
    });

    const globals = (gid: number, bufidx: number) => {
      if (gid < 0 || gid >= inputArrays.length)
        throw new Error("gid out of bounds: " + gid);
      if (bufidx < 0 || bufidx >= inputArrays[gid].length)
        throw new Error("bufidx out of bounds: " + bufidx);
      return inputArrays[gid][bufidx];
    };

    // Evaluate each output
    for (let oi = 0; oi < kernel.numOutputs; oi++) {
      const tune = tunes[oi];
      const out = kernel.outputs[oi];
      const outputArray = dtypedArray(out.dtype, outputBuffers[oi]);

      if (!out.reduction) {
        for (let i = 0; i < (kernel.size as number); i++) {
          outputArray[i] = tune.exp.evaluate({ gidx: i }, globals);
        }
      } else {
        const useKahan =
          out.reduction.dtype === DType.Float64 &&
          out.reduction.op === AluOp.Add;
        for (let i = 0; i < (kernel.size as number); i++) {
          let acc = out.reduction.identity;
          let comp = 0; // Kahan compensation
          for (let j = 0; j < (out.reduction.size as number); j++) {
            const item = tune.exp.evaluate({ gidx: i, ridx: j }, globals);
            if (useKahan) {
              const y = item - comp;
              const t = acc + y;
              comp = t - acc - y;
              acc = t;
            } else {
              acc = out.reduction.evaluate(acc, item);
            }
          }
          outputArray[i] = tune.epilogue!.evaluate({ acc, gidx: i }, globals);
        }
      }
    }
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
