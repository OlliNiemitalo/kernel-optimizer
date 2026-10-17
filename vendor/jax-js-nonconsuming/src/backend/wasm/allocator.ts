/** Simple tensor memory allocator for WebAssembly linear memory. */
export class WasmAllocator {
  #memory: WebAssembly.Memory;
  #headPtr: number;
  #freeLists: Map<number, number[]>;
  #allocatedBuffers: Map<number, number>; // ptr -> sizeClass
  /** Reverse index: ptr → sizeClass for freed blocks (enables top-of-heap compaction). */
  #freeBlocks: Map<number, number>;

  constructor(memory: WebAssembly.Memory) {
    this.#memory = memory;
    this.#headPtr = 64; // Address 0 is reserved for empty slices.
    this.#freeLists = new Map();
    this.#allocatedBuffers = new Map();
    this.#freeBlocks = new Map();
  }

  malloc(size: number): number {
    if (size === 0) return 0;

    const sizeClass = this.#findSizeClass(size);
    const freeList = this.#freeLists.get(sizeClass);

    let ptr: number;
    if (freeList && freeList.length > 0) {
      ptr = freeList.pop()!;
      this.#freeBlocks.delete(ptr);
      new Uint8Array(this.#memory.buffer, ptr, sizeClass).fill(0);
    } else {
      ptr = this.#bumpAlloc(sizeClass);
    }

    this.#allocatedBuffers.set(ptr, sizeClass);
    return ptr;
  }

  free(ptr: number): void {
    if (ptr === 0) return;

    const sizeClass = this.#allocatedBuffers.get(ptr);
    if (sizeClass === undefined) {
      throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
    }

    this.#allocatedBuffers.delete(ptr);

    // When all allocations are freed, reset the bump pointer entirely.
    // This reclaims ALL memory — critical for cross-shape workloads where
    // each JIT call uses different buffer sizes (the old size-class free
    // lists are useless for the new sizes).
    if (this.#allocatedBuffers.size === 0) {
      this.#headPtr = 64;
      this.#freeLists.clear();
      this.#freeBlocks.clear();
      return;
    }

    // Top-of-heap compaction: if the freed block is at the top of the
    // heap, move the bump pointer backward instead of adding to free list.
    const alignedSize = Math.ceil(sizeClass / 64) * 64;
    if (ptr + alignedSize === this.#headPtr) {
      this.#headPtr = ptr;
      // Recursively compact: check if the block now at the top is also free.
      this.#compactTop();
      return;
    }

    const freeList = this.#freeLists.get(sizeClass);
    if (freeList) freeList.push(ptr);
    else this.#freeLists.set(sizeClass, [ptr]);
    this.#freeBlocks.set(ptr, sizeClass);
  }

  /** Remove freed blocks from the top of the heap by walking downward. */
  #compactTop(): void {
    // Scan free blocks for one that ends exactly at #headPtr.
    // This is O(free blocks) but only runs during compaction cascades.
    let found = true;
    while (found) {
      found = false;
      for (const [freePtr, freeSizeClass] of this.#freeBlocks) {
        const alignedSize = Math.ceil(freeSizeClass / 64) * 64;
        if (freePtr + alignedSize === this.#headPtr) {
          this.#headPtr = freePtr;
          this.#freeBlocks.delete(freePtr);
          // Remove from the size-class free list too.
          const list = this.#freeLists.get(freeSizeClass);
          if (list) {
            const idx = list.indexOf(freePtr);
            if (idx !== -1) list.splice(idx, 1);
            if (list.length === 0) this.#freeLists.delete(freeSizeClass);
          }
          found = true;
          break; // restart scan since #headPtr changed
        }
      }
    }
  }

  #bumpAlloc(size: number): number {
    const ptr = this.#headPtr;
    // Use Math.ceil (not bitwise & -64) to avoid int32 overflow above 2 GiB.
    size = Math.ceil(size / 64) * 64;
    this.#headPtr += size;
    if (ptr + size > this.#memory.buffer.byteLength) {
      // Note: 4 GiB = max memory32 size
      // https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html
      const needed = ptr + size;
      if (needed > 0xffff_ffff) {
        throw new RangeError(
          `WASM allocation would exceed 4 GiB limit (need ${(needed / 2 ** 30).toFixed(1)} GiB)`,
        );
      }
      // Use >>> (unsigned right shift) — signed >> wraps negative above 2 GiB.
      this.#memory.grow(
        ((needed + 65535) >>> 16) - (this.#memory.buffer.byteLength >>> 16),
      );
    }
    return ptr;
  }

  #findSizeClass(size: number): number {
    // Small sizes: 64-byte increments from 64 to 512.
    if (size <= 512) {
      return (size + 63) & -64;
    }
    // Medium sizes: 768 (512+256), then 256-byte increments from 1024 to 2048.
    if (size <= 2048) {
      return (size + 511) & -512;
    }
    // Large sizes: powers of 2 from 4 KiB to 64 KiB.
    if (size <= 65536) {
      let sizeClass = 4096;
      while (sizeClass < size) sizeClass *= 2;
      return sizeClass;
    }
    // Very large sizes: 64 KiB increments starting from 128 KiB.
    // Use Math.ceil (not bitwise & -65536) to avoid int32 overflow above 2 GiB.
    return Math.ceil(size / 65536) * 65536;
  }

  // Debug methods
  getStats(): { totalAllocated: number; freeListSizes: Map<number, number> } {
    const freeListSizes = new Map<number, number>();
    for (const [sizeClass, freeList] of this.#freeLists) {
      if (freeList.length > 0) {
        freeListSizes.set(sizeClass, freeList.length);
      }
    }

    return {
      totalAllocated: this.#headPtr,
      freeListSizes,
    };
  }
}
