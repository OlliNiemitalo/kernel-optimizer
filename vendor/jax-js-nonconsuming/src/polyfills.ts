/** @file Polyfills for using this library. */

export function installExplicitResourceManagementPolyfills(
  target: typeof globalThis = globalThis,
) {
  // Polyfill for `Symbol.dispose` and `Symbol.asyncDispose`
  // https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-2.html
  (Symbol as any).dispose ??= Symbol.for("Symbol.dispose");
  (Symbol as any).asyncDispose ??= Symbol.for("Symbol.asyncDispose");

  // Polyfill for `SuppressedError` (required by the `using` keyword).
  // When a `using` block throws AND the `[Symbol.dispose]()` also throws,
  // the runtime wraps both errors in a `SuppressedError`. Safari does not
  // have this yet, so we provide a minimal shim.
  if (typeof target.SuppressedError === "undefined") {
    (target as any).SuppressedError = class SuppressedError extends Error {
      error: unknown;
      suppressed: unknown;
      constructor(error: unknown, suppressed: unknown, message?: string) {
        super(message);
        this.name = "SuppressedError";
        this.error = error;
        this.suppressed = suppressed;
      }
    };
  }

  // Polyfill for `DisposableStack` (TC39 Explicit Resource Management).
  // Safari and older browsers/runtimes lack native support.
  if (typeof target.DisposableStack === "undefined") {
    const SuppressedErrorCtor = target.SuppressedError;

    (target as any).DisposableStack = class DisposableStack {
      #disposed = false;
      #stack: (() => void)[] = [];

      #throwIfDisposed() {
        if (this.#disposed)
          throw new ReferenceError("DisposableStack already disposed");
      }

      get disposed() {
        return this.#disposed;
      }

      get [Symbol.toStringTag]() {
        return "DisposableStack";
      }

      use<T extends Disposable | null | undefined>(value: T): T {
        this.#throwIfDisposed();
        if (value == null) {
          return value;
        }
        const dispose = (value as any)[Symbol.dispose];
        if (typeof dispose !== "function") {
          throw new TypeError(
            "DisposableStack.use requires a value with [Symbol.dispose]",
          );
        }
        this.#stack.push(() => dispose.call(value));
        return value;
      }

      adopt<T>(value: T, onDispose: (value: T) => void): T {
        this.#throwIfDisposed();
        if (typeof onDispose !== "function") {
          throw new TypeError(
            "DisposableStack.adopt requires a callable disposer",
          );
        }
        this.#stack.push(() => onDispose(value));
        return value;
      }

      defer(onDispose: () => void): void {
        this.#throwIfDisposed();
        if (typeof onDispose !== "function") {
          throw new TypeError(
            "DisposableStack.defer requires a callable disposer",
          );
        }
        this.#stack.push(onDispose);
      }

      move(): DisposableStack {
        this.#throwIfDisposed();
        const next = new DisposableStack();
        (next as any).#stack = this.#stack;
        this.#stack = [];
        this.#disposed = true;
        return next;
      }

      [Symbol.dispose]() {
        if (this.#disposed) return;
        this.#disposed = true;
        let firstError: unknown;
        let hasSuppressed = false;
        for (let i = this.#stack.length - 1; i >= 0; i--) {
          try {
            this.#stack[i]();
          } catch (e) {
            if (!hasSuppressed) {
              firstError = e;
              hasSuppressed = true;
            } else {
              firstError = new SuppressedErrorCtor(
                e,
                firstError,
                "An error was suppressed during disposal.",
              );
            }
          }
        }
        this.#stack.length = 0;
        if (hasSuppressed) throw firstError;
      }

      dispose() {
        this[Symbol.dispose]();
      }
    };
  }
}

installExplicitResourceManagementPolyfills();
