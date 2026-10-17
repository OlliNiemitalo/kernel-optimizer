import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const vendor = (p) =>
  fileURLToPath(new URL(`./vendor/jax-js-nonconsuming/${p}`, import.meta.url));

// Cross-origin isolation lets jax-js use SharedArrayBuffer (parallel Wasm workers).
const isolation = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
};

export default defineConfig({
  base: "./",
  resolve: {
    alias: [
      { find: /^@hamk-uas\/jax-js-nonconsuming\/optax$/, replacement: vendor("packages/optax/src/index.ts") },
      { find: /^@hamk-uas\/jax-js-nonconsuming$/, replacement: vendor("src/index.ts") },
    ],
  },
  // Also run esbuild on .js so `using` declarations are lowered for Node 22 and older browsers.
  esbuild: { target: "es2022", include: /\.(m?[jt]s|[jt]sx)$/, exclude: /node_modules/ },
  build: { target: "es2022" },
  server: { headers: isolation },
  preview: { headers: isolation },
});
