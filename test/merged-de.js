// Compare DE convergence for all-single vs merged-pair structures at equal channel count.
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { DifferentialEvolution } from "../src/de.js";
import { bounds, defaultProblem, maxError, sampleGrid } from "../src/problem.js";

await init("wasm");
defaultDevice("wasm");
for (const multiplicity of [[1, 1, 1], [2, 1], [1, 2]]) {
  const problem = { ...defaultProblem(), components: multiplicity.length, multiplicity };
  const grid = sampleGrid(problem);
  const de = new DifferentialEvolution({
    components: multiplicity.length, grid, bounds: bounds(problem), populationSize: 512, F: 0.76, CR: 0.999, seed: 3,
  });
  const trace = [];
  for (let k = 0; k < 6; k++) {
    de.step(250);
    trace.push((await de.snapshot()).bestCost.toFixed(5));
  }
  const snap = await de.snapshot();
  console.log(`${multiplicity}: ${trace.join(" ")} | best ${maxError(snap.best, grid).toFixed(6)} at`, snap.best.map((v) => +v.toFixed(3)).join(","));
  de.dispose();
}
