import { bounds, defaultProblem, sampleGrid } from "../src/problem.js";
import { ProjectedEvolution } from "../src/projected-de.js";

const n = Number(process.argv[2] ?? 5), evals = Number(process.argv[3] ?? 40000);
const problem = { ...defaultProblem(), components: n, multiplicity: Array(n).fill(1) };
for (const P of (process.argv[4] ?? "32,64,128,256").split(",").map(Number)) {
  const results = [];
  for (const seed of [1, 2, 3]) {
    const de = new ProjectedEvolution({ components: n, multiplicity: problem.multiplicity, grid: sampleGrid(problem), bounds: bounds(problem), populationSize: P, F: 0.76, CR: 0.999, seed });
    de.step(Math.round(evals / P));
    results.push((await de.snapshot()).bestCost.toFixed(5));
  }
  console.log(`P=${P}: ${results.join(" ")}`);
}
