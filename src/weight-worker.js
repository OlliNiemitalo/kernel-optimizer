// Web Worker for ProjectedEvolution: solves candidates' optimal weights.
import { WeightSolver, solveCandidate } from "./remez.js";

let solver = null;
self.onmessage = ({ data }) => {
  if (data.type === "problem") solver = new WeightSolver(data.grid, data.multiplicity);
  else self.postMessage({ id: data.id, out: data.items.map((item) => solveCandidate(solver, item)) });
};
