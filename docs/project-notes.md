# Project notes

Main question: can some Branch-and-Cut decisions be learned so as to reduce the effort of solving combinatorial problems exactly?

## Research questions and status

The detailed results are in [results.md](results.md).

| Question | Status on MIS |
|---|---|
| 1. Can branching decisions be learned from solver data? | Yes. A small MLP that imitates full strong branching is faster than `relpscost` on every set ([Final run](results.md#final-run)). Its trees stay larger than the expert's ([How far imitation can go](results.md#how-far-imitation-can-go)). |
| 2. Can machine learning improve cutting-plane selection? | Not on MIS. Root selection has no headroom, and a per-instance oracle over fixed separator settings gains only a few percent. A good fixed configuration takes the gain ([Cut selection at the root](results.md#cut-selection-at-the-root), [Cutting planes at the nodes](results.md#cutting-planes-at-the-nodes)). |
| 3. Are learned branching and cuts complementary? | Yes. With cuts at the nodes, the lead of the learned rule over `relpscost` erodes much less with the instance size ([Final run](results.md#final-run)). |
| 4. Do the learned policies generalize across sizes and distributions? | Across sizes: yes, without retraining ([Larger instances](results.md#larger-instances)). Across graph families: the history features transfer, but mixing families in training does not help. Only synthetic graphs have been tested so far. |
| 5. Does the saving in solver effort pay for the inference? | Yes. Features and inference take 2 to 7% of the time ([Where the time goes](results.md#where-the-time-goes)). |

## Local or global signal

Is it better to learn local decisions by imitating an expert, or to learn with an objective tied to the global cost of the search? On MIS, imitation reaches within 0 to 16% of the expert's nodes in the deployed settings. The expert's remaining lead comes from its strong branching pruning, which is too costly to recover ([Headroom of imitation](results.md#headroom-of-imitation-in-the-deployed-settings)). The question stays open for problems where the expert itself is weak.

## Next: Job-Shop Scheduling

The JSSP tests the opposite regime: in its disjunctive big-M formulation the LP bound is weak, so trees are large and cuts help little. The literature on learning for the JSSP is almost entirely heuristic, mostly learned dispatching rules. Learned branching inside a MILP solver is not covered there, and constraint programming dominates MILP on this problem. A pilot in `experiments/jssp/pilot/` measures what SCIP does on small Taillard-style instances before any data is collected.
