# Project notes

Main question: can some Branch-and-Cut decisions be learned so as to reduce the effort of solving combinatorial problems exactly?

## Research questions and status

The detailed results are in [results.md](results.md).

| Question | Status |
|---|---|
| 1. Can branching decisions be learned from solver data? | Yes. A small MLP that imitates full strong branching is faster than `relpscost` on every set ([Final run](results.md#final-run)). Its trees stay larger than the expert's ([How far imitation can go](results.md#how-far-imitation-can-go)). |
| 2. Can machine learning improve cutting-plane selection? | Not on MIS. Root selection has no headroom, and a per-instance oracle over fixed separator settings gains only a few percent. A good fixed configuration takes the gain ([Cut selection at the root](results.md#cut-selection-at-the-root), [Cutting planes at the nodes](results.md#cutting-planes-at-the-nodes)). |
| 3. Are learned branching and cuts complementary? | Yes. With cuts at the nodes, the lead of the learned rule over `relpscost` erodes much less with the instance size ([Final run](results.md#final-run)). |
| 4. Do the learned policies generalize across sizes and distributions? | Across sizes: yes, without retraining ([Larger instances](results.md#larger-instances)). Across graph families: the history features transfer, but mixing families in training does not help. Only synthetic graphs have been tested so far. Across problems: on the job shop, there was nothing to learn in the branching ([Job-shop pilot](results.md#job-shop-pilot)). |
| 5. Does the saving in solver effort pay for the inference? | Yes. Features and inference take 2 to 7% of the time ([Where the time goes](results.md#where-the-time-goes)). |

## Local or global signal

Is it better to learn local decisions by imitating an expert, or to learn with an objective tied to the global cost of the search? On MIS, imitation reaches within 0 to 16% of the expert's nodes in the deployed settings. The expert's remaining lead comes from its strong branching pruning, which is too costly to recover ([Headroom of imitation](results.md#headroom-of-imitation-in-the-deployed-settings)). The question stays open for problems where the expert itself is weak.

## Job-Shop Scheduling

The JSSP was chosen for the opposite regime: in its disjunctive big-M formulation the LP bound is weak. The literature on learning for the JSSP is almost entirely heuristic, and learned branching inside a MILP solver is not covered there.

A pilot on Taillard-style instances (`experiments/jssp/pilot/`) first showed a large margin for imitation: on 10×10 instances, the choices of full strong branching alone explored half the nodes of `relpscost`. The margin went away step by step. Given the optimum as first solution, the expert's choices do no better than `relpscost`, and a tabu search finds solutions almost as good as the optimum. With integer start times and one `cumulative` constraint per machine, which brings SCIP the disjunctive reasoning of constraint programming, trees shrink to about ten nodes. On 15×15 instances, the bound limits the search and no branching rule raises it. No data was collected.

What the MILP lacks on this problem is propagation and a primal heuristic, both classical. The approach used on MIS does not carry over to it.
