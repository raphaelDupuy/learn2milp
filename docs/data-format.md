# Data formats

## Instances

An instance is a pair of text files:

- `NAME.mis`: the graph, with a header `MIS_INSTANCE 1`, the number of vertices `n`, the number of edges `m` and one edge `u v` per line (`u < v`);
- `NAME.mis.meta`: the problem type, the generator, its version, its seed and its parameters (`parameter.n`, `parameter.p` or `parameter.k`).

A batch of instances comes with a `manifest.csv`, one row per instance with its parameters and number of edges `m`. The formats are documented in `include/io/MISInstanceIO.hpp` and `include/io/InstanceMetadataIO.hpp`.

## Strong branching samples

For an instance `NAME.mis`, `mis-collector` writes three files to the output directory:

| File | Content |
|---|---|
| `NAME.sb.nodes.csv` | one row per sample (one branching decision) |
| `NAME.sb.candidates.csv` | one row per branching candidate of each sample |
| `NAME.sb.meta` | instance, SCIP version, settings, final status and counts |

The metadata file is written last: its presence means that both tables are complete. The two tables are joined on `sample_index`. For a given instance, limits and seed, they are reproducible byte for byte, unless the time limit is what stopped the run.

Main columns of the node table:

- `node_id`, `parent_node_id`, `node_depth`: position in the search tree;
- `lp_objective`, `primal_bound`, `dual_bound`: objective bounds at the node;
- `number_of_candidates`, `expert_selected_vertex_id`: the decision;
- `node_cutoff`: 1 when strong branching proved that the node can be pruned. The node is then not branched, and the expert selection is arbitrary among tied candidates;
- `vertex_states`: one character per vertex, indexed by vertex id.

| Code | Vertex state |
|---|---|
| `Z` / `O` | fixed to 0 / fixed to 1 by the local bounds |
| `z` / `o` | not fixed, LP value 0 / LP value 1 |
| `f` | not fixed, fractional LP value: a branching candidate |
| `u` | unknown (multi-aggregated by presolving) |

Main columns of the candidate table:

- `vertex_id`, `lp_value`, `fractionality`, `reduced_cost`: state of the candidate;
- `down_*` and `up_*`: strong branching outcome when the vertex is excluded (`x_v = 0`) or selected (`x_v = 1`): `bound`, `gain`, `infeasible`, ...;
- `score`: strong branching score, computed as in SCIP's `vanillafullstrong` rule;
- `expert_rank`: number of candidates with a strictly higher score (ties share a rank);
- `expert_selected`: 1 for the candidate the expert branched on;
- `pseudocost_down`, `pseudocost_up`, `pseudocost_count_*`, `inferences_*`, `cutoffs_*`, `conflict_score`: what the search has learned about the vertex before the node is branched, as kept by SCIP. Strong branching does not change these statistics: a rule without strong branching sees the same kind of values.

On-policy collections (`--policy-model`) have the same tables. Their metadata records the model (`branching_policy`, `policy_model`), the number of branching decisions it took and how many of them agree with the expert (`policy_decisions`, `policy_expert_agreements`).

Conventions:

- objective values (`lp_objective`, `primal_bound`, `dual_bound`, strong branching bounds and gains) are in SCIP's internal space, which is a minimization: the MIS objective appears negated. The metadata gives the map back to the original objective, `original = original_objective_intercept + original_objective_slope * internal`;
- every `*_valid` column tells whether the value next to it is meaningful;
- a node reached while the solver is stopping (time limit) is not recorded, so every sample is a complete expert decision;
- collection fails, without writing anything, if presolving merges two vertices into one LP column: a sample is never attributed to the wrong vertex.

A sample is **usable** for training when the node was branched (`node_cutoff = 0`), the expert's choice is valid (`expert_selection_valid = 1`) and strong branching evaluated every candidate; in the other samples the expert's choice is arbitrary or missing.

The exact column lists and the meaning of every field are documented in `include/io/MISStrongBranchingSampleIO.hpp` and `include/branching/MISStrongBranchingCollector.h`.

Loading a dataset in Python:

```python
import pandas as pd

nodes = pd.read_csv("NAME.sb.nodes.csv")
candidates = pd.read_csv("NAME.sb.candidates.csv")
samples = candidates.merge(nodes, on=["sample_index", "node_id"])
```

## Training tables

`mis-featurizer` writes `NAME.sb.features.csv`, one row per candidate: `sample_index`, `candidate_position`, `vertex_id`, `node_cutoff`, `expert_selection_valid`, `evaluated`, the expert's answer (`score`, `expert_rank`, `expert_selected`), then one column per feature, named after the feature (see [learning-to-branch.md](learning-to-branch.md#1-features)). The format is documented in `include/services/MISFeaturizationService.hpp`.

## Benchmark results

`mis-benchmark` writes one CSV row per run, with the columns `instance,n,m,settings,rule,seed,status,solved,nodes,lp_iterations,time_seconds,primal_bound,dual_bound,root_first_lp_bound,root_dual_bound,cuts_applied,lookahead_scored,lookahead_improving,gate_separated,gate_skipped`. The bounds are those of the size of a maximum independent set:

- `settings` names the SCIP settings, followed by `+` and the parameter file and cut selector when there are any (`default+no_node_gomory`, `study+lookahead`);
- `root_first_lp_bound` is the bound of the first LP of the root, before any cut, and `root_dual_bound` the bound at the end of the root, after its cuts (the optimum when the root closes the gap); the share of the gap closed at the root is (`root_first_lp_bound` − `root_dual_bound`) / (`root_first_lp_bound` − optimum);
- `cuts_applied` counts the cuts added to the LP over the whole solve;
- `lookahead_scored` and `lookahead_improving`, with `--cut-selector lookahead` only (0 otherwise), count the root cuts scored by the selector and those that raised the LP bound when added alone;
- `gate_separated` and `gate_skipped`, when a parameter file turns on the separation gate (`separating/misgate/freq = 1`, see [command-line-tools.md](command-line-tools.md)), count the nodes below the root where the gated separators were called and those where they were not (0 otherwise).

Results written before the root columns existed have the first 13 columns only, and those written before the gate columns the first 18.

## Node logs

With `--node-log-dir DIR`, every run of `mis-benchmark` also writes `DIR/INSTANCE__RULE__sSEED.nodes.csv`, one line per node solved, in the order they were solved (written by `cuts/MISNodeLog.h`). A node's LP is first solved without new cuts (the first LP), then with its cut loop: separation rounds, each followed by a new LP solve. It can be solved again after its cut loop, when the branching rule tightens bounds (strong branching proving a child infeasible). The values are in SCIP's transformed problem, a minimization of −(size of the set): a node is pruned when its lower bound reaches the cutoff bound, `inf` stands for an infeasible LP or one that reached the cutoff bound, and an empty value for one that was not measured.

- `run` (SCIP restarts start a new run), `node`, `parent` (0 for the root), `depth`;
- `status`: `branched`, `infeasible` (pruned, or infeasible) or `feasible` (its LP solution is a set);
- `global_lower`: global lower bound when the node is selected; `cutoff_first`, `cutoff_final`: cutoff bound at the first LP and at the end of the node;
- `lp_first`, `lp_cut_loop`, `lp_final`: LP bound after the first LP, at the end of the cut loop, and after the node's last LP solve; `lower_bound`: the node's lower bound at the end;
- `fractional`, `half`: fractional variables of the first LP solution, and how many of them are at ½;
- `separation_rounds`, `cuts`, `pool_cuts` (of which taken from the global cut pool), `zerohalf_cuts`, `oddcycle_cuts`, `clique_cuts` (cuts of these separators, from the pool or not): rounds of the cut loop and cuts applied to the LP;
- `node_seconds`: time spent on the node; `first_lp_seconds`: until the first LP is solved (propagation included); `cut_loop_seconds`: the cut loop, LP solves included; `separation_seconds` and the times of the three separators: the separators and the cut pool during the cut loop; `node_separation_seconds`: the separators and the cut pool over the whole node;
- `first_lp_iterations`, `cut_loop_iterations`: simplex iterations of the first LP and of the cut loop;
- `lp_resolves`: LP solves of the node after its cut loop.

The cut loop raised the bound of the node by `lp_cut_loop` − `lp_first`, and pruned it when `lp_cut_loop` reaches `cutoff_first` while `lp_first` did not.
