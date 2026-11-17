# Results

Learned branching rules (see [learning-to-branch.md](learning-to-branch.md)) compared with SCIP's default rule, `relpscost`. All figures are shifted geometric means over the instances and SCIP seeds, as computed by `mis-benchmark`; models and result tables are in `experiments/`.

## Erdős–Rényi G(125, 0.1)

20 test instances and 20 transfer instances G(150, 0.05), 3 seeds, node counts (`experiments/er125_p01/`):

| Model | Features | Training samples | Test nodes | Transfer nodes |
|---|---|---|---|---|
| `relpscost` (SCIP default) | | | 3,584 | 732 |
| `mlp32` | 18 | expert | 2,711 | 733 |
| `mlp32_dagger1` | 18 | expert + on-policy | 2,606 | 699 |
| `mlp32_onpolicy1` | 18 | on-policy | 2,582 | 713 |
| `mlp32_v2` | 30 | expert + on-policy | 2,485 | 732 |
| `mlp32_h0` | 44 | expert | 2,940 | |
| `mlp32_h1` | 44 | expert + one on-policy round | 2,583 | 748 |
| `mlp32_h2` | 44 | expert + two on-policy rounds | 2,598 | 718 |

On-policy samples bring about 4% fewer nodes, on about 42 of the 60 runs of each set. The gain is small (within the instance-to-instance noise), and the offline validation metrics do not improve: with these features and this model size, the nodes visited during training are not what limits the learned rule. The training seed, measured later on the history models, moves the node counts by about 2%.

The 12 features of version 2 do not settle the matter either: `mlp32_v2` explores 4.5% fewer nodes than `mlp32_dagger1` on the test set but 4% more on the transfer set, both within the noise. A network twice as wide reaches the same validation accuracy (0.137), so the inputs, not the size of the model, are the limit. Two side results are worth keeping in mind. Training the model to maximize the strong branching score of its choice, instead of imitating the expert's choice, raised the offline score ratio (0.64 to 0.69) but gave larger trees (3,096 nodes on the test set): the score ratio must not be used to select a model. And the strong branching score of a vertex at the parent node, used alone to choose, reaches a score ratio of 0.65 on the expert's nodes and 0.83 on the on-policy ones, above every model here: what the search has already learned about a variable says more than the local structure of the graph.

The search history of version 3 (pseudo-costs, inferences, cutoffs, distance to the incumbent) raises the validation accuracy from 0.14 to 0.19 but not the performance in the solver: `mlp32_h1` and `mlp32_h2` are within the noise of `mlp32_v2`. With these features the on-policy samples do matter, since the pseudo-costs seen at a node depend on the rule that drove the search: the model trained on the expert's samples alone is the worst of all, and one on-policy round removes 12% of its nodes. A second round brings nothing more. The `mlp32_v2` samples were collected in the previous sample format and are not reproducible with the current collector; the three `mlp32_h*` models are.

Across all these models, neither the score ratio nor the accuracy measured on the validation samples predicted the size of the search trees: only the benchmark does.

## Generalization to Barabási–Albert graphs

Another graph family tells the two kinds of features apart. On 20 Barabási–Albert graphs with 200 vertices and \(k = 8\) (`data/instances/ba200_k8/test`, seeds 81 to 100, never seen during training; 3 seeds, all solved), relpscost explores 846 nodes in 6.1 s, `mlp32_v2` 1,033 nodes in 3.1 s and `mlp32_h2` 861 nodes in 2.8 s. Both learned rules stay about twice as fast as relpscost on every instance, and the model with the search history explores 15% fewer nodes than the one without (on 17 of the 20 instances), as many as relpscost: the history generalizes across families where the static features do not. Results are in `experiments/ba200_k8/results/`; the instances are generated with

```bash
./build/mis-generator batch --family barabasi_albert --n-values 200 --k-values 8 \
    --seeds $(seq -s, 81 100) --output-dir data/instances/ba200_k8/test
```

## Training on Barabási–Albert graphs

Training on this family does not do better. `experiments/ba200_k8/build_dataset.sh` builds a dataset with the same splits as `er125_p01` (training seeds 1 to 60, validation 61 to 80, test 81 to 100): the expert's samples took about 1.5 hours with 7 parallel jobs (383 s per training instance on average, 19 minutes at most), and one on-policy round (`ba200_k8_mix1`, driven by a first two-family model) as long again. Three models are compared, each trained twice with different training seeds (results of the first in `experiments/*/results/`):

| Model | Training samples | Barabási–Albert test | Erdős–Rényi test | Transfer (first seed) |
|---|---|---|---|---|
| `relpscost` | | 846 | 3,584 | 732 |
| `mlp32_h2` | Erdős–Rényi: expert + two rounds | 861 / 877 | 2,598 / 2,660 | 718 |
| `mlp32_ba1` | Barabási–Albert: expert + one round | 864 / 854 | 2,788 / 2,750 | 746 |
| `mlp32_mix1` | both | 936 / 1,000 | 2,712 / 2,684 | 742 |

The training seed moves the node counts by about 2%. The model trained on Erdős–Rényi graphs alone is as good on Barabási–Albert graphs as the one trained on them, and the model trained on both families is the worst on Barabási–Albert graphs (8% and 15% more nodes than `mlp32_h2`), although each family alone does well; why mixing hurts has not been established. More data of the family would not help either: `mlp32_ba1` trained on half of its instances (seeds 1 to 30) explores 848 and 862 nodes. No learned rule explores fewer nodes than relpscost on this family, while all of them solve about twice as fast.

## Graph neural network

A graph neural network over the candidates of the node (`--hidden 32 --graph-layers 2`, see [learning-to-branch.md](learning-to-branch.md#graph-models)) imitates the expert better than the multilayer perceptron trained on the same samples, on every validation set and at every epoch, and yet explores more nodes in SCIP. Validation metrics of the exported model; node counts on the test sets as above, two values for two training seeds:

| Model | Training samples | Validation accuracy | Validation score ratio | Barabási–Albert test | Erdős–Rényi test | Transfer |
|---|---|---|---|---|---|---|
| `mlp32_ba1` | Barabási–Albert: expert + mix1 | 0.194 | 0.645 | 864 / 854 | 2,788 / 2,750 | 746 |
| `gnn32x2_ba1` | Barabási–Albert: expert + mix1 | 0.211 | 0.662 | 946 / 981 | 2,920 | |
| `mlp32_gnn1` | Barabási–Albert: expert + mix1 + gnn1 | 0.204 | 0.659 | 874 | | |
| `gnn32x2_gnn1` | Barabási–Albert: expert + mix1 + gnn1 | 0.222 | 0.680 | 959 | | |
| `mlp32_h2` | Erdős–Rényi: expert + h1 + h2 | 0.185 | 0.671 | 861 / 877 | 2,598 / 2,660 | 718 |
| `gnn32x2_h2` | Erdős–Rényi: expert + h1 + h2 | 0.214 | 0.687 | 915 | 2,795 | 763 |

(The two `gnn1` models are validated on a set that also contains the `gnn1` samples.) On Barabási–Albert graphs, `gnn32x2_ba1` explores 8.5% more nodes than `mlp32_ba1` (95% interval [1.04, 1.13] on the ratio), on 15 of the 20 instances; on Erdős–Rényi graphs, `gnn32x2_h2` explores 7% more than `mlp32_h2` ([1.01, 1.15]).

A graph model leads the search to other nodes than the samples it was trained on, which a perceptron's on-policy round does not cover. An on-policy round driven by the graph model itself (`ba200_k8_gnn1`, collected with `scripts/collect_on_policy.sh` and `gnn32x2_ba1`: 41,150 samples, the model agreeing with the expert at 29% of the nodes) does not close the gap: trained on the same samples, the graph model explores 959 nodes and the perceptron 874 (ratio 1.087, [1.02, 1.17]). The cost of the graph model inside SCIP is small: on an idle machine, `gnn32x2_ba1` solves in 2.97 s per instance against 2.86 s for `mlp32_ba1`, and `gnn32x2_h2` in 2.83 s against 2.63 s; the solving times follow the node counts. (The other graph-model runs were timed while other computations ran: only their node counts are reported.)

This network sees only the candidates and the edges between them. The reference graph model for imitating strong branching (Gasse et al., 2019) works on the bipartite graph of all the variables and the constraints, with features on both sides; the next section tries it.

## Bipartite graph model

The bipartite model (`--hidden 32 --bipartite-rounds 1`, see [learning-to-branch.md](learning-to-branch.md#graph-models)) follows Gasse et al. (2019) on the free vertices and the edge constraints between them, without the constraint duals that the samples do not record. Trained on the same Barabási–Albert samples as `mlp32_ba1` (expert + mix1, 20 epochs, 9 minutes on a GTX 1050 Ti), it sits between the perceptron and the candidate graph model offline and explores the most nodes of the three in SCIP, while each node costs about 0.9 ms more to score:

| Model | Validation accuracy | Validation score ratio | Barabási–Albert test nodes | Time per instance |
|---|---|---|---|---|
| `mlp32_ba1` | 0.194 | 0.645 | 864 | 2.86 s |
| `gnn32x2_ba1` | 0.211 | 0.662 | 946 | 2.97 s |
| `gcnn32x1_ba1` | 0.205 | 0.653 | 966 | 4.17 s |

(times on an idle machine; ratio of `gcnn32x1_ba1` to `mlp32_ba1` 1.106, 95% interval [1.04, 1.19], more nodes on 17 of the 20 instances.)

## How far imitation can go

The expert prunes a node as soon as strong branching proves that its bound reaches the cutoff. A learned rule cannot: it does not solve the children's LPs, so SCIP creates, solves and prunes those children, each of them a node. The rule `misfullstrong_nocutoff` makes the expert's choices without this pruning, which gives the tree a perfect imitation of the expert would explore (SCIP seed 0, 20 test instances, all solved; results in `experiments/*/results/test_expert*.csv`):

| SCIP seed 0, test nodes | Erdős–Rényi | Barabási–Albert |
|---|---|---|
| expert, `misfullstrong` | 1,282 | 514 |
| expert without pruning, `misfullstrong_nocutoff` | 1,882 | 725 |
| `mlp32_h2` (Erdős–Rényi) | 2,679 | 815 |
| `mlp32_ba1` (Barabási–Albert) | | 903 |
| `relpscost` | 3,596 | 863 |

The pruning accounts for about 30% of the expert's tree (47% more nodes without it on Erdős–Rényi, 41% on Barabási–Albert): the target of imitation is 1,882 nodes, not 1,282. The best perceptron is still 42% above it on Erdős–Rényi and 12% on Barabási–Albert, so a better imitation would pay; but the models agree with the expert at about one node in five, and the graph models, closer to the expert offline, are not closer to it in the solver.

## Lookback

The lookback options (`--smoothing`, `--lookback-weight`, see [learning-to-branch.md](learning-to-branch.md#lookback)) were tried on the perceptron, with the recipes of `mlp32_h2` and `mlp32_ba1`: smoothing \(\varepsilon = 0.1\) alone (`_z`), smoothing and the Parent-as-Target term with \(\lambda = 0.2\), the choice of Gupta et al. (2022) for independent sets (`_zpat`), and the term alone (`_ypat`). Test nodes, training seed 0 (reference: two seeds):

| Variant | Erdős–Rényi test | Barabási–Albert test |
|---|---|---|
| reference | 2,598 / 2,660 | 864 / 854 |
| `_z` | 2,612 | 911 |
| `_zpat` | 2,558 | 839 |
| `_ypat` | 2,653 | 855 |

All the differences are within the noise of the training seed, and the validation accuracy does not move (0.18 to 0.19). The Parent-as-Target term does not change the behaviour it targets: at the nodes where the lookback condition holds, the models pick one of the parent's second-best candidates 24 to 25% of the time, with or without it. Its target is the model's own distribution at the parent, which says little when the model finds the parent's best candidate at one node in five; and which candidate strong branching ranked second at the parent is not known to a rule that does not run strong branching.

## Strong branching on the model's first candidates

Strong branching does what no imitation can: it proves that a node can be pruned before creating its children. With `--strong-branching K` (see [learning-to-branch.md](learning-to-branch.md#3-evaluation-inside-scip)), the learned rule runs it on the K candidates its model ranks first. Shifted geometric means over the 60 test runs, times on an idle machine, `mlp32_h2` on Erdős–Rényi graphs and `mlp32_ba1` on Barabási–Albert graphs:

| Rule | Erdős–Rényi nodes | time | Barabási–Albert nodes | time |
|---|---|---|---|---|
| SCIP with default settings | 1,537 | 11.2 s | 487 | 9.06 s |
| `relpscost` | 3,584 | 3.85 s | 846 | 5.94 s |
| learned rule | 2,598 | 2.63 s | 864 | 2.86 s |
| learned rule, K = 1 | 1,851 | 3.69 s | 612 | 3.96 s |
| learned rule, K = 2 | 1,460 | 4.32 s | 519 | 4.74 s |
| learned rule, K = 3 | 1,350 | 4.91 s | 512 | 5.55 s |
| learned rule, K = 5 | 1,224 | 6.44 s | 454 | 7.10 s |
| learned rule, K = 10 | 1,222 | 11.0 s | 471 | 12.3 s |

(All rules but the first use the study settings: presolving on, no restarts, cuts at the root only. Full strong branching explores 1,282 and 514 nodes with SCIP seed 0.)

From K = 5 on, the trees are smaller than those of SCIP with its default settings and its cuts at every node, and smaller than full strong branching's: the model's ranking and the pruning together do better than strong branching on every candidate. Each evaluated candidate costs two LP solves, so the solving time grows with K and the learned rule without strong branching stays the fastest: 32% faster than `relpscost` on Erdős–Rényi graphs and twice as fast on Barabási–Albert graphs. With K = 1, the rule beats `relpscost` on both counts on Barabási–Albert graphs (28% less time, 95% interval [0.66, 0.77] on the ratio, and 28% fewer nodes) and matches its time with half the nodes on Erdős–Rényi graphs ([0.92, 1.02]).

Against SCIP with its default settings, every rule run with the study settings is faster, which comes from the settings rather than from learning (`relpscost` alone is three times faster); with K = 5, the learned rule also explores fewer nodes, on both families.

## Larger instances

`experiments/generalization/run_benchmark.sh` evaluates the models trained on G(125, 0.1) and BA(200, 8), without retraining, on larger graphs of the same families (same mean degree for Erdős–Rényi graphs, same \(k\) for Barabási–Albert graphs), 10 instances per size, SCIP seed 0, one run at a time on an idle machine (12 h in all, every run solved; `scripts/watch_progress.sh` follows its progress log). The learned rule uses the perceptron (`mlp32_h2`, `mlp32_ba1`), alone or with strong branching on its K first candidates, the candidate graph model (`gnn32x2_h2`, `gnn32x2_ba1`) or the bipartite model (`gcnn32x1_ba1`, Barabási–Albert only). Shifted geometric means of nodes and seconds; tables in `experiments/generalization/results/`:

| Instances | SCIP, default settings | `relpscost` | MLP | MLP, K = 1 | MLP, K = 2 | GNN | GCNN |
|---|---|---|---|---|---|---|---|
| G(150, 0.083) | 4,825 · 29.3 s | 15,116 · 13.6 s | 10,642 · 10.2 s | 7,083 · 14.4 s | 5,388 · 17.0 s | 10,199 · 10.5 s | |
| G(200, 0.0625) | 93,064 · 316.5 s | 388,752 · 322.6 s | 320,583 · 311.5 s | 239,866 · 500.4 s | 169,015 · 566.2 s | 297,819 · 326.9 s | |
| BA(250, 8) | 2,301 · 29.4 s | 5,498 · 18.4 s | 4,149 · 11.5 s | 2,892 · 16.9 s | 2,474 · 22.2 s | 4,939 · 13.3 s | 5,545 · 22.1 s |
| BA(300, 8) | 6,172 · 71.3 s | 22,174 · 65.1 s | 19,821 · 54.3 s | 14,208 · 86.3 s | 11,618 · 111.8 s | 19,575 · 57.2 s | 24,428 · 104.2 s |

Ratio to `relpscost`, from the training size (20 test instances, 3 seeds) up: geometric mean over the instances of the per-instance ratio (time unshifted, nodes shifted by 100), with a paired bootstrap 95% interval:

| Family | size | MLP time | MLP nodes | GNN time | GNN nodes | GCNN time | GCNN nodes |
|---|---|---|---|---|---|---|---|
| Erdős–Rényi | 125 (training) | 0.68 [0.64, 0.72] | 0.73 [0.67, 0.80] | 0.73 [0.69, 0.77] | 0.79 [0.72, 0.86] | | |
| | 150 | 0.75 [0.67, 0.83] | 0.71 [0.61, 0.81] | 0.77 [0.69, 0.86] | 0.68 [0.57, 0.77] | | |
| | 200 | 0.97 [0.90, 1.03] | 0.83 [0.77, 0.88] | 1.01 [0.88, 1.17] | 0.77 [0.65, 0.89] | | |
| Barabási–Albert | 200 (training) | 0.47 [0.44, 0.51] | 1.02 [0.89, 1.18] | 0.49 [0.45, 0.53] | 1.11 [0.97, 1.28] | 0.68 [0.59, 0.78] | 1.13 [0.99, 1.30] |
| | 250 | 0.62 [0.57, 0.69] | 0.76 [0.68, 0.85] | 0.72 [0.63, 0.85] | 0.90 [0.80, 1.04] | 1.20 [1.05, 1.38] | 1.01 [0.93, 1.09] |
| | 300 | 0.83 [0.69, 1.00] | 0.89 [0.74, 1.08] | 0.88 [0.80, 0.97] | 0.88 [0.81, 0.96] | 1.60 [1.46, 1.76] | 1.10 [1.03, 1.18] |

The models carry over to larger graphs in node counts: the perceptron and the candidate graph model explore fewer nodes than `relpscost` on every larger set, and strong branching on the perceptron's first candidates divides them again by 1.3 to 2. The gain in time shrinks with the size and vanishes on G(200, 0.0625).

Time per node (total time over total nodes) explains it. A learned decision costs about the same at every node: 0.92, 0.92 and 0.89 ms for the perceptron on Erdős–Rényi graphs of 125, 150 and 200 vertices, 2.44, 2.31 and 2.33 ms on Barabási–Albert graphs of 200, 250 and 300 vertices. `relpscost` pays for strong branching only until its pseudo-costs are reliable, which a large tree amortizes: 0.95, 0.84 and 0.74 ms, and 4.02, 2.83 and 2.40 ms. Strong branching on the model's candidates, which pays at every node, is 1.3 to 1.8 times slower than `relpscost` on the largest sets. SCIP with its default settings, whose cuts at every node shrink the trees two to four times, catches up with the study settings in time on the largest sets.

The graph models do not generalize better than the perceptron. The candidate graph model, which explores 7 to 9% more nodes than the perceptron at the training size, explores as many on the largest sets (ratio to the perceptron 0.93 [0.79, 1.10] on G(200, 0.0625), 0.99 [0.83, 1.13] on BA(300, 8)), 19% more on BA(250, 8), and is never faster. The bipartite model scores a graph that grows with the instance: its time per node rises from 3.3 to 3.8 ms, and it is slower than `relpscost` from 250 vertices on (1.6 times at 300).

## Two more baselines

Two checks on the test sets (SCIP seed 0, 20 instances each, all solved; results in `experiments/*/results/test_pscost.csv` and `test_learned_*_default.csv`) tell what the learned rule's gain is made of. Shifted geometric means; time per node is the total time over the total number of nodes:

| Rule | Settings | Erdős–Rényi nodes | time | ms per node | Barabási–Albert nodes | time | ms per node |
|---|---|---|---|---|---|---|---|
| `pscost` | study | 6,955 | 4.27 s | 0.57 | 3,186 | 5.53 s | 1.43 |
| `relpscost` | study | 3,596 | 3.86 s | 0.96 | 863 | 5.96 s | 4.01 |
| learned rule | study | 2,679 | 2.69 s | 0.92 | 903 | 2.92 s | 2.39 |
| `relpscost` | default | 1,560 | 11.44 s | 6.95 | 492 | 9.29 s | 14.80 |
| learned rule | default | 1,372 | 7.09 s | 4.88 | 563 | 5.25 s | 8.07 |

`pscost` branches on pseudo-costs without ever running strong branching: it is the cheapest informed decision. The learned rule is faster than `pscost` on every instance (time ratio 0.63, 95% interval [0.58, 0.68], on Erdős–Rényi graphs; 0.52, [0.48, 0.57], on Barabási–Albert graphs) and explores 2.5 to 3 times fewer nodes (ratios 0.39 and 0.31). Its gain over `relpscost` is therefore not only the strong branching it saves: it recovers most of what strong branching tells, at a fraction of its cost. On Barabási–Albert graphs it builds trees of the size of `relpscost`'s with nodes 40% cheaper; on Erdős–Rényi graphs, trees 27% smaller at the same cost per node. Compared with `pscost`, computing the features and evaluating the model take about 0.35 ms per node on Erdős–Rényi graphs and 1 ms on Barabási–Albert graphs, some 40% of the learned rule's time per node (an estimate: the two rules do not solve the same nodes).

The learned rule was trained without cuts at the nodes. With SCIP's default settings, which separate cuts at the nodes, it keeps its lead over `relpscost` on every instance: time ratio 0.62 ([0.55, 0.69]) on Erdős–Rényi graphs and 0.56 ([0.49, 0.63]) on Barabási–Albert graphs, with 0.89 and 1.12 times the nodes. Branching and cutting look complementary rather than redundant. At this size, the cuts at the nodes halve the trees but multiply the time per node by five to seven, so that both rules are slower with the default settings than with the study settings.
