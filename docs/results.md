# Results

Learned branching rules (see [learning-to-branch.md](learning-to-branch.md)) compared with SCIP's default rule, `relpscost`. All figures are shifted geometric means over the instances and SCIP seeds, as computed by `mis-benchmark`; models and result tables are in `experiments/`. The last two sections turn to the cutting planes: which separators are worth calling at the nodes, and what a better selection of the cuts at the root would be worth.

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

## Cutting planes at the nodes

The study settings separate cuts at the root only, so that branching is measured alone. SCIP's default settings also separate at the nodes, which halves the trees but makes SCIP slower at the training sizes (previous section). `experiments/cuts/run_ablation.sh` measures which separators are worth calling at the nodes. It runs `mis-benchmark` with SCIP's default settings and one of the parameter files of `experiments/cuts/settings/` (`--parameters`), SCIP seed 0, one run at a time on an idle machine, on the 20 test instances of each family and the 10 instances of each larger set; every run is solved, and the tables are in `experiments/cuts/results/`. The configurations differ from SCIP's default settings as follows:

- `no_node_gomory`: Gomory cuts at the root only;
- `zerohalf_every_node`: {0,½}-cuts (zero-half separator) at every node, every other separator at the root only;
- `oddcycle`: zero-half cuts at depths 10, 40 and 160 as by default (SCIP calls its separators of frequency 10 at depths that grow fourfold), the odd-cycle separator, off by default, at the same depths, every other separator at the root only.

Ratios to the study settings with `relpscost`, time (unshifted) · nodes (shifted by 100), geometric means of the per-instance ratios. With the study settings, `relpscost` takes 3.83 s and 3,596 nodes on G(125, 0.1), 13.1 s and 15,116 nodes on G(150, 0.083), 322 s and 388,752 nodes on G(200, 0.0625), 5.90 s and 863 nodes on BA(200, 8), 18.0 s and 5,498 nodes on BA(250, 8), 64.8 s and 22,174 nodes on BA(300, 8):

| Configuration | G(125, 0.1) | G(150, 0.083) | G(200, 0.0625) | BA(200, 8) | BA(250, 8) | BA(300, 8) |
|---|---|---|---|---|---|---|
| SCIP, default settings | 2.97 · 0.45 | 2.19 · 0.32 | 0.98 · 0.24 | 1.55 · 0.61 | 1.60 · 0.43 | 1.10 · 0.28 |
| `no_node_gomory` | 0.95 · 0.42 | 0.84 · 0.34 | 0.60 · 0.23 | 0.95 · 0.62 | 0.96 · 0.40 | 0.77 · 0.30 |
| `zerohalf_every_node` | 1.02 · 0.34 | 0.89 · 0.23 | 0.71 · 0.14 | 0.99 · 0.49 | 0.93 · 0.29 | 0.84 · 0.17 |
| `oddcycle` | 1.01 · 0.33 | 0.93 · 0.25 | 0.55 · 0.14 | 1.03 · 0.43 | 0.85 · 0.19 | 0.68 · 0.12 |
| learned rule, study settings | 0.69 · 0.75 | 0.77 · 0.71 | 0.97 · 0.82 | 0.48 · 1.04 | 0.63 · 0.76 | 0.83 · 0.89 |
| learned rule, `no_node_gomory` | 0.63 · 0.38 | 0.56 · 0.27 | 0.48 · 0.20 | 0.44 · 0.68 | 0.47 · 0.32 | 0.49 · 0.22 |

The reference runs (study settings, default settings, learned rule with the study settings) are those of the previous sections, made on other days. On the 5 instances of G(150, 0.083) and BA(250, 8) where they were made again, the trees were the same and the times 1 to 15% shorter (7 and 3% on average): a difference of a few percent with a reference is not significant, and the conclusions below do not rest on one.

**Gomory cuts at the nodes cost time and save nothing.** Without them, the trees keep their size (nodes of `no_node_gomory` over those of the default settings: 0.94 to 1.05, every interval but one containing 1) and the time falls by 30 to 68%, on every instance (time ratio 0.32 [0.30, 0.34] on G(125, 0.1), up to 0.70 [0.64, 0.76] on BA(300, 8)). This is a setting to change once, not a decision to learn.

**Cuts at the nodes pay off more as the graphs grow.** They make the trees 1.6 to 8 times smaller, but at the training sizes they cost about what they save: without Gomory cuts at the nodes, every configuration takes about the time of the study settings (0.95 to 1.03) on G(125, 0.1) and BA(200, 8). On the larger sets they are faster: 0.84 to 0.93 of the study time on G(150, 0.083), 0.55 to 0.71 on G(200, 0.0625), 0.68 to 0.84 on BA(300, 8). The zero-half separator does the work at the nodes. On a pilot (5 instances of G(150, 0.083) and BA(250, 8)), the cut pool alone keeps 60 and 84% of the study settings' nodes, and zero-half cuts at depths 10, 40 and 160 bring this down to 34 and 31%; zero-half cuts every 10 levels and clique cuts at every node (SCIP: root only) are within the noise of 5 instances in time (0.79 to 0.97 of the study time for all the variants). The odd-cycle separator gives the smallest trees on Barabási–Albert graphs and the best time on the three largest sets, G(200, 0.0625), BA(250, 8) and BA(300, 8); `no_node_gomory` is the fastest on the smaller ones.

**The best configuration depends on the instances, but choosing it per instance would gain little.** The per-instance best of the five configurations with `relpscost` (study settings, default settings, `no_node_gomory`, `zerohalf_every_node`, `oddcycle`) is 1 to 7% faster than the best single one of each set, an optimistic figure since it takes the minimum of five noisy times. At the training sizes, `oddcycle` takes the time of the study settings with two to three times fewer nodes: over a whole instance, the cost of the cuts and their savings balance out, which leaves open where in the tree they pay off.

**The learned rule and the cuts at the nodes add up.** With `no_node_gomory`, the learned rule is the fastest configuration on every set, at 0.44 to 0.63 of the study time and 0.46 to 0.87 of the time of the best configuration with `relpscost` (faster on every instance of the largest sets). Its lead over `relpscost` fades much less with the size of the graphs. Time ratio to `relpscost`, both with `no_node_gomory`: 0.66 [0.61, 0.71], 0.67 [0.60, 0.74] and 0.81 [0.75, 0.86] on Erdős–Rényi graphs of 125, 150 and 200 vertices (0.69, 0.77 and 0.97 with the study settings), 0.46 [0.42, 0.51], 0.49 [0.45, 0.53] and 0.64 [0.54, 0.78] on Barabási–Albert graphs of 200, 250 and 300 vertices (0.48, 0.63 and 0.83). A likely reason: the cuts add rows to the node LPs, which makes every strong branching LP of `relpscost` dearer, while a learned decision costs the same. On the larger sets, the two gains combine better than their product (0.48 against 0.97 × 0.60 = 0.58 on G(200, 0.0625), 0.49 against 0.83 × 0.77 = 0.64 on BA(300, 8)), with a model trained without cuts at the nodes and not retrained.

## Cut selection at the root

At each round of the root's cut loop, the separators find more cuts than SCIP adds to the LP. SCIP's selector (`hybrid`) scores them mostly by efficacy (how far the cut is from the LP solution), with small weights on parallelism to the objective and integer support, and skips the cuts too parallel to those already taken. The lookahead expert of Paulus et al. (2022), `--cut-selector lookahead` (see [command-line-tools.md](command-line-tools.md#mis-benchmark-evaluating-branching-rules)), scores each candidate by the increase of the LP bound when it alone is added, and takes the cuts in that order with the same parallelism filter. It is to cut selection what full strong branching is to branching: too slow to use, but it tells what a better selection is worth. Runs stopped after the root (`limits/nodes = 1`) give the bound at the end of SCIP's root; same instances, settings and seed as above, ratios to the study settings with SCIP's selector:

| Instances | gap closed at the root, SCIP · lookahead | cuts in the LP | candidates raising the bound alone | time | nodes |
|---|---|---|---|---|---|
| G(125, 0.1) | 0.728 · 0.730 | 194 · 197 | 40% of 1,151 | 1.61 [1.49, 1.74] | 1.14 [0.99, 1.31] |
| G(150, 0.083) | 0.709 · 0.704 | 218 · 220 | 44% of 1,177 | 1.29 [1.11, 1.49] | 0.97 [0.81, 1.16] |
| G(200, 0.0625) | 0.663 · 0.662 | 252 · 251 | 42% of 1,171 | 0.96 [0.86, 1.07] | 0.94 [0.84, 1.07] |
| BA(200, 8) | 0.790 · 0.786 | 232 · 236 | 53% of 1,295 | 1.93 [1.74, 2.14] | 1.11 [0.92, 1.36] |
| BA(250, 8) | 0.747 · 0.749 | 278 · 281 | 62% of 1,327 | 1.57 [1.47, 1.67] | 1.15 [0.97, 1.36] |
| BA(300, 8) | 0.726 · 0.725 | 323 · 325 | 52% of 1,418 | 1.46 [1.22, 1.69] | 1.24 [1.04, 1.42] |

The gap closed is (first LP bound − bound at the end of the root) / (first LP bound − optimum); the first LP bound is \(n/2\), reached by \(x = ½\) everywhere.

The expert closes the same share of the gap as SCIP's selector on every set, within 0.005, with as many cuts in the LP. Candidates are not lacking: 40 to 62% of them raise the bound on their own. The trees are no smaller (node ratios between 0.94 and 1.24, every interval but one containing 1), and the expert's LP solves make the runs 1.3 to 1.9 times slower, except on G(200, 0.0625), where they are a small part of five minutes of solving. With cuts at the nodes (`no_node_gomory`, the expert at the root only), the result is the same: time ratio 1.11 to 1.99, node ratio 0.98 to 1.19.

The reason is the cut loop. At the root, SCIP separates until the bound stalls, with no limit on the number of rounds: which cuts enter at each round changes the path of the bound, not where it ends, which is set by what the separators can find. Paulus et al. measured their expert after 30 rounds of one cut each, where the choice of every cut counts. Here the loop runs to its end, and the root takes 0.4 to 18% of the solving time with the study settings (the larger the graph, the smaller its share), so that even a selection reaching the same bound in fewer rounds would save little. For this problem and this formulation, there is nothing to learn in the selection of the cuts at the root: the decision that matters is where to separate in the tree ([cutting planes at the nodes](#cutting-planes-at-the-nodes) and next section).

## Where to separate in the tree

At the training sizes, odd-cycle cuts at the nodes take the time of the study settings with three times fewer nodes ([cutting planes at the nodes](#cutting-planes-at-the-nodes)): over a whole instance, what the cuts cost and what they save balance out. Two experiments look at where in the tree they pay off, with the protocol of the cut ablation (SCIP's default settings and a parameter file of `experiments/cuts/settings/`, `relpscost`, SCIP seed 0, the 20 test instances of G(125, 0.1) and BA(200, 8), every run solved). Their runs wrote a log of every node (`--node-log-dir`, [data-format.md](data-format.md#node-logs)); `experiments/cuts/compare_configs.py` computes the ratios of their tables.

### Which mechanism makes the trees smaller

`EXPERIMENT=node_mechanisms` adds four configurations to those of the ablation; like them, they call Gomory and the other separators SCIP uses below the root at the root only:

- `pool_nodes`: no separator at the nodes, only the cut pool (the cuts found so far, separated again; SCIP: every 10 levels); `pool_every_node`: the cut pool at every node;
- `zerohalf_local_every_node`: zero-half cuts at every node, local cuts allowed at every node (SCIP: only at the nodes of best bound);
- `oddcycle_every_node`: odd-cycle cuts at every node, zero-half cuts as by default.

Ratios to the study settings, time (unshifted) · nodes (shifted by 100), with 95% paired bootstrap intervals. With the study settings, `relpscost` takes 3.78 s and 3,596 nodes on G(125, 0.1), 7.14 s and 863 nodes on BA(200, 8):

| Configuration | G(125, 0.1), time | nodes | BA(200, 8), time | nodes |
|---|---|---|---|---|
| `no_node_gomory` | 0.94 [0.91, 0.99] | 0.42 | 0.98 [0.94, 1.01] | 0.62 |
| `pool_nodes` | **0.89** [0.86, 0.92] | 0.63 | 1.00 [0.98, 1.02] | 0.90 |
| `pool_every_node` | 0.95 [0.89, 1.00] | 0.63 | 1.00 [0.93, 1.10] | 0.83 |
| `zerohalf_every_node` | 1.01 [0.97, 1.04] | 0.34 | 1.02 [0.98, 1.06] | 0.49 |
| `zerohalf_local_every_node` | 1.21 [1.14, 1.29] | 0.33 | 1.15 [1.03, 1.26] | 0.49 |
| `oddcycle` | 1.00 [0.95, 1.05] | 0.33 | 1.05 [0.99, 1.12] | 0.43 |
| `oddcycle_every_node` | 1.56 [1.45, 1.69] | **0.15** | 1.36 [1.18, 1.53] | **0.22** |

The cut pool alone, at depths 10, 20, 30…, is the fastest configuration on Erdős–Rényi graphs, with trees 1.6 times smaller than the study settings'; separating it at every node gives no smaller trees. Odd-cycle cuts at every node give trees 4 to 7 times smaller, at 1.4 to 1.6 times the time. Local zero-half cuts cost time and save no node.

The node logs of the configurations that separate at every node tell what each cut loop below the root did: it pruned the node (the first LP did not reach the cutoff bound, the cut loop did), raised its bound without pruning it, or left it unchanged (`experiments/cuts/analyze_node_logs.py`, outputs in `experiments/cuts/results/node_mechanisms/analysis/`). With odd-cycle cuts at every node, the cut loop takes 49% of the time spent on the nodes on G(125, 0.1) and 27% on BA(200, 8):

| Loops of `oddcycle_every_node` | G(125, 0.1): loops · loop time | BA(200, 8): loops · loop time |
|---|---|---|
| pruned the node | 16% · 8% | 14% · 8% |
| raised the bound (median: 15% and 11% of the gap to the cutoff) | 73% · 89% | 71% · 89% |
| bound unchanged | 10% · 2% | 16% · 3% |

With zero-half cuts at every node, 85 to 87% of the loops leave the bound unchanged, and take 43 to 44% of the loop time.

**The loops that prune are cheap and can be told apart before separating.** The gap between the cutoff bound and the LP bound after the node's first LP tells them from the others with an AUC of 0.90 on both families; the depth does not (0.52 to 0.54). Separating only where this gap is at most 0.5 (the same threshold on both families) would keep 80% of the pruning loops with 27% of the loops and 13% of the loop time; a logistic regression on six features (gap, depth, distance to the global lower bound, fractional variables, share of them at ½, iterations of the first LP), trained on 10 instances and measured on the 10 others, needs 8.5% (Erdős–Rényi) and 12% (Barabási–Albert) of the loop time for 80%, little more than the gap alone. SCIP's own filter does the opposite: `separating/maxbounddist` separates only at the nodes whose bound is close to the global lower bound, far from the cutoff bound, where the loops seldom prune.

These figures are measured on the tree that separates everywhere: a node whose loop is skipped is branched on, and its cuts do not enter the pool. They tell what a gate could keep, not what it would gain.

### A separation gate

`src/cuts/MISSeparationGate.c` makes that rule a SCIP plugin: a separator that finds no cuts, called first at every node, which lets the odd-cycle separator run at the node only when the gap after the first LP is at most `separating/misgate/maxgap` ([command-line-tools.md](command-line-tools.md#mis-benchmark-evaluating-branching-rules)). `EXPERIMENT=node_gate` compares it, with thresholds 0.25, 0.5 and 1 (`oddcycle_gate025`, `oddcycle_gate050`, `oddcycle_gate100`), to the configurations of the previous experiment, made again in the same session. Ratios to the study settings as above; the last column is the share of the nodes with a cut loop where the gate let the odd-cycle separator run:

| Configuration | G(125, 0.1), time | nodes | BA(200, 8), time | nodes | odd-cycle cuts at |
|---|---|---|---|---|---|
| `pool_nodes` | **0.91** [0.87, 0.95] | 0.63 | 0.97 [0.94, 1.01] | 0.90 | no node |
| `no_node_gomory` | 0.94 [0.89, 0.99] | 0.42 | 0.97 [0.93, 1.00] | 0.62 | no node |
| `oddcycle` | 1.02 [0.97, 1.07] | 0.33 | 1.03 [0.96, 1.12] | 0.43 | depths 10, 40, 160 |
| `oddcycle_gate025` | 0.99 [0.93, 1.05] | 0.32 | 1.05 [0.97, 1.13] | 0.45 | 18% · 16% |
| `oddcycle_gate050` | 1.01 [0.96, 1.07] | 0.27 | 1.06 [0.99, 1.14] | 0.40 | 25% · 23% |
| `oddcycle_gate100` | 1.09 [1.01, 1.17] | 0.22 | 1.09 [1.00, 1.20] | 0.33 | 42% · 38% |
| `oddcycle_every_node` | 1.55 [1.41, 1.69] | **0.15** | 1.29 [1.15, 1.43] | **0.22** | every node |

The configurations made again explored the same trees as in the previous experiment, so the gate, when off, changes nothing. The times of this session were about 1.65 times those of the previous one on every instance and configuration (the state of the host machine): the ratios, made within one session with the configurations interleaved on each instance, are comparable; the times in seconds are not.

**The gate does what it is meant to, but the loops it skips were worth half of the tree reduction.** With a threshold of 0.5, the odd-cycle separator runs at a quarter of the nodes, and the gate takes 35% (Erdős–Rényi) and 18% (Barabási–Albert) less time than separating at every node. Its trees are, however, 1.8 times larger (900 nodes against 453 on G(125, 0.1)): the loops that raise the bound without pruning, which the node logs could not value, help the descendants of the node and fill the cut pool. The three thresholds move along the same trade-off between nodes and time as the fixed configurations, and none is faster than the study settings or `oddcycle`; the cut pool alone stays the fastest on Erdős–Rényi graphs, and on Barabási–Albert graphs every configuration but the two that separate the most is within the noise.

At the training sizes, a decision per node that aims at the nodes the cut loop prunes does not pay, and a model trained to recognize them would learn the wrong target. A decision that pays would have to value the bound gained for the descendants of the node, which requires counterfactual runs. Where cuts at the nodes do pay, on the larger sets (G(200, 0.0625), BA(300, 8)), the gate has not been measured.

## Clique formulation

`src/problems/MIS.cpp` writes one constraint x_u + x_v ≤ 1 per edge, whose LP relaxation is weak: x = 1/2 everywhere is feasible, and the first LP bound is n/2 on every instance. The MIS instances of the literature (Gasse et al. 2019, Ecole, and the learning-to-cut papers built on them) use a clique formulation instead: a greedy partition of the edges into cliques, each written x(K) ≤ 1. Their figures are therefore not comparable with ours. SCIP can make that change itself: it upgrades the edge constraints to set packing constraints, and `constraints/setppc/cliquelifting` (off by default) merges them into cliques in presolve.

`EXPERIMENT=clique_lifting` adds that parameter to five configurations of the cut ablation (`cliquelifting.set`, `no_node_gomory_cliquelifting.set`, `oddcycle_cliquelifting.set` in `experiments/cuts/settings/`) and runs them with the unlifted ones, the ten configurations interleaved on each instance, with the protocol above, on G(125, 0.1), BA(200, 8) (20 test instances each), G(150, 0.083), BA(250, 8), G(200, 0.0625) and BA(300, 8) (10 each). The first four sets were made in one session; the last two in two sessions (the first four instances of G(200, 0.0625), then the rest overnight), the ten configurations of an instance always in the same one. Every run is solved, and both formulations find the same optimum on all 80 instances. The unlifted configurations explored the same trees as in the earlier experiments.

The lifting lowers the first LP bound by about a quarter (62.5 → 42.7 on G(125, 0.1), 100 → 74.7 on BA(200, 8)), but the root cut loop, which stops at stagnation, nearly makes up the difference: after the root cuts, the bounds differ by 0.3 to 1.4 only (41.6 against 41.3 on G(125, 0.1), 70.1 against 69.7 on G(200, 0.0625), with the study settings). The gain is below the root, where the clique constraints stay in every node LP while root cuts age out of the LP, and where those LPs have fewer rows.

Ratio of each lifted configuration to the same configuration unlifted, time (unshifted) · nodes (shifted by 100), with 95% bootstrap intervals:

| Configuration | G(125, 0.1) | BA(200, 8) | G(150, 0.083) | BA(250, 8) | G(200, 0.0625) | BA(300, 8) |
|---|---|---|---|---|---|---|
| study | 0.77 [0.73, 0.81] · 0.38 | 0.84 [0.78, 0.89] · 0.81 | 0.62 [0.51, 0.75] · 0.43 | 0.84 [0.78, 0.91] · 0.80 | 0.56 [0.50, 0.62] · 0.45 | 0.75 [0.66, 0.85] · 0.70 |
| `no_node_gomory` | 0.94 [0.90, 0.98] · 0.64 | 0.89 [0.82, 0.96] · 0.93 | 0.83 [0.74, 0.91] · 0.67 | 0.84 [0.75, 0.93] · 1.01 | 0.89 [0.77, 1.03] · 0.93 | 0.78 [0.69, 0.89] · 0.83 |
| `oddcycle` | 0.89 [0.84, 0.93] · 0.64 | 0.83 [0.77, 0.88] · 0.73 | 0.81 [0.73, 0.90] · 0.72 | 0.81 [0.72, 0.92] · 0.87 | 0.78 [0.75, 0.82] · 0.86 | 0.78 [0.69, 0.88] · 0.75 |
| learned, study | 0.69 [0.65, 0.73] · 0.45 | 0.83 [0.77, 0.90] · 0.69 | 0.66 [0.58, 0.75] · 0.46 | 0.81 [0.72, 0.91] · 0.77 | 0.51 [0.45, 0.58] · 0.39 | 0.78 [0.67, 0.90] · 0.73 |
| learned, `no_node_gomory` | 0.89 [0.83, 0.96] · 0.67 | 0.88 [0.83, 0.94] · 0.79 | 0.92 [0.79, 1.08] · 0.77 | 0.96 [0.87, 1.05] · 1.06 | 0.80 [0.74, 0.89] · 0.76 | 0.78 [0.66, 0.90] · 0.94 |

Ratio to the study settings with `relpscost` (unlifted), same notation:

| Configuration | G(125, 0.1) | BA(200, 8) | G(150, 0.083) | BA(250, 8) | G(200, 0.0625) | BA(300, 8) |
|---|---|---|---|---|---|---|
| study, lifted | 0.77 · 0.38 | 0.84 · 0.81 | 0.62 · 0.43 | 0.84 · 0.80 | 0.56 · 0.45 | 0.75 · 0.70 |
| `oddcycle`, lifted | 0.90 · 0.21 | 0.88 · **0.31** | 0.77 · **0.18** | 0.71 · **0.16** | 0.44 · **0.12** | 0.53 · **0.09** |
| learned, `no_node_gomory` | 0.63 · 0.38 | 0.46 · 0.68 | 0.58 · 0.27 | 0.49 · 0.32 | 0.49 · 0.20 | 0.49 · 0.22 |
| learned, study, lifted | **0.48** · 0.33 | **0.40** · 0.72 | **0.50** · 0.32 | 0.51 · 0.58 | 0.49 · 0.32 | 0.65 · 0.65 |
| learned, `no_node_gomory`, lifted | 0.56 · **0.26** | **0.40** · 0.53 | 0.54 · 0.21 | **0.46** · 0.34 | **0.39** · 0.15 | **0.38** · 0.21 |

With the study settings, `relpscost` takes 316 s and 388,752 nodes on G(200, 0.0625), 64.2 s and 22,174 nodes on BA(300, 8), as in the earlier sessions.

**The lifting speeds up every configuration, the learned branching rule included, at every size.** It saves the most where no separator works at the nodes (study settings: 0.56 to 0.84 of the time, the most on G(200, 0.0625)), and less where node cuts already strengthen the node LPs (`no_node_gomory`, learned with `no_node_gomory`: 0.78 to 0.96, not significant in three cases out of twelve): the clique constraints and the node cuts partly do the same work.

**The learned rule with `no_node_gomory`, lifted, is the fastest configuration measured so far.** It takes 0.39 and 0.38 of the time of the study settings on the two largest sets, against 0.49 unlifted, and 0.40 to 0.56 at the smaller sizes. At the training sizes, the learned rule with the study settings, lifted, is as fast or faster (0.48 against 0.56 on G(125, 0.1)), but it falls behind as the instances grow (0.65 against 0.38 on BA(300, 8)), where cuts at the nodes pay the most. The model was trained on the edge formulation and its features were computed on it; it keeps its lead over `relpscost` after the lifting: with the study settings, lifted, it takes 0.63, 0.47, 0.80, 0.60, 0.88 and 0.87 of the time of `relpscost`, lifted, on the six sets (0.69, 0.48, 0.75, 0.63, 0.97 and 0.83 unlifted).

These are single runs (SCIP seed 0). Before comparing with the figures of the literature, one would also need its density and sizes: BA with affinity 4, sparser than ours, on 500 to 1500 vertices. A test on three BA(500, 4) instances (default settings, `relpscost`) solved them in 2 to 14 s and 1 to 265 nodes with either formulation, as easy as reported there: at equal size, density makes the instances hard more than the number of vertices.
