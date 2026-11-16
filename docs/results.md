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
