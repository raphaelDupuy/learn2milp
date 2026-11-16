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
