# Learning a branching rule

The first learned rule ranks the branching candidates of a node with a small neural network and branches on the best one. It imitates the full strong branching expert recorded by the data collection (see [command-line-tools.md](command-line-tools.md#mis-collector-strong-branching-samples)). Results are in [results.md](results.md).

## 1. Features

`mis-featurizer` turns the samples collected on one instance into a training table, `NAME.sb.features.csv`, with one row per candidate: the features and the expert's answer (`score`, `expert_rank`, `expert_selected`).

```bash
scripts/featurize_strong_branching.sh build/mis-featurizer INSTANCE_DIR SAMPLES_DIR OUTPUT_DIR 4
```

Each candidate is described by 44 features computed from the graph, the state of every vertex, the LP values of the candidates and the history of the search:

- **version 1**: LP value and fractionality, degree in the graph and among the free or fractional vertices, sum and maximum of the LP values of its neighbours, tight edges, triangles of fractional neighbours, values relative to the other candidates, and the depth and size of the node;
- **version 2**: the LP values two steps away, the room left for the neighbours to grow if the candidate is set to 0, the LP weight of cliques through the candidate (a clique of weight 1 is a tight clique inequality) and the rank of the candidate among those of the node;
- **version 3**: what the search has learned about the candidate (pseudo-costs and their number of observations, inferences, cutoffs, conflict score) and the distance of the node to the incumbent.

They are listed in `include/learning/MISBranchingFeatures.h`.

The features are implemented once, in C, and that single function is used both by the featurizer and by the learned rule inside SCIP. A test checks that the features recomputed from the collected files are identical, bit for bit, to the ones computed during the solve: training and inference cannot disagree on what a feature means. After a change of the feature set, the tables are recomputed in seconds; the samples do not need to be collected again.

## 2. Training

The Python code lives in `ml/` and uses [uv](https://docs.astral.sh/uv/) for its environment:

```bash
uv sync

uv run python ml/train_branching_model.py \
    --train data/processed/features/er125_p01/train \
    --valid data/processed/features/er125_p01/valid \
    --hidden 32 32 \
    --output experiments/er125_p01/models/mlp32.model
```

`--hidden` takes the sizes of the hidden layers; with no value, the model is linear. Training keeps the [usable samples](data-format.md#strong-branching-samples), in which the expert's choice is meaningful, and minimizes a cross-entropy: among the candidates of a node, the one chosen by the expert should get the highest score.

The script reports, on the validation set, how the decisions of the model compare with the expert's: `accuracy` (same candidate), `top5` (the expert's candidate is among the model's five best) and `score ratio` (strong branching score of the model's candidate divided by the best score of the node; 1 means as good as the expert). It writes the model as a text file, checks that this file reproduces the decisions of the trained network, and stores the training history next to it as JSON.

`--train` and `--valid` accept several directories, aggregated into one dataset; the metrics are also reported per validation directory. The candidates of all samples are stored one after the other, without padding, so that a family with many candidates per node does not inflate the others: training on Erdős–Rényi and Barabási–Albert samples together (200,000 samples, see [results.md](results.md#training-on-barabásialbert-graphs)) peaks at 5.3 GB.

### Graph models

By default every candidate is scored on its own (a multilayer perceptron). With `--graph-layers K`, the network is a graph neural network over the candidates of the node, linked by the edges of the graph between them: the `--hidden` layers transform each candidate, then each of the K graph layers computes

\[
h'_v = \mathrm{ReLU}\big(W h_v + W_N\, \overline{h}_{N(v)} + W_G\, \overline{h} + b\big),
\]

where \(\overline{h}_{N(v)}\) is the mean state of the candidate neighbours of \(v\) (0 if it has none) and \(\overline{h}\) the mean state of all the candidates of the node; a linear layer gives the score. Means rather than sums keep the scale independent of the degree, which varies widely on Barabási–Albert graphs.

```bash
uv run python ml/train_branching_model.py \
    --train data/processed/features/ba200_k8/train \
            data/processed/features/ba200_k8_mix1/train \
    --valid data/processed/features/ba200_k8/valid \
            data/processed/features/ba200_k8_mix1/valid \
    --hidden 32 --graph-layers 2 \
    --output experiments/ba200_k8/models/gnn32x2_ba1.model
```

The samples need no new collection: the feature tables keep the vertex of every candidate, and the edges between candidates are read from the instance files, found by name under `--instances` (`data/instances` by default). A graph model is written in version 2 of the model format (`graph_layer` entries, see `ml/branching_model_file.py`); a model without graph layer is still written in version 1, byte for byte as before.

Training on the CPU, the default, is reproducible bit for bit. `--device cuda` trains on the GPU, about four times faster per epoch for a graph model; it needs a CUDA build of PyTorch, which the project's environment does not install, and its results are not reproducible bit for bit (the GPU sums in parallel, in no fixed order). SCIP, the collection and the evaluation inside SCIP always run on the CPU.

## 3. Evaluation inside SCIP

The `mislearned` rule loads a model file and evaluates it in C at every node. A model names the features it reads, which may be any subset of the computed ones: models trained before a feature was added keep working, and a model that names an unknown feature is refused. `--features` of the training script selects the features of a model.

```bash
./build/mis-benchmark \
    --instances data/instances/er125_p01/test \
    --rules relpscost,mislearned \
    --model experiments/er125_p01/models/mlp32.model \
    --seeds 0,1,2 \
    --output experiments/er125_p01/results/test_learned_mlp32.csv
```

The rule scores all the candidates of a node at once (`MISevaluateBranchingModelOnNode`), which graph models need; a multilayer perceptron gives the same scores as before, bit for bit. On real samples, the scores computed in C agree with those of the Python model file to about \(10^{-14}\), and a test solves an instance with a graph model while an independent implementation recomputes every score.

A model can also be tried on one instance, or compared with other rules and models, from the [interactive menu](interactive-menu.md).

## 4. On-policy samples

The first dataset only contains nodes visited by the expert's own search. Inside the solver, a learned rule makes different choices and reaches nodes that the expert never visits, on which it was not trained. With `--policy-model`, the collector lets a trained model drive the search while full strong branching still labels every visited node (the DAgger scheme):

```bash
./build/mis-collector \
    --instance INSTANCE.mis \
    --output-dir OUTPUT_DIR \
    --policy-model experiments/er125_p01/models/mlp32.model
```

The policy may be a multilayer perceptron or a graph model. The tables have the same format, and the metadata records the model and its agreement with the expert (see [data-format.md](data-format.md#strong-branching-samples)). Nodes that strong branching proves prunable are still pruned, so the tree is smaller than the one the model explores on its own.

`scripts/collect_on_policy.sh DATASET MODEL NAME [BUILD_DIR] [JOBS]` collects and featurizes such samples for the training and validation instances of a dataset, into `data/raw/strong_branching/DATASET_NAME` and `data/processed/features/DATASET_NAME`; `experiments/er125_p01/collect_on_policy.sh` is the same for `er125_p01`. A new model is then trained on the expert's samples and the on-policy ones together:

```bash
experiments/er125_p01/collect_on_policy.sh experiments/er125_p01/models/mlp32.model dagger1 build 4

uv run python ml/train_branching_model.py \
    --train data/processed/features/er125_p01/train \
            data/processed/features/er125_p01_dagger1/train \
    --valid data/processed/features/er125_p01/valid \
            data/processed/features/er125_p01_dagger1/valid \
    --hidden 32 32 \
    --output experiments/er125_p01/models/mlp32_dagger1.model
```
