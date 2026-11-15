# Learn2MILP

## Overview

Learn2MILP is a research-oriented project on learning-augmented exact methods for combinatorial optimization.

The main objective is to investigate whether machine learning can improve selected decision mechanisms of Mixed-Integer Linear Programming (MILP) solvers while preserving exactness.

The project initially focuses on two components of Branch-and-Cut:

- branching decisions;
- cutting-plane selection.

The first problem considered is the Maximum Independent Set (MIS). The framework is intended to be extended later to the Job-Shop Scheduling Problem (JSSP).

## Research questions

The project investigates the following questions:

1. Can branching decisions in exact MILP solving be learned from solver data?
2. Can cutting-plane selection be improved using machine learning?
3. Are learned branching and cutting policies complementary?
4. How well do learned policies generalize across instance sizes and distributions?
5. Does the reduction in solver effort compensate for the computational cost of ML inference?

A later direction is to compare policies trained to imitate local solver decisions with approaches whose objective is more directly related to the global performance of the search process.

## Current status

The current version provides a basic experimental framework for the Maximum Independent Set problem.

### Implemented

- C++17 project using CMake;
- integration with SCIP;
- MILP formulation of Maximum Independent Set;
- Erdős–Rényi random graph generation;
- reproducible instance generation using explicit random seeds, one instance at a time or in batches;
- exact optimization and basic solver statistics;
- full strong branching rule for SCIP that records every branching decision, checked node by node against SCIP's own `vanillafullstrong` rule;
- learning-to-branch dataset generation (`mis-collector`);
- benchmark of branching rules on a set of instances (`mis-benchmark`);
- a first learned branching rule: candidate features, training of a small ranking model in Python, and its evaluation inside SCIP (`mislearned`).

### Planned

- richer learned branching policies (graph neural networks, other training signals);
- learning-to-cut;
- comparison of classical and learned Branch-and-Cut configurations;
- experiments on generalization across instance distributions;
- extension to Job-Shop Scheduling.

## Maximum Independent Set

For a graph \(G=(V,E)\), the Maximum Independent Set problem is formulated as a binary linear program.

Each vertex \(v\) is associated with a binary variable \(x_v\).

The objective is:

\[
\max \sum_{v \in V} x_v
\]

subject to:

\[
x_u + x_v \leq 1
\qquad \forall (u,v) \in E
\]

and:

\[
x_v \in \{0,1\}
\qquad \forall v \in V.
\]

The current implementation generates Erdős–Rényi graphs \(G(n,p)\), builds the corresponding MILP model, and solves it exactly using SCIP.

## Strong branching dataset

`mis-collector` solves one MIS instance with full strong branching and records one sample per branching decision: the candidates, the outcome of strong branching on each of them, and the variable chosen by this expert.

```bash
./build/mis-generator single --n 125 --p 0.1 --seed 1 --output data/instances/g125.mis

./build/mis-collector \
    --instance data/instances/g125.mis \
    --output-dir data/raw/strong_branching \
    --time-limit 600
```

Optional limits are `--time-limit SECONDS` (default 3600), `--node-limit N` and `--scip-seed S`. Existing output files are never overwritten.

### Solver settings

The same SCIP settings are used to collect the data and, later, to evaluate the learned policies:

- presolving enabled;
- restarts disabled (`presolving/maxrestarts = 0`), so that the variable space does not change during the search;
- cutting planes at the root node only (`separating/maxrounds = 0`), so that the effect of branching is isolated.

### Output files

For an instance `NAME.mis`, three files are written to the output directory:

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
- `expert_selected`: 1 for the candidate the expert branched on.
- `pseudocost_down`, `pseudocost_up`, `pseudocost_count_*`, `inferences_*`, `cutoffs_*`, `conflict_score`: what the search has learned about the vertex before the node is branched, as kept by SCIP. Strong branching does not change these statistics: a rule without strong branching sees the same kind of values.

Conventions:

- objective values (`lp_objective`, `primal_bound`, `dual_bound`, strong branching bounds and gains) are in SCIP's internal space, which is a minimization: the MIS objective appears negated. The metadata gives the map back to the original objective, `original = original_objective_intercept + original_objective_slope * internal`;
- every `*_valid` column tells whether the value next to it is meaningful;
- a node reached while the solver is stopping (time limit) is not recorded, so every sample is a complete expert decision;
- collection fails, without writing anything, if presolving merges two vertices into one LP column: a sample is never attributed to the wrong vertex.

The exact column lists and the meaning of every field are documented in `include/io/MISStrongBranchingSampleIO.hpp` and `include/branching/MISStrongBranchingCollector.h`.

Loading a dataset in Python:

```python
import pandas as pd

nodes = pd.read_csv("NAME.sb.nodes.csv")
candidates = pd.read_csv("NAME.sb.candidates.csv")
samples = candidates.merge(nodes, on=["sample_index", "node_id"])
```

### Collecting a whole directory

`scripts/collect_strong_branching.sh` runs the collector on every instance of a directory, several instances at a time. It can be interrupted and restarted: instances already collected are skipped.

```bash
scripts/collect_strong_branching.sh build/mis-collector INSTANCE_DIR OUTPUT_DIR 4
```

Two more optional arguments give the time limit per instance and a policy model (see "On-policy samples" below).

`experiments/er125_p01/build_dataset.sh` is the recipe of the first dataset: Erdős–Rényi graphs G(125, 0.1) split into training (60 graphs), validation (20) and test (20) sets, plus 20 graphs G(150, 0.05) to measure generalization. Samples are collected for the training and validation sets only; the other two sets are used to evaluate the solver.

```bash
experiments/er125_p01/build_dataset.sh build 4
```

## Evaluating branching rules

`mis-benchmark` solves every instance of a directory with each branching rule and each SCIP seed, and writes one CSV row per run: status, explored nodes, LP iterations and solving time.

```bash
./build/mis-benchmark \
    --instances data/instances/er125_p01/test \
    --rules relpscost,pscost,misfullstrong \
    --seeds 0,1,2 \
    --output experiments/er125_p01/baselines.csv
```

- `--rules` takes the name of any SCIP branching rule (`relpscost` is SCIP's default), or `misfullstrong`, the full strong branching expert recorded by the data collection;
- `--settings study` (the default) uses the solver settings of the data collection, so that rules are compared in the conditions they were learned in; `--settings default` runs SCIP with no parameter changed, as an outside reference;
- runs are executed one at a time, with the rules interleaved on each instance: do not run anything else on the machine if solving times matter.

The tool ends with a summary per rule: number of solved runs and shifted geometric means of the node counts (shift 100) and of the solving times (shift 1 second). Node counts are reproducible for a given seed; solving times are not, and should be compared over several runs.

## Learning a branching rule

The first learned rule ranks the branching candidates of a node with a small neural network and branches on the best one. It imitates the full strong branching expert recorded by the data collection.

### 1. Features

`mis-featurizer` turns the samples collected on one instance into a training table, `NAME.sb.features.csv`, with one row per candidate: the features and the expert's answer (`score`, `expert_rank`, `expert_selected`).

```bash
scripts/featurize_strong_branching.sh build/mis-featurizer INSTANCE_DIR SAMPLES_DIR OUTPUT_DIR 4
```

Each candidate is described by 44 features computed from the graph, the state of every vertex, the LP values of the candidates and the history of the search: LP value and fractionality, degree in the graph and among the free or fractional vertices, sum and maximum of the LP values of its neighbours, tight edges, triangles of fractional neighbours, values relative to the other candidates, and the depth and size of the node; then, since feature set version 2, the LP values two steps away, the room left for the neighbours to grow if the candidate is set to 0, the LP weight of cliques through the candidate (a clique of weight 1 is a tight clique inequality) and the rank of the candidate among those of the node; and, since version 3, what the search has learned about the candidate (pseudo-costs and their number of observations, inferences, cutoffs, conflict score) and the distance of the node to the incumbent. They are listed in `include/learning/MISBranchingFeatures.h`.

The features are implemented once, in C, and that single function is used both by the featurizer and by the learned rule inside SCIP. A test checks that the features recomputed from the collected files are identical, bit for bit, to the ones computed during the solve: training and inference cannot disagree on what a feature means. After a change of the feature set, the tables are recomputed in seconds; the samples do not need to be collected again.

### 2. Training

The Python code lives in `ml/` and uses [uv](https://docs.astral.sh/uv/) for its environment:

```bash
uv sync

uv run python ml/train_branching_model.py \
    --train data/processed/features/er125_p01/train \
    --valid data/processed/features/er125_p01/valid \
    --hidden 32 32 \
    --output experiments/er125_p01/models/mlp32.model
```

`--hidden` takes the sizes of the hidden layers; with no value, the model is linear. Training keeps the samples in which the expert's choice is meaningful (see `node_cutoff` above) and minimizes a cross-entropy: among the candidates of a node, the one chosen by the expert should get the highest score.

The script reports, on the validation set, how the decisions of the model compare with the expert's: `accuracy` (same candidate), `top5` (the expert's candidate is among the model's five best) and `score ratio` (strong branching score of the model's candidate divided by the best score of the node; 1 means as good as the expert). It writes the model as a text file, checks that this file reproduces the decisions of the trained network, and stores the training history next to it as JSON.

### 3. Evaluation inside SCIP

The `mislearned` rule loads a model file and evaluates it in C at every node. A model names the features it reads, which may be any subset of the computed ones: models trained before a feature was added keep working, and a model that names an unknown feature is refused. `--features` of the training script selects the features of a model.

```bash
./build/mis-benchmark \
    --instances data/instances/er125_p01/test \
    --rules relpscost,mislearned \
    --model experiments/er125_p01/models/mlp32.model \
    --seeds 0,1,2 \
    --output experiments/er125_p01/results/test_learned_mlp32.csv
```

### 4. On-policy samples

The first dataset only contains nodes visited by the expert's own search. Inside the solver, a learned rule makes different choices and reaches nodes that the expert never visits, on which it was not trained. With `--policy-model`, the collector lets a trained model drive the search while full strong branching still labels every visited node (the DAgger scheme):

```bash
./build/mis-collector \
    --instance INSTANCE.mis \
    --output-dir OUTPUT_DIR \
    --policy-model experiments/er125_p01/models/mlp32.model
```

The tables have the same format. The metadata file records the model (`branching_policy`, `policy_model`), the number of branching decisions it took and how many of them agree with the expert (`policy_decisions`, `policy_expert_agreements`). Nodes that strong branching proves prunable are still pruned, so the tree is smaller than the one the model explores on its own.

`experiments/er125_p01/collect_on_policy.sh` collects and featurizes such samples for the training and validation instances. The training script accepts several directories, so that a new model is trained on the expert's samples and the on-policy ones together:

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

First iteration on er125_p01 (20 instances, 3 seeds, shifted geometric means of the node counts; models and result tables are in `experiments/er125_p01/`):

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

On-policy samples bring about 4% fewer nodes, on about 42 of the 60 runs of each set. The gain is small (within the instance-to-instance noise), and the offline validation metrics do not improve: with these features and this model size, the nodes visited during training are not what limits the learned rule. The effect of the training seed has not been measured.

The 12 features of version 2 do not settle the matter either: `mlp32_v2` explores 4.5% fewer nodes than `mlp32_dagger1` on the test set but 4% more on the transfer set, both within the noise. A network twice as wide reaches the same validation accuracy (0.137), so the inputs, not the size of the model, are the limit. Two side results are worth keeping in mind. Training the model to maximize the strong branching score of its choice, instead of imitating the expert's choice, raised the offline score ratio (0.64 to 0.69) but gave larger trees (3,096 nodes on the test set): the score ratio must not be used to select a model. And the strong branching score of a vertex at the parent node, used alone to choose, reaches a score ratio of 0.65 on the expert's nodes and 0.83 on the on-policy ones, above every model here: what the search has already learned about a variable says more than the local structure of the graph.

The search history of version 3 (pseudo-costs, inferences, cutoffs, distance to the incumbent) raises the validation accuracy from 0.14 to 0.19 but not the performance in the solver: `mlp32_h1` and `mlp32_h2` are within the noise of `mlp32_v2`. With these features the on-policy samples do matter, since the pseudo-costs seen at a node depend on the rule that drove the search: the model trained on the expert's samples alone is the worst of all, and one on-policy round removes 12% of its nodes. A second round brings nothing more. The `mlp32_v2` samples were collected in the previous sample format and are not reproducible with the current collector; the three `mlp32_h*` models are.

Across all these models, neither the score ratio nor the accuracy measured on the validation samples predicted the size of the search trees: only the benchmark does.

A pilot on another graph family tells the two kinds of features apart. On 20 Barabási–Albert graphs with 200 vertices, each new vertex joined to 8 existing ones (never seen during training; 3 seeds, all solved), relpscost explores 916 nodes in 5.6 s, `mlp32_v2` 1,064 nodes in 3.0 s and `mlp32_h2` 861 nodes in 2.7 s. Both learned rules stay faster than relpscost on every instance, and the model with the search history explores 17% fewer nodes than the one without, on all 20 instances: the history generalizes across families where the static features do not. The instances come from a temporary Python script, kept with the results in `experiments/ba200_m8_pilot/` until the C++ generator supports this family.

## Project structure

```text
learn2milp/
├── CMakeLists.txt
├── README.md
├── docs/
├── data/
├── experiments/
├── include/
│   ├── branching/
│   ├── features/
│   ├── generators/
│   ├── instances/
│   ├── io/
│   ├── learning/
│   ├── problems/
│   ├── services/
│   └── ui/
├── ml/
├── pyproject.toml
├── src/
│   ├── branching/
│   ├── generators/
│   ├── io/
│   ├── learning/
│   ├── problems/
│   ├── services/
│   ├── tools/
│   ├── ui/
│   └── main.cpp
├── scripts/
└── tests/
```

The code is deliberately separated into problem definitions, instance generators, and solver-related components. This structure makes it possible to add new instance distributions and optimization problems without coupling them to the data generation process.

## Dependencies

- C++17 compiler;
- CMake (version 3.20 or newer);
- SCIP Optimization Suite 10.1.0;
- Git.

The machine learning components are written in Python (3.11 or newer) with PyTorch; their dependencies are declared in `pyproject.toml` and installed with `uv sync`. Trained models are evaluated in C inside SCIP, so Python is not needed to run them.

## Setup and installation

The following instructions describe a source-based installation of SCIP and the configuration of Learn2MILP on Ubuntu.

### 1. Install system dependencies

Update the package list and install the required development tools:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libmpfr-dev
```

Additional dependencies may be required depending on the SCIP build configuration.

### 2. Download and extract SCIP

Download SCIP Optimization Suite 10.1.0 from the official [SCIP website](https://scipopt.org/).

Extract the archive and place the source directory under `~/opt/`:

```bash
mkdir -p ~/opt
```

For the following instructions, the extracted directory is assumed to be:

```text
~/opt/scipoptsuite-10.1.0
```

### 3. Configure SCIP

Navigate to the SCIP source directory:

```bash
cd ~/opt/scipoptsuite-10.1.0
```

Configure the build:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DAUTOBUILD=ON \
    -DPAPILO=OFF
```

`PAPILO=OFF` disables the optional PaPILO component. This can reduce compilation time and memory consumption when PaPILO is not needed.

### 4. Build and install SCIP

Compile SCIP using a limited number of parallel jobs:

```bash
cmake --build build --parallel 2
```

The number of parallel jobs can be increased depending on available system memory.

Install the compiled libraries and headers:

```bash
sudo cmake --install build
```

### 5. Configure the dynamic library loader

Add `/usr/local/lib` to the dynamic linker configuration:

```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/scip.conf
```

Update the library cache:

```bash
sudo ldconfig
```

Verify that SCIP is available:

```bash
ldconfig -p | grep scip
```

The output should contain entries for `libscip.so` and `libscip.so.10.1`.

The main SCIP headers should also be installed:

```bash
ls /usr/local/include/scip/scip.h
ls /usr/local/include/scip/scipdefplugins.h
```

### 6. Clone Learn2MILP

Clone the repository:

```bash
mkdir -p ~/projects
cd ~/projects

git clone https://github.com/raphaelDupuy/learn2milp.git
cd learn2milp
```

### 7. Configure and build Learn2MILP

Configure the project using CMake:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release
```

Compile the project:

```bash
cmake --build build --parallel 2
```

The build produces the following executables:

- `learn2milp`: main optimization program;
- `mis-generator`: MIS instance generator;
- `mis-collector`: strong branching dataset collector;
- `mis-featurizer`: computes training tables from collected samples;
- `mis-benchmark`: branching rule benchmark;
- one `test_*` executable per test.

### 8. Run the tests

Run the test suite using CTest:

```bash
ctest --test-dir build --output-on-failure
```

All tests should pass before starting experiments.

### 9. Run the main program

Execute the main application:

```bash
./build/learn2milp
```

The program generates an Erdős–Rényi graph, builds the corresponding MIS model, and solves it using SCIP.

The solver output includes the optimization status, objective value, Branch-and-Bound statistics, and solving time.

## Reproducibility

Random instances are generated using explicit seeds. This makes individual experiments reproducible and allows solver behaviour to be compared across different configurations.

## Long-term objective

The long-term objective is to develop and evaluate a learning-augmented Branch-and-Cut framework in which machine learning assists the solver in selected algorithmic decisions while the underlying optimization remains exact.

The project will be evaluated not only through prediction accuracy, but primarily through computational measures such as solving time, branch-and-bound nodes, LP iterations, cuts, and ML inference overhead.
