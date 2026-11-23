# Learn2MILP

## Overview

Learn2MILP is a research-oriented project on learning-augmented exact methods for combinatorial optimization.

The main objective is to investigate whether machine learning can improve selected decision mechanisms of Mixed-Integer Linear Programming (MILP) solvers while preserving exactness.

The project initially focuses on two components of Branch-and-Cut:

- branching decisions;
- cutting-plane selection.

The first problem considered is the Maximum Independent Set (MIS). The second is the Job-Shop Scheduling Problem (JSSP), at the pilot stage.

## Research questions

The project investigates the following questions:

1. Can branching decisions in exact MILP solving be learned from solver data?
2. Can cutting-plane selection be improved using machine learning?
3. Are learned branching and cutting policies complementary?
4. How well do learned policies generalize across instance sizes and distributions?
5. Does the reduction in solver effort compensate for the computational cost of ML inference?

A later direction is to compare policies trained to imitate local solver decisions with approaches whose objective is more directly related to the global performance of the search process.

## Documentation

| Document | Content |
|---|---|
| [Installation](docs/installation.md) | SCIP 10.1 from source, build, tests, Python environment |
| [Interactive menu](docs/interactive-menu.md) | `learn2milp`: generate, solve, compare rules, inspect datasets, file browser |
| [Command line tools](docs/command-line-tools.md) | `mis-generator`, `mis-collector`, `mis-featurizer`, `mis-benchmark`, dataset scripts |
| [Data formats](docs/data-format.md) | instances, strong branching samples, training tables, benchmark results |
| [Learning a branching rule](docs/learning-to-branch.md) | features, training, the `mislearned` rule, on-policy samples |
| [Results](docs/results.md) | MIS: learned branching, cutting planes, solver settings, time profiles, final run |
| [Project notes](docs/project-notes.md) | research questions, status on MIS, next steps |

## Quick start

After [installing SCIP and building the project](docs/installation.md):

```bash
# Generate an instance and solve it with SCIP's default rule and a learned model
./build/mis-generator single --n 125 --p 0.1 --seed 1 --output data/instances/g125.mis
./build/mis-benchmark --instances data/instances --rules relpscost,mislearned \
    --model experiments/er125_p01/models/mlp32_h2.model --output results.csv

# The same with the settings of the final configuration (cuts at the nodes)
./build/mis-benchmark --instances data/instances --rules relpscost,mislearned \
    --model experiments/er125_p01/models/mlp32_h2.model \
    --parameters experiments/cuts/settings/oddcycle_cliquelifting_nogomory.set \
    --output results.csv

# Or do the same, and more, from the interactive menu
./build/learn2milp
```

## Current status

The experiments on MIS are done. The JSSP pilot is in progress (`experiments/jssp/pilot/`).

### Implemented

- C++17 project using CMake, integrated with SCIP 10.1;
- MILP formulation of Maximum Independent Set;
- reproducible instance generation (Erdős–Rényi and Barabási–Albert graphs) using explicit random seeds, one instance at a time or in batches;
- full strong branching rule for SCIP that records every branching decision, checked node by node against SCIP's own `vanillafullstrong` rule;
- learning-to-branch datasets, from the expert's search and on-policy (`mis-collector`);
- a learned branching rule: candidate features in C, training of a small ranking model in Python (a multilayer perceptron, a graph neural network over the candidates or a bipartite model of variables and constraints after Gasse et al. 2019, trained on the CPU or a GPU), and its evaluation inside SCIP (`mislearned`), optionally with strong branching on the model's first candidates;
- cutting-plane experiments: separator settings, a lookahead cut selector (Paulus et al. 2022), a separation gate and node logs (SCIP plugins);
- benchmark of rules and settings on a set of instances, with SCIP's statistics and an optimal-solution oracle for the primal heuristics (`mis-benchmark`);
- an interactive menu to try rules and models and to inspect datasets (`learn2milp`).

### Results on MIS

- **Branching.** The learned perceptron is faster than SCIP's `relpscost` on every test set, and on graphs up to 1.6 times larger than in training without retraining. Graph neural networks imitate the expert better offline but give larger trees in SCIP.
- **Cutting planes.** Learned cut selection has no room on MIS: selection at the root closes no more gap than SCIP's, and choosing the separators per instance gains only a few percent over the best fixed choice. The gain comes from fixed settings: no Gomory cuts, odd-cycle cuts at the nodes, and SCIP's clique lifting.
- **Together.** The learned rule with these settings takes 0.14 to 0.30 of the time of SCIP as shipped, and 0.18 to 0.67 of the time of the best SCIP configuration, on nine test sets. Features and inference take 2 to 7% of its time.

Details are in [Results](docs/results.md) and the status of each research question in [Project notes](docs/project-notes.md).

### Planned

- extension to Job-Shop Scheduling: generator, formulation, and whether strong branching is worth imitating when the LP bound is weak;
- evaluation of the MIS models on other graph families;
- comparison of imitation with training signals tied to the whole search (tree-level objectives).

## Maximum Independent Set

For a graph $`G=(V,E)`$, the Maximum Independent Set problem is formulated as a binary linear program.

Each vertex $`v`$ is associated with a binary variable $`x_v`$.

The objective is:

```math
\max \sum_{v \in V} x_v
```

subject to:

```math
x_u + x_v \leq 1
\qquad \forall (u,v) \in E
```

and:

```math
x_v \in \{0,1\}
\qquad \forall v \in V.
```

The implementation generates random graphs of two families, builds the corresponding MILP model, and solves it exactly using SCIP. In the final configuration, SCIP's clique lifting (`constraints/setppc/cliquelifting`) turns the edge constraints into clique constraints.

## Project structure

```text
learn2milp/
├── CMakeLists.txt
├── README.md
├── docs/            documentation
├── data/            instances and datasets (not versioned)
├── experiments/     dataset recipes, trained models, settings, result tables, JSSP pilot
├── include/
│   ├── branching/   SCIP branching rules (expert, learned)
│   ├── cuts/        cut selector, separation gate and node logs (SCIP plugins)
│   ├── features/
│   ├── generators/  graph families
│   ├── instances/
│   ├── io/          file formats
│   ├── learning/    branching features and model evaluation (C)
│   ├── problems/
│   ├── services/    generation, collection, featurization, benchmark, ...
│   ├── tools/       parsing of typed values, shared by the tools and the menu
│   ├── ui/          interactive menu: input, file browser, screens
│   └── util/        small helpers shared by every part
├── ml/              training (Python)
├── pyproject.toml
├── src/             implementations, the tools' main files and main.cpp
├── scripts/         batch collection and featurization
└── tests/
```

The code is deliberately separated into problem definitions, instance generators, and solver-related components. This structure makes it possible to add new instance distributions and optimization problems without coupling them to the data generation process.

## Reproducibility

Random instances are generated using explicit seeds, and the strong branching samples are reproducible byte for byte for a given instance, limits and seed. This makes individual experiments reproducible and allows solver behaviour to be compared across different configurations.

## Long-term objective

The long-term objective is to develop and evaluate a learning-augmented Branch-and-Cut framework in which machine learning assists the solver in selected algorithmic decisions while the underlying optimization remains exact.

The project will be evaluated not only through prediction accuracy, but primarily through computational measures such as solving time, branch-and-bound nodes, LP iterations, cuts, and ML inference overhead.
