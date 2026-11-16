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

## Documentation

| Document | Content |
|---|---|
| [Installation](docs/installation.md) | SCIP 10.1 from source, build, tests, Python environment |
| [Interactive menu](docs/interactive-menu.md) | `learn2milp`: generate, solve, compare rules, inspect datasets, file browser |
| [Command line tools](docs/command-line-tools.md) | `mis-generator`, `mis-collector`, `mis-featurizer`, `mis-benchmark`, dataset scripts |
| [Data formats](docs/data-format.md) | instances, strong branching samples, training tables, benchmark results |
| [Learning a branching rule](docs/learning-to-branch.md) | features, training, the `mislearned` rule, on-policy samples |
| [Results](docs/results.md) | learned rules against SCIP's default on Erdős–Rényi and Barabási–Albert graphs |
| [Project notes](docs/project-notes.md) | working notes on the research questions |

## Quick start

After [installing SCIP and building the project](docs/installation.md):

```bash
# Generate an instance and solve it with SCIP's default rule and a learned model
./build/mis-generator single --n 125 --p 0.1 --seed 1 --output data/instances/g125.mis
./build/mis-benchmark --instances data/instances --rules relpscost,mislearned \
    --model experiments/er125_p01/models/mlp32_h2.model --output results.csv

# Or do the same, and more, from the interactive menu
./build/learn2milp
```

## Current status

The current version provides an experimental framework for learning to branch on the Maximum Independent Set problem.

### Implemented

- C++17 project using CMake, integrated with SCIP;
- MILP formulation of Maximum Independent Set;
- reproducible instance generation (Erdős–Rényi and Barabási–Albert graphs) using explicit random seeds, one instance at a time or in batches;
- full strong branching rule for SCIP that records every branching decision, checked node by node against SCIP's own `vanillafullstrong` rule;
- learning-to-branch datasets, from the expert's search and on-policy (`mis-collector`);
- benchmark of branching rules on a set of instances (`mis-benchmark`);
- a learned branching rule: candidate features, training of a small ranking model in Python (a multilayer perceptron, or a graph neural network over the candidates, on the CPU or a GPU), and its evaluation inside SCIP (`mislearned`);
- an interactive menu to try rules and models and to inspect datasets (`learn2milp`).

Learned rules explore 25 to 30% fewer nodes than SCIP's default rule on the Erdős–Rényi graphs they were trained on, and the model using the search history keeps up with it on Barabási–Albert graphs never seen in training, about twice as fast. A graph neural network over the candidates imitates the expert better than the perceptron but explores 5 to 12% more nodes; see [Results](docs/results.md).

### Planned

- richer learned branching policies (a bipartite graph model over variables and constraints, as in Gasse et al. 2019; other training signals);
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

The implementation generates random graphs of two families, builds the corresponding MILP model, and solves it exactly using SCIP.

## Project structure

```text
learn2milp/
├── CMakeLists.txt
├── README.md
├── docs/            documentation
├── data/            instances and datasets (not versioned)
├── experiments/     dataset recipes, trained models and result tables
├── include/
│   ├── branching/   SCIP branching rules (expert, learned)
│   ├── features/
│   ├── generators/  graph families
│   ├── instances/
│   ├── io/          file formats
│   ├── learning/    branching features and model evaluation (C)
│   ├── problems/
│   ├── services/    generation, collection, featurization, benchmark, ...
│   ├── tools/       parsing of typed values, shared by the tools and the menu
│   └── ui/          interactive menu
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
