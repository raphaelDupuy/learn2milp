# Learn2MILP

## Overview

Learn2MILP is a research-oriented project on learning-augmented exact methods for combinatorial optimization.

The main objective is to investigate whether machine learning can improve selected decision mechanisms of Mixed-Integer Linear Programming (MILP) solvers while preserving exactness.

The project initially focuses on two components of Branch-and-Cut:

* branching decisions;
* cutting-plane selection.

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

* C++17 project using CMake;
* integration with SCIP;
* MILP formulation of Maximum Independent Set;
* Erdős–Rényi random graph generation;
* reproducible instance generation using explicit random seeds;
* exact optimization and basic solver statistics.

### Planned

* systematic solver statistics collection;
* batch instance generation and benchmarking;
* learning-to-branch dataset generation;
* learned branching policies;
* learning-to-cut;
* comparison of classical and learned Branch-and-Cut configurations;
* experiments on generalization across instance distributions;
* extension to Job-Shop Scheduling.

## Maximum Independent Set

For a graph \(G=(V,E)\), the Maximum Independent Set problem is formulated as a binary linear program.

Each vertex \(v\) is associated with a binary variable \(x_v\).

The objective is:

$$
\max \sum_{v \in V} x_v
$$

subject to:

$$
x_u + x_v \leq 1
\qquad \forall (u,v) \in E
$$

and:

$$
x_v \in \{0,1\}
\qquad \forall v \in V.
$$

The current implementation generates Erdős–Rényi graphs \(G(n,p)\), builds the corresponding MILP model, and solves it exactly using SCIP.

## Project structure

```text
learn2milp/
├── CMakeLists.txt
├── README.md
├── docs/
├── data/
├── include/
│   ├── generators/
│   └── problems/
└── src/
    ├── generators/
    ├── problems/
    └── main.cpp
```

The code is deliberately separated into problem definitions, instance generators, and solver-related components. This structure makes it possible to add new instance distributions and optimization problems without coupling them to the data generation process.

## Dependencies

* C++17
* CMake
* SCIP

The machine learning components will be developed separately, primarily using Python.

## Reproducibility

Random instances are generated using explicit seeds. This makes individual experiments reproducible and allows solver behaviour to be compared across different configurations.

## Long-term objective

The long-term objective is to develop and evaluate a learning-augmented Branch-and-Cut framework in which machine learning assists the solver in selected algorithmic decisions while the underlying optimization remains exact.

The project will be evaluated not only through prediction accuracy, but primarily through computational measures such as solving time, branch-and-bound nodes, LP iterations, cuts, and ML inference overhead.
