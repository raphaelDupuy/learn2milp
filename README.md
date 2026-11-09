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
- reproducible instance generation using explicit random seeds;
- exact optimization and basic solver statistics.

### Planned

- systematic solver statistics collection;
- batch instance generation and benchmarking;
- learning-to-branch dataset generation;
- learned branching policies;
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

## Project structure

```text
learn2milp/
├── CMakeLists.txt
├── README.md
├── docs/
├── data/
├── include/
│   ├── generators/
│   ├── io/
│   └── problems/
├── src/
│   ├── generators/
│   ├── io/
│   ├── problems/
│   ├── tools/
│   └── main.cpp
└── tests/
```

The code is deliberately separated into problem definitions, instance generators, and solver-related components. This structure makes it possible to add new instance distributions and optimization problems without coupling them to the data generation process.

## Dependencies

- C++17 compiler;
- CMake (version 3.20 or newer);
- SCIP Optimization Suite 10.1.0;
- Git.

The machine learning components will be developed separately, primarily using Python.

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
- `test_MISInstanceIO`: instance serialization test executable.

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
