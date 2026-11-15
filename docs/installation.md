# Installation

These instructions install SCIP from source and build Learn2MILP on Ubuntu.

## Dependencies

- C++17 compiler;
- CMake (version 3.20 or newer);
- SCIP Optimization Suite 10.1.0;
- Git.

The machine learning components are written in Python (3.11 or newer) with PyTorch; their dependencies are declared in `pyproject.toml` and installed with [uv](https://docs.astral.sh/uv/). Trained models are evaluated in C inside SCIP, so Python is not needed to run them.

## 1. Install system dependencies

Update the package list and install the required development tools:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libmpfr-dev
```

Additional dependencies may be required depending on the SCIP build configuration.

## 2. Download and extract SCIP

Download SCIP Optimization Suite 10.1.0 from the official [SCIP website](https://scipopt.org/).

Extract the archive and place the source directory under `~/opt/`:

```bash
mkdir -p ~/opt
```

For the following instructions, the extracted directory is assumed to be:

```text
~/opt/scipoptsuite-10.1.0
```

## 3. Configure SCIP

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

## 4. Build and install SCIP

Compile SCIP using a limited number of parallel jobs:

```bash
cmake --build build --parallel 2
```

The number of parallel jobs can be increased depending on available system memory.

Install the compiled libraries and headers:

```bash
sudo cmake --install build
```

## 5. Configure the dynamic library loader

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

## 6. Clone Learn2MILP

Clone the repository:

```bash
mkdir -p ~/projects
cd ~/projects

git clone https://github.com/raphaelDupuy/learn2milp.git
cd learn2milp
```

## 7. Configure and build Learn2MILP

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

- `learn2milp`: interactive terminal menu (see [interactive-menu.md](interactive-menu.md));
- `mis-generator`: MIS instance generator;
- `mis-collector`: strong branching dataset collector;
- `mis-featurizer`: computes training tables from collected samples;
- `mis-benchmark`: branching rule benchmark;
- one `test_*` executable per test.

The four `mis-*` tools are described in [command-line-tools.md](command-line-tools.md).

## 8. Run the tests

Run the test suite using CTest:

```bash
ctest --test-dir build --output-on-failure
```

All tests should pass before starting experiments.

## 9. Set up the Python environment

Only needed to train models:

```bash
uv sync
```

Python scripts are then run with `uv run python ml/...` (see [learning-to-branch.md](learning-to-branch.md)).
