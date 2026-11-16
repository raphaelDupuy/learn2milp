# Command line tools

The build produces four tools, used by the scripts of `scripts/` and `experiments/`. Every tool prints its options with `--help`. A malformed argument is reported with what is wrong (unknown, repeated or missing option, invalid value) and exit code 2. Existing output files are never overwritten.

The files they read and write are described in [data-format.md](data-format.md).

## `mis-generator`: instances

`mis-generator` writes MIS instances, one graph per `.mis` file with its generator and parameters in a `.mis.meta` file next to it. Two graph families are available (`--family`):

- `erdos_renyi` (the default): \(G(n, p)\), every edge present with probability \(p\);
- `barabasi_albert`: preferential attachment. The graph starts with \(k\) isolated vertices; every new vertex is joined to \(k\) distinct existing vertices, drawn with probability proportional to their degree, until there are \(n\) vertices. The graph has \(k(n-k)\) edges, is connected and has a few vertices of high degree, unlike \(G(n, p)\). (\(k\) is often written \(m\) in the literature; here \(m\) is the number of edges, as in the instance files.)

```bash
./build/mis-generator single --n 125 --p 0.1 --seed 1 --output data/instances/g125.mis
./build/mis-generator single --family barabasi_albert --n 200 --k 8 --seed 1 --output data/instances/ba200.mis

./build/mis-generator batch --family barabasi_albert \
    --n-values 200 --k-values 8 --seeds 81,82,83 \
    --output-dir data/instances/ba_test
```

A batch generates every combination of the parameter lists and names each instance after its family and parameters (`erdos_renyi_n125_p0_1_seed81`, `barabasi_albert_n200_k8_seed81`). Its `manifest.csv` lists the instances with their parameters and number of edges `m`; it is written last, so its presence means the batch is complete. A graph only depends on its parameters and seed, on every platform.

## `mis-collector`: strong branching samples

`mis-collector` solves one MIS instance with full strong branching and records one sample per branching decision: the candidates, the outcome of strong branching on each of them, and the variable chosen by this expert.

```bash
./build/mis-generator single --n 125 --p 0.1 --seed 1 --output data/instances/g125.mis

./build/mis-collector \
    --instance data/instances/g125.mis \
    --output-dir data/raw/strong_branching \
    --time-limit 600
```

Optional limits are `--time-limit SECONDS` (default 3600), `--node-limit N` and `--scip-seed S`. With `--policy-model MODEL`, a trained model drives the search while strong branching still labels every visited node (on-policy samples, see [learning-to-branch.md](learning-to-branch.md#4-on-policy-samples)).

For an instance `NAME.mis`, the collector writes `NAME.sb.nodes.csv`, `NAME.sb.candidates.csv` and `NAME.sb.meta` (see [data-format.md](data-format.md#strong-branching-samples)).

### Solver settings

The same SCIP settings, called the study settings, are used to collect the data and to evaluate the learned policies:

- presolving enabled;
- restarts disabled (`presolving/maxrestarts = 0`), so that the variable space does not change during the search;
- cutting planes at the root node only (`separating/maxrounds = 0`), so that the effect of branching is isolated.

### Collecting a whole directory

`scripts/collect_strong_branching.sh` runs the collector on every instance of a directory, several instances at a time. It can be interrupted and restarted: instances already collected are skipped.

```bash
scripts/collect_strong_branching.sh build/mis-collector INSTANCE_DIR OUTPUT_DIR 4
```

Two more optional arguments give the time limit per instance and a policy model.

`experiments/er125_p01/build_dataset.sh` is the recipe of the first dataset: Erdős–Rényi graphs G(125, 0.1) split into training (60 graphs), validation (20) and test (20) sets, plus 20 graphs G(150, 0.05) to measure generalization. Samples are collected for the training and validation sets only; the other two sets are used to evaluate the solver.

```bash
experiments/er125_p01/build_dataset.sh build 4
```

`experiments/ba200_k8/build_dataset.sh` does the same with Barabási–Albert graphs (200 vertices, \(k = 8\)), with the same splits and seeds. `scripts/collect_on_policy.sh` adds [on-policy samples](learning-to-branch.md#4-on-policy-samples) to either dataset.

## `mis-featurizer`: training tables

`mis-featurizer` turns the samples collected on one instance into a training table, `NAME.sb.features.csv`, with one row per candidate: the features and the expert's answer.

```bash
./build/mis-featurizer \
    --instance data/instances/g125.mis \
    --samples-dir data/raw/strong_branching \
    --output-dir data/processed/features
```

`scripts/featurize_strong_branching.sh` does the same for every collection of a directory:

```bash
scripts/featurize_strong_branching.sh build/mis-featurizer INSTANCE_DIR SAMPLES_DIR OUTPUT_DIR 4
```

The features are described in [learning-to-branch.md](learning-to-branch.md#1-features).

## `mis-benchmark`: evaluating branching rules

`mis-benchmark` solves every instance of a directory with each branching rule and each SCIP seed, and writes one CSV row per run: status, explored nodes, LP iterations and solving time.

```bash
./build/mis-benchmark \
    --instances data/instances/er125_p01/test \
    --rules relpscost,pscost,misfullstrong \
    --seeds 0,1,2 \
    --output experiments/er125_p01/baselines.csv
```

- `--rules` takes the name of any SCIP branching rule (`relpscost` is SCIP's default), `misfullstrong`, the full strong branching expert recorded by the data collection, `misfullstrong_nocutoff`, the same expert without its pruning of the nodes that strong branching proves prunable (the tree a perfect imitation of its choices would explore), or `mislearned`, the learned rule driven by the model of `--model`;
- `--settings study` (the default) uses the [study settings](#solver-settings), so that rules are compared in the conditions they were learned in; `--settings default` runs SCIP with no parameter changed, as an outside reference;
- `--time-limit SECONDS` limits each run (default 3600);
- runs are executed one at a time, with the rules interleaved on each instance: do not run anything else on the machine if solving times matter.

The tool ends with a summary per rule: number of solved runs and shifted geometric means of the node counts (shift 100) and of the solving times (shift 1 second). Node counts are reproducible for a given seed; solving times are not, and should be compared over several runs.
