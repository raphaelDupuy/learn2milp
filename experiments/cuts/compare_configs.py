"""Ratios of the configurations of a cut experiment to a reference one.

For every set and configuration of an experiment directory written by
run_ablation.sh (SET_CONFIG.csv), prints the geometric mean over the instances
of the per-instance ratio to the reference configuration, for the solving
time (unshifted) and the node count (shifted by 100), with a 95% paired
bootstrap interval, and the shifted geometric means of both. Only the
instances solved by both configurations count.

    ~/.local/bin/uv run --project ml python experiments/cuts/compare_configs.py \
        experiments/cuts/results/node_gate --reference study
"""
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

NODE_SHIFT = 100
TIME_SHIFT = 1
BOOTSTRAP = 10000


def load(path):
    """Instance -> (time, nodes, gate separated, gate skipped), solved runs."""
    runs = {}
    with open(path) as table:
        for row in csv.DictReader(table):
            if row["solved"] != "1":
                continue
            runs[row["instance"]] = (
                float(row["time_seconds"]),
                int(row["nodes"]),
                int(row.get("gate_separated") or 0),
                int(row.get("gate_skipped") or 0),
            )
    return runs


def mean_log(values):
    return sum(values) / len(values)


def ratio_with_interval(logs, rng):
    """Geometric mean of exp(logs) and its 95% bootstrap interval."""
    n = len(logs)
    means = sorted(mean_log([logs[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOTSTRAP))
    low, high = means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]
    return math.exp(mean_log(logs)), math.exp(low), math.exp(high)


def shifted_geomean(values, shift):
    return math.exp(mean_log([math.log(v + shift) for v in values])) - shift


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", default="study")
    parser.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    args = parser.parse_args()

    tables = defaultdict(dict)  # set -> config -> runs
    for path in sorted(args.directory.glob("*.csv")):
        set_name, config = path.stem.split("_", 1)
        tables[set_name][config] = load(path)

    for set_name, configs in tables.items():
        reference = configs.get(args.reference)
        if reference is None:
            print(f"{set_name}: no {args.reference} table, skipped")
            continue
        print(f"\n{set_name} (ratios to {args.reference}: time unshifted, nodes shifted by {NODE_SHIFT})")
        print(f"{'configuration':28} {'n':>3} {'time ratio [95%]':>22} {'node ratio [95%]':>22}"
              f" {'time sgm':>9} {'nodes sgm':>10} {'gate sep.':>9}")
        for config, runs in sorted(configs.items()):
            common = sorted(set(runs) & set(reference))
            if not common:
                continue
            rng = random.Random(args.seed)
            time_logs = [math.log(runs[i][0] / reference[i][0]) for i in common]
            node_logs = [math.log((runs[i][1] + NODE_SHIFT) / (reference[i][1] + NODE_SHIFT)) for i in common]
            time = ratio_with_interval(time_logs, rng)
            nodes = ratio_with_interval(node_logs, rng)
            separated = sum(runs[i][2] for i in common)
            decided = separated + sum(runs[i][3] for i in common)
            gate = f"{separated / decided:9.0%}" if decided else f"{'':9}"
            print(f"{config:28} {len(common):3d}"
                  f" {time[0]:6.2f} [{time[1]:5.2f}, {time[2]:5.2f}]"
                  f" {nodes[0]:6.2f} [{nodes[1]:5.2f}, {nodes[2]:5.2f}]"
                  f" {shifted_geomean([runs[i][0] for i in common], TIME_SHIFT):9.2f}"
                  f" {shifted_geomean([runs[i][1] for i in common], NODE_SHIFT):10.0f} {gate}")


if __name__ == "__main__":
    main()
