"""Where the solving time goes, part by part of SCIP.

Reads the statistics that run_ablation.sh writes with STATISTICS=1 (one SCIP
JSON statistics file per run, stats/<set>__<config>/) and the result tables
of the same experiment (<set>_<config>.csv, for the time of the learned
rule), splits the solving time of every run into parts that do not overlap,
and averages them over the instances of each set and configuration.

    ~/.local/bin/uv run --project ml python experiments/profile/time_breakdown.py \\
        experiments/cuts/results/time_profile

Prints, for each set and configuration, the mean solving time and the share
of each part (mean over the instances of the per-instance shares, so that
every instance weighs the same), and writes every part, in seconds and as a
share, to time_breakdown.csv in the experiment directory.

The parts:
  presolve        presolving, clique lifting included
  lp              LP solves of the nodes (first LP and cut loop resolves)
  strong_branch   LP solves of strong branching
  branching       the rest of the branching rule (scores, pseudo-costs)
  features        learned rule: features of the candidates
  inference       learned rule: evaluation of the model
  sep_<name>      each separator; sep_other gathers the other separators,
                  the separation of the constraint handlers and the cut
                  selector
  heuristics      primal heuristics, their diving LPs included
  propagation     propagators and constraint propagation
  conflict        conflict analysis
  other           the rest of the solving time (node selection, tree
                  management, LP setup); a large value would mean that
                  the parts above miss something
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

SEPARATORS = ["gomory", "zerohalf", "oddcycle", "clique", "aggregation", "cut_pool"]

PARTS = (["presolve", "lp", "strong_branch", "branching", "features", "inference"]
         + [f"sep_{name}" for name in SEPARATORS]
         + ["sep_other", "heuristics", "propagation", "conflict", "other"])

# Groups printed in the summary table.
GROUPS = {
    "presolve": ["presolve"],
    "LP": ["lp"],
    "SB": ["strong_branch"],
    "branch": ["branching"],
    "ML": ["features", "inference"],
    "sep": [p for p in PARTS if p.startswith("sep_")],
    "heur": ["heuristics"],
    "prop": ["propagation"],
    "confl": ["conflict"],
    "other": ["other"],
}


def plugins(table):
    """Plugins of a statistics table, without the entries that are not
    plugins (description, solutions found outside of any heuristic)."""
    return {name: data for name, data in table.get("plugins", table).items()
            if isinstance(data, dict)}


def breakdown(stats, features, inference):
    """Seconds of each part of one run."""
    seconds = dict.fromkeys(PARTS, 0.0)
    total = stats["timing"]["solving_time"]
    seconds["presolve"] = stats["timing"]["presolving_time"]

    lp = stats["lp"]
    seconds["lp"] = sum(lp[name]["time"] for name in
                        ["primal_lp", "dual_lp", "barrier_lp", "lex_dual_lp", "resolve_lp"])
    seconds["strong_branch"] = lp["strongbranch_lp"]["time"]

    heuristics = {name: data.get("time", 0.0) for name, data in plugins(stats["heuristics"]).items()}
    # Solutions found by strong branching are checked inside it.
    strong_branching_solutions = heuristics.pop("strong_branching", 0.0)

    rules = sum(data["time"] for data in plugins(stats["branchrules"]).values())
    seconds["features"] = features
    seconds["inference"] = inference
    seconds["branching"] = max(0.0, rules - seconds["strong_branch"] - features - inference
                               - strong_branching_solutions)

    for name, data in plugins(stats["separator"]).items():
        key = f"sep_{name}" if name in SEPARATORS else "sep_other"
        seconds[key] += data["exec_time"]

    constraint_times = plugins(stats["constiming"])
    seconds["sep_other"] += sum(data["separation_time"] for data in constraint_times.values())
    seconds["sep_other"] += sum(data["exec_time"] for data in plugins(stats["cutsel"]).values())

    seconds["heuristics"] = sum(heuristics.values())

    seconds["propagation"] = sum(data.get("total_time", 0.0) - data.get("presolve_time", 0.0)
                                 for data in plugins(stats["propagator"]).values())
    seconds["propagation"] += sum(data["propagation_time"] for data in constraint_times.values())

    conflict = stats["conflict"]
    seconds["conflict"] = sum(conflict[name] for name in
                              ["propagation_time", "infeasible_lp_time",
                               "bound_exceeding_lp_time", "strong_branching_time"])

    seconds["other"] = total - sum(seconds.values())
    return total, seconds


def learned_times(table):
    """Instance -> (features, inference) seconds, from a result table."""
    times = {}
    with open(table) as rows:
        for row in csv.DictReader(rows):
            times[row["instance"]] = (float(row.get("features_seconds") or 0.0),
                                      float(row.get("inference_seconds") or 0.0))
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    runs = defaultdict(list)  # (set, config) -> [(total, seconds)]
    for folder in sorted((args.directory / "stats").iterdir()):
        set_name, config = folder.name.split("__", 1)
        table = args.directory / f"{set_name}_{config}.csv"
        learned = learned_times(table) if table.exists() else {}
        for path in sorted(folder.glob("*.stats.json")):
            instance = path.name.split("__")[0] + ".mis"
            with open(path) as file:
                stats = json.load(file)
            runs[set_name, config].append(breakdown(stats, *learned.get(instance, (0.0, 0.0))))

    output = args.directory / "time_breakdown.csv"
    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["set", "config", "runs", "mean_seconds"]
                        + [f"{p}_seconds" for p in PARTS] + [f"{p}_share" for p in PARTS])
        print(f"{'set':6} {'configuration':34} {'runs':>4} {'time':>8}  "
              + " ".join(f"{g:>7}" for g in GROUPS))
        for (set_name, config), items in runs.items():
            n = len(items)
            mean_total = sum(total for total, _ in items) / n
            mean_seconds = {p: sum(s[p] for _, s in items) / n for p in PARTS}
            share = {p: sum(s[p] / total for total, s in items if total > 0) / n for p in PARTS}
            writer.writerow([set_name, config, n, f"{mean_total:.4f}"]
                            + [f"{mean_seconds[p]:.4f}" for p in PARTS]
                            + [f"{share[p]:.4f}" for p in PARTS])
            print(f"{set_name:6} {config:34} {n:4d} {mean_total:8.2f}  "
                  + " ".join(f"{100 * sum(share[p] for p in parts):6.1f}%"
                             for parts in GROUPS.values()))
    print(f"\nWritten: {output}")


if __name__ == "__main__":
    main()
