"""Per-node headroom of the cut loop, from the node logs of mis-benchmark.

For the configurations that separate at every node, each node below the root
is classified by what its cut loop did:

  pruned   the first LP did not reach the cutoff bound, the cut loop did:
           without the cuts, the node would have been branched on;
  raised   the bound rose without pruning the node;
  idle     the bound did not move.

The script then asks whether the useful loops can be told apart before
separating, from what is known after the first LP of the node: the gap to the
cutoff bound, the depth, the fractional variables, the distance to the global
lower bound. A gate that separates only at the nodes it ranks first is scored
by the share of the cut-loop time it spends against the share of the pruned
nodes it keeps (a curve like a ROC curve, where time replaces the count of
nodes).

These are measures on the tree of the configuration that separates
everywhere. Skipping a loop changes the tree (the node is branched on, its
cuts are not in the pool for the others), so the curves bound what a gate can
keep, they do not simulate it.

Usage (from the repository root):
  ~/.local/bin/uv run --project ml python experiments/cuts/analyze_node_logs.py \
      [experiments/cuts/results/node_mechanisms]

Outputs, in <results>/analysis/:
  outcomes.csv   per set and configuration: loops by outcome, share of
                 nodes and of cut-loop time
  features.csv   per set and configuration: AUC of each feature, alone, for
                 telling pruned loops from the others
  gate.csv       per set, configuration and gate: share of cut-loop time
                 needed to keep 50, 80 and 90 % of the pruned nodes
  gate_curves.csv  the curves themselves (100 points per gate)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/cuts/results/node_mechanisms")
OUT = RESULTS / "analysis"
SETS = ["er125", "ba200"]
CONFIGS = ["oddcycle_every_node", "zerohalf_every_node", "zerohalf_local_every_node"]
TOL = 1e-6
TRAIN = range(81, 91)   # instances whose nodes fit the gate model
TEST = range(91, 101)   # instances it is scored on


def load(set_name: str, config: str) -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "nodes" / f"{set_name}__{config}").glob("*.nodes.csv")):
        frame = pd.read_csv(path)
        frame["instance"] = int(path.name.split("__")[0].rsplit("seed", 1)[1])
        frames.append(frame)
    nodes = pd.concat(frames, ignore_index=True)
    return nodes[nodes["depth"] > 0].copy()


def classify(nodes: pd.DataFrame) -> pd.DataFrame:
    lp_first, lp_loop = nodes["lp_first"], nodes["lp_cut_loop"]
    cutoff = nodes["cutoff_final"]
    looped = nodes["separation_rounds"] > 0
    # A first LP that is infeasible or reaches the cutoff leaves no loop to judge.
    open_first = np.isfinite(lp_first) & (lp_first < cutoff - TOL)
    nodes["looped"] = looped & open_first
    nodes["pruned"] = nodes["looped"] & (~np.isfinite(lp_loop) | (lp_loop >= cutoff - TOL))
    gain = (lp_loop - lp_first).where(np.isfinite(lp_loop), np.nan)
    nodes["gain"] = gain
    nodes["raised"] = nodes["looped"] & ~nodes["pruned"] & (gain > TOL)
    nodes["idle"] = nodes["looped"] & ~nodes["pruned"] & ~(gain > TOL)

    # What is known before separating (after the node's first LP).
    gap = nodes["cutoff_first"] - lp_first
    nodes["gap"] = gap
    span = nodes["cutoff_first"] - nodes["global_lower"]
    nodes["bound_distance"] = ((lp_first - nodes["global_lower"]) / span).clip(0, 1)
    nodes["frac_share"] = nodes["fractional"] / nodes["fractional"].groupby(nodes["instance"]).transform("max")
    nodes["half_share"] = (nodes["half"] / nodes["fractional"].replace(0, np.nan)).fillna(0)
    nodes["log_first_iters"] = np.log1p(nodes["first_lp_iterations"])
    return nodes


FEATURES = {
    # name: (column, sign) with sign +1 when larger means "more likely pruned"
    "gap (small)": ("gap", -1),
    "depth (deep)": ("depth", +1),
    "bound distance (far from global lower bound)": ("bound_distance", +1),
    "fractional variables (few)": ("frac_share", -1),
    "share of fractional values at 1/2 (high)": ("half_share", +1),
    "first LP iterations (few)": ("log_first_iters", -1),
}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a pruned loop scores above another (ties count half)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    positives = label.sum()
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def logistic(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Logistic regression fitted by Newton's method; returns test scores."""
    def design(frame):
        return frame[columns].to_numpy(float)
    x_train, x_test = design(train), design(test)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-12
    x_train = np.c_[np.ones(len(x_train)), (x_train - mean) / std]
    x_test = np.c_[np.ones(len(x_test)), (x_test - mean) / std]
    y = train["pruned"].to_numpy(float)
    w = np.zeros(x_train.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-x_train @ w))
        grad = x_train.T @ (p - y) + 1e-3 * w
        hess = (x_train * (p * (1 - p))[:, None]).T @ x_train + 1e-3 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return x_test @ w


def gate_curve(score: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    """Separate at the nodes in decreasing score: cumulative shares."""
    order = np.argsort(-score, kind="mergesort")
    cost = frame["cut_loop_seconds"].to_numpy(float)[order]
    pruned = frame["pruned"].to_numpy(float)[order]
    gain = frame["gain"].fillna(0).clip(lower=0).to_numpy(float)[order]
    return pd.DataFrame({
        "time_share": np.cumsum(cost) / cost.sum(),
        "pruned_share": np.cumsum(pruned) / pruned.sum(),
        "gain_share": np.cumsum(gain) / max(gain.sum(), 1e-12),
    })


def time_for(curve: pd.DataFrame, target: float) -> float:
    reached = curve.index[curve["pruned_share"] >= target - 1e-12]
    return float(curve["time_share"].iloc[reached[0]]) if len(reached) else float("nan")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    outcomes, features, gates, curves = [], [], [], []
    rng = np.random.default_rng(0)

    for set_name in SETS:
        for config in CONFIGS:
            nodes = classify(load(set_name, config))
            loops = nodes[nodes["looped"]]
            total_time = nodes["node_seconds"].sum()
            loop_time = loops["cut_loop_seconds"].sum()
            row = {"set": set_name, "config": config,
                   "nodes": len(nodes), "loops": len(loops),
                   "loop_time_share_of_node_time": loop_time / total_time}
            for outcome in ["pruned", "raised", "idle"]:
                part = loops[loops[outcome]]
                row[f"{outcome}_share"] = len(part) / len(loops)
                row[f"{outcome}_time_share"] = part["cut_loop_seconds"].sum() / loop_time
            row["median_relative_gain_raised"] = float(
                (loops.loc[loops["raised"], "gain"] / loops.loc[loops["raised"], "gap"]).median())
            outcomes.append(row)

            label = loops["pruned"].to_numpy(bool)
            for name, (column, sign) in FEATURES.items():
                features.append({"set": set_name, "config": config, "feature": name,
                                 "auc": auc(sign * loops[column].to_numpy(float), label)})

            # Gates scored on the test instances.
            train = loops[loops["instance"].isin(TRAIN)]
            test = loops[loops["instance"].isin(TEST)].reset_index(drop=True)
            columns = [c for c, _ in FEATURES.values()]
            scores = {
                "oracle (pruned nodes first)": test["pruned"].to_numpy(float) + 1e-9 * rng.random(len(test)),
                "random": rng.random(len(test)),
                "smallest gap first": -test["gap"].to_numpy(float),
                "shallowest first": -test["depth"].to_numpy(float) + 1e-9 * rng.random(len(test)),
                "logistic regression (6 features)": logistic(train, test, columns),
            }
            for gate, score in scores.items():
                curve = gate_curve(score, test)
                gates.append({"set": set_name, "config": config, "gate": gate,
                              "auc": auc(score, test["pruned"].to_numpy(bool)),
                              "time_for_50": time_for(curve, 0.5),
                              "time_for_80": time_for(curve, 0.8),
                              "time_for_90": time_for(curve, 0.9)})
                points = np.linspace(0, len(curve) - 1, 100).round().astype(int)
                sampled = curve.iloc[points].assign(set=set_name, config=config, gate=gate)
                curves.append(sampled)

    pd.DataFrame(outcomes).to_csv(OUT / "outcomes.csv", index=False, float_format="%.4f")
    pd.DataFrame(features).to_csv(OUT / "features.csv", index=False, float_format="%.4f")
    pd.DataFrame(gates).to_csv(OUT / "gate.csv", index=False, float_format="%.4f")
    pd.concat(curves).to_csv(OUT / "gate_curves.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(pd.DataFrame(outcomes).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(features).pivot_table(index="feature", columns=["set", "config"], values="auc").round(3).to_string())
    print()
    print(pd.DataFrame(gates).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
