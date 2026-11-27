"""Covering pilot: is learning to branch in SCIP worth it on minimum
dominating set and set cover?

Writes each instance as a MILP, solves it with the scip binary under a few
branching rules (with or without the optimum given first) and summarizes
nodes, time and root bounds. A run can be stopped and resumed: finished runs
are kept in results.csv and skipped when the same command is run again.

    uv run --project ml python experiments/covering/pilot/covering_pilot.py \\
        generate-setcover --rows 500 --cols 1000 --density 0.05 \\
        --seeds 1-20 --output DIR
    ... run --instances DIR --pattern '*.scp' --rules default fullstrong \\
        --output DIR [--write-solutions DIR] [--oracle DIR]
    ... summarize DIR/results.csv

Instances:
  .mis  a graph (mis-generator); the problem is minimum dominating set:
        min sum x_v  s.t.  x_v + sum of x_u over the neighbours u of v >= 1.
  .scp  set cover, OR-Library format: "rows cols", the cost of every
        column, then for every row the number of columns that cover it and
        their indices (from 1). min c.x  s.t.  every row covered.

Stopping: Ctrl+C or SIGTERM stops the solves in progress (they are run
again on resume); a file named STOP in the output directory lets them
finish and starts no new one.
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np


# --- Instances -------------------------------------------------------------

def generate_setcover(rows, cols, density, seed, max_cost=100):
    """The set cover generator of Gasse et al. (2019), after Balas and Ho
    (1980), with numpy's default generator: every column covers at least two
    rows, every row is covered at least once, costs uniform in
    [1, max_cost]. Returns (costs, rows covered by each column)."""
    rng = np.random.default_rng(seed)
    nonzeros = int(rows * cols * density)
    if nonzeros < rows or nonzeros < 2 * cols:
        raise ValueError("density too low for this size")
    indices = np.empty(nonzeros, dtype=int)
    indices[:2 * cols] = np.arange(2 * cols) % cols
    indices[2 * cols:] = rng.choice(cols, size=nonzeros - 2 * cols, replace=True)
    _, counts = np.unique(indices, return_counts=True)
    # The same array now holds row indices, column after column; its first
    # `rows` entries are a permutation so that every row is covered.
    indices[:rows] = rng.permutation(rows)
    covered, i = [], 0
    for n in counts:
        if i >= rows:
            indices[i:i + n] = rng.choice(rows, size=n, replace=False)
        elif i + n > rows:
            remaining = np.setdiff1d(np.arange(rows), indices[i:rows],
                                     assume_unique=True)
            indices[rows:i + n] = rng.choice(remaining, size=i + n - rows,
                                             replace=False)
        covered.append(sorted(indices[i:i + n].tolist()))
        i += n
    costs = rng.integers(1, max_cost + 1, size=cols).tolist()
    return costs, covered


def write_setcover(path, rows, costs, covered):
    by_row = [[] for _ in range(rows)]
    for j, column in enumerate(covered):
        for i in column:
            by_row[i].append(j + 1)
    lines = [f"{rows} {len(costs)}"]
    lines += [" ".join(map(str, costs[k:k + 20])) for k in range(0, len(costs), 20)]
    for columns in by_row:
        lines.append(str(len(columns)))
        lines += [" ".join(map(str, columns[k:k + 20]))
                  for k in range(0, len(columns), 20)]
    Path(path).write_text("\n".join(lines) + "\n")


def read_setcover(path):
    """Returns (costs, columns covering each row, from 0)."""
    numbers = [int(x) for x in Path(path).read_text().split()]
    rows, cols = numbers[0], numbers[1]
    costs = numbers[2:2 + cols]
    position = 2 + cols
    by_row = []
    for _ in range(rows):
        n = numbers[position]
        by_row.append([j - 1 for j in numbers[position + 1:position + 1 + n]])
        position += 1 + n
    return costs, by_row


def read_graph(path):
    """Returns (number of vertices, edges) of a .mis file."""
    n, edges = None, []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "n":
            n = int(parts[1])
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            edges.append((int(parts[0]), int(parts[1])))
    return n, edges


def covering_problem(path):
    """(problem name, costs, covering rows: lists of variable indices)."""
    if path.suffix == ".mis":
        n, edges = read_graph(path)
        closed = [[v] for v in range(n)]
        for u, v in edges:
            closed[u].append(v)
            closed[v].append(u)
        return "dominating", [1] * n, [sorted(r) for r in closed]
    if path.suffix == ".scp":
        costs, by_row = read_setcover(path)
        return "setcover", costs, by_row
    raise ValueError(f"unknown instance type: {path}")


def wrapped(terms, indent="   "):
    """LP terms, ten to a line."""
    return f"\n{indent}".join(" ".join(terms[k:k + 10])
                              for k in range(0, len(terms), 10))


def write_lp(path, costs, rows):
    objective = [f"+ {c} x{j}" for j, c in enumerate(costs)]
    lines = ["Minimize", " obj: " + wrapped(objective), "Subject To"]
    for i, row in enumerate(rows):
        lines.append(f" c{i}: " + wrapped([f"+ x{j}" for j in row]) + " >= 1")
    lines.append("Binary")
    lines.append(" " + wrapped([f"x{j}" for j in range(len(costs))], " "))
    lines.append("End")
    Path(path).write_text("\n".join(lines) + "\n")


def group_of(stem):
    """The instance name without its seed: the size or distribution."""
    return re.sub(r"_(seed|s)\d+$", "", stem)


# --- Solving ---------------------------------------------------------------

# The settings of the MIS chain (as Gasse et al. 2019): no restart, cuts at
# the root only.
STUDY = ["set presolving maxrestarts 0", "set separating maxrounds 0"]

RULES = {
    # relpscost, SCIP's default rule, with its strong branching start.
    "default": [],
    # Full strong branching, the expert of the MIS chain.
    "fullstrong": ["set branching fullstrong priority 1000000"],
    # Full strong branching without its side effects (no domain reduction
    # from an infeasible child, no node pruned): the tree an imitation of
    # its choices could reach at best.
    "vanillafullstrong": ["set branching vanillafullstrong priority 1000000"],
    # Pseudo-costs without strong branching.
    "pscost": ["set branching pscost priority 1000000"],
}

STATS = {
    "status": r"SCIP Status\s*:\s*(.*)",
    "time": r"Solving Time \(sec\)\s*:\s*([\d.]+)",
    "nodes": r"Solving Nodes\s*:\s*(\d+)",
    "primal": r"Primal Bound\s*:\s*([-+\d.e]+)",
    "dual": r"Dual Bound\s*:\s*([-+\d.e]+)",
    "root_first_lp": r"First LP value\s*:\s*([-+\d.e]+)",
    "root_dual": r"Final Dual Bound\s*:\s*([-+\d.e]+)",
    "lp_iterations": r"^\s*primal LP\s*:.*\n\s*dual LP\s*:\s*[\d.]+\s+\d+\s+(\d+)",
}


class Runner:
    """Runs scip processes and kills them all on request."""

    def __init__(self):
        self.processes = set()
        self.lock = threading.Lock()
        self.stopping = False

    def solve(self, lp_file, rule, time_limit, log_file, start_solution=None,
              write_solution=None):
        """One solve; None if it was stopped."""
        commands = [*STUDY, "set display verblevel 3",
                    f"set limits time {time_limit}", "set limits memory 2500",
                    *RULES[rule], f"read {lp_file}",
                    *([f"read {start_solution}"] if start_solution else []),
                    "optimize",
                    *([f"write solution {write_solution}"] if write_solution else []),
                    "display statistics", "quit"]
        with open(log_file, "w") as log:
            with self.lock:
                if self.stopping:
                    return None
                process = subprocess.Popen(["scip", "-c", " ".join(commands)],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.processes.add(process)
            process.wait()
            with self.lock:
                self.processes.discard(process)
                if self.stopping:
                    return None
        text = Path(log_file).read_text()
        row = {}
        for key, pattern in STATS.items():
            match = re.search(pattern, text, re.MULTILINE)
            row[key] = match.group(1).strip() if match else ""
        return row

    def stop(self):
        with self.lock:
            self.stopping = True
            for process in self.processes:
                process.kill()


def short_status(status):
    for text, short in (("optimal solution found", "optimal"),
                        ("time limit", "time limit"),
                        ("memory limit", "memory limit"),
                        ("infeasible", "infeasible")):
        if text in status:
            return short
    return status or "failed"


# --- Commands --------------------------------------------------------------

def seed_range(text):
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def command_generate_setcover(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for seed in seed_range(args.seeds):
        path = out / f"setcover_r{args.rows}_c{args.cols}_d{args.density:g}_s{seed}.scp"
        if path.exists():
            continue
        costs, covered = generate_setcover(args.rows, args.cols, args.density, seed)
        write_setcover(path, args.rows, costs, covered)
    print(f"instances in {out}")


FIELDS = ["instance", "group", "problem", "rule", "oracle", "variables", "rows",
          *STATS]


def command_run(args):
    out = Path(args.output)
    for sub in ("lp", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    done = set()
    if results.exists():
        with open(results) as handle:
            done = {(r["instance"], r["rule"], r["oracle"])
                    for r in csv.DictReader(handle)}
    oracle = "1" if args.oracle else "0"
    if args.write_solutions:
        Path(args.write_solutions).mkdir(parents=True, exist_ok=True)

    tasks = []
    paths = [p for p in sorted(Path(args.instances).glob(args.pattern))
             if p.suffix in (".mis", ".scp")]
    for path in paths:
        if all((path.stem, rule, oracle) in done for rule in args.rules):
            continue
        problem, costs, rows = covering_problem(path)
        lp_file = out / "lp" / f"{path.stem}.lp"
        if not lp_file.exists():
            write_lp(lp_file, costs, rows)
        info = {"instance": path.stem, "group": group_of(path.stem),
                "problem": problem, "oracle": oracle,
                "variables": len(costs), "rows": len(rows)}
        for rule in args.rules:
            if (path.stem, rule, oracle) not in done:
                tasks.append((dict(info, rule=rule), lp_file))

    progress_log = Path(args.progress_log)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def report(line):
        with lock:
            print(line, flush=True)
            with open(progress_log, "a") as handle:
                handle.write(line + "\n")

    report(f"{out}: {len(tasks)} runs left ({len(done)} already done), "
           f"{args.jobs} at a time, time limit {args.time_limit} s, "
           f"oracle {'on' if args.oracle else 'off'}")
    if not tasks:
        return

    runner = Runner()
    finished = [0]
    stop_file = out / "STOP"

    def work(task):
        info, lp_file = task
        if runner.stopping or stop_file.exists():
            return
        suffix = "_oracle" if args.oracle else ""
        log = out / "logs" / f"{info['instance']}_{info['rule']}{suffix}.log"
        start = None
        if args.oracle:
            start = Path(args.oracle) / f"{info['instance']}_{args.oracle_rule}.sol"
            if not start.exists():
                report(f"no oracle solution {start}, skipped")
                return
        written = (Path(args.write_solutions) / f"{info['instance']}_{info['rule']}.sol"
                   if args.write_solutions else None)
        row = runner.solve(lp_file, info["rule"], args.time_limit, log, start, written)
        if row is None:
            return
        row = dict(info, **row)
        with lock:
            new = not results.exists()
            with open(results, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
            finished[0] += 1
            count = finished[0]
        try:
            seconds, nodes = f"{float(row['time']):.1f}", f"{int(row['nodes']):,}"
        except ValueError:
            seconds, nodes = row["time"], row["nodes"]
        report(f"{time.strftime('%H:%M:%S')}  {count}/{len(tasks)}  "
               f"{info['instance']}  {info['rule']}  {short_status(row['status'])}  "
               f"{seconds} s  {nodes} nodes")

    def interrupted(signum, frame):
        report(f"{time.strftime('%H:%M:%S')}  stopping: solves in progress are "
               "killed and will run again on resume")
        runner.stop()

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    with ThreadPoolExecutor(args.jobs) as pool:
        pending = {pool.submit(work, task) for task in tasks}
        while pending:
            finished_now, pending = wait(pending, timeout=1,
                                         return_when=FIRST_COMPLETED)
            for future in finished_now:
                future.result()
    if runner.stopping or stop_file.exists():
        report(f"{time.strftime('%H:%M:%S')}  stopped after {finished[0]} runs; "
               "run the same command to resume")
        if stop_file.exists():
            stop_file.unlink()
        sys.exit(1)
    report(f"{time.strftime('%H:%M:%S')}  done, results in {results}")


def shifted_geomean(values, shift):
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def command_summarize(args):
    import pandas as pd

    data = pd.read_csv(args.results)
    for column in ("time", "nodes", "primal", "dual", "root_first_lp"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["solved"] = data["status"].str.contains("optimal solution found", na=False)
    data["primal"] = data["primal"].where(data["primal"] < 1e19)
    best = data.groupby("instance")["primal"].transform("min")
    root_dual = pd.to_numeric(data["root_dual"], errors="coerce")
    root_dual = root_dual.fillna(data["dual"].where(data["nodes"] == 1))
    data["root_lp/best"] = data["root_first_lp"] / best
    data["root_dual/best"] = root_dual / best
    # Nodes against relpscost on the same instance, with the same oracle.
    reference = data[data["rule"] == "default"].set_index(
        ["instance", "oracle"])["nodes"]
    data["ratio"] = [n / reference.get((i, o), np.nan)
                     for i, o, n in zip(data["instance"], data["oracle"], data["nodes"])]
    rows = []
    for (group, oracle, rule), part in data.groupby(["group", "oracle", "rule"]):
        rows.append({
            "group": group, "oracle": oracle, "rule": rule,
            "solved": f"{part['solved'].sum()}/{len(part)}",
            "nodes_sgm100": round(shifted_geomean(part["nodes"], 100)),
            "time_sgm1": round(shifted_geomean(part["time"], 1), 1),
            "nodes/default": round(float(np.exp(np.log(part["ratio"]).mean())), 2),
            "root_lp/best": round(part["root_lp/best"].mean(), 3),
            "root_dual/best": round(part["root_dual/best"].mean(), 3),
            "vars": int(part["variables"].mean()),
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-setcover")
    gen.add_argument("--rows", type=int, required=True)
    gen.add_argument("--cols", type=int, required=True)
    gen.add_argument("--density", type=float, default=0.05)
    gen.add_argument("--seeds", required=True, help="e.g. 1-20")
    gen.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--instances", required=True)
    run.add_argument("--pattern", default="*")
    run.add_argument("--rules", nargs="+", default=["default"], choices=list(RULES))
    run.add_argument("--time-limit", type=int, default=600)
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--progress-log", default="data/logs/covering_pilot.log",
                     help="file that scripts/watch_progress.sh can follow")
    run.add_argument("--write-solutions", metavar="DIR",
                     help="write the best solution of every run to DIR")
    run.add_argument("--oracle", metavar="DIR",
                     help="start every run from a solution in DIR (written by "
                          "--write-solutions): the primal oracle")
    run.add_argument("--oracle-rule", default="default",
                     help="the rule whose solutions --oracle reads")
    run.add_argument("--output", required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("results")

    args = parser.parse_args()
    {"generate-setcover": command_generate_setcover, "run": command_run,
     "summarize": command_summarize}[args.command](args)


if __name__ == "__main__":
    main()
