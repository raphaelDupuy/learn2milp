"""JSSP pilot: is learning to branch in SCIP worth it on the job shop?

Generates Taillard-style instances (explicit seeds), writes them as MILPs in
several formulations, solves them with the scip binary under a few branching
rules and summarizes nodes, time and root bounds.

    uv run python experiments/jssp/pilot/jssp_pilot.py generate ...
    uv run python experiments/jssp/pilot/jssp_pilot.py run ...
    uv run python experiments/jssp/pilot/jssp_pilot.py summarize ...

Instances use the OR-Library text format: "n m", then one line per job of
m pairs "machine duration" in processing order (machines from 0).
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


# --- Instances -------------------------------------------------------------

def generate_taillard_like(jobs, machines, seed, low=1, high=99):
    """Each job visits every machine once, in a random order; durations are
    uniform integers in [low, high] (Taillard 1993, with numpy's generator
    instead of Taillard's own)."""
    rng = np.random.default_rng(seed)
    routes = [rng.permutation(machines).tolist() for _ in range(jobs)]
    durations = rng.integers(low, high + 1, size=(jobs, machines)).tolist()
    return [list(zip(routes[j], durations[j])) for j in range(jobs)]


def write_instance(path, instance):
    machines = len(instance[0])
    with open(path, "w") as out:
        out.write(f"{len(instance)} {machines}\n")
        for job in instance:
            out.write(" ".join(f"{m} {p}" for m, p in job) + "\n")


def read_instance(path):
    numbers = [int(x) for x in Path(path).read_text().split()]
    jobs, machines = numbers[0], numbers[1]
    values = numbers[2:]
    return [[(values[2 * (j * machines + k)], values[2 * (j * machines + k) + 1])
             for k in range(machines)] for j in range(jobs)]


def dispatch_makespan(instance):
    """Makespan of a greedy schedule: repeatedly start the next operation
    that can start earliest, ties broken by the most work remaining."""
    jobs = len(instance)
    machines = len(instance[0])
    next_op = [0] * jobs
    job_ready = [0] * jobs
    machine_ready = [0] * machines
    remaining = [sum(p for _, p in job) for job in instance]
    for _ in range(jobs * machines):
        best = None
        for j in range(jobs):
            if next_op[j] == machines:
                continue
            m, p = instance[j][next_op[j]]
            start = max(job_ready[j], machine_ready[m])
            key = (start, -remaining[j])
            if best is None or key < best[0]:
                best = (key, j, m, p, start)
        _, j, m, p, start = best
        job_ready[j] = machine_ready[m] = start + p
        remaining[j] -= p
        next_op[j] += 1
    return max(job_ready)


def trivial_lower_bound(instance):
    machines = len(instance[0])
    job_length = max(sum(p for _, p in job) for job in instance)
    load = [0] * machines
    for job in instance:
        for m, p in job:
            load[m] += p
    return max(job_length, max(load))


# --- Formulations ----------------------------------------------------------

def heads_and_tails(instance):
    """head[j][k]: work of job j before operation k; tail[j][k]: work from
    operation k (included) to the end of job j."""
    heads, tails = [], []
    for job in instance:
        durations = [p for _, p in job]
        heads.append([sum(durations[:k]) for k in range(len(job))])
        tails.append([sum(durations[k:]) for k in range(len(job))])
    return heads, tails


def operations_by_machine(instance):
    machines = len(instance[0])
    on = [[] for _ in range(machines)]
    for j, job in enumerate(instance):
        for k, (m, p) in enumerate(job):
            on[m].append((j, k, p))
    return on


def machine_bound(instance):
    """One-machine bound: on each machine, the smallest head, the load and
    the smallest tail after the operation."""
    heads, tails = heads_and_tails(instance)
    bound = 0
    for ops in operations_by_machine(instance):
        head = min(heads[j][k] for j, k, _ in ops)
        tail = min(tails[j][k] - p for j, k, p in ops)
        bound = max(bound, head + sum(p for _, _, p in ops) + tail)
    return max(bound, max(sum(p for _, p in job) for job in instance))


def write_disjunctive(path, instance, tight, bounded=False, integer=False):
    """Manne's disjunctive formulation. y_j_l_m = 1 if job j precedes job l
    on machine m (j < l). Naive: big-M = sum of all durations, no bounds.
    Tight: big-M and bounds from the makespan H of a dispatching rule,
    head <= s <= H - tail. Bounded: also cmax >= the one-machine bound,
    which the LP of the big-M formulation does not see. Integer: start
    times and cmax are integer variables (the durations are integers)."""
    heads, tails = heads_and_tails(instance)
    total = sum(p for job in instance for _, p in job)
    horizon = dispatch_makespan(instance) if tight else total
    big_m = horizon
    rows, binaries = [], []
    for j, job in enumerate(instance):
        for k in range(len(job) - 1):
            rows.append(f" prec_{j}_{k}: s_{j}_{k + 1} - s_{j}_{k} >= {job[k][1]}")
        last = len(job) - 1
        rows.append(f" end_{j}: cmax - s_{j}_{last} >= {job[last][1]}")
    for m, ops in enumerate(operations_by_machine(instance)):
        for (j, k, p), (l, h, q) in itertools.combinations(ops, 2):
            y = f"y_{j}_{l}_{m}"
            binaries.append(y)
            # y = 1: s_l >= s_j + p ; y = 0: s_j >= s_l + q.
            rows.append(f" a_{j}_{l}_{m}: s_{l}_{h} - s_{j}_{k} - {big_m} {y} >= {p - big_m}")
            rows.append(f" b_{j}_{l}_{m}: s_{j}_{k} - s_{l}_{h} + {big_m} {y} >= {q}")
    bounds = []
    for j, job in enumerate(instance):
        for k in range(len(job)):
            if tight:
                bounds.append(f" {heads[j][k]} <= s_{j}_{k} <= {horizon - tails[j][k]}")
            else:
                bounds.append(f" s_{j}_{k} >= 0")
    lower = machine_bound(instance) if bounded else 0
    bounds.append(f" {lower} <= cmax <= {horizon}" if tight else " cmax >= 0")
    with open(path, "w") as out:
        out.write("Minimize\n obj: cmax\nSubject To\n")
        out.write("\n".join(rows) + "\nBounds\n" + "\n".join(bounds))
        if integer:
            starts = [f"s_{j}_{k}" for j, job in enumerate(instance)
                      for k in range(len(job))]
            out.write("\nGeneral\n" + "\n".join(f" {v}" for v in starts + ["cmax"]))
        out.write("\nBinary\n" + "\n".join(f" {b}" for b in binaries) + "\nEnd\n")
    return len(binaries)


def write_with_machines(path, instance):
    """The bounded formulation with integer start times, plus one
    cumulative constraint of capacity 1 per machine: SCIP then reasons on
    the time windows of all the operations of a machine together (time
    tabling, edge finding), not only pair by pair through the big-M rows.
    The LP is unchanged. Written in SCIP's CIP format, the LP file being
    converted by the scip binary."""
    path = Path(path)
    base = path.with_suffix(".base.lp")
    binaries = write_disjunctive(base, instance, tight=True, bounded=True,
                                 integer=True)
    subprocess.run(["scip", "-c", f"read {base} write problem {path} quit"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    text = path.read_text()
    machines = []
    for m, ops in enumerate(operations_by_machine(instance)):
        jobs = ", ".join(f"<s_{j}_{k}>({p})[1]" for j, k, p in ops)
        machines.append(f"  [cumulative] <machine_{m}>: cumulative({jobs})"
                        f"[0,{dispatch_makespan(instance)}) <= 1;")
    end = text.rindex("END")
    path.write_text(text[:end] + "\n".join(machines) + "\n" + text[end:])
    base.unlink()
    return binaries


def write_time_indexed(path, instance):
    """Time-indexed formulation on the horizon H of a dispatching rule:
    x_j_k_t = 1 if operation k of job j starts at t."""
    heads, tails = heads_and_tails(instance)
    horizon = dispatch_makespan(instance)
    starts = {}
    for j, job in enumerate(instance):
        for k in range(len(job)):
            starts[j, k] = range(heads[j][k], horizon - tails[j][k] + 1)

    def start_time(j, k):
        return " + ".join(f"{t} x_{j}_{k}_{t}" for t in starts[j, k] if t > 0)

    def linear(terms):
        text = " ".join(terms)
        return text[2:] if text.startswith("+ ") else text

    rows, binaries = [], []
    for j, job in enumerate(instance):
        for k in range(len(job)):
            names = [f"x_{j}_{k}_{t}" for t in starts[j, k]]
            binaries.extend(names)
            rows.append(f" once_{j}_{k}: " + " + ".join(names) + " = 1")
        for k in range(len(job) - 1):
            terms = [f"+ {t} x_{j}_{k + 1}_{t}" for t in starts[j, k + 1] if t > 0]
            terms += [f"- {t} x_{j}_{k}_{t}" for t in starts[j, k] if t > 0]
            rows.append(f" prec_{j}_{k}: {linear(terms)} >= {job[k][1]}")
        last = len(job) - 1
        terms = ["+ cmax"] + [f"- {t} x_{j}_{last}_{t}" for t in starts[j, last] if t > 0]
        rows.append(f" end_{j}: {linear(terms)} >= {job[last][1]}")
    for m, ops in enumerate(operations_by_machine(instance)):
        for t in range(horizon):
            terms = []
            for j, k, p in ops:
                for u in range(t - p + 1, t + 1):
                    if u in starts[j, k]:
                        terms.append(f"x_{j}_{k}_{u}")
            if len(terms) > 1:
                rows.append(f" cap_{m}_{t}: " + " + ".join(terms) + " <= 1")
    with open(path, "w") as out:
        out.write("Minimize\n obj: cmax\nSubject To\n")
        out.write("\n".join(rows) + f"\nBounds\n cmax <= {horizon}\n")
        out.write("Binary\n" + "\n".join(f" {b}" for b in binaries) + "\nEnd\n")
    return len(binaries)


FORMULATIONS = {
    "naive": lambda path, inst: write_disjunctive(path, inst, tight=False),
    "tight": lambda path, inst: write_disjunctive(path, inst, tight=True),
    "bounded": lambda path, inst: write_disjunctive(path, inst, tight=True,
                                                    bounded=True),
    "timeidx": write_time_indexed,
    "boundedint": lambda path, inst: write_disjunctive(path, inst, tight=True,
                                                       bounded=True, integer=True),
    "disjunctive": write_with_machines,
    "disjunctive_ef": write_with_machines,
}

# File format of each formulation (LP by default).
FORMULATION_FORMAT = {"disjunctive": "cip", "disjunctive_ef": "cip"}

# SCIP settings that go with a formulation. With integer start times,
# branching keeps to the order variables. The cumulative constraints only
# propagate: no cuts, and no extra disjunctive constraints from presolving.
_MACHINES = ["set branching preferbinary TRUE",
             "set constraints cumulative sepafreq -1",
             "set constraints cumulative disjunctive FALSE"]
FORMULATION_SETTINGS = {
    "boundedint": ["set branching preferbinary TRUE"],
    "disjunctive": _MACHINES,
    "disjunctive_ef": _MACHINES + ["set constraints cumulative efcheck TRUE",
                                   "set constraints cumulative efinfer TRUE"],
}


# --- Solving ---------------------------------------------------------------

RULES = {
    # SCIP as shipped: relpscost with its strong branching initialization.
    "default": [],
    # Full strong branching, the expert of the MIS chain.
    "fullstrong": ["set branching fullstrong priority 1000000"],
    # Full strong branching without its side effects: no domain reduction
    # from an infeasible child, no node pruned; the tree an imitation of
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


def solve(lp_file, rule, seed, time_limit, log_file, restarts=True,
          start_solution=None, write_solution=None, settings=()):
    """Solves one MILP with the scip binary. start_solution: a .sol file
    read before solving, the first incumbent (the primal oracle);
    write_solution: where to write the best solution found; settings:
    more scip commands, run before reading the problem."""
    commands = [*([] if restarts else ["set presolving maxrestarts 0"]),
                *settings,
                "set display verblevel 3", f"set limits time {time_limit}",
                f"set randomization randomseedshift {seed}",
                "set limits memory 2500", *RULES[rule], f"read {lp_file}",
                *([f"read {start_solution}"] if start_solution else []),
                "optimize",
                *([f"write solution {write_solution}"] if write_solution else []),
                "display statistics", "quit"]
    with open(log_file, "w") as log:
        subprocess.run(["scip", "-c", " ".join(commands)], stdout=log,
                       stderr=subprocess.STDOUT, check=False)
    text = Path(log_file).read_text()
    row = {}
    for key, pattern in STATS.items():
        match = re.search(pattern, text, re.MULTILINE)
        row[key] = match.group(1).strip() if match else ""
    return row


# --- Progress ---------------------------------------------------------------

def short_status(status):
    """SCIP's status in a word or two."""
    for text, short in (("optimal solution found", "optimal"),
                        ("time limit", "time limit"),
                        ("memory limit", "memory limit"),
                        ("infeasible", "infeasible")):
        if text in status:
            return short
    return status or "failed"


class ProgressTable:
    """Aligned lines for the progress of a run: the widths fit the names of
    the instances, formulations and rules of the run."""

    def __init__(self, total, instances, formulations, rules):
        self.total = total
        self.done = 0
        self.columns = [  # title, width, right-aligned
            ("time", 8, False),
            ("run", 2 * len(str(total)) + 1, True),
            ("instance", max(map(len, instances), default=0), False),
            ("formulation", max(map(len, formulations), default=0), False),
            ("rule", max(map(len, rules), default=0), False),
            ("status", len("memory limit"), False),
            ("seconds", 8, True),
            ("nodes", 11, True),
        ]
        self.columns = [(title, max(width, len(title)), right)
                        for title, width, right in self.columns]

    def _line(self, values):
        cells = [value.rjust(width) if right else value.ljust(width)
                 for value, (_, width, right) in zip(values, self.columns)]
        return "  ".join(cells).rstrip()

    def header(self):
        return [self._line([title for title, _, _ in self.columns]),
                self._line(["-" * width for _, width, _ in self.columns])]

    def row(self, instance, formulation, rule, status, seconds, nodes):
        self.done += 1
        try:
            seconds = f"{float(seconds):.1f}"
            nodes = f"{int(nodes):,}"
        except ValueError:
            pass
        return self._line([time.strftime("%H:%M:%S"),
                           f"{self.done}/{self.total}", instance, formulation,
                           rule, short_status(status), str(seconds), str(nodes)])


# --- Commands --------------------------------------------------------------

def command_generate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        jobs, machines = map(int, size.split("x"))
        for seed in range(args.first_seed, args.first_seed + args.count):
            instance = generate_taillard_like(jobs, machines, seed)
            write_instance(out / f"j{jobs}x{machines}_s{seed}.jsp", instance)
    print(f"instances in {out}")


def command_run(args):
    out = Path(args.output)
    (out / "lp").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    instances = sorted(Path(args.instances).glob(args.pattern))
    tasks = []
    for path in instances:
        instance = read_instance(path)
        info = {"instance": path.stem, "jobs": len(instance),
                "machines": len(instance[0]),
                "trivial_lb": trivial_lower_bound(instance),
                "dispatch": dispatch_makespan(instance)}
        for formulation in args.formulations:
            extension = FORMULATION_FORMAT.get(formulation, "lp")
            lp_file = out / "lp" / f"{path.stem}_{formulation}.{extension}"
            binaries = FORMULATIONS[formulation](lp_file, instance)
            for rule in args.rules:
                log = out / "logs" / f"{path.stem}_{formulation}_{rule}.log"
                tasks.append((dict(info, formulation=formulation, rule=rule,
                                   binaries=binaries), lp_file, log))

    # One line per finished run, on the terminal and in the progress log
    # that scripts/watch_progress.sh follows.
    progress_log = Path(args.progress_log)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    table = ProgressTable(len(tasks),
                          [t[0]["instance"] for t in tasks],
                          [t[0]["formulation"] for t in tasks],
                          [t[0]["rule"] for t in tasks])
    lock = threading.Lock()

    def report(lines):
        with lock:
            with open(progress_log, "a") as handle:
                for line in lines:
                    print(line, flush=True)
                    handle.write(line + "\n")

    report([f"{out}: {len(tasks)} runs, {args.jobs} at a time, "
            f"time limit {args.time_limit} s", "", *table.header()])

    if args.write_solutions:
        Path(args.write_solutions).mkdir(parents=True, exist_ok=True)

    def solution_file(directory, info, formulation=None):
        formulation = formulation or info["formulation"]
        return Path(directory) / f"{info['instance']}_{formulation}.sol"

    def work(task):
        info, lp_file, log = task
        start = (solution_file(args.oracle, info, args.oracle_formulation)
                 if args.oracle else None)
        if start is not None and not start.exists():
            raise FileNotFoundError(f"no oracle solution {start}")
        written = (solution_file(args.write_solutions, info)
                   if args.write_solutions else None)
        row = dict(info, **solve(lp_file, info["rule"], 0, args.time_limit, log,
                                     restarts=not args.no_restarts,
                                     start_solution=start,
                                     write_solution=written,
                                     settings=FORMULATION_SETTINGS.get(
                                         info["formulation"], [])))
        row["oracle"] = int(start is not None)
        with lock:
            line = table.row(info["instance"], info["formulation"], info["rule"],
                             row["status"], row["time"], row["nodes"])
        report([line])
        return row

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(work, tasks))
    results = out / "results.csv"
    with open(results, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"results in {results}")


def shifted_geomean(values, shift):
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def command_summarize(args):
    import pandas as pd

    data = pd.read_csv(args.results)
    data["solved"] = data["status"].str.contains("optimal solution found", na=False)
    # SCIP prints 1e+20 as the primal bound when it found no solution.
    data["primal"] = data["primal"].where(data["primal"] < 1e19)
    data["size"] = data["jobs"].astype(str) + "x" + data["machines"].astype(str)
    # Root gap closed: share of the gap between the first LP value and the
    # best primal bound that the root closes; and the first LP value against
    # the trivial bound max(job length, machine load).
    best = data.groupby("instance")["primal"].transform("min")
    data["root_lp_over_best"] = data["root_first_lp"] / best
    data["trivial_over_best"] = data["trivial_lb"] / best
    # SCIP prints no root dual bound when the root solves the problem.
    root_dual = pd.to_numeric(data["root_dual"], errors="coerce")
    root_dual = root_dual.fillna(data["dual"].where(data["nodes"] == 1))
    data["root_dual_over_best"] = root_dual / best
    data["dispatch_over_best"] = data["dispatch"] / best
    rows = []
    for (size, formulation, rule), group in data.groupby(
            ["size", "formulation", "rule"], sort=False):
        rows.append({
            "size": size, "formulation": formulation, "rule": rule,
            "solved": f"{group['solved'].sum()}/{len(group)}",
            "nodes_sgm100": round(shifted_geomean(group["nodes"], 100)),
            "time_sgm1": round(shifted_geomean(group["time"], 1), 1),
            "root_lp/best": round(group["root_lp_over_best"].mean(), 3),
            "root_dual/best": round(group["root_dual_over_best"].mean(), 3),
            "trivial_lb/best": round(group["trivial_over_best"].mean(), 3),
            "dispatch/best": round(group["dispatch_over_best"].mean(), 3),
            "binaries": int(group["binaries"].mean()),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--sizes", nargs="+", required=True, help="e.g. 6x6 10x5")
    gen.add_argument("--first-seed", type=int, default=1)
    gen.add_argument("--count", type=int, default=5)
    gen.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--instances", required=True)
    run.add_argument("--pattern", default="*.jsp")
    run.add_argument("--formulations", nargs="+", default=["tight"],
                     choices=list(FORMULATIONS))
    run.add_argument("--rules", nargs="+", default=["default"],
                     choices=list(RULES))
    run.add_argument("--time-limit", type=int, default=600)
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--progress-log", default="data/logs/jssp_pilot.log",
                     help="file that scripts/watch_progress.sh can follow")
    run.add_argument("--write-solutions", metavar="DIR",
                     help="write the best solution of every run to DIR")
    run.add_argument("--oracle", metavar="DIR",
                     help="start every run from the solution in DIR "
                          "(written by --write-solutions): the primal oracle")
    run.add_argument("--oracle-formulation", metavar="NAME",
                     help="read the oracle solutions written for this "
                          "formulation (the variables are the same in all "
                          "the disjunctive ones)")
    run.add_argument("--no-restarts", action="store_true",
                     help="set presolving/maxrestarts to 0")
    run.add_argument("--output", required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("results")

    args = parser.parse_args()
    {"generate": command_generate, "run": command_run,
     "summarize": command_summarize}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
