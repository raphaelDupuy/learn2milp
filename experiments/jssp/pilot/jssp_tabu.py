#!/usr/bin/env python3
"""Tabu search for the job-shop (N5 neighbourhood of Nowicki and Smutnicki:
swap the first two or the last two operations of the blocks of a critical
path). Writes the best schedule found as a SCIP solution of the disjunctive
formulation of jssp_pilot.py, to give SCIP a first incumbent found without
the solver.

    jssp_tabu.py --instances DIR --pattern 'j10x10_s*.jsp' \\
        --time-limit 10 --output DIR [--optima DIR]

--optima: a directory of optimal .sol files (from jssp_pilot.py
--write-solutions), only to report the gap and the time to the optimum.
"""

import argparse
import csv
import random
import re
import time
from pathlib import Path

from jssp_pilot import read_instance


class Schedule:
    """Operations o = j * machines + k; machine sequences seq[m]."""

    def __init__(self, instance):
        self.jobs = len(instance)
        self.machines = len(instance[0])
        self.n = self.jobs * self.machines
        self.machine = [m for job in instance for m, _ in job]
        self.duration = [p for job in instance for _, p in job]

    def evaluate(self, seq):
        """Earliest start times. Returns (makespan, heads, critical
        predecessor of each operation), or None if the orders have a
        cycle."""
        n, machines = self.n, self.machines
        machine_prev = [-1] * n
        machine_next = [-1] * n
        for order in seq:
            for a, b in zip(order, order[1:]):
                machine_prev[b] = a
                machine_next[a] = b
        indegree = [0] * n
        for o in range(n):
            indegree[o] = (o % machines != 0) + (machine_prev[o] >= 0)
        ready = [o for o in range(n) if indegree[o] == 0]
        head = [0] * n
        critical = [-1] * n
        duration = self.duration
        visited = 0
        while ready:
            o = ready.pop()
            visited += 1
            end = head[o] + duration[o]
            successors = (o + 1 if o % machines != machines - 1 else -1, machine_next[o])
            for s in successors:
                if s < 0:
                    continue
                if end > head[s]:
                    head[s] = end
                    critical[s] = o
                indegree[s] -= 1
                if indegree[s] == 0:
                    ready.append(s)
        if visited != n:
            return None
        makespan = max(head[o] + duration[o] for o in range(n))
        return makespan, head, critical

    def critical_moves(self, seq, head, critical, makespan):
        """N5 moves on one critical path: (machine, position) pairs, the
        swap of seq[m][i] and seq[m][i + 1]."""
        last = next(o for o in range(self.n) if head[o] + self.duration[o] == makespan)
        path = []
        o = last
        while o >= 0:
            path.append(o)
            o = critical[o]
        path.reverse()
        blocks = []
        for o in path:
            if blocks and self.machine[blocks[-1][-1]] == self.machine[o]:
                blocks[-1].append(o)
            else:
                blocks.append([o])
        position = {}
        for order in seq:
            for i, o in enumerate(order):
                position[o] = i
        moves = set()
        for b, block in enumerate(blocks):
            if len(block) < 2:
                continue
            m = self.machine[block[0]]
            if b > 0:
                moves.add((m, position[block[0]]))
            if b < len(blocks) - 1:
                moves.add((m, position[block[-2]]))
        return sorted(moves)


def initial_sequences(schedule, instance):
    """The greedy dispatching rule of jssp_pilot.py (earliest start, then
    most work remaining), as machine sequences."""
    jobs, machines = schedule.jobs, schedule.machines
    next_op = [0] * jobs
    job_ready = [0] * jobs
    machine_ready = [0] * machines
    remaining = [sum(p for _, p in job) for job in instance]
    seq = [[] for _ in range(machines)]
    for _ in range(jobs * machines):
        best = None
        for j in range(jobs):
            if next_op[j] == machines:
                continue
            m, p = instance[j][next_op[j]]
            key = (max(job_ready[j], machine_ready[m]), -remaining[j])
            if best is None or key < best[0]:
                best = (key, j, m, p)
        (start, _), j, m, p = best
        job_ready[j] = machine_ready[m] = start + p
        remaining[j] -= p
        seq[m].append(j * machines + next_op[j])
        next_op[j] += 1
    return seq


def tabu_search(instance, time_limit, seed, max_stall=2000, optimum=None):
    """Returns (best makespan, best sequences, seconds to the best,
    iterations, seconds to the optimum or None)."""
    rng = random.Random(seed)
    schedule = Schedule(instance)
    seq = initial_sequences(schedule, instance)
    makespan, head, critical = schedule.evaluate(seq)
    best = (makespan, [list(order) for order in seq])
    started = time.perf_counter()
    found_at = 0.0
    optimum_at = 0.0 if optimum is not None and makespan <= optimum else None
    tabu = {}  # (a, b): "a before b" forbidden until that iteration
    iteration = stall = 0
    while time.perf_counter() - started < time_limit:
        iteration += 1
        moves = schedule.critical_moves(seq, head, critical, makespan)
        if not moves:  # one block: the makespan is a machine load, optimal
            break
        chosen = None
        for m, i in moves:
            a, b = seq[m][i], seq[m][i + 1]
            seq[m][i], seq[m][i + 1] = b, a
            result = schedule.evaluate(seq)
            seq[m][i], seq[m][i + 1] = a, b
            if result is None:
                continue
            forbidden = tabu.get((b, a), 0) > iteration
            if forbidden and result[0] >= best[0]:
                continue
            if chosen is None or result[0] < chosen[0][0]:
                chosen = (result, m, i)
        if chosen is None:
            m, i = rng.choice(moves)
            a, b = seq[m][i], seq[m][i + 1]
            seq[m][i], seq[m][i + 1] = b, a
            result = schedule.evaluate(seq)
            if result is None:
                seq[m][i], seq[m][i + 1] = a, b
                continue
        else:
            result, m, i = chosen
            a, b = seq[m][i], seq[m][i + 1]
            seq[m][i], seq[m][i + 1] = b, a
        tabu[(a, b)] = iteration + 8 + rng.randrange(6)
        makespan, head, critical = result
        if makespan < best[0]:
            best = (makespan, [list(order) for order in seq])
            found_at = time.perf_counter() - started
            stall = 0
            if optimum is not None and optimum_at is None and makespan <= optimum:
                optimum_at = found_at
        else:
            stall += 1
        if stall >= max_stall:
            # Restart from the best schedule with a few random critical swaps.
            seq = [list(order) for order in best[1]]
            makespan, head, critical = schedule.evaluate(seq)
            for _ in range(3):
                moves = schedule.critical_moves(seq, head, critical, makespan)
                if not moves:
                    break
                m, i = rng.choice(moves)
                seq[m][i], seq[m][i + 1] = seq[m][i + 1], seq[m][i]
                result = schedule.evaluate(seq)
                if result is None:
                    seq[m][i], seq[m][i + 1] = seq[m][i + 1], seq[m][i]
                    result = schedule.evaluate(seq)
                makespan, head, critical = result
            tabu.clear()
            stall = 0
    return best[0], best[1], found_at, iteration, optimum_at


def write_solution(path, instance, seq, makespan):
    """Values of cmax, s_j_k and y_j_l_m (1 if job j < l comes first on m)
    of the disjunctive formulation of jssp_pilot.py."""
    schedule = Schedule(instance)
    result = schedule.evaluate(seq)
    assert result is not None and result[0] == makespan
    head = result[1]
    machines = schedule.machines
    lines = ["solution status: feasible solution found",
             f"objective value: {makespan}", f"cmax {makespan}"]
    for o in range(schedule.n):
        lines.append(f"s_{o // machines}_{o % machines} {head[o]}")
    for m, order in enumerate(seq):
        rank = {o // machines: i for i, o in enumerate(order)}
        for j in range(schedule.jobs):
            for l in range(j + 1, schedule.jobs):
                lines.append(f"y_{j}_{l}_{m} {int(rank[j] < rank[l])}")
    Path(path).write_text("\n".join(lines) + "\n")


def known_optimum(directory, stem):
    if directory is None:
        return None
    path = Path(directory) / f"{stem}_bounded.sol"
    if not path.exists():
        return None
    match = re.search(r"objective value:\s+(\S+)", path.read_text())
    return round(float(match.group(1))) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--pattern", default="*.jsp")
    parser.add_argument("--time-limit", type=float, default=10.0,
                        help="seconds per instance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--optima", help="directory of optimal .sol files")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.instances).glob(args.pattern)):
        instance = read_instance(path)
        optimum = known_optimum(args.optima, path.stem)
        makespan, seq, found_at, iterations, optimum_at = tabu_search(
            instance, args.time_limit, args.seed, optimum=optimum)
        write_solution(out / f"{path.stem}_bounded.sol", instance, seq, makespan)
        row = {"instance": path.stem, "makespan": makespan, "optimum": optimum,
               "gap_percent": (f"{100 * (makespan - optimum) / optimum:.2f}"
                               if optimum else ""),
               "seconds_to_best": f"{found_at:.2f}",
               "seconds_to_optimum": (f"{optimum_at:.2f}"
                                      if optimum_at is not None else ""),
               "iterations": iterations, "time_limit": args.time_limit}
        rows.append(row)
        print(", ".join(f"{k} {v}" for k, v in row.items()), flush=True)
    with open(out / "tabu.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
