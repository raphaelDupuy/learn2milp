"""Barabasi-Albert MIS instances for a one-off generalization pilot.

This script is temporary: it only exists to test, before writing a proper
generator in C++, whether the branching models trained on Erdos-Renyi
graphs (experiments/er125_p01) still work on another graph family. It
writes instances and metadata in the formats of the repository, so that
mis-benchmark reads them like any other instance.

Graph model: the graph starts with m isolated vertices; every new vertex
is joined to m distinct existing vertices, drawn with probability
proportional to their degree (vertices of the initial set are drawn
uniformly for the first new vertex).

Usage, from the repository root:
    python3 experiments/ba200_m8_pilot/generate_pilot_instances.py \
        OUTPUT_DIR N M FIRST_SEED LAST_SEED

The results in this directory were produced with
    python3 experiments/ba200_m8_pilot/generate_pilot_instances.py \
        data/instances/ba200_m8_pilot/test 200 8 1 20
    ./build/mis-benchmark --instances data/instances/ba200_m8_pilot/test \
        --rules relpscost,mislearned \
        --model experiments/er125_p01/models/mlp32_v2.model \
        --seeds 0,1,2 --time-limit 300 \
        --output experiments/ba200_m8_pilot/test_learned_mlp32_v2.csv
and the same command with mlp32_h2.model.
"""

from __future__ import annotations

import pathlib
import random
import sys


def barabasi_albert(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    """Edges (u, v), u < v, sorted, of a Barabasi-Albert graph."""
    rng = random.Random(seed)
    edges = []
    targets = list(range(m))

    # Every vertex appears once per incident edge: drawing from this list
    # is drawing proportionally to the degree.
    repeated: list[int] = []

    for source in range(m, n):
        edges.extend((target, source) for target in targets)
        repeated.extend(targets)
        repeated.extend([source] * m)

        chosen: set[int] = set()

        while len(chosen) < m:
            chosen.add(rng.choice(repeated))

        targets = sorted(chosen)

    return sorted(edges)


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)

    output = pathlib.Path(sys.argv[1])
    n, m, first_seed, last_seed = (int(value) for value in sys.argv[2:])

    output.mkdir(parents=True, exist_ok=True)

    for seed in range(first_seed, last_seed + 1):
        edges = barabasi_albert(n, m, seed)
        name = f"barabasi_albert_n{n}_m{m}_seed{seed}.mis"
        graph = output / name
        metadata = output / (name + ".meta")

        if graph.exists() or metadata.exists():
            raise SystemExit(f"Output file already exists: {graph}")

        graph.write_text(
            f"MIS_INSTANCE 1\nn {n}\nm {len(edges)}\n"
            + "".join(f"{u} {v}\n" for u, v in edges)
        )
        metadata.write_text(
            "INSTANCE_METADATA 1\n"
            'problem_type "MIS"\n'
            'generator_type "barabasi_albert_pilot"\n'
            'generator_version "0.1"\n'
            f"seed {seed}\n"
            f'parameter.m "{m}"\n'
            f'parameter.n "{n}"\n'
        )


if __name__ == "__main__":
    main()
