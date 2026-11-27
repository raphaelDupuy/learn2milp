#!/usr/bin/env bash
#
# Minimum dominating set on Erdos-Renyi graphs G(150, 0.067), mean degree
# 10 (the size of the covering pilot, experiments/covering).
#
#   train  seeds   1..60   strong branching samples collected
#   valid  seeds  61..80   strong branching samples collected
#   test   seeds  81..100  instances only (solver evaluation)
#
# Usage, from the repository root:
#   experiments/dom_er150/build_dataset.sh [BUILD_DIR] [JOBS]
#
# Outputs (all ignored by git):
#   data/instances/dom_er150/{train,valid,test}/        instances
#   data/raw/strong_branching/dom_er150/{train,valid}/  samples
#   data/processed/features/dom_er150/{train,valid}/    training tables
#
# Collection and feature computation resume where they stopped.
set -euo pipefail

BUILD_DIR="${1:-build}"
JOBS="${2:-4}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTANCES="$ROOT/data/instances/dom_er150"
SAMPLES="$ROOT/data/raw/strong_branching/dom_er150"
FEATURES="$ROOT/data/processed/features/dom_er150"

GENERATOR="$BUILD_DIR/mis-generator"
COLLECTOR="$BUILD_DIR/mis-collector"
FEATURIZER="$BUILD_DIR/mis-featurizer"

generate() {
    local split="$1" first="$2" last="$3"

    if [[ -e "$INSTANCES/$split/manifest.csv" ]]; then
        echo "Instances already generated: $split"
        return 0
    fi

    "$GENERATOR" batch \
        --n-values 150 \
        --p-values 0.067 \
        --seeds "$(seq -s, "$first" "$last")" \
        --output-dir "$INSTANCES/$split" \
        >/dev/null

    echo "Generated instances: $split"
}

generate train 1 60
generate valid 61 80
generate test 81 100

for split in train valid; do
    PROBLEM=dominating "$ROOT/scripts/collect_strong_branching.sh" \
        "$COLLECTOR" \
        "$INSTANCES/$split" \
        "$SAMPLES/$split" \
        "$JOBS"
done

for split in train valid; do
    "$ROOT/scripts/featurize_strong_branching.sh" \
        "$FEATURIZER" \
        "$INSTANCES/$split" \
        "$SAMPLES/$split" \
        "$FEATURES/$split" \
        "$JOBS"
done
