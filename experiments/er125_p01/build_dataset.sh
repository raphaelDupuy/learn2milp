#!/usr/bin/env bash
#
# First learning-to-branch dataset: Erdos-Renyi graphs G(125, 0.1).
#
#   train     seeds   1..60   strong branching samples collected
#   valid     seeds  61..80   strong branching samples collected
#   test      seeds  81..100  instances only (solver evaluation)
#   transfer  G(150, 0.05), seeds 1..20, instances only (generalization)
#
# Usage, from the repository root:
#   experiments/er125_p01/build_dataset.sh [BUILD_DIR] [JOBS]
#
# Outputs (all ignored by git):
#   data/instances/er125_p01/{train,valid,test,transfer}/   instances
#   data/raw/strong_branching/er125_p01/{train,valid}/      samples
#   data/processed/features/er125_p01/{train,valid}/        training tables
#
# Instance generation refuses to overwrite an existing split: remove the
# directories above to rebuild from scratch. Collection and feature
# computation resume where they stopped. After a change of the feature
# set, remove data/processed/features/er125_p01 and run the script again:
# the samples do not need to be collected again.
set -euo pipefail

BUILD_DIR="${1:-build}"
JOBS="${2:-4}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTANCES="$ROOT/data/instances/er125_p01"
SAMPLES="$ROOT/data/raw/strong_branching/er125_p01"
FEATURES="$ROOT/data/processed/features/er125_p01"

GENERATOR="$BUILD_DIR/mis-generator"
COLLECTOR="$BUILD_DIR/mis-collector"
FEATURIZER="$BUILD_DIR/mis-featurizer"

seeds() {
    seq -s, "$1" "$2"
}

generate() {
    local split="$1" n="$2" p="$3" first="$4" last="$5"

    if [[ -e "$INSTANCES/$split/manifest.csv" ]]; then
        echo "Instances already generated: $split"
        return 0
    fi

    "$GENERATOR" batch \
        --n-values "$n" \
        --p-values "$p" \
        --seeds "$(seeds "$first" "$last")" \
        --output-dir "$INSTANCES/$split" \
        >/dev/null

    echo "Generated instances: $split"
}

generate train 125 0.1 1 60
generate valid 125 0.1 61 80
generate test 125 0.1 81 100
generate transfer 150 0.05 1 20

for split in train valid; do
    "$ROOT/scripts/collect_strong_branching.sh" \
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
