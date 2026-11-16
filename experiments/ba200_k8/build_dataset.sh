#!/usr/bin/env bash
#
# Learning-to-branch dataset on Barabasi-Albert graphs, n = 200, k = 8.
#
#   train     seeds   1..60   strong branching samples collected
#   valid     seeds  61..80   strong branching samples collected
#   test      seeds  81..100  instances only (solver evaluation)
#
# Same splits as experiments/er125_p01, so that models can be trained on
# one family or both and evaluated on the test set of each.
#
# Usage, from the repository root:
#   experiments/ba200_k8/build_dataset.sh [BUILD_DIR] [JOBS]
#
# Outputs (all ignored by git):
#   data/instances/ba200_k8/{train,valid,test}/           instances
#   data/raw/strong_branching/ba200_k8/{train,valid}/     samples
#   data/processed/features/ba200_k8/{train,valid}/       training tables
#
# Instance generation refuses to overwrite an existing split. Collection
# and feature computation resume where they stopped, except that an
# interrupted collector leaves NAME.sb.*.tmp files behind: remove them
# before running the script again.
#
# Cost: about 1.5 hours with 7 jobs, 383 s per training instance on
# average and 19 minutes at most (one instance alone on the machine took
# 150 s), 13 MB of samples per instance. A node costs three times as much
# as on er125_p01 (152 candidates on average instead of 70) but the trees
# are three times smaller.
set -euo pipefail

BUILD_DIR="${1:-build}"
JOBS="${2:-4}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTANCES="$ROOT/data/instances/ba200_k8"
SAMPLES="$ROOT/data/raw/strong_branching/ba200_k8"
FEATURES="$ROOT/data/processed/features/ba200_k8"

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
        --family barabasi_albert \
        --n-values 200 \
        --k-values 8 \
        --seeds "$(seq -s, "$first" "$last")" \
        --output-dir "$INSTANCES/$split" \
        >/dev/null

    echo "Generated instances: $split"
}

generate train 1 60
generate valid 61 80
generate test 81 100

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
