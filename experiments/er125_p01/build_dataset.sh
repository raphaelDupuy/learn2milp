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
# Outputs (both ignored by git):
#   data/instances/er125_p01/{train,valid,test,transfer}/
#   data/raw/strong_branching/er125_p01/{train,valid}/
#
# Instance generation refuses to overwrite an existing split: remove the
# directories above to rebuild from scratch. Collection resumes where it
# stopped.
set -euo pipefail

BUILD_DIR="${1:-build}"
JOBS="${2:-4}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTANCES="$ROOT/data/instances/er125_p01"
SAMPLES="$ROOT/data/raw/strong_branching/er125_p01"

GENERATOR="$BUILD_DIR/mis-generator"
COLLECTOR="$BUILD_DIR/mis-collector"

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
