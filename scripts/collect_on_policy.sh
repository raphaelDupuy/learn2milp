#!/usr/bin/env bash
#
# On-policy samples for a dataset: its training and validation instances
# are solved again, the search following a trained model while full
# strong branching still labels every visited node. The samples describe
# the nodes that the model leads to, which the expert's own search does
# not visit.
#
# Usage, from the repository root:
#   scripts/collect_on_policy.sh DATASET MODEL NAME [BUILD_DIR] [JOBS]
#
#   DATASET  dataset whose instances are in data/instances/DATASET/{train,valid},
#            e.g. er125_p01 or ba200_k8
#   MODEL    model file driving the search
#   NAME     name of the new sample set, e.g. dagger1
#
# Outputs (ignored by git):
#   data/raw/strong_branching/DATASET_NAME/{train,valid}/   samples
#   data/processed/features/DATASET_NAME/{train,valid}/     training tables
#
# Collection and feature computation resume where they stopped (an
# interrupted collector leaves NAME.sb.*.tmp files behind: remove them
# first). A model is then trained on the expert's samples and the new
# ones together, e.g.
#
#   uv run python ml/train_branching_model.py \
#       --train data/processed/features/DATASET/train \
#               data/processed/features/DATASET_NAME/train \
#       --valid data/processed/features/DATASET/valid \
#               data/processed/features/DATASET_NAME/valid \
#       --output MODEL_FILE
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

DATASET="$1"
MODEL="$2"
NAME="$3"
BUILD_DIR="${4:-build}"
JOBS="${5:-4}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCES="$ROOT/data/instances/$DATASET"
SAMPLES="$ROOT/data/raw/strong_branching/${DATASET}_$NAME"
FEATURES="$ROOT/data/processed/features/${DATASET}_$NAME"

if [[ ! -d "$INSTANCES/train" || ! -d "$INSTANCES/valid" ]]; then
    echo "Training or validation instances not found in $INSTANCES" >&2
    exit 2
fi

for split in train valid; do
    "$ROOT/scripts/collect_strong_branching.sh" \
        "$BUILD_DIR/mis-collector" \
        "$INSTANCES/$split" \
        "$SAMPLES/$split" \
        "$JOBS" \
        3600 \
        "$MODEL"
done

for split in train valid; do
    "$ROOT/scripts/featurize_strong_branching.sh" \
        "$BUILD_DIR/mis-featurizer" \
        "$INSTANCES/$split" \
        "$SAMPLES/$split" \
        "$FEATURES/$split" \
        "$JOBS"
done
