#!/usr/bin/env bash
#
# On-policy samples for the er125_p01 dataset: the training and validation
# instances are solved again, the search following a trained model while
# full strong branching still labels every visited node. The samples
# describe the nodes that the model leads to, which the expert's own
# search (build_dataset.sh) does not visit.
#
# Usage, from the repository root:
#   experiments/er125_p01/collect_on_policy.sh MODEL NAME [BUILD_DIR] [JOBS]
#
#   MODEL  model file driving the search, e.g.
#          experiments/er125_p01/models/mlp32.model
#   NAME   name of the new sample set, e.g. dagger1
#
# Outputs (ignored by git):
#   data/raw/strong_branching/er125_p01_NAME/{train,valid}/   samples
#   data/processed/features/er125_p01_NAME/{train,valid}/     training tables
#
# Collection and feature computation resume where they stopped. A model
# is then trained on the expert's samples and the new ones together:
#
#   uv run python ml/train_branching_model.py \
#       --train data/processed/features/er125_p01/train \
#               data/processed/features/er125_p01_NAME/train \
#       --valid data/processed/features/er125_p01/valid \
#               data/processed/features/er125_p01_NAME/valid \
#       --output experiments/er125_p01/models/mlp32_NAME.model
set -euo pipefail

# Kept for the commands of the README: same as
#   scripts/collect_on_policy.sh er125_p01 MODEL NAME [BUILD_DIR] [JOBS]
if [[ $# -lt 2 || $# -gt 4 ]]; then
    sed -n '3,28p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$ROOT/scripts/collect_on_policy.sh" er125_p01 "$@"
