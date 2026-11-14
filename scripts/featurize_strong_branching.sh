#!/usr/bin/env bash
#
# Computes the feature table of every collected instance of a directory,
# several instances at a time.
#
# Usage:
#   scripts/featurize_strong_branching.sh FEATURIZER INSTANCE_DIR SAMPLES_DIR OUTPUT_DIR [JOBS]
#
#   FEATURIZER    path of the mis-featurizer executable
#   INSTANCE_DIR  directory containing the .mis instances
#   SAMPLES_DIR   directory containing the collected .sb.* files
#   OUTPUT_DIR    directory receiving the .sb.features.csv files
#   JOBS          instances processed in parallel (default 4)
#
# An instance whose feature table already exists is skipped: remove
# OUTPUT_DIR to recompute everything, for example after a change of the
# feature set.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

FEATURIZER="$1"
INSTANCE_DIR="$2"
SAMPLES_DIR="$3"
OUTPUT_DIR="$4"
JOBS="${5:-4}"

if [[ ! -x "$FEATURIZER" ]]; then
    echo "Featurizer not found or not executable: $FEATURIZER" >&2
    exit 2
fi

if [[ ! -d "$INSTANCE_DIR" || ! -d "$SAMPLES_DIR" ]]; then
    echo "Instance or samples directory not found." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"

featurize_one() {
    local instance="$1"
    local name
    name="$(basename "$instance" .mis)"

    if [[ -e "$OUTPUT_DIR/$name.sb.features.csv" ]]; then
        echo "skip  $name"
        return 0
    fi

    if "$FEATURIZER" \
        --instance "$instance" \
        --samples-dir "$SAMPLES_DIR" \
        --output-dir "$OUTPUT_DIR" \
        >"$OUTPUT_DIR/logs/$name.log" 2>&1; then
        echo "done  $name"
    else
        echo "FAIL  $name (see $OUTPUT_DIR/logs/$name.log)"
        return 1
    fi
}

export -f featurize_one
export FEATURIZER SAMPLES_DIR OUTPUT_DIR

FAILED=0

find "$INSTANCE_DIR" -maxdepth 1 -name '*.mis' -print0 |
    sort -z |
    xargs -0 -P "$JOBS" -I {} bash -c 'featurize_one "$1"' _ {} ||
    FAILED=1

TOTAL="$(find "$INSTANCE_DIR" -maxdepth 1 -name '*.mis' | wc -l)"
DONE="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.sb.features.csv' | wc -l)"

echo "Feature tables: $DONE of $TOTAL instances in $OUTPUT_DIR"

if [[ "$FAILED" -ne 0 || "$DONE" -ne "$TOTAL" ]]; then
    exit 1
fi
