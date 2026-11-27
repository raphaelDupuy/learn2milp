#!/usr/bin/env bash
#
# Collects strong branching samples for every .mis instance of a directory,
# several instances at a time.
#
# Usage:
#   scripts/collect_strong_branching.sh COLLECTOR INSTANCE_DIR OUTPUT_DIR [JOBS] [TIME_LIMIT] [POLICY_MODEL]
#
#   COLLECTOR     path of the mis-collector executable
#   INSTANCE_DIR  directory containing the .mis instances (and their .meta)
#   OUTPUT_DIR    directory receiving the .sb.* files
#   JOBS          instances collected in parallel (default 4)
#   TIME_LIMIT    solving time limit per instance, in seconds (default 3600)
#   POLICY_MODEL  branching model driving the search instead of the expert
#                 (default: none, the search follows the expert)
#
# The environment variable PROBLEM chooses the problem posed on each graph:
# mis (default) or dominating.
#
# The script can be interrupted and restarted: an instance whose .sb.meta
# file already exists is skipped. The collected tables do not depend on
# JOBS, but a run stopped by the time limit depends on the machine load.
# One log file per instance is written to OUTPUT_DIR/logs.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

COLLECTOR="$1"
INSTANCE_DIR="$2"
OUTPUT_DIR="$3"
JOBS="${4:-4}"
TIME_LIMIT="${5:-3600}"
POLICY_MODEL="${6:-}"
PROBLEM="${PROBLEM:-mis}"

if [[ ! -x "$COLLECTOR" ]]; then
    echo "Collector not found or not executable: $COLLECTOR" >&2
    exit 2
fi

if [[ -n "$POLICY_MODEL" && ! -f "$POLICY_MODEL" ]]; then
    echo "Policy model not found: $POLICY_MODEL" >&2
    exit 2
fi

if [[ ! -d "$INSTANCE_DIR" ]]; then
    echo "Instance directory not found: $INSTANCE_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"

collect_one() {
    local instance="$1"
    local name
    name="$(basename "$instance" .mis)"

    if [[ -e "$OUTPUT_DIR/$name.sb.meta" ]]; then
        echo "skip  $name"
        return 0
    fi

    local policy=()

    if [[ -n "$POLICY_MODEL" ]]; then
        policy=(--policy-model "$POLICY_MODEL")
    fi

    if "$COLLECTOR" \
        --instance "$instance" \
        --output-dir "$OUTPUT_DIR" \
        --time-limit "$TIME_LIMIT" \
        --problem "$PROBLEM" \
        "${policy[@]}" \
        >"$OUTPUT_DIR/logs/$name.log" 2>&1; then
        echo "done  $name"
    else
        echo "FAIL  $name (see $OUTPUT_DIR/logs/$name.log)"
        return 1
    fi
}

export -f collect_one
export COLLECTOR OUTPUT_DIR TIME_LIMIT POLICY_MODEL

FAILED=0

find "$INSTANCE_DIR" -maxdepth 1 -name '*.mis' -print0 |
    sort -z |
    xargs -0 -P "$JOBS" -I {} bash -c 'collect_one "$1"' _ {} ||
    FAILED=1

TOTAL="$(find "$INSTANCE_DIR" -maxdepth 1 -name '*.mis' | wc -l)"
COLLECTED="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.sb.meta' | wc -l)"

echo "Collected $COLLECTED of $TOTAL instances into $OUTPUT_DIR"

if [[ "$FAILED" -ne 0 || "$COLLECTED" -ne "$TOTAL" ]]; then
    exit 1
fi
