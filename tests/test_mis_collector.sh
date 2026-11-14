
#!/usr/bin/env bash
set -euo pipefail

GENERATOR="$1"
COLLECTOR="$2"
WORK_DIR="$3"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

INSTANCE="$WORK_DIR/instance_n100.mis"

# An instance that needs branching with presolving and root cuts.
"$GENERATOR" single \
    --n 100 \
    --p 0.10 \
    --seed 1 \
    --output "$INSTANCE" \
    >/dev/null

# Collect the same instance twice, in two directories.
"$COLLECTOR" \
    --instance "$INSTANCE" \
    --output-dir "$WORK_DIR/first" \
    >"$WORK_DIR/first.stdout"

"$COLLECTOR" \
    --instance "$INSTANCE" \
    --output-dir "$WORK_DIR/second" \
    --time-limit 600 \
    --scip-seed 0 \
    >/dev/null

NODES="$WORK_DIR/first/instance_n100.sb.nodes.csv"
CANDIDATES="$WORK_DIR/first/instance_n100.sb.candidates.csv"
META="$WORK_DIR/first/instance_n100.sb.meta"

test -s "$NODES"
test -s "$CANDIDATES"
test -s "$META"

# No temporary file is left behind.
if compgen -G "$WORK_DIR/first/*.tmp" >/dev/null; then
    echo "ERROR: temporary files were left behind" >&2
    exit 1
fi

# The tables must be reproducible.
cmp "$NODES" "$WORK_DIR/second/instance_n100.sb.nodes.csv"
cmp "$CANDIDATES" "$WORK_DIR/second/instance_n100.sb.candidates.csv"

# Check the table headers and the metadata.
head -n 1 "$NODES" | grep -q '^sample_index,node_id,'
head -n 1 "$NODES" | grep -q ',vertex_states$'
head -n 1 "$CANDIDATES" | grep -q '^sample_index,node_id,candidate_position,vertex_id,'

grep -qx 'MIS_SB_DATASET 1' "$META"
grep -qx 'instance_name "instance_n100.mis"' "$META"
grep -qx 'number_of_vertices 100' "$META"
grep -qx 'branching_rule "misfullstrong"' "$META"
grep -qx 'parameter.presolving/maxrestarts 0' "$META"
grep -qx 'parameter.separating/maxrounds 0' "$META"
grep -qx 'parameter.limits/nodes -1' "$META"
grep -qx 'parameter.randomization/randomseedshift 0' "$META"
grep -qx 'status "OPTIMAL"' "$META"
grep -qx 'complete 1' "$META"

# The metadata counts must match the tables (minus their header row).
SAMPLES=$(($(wc -l <"$NODES") - 1))
ROWS=$(($(wc -l <"$CANDIDATES") - 1))

test "$SAMPLES" -gt 1
grep -qx "number_of_samples $SAMPLES" "$META"
grep -qx "number_of_candidates $ROWS" "$META"
grep -q "samples: $SAMPLES" "$WORK_DIR/first.stdout"

# Every node row holds one state per vertex.
if tail -n +2 "$NODES" | grep -qvE ',[ZOzofu]{100}$'; then
    echo "ERROR: malformed vertex_states column" >&2
    exit 1
fi

# A node limit truncates the dataset and is reported as such.
"$COLLECTOR" \
    --instance "$INSTANCE" \
    --output-dir "$WORK_DIR/limited" \
    --node-limit 5 \
    >"$WORK_DIR/limited.stdout"

grep -qx 'status "NODELIMIT"' "$WORK_DIR/limited/instance_n100.sb.meta"
grep -qx 'complete 0' "$WORK_DIR/limited/instance_n100.sb.meta"
grep -qx 'parameter.limits/nodes 5' "$WORK_DIR/limited/instance_n100.sb.meta"
grep -q 'truncated search tree' "$WORK_DIR/limited.stdout"

# Existing outputs must not be overwritten.
cp "$NODES" "$WORK_DIR/nodes.before"

if "$COLLECTOR" \
    --instance "$INSTANCE" \
    --output-dir "$WORK_DIR/first" \
    >"$WORK_DIR/overwrite.stdout" \
    2>"$WORK_DIR/overwrite.stderr"; then
    echo "ERROR: existing outputs were overwritten" >&2
    exit 1
fi

cmp "$NODES" "$WORK_DIR/nodes.before"

# A missing instance must be rejected without creating the directory.
if "$COLLECTOR" \
    --instance "$WORK_DIR/missing.mis" \
    --output-dir "$WORK_DIR/missing" \
    >"$WORK_DIR/missing.stdout" \
    2>"$WORK_DIR/missing.stderr"; then
    echo "ERROR: missing instance was accepted" >&2
    exit 1
fi

test ! -e "$WORK_DIR/missing"

# Invalid limits must be rejected.
for INVALID in "--time-limit 0" "--node-limit 0" "--node-limit abc" "--scip-seed -1"; do
    # shellcheck disable=SC2086
    if "$COLLECTOR" \
        --instance "$INSTANCE" \
        --output-dir "$WORK_DIR/invalid" \
        $INVALID \
        >"$WORK_DIR/invalid.stdout" \
        2>"$WORK_DIR/invalid.stderr"; then
        echo "ERROR: invalid option was accepted: $INVALID" >&2
        exit 1
    fi
done

test ! -e "$WORK_DIR/invalid"

# Unknown options must be rejected.
if "$COLLECTOR" \
    --instance "$INSTANCE" \
    --output-dir "$WORK_DIR/unknown" \
    --unknown value \
    >"$WORK_DIR/unknown.stdout" \
    2>"$WORK_DIR/unknown.stderr"; then
    echo "ERROR: unknown option was accepted" >&2
    exit 1
fi

echo "All MIS collector CLI tests passed."
