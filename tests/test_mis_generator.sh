
#!/usr/bin/env bash
set -euo pipefail

GENERATOR="$1"
WORK_DIR="$2"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

GRAPH1="$WORK_DIR/first.mis"
GRAPH2="$WORK_DIR/second.mis"

# Generate the same instance twice.
"$GENERATOR" single \
    --n 30 \
    --p 0.20 \
    --seed 123 \
    --output "$GRAPH1"

"$GENERATOR" single \
    --n 30 \
    --p 0.20 \
    --seed 123 \
    --output "$GRAPH2"

# Graphs and metadata must be reproducible.
cmp "$GRAPH1" "$GRAPH2"
cmp "$GRAPH1.meta" "$GRAPH2.meta"

# Check the file headers and metadata.
grep -qx 'MIS_INSTANCE 1' "$GRAPH1"
grep -qx 'n 30' "$GRAPH1"
grep -qx 'MIS_GENERATION_METADATA 1' "$GRAPH1.meta"
grep -qx 'family erdos_renyi' "$GRAPH1.meta"
grep -qx 'seed 123' "$GRAPH1.meta"
grep -qx 'n 30' "$GRAPH1.meta"

# Invalid probability must be rejected.
if "$GENERATOR" single \
    --n 30 \
    --p 1.5 \
    --seed 123 \
    --output "$WORK_DIR/invalid.mis" \
    >"$WORK_DIR/invalid.stdout" \
    2>"$WORK_DIR/invalid.stderr"; then
    echo "ERROR: invalid probability was accepted" >&2
    exit 1
fi

# Unknown options must be rejected.
if "$GENERATOR" single \
    --n 30 \
    --p 0.2 \
    --seed 123 \
    --output "$WORK_DIR/unknown.mis" \
    --unknown value \
    >"$WORK_DIR/unknown.stdout" \
    2>"$WORK_DIR/unknown.stderr"; then
    echo "ERROR: unknown option was accepted" >&2
    exit 1
fi

echo "All MIS generator CLI tests passed."
