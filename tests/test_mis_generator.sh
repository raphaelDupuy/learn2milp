
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

grep -qx 'INSTANCE_METADATA 1' "$GRAPH1.meta"
grep -qx 'problem_type "MIS"' "$GRAPH1.meta"
grep -qx 'generator_type "erdos_renyi"' "$GRAPH1.meta"
grep -qx 'generator_version "1.0"' "$GRAPH1.meta"
grep -qx 'seed 123' "$GRAPH1.meta"
grep -qx 'parameter.n "30"' "$GRAPH1.meta"
grep -q '^parameter.p "' "$GRAPH1.meta"

# The family can be named explicitly; erdos_renyi is the default.
"$GENERATOR" single \
    --family erdos_renyi \
    --n 30 \
    --p 0.20 \
    --seed 123 \
    --output "$WORK_DIR/explicit.mis" \
    >/dev/null

cmp "$GRAPH1" "$WORK_DIR/explicit.mis"

# Barabasi-Albert instances: reproducible, with k (n - k) edges.
BA1="$WORK_DIR/ba_first.mis"
BA2="$WORK_DIR/ba_second.mis"

for output in "$BA1" "$BA2"; do
    "$GENERATOR" single \
        --family barabasi_albert \
        --n 40 \
        --k 3 \
        --seed 5 \
        --output "$output" \
        >"$WORK_DIR/ba.stdout"
done

cmp "$BA1" "$BA2"
cmp "$BA1.meta" "$BA2.meta"

grep -qx 'n 40' "$BA1"
grep -qx 'm 111' "$BA1"
grep -qx 'generator_type "barabasi_albert"' "$BA1.meta"
grep -qx 'generator_version "1.0"' "$BA1.meta"
grep -qx 'seed 5' "$BA1.meta"
grep -qx 'parameter.k "3"' "$BA1.meta"
grep -qx 'parameter.n "40"' "$BA1.meta"
grep -q 'edges: 111' "$WORK_DIR/ba.stdout"

# An existing instance is never overwritten.
cp "$GRAPH1" "$WORK_DIR/before.mis"

if "$GENERATOR" single \
    --n 10 \
    --p 0.5 \
    --seed 1 \
    --output "$GRAPH1" \
    >"$WORK_DIR/overwrite.stdout" \
    2>"$WORK_DIR/overwrite.stderr"; then
    echo "ERROR: existing instance was overwritten" >&2
    exit 1
fi

cmp "$GRAPH1" "$WORK_DIR/before.mis"

# Arguments that do not fit the family are usage errors (exit code 2),
# and no file is written.
expect_usage_error() {
    local description="$1"
    shift

    local status=0
    "$GENERATOR" single "$@" \
        >"$WORK_DIR/usage.stdout" \
        2>"$WORK_DIR/usage.stderr" || status=$?

    if [[ "$status" -ne 2 ]]; then
        echo "ERROR: $description (exit code $status, expected 2)" >&2
        exit 1
    fi
}

expect_usage_error "k accepted for erdos_renyi" \
    --n 30 --p 0.2 --k 3 --seed 1 --output "$WORK_DIR/bad1.mis"
expect_usage_error "p accepted for barabasi_albert" \
    --family barabasi_albert --n 30 --k 3 --p 0.2 --seed 1 \
    --output "$WORK_DIR/bad2.mis"
expect_usage_error "barabasi_albert without k accepted" \
    --family barabasi_albert --n 30 --seed 1 --output "$WORK_DIR/bad3.mis"
expect_usage_error "k not smaller than n accepted" \
    --family barabasi_albert --n 3 --k 3 --seed 1 --output "$WORK_DIR/bad4.mis"
expect_usage_error "k = 0 accepted" \
    --family barabasi_albert --n 30 --k 0 --seed 1 --output "$WORK_DIR/bad5.mis"
expect_usage_error "unknown family accepted" \
    --family watts_strogatz --n 30 --k 3 --seed 1 --output "$WORK_DIR/bad6.mis"

if compgen -G "$WORK_DIR/bad*" >/dev/null; then
    echo "ERROR: a refused command wrote a file" >&2
    exit 1
fi

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
