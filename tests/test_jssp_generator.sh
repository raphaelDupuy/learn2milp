#!/usr/bin/env bash
set -euo pipefail

GENERATOR="$1"
WORK_DIR="$2"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# Same seed twice: same instance and metadata.
"$GENERATOR" single --jobs 6 --machines 4 --seed 7 --output "$WORK_DIR/first.jsp" > /dev/null
"$GENERATOR" single --jobs 6 --machines 4 --seed 7 --output "$WORK_DIR/second.jsp" > /dev/null
cmp "$WORK_DIR/first.jsp" "$WORK_DIR/second.jsp"
cmp "$WORK_DIR/first.jsp.meta" "$WORK_DIR/second.jsp.meta"

head -n 1 "$WORK_DIR/first.jsp" | grep -qx '6 4' || fail "header line"
[[ $(wc -l < "$WORK_DIR/first.jsp") -eq 7 ]] || fail "one line per job"
grep -qx 'problem_type "JSSP"' "$WORK_DIR/first.jsp.meta" || fail "problem type"
grep -qx 'generator_type "taillard"' "$WORK_DIR/first.jsp.meta" || fail "generator type"
grep -qx 'seed 7' "$WORK_DIR/first.jsp.meta" || fail "seed"
grep -qx 'parameter.jobs "6"' "$WORK_DIR/first.jsp.meta" || fail "jobs parameter"

# Existing files are never overwritten.
if "$GENERATOR" single --jobs 6 --machines 4 --seed 8 --output "$WORK_DIR/first.jsp" \
    > /dev/null 2>&1; then
    fail "an existing file was overwritten"
fi
cmp "$WORK_DIR/first.jsp" "$WORK_DIR/second.jsp"

# The published seeds give ta01 (first job of the published file).
"$GENERATOR" single --jobs 15 --machines 15 \
    --time-seed 840612802 --machine-seed 398197754 \
    --output "$WORK_DIR/ta01.jsp" > /dev/null
sed -n 2p "$WORK_DIR/ta01.jsp" \
    | grep -qx '6 94 12 66 4 10 7 53 3 26 2 15 10 65 11 82 8 10 14 27 9 93 13 92 5 96 0 70 1 83' \
    || fail "ta01"
if grep -q '^seed' "$WORK_DIR/ta01.jsp.meta"; then
    fail "no seed is recorded when the two seeds are given"
fi

# Seeds: either --seed or both Taillard seeds.
for arguments in \
    "--seed 1 --time-seed 5 --machine-seed 6" \
    "--time-seed 5" \
    "--time-seed 0 --machine-seed 6" \
    "--time-seed 2147483647 --machine-seed 6" \
    "--jobs 0 --seed 1"; do
    # shellcheck disable=SC2086
    if "$GENERATOR" single --jobs 3 --machines 3 $arguments \
        --output "$WORK_DIR/refused.jsp" > /dev/null 2>&1; then
        fail "accepted: $arguments"
    fi
done
[[ ! -e "$WORK_DIR/refused.jsp" ]] || fail "a refused request left a file"

# Batch: every combination, a manifest written last.
"$GENERATOR" batch --jobs-values 4,5 --machines-values 3 --seeds 1,2,3 \
    --output-dir "$WORK_DIR/batch" > /dev/null
[[ $(ls "$WORK_DIR/batch"/*.jsp | wc -l) -eq 6 ]] || fail "batch instances"
[[ $(ls "$WORK_DIR/batch"/*.jsp.meta | wc -l) -eq 6 ]] || fail "batch metadata"
head -n 1 "$WORK_DIR/batch/manifest.csv" \
    | grep -qx 'instance_id,file,generator_type,generator_version,seed,jobs,machines,time_seed,machine_seed,lower_bound,dispatch_makespan' \
    || fail "manifest header"
[[ $(wc -l < "$WORK_DIR/batch/manifest.csv") -eq 7 ]] || fail "manifest rows"
grep -q '^taillard_j5_m3_seed2,taillard_j5_m3_seed2.jsp,taillard,1.0,2,5,3,' \
    "$WORK_DIR/batch/manifest.csv" || fail "manifest row"

# A batch instance is the single instance of the same seed.
"$GENERATOR" single --jobs 5 --machines 3 --seed 2 --output "$WORK_DIR/single.jsp" > /dev/null
cmp "$WORK_DIR/single.jsp" "$WORK_DIR/batch/taillard_j5_m3_seed2.jsp"

# A batch is not written over another.
if "$GENERATOR" batch --jobs-values 4 --machines-values 3 --seeds 9 \
    --output-dir "$WORK_DIR/batch" > /dev/null 2>&1; then
    fail "a second batch was written in the same directory"
fi

echo "All jssp-generator tests passed"
