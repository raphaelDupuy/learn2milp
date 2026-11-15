
#!/usr/bin/env bash
set -euo pipefail

GENERATOR="$1"
WORK_DIR="$2"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

OUT1="$WORK_DIR/first"
OUT2="$WORK_DIR/second"

generate_batch() {
    local output_dir="$1"

    "$GENERATOR" batch \
        --n-values 10,12 \
        --p-values 0.1,0.2 \
        --seeds 1,2 \
        --output-dir "$output_dir"
}

generate_batch "$OUT1"

# 2 sizes x 2 probabilities x 2 seeds = 8 instances.
test "$(find "$OUT1" -maxdepth 1 -name '*.mis' | wc -l)" -eq 8
test "$(find "$OUT1" -maxdepth 1 -name '*.mis.meta' | wc -l)" -eq 8
test "$(wc -l < "$OUT1/manifest.csv")" -eq 9

# File names use the shortest text of each probability.
for expected in \
    erdos_renyi_n10_p0_1_seed1 \
    erdos_renyi_n10_p0_2_seed2 \
    erdos_renyi_n12_p0_1_seed2 \
    erdos_renyi_n12_p0_2_seed1; do
    test -f "$OUT1/$expected.mis"
    test -f "$OUT1/$expected.mis.meta"
    grep -q "^$expected,$expected.mis," "$OUT1/manifest.csv"
done

# Every instance ID must be unique.
test "$(
    tail -n +2 "$OUT1/manifest.csv" |
    cut -d, -f1 |
    sort -u |
    wc -l
)" -eq 8

# Check the CSV header.
head -n 1 "$OUT1/manifest.csv" |
    grep -qx 'instance_id,file,generator_type,generator_version,seed,n,p,m'

# Check the generic metadata format of a generated instance.
FIRST_META="$(find "$OUT1" -maxdepth 1 -name '*.mis.meta' | sort | head -n 1)"

grep -qx 'INSTANCE_METADATA 1' "$FIRST_META"
grep -qx 'problem_type "MIS"' "$FIRST_META"
grep -qx 'generator_type "erdos_renyi"' "$FIRST_META"
grep -qx 'generator_version "1.0"' "$FIRST_META"
grep -q '^seed [0-9]\+$' "$FIRST_META"
grep -q '^parameter.n "' "$FIRST_META"
grep -q '^parameter.p "' "$FIRST_META"

# Repeating the batch in another directory must reproduce all outputs.
generate_batch "$OUT2"

cmp "$OUT1/manifest.csv" "$OUT2/manifest.csv"

while IFS=, read -r instance_id filename family version seed n p m
do
    [[ "$instance_id" == "instance_id" ]] && continue
    cmp "$OUT1/$filename" "$OUT2/$filename"
    cmp "$OUT1/$filename.meta" "$OUT2/$filename.meta"
done < "$OUT1/manifest.csv"

# Barabasi-Albert batch: named after n, k and the seed, with n and k in
# the manifest and m = k (n - k) edges.
BA="$WORK_DIR/barabasi_albert"

"$GENERATOR" batch \
    --family barabasi_albert \
    --n-values 20,30 \
    --k-values 2,3 \
    --seeds 1,2 \
    --output-dir "$BA" \
    >"$WORK_DIR/ba.stdout"

test "$(find "$BA" -maxdepth 1 -name '*.mis' | wc -l)" -eq 8
head -n 1 "$BA/manifest.csv" |
    grep -qx 'instance_id,file,generator_type,generator_version,seed,n,k,m'
grep -qx 'barabasi_albert_n20_k2_seed1,barabasi_albert_n20_k2_seed1.mis,barabasi_albert,1.0,1,20,2,36' \
    "$BA/manifest.csv"
grep -qx 'barabasi_albert_n30_k3_seed2,barabasi_albert_n30_k3_seed2.mis,barabasi_albert,1.0,2,30,3,81' \
    "$BA/manifest.csv"
grep -qx 'parameter.k "3"' "$BA/barabasi_albert_n30_k3_seed2.mis.meta"
grep -q '^\[8/8\] barabasi_albert_n30_k3_seed2.mis (m=81)$' "$WORK_DIR/ba.stdout"

# Parameters of the other family, or k not smaller than every n, are
# refused before anything is written.
for invalid in \
    "--family barabasi_albert --n-values 20 --k-values 2 --p-values 0.1" \
    "--family barabasi_albert --n-values 20 --p-values 0.1" \
    "--n-values 20 --p-values 0.1 --k-values 2" \
    "--family barabasi_albert --n-values 5,20 --k-values 5" \
    "--family watts_strogatz --n-values 20 --k-values 2"; do
    # shellcheck disable=SC2086
    if "$GENERATOR" batch $invalid \
        --seeds 1 \
        --output-dir "$WORK_DIR/invalid-family" \
        >"$WORK_DIR/invalid-family.stdout" \
        2>"$WORK_DIR/invalid-family.stderr"; then
        echo "ERROR: invalid batch was accepted: $invalid" >&2
        exit 1
    fi

    test ! -e "$WORK_DIR/invalid-family"
done

# Duplicate values must be rejected.
if "$GENERATOR" batch \
    --n-values 10,10 \
    --p-values 0.1 \
    --seeds 1 \
    --output-dir "$WORK_DIR/invalid" \
    >"$WORK_DIR/invalid.stdout" \
    2>"$WORK_DIR/invalid.stderr"; then
    echo "ERROR: duplicate n value was accepted" >&2
    exit 1
fi

# Invalid probabilities must be rejected.
if "$GENERATOR" batch \
    --n-values 10 \
    --p-values 0.1,1.5 \
    --seeds 1 \
    --output-dir "$WORK_DIR/invalid-p" \
    >"$WORK_DIR/invalid-p.stdout" \
    2>"$WORK_DIR/invalid-p.stderr"; then
    echo "ERROR: invalid probability was accepted" >&2
    exit 1
fi

# Malformed comma-separated lists must be rejected.
for malformed_list in "10," "10,,20" ","; do
    if "$GENERATOR" batch \
        --n-values "$malformed_list" \
        --p-values 0.1 \
        --seeds 1 \
        --output-dir "$WORK_DIR/invalid-list" \
        >"$WORK_DIR/invalid-list.stdout" \
        2>"$WORK_DIR/invalid-list.stderr"; then
        echo "ERROR: malformed list '$malformed_list' was accepted" >&2
        exit 1
    fi
done

# A pre-existing temporary manifest must be preserved.
PREEXISTING="$WORK_DIR/preexisting-temp"
mkdir -p "$PREEXISTING"
printf 'keep this file\n' > "$PREEXISTING/manifest.csv.tmp"

if "$GENERATOR" batch \
    --n-values 10 \
    --p-values 0.1 \
    --seeds 1 \
    --output-dir "$PREEXISTING" \
    >"$WORK_DIR/preexisting.stdout" \
    2>"$WORK_DIR/preexisting.stderr"; then
    echo "ERROR: existing temporary manifest was accepted" >&2
    exit 1
fi

test "$(cat "$PREEXISTING/manifest.csv.tmp")" = "keep this file"
test ! -e "$PREEXISTING/manifest.csv"
test "$(find "$PREEXISTING" -maxdepth 1 -name '*.mis' | wc -l)" -eq 0


echo "All MIS generator batch tests passed."
