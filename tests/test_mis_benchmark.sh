
#!/usr/bin/env bash
set -euo pipefail

GENERATOR="$1"
BENCHMARK="$2"
WORK_DIR="$3"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Two instances that need branching with the study settings.
"$GENERATOR" batch \
    --n-values 100 \
    --p-values 0.1 \
    --seeds 1,2 \
    --output-dir "$WORK_DIR/instances" \
    >/dev/null

RESULTS="$WORK_DIR/results.csv"

"$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost,pscost \
    --seeds 0,1 \
    --output "$RESULTS" \
    >"$WORK_DIR/stdout"

# One row per (instance, seed, rule), plus the header.
head -n 1 "$RESULTS" |
    grep -qx 'instance,n,m,settings,rule,seed,status,solved,nodes,lp_iterations,time_seconds,primal_bound,dual_bound,root_first_lp_bound,root_dual_bound,cuts_applied,lookahead_scored,lookahead_improving,gate_separated,gate_skipped,features_seconds,inference_seconds,problem'

test "$(wc -l <"$RESULTS")" -eq 9
test "$(grep -c ',study,relpscost,' "$RESULTS")" -eq 4
test "$(grep -c ',study,pscost,' "$RESULTS")" -eq 4
test "$(grep -c ',OPTIMAL,1,' "$RESULTS")" -eq 8
grep -q '^erdos_renyi_n100_p0_1_seed1.mis,100,' "$RESULTS"

# Rules are interleaved on each instance.
sed -n '2p' "$RESULTS" | grep -q ',relpscost,0,'
sed -n '3p' "$RESULTS" | grep -q ',pscost,0,'

# Every row has the same number of fields.
test "$(awk -F, '{print NF}' "$RESULTS" | sort -u | wc -l)" -eq 1

# All rules and seeds prove the same optimum on a given instance.
test "$(
    grep '^erdos_renyi_n100_p0_1_seed1.mis,' "$RESULTS" |
    cut -d, -f12 |
    sort -u |
    wc -l
)" -eq 1

# The summary lists every rule.
grep -q '^relpscost .*4/4' "$WORK_DIR/stdout"
grep -q '^pscost .*4/4' "$WORK_DIR/stdout"

# Node counts are reproducible.
"$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost,pscost \
    --seeds 0,1 \
    --output "$WORK_DIR/again.csv" \
    >/dev/null

cmp <(cut -d, -f1-10 "$RESULTS") <(cut -d, -f1-10 "$WORK_DIR/again.csv")

# SCIP as shipped is available as an outside reference.
"$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --settings default \
    --output "$WORK_DIR/default.csv" \
    >/dev/null

test "$(grep -c ',default,relpscost,0,OPTIMAL,1,' "$WORK_DIR/default.csv")" -eq 2

# Node logs: one file per run, with one line per node after the header.
"$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --node-log-dir "$WORK_DIR/nodes" \
    --output "$WORK_DIR/logged.csv" \
    >/dev/null

for INSTANCE in erdos_renyi_n100_p0_1_seed1 erdos_renyi_n100_p0_1_seed2; do
    NODE_LOG="$WORK_DIR/nodes/${INSTANCE}__relpscost__s0.nodes.csv"
    NODES="$(grep "^${INSTANCE}.mis," "$WORK_DIR/logged.csv" | cut -d, -f9)"
    test "$(($(wc -l <"$NODE_LOG") - 1))" -eq "$NODES"
done

# An existing node log is never overwritten, and nothing is run.
if "$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --node-log-dir "$WORK_DIR/nodes" \
    --output "$WORK_DIR/relogged.csv" \
    >/dev/null \
    2>"$WORK_DIR/relogged.stderr"; then
    echo "ERROR: an existing node log was overwritten" >&2
    exit 1
fi

test ! -e "$WORK_DIR/relogged.csv"
grep -q 'File already exists' "$WORK_DIR/relogged.stderr"

# SCIP's statistics of every run, as JSON, never overwritten.
"$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --statistics-dir "$WORK_DIR/stats" \
    --output "$WORK_DIR/stats.csv" \
    >/dev/null

for INSTANCE in erdos_renyi_n100_p0_1_seed1 erdos_renyi_n100_p0_1_seed2; do
    python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["timing"]["solving_time"] > 0' \
        "$WORK_DIR/stats/${INSTANCE}__relpscost__s0.stats.json"
done

if "$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --statistics-dir "$WORK_DIR/stats" \
    --output "$WORK_DIR/restats.csv" \
    >/dev/null \
    2>"$WORK_DIR/restats.stderr"; then
    echo "ERROR: existing statistics were overwritten" >&2
    exit 1
fi

test ! -e "$WORK_DIR/restats.csv"
grep -q 'File already exists' "$WORK_DIR/restats.stderr"

# Existing results must not be overwritten.
cp "$RESULTS" "$WORK_DIR/results.before"

if "$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules relpscost \
    --output "$RESULTS" \
    >"$WORK_DIR/overwrite.stdout" \
    2>"$WORK_DIR/overwrite.stderr"; then
    echo "ERROR: existing results were overwritten" >&2
    exit 1
fi

cmp "$RESULTS" "$WORK_DIR/results.before"

# Invalid requests must be rejected without creating a result table.
for INVALID in \
    "--rules no_such_rule" \
    "--rules relpscost,relpscost" \
    "--rules relpscost, " \
    "--rules relpscost --seeds 0,x" \
    "--rules relpscost --settings other" \
    "--rules relpscost --time-limit 0" \
    "--rules relpscost --unknown value" \
    "--rules mislearned" \
    "--rules relpscost --model $WORK_DIR/instances/manifest.csv"; do
    # shellcheck disable=SC2086
    if "$BENCHMARK" \
        --instances "$WORK_DIR/instances" \
        --output "$WORK_DIR/invalid.csv" \
        $INVALID \
        >"$WORK_DIR/invalid.stdout" \
        2>"$WORK_DIR/invalid.stderr"; then
        echo "ERROR: invalid request was accepted: $INVALID" >&2
        exit 1
    fi

    test ! -e "$WORK_DIR/invalid.csv"
done

# A file that is not a model must be rejected before any run.
if "$BENCHMARK" \
    --instances "$WORK_DIR/instances" \
    --rules mislearned \
    --model "$WORK_DIR/instances/manifest.csv" \
    --output "$WORK_DIR/invalid.csv" \
    >"$WORK_DIR/invalid.stdout" \
    2>"$WORK_DIR/invalid.stderr"; then
    echo "ERROR: invalid model file was accepted" >&2
    exit 1
fi

test ! -e "$WORK_DIR/invalid.csv"
grep -q 'Invalid model file' "$WORK_DIR/invalid.stderr"

# A directory without instances must be rejected.
mkdir -p "$WORK_DIR/empty"

if "$BENCHMARK" \
    --instances "$WORK_DIR/empty" \
    --rules relpscost \
    --output "$WORK_DIR/empty.csv" \
    >"$WORK_DIR/empty.stdout" \
    2>"$WORK_DIR/empty.stderr"; then
    echo "ERROR: empty instance directory was accepted" >&2
    exit 1
fi

test ! -e "$WORK_DIR/empty.csv"

echo "All MIS benchmark CLI tests passed."
