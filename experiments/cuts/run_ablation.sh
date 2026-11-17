#!/usr/bin/env bash
#
# Cutting planes at the nodes: which of SCIP's separators make the trees
# smaller, and at what cost. A configuration is a SCIP preset (study:
# cuts at the root only; default: SCIP as shipped), a parameter file of
# experiments/cuts/settings applied on top of it, a branching rule (the
# learned rule uses the model of the set) and a cut selector.
#
# Usage, from the repository root:
#   experiments/cuts/run_ablation.sh [BUILD_DIR]
#
# Environment:
#   SETS               sets to run, comma-separated (default: er150,ba250)
#   CONFIGS            configurations to run, comma-separated (default: all)
#   INSTANCES_PER_SET  first instances of each set (default: all)
#   TIME_LIMIT         seconds per run (default 600)
#   SEEDS              SCIP seeds, comma-separated (default 0)
#
# Runs are made one at a time; a run already made is skipped, so the script
# resumes where it stopped and later calls can add sets or configurations.
#
# Outputs:
#   experiments/cuts/results/<set>_<config>.csv   one row per run
#   data/logs/cuts.log                            progress, one line per run
set -euo pipefail

BUILD_DIR="${1:-cmake-build-release}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RESULTS="experiments/cuts/results"
PARTS="$RESULTS/runs"
SETTINGS="experiments/cuts/settings"
LOG="data/logs/cuts.log"
BENCHMARK="$BUILD_DIR/mis-benchmark"

TIME_LIMIT="${TIME_LIMIT:-600}"
SEEDS="${SEEDS:-0}"
INSTANCES_PER_SET="${INSTANCES_PER_SET:-1000}"
SELECTED_SETS="${SETS:-er150,ba250}"
SELECTED_CONFIGS="${CONFIGS:-all}"

ER_MLP="experiments/er125_p01/models/mlp32_h2.model"
BA_MLP="experiments/ba200_k8/models/mlp32_ba1.model"

# Sets: name, instance directory, model of the learned rule.
ALL_SETS=(
    "er125 data/instances/er125_p01/test $ER_MLP"
    "ba200 data/instances/ba200_k8/test $BA_MLP"
    "er150 data/instances/generalization/er150_p0083 $ER_MLP"
    "ba250 data/instances/generalization/ba250_k8 $BA_MLP"
    "er200 data/instances/generalization/er200_p00625 $ER_MLP"
    "ba300 data/instances/generalization/ba300_k8 $BA_MLP"
)

# Configurations: name, preset, parameter file (- for none), rule, cut
# selector (- or omitted: SCIP's own).
ALL_CONFIGS=(
    "study study - relpscost"
    "default default - relpscost"
    "no_node_gomory default no_node_gomory relpscost"
    "zerohalf_nodes default zerohalf_nodes relpscost"
    "pool_nodes default pool_nodes relpscost"
    "zerohalf_depth10 default zerohalf_depth10 relpscost"
    "zerohalf_every_node default zerohalf_every_node relpscost"
    "clique_every_node default clique_every_node relpscost"
    "oddcycle default oddcycle relpscost"
    "learned_study study - mislearned"
    "learned_no_node_gomory default no_node_gomory mislearned"
    "study_root study root_only relpscost"
    "study_lookahead study - relpscost lookahead"
    "no_node_gomory_lookahead default no_node_gomory relpscost lookahead"
)

selected() {  # selected NAME LIST
    [[ "$2" == all || ",$2," == *",$1,"* ]]
}

mkdir -p "$PARTS" "$(dirname "$LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# Instances of a set, in a fixed order.
instances_of() {
    find "$1" -maxdepth 1 -name '*.mis' | sort | head -n "$INSTANCES_PER_SET"
}

total=0
done_runs=0

for entry in "${ALL_SETS[@]}"; do
    read -r name directory _ <<< "$entry"
    selected "$name" "$SELECTED_SETS" || continue

    for instance in $(instances_of "$directory"); do
        for config in "${ALL_CONFIGS[@]}"; do
            read -r config_name _ <<< "$config"
            selected "$config_name" "$SELECTED_CONFIGS" || continue

            for seed in ${SEEDS//,/ }; do
                total=$((total + 1))
                part="$PARTS/${name}__${config_name}__$(basename "$instance" .mis)__s$seed.csv"
                [[ -e "$part" ]] && done_runs=$((done_runs + 1))
            done
        done
    done
done

log "start: $total runs, $done_runs already made (sets $SELECTED_SETS, configurations $SELECTED_CONFIGS, time limit $TIME_LIMIT s)"
started=$(date +%s)
made=0

for entry in "${ALL_SETS[@]}"; do
    read -r name directory model <<< "$entry"
    selected "$name" "$SELECTED_SETS" || continue

    for instance in $(instances_of "$directory"); do
        base="$(basename "$instance" .mis)"
        single="$PARTS/.instance_$base"
        mkdir -p "$single"
        ln -sf "$ROOT/$instance" "$single/"
        [[ -e "$instance.meta" ]] && ln -sf "$ROOT/$instance.meta" "$single/"

        for config in "${ALL_CONFIGS[@]}"; do
            read -r config_name preset parameters rule selector <<< "$config"
            selected "$config_name" "$SELECTED_CONFIGS" || continue

            for seed in ${SEEDS//,/ }; do
                part="$PARTS/${name}__${config_name}__${base}__s$seed.csv"
                [[ -e "$part" ]] && continue

                options=(--instances "$single" --rules "$rule" --settings "$preset"
                         --seeds "$seed" --time-limit "$TIME_LIMIT" --output "$part.tmp")
                [[ "$parameters" != - ]] && options+=(--parameters "$SETTINGS/$parameters.set")
                [[ "$rule" == mislearned ]] && options+=(--model "$model")
                [[ "${selector:--}" != - ]] && options+=(--cut-selector "$selector")

                rm -f "$part.tmp"

                if ! "$BENCHMARK" "${options[@]}" > /dev/null 2> "$part.err"; then
                    log "FAILED $name $base $config_name seed $seed: $(tail -n 1 "$part.err")"
                    continue
                fi

                rm -f "$part.err"
                mv "$part.tmp" "$part"

                made=$((made + 1))
                done_runs=$((done_runs + 1))
                elapsed=$(( $(date +%s) - started ))
                remaining=$(( elapsed * (total - done_runs) / made ))

                result=$(tail -n 1 "$part" | awk -F, '{printf "%-8s %9d nodes %8.1f s", $7, $9, $11}')
                log "$(printf '[%3d/%d] %-6s %-34s %-20s %s | elapsed %dh%02d, about %dh%02d left' \
                    "$done_runs" "$total" "$name" "$base" "$config_name" "$result" \
                    $((elapsed / 3600)) $((elapsed % 3600 / 60)) \
                    $((remaining / 3600)) $((remaining % 3600 / 60)))"
            done
        done
    done
done

# One table per set and configuration, and the geometric means over the
# runs made so far (nodes shifted by 100, time in seconds unshifted).
log "summary (geometric means over the runs of each set):"

for entry in "${ALL_SETS[@]}"; do
    read -r name _ <<< "$entry"
    selected "$name" "$SELECTED_SETS" || continue

    for config in "${ALL_CONFIGS[@]}"; do
        read -r config_name _ <<< "$config"
        selected "$config_name" "$SELECTED_CONFIGS" || continue

        files=("$PARTS/${name}__${config_name}__"*.csv)
        [[ -e "${files[0]}" ]] || continue

        table="$RESULTS/${name}_${config_name}.csv"
        head -n 1 "${files[0]}" > "$table"

        for file in "${files[@]}"; do
            tail -n +2 "$file" >> "$table"
        done

        log "$(awk -F, -v s="$name" -v c="$config_name" 'NR > 1 {
                nodes += log($9 + 100); time += log($11); solved += $8; k++ }
            END { printf "  %-6s %-20s %2d/%d solved %9.0f nodes %8.2f s", s, c, solved, k,
                exp(nodes / k) - 100, exp(time / k) }' "$table")"
    done
done

log "done"
