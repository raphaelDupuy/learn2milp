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
#   EXPERIMENT         name of a separate experiment: its runs and tables go
#                      to experiments/cuts/results/EXPERIMENT/, so that
#                      configurations already run elsewhere are run again
#                      in the same session (default: none)
#   NODE_LOGS          1 to write the node log of every run (mis-benchmark
#                      --node-log-dir) to the nodes/ folder of the results
#   STATISTICS         1 to write SCIP's statistics of every run as JSON
#                      (mis-benchmark --statistics-dir) to the stats/ folder
#                      of the results, for experiments/profile/time_breakdown.py
#
# Runs are made one at a time; a run already made is skipped, so the script
# resumes where it stopped and later calls can add sets or configurations.
#
# A run during which the machine slept is discarded: SCIP measures time on
# the wall clock, which keeps going while the WSL virtual machine is
# suspended, whereas /proc/uptime stops. Its table is kept as
# runs/*.csv.suspended, and the next call makes the run again.
#
# Outputs:
#   experiments/cuts/results/[EXPERIMENT/]<set>_<config>.csv   one row per run
#   experiments/cuts/results/[EXPERIMENT/]nodes/<set>__<config>/   node logs
#   experiments/cuts/results/[EXPERIMENT/]stats/<set>__<config>/   statistics
#   data/logs/cuts.log                     progress, one line per run
set -euo pipefail

BUILD_DIR="${1:-cmake-build-release}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RESULTS="experiments/cuts/results${EXPERIMENT:+/$EXPERIMENT}"
PARTS="$RESULTS/runs"
NODES="$RESULTS/nodes"
STATS="$RESULTS/stats"
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
    # Density sweep at fixed size (seeds 81-90, mis-generator batch): mean
    # degree 6 to 20 for G(150, p), with er150 at 12.4; k = 4 to 12 for
    # BA(250, k), with ba250 at k = 8.
    "er150_d6 data/instances/density/er150_p0040 $ER_MLP"
    "er150_d8 data/instances/density/er150_p0054 $ER_MLP"
    "er150_d10 data/instances/density/er150_p0067 $ER_MLP"
    "er150_d14 data/instances/density/er150_p0094 $ER_MLP"
    "er150_d17 data/instances/density/er150_p0114 $ER_MLP"
    "er150_d20 data/instances/density/er150_p0134 $ER_MLP"
    "ba250_k4 data/instances/density/ba250_k4 $BA_MLP"
    "ba250_k6 data/instances/density/ba250_k6 $BA_MLP"
    "ba250_k10 data/instances/density/ba250_k10 $BA_MLP"
    "ba250_k12 data/instances/density/ba250_k12 $BA_MLP"
    # Validation instances, to check on fresh instances the settings chosen
    # on the test sets: seeds 61-80 of the training sizes, and seeds 61-70
    # (mis-generator batch) of the other sets. G(200) comes last, as it
    # takes most of the time.
    "er125_valid data/instances/er125_p01/valid $ER_MLP"
    "ba200_valid data/instances/ba200_k8/valid $BA_MLP"
    "er150_d8_valid data/instances/validation/er150_p0054 $ER_MLP"
    "ba250_k6_valid data/instances/validation/ba250_k6 $BA_MLP"
    "er150_valid data/instances/validation/er150_p0083 $ER_MLP"
    "ba250_valid data/instances/validation/ba250_k8 $BA_MLP"
    "ba300_valid data/instances/validation/ba300_k8 $BA_MLP"
    "er200_valid data/instances/validation/er200_p00625 $ER_MLP"
)

# Configurations: name, preset, parameter file (- for none), rule, cut
# selector (- or omitted: SCIP's own), candidates of the learned rule
# evaluated by strong branching (- or omitted: none), and "oracle" for runs
# that start from an optimal solution (mis-benchmark --primal-oracle on).
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
    "oddcycle_every_node default oddcycle_every_node relpscost"
    "oddcycle_gate025 default oddcycle_gate025 relpscost"
    "oddcycle_gate050 default oddcycle_gate050 relpscost"
    "oddcycle_gate100 default oddcycle_gate100 relpscost"
    "pool_every_node default pool_every_node relpscost"
    "zerohalf_local_every_node default zerohalf_local_every_node relpscost"
    "learned_study study - mislearned"
    "learned_no_node_gomory default no_node_gomory mislearned"
    "learned_oddcycle default oddcycle mislearned"
    "study_cliquelifting study cliquelifting relpscost"
    "no_node_gomory_cliquelifting default no_node_gomory_cliquelifting relpscost"
    "oddcycle_cliquelifting default oddcycle_cliquelifting relpscost"
    "oddcycle_cliquelifting_nodive default oddcycle_cliquelifting_nodive relpscost"
    "oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory relpscost"
    "oddcycle_cliquelifting_nodive_nogomory default oddcycle_cliquelifting_nodive_nogomory relpscost"
    "learned_study_cliquelifting study cliquelifting mislearned"
    "learned_no_node_gomory_cliquelifting default no_node_gomory_cliquelifting mislearned"
    "learned_oddcycle_cliquelifting default oddcycle_cliquelifting mislearned"
    "learned_oddcycle_cliquelifting_nodive default oddcycle_cliquelifting_nodive mislearned"
    "learned_oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory mislearned"
    "learned_oddcycle_cliquelifting_nodive_nogomory default oddcycle_cliquelifting_nodive_nogomory mislearned"
    # Strong branching on the first K candidates of the model, which can
    # prune like the expert.
    "learned_sb1_oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory mislearned - 1"
    "learned_sb2_oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory mislearned - 2"
    # Without primal heuristics, then also starting from an optimal
    # solution: what any policy of the heuristics could gain at most.
    "oddcycle_cliquelifting_nogomory_noheur default oddcycle_cliquelifting_nogomory_noheur relpscost"
    "oddcycle_cliquelifting_nogomory_noheur_oracle default oddcycle_cliquelifting_nogomory_noheur relpscost - - oracle"
    "learned_oddcycle_cliquelifting_nogomory_noheur default oddcycle_cliquelifting_nogomory_noheur mislearned"
    "learned_oddcycle_cliquelifting_nogomory_noheur_oracle default oddcycle_cliquelifting_nogomory_noheur mislearned - - oracle"
    # The expert in the deployed settings, for the headroom of the learned
    # rule: with and without its strong branching pruning.
    "fullstrong_oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory misfullstrong"
    "fullstrong_nocutoff_oddcycle_cliquelifting_nogomory default oddcycle_cliquelifting_nogomory misfullstrong_nocutoff"
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
# Column widths of the progress lines and of the summary, so that every
# line is aligned whatever the names of the sets and configurations.
name_width=0
base_width=0
config_width=0

for entry in "${ALL_SETS[@]}"; do
    read -r name directory _ <<< "$entry"
    selected "$name" "$SELECTED_SETS" || continue
    name_width=$(( ${#name} > name_width ? ${#name} : name_width ))

    for instance in $(instances_of "$directory"); do
        base="$(basename "$instance" .mis)"
        base_width=$(( ${#base} > base_width ? ${#base} : base_width ))

        for config in "${ALL_CONFIGS[@]}"; do
            read -r config_name _ <<< "$config"
            selected "$config_name" "$SELECTED_CONFIGS" || continue
            config_width=$(( ${#config_name} > config_width ? ${#config_name} : config_width ))

            for seed in ${SEEDS//,/ }; do
                total=$((total + 1))
                part="$PARTS/${name}__${config_name}__$(basename "$instance" .mis)__s$seed.csv"
                [[ -s "$part" ]] && done_runs=$((done_runs + 1))
            done
        done
    done
done

log "start: $total runs, $done_runs already made (sets $SELECTED_SETS, configurations $SELECTED_CONFIGS, time limit $TIME_LIMIT s)"
started=$(date +%s)
made=0
suspended_runs=0

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
            read -r config_name preset parameters rule selector strong oracle <<< "$config"
            selected "$config_name" "$SELECTED_CONFIGS" || continue

            for seed in ${SEEDS//,/ }; do
                part="$PARTS/${name}__${config_name}__${base}__s$seed.csv"
                [[ -s "$part" ]] && continue

                options=(--instances "$single" --rules "$rule" --settings "$preset"
                         --seeds "$seed" --time-limit "$TIME_LIMIT" --output "$part.tmp")
                [[ "$parameters" != - ]] && options+=(--parameters "$SETTINGS/$parameters.set")
                [[ "$rule" == mislearned ]] && options+=(--model "$model")
                [[ "${selector:--}" != - ]] && options+=(--cut-selector "$selector")
                [[ "${strong:--}" != - ]] && options+=(--strong-branching "$strong")
                [[ "${oracle:-}" == oracle ]] && options+=(--primal-oracle on)

                # The node log of a run that was not completed is stale.
                if [[ "${NODE_LOGS:-0}" == 1 ]]; then
                    node_dir="$NODES/${name}__${config_name}"
                    mkdir -p "$node_dir"
                    rm -f "$node_dir/${base}__${rule}__s$seed.nodes.csv"
                    options+=(--node-log-dir "$node_dir")
                fi

                # Likewise for the statistics, written at the end of a run.
                if [[ "${STATISTICS:-0}" == 1 ]]; then
                    stats_dir="$STATS/${name}__${config_name}"
                    mkdir -p "$stats_dir"
                    rm -f "$stats_dir/${base}__${rule}__s$seed.stats.json"
                    options+=(--statistics-dir "$stats_dir")
                fi

                rm -f "$part.tmp"
                read -r uptime_before _ < /proc/uptime
                wall_before=$(date +%s.%N)

                if ! "$BENCHMARK" "${options[@]}" > /dev/null 2> "$part.err"; then
                    log "FAILED $name $base $config_name seed $seed: $(tail -n 1 "$part.err")"
                    continue
                fi

                rm -f "$part.err"

                read -r uptime_after _ < /proc/uptime
                wall_after=$(date +%s.%N)
                slept=$(awk -v w0="$wall_before" -v w1="$wall_after" -v u0="$uptime_before" -v u1="$uptime_after" \
                    'BEGIN { d = (w1 - w0) - (u1 - u0); print (d > 2 ? int(d + 0.5) : 0) }')

                if (( slept > 0 )); then
                    mv "$part.tmp" "$part.suspended"
                    suspended_runs=$((suspended_runs + 1))
                    log "SUSPENDED $name $base $config_name seed $seed: the machine slept about $slept s during the run, which is discarded"
                    continue
                fi

                # Written to disk before the rename: after a host crash, the
                # rename alone can survive with an empty file.
                sync "$part.tmp"
                [[ "${NODE_LOGS:-0}" == 1 ]] && sync "$node_dir/${base}__${rule}__s$seed.nodes.csv"
                [[ "${STATISTICS:-0}" == 1 ]] && sync "$stats_dir/${base}__${rule}__s$seed.stats.json"
                mv "$part.tmp" "$part"

                made=$((made + 1))
                done_runs=$((done_runs + 1))
                elapsed=$(( $(date +%s) - started ))
                remaining=$(( elapsed * (total - done_runs) / made ))

                # nodes and time of the run (columns 9 and 11), and its
                # status (column 7) at the end of the line unless optimal
                last=$(tail -n 1 "$part")
                result=$(awk -F, '{printf "%7d nodes %7.1f s", $9, $11}' <<< "$last")
                status=$(cut -d, -f7 <<< "$last")
                note=""
                [[ "$status" != OPTIMAL ]] && note="  $status"
                log "$(printf '[%*d/%d] %-*s %-*s %-*s %s | elapsed %dh%02d, about %dh%02d left%s' \
                    "${#total}" "$done_runs" "$total" "$name_width" "$name" "$base_width" "$base" \
                    "$config_width" "$config_name" "$result" \
                    $((elapsed / 3600)) $((elapsed % 3600 / 60)) \
                    $((remaining / 3600)) $((remaining % 3600 / 60)) "$note")"
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

        log "$(awk -F, -v s="$name" -v c="$config_name" -v sw="$name_width" -v cw="$config_width" 'NR > 1 {
                nodes += log($9 + 100); time += log($11); solved += $8; k++ }
            END { printf "  %-" sw "s %-" cw "s %2d/%d solved %9.0f nodes %8.2f s", s, c, solved, k,
                exp(nodes / k) - 100, exp(time / k) }' "$table")"
    done
done

if (( suspended_runs > 0 )); then
    log "$suspended_runs runs discarded because the machine slept: call the script again to make them"
fi

log "done"
