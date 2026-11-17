#!/usr/bin/env bash
#
# Generalization to larger graphs: the models trained on er125_p01 and
# ba200_k8 are evaluated, without retraining, on larger instances of the
# same families, against SCIP's default rule and SCIP's default settings.
#
# Usage, from the repository root:
#   experiments/generalization/run_benchmark.sh [BUILD_DIR]
#
# The runs are made one at a time (the solving times are measured), one
# instance after the other with every configuration in turn, so that the
# comparison builds up as the instances are solved. A run already made is
# skipped: the script resumes where it stopped.
#
# Outputs:
#   data/instances/generalization/<set>/                   instances (ignored by git)
#   experiments/generalization/results/<set>_<config>.csv  one row per run
#   data/logs/generalization.log                           progress, one line per run
#
# Follow the progress with
#   tail -n +1 -F data/logs/generalization.log
set -euo pipefail

BUILD_DIR="${1:-cmake-build-release}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTANCES="$ROOT/data/instances/generalization"
RESULTS="$ROOT/experiments/generalization/results"
PARTS="$RESULTS/runs"
LOG="$ROOT/data/logs/generalization.log"

GENERATOR="$BUILD_DIR/mis-generator"
BENCHMARK="$BUILD_DIR/mis-benchmark"

TIME_LIMIT="${TIME_LIMIT:-3600}"
SEEDS="${SEEDS:-0}"
INSTANCE_SEEDS="${INSTANCE_SEEDS:-$(seq -s, 81 90)}"

# Instance sets: name, family, n, p or k, models of the learned rule by
# kind (kind:path, comma-separated). The Erdos-Renyi graphs keep the mean
# degree of er125_p01 (about 12.4), the Barabasi-Albert ones the k of
# ba200_k8. Calibration (relpscost, one instance): G(150, 0.083) 24 s,
# G(200, 0.0625) 4 min, BA(250, 8) 1 min, BA(300, 8) 3 min, while
# G(250, 0.05) and BA(400, 8) were not solved within 10 minutes.
ER_MODELS="mlp:experiments/er125_p01/models/mlp32_h2.model,gnn:experiments/er125_p01/models/gnn32x2_h2.model"
BA_MODELS="mlp:experiments/ba200_k8/models/mlp32_ba1.model,gnn:experiments/ba200_k8/models/gnn32x2_ba1.model,gcnn:experiments/ba200_k8/models/gcnn32x1_ba1.model"

SETS=(
    "er150_p0083 erdos_renyi 150 0.083 $ER_MODELS"
    "ba250_k8 barabasi_albert 250 8 $BA_MODELS"
    "er200_p00625 erdos_renyi 200 0.0625 $ER_MODELS"
    "ba300_k8 barabasi_albert 300 8 $BA_MODELS"
)

# Configurations: name, rule, settings, strong branching candidates, kind
# of model (- for none). A configuration whose kind of model a set does
# not have is skipped on that set (no bipartite model for Erdos-Renyi).
CONFIGS=(
    "scip_default relpscost default 0 -"
    "relpscost relpscost study 0 -"
    "learned mislearned study 0 mlp"
    "learned_sb1 mislearned study 1 mlp"
    "learned_sb2 mislearned study 2 mlp"
    "gnn mislearned study 0 gnn"
    "gcnn mislearned study 0 gcnn"
)

# model_for KIND MODELS: the path of the model of that kind, or nothing.
model_for() {
    local entry
    for entry in ${2//,/ }; do
        [[ "${entry%%:*}" == "$1" ]] && { echo "${entry#*:}"; return; }
    done
    return 0
}

# applies KIND MODELS: whether a configuration runs on a set.
applies() {
    [[ "$1" == - || -n "$(model_for "$1" "$2")" ]]
}

mkdir -p "$INSTANCES" "$PARTS" "$(dirname "$LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

for entry in "${SETS[@]}"; do
    read -r name family n parameter _ <<< "$entry"
    directory="$INSTANCES/$name"

    if [[ ! -e "$directory/manifest.csv" ]]; then
        if [[ "$family" == erdos_renyi ]]; then
            values=(--n-values "$n" --p-values "$parameter")
        else
            values=(--n-values "$n" --k-values "$parameter")
        fi

        "$GENERATOR" batch --family "$family" "${values[@]}" \
            --seeds "$INSTANCE_SEEDS" --output-dir "$directory" > /dev/null
        log "generated $name: $(ls "$directory"/*.mis | wc -l) instances"
    fi
done

# Total and remaining runs, for the progress lines.
total=0
done_runs=0

for entry in "${SETS[@]}"; do
    read -r name _ _ _ models <<< "$entry"

    for instance in "$INSTANCES/$name"/*.mis; do
        for config in "${CONFIGS[@]}"; do
            read -r config_name _ _ _ kind <<< "$config"
            applies "$kind" "$models" || continue

            for seed in ${SEEDS//,/ }; do
                total=$((total + 1))
                part="$PARTS/${name}__${config_name}__$(basename "$instance" .mis)__s$seed.csv"
                [[ -e "$part" ]] && done_runs=$((done_runs + 1))
            done
        done
    done
done

log "start: $total runs, $done_runs already made, time limit $TIME_LIMIT s, SCIP seeds $SEEDS"
started=$(date +%s)
made=0

for entry in "${SETS[@]}"; do
    read -r name _ _ _ models <<< "$entry"

    for instance in "$INSTANCES/$name"/*.mis; do
        base="$(basename "$instance" .mis)"
        single="$PARTS/.instance_$base"
        mkdir -p "$single"
        ln -sf "$instance" "$single/"
        [[ -e "$instance.meta" ]] && ln -sf "$instance.meta" "$single/"

        for config in "${CONFIGS[@]}"; do
            read -r config_name rule settings strong kind <<< "$config"
            applies "$kind" "$models" || continue
            model="$(model_for "$kind" "$models")"

            for seed in ${SEEDS//,/ }; do
                part="$PARTS/${name}__${config_name}__${base}__s$seed.csv"

                [[ -e "$part" ]] && continue

                options=(--instances "$single" --rules "$rule" --settings "$settings"
                         --seeds "$seed" --time-limit "$TIME_LIMIT" --output "$part.tmp")

                if [[ "$rule" == mislearned ]]; then
                    options+=(--model "$model")
                    [[ "$strong" != 0 ]] && options+=(--strong-branching "$strong")
                fi

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

                # status, nodes and time of the run (columns 7, 9 and 11)
                result=$(tail -n 1 "$part" | awk -F, '{printf "%-8s %9d nodes %9.1f s", $7, $9, $11}')
                log "$(printf '[%3d/%d] %-10s %-36s %-12s %s | elapsed %dh%02d, about %dh%02d left' \
                    "$done_runs" "$total" "$name" "$base" "$config_name" "$result" \
                    $((elapsed / 3600)) $((elapsed % 3600 / 60)) \
                    $((remaining / 3600)) $((remaining % 3600 / 60)))"
            done
        done

        # Comparison of the configurations on this instance.
        summary=$(for config in "${CONFIGS[@]}"; do
            read -r config_name _ _ _ kind <<< "$config"
            applies "$kind" "$models" || continue
            cat "$PARTS/${name}__${config_name}__${base}"__s*.csv 2>/dev/null \
                | awk -F, -v c="$config_name" '$1 != "instance" {n += $9; t += $11; k++}
                    END {if (k) printf "%s %.0f nodes %.1f s | ", c, n / k, t / k}'
        done)
        log "  $base: ${summary% | }"
    done
done

# One table per set and configuration, and the shifted geometric means
# (nodes shift 100, time shift 1 s, as mis-benchmark).
log "summary (shifted geometric means over the runs of each set):"

for entry in "${SETS[@]}"; do
    read -r name _ _ _ models <<< "$entry"

    for config in "${CONFIGS[@]}"; do
        read -r config_name _ _ _ kind <<< "$config"
        applies "$kind" "$models" || continue
        table="$RESULTS/${name}_${config_name}.csv"
        files=("$PARTS/${name}__${config_name}__"*.csv)

        head -n 1 "${files[0]}" > "$table"

        for file in "${files[@]}"; do
            tail -n +2 "$file" >> "$table"
        done

        log "$(awk -F, -v s="$name" -v c="$config_name" 'NR > 1 {
                nodes += log($9 + 100); time += log($11 + 1); solved += $8; k++ }
            END { printf "  %-10s %-12s %2d/%d solved %10.0f nodes %9.1f s", s, c, solved, k,
                exp(nodes / k) - 100, exp(time / k) - 1 }' "$table")"
    done
done

log "done"
