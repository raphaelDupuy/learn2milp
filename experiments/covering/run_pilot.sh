#!/usr/bin/env bash
#
# Covering pilot: minimum dominating set on G(150, 0.067) and set cover
# 1000 x 1000 (density 0.05), 20 instances each. For each problem:
#   1. relpscost, writing the best solution (the oracle of step 3);
#   2. full strong branching and vanillafullstrong;
#   3. the three rules again, from the solution of step 1.
#
# Usage, from the repository root:
#   experiments/covering/run_pilot.sh
#
# Stop: Ctrl+C, or kill the python process (the solves in progress are run
# again on resume), or touch experiments/covering/results/pilot/<problem>/STOP
# (the solves in progress finish). Resume: run the script again; finished
# runs are skipped.
set -euo pipefail

UV="${UV:-$HOME/.local/bin/uv}"
PILOT=(experiments/covering/pilot/covering_pilot.py)
INSTANCES=data/instances/covering/pilot
RESULTS=experiments/covering/results/pilot
TIME_LIMIT="${TIME_LIMIT:-900}"
JOBS="${JOBS:-4}"

pilot() { "$UV" run --project ml python "${PILOT[@]}" "$@"; }

mkdir -p "$INSTANCES"
for seed in $(seq 1 20); do
    path="$INSTANCES/er_n150_p0.067_s$seed.mis"
    [[ -e "$path" ]] || ./cmake-build-release/mis-generator single \
        --family erdos_renyi --n 150 --p 0.067 --seed "$seed" --output "$path" >/dev/null
done
pilot generate-setcover --rows 1000 --cols 1000 --density 0.05 --seeds 1-20 \
    --output "$INSTANCES" >/dev/null

for problem in dominating setcover; do
    case "$problem" in
        dominating) pattern='er_n150_p0.067_s*.mis' ;;
        setcover) pattern='setcover_r1000_c1000_d0.05_s*.scp' ;;
    esac
    out="$RESULTS/$problem"
    common=(--instances "$INSTANCES" --pattern "$pattern" --output "$out"
            --time-limit "$TIME_LIMIT" --jobs "$JOBS")
    pilot run "${common[@]}" --rules default --write-solutions "$out/sol"
    pilot run "${common[@]}" --rules fullstrong vanillafullstrong
    pilot run "${common[@]}" --rules default fullstrong vanillafullstrong \
        --oracle "$out/sol"
done

for problem in dominating setcover; do
    pilot summarize "$RESULTS/$problem/results.csv"
done
