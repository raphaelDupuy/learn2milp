#!/usr/bin/env bash
# Drives the interactive menu with answers on its standard input: generate
# and solve an instance, compare rules on it, inspect its directory, exit.
set -euo pipefail

MENU="$1"
WORK_DIR="$2"
MODEL="$(cd "$(dirname "$0")/.." && pwd)/experiments/er125_p01/models/linear.model"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"  # the menu writes under data/ of the working directory

INSTANCE=data/instances/generated_and_solved/erdos_renyi_n40_p0_1_seed1.mis

# Settings files: one valid, one that SCIP would stop reading at line 1.
printf 'separating/zerohalf/freq = 1\n' >zerohalf.set
printf 'separating/misgate/separators = oddcycle\n' >unquoted.set

# The same model under the same file name in another directory.
mkdir models
cp "$MODEL" models/linear.model

# Answers:
# 4 generate and solve: Erdos-Renyi, n 40, p 0.1, seed 1; the expert
#   without pruning, study settings, an unquoted settings file (refused),
#   then zerohalf.set, default time limit, two invalid seeds, then 0;
# 5 compare relpscost and misfullstrong_nocutoff, without settings file
#   and with zerohalf.set, seeds 0 and 1;
# 5 compare the two models (after a rule given twice), with strong
#   branching on their first candidate;
# 6 inspect the instance directory; 7 exit.
printf '%s\n' \
    4 1 40 0.1 1 4 '' unquoted.set zerohalf.set '' abc -1 '' \
    5 "$INSTANCE" relpscost,misfullstrong_nocutoff '' '-, zerohalf.set' '' 0,1 \
    5 "$INSTANCE" 'relpscost, relpscost' "$MODEL, models/linear.model" 1 '' '' '' 0 \
    6 data/instances/generated_and_solved \
    7 |
    "$MENU" >stdout 2>&1

test -f "$INSTANCE"
grep -q 'Parameters: n = 40, p = 0.1$' stdout
grep -q 'did not apply line 1' stdout
grep -q 'Branching rule:    misfullstrong_nocutoff$' stdout
grep -q 'Settings:          study + zerohalf.set,' stdout
test "$(grep -c 'Invalid value: an integer from 0' stdout)" -eq 2
grep -q 'Maximum independent set: 16 vertices (optimal)' stdout

# 2 rules x 2 settings files x 2 seeds, labelled after the rule and file.
test "$(grep -cE '^(relpscost|misfullstrong_nocutoff)(\+zerohalf)? +[01] +OPTIMAL ' stdout)" -eq 8
grep -qE '^misfullstrong_nocutoff\+zerohalf +solved 2/2' stdout
grep -qE '^relpscost +solved 2/2' stdout

grep -q 'Invalid list: rule given twice: relpscost' stdout
grep -qE '^linear\+sb1 +0 +OPTIMAL ' stdout
grep -qE '^models/linear.model\+sb1 +0 +OPTIMAL ' stdout
! grep -q 'WARNING' stdout
grep -q 'Instances (.mis):  1 (erdos_renyi: 1)' stdout
grep -q 'Goodbye' stdout

# The end of the input leaves the menu cleanly.
printf '' | "$MENU" >stdout_eof 2>&1
grep -q 'Input interrupted' stdout_eof

echo "Terminal UI test passed."
