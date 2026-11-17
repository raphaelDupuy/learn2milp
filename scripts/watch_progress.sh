#!/usr/bin/env bash
#
# Follows the progress log of a long experiment: every line written so
# far, then each new one as it comes (one line per finished run).
#
# Usage, from the repository root:
#   scripts/watch_progress.sh [LOG]     (default: data/logs/generalization.log)
#
# From WSL, in a new Windows Terminal window:
#   wt.exe -w new new-tab --title Learn2MILP wsl.exe -d "$WSL_DISTRO_NAME" \
#       --cd "$PWD" -- scripts/watch_progress.sh
set -u

LOG="${1:-data/logs/generalization.log}"

printf 'Learn2MILP: following %s (Ctrl+C to close)\n\n' "$LOG"

# -F waits for the file if it does not exist yet, and follows it if it is
# recreated.
exec tail -n +1 -F "$LOG"
