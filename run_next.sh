#!/usr/bin/env bash
# Claim the next free run from pipeline/tasks.csv and execute it.
#
#   ./run_next.sh                one run, then stop
#   ./run_next.sh --loop         keep going until all runs are done or claimed
#   ./run_next.sh --loop --max 4 stop after 4 runs
#   ./run_next.sh --run T3_attack_r2   claim and run one specific id
#   ./run_next.sh --status       table of done / claimed / to do (no claiming)
#   ./run_next.sh --dry-run      show which run would be claimed next (no claiming)
#   ./run_next.sh --release ID   give back a run you claimed but won't finish
#   ./run_next.sh --reupload ID LOG   retry a failed upload from local files
#
# Runs start inside tmux session "d33" so they survive SSH disconnects. Watch with
# `tmux attach -t d33`, detach with Ctrl+b then d. Add --no-tmux to run in the foreground.
# To stop a --loop after the current run finishes: touch ~/d33.stop
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/pipeline/config.env"
export PATH="$HOME/.local/bin:$PATH"
export HF_REPO CT_DIR CLAIM_TTL_HOURS

tracker() { uv run --quiet --project "$CT_DIR" python "$REPO_DIR/pipeline/tracker.py" "$@"; }

MODE=once MAX=0 TMUX_OK=1 WANT=""
ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status) tracker status; exit $? ;;
    --dry-run) tracker next; exit $? ;;
    --release) tracker release "${2:?run id}" --reason "manual release"; exit $? ;;
    --reupload) tracker reupload "${2:?run id}" --log "${3:?path to .eval log}"; exit $? ;;
    --loop) MODE=loop ;;
    --max) MAX="${2:?number}"; shift ;;
    --run) WANT="${2:?run id}"; shift ;;
    --no-tmux) TMUX_OK=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [[ $TMUX_OK -eq 1 && -z "${TMUX:-}" ]]; then
  if tmux has-session -t d33 2>/dev/null; then
    echo "tmux session d33 already exists: something may be running. Check with: tmux attach -t d33" >&2
    exit 1
  fi
  tmux new -d -s d33 -c "$REPO_DIR" \
    "./run_next.sh --no-tmux ${ORIG_ARGS[*]} 2>&1 | tee -a $HOME/d33-runs.log; echo; echo '[run_next finished, press Enter to close]'; read"
  echo "started in tmux session d33 (log: ~/d33-runs.log)"
  echo "  watch:  tmux attach -t d33      detach: Ctrl+b then d"
  echo "  stop after the current run:  touch ~/d33.stop"
  exit 0
fi

rm -f "$HOME/d33.stop"
CURRENT=""
on_interrupt() {
  if [[ -n "$CURRENT" ]]; then
    echo; echo "interrupted, releasing $CURRENT so someone else can run it"
    tracker release "$CURRENT" --reason "interrupted"
  fi
  exit 130
}
trap on_interrupt INT TERM

DONE=0 INFRA_FAILS=0
while true; do
  if [[ -n "$WANT" ]]; then
    RID="$(tracker claim "$WANT")" || exit 1
  else
    RID="$(tracker claim-next)"
    if [[ $? -eq 3 || -z "$RID" ]]; then
      echo "nothing left to claim: every run is done or claimed by someone"
      break
    fi
  fi
  CURRENT="$RID"
  echo; echo "### $(date '+%F %T') claimed $RID"

  "$REPO_DIR/run_task.sh" "$RID"
  CODE=$?
  CURRENT=""
  case $CODE in
    0)  DONE=$((DONE + 1)); INFRA_FAILS=0
        echo "### $RID done ($DONE this session)" ;;
    10) INFRA_FAILS=$((INFRA_FAILS + 1))
        echo "### $RID did not finish. Releasing it so it can be retried. Its partial upload is kept as a failed attempt."
        tracker release "$RID" --reason "eval did not finish" ;;
    11) echo "### $RID finished but the upload failed. Keeping the claim. Fix HF access, then run the --reupload command above."
        exit 11 ;;
    *)  echo "### run_task.sh exited $CODE for $RID. Releasing it and stopping."
        tracker release "$RID" --reason "run_task exit $CODE"
        exit "$CODE" ;;
  esac

  if [[ $INFRA_FAILS -ge 2 ]]; then
    echo "### two runs in a row did not finish. Stopping so a broken machine doesn't burn through tasks."
    echo "    Look at the last log in $CT_DIR/logs/ before starting again."
    exit 10
  fi
  [[ "$MODE" == loop && -z "$WANT" ]] || break
  [[ $MAX -gt 0 && $DONE -ge $MAX ]] && { echo "### reached --max $MAX"; break; }
  [[ -f "$HOME/d33.stop" ]] && { echo "### found ~/d33.stop, stopping"; rm -f "$HOME/d33.stop"; break; }
done
