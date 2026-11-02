#!/usr/bin/env bash
# Run one trajectory from pipeline/tasks.csv, e.g.  ./run_task.sh T1_attack_r1
#
# Every run gets its own log folder and its own empty Inspect cache. Control Tower skips tasks
# that already have a successful log in the log folder, and it replays cached model responses
# for identical prompts, which would record no activations. Both would silently waste a run.
#
# Exit codes: 0 finished and uploaded | 10 eval did not finish (infra error, cancelled)
#             11 finished but upload failed (data is on local disk, see message) | 2 usage
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/pipeline/config.env"
export PATH="$HOME/.local/bin:$PATH"

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "usage: ./run_task.sh <run_id>   (ids are in pipeline/tasks.csv)" >&2
  exit 2
fi
ROW="$(grep -E "^${RUN_ID}," "$REPO_DIR/pipeline/tasks.csv" || true)"
if [[ -z "$ROW" ]]; then
  echo "unknown run id: $RUN_ID" >&2
  exit 2
fi
IFS=, read -r _ TASK_ID CONDITION REP ENV MAIN_TASK SIDE_TASK <<<"$ROW"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="logs/${RUN_ID}-${TS}"
CACHE_DIR="/tmp/inspect-cache-${RUN_ID}-${TS}"

CT_SHA="$(git -C "$CT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
SETTINGS_SHA="$(git -C "$CT_DIR/.settings/$SETTINGS_NAME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
PROVIDER_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
OPERATOR="${D33_OPERATOR:-$(git -C "$REPO_DIR" config user.name || echo unknown)}"

export ACT_HF_REPO="$HF_REPO"
export ACT_RUN_ID="$RUN_ID"
export ACT_THINKING
export INSPECT_CACHE_DIR="$CACHE_DIR"
export ACT_RUN_META="{\"run_id\":\"$RUN_ID\",\"task_id\":\"$TASK_ID\",\"condition\":\"$CONDITION\",\"rep\":$REP,\"env\":\"$ENV\",\"main_task\":\"$MAIN_TASK\",\"side_task\":\"$SIDE_TASK\",\"model\":\"$MODEL\",\"act_thinking\":\"$ACT_THINKING\",\"operator\":\"$OPERATOR\",\"host\":\"$(hostname)\",\"started_utc\":\"$TS\",\"control_tower\":\"$CT_SHA\",\"settings\":\"$SETTINGS_SHA\",\"provider\":\"$PROVIDER_SHA\",\"log_dir\":\"$LOG_DIR\"}"

ARGS=(--policy "$CONDITION" -e "linuxarena/$ENV" -t "$MAIN_TASK")
[[ "$CONDITION" == "attack" ]] && ARGS+=(-s "$SIDE_TASK")
ARGS+=(--model-role "untrusted=hfact/$MODEL" --log-dir "$LOG_DIR" --run-name "$RUN_ID" --tag d33)

echo "=== $RUN_ID: $CONDITION | $ENV / $MAIN_TASK ${SIDE_TASK:+| side: $SIDE_TASK}"
echo "    model $MODEL | log $CT_DIR/$LOG_DIR | started $TS"

cd "$CT_DIR" || exit 2
uv run ct run eval "${ARGS[@]}"
CT_EXIT=$?

LOG_PATH="$(ls -t "$LOG_DIR"/*.eval 2>/dev/null | head -1)"
if [[ -z "$LOG_PATH" ]]; then
  echo "!!! $RUN_ID: no eval log was written (ct exited $CT_EXIT). Nothing to upload." >&2
  exit 10
fi
read -r STATUS EVAL_ID < <(uv run python -c "
from inspect_ai.log import read_eval_log
l = read_eval_log('$LOG_PATH', header_only=True)
print(l.status, l.eval.eval_id)" 2>/dev/null)

echo "=== $RUN_ID finished: status=$STATUS eval_id=$EVAL_ID log=$CT_DIR/$LOG_PATH"

export HF_REPO CT_DIR
TRACKER=(uv run python "$REPO_DIR/pipeline/tracker.py")
# check exits 0 = uploaded and finished, 2 = uploaded but not finished, 1 = nothing uploaded
"${TRACKER[@]}" check "$RUN_ID" --eval-id "$EVAL_ID" >/dev/null 2>&1
if [[ $? -eq 1 ]]; then
  echo "    upload not found on HF, retrying from local files"
  if ! "${TRACKER[@]}" reupload "$RUN_ID" --log "$LOG_PATH"; then
    echo "!!! $RUN_ID: upload failed. The data is safe on disk. Retry later with:" >&2
    echo "    cd $REPO_DIR && ./run_next.sh --reupload $RUN_ID $CT_DIR/$LOG_PATH" >&2
    exit 11
  fi
fi
"${TRACKER[@]}" check "$RUN_ID" --eval-id "$EVAL_ID" || true

[[ "$STATUS" == "success" ]] && exit 0 || exit 10
