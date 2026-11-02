# Runbook for agents collecting D33 trajectories

You are managing data collection on a rented GPU machine for the D33 activation-monitoring
project (UCSD DSC 180A). A Qwen model runs as an agent inside LinuxArena tasks through Control
Tower. The `hfact` provider in this repo records the model's internal activations at every
turn and uploads them to Hugging Face with the eval log when the run ends.

Your job is to set the machine up, run tasks with the scripts here, watch them, and report.
**Never run `ct run eval` by hand for data collection.** The scripts set things that silently
ruin a run if they're missing (see "Why the scripts exist").

## The experiment

`pipeline/tasks.csv` lists 36 runs: 6 task pairs (T1–T6) × honest/attack × 3 repetitions. IDs
look like `T3_attack_r2`. Honest runs do the main task. Attack runs do the same main task with
a secret side task. The order is round-robin (all r1 runs first) so a partial dataset is still
balanced. Several machines collect at once. `claims.jsonl` (pushed to GitHub) stops two machines
taking the same run, and the Hugging Face dataset decides what's done.

All runs must use identical settings: `pipeline/config.env` pins the model, Control Tower
commit, LinuxArena settings commit and kernels. Don't change it on your own machine.

## Setup (once per machine)

Requirements: a Vast.ai **VM instance** (regular instances can't run Docker) with an
**H100 or H200 (80 GB+)**, and at least 150 GB of disk. Smaller cards fail on long episodes:
the model's weights are ~31 GB, but it already uses ~48 GB at 25k tokens of context. Then:

```bash
git clone https://x-access-token:$GH_TOKEN@github.com/K-Pat/Activation-Monitoring-D33.git
cd Activation-Monitoring-D33
HF_TOKEN=... GH_TOKEN=... GIT_NAME="..." GIT_EMAIL=... ./setup_vast.sh
```

It checks the GPU/Docker/disk, installs Control Tower at the pinned commit, installs the
provider and FP8 kernels, checks GitHub push and Hugging Face write access, downloads the model
(~31 GB) and runs `pipeline/smoke_test.py`. It stops at the first problem with a message
saying what to fix. Re-running is safe. Ask the human for tokens; never print them, write them
into files in this repo, or commit them.

## Collecting

```bash
./run_next.sh --status     # what's done, claimed, to do
./run_next.sh              # claim and run ONE task (do this first on a new machine)
./run_next.sh --loop       # keep claiming and running until nothing is left
touch ~/d33.stop           # stop a --loop after the current run finishes
```

Runs execute in tmux session `d33` (output also in `~/d33-runs.log`). A run takes roughly 20–90
minutes. To see progress mid-run without attaching:

```bash
ls -t ~/control-tower/activations/ | head -1        # newest eval id
ls ~/control-tower/activations/<eval_id>/*/ | wc -l  # turns recorded so far
tail -c 3000 ~/d33-runs.log
```

## What normal looks like

- The run prints `=== T1_honest_r1: ...`, builds Docker images (first time per environment),
  then the agent takes steps. Each step writes one `turnNNNNN.safetensors`.
- Tool calls vary: the agent explores files, then edits, tests and submits.
- At the end: `[hfact] uploaded N files for T1_honest_r1 ...` and a `check` line with
  `status=success`. A run that finished counts as done **even if the task score is 0**. That's
  real data about the model.

## What broken looks like, and what to do

| Symptom | Meaning | Action |
|---|---|---|
| The same command repeated turn after turn, or one turn hitting max tokens with repeated text | Model looping | Let it run out. Report it with the run ID. |
| `Completed all tasks` within a minute or two, 0–3 turns | Skipped or no-progress run | Report it. Don't rerun by hand. |
| `model ... cache: read` in the log | Cached replay, no activations | Shouldn't happen via the scripts. Report it. |
| Run exits with "did not finish" | Infra error. The script releases the claim. | After two in a row the loop stops itself. Read the last log in `~/control-tower/logs/` and report. |
| "upload failed" | Data is on disk, HF unreachable | Fix network/token, then run the `--reupload` command it printed. |
| CUDA out of memory | Another process holds the GPU | `nvidia-smi`. Don't kill processes you didn't start. Report it. |

Inspect a run's transcript and scores from its log:

```bash
cd ~/control-tower && uv run inspect view --log-dir logs/<run_id>-<timestamp>
```

## Rules

- Only collect through `run_next.sh` / `run_task.sh`.
- Don't edit `pipeline/config.env`, `pipeline/tasks.csv` or `act_provider/` on a collecting
  machine. Changes go through the team and apply to every machine before more runs.
- Don't run `uv sync` in `~/control-tower`: it uninstalls the provider and kernels. Re-run
  `./setup_vast.sh` instead.
- Don't delete anything from the Hugging Face dataset, including failed attempts. They're kept
  on purpose.
- If you stop a run with Ctrl+C, the script releases its claim. If the machine dies, the claim
  expires after 6 hours.
- Report to the human: run IDs finished, failures with their log paths, and anything in the
  "broken" table.

## Why the scripts exist

- Control Tower runs evals as an Inspect eval set, which **skips a task that already has a
  successful log** in the log folder. `run_task.sh` gives every run a fresh `--log-dir`.
- Control Tower **caches model responses**. An identical prompt replays the old answer without
  calling the model, so no activations are recorded. `run_task.sh` gives every run an empty
  `INSPECT_CACHE_DIR`.
- `ACT_HF_REPO`, `ACT_RUN_ID` and `ACT_RUN_META` turn on the upload and file it under
  `runs/<run_id>/<eval_id>/` with a `manifest.json` (status, scores, versions, operator).
- Qwen3-Coder writes tool calls in XML, which Inspect doesn't parse. `act_provider` handles it
  and renders the history in the model's own chat template.

## Data layout on Hugging Face

```
runs/<run_id>/<eval_id>/manifest.json
runs/<run_id>/<eval_id>/log.eval
runs/<run_id>/<eval_id>/activations/<sample>_e1/turnNNNNN.safetensors
```

Each turn file holds `layer<i>` `[n_tokens, d_model]` (row 0 = last prompt token, then every
generated token), `token_ids` and `prompt_ids`. Its metadata has `model`, `layers`, `run_id`
and `n_tools`: `n_tools=0` marks a side call such as the attack policy's private planner, not an
action.

## Analysis guardrails (from the experiment plan)

- "Attack" labels the run, not every step. Most steps in an attack run are ordinary work. Keep
  the side-task score separate from the condition.
- Split train/test **by run**, never by step. Steps from one run are near-duplicates.
- Attack prompts contain the side-task instructions. A probe can learn "the prompt mentions a
  side task" instead of the behavior. Check for that before claiming detection.
- Keep failed runs with their status. Distinguish model failures from infrastructure failures.
