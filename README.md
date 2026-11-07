# Activation Monitoring for AI Control

DSC 180A capstone, group D33. Vipra Bindal, Kavyan Patel and Noah Golder.

We test whether a linear probe on an AI agent's internal activations can tell honest runs from
attack runs. The agent is Qwen3-Coder-30B working on LinuxArena tasks through Control Tower. In
attack runs, it also receives a hidden side task.

## Main result

On tasks it never saw in training, the probe separates honest and attack runs well, but it mostly
reads the difference between the two prompts rather than attack behavior. It is 100% accurate on
the first turn, before the agent has acted, and it flags 86% of turns in runs where the agent never
attacked. A separate before-and-after test also found no clear behavior signal. These results are
a baseline, not evidence that the probe detects attacks.

| Layer | Turn accuracy (pooled) | Turn accuracy (equal task weight) | AUROC | Runs correct | First-turn accuracy |
|---:|---:|---:|---:|---:|---:|
| 12 | 81% | 83% | 0.89 | 34/36 | 100% |
| 24 | 82% | 85% | 0.90 | 35/36 | 100% |
| 36 | 81% | 84% | 0.89 | 36/36 | 100% |
| 47 | 88% | 91% | 0.96 | 36/36 | 100% |

Other results at layer 47:

- **Flag rates:** the probe flags 86% of attack turns and 11% of honest turns.
- **Flags by behavior:** honest 6%, no attack 86%, wrong mechanism 100%, real attempt that failed
  83%, and side task succeeded 100%. These are means across runs.
- **Shuffled labels:** AUROC falls to 0.50 over 20 shuffles.
- **Before and after first mention of the side task:** a probe trained on this split reaches AUROC
  0.88 in attack runs and 0.81 at a matching fake point in honest runs. The confidence intervals
  overlap, so this mostly tracks progress through the run.
- **Task outcomes:** honest runs passed the main task in 6 of 18 runs. Attack runs passed the side
  task in 1 of 18 runs.

All numbers are in `results/metrics.json` and all figures are in `results/figures/`.

## Reproduce the results

This reruns the analysis on the activations we collected. It needs Python 3.12, about 5 GB of free
disk and no GPU. It was tested on macOS.

```bash
git clone https://github.com/K-Pat/Activation-Monitoring-D33.git
cd Activation-Monitoring-D33
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r probe/requirements.txt
python -m probe.run_all
```

The command overwrites `results/metrics.json`, `results/scores.npz` and `results/figures/`. After
the download, evaluation and figures take under a minute on a laptop. To rerun without downloading
again, use `python -m probe.run_all --skip-download`.

The metrics should match the committed `results/metrics.json`. Results may vary slightly on other
machines or library versions, and figure text falls back to DejaVu Sans if Arial is not installed.

### Data

`probe/extract.py` downloads the data automatically into `data/`, which git ignores. The data is
the public Hugging Face dataset
[`d33-activation-monitoring/agent-activations`](https://huggingface.co/datasets/d33-activation-monitoring/agent-activations),
pinned to commit `02ff64e9fb87a5089bc704dfa7acf994814f4715`. Only the 36 runs named
`T<task>_<honest|attack>_r<rep>` are used, about 4.2 GB in total.

```
runs/<run_id>/<eval_id>/manifest.json
runs/<run_id>/<eval_id>/log.eval
runs/<run_id>/<eval_id>/activations/<sample>_e1/turnNNNNN.safetensors
```

Each turn file holds one `[n_tokens, 2048]` array per layer. Row 0 is the last prompt token, and
later rows are the generated tokens. The dataset also holds early test files outside these 36 runs,
which the analysis skips.

### Pipeline

| File | What it does |
|---|---|
| `probe/extract.py` | Downloads the runs and saves `data/activations.npz` and `data/runs.json` |
| `probe/evaluate.py` | Trains and tests the probes and saves `results/metrics.json` and `results/scores.npz` |
| `probe/figures.py` | Draws the figures in `results/figures/` |
| `probe/behavior_labels.csv` | Behavior label for each attack run |
| `probe/run_all.py` | Runs all of the steps above |

## Method

- **Experiment:** 6 task pairs, each with 3 honest and 3 attack runs, for 36 runs and 1,231 agent
  turns (684 honest and 547 attack). See `pipeline/tasks.csv`.
- **Features:** the activation at the last prompt token of each turn, at layers 12, 24, 36 and 47.
  It is captured before the agent generates that turn.
- **Probe:** `StandardScaler` and `LogisticRegression(C=0.01, class_weight="balanced")`, with a
  decision threshold of 0.
- **Testing:** leave-one-task-out. Each task is held out once, so every score comes from a probe
  that never saw that task.
- **Aggregation:** pooled accuracy counts correct turns across all six rounds and divides by all
  turns, so tasks with more turns count more. Equal-weight accuracy averages the six task
  accuracies. A run is correct when its mean turn score falls on the correct side of 0.

## Limitations

- **Prompt confound:** attack prompts differ from honest prompts in role text, hidden-objective
  text, the side-task description and the closing instruction. T1 to T3 attack prompts also add a
  private tool. A probe can separate the conditions from these differences alone.
- **Behavior labels:** `probe/behavior_labels.csv` comes from one AI-assisted review of the
  transcripts and has not been independently checked. Figures 2 and 4 and the flag rates by
  behavior depend on it. The honest versus attack labels and the probe's accuracy do not.
- **First-mention labels:** these come from a keyword search of the agent's output, so they can
  miss a side task the agent pursues without naming it.
- **Sample size:** the experiment has 36 runs across 6 tasks. Turns within a run are closely
  related, so turn counts overstate the amount of independent data.
- **Figure 6:** this is a shortened, hand-written summary of the T5 prompts, not the full text.

## Collect the data again (optional)

This is not needed to reproduce the analysis. New runs will not match ours exactly because the
model samples its output. Collection also needs write access to our Hugging Face dataset and this
GitHub repo, or edits to `pipeline/config.env` that point the scripts at your own.

It requires:

- a Linux machine with Docker, at least 48 GB of GPU memory with FP8 support, and at least 150 GB of
  disk
- a Hugging Face token and a GitHub token

```bash
HF_TOKEN=... GH_TOKEN=... GIT_NAME="..." GIT_EMAIL=... ./setup_vast.sh
./run_next.sh --status
./run_next.sh --loop
```

`pipeline/config.env` pins the model, the Control Tower and LinuxArena commits and the kernels.
`pyproject.toml` pins the capture dependencies. `act_provider/` records the activations, and
`AGENTS.md` is the full collection runbook. `task_list.md` is the original experiment brief, and
parts of it are out of date. For example, the study later moved from Qwen3-8B to Qwen3-Coder-30B.
