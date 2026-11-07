## AI use

We used Claude Code as a coding assistant. We chose the experiment design, task matrix and probe-testing direction, and we questioned its results as we went.

### Experiment runbook and multi-agent collection

AI helped write the setup script, the run tracker and the AGENTS.md runbook. Any agent following the runbook can set up a machine, claim the next unfinished run through claims.jsonl, run it and upload the results, so several machines can work without repeating a run. In practice, runs didn't overlap: 35 ran on our DGX Spark and 1 on a Vast.ai VM.

### Probe implementation

AI helped write the logistic regression probe using our specific instructions, the leave-one-task-out testing and the control checks. It later probe/ pipeline andchecked that a fresh download reproduced the results.

### Figures

AI helped write the plotting code and revised the figures after our feedback.

### Prompts

The prompts we use for these parts are listed below, cleaned up for grammar and style.

#### Runbook

> Write a runbook and scripts that let agents on separate GPU machines collect the 36 runs in pipeline/tasks.csv. Pin the model and software versions in one config file. Use a claim file in Git so two machines can't take the same run, and expire claims from dead machines. Upload each finished run to Hugging Face. Document what a healthy run looks like, what failures look like and what the agent should report. Never store tokens in the repo.

#### Probe

> Train a logistic regression probe to separate honest and attack runs, using the last-prompt-token activations at layers 12, 24, 36 and 47. Hold out one task at a time, and never split a run across train and test. Report pooled and equal-task-weight accuracy, AUROC and run-level accuracy. Add controls for shuffled labels, first-turn accuracy and flag rates on attack runs where the agent never attacked, so we can tell whether the probe reads behavior or the prompt.

#### Figures

> Create slide figures from results/metrics.json, with one claim per figure. Show task outcomes, accuracy by layer, flag rates by behavior, flag rates over a run and run-level separation. Show individual runs as well as averages. Don't let any title claim attack detection unless the prompt-confound controls support it.

> Can we do one of [the] separability [figures], or one figure that shows a positive result [from the probe]?
