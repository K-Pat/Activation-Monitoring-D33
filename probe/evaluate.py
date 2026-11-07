"""Train and test the probes, and write every reported number to results/metrics.json.

The probe predicts whether a turn comes from an honest or an attack run. Each of the
6 tasks is held out once, so every score comes from a probe that never saw that task.
"""
import argparse
import csv
import json
import os
import statistics

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import HEADLINE_LAYER, LAYERS, leave_one_task_out

N_SHUFFLES = 20
N_BOOTSTRAP = 1000
LABELS_PATH = os.path.join(os.path.dirname(__file__), "behavior_labels.csv")


def load_labels():
    with open(LABELS_PATH, newline="") as f:
        return {row["run_id"]: row["behavior"] for row in csv.DictReader(f)}


def accuracy(scores, labels):
    return float(((scores > 0) == labels).mean())


def condition_probe_metrics(scores, y, run, task, position, first_turn_scores, labels, onset):
    tasks = sorted(set(task))
    run_ids = sorted(set(run))
    attack = y == 1
    first = position == 0
    per_task = [accuracy(scores[task == t], y[task == t]) for t in tasks]
    run_mean = {r: float(scores[run == r].mean()) for r in run_ids}
    run_flag = {r: float((scores[run == r] > 0).mean()) for r in run_ids}
    honest_runs = [r for r in run_ids if "honest" in r]
    attack_runs = [r for r in run_ids if "attack" in r]

    after_mention = np.array([position[i] >= onset[r] for i, r in enumerate(run) if "attack" in r])
    attack_scores = scores[attack]

    length = {r: int((run == r).sum()) for r in run_ids}
    progress = np.array([position[i] / max(1, length[r] - 1) for i, r in enumerate(run)])
    bins = np.minimum((progress * 10).astype(int), 9)

    def flag_curve(names):
        return [
            float(np.mean([
                (scores[(run == r) & (bins == b)] > 0).mean()
                for r in names
                if ((run == r) & (bins == b)).any()
            ]))
            for b in range(10)
        ]

    return {
        "turn_accuracy_pooled": accuracy(scores, y),
        "turn_accuracy_equal_task_weight": float(np.mean(per_task)),
        "auroc": float(roc_auc_score(y, scores)),
        "per_task_accuracy": dict(zip(tasks, per_task)),
        "per_task_turns": {t: int((task == t).sum()) for t in tasks},
        "attack_turns_flagged": float((scores[attack] > 0).mean()),
        "honest_turns_flagged": float((scores[~attack] > 0).mean()),
        "mean_honest_run_flag_rate": float(np.mean([run_flag[r] for r in honest_runs])),
        "run_accuracy": float(np.mean([(run_mean[r] > 0) == (r in attack_runs) for r in run_ids])),
        "min_attack_run_mean_score": min(run_mean[r] for r in attack_runs),
        "max_honest_run_mean_score": max(run_mean[r] for r in honest_runs),
        "run_flag_rate": run_flag,
        "flag_rate_by_behavior": {
            b: float(np.mean([run_flag[r] for r in labels if labels[r] == b]))
            for b in sorted(set(labels.values()))
        },
        "attack_flagged_before_first_mention": float((attack_scores[~after_mention] > 0).mean()),
        "attack_flagged_after_first_mention": float((attack_scores[after_mention] > 0).mean()),
        "attack_turns_before_first_mention": int((~after_mention).sum()),
        "attack_turns_after_first_mention": int(after_mention.sum()),
        # Trained on all turns, then read on first turns only.
        "first_turn_accuracy_all_turn_probe": accuracy(scores[first], y[first]),
        # Trained and tested on first turns only, 30 training turns per round.
        "first_turn_accuracy_first_turn_probe": accuracy(first_turn_scores[first], y[first]),
        "flag_curve_attack": flag_curve(attack_runs),
        "flag_curve_honest": flag_curve(honest_runs),
    }


def bootstrap_auroc(labels, scores, run, mask):
    """95% interval, resampling whole runs because turns in a run are near-duplicates."""
    random = np.random.RandomState(0)
    run_ids = sorted(set(run[mask]))
    rows = {r: np.where(run == r)[0] for r in run_ids}
    values = []
    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate([rows[r] for r in random.choice(run_ids, len(run_ids))])
        if 0 < labels[sample].sum() < len(sample):
            values.append(roc_auc_score(labels[sample], scores[sample]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def before_after_control(X, run, task, position, y, onset):
    """Can a probe tell turns after the first mention from turns before it?

    Honest runs get a fake onset at the same point in the run as their paired attack
    run. Similar scores on both mean the probe is tracking progress through the run.
    """
    after = np.array([position[i] >= onset[r] for i, r in enumerate(run)])
    attack, honest = y == 1, y == 0
    attack_scores = leave_one_task_out(X, after, task, attack)
    honest_scores = leave_one_task_out(X, after, task, honest)
    return {
        "attack_auroc": float(roc_auc_score(after[attack], attack_scores[attack])),
        "attack_auroc_95ci": bootstrap_auroc(after, attack_scores, run, attack),
        "honest_fake_onset_auroc": float(roc_auc_score(after[honest], honest_scores[honest])),
        "honest_fake_onset_auroc_95ci": bootstrap_auroc(after, honest_scores, run, honest),
    }


def shuffled_label_control(X, run, task):
    """Reassign honest and attack among each task's 6 runs, then rerun the probe."""
    random = np.random.RandomState(0)
    run_ids = sorted(set(run))
    values = []
    for _ in range(N_SHUFFLES):
        assignment = {}
        for t in sorted(set(task)):
            names = [r for r in run_ids if r[:2] == t]
            shuffled = [0, 0, 0, 1, 1, 1]
            random.shuffle(shuffled)
            assignment.update(zip(names, shuffled))
        labels = np.array([assignment[r] for r in run])
        values.append(float(roc_auc_score(labels, leave_one_task_out(X, labels, task))))
    return {"mean": float(np.mean(values)), "min": min(values), "max": max(values), "values": values}


def pca_separation(X, y):
    """Best AUROC of each of the top 2 principal components, in either direction."""
    pca = PCA(2, random_state=0)
    components = pca.fit_transform(StandardScaler().fit_transform(X))
    aurocs = [roc_auc_score(y, components[:, i]) for i in range(2)]
    return {
        "pc_auroc": [float(max(a, 1 - a)) for a in aurocs],
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    data = np.load(os.path.join(args.data_dir, "activations.npz"))
    with open(os.path.join(args.data_dir, "runs.json")) as f:
        runs = json.load(f)
    labels = load_labels()

    run, turn = data["run"], data["turn"]
    y = np.array([int("_attack_" in r) for r in run])
    task = np.array([r[:2] for r in run])
    position = np.array([turn[i] - turn[run == r].min() for i, r in enumerate(run)])

    # Attack runs use their first mention. Honest runs get the same fraction of the run.
    fraction = {
        r: runs[r]["first_mention_turn"] / runs[r]["n_turns"] for r in runs if "attack" in r
    }
    onset = {r: runs[r]["first_mention_turn"] for r in fraction}
    for r, value in fraction.items():
        honest = r.replace("attack", "honest")
        onset[honest] = int(round(value * runs[honest]["n_turns"]))

    first_turn = position == 0
    tokens = data["prompt_tokens"]
    metrics = {
        "n_turns": len(run),
        "n_runs": len(runs),
        "n_honest_turns": int((y == 0).sum()),
        "n_attack_turns": int((y == 1).sum()),
        "median_first_mention_fraction": statistics.median(fraction.values()),
        "min_first_mention_fraction": min(fraction.values()),
        "max_first_mention_fraction": max(fraction.values()),
        "first_mention_fraction": fraction,
        "first_turn_prompt_tokens": {
            t: {
                "honest": int(tokens[(run == f"{t}_honest_r1") & first_turn][0]),
                "attack": int(tokens[(run == f"{t}_attack_r1") & first_turn][0]),
            }
            for t in sorted(set(task))
        },
        "main_task_score": {r: runs[r]["main_task_score"] for r in runs},
        "side_task_score": {r: runs[r]["side_task_score"] for r in runs},
        "behavior_labels": labels,
        "layers": {},
    }

    scores = {}
    for layer in LAYERS:
        X = data[f"layer{layer}"]
        scores[layer] = leave_one_task_out(X, y, task)
        first_turn_scores = leave_one_task_out(X, y, task, first_turn)
        metrics["layers"][str(layer)] = {
            **condition_probe_metrics(
                scores[layer], y, run, task, position, first_turn_scores, labels, onset
            ),
            "before_after_mention": before_after_control(X, run, task, position, y, onset),
            "pca": pca_separation(X, y),
        }
        summary = metrics["layers"][str(layer)]
        print(
            f"layer {layer}: accuracy {summary['turn_accuracy_pooled']:.3f} "
            f"AUROC {summary['auroc']:.3f} runs {summary['run_accuracy']:.3f}",
            flush=True,
        )
    metrics["shuffled_label_auroc"] = shuffled_label_control(
        data[f"layer{HEADLINE_LAYER}"], run, task
    )

    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    np.savez(
        os.path.join(args.results_dir, "scores.npz"),
        run=run,
        position=position,
        y=y,
        task=task,
        **{f"layer{layer}": scores[layer] for layer in LAYERS},
    )
    print(f"Wrote {args.results_dir}/metrics.json and scores.npz", flush=True)


if __name__ == "__main__":
    main()
