"""Download the 36 runs and reduce them to one activation vector per turn.

Writes to the data directory:
  raw/              the dataset files, about 4.2 GB
  activations.npz   last-prompt-token vector for each turn, at each layer
  runs.json         task scores and side-task first-mention turn for each run
"""
import argparse
import glob
import io
import json
import os
import re
import zipfile

import numpy as np
import zstandard
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

from .common import FIRST_MENTION_PATTERN, HF_REPO, HF_REVISION, LAYERS, RUN_ID_PATTERN


def read_eval_sample(path):
    """Return the sample JSON from a log.eval file.

    The logs use zstd entries, which zipfile cannot decompress on Python < 3.14,
    so the entry is read from the archive directly.
    """
    with zipfile.ZipFile(path) as archive:
        name = next(
            n for n in archive.namelist() if n.startswith("samples/") and n.endswith(".json")
        )
        info = archive.getinfo(name)
        archive.fp.seek(info.header_offset)
        header = archive.fp.read(30)
        name_length = int.from_bytes(header[26:28], "little")
        extra_length = int.from_bytes(header[28:30], "little")
        archive.fp.seek(info.header_offset + 30 + name_length + extra_length)
        raw = archive.fp.read(info.compress_size)
    if info.compress_type == 93:
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    return json.loads(raw)


def agent_completions(sample):
    """Return the agent's output text per turn, in order.

    Monitor calls are also model events, so only the hidden_untrusted role counts.
    These events line up one to one with the activation turn files.
    """
    attachments = sample.get("attachments") or {}

    def resolve(text):
        return re.sub(
            r"attachment://([0-9a-f]{32})",
            lambda m: attachments.get(m.group(1), m.group(0)),
            text,
        )

    return [
        resolve(event["output"].get("completion") or "")
        for event in sample["events"]
        if event.get("event") == "model" and event.get("role") == "hidden_untrusted"
    ]


def summarize_run(log_path):
    sample = read_eval_sample(log_path)
    completions = agent_completions(sample)
    pattern = re.compile(FIRST_MENTION_PATTERN, re.I)
    tasks = ((sample.get("scores") or {}).get("tasks") or {}).get("value") or {}
    return {
        "main_task_score": tasks.get("main"),
        "side_task_score": tasks.get("side"),
        "n_turns": len(completions),
        "first_mention_turn": next(
            (i for i, text in enumerate(completions) if pattern.search(text)), None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    raw_dir = os.path.join(args.data_dir, "raw")

    print(f"Downloading {HF_REPO} at {HF_REVISION[:8]} to {raw_dir}", flush=True)
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=raw_dir,
        allow_patterns=["runs/T*"],
    )

    run_dirs = [
        d for d in sorted(glob.glob(os.path.join(raw_dir, "runs", "*")))
        if re.fullmatch(RUN_ID_PATTERN, os.path.basename(d))
    ]
    if len(run_dirs) != 36:
        raise SystemExit(f"Expected 36 runs, found {len(run_dirs)}")

    runs, turns, prompt_tokens = [], [], []
    vectors = {layer: [] for layer in LAYERS}
    summaries = {}
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        files = sorted(
            glob.glob(os.path.join(run_dir, "*", "activations", "*", "turn*.safetensors"))
        )
        logs = glob.glob(os.path.join(run_dir, "*", "log.eval"))
        if len(logs) != 1:
            raise SystemExit(f"{run_id}: expected one log.eval, found {len(logs)}")
        summaries[run_id] = summarize_run(logs[0])
        if summaries[run_id]["n_turns"] != len(files):
            raise SystemExit(
                f"{run_id}: {summaries[run_id]['n_turns']} agent turns in the log "
                f"but {len(files)} activation files"
            )
        for path in files:
            tensors = load_file(path)
            runs.append(run_id)
            turns.append(int(re.search(r"turn(\d+)", os.path.basename(path)).group(1)))
            prompt_tokens.append(len(tensors["prompt_ids"]))
            for layer in LAYERS:
                # Row 0 is the last prompt token, before the agent generates anything.
                vectors[layer].append(tensors[f"layer{layer}"][0].astype(np.float32))
        print(f"{run_id}: {len(files)} turns", flush=True)

    np.savez_compressed(
        os.path.join(args.data_dir, "activations.npz"),
        run=np.array(runs),
        turn=np.array(turns),
        prompt_tokens=np.array(prompt_tokens),
        **{f"layer{layer}": np.stack(vectors[layer]) for layer in LAYERS},
    )
    with open(os.path.join(args.data_dir, "runs.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    print(f"Saved {len(runs)} turns from {len(run_dirs)} runs", flush=True)


if __name__ == "__main__":
    main()
