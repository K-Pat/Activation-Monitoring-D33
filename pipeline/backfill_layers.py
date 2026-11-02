"""Add a layer to runs recorded before it was captured (e.g. layer 47, added 2026-11-02).

Each turn file keeps the full prompt (`prompt_ids`) and every generated token (`token_ids`, row 0
= last prompt token). Feeding prompt + generated tokens through the model once reproduces the
activations at those positions: the model is causal, so position k only sees tokens <= k, the same
inputs it had during the run. Values differ only by floating-point noise (batched vs one-at-a-time
processing), and rarely by an MoE routing flip that noise causes.

Before writing anything, each run is verified: the layers it already has are recomputed and
compared with the saved ones. A run is only updated if they match (mean cosine >= --min-cos at
row 0, >= --min-cos-gen over generated rows).
The check results go into the turn metadata and the run's manifest.

    uv run --project ~/control-tower python pipeline/backfill_layers.py --dry-run   # verify only
    uv run --project ~/control-tower python pipeline/backfill_layers.py             # verify + upload
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from act_provider.provider import _decoder_layers  # noqa: E402

REPO = os.environ.get("HF_REPO", "d33-activation-monitoring/agent-activations")


def run_dirs_on_hub(api: HfApi) -> list[str]:
    """runs/<run_id>/<eval_id> folders that have a manifest."""
    files = api.list_repo_files(REPO, repo_type="dataset")
    return sorted({f.rsplit("/", 1)[0] for f in files if f.startswith("runs/") and f.endswith("/manifest.json")})


def file_layers(path: str) -> list[int]:
    with safe_open(path, "pt") as h:
        return sorted(int(k[5:]) for k in h.keys() if k.startswith("layer"))


def recompute(model, blocks, prompt_ids: torch.Tensor, token_ids: torch.Tensor, layers: list[int]) -> dict:
    """Activations at the last prompt token and every generated token, for the given layers."""
    seq = torch.cat([prompt_ids, token_ids[1:]]).unsqueeze(0).to(model.device)
    start = len(prompt_ids) - 1
    got: dict[int, torch.Tensor] = {}

    def hook_for(i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            got[i] = h[0, start:].detach().to(torch.float16).cpu()
        return hook

    handles = [blocks[i].register_forward_hook(hook_for(i)) for i in layers]
    try:
        with torch.no_grad():
            model.model(input_ids=seq, use_cache=False)  # base model only: no 150k-wide logits
    finally:
        for h in handles:
            h.remove()
    return got


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a, b = a.float(), b.float()
    return (torch.nn.functional.cosine_similarity(a, b, dim=-1)).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", default="last", help="layer index to add, or 'last' (default)")
    p.add_argument("--runs", default="", help="comma-separated run ids (default: every run missing the layer)")
    p.add_argument("--min-cos", type=float, default=0.99,
                   help="minimum mean cosine at row 0 (last prompt token) to accept a run")
    p.add_argument("--min-cos-gen", type=float, default=0.98,
                   help="minimum mean cosine over generated-token rows to accept a run")
    p.add_argument("--dry-run", action="store_true", help="verify only, change nothing")
    p.add_argument("--mem-fraction", type=float, default=0.4,
                   help="cap on this process's share of GPU memory (keep room for a running collection loop)")
    p.add_argument("--work-dir", default=os.path.expanduser("~/d33-backfill"))
    args = p.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    api = HfApi()
    wanted = set(filter(None, args.runs.split(",")))
    dirs = [d for d in run_dirs_on_hub(api) if not wanted or d.split("/")[1] in wanted]

    model = None
    for d in dirs:
        run_id = d.split("/")[1]
        local = snapshot_download(REPO, repo_type="dataset", local_dir=args.work_dir,
                                  allow_patterns=[f"{d}/manifest.json", f"{d}/activations/*/*.safetensors"])
        man_path = os.path.join(local, d, "manifest.json")
        manifest = json.load(open(man_path))
        files = sorted(glob.glob(os.path.join(local, d, "activations/*/*.safetensors")))
        if not files:
            continue
        model_name = safe_open(files[0], "pt").metadata().get("model")
        if model is None:
            print(f"loading {model_name}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0").eval()
            blocks = _decoder_layers(model)
            target = len(blocks) - 1 if args.layer == "last" else int(args.layer)
        have = file_layers(files[0])
        if target in have:
            print(f"{run_id}: already has layer{target}, skipping")
            continue

        print(f"{run_id}: {len(files)} turns, has layers {have}, adding layer{target}", flush=True)
        new, cos_all = {}, {i: [] for i in have}
        for f in files:
            with safe_open(f, "pt") as h:
                saved = {k: h.get_tensor(k) for k in h.keys()}
                meta = dict(h.metadata() or {})
            got = recompute(model, blocks, saved["prompt_ids"], saved["token_ids"], have + [target])
            n = saved["token_ids"].shape[0]
            for i in have:
                cos_all[i].append(cosine_rows(got[i][:n], saved[f"layer{i}"][:n]))
            new[f] = (saved, meta, got[target][:n])

        def summary(c: np.ndarray) -> dict:
            return {"mean_cos": round(float(c.mean()), 5), "p01_cos": round(float(np.quantile(c, 0.01)), 5),
                    "min_cos": round(float(c.min()), 5), "rows": int(c.size)}

        stats = {}
        for i in have:
            stats[f"layer{i}"] = summary(np.concatenate(cos_all[i]))
            # row 0 (last prompt token, the pre-action state) was a prefill position during the run
            # too; generated rows were computed one token at a time, so they carry more noise
            stats[f"layer{i}_row0"] = summary(np.array([c[0] for c in cos_all[i]]))
        # row 0 must reproduce almost exactly. Generated rows drift more (decode-time noise plus the
        # odd MoE routing flip, worst at deeper layers), so they get a looser bar.
        ok = all(s["mean_cos"] >= (args.min_cos if k.endswith("_row0") else args.min_cos_gen)
                 for k, s in stats.items())
        print(f"  verification vs saved layers: {json.dumps(stats)}  -> {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok or args.dry_run:
            continue

        ops = []
        for f, (saved, meta, extra) in new.items():
            saved[f"layer{target}"] = extra.contiguous()
            meta["layers"] = ",".join(str(i) for i in sorted(have + [target]))
            meta["backfilled_layers"] = str(target)
            save_file({k: v.contiguous() for k, v in saved.items()}, f, metadata=meta)
            ops.append(CommitOperationAdd(os.path.relpath(f, local).replace(os.sep, "/"), f))
        manifest.setdefault("backfill", []).append(
            {"layer": target, "method": "single forward pass over prompt_ids + token_ids", "verification": stats})
        json.dump(manifest, open(man_path, "w"), indent=2)
        ops.append(CommitOperationAdd(f"{d}/manifest.json", man_path))
        api.create_commit(REPO, repo_type="dataset", operations=ops,
                          commit_message=f"Backfill layer{target} for {run_id}")
        print(f"  uploaded {len(ops)} files", flush=True)


if __name__ == "__main__":
    main()
