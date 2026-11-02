"""Measure how peak GPU memory grows with prompt length, to find what makes long episodes OOM.

Loads the model the way Inspect's HF provider does, then for each setting and prompt length runs
one short generate() and records torch's peak allocated memory. Run on an otherwise idle GPU:
    uv run --project ~/control-tower python pipeline/profile_memory.py [--lengths 4000,8000,16000]
"""

import argparse
import gc
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
GB = 1024**3


def peak_for(model, tok, n_tokens: int, **gen_kwargs) -> tuple[float, float, float]:
    ids = torch.randint(1000, 50000, (1, n_tokens), device=model.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.time()
    with torch.no_grad():
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False,
                       pad_token_id=tok.eos_token_id, **gen_kwargs)
    torch.cuda.synchronize()
    return (
        (torch.cuda.max_memory_allocated() - base) / GB,
        torch.cuda.memory_reserved() / GB,
        time.time() - t,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", default="2000,4000,8000,16000,24000")
    p.add_argument("--attn", default="sdpa", help="attention implementations to compare, e.g. sdpa,eager")
    p.add_argument("--mem-fraction", type=float, default=0.6,
                   help="cap torch at this share of GPU memory so an overshoot raises OOM instead of "
                        "freezing a unified-memory machine like the DGX Spark")
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    lengths = [int(x) for x in args.lengths.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    for attn in args.attn.split(","):
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", attn_implementation=attn)
        model.eval()
        w = torch.cuda.memory_allocated() / GB
        print(f"\n== attn={attn} (resolved {model.config._attn_implementation}), weights {w:.1f} GB")
        print(f"{'tokens':>7} {'extra GB':>9} {'GB/1k tok':>9} {'reserved':>9} {'seconds':>8}")
        for n in lengths:
            try:
                extra, reserved, secs = peak_for(model, tok, n)
                print(f"{n:>7} {extra:>9.1f} {extra / n * 1000:>9.2f} {reserved:>9.1f} {secs:>8.1f}", flush=True)
            except torch.OutOfMemoryError:
                print(f"{n:>7}  out of memory")
                break
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
