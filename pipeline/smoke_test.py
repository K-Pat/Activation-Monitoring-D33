"""Check a machine can collect data before it claims real runs.

Loads the configured model through hfact, gives it one toy tool, and checks that:
  1. the weights loaded in FP8 (not silently converted to 16-bit)
  2. the model's tool call was parsed and the tool actually ran
  3. one activation file was written per model call, with the expected layers and metadata
Nothing is uploaded. Takes a few minutes, mostly loading weights.

Run from Control Tower's environment:  uv run --project ~/control-tower python pipeline/smoke_test.py
"""

import glob
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("ACT_HF_REPO", None)  # never upload smoke-test data

from inspect_ai import Task, eval  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402
from inspect_ai.solver import generate, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402
from safetensors import safe_open  # noqa: E402

import act_provider  # noqa: E402,F401  registers hfact

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


@tool
def add():
    async def execute(x: int, y: int) -> str:
        """Add two integers.

        Args:
            x: first number
            y: second number
        """
        return str(x + y)

    return execute


def main() -> None:
    out = tempfile.mkdtemp(prefix="d33-smoke-")
    print(f"loading hfact/{MODEL} (activations to {out})")
    model = get_model(f"hfact/{MODEL}", out_dir=out)
    api = model.api

    fp8 = any("float8" in str(p.dtype) for p in api.model.parameters())
    check(fp8 or "FP8" not in MODEL, "weights loaded in FP8")

    task = Task(
        dataset=[Sample(input="Use the add tool to compute 1234 + 4321, then reply with the result.")],
        solver=[use_tools(add()), generate()],
    )
    log = eval(task, model=model, max_tokens=256, max_connections=1, display="none", log_dir=out)[0]
    check(log.status == "success", f"eval finished (status={log.status})")

    msgs = log.samples[0].messages if log.samples else []
    calls = [tc for m in msgs if m.role == "assistant" for tc in (m.tool_calls or [])]
    results = [m.text for m in msgs if m.role == "tool"]
    check(any(tc.function == "add" and not tc.parse_error for tc in calls), "tool call parsed")
    check(any("5555" in r for r in results), "tool ran and returned 5555")
    n_model_calls = sum(m.role == "assistant" for m in msgs)

    files = sorted(glob.glob(f"{out}/*/*/*.safetensors"))
    check(len(files) == n_model_calls and files, f"{len(files)} activation files for {n_model_calls} model calls")
    if files:
        with safe_open(files[0], "pt") as f:
            keys, meta = set(f.keys()), f.metadata() or {}
            layer_keys = sorted(k for k in keys if k.startswith("layer"))
            check(len(layer_keys) == len(api.layers), f"recorded layers {layer_keys}")
            check({"token_ids", "prompt_ids"} <= keys, "token_ids and prompt_ids saved")
            check(meta.get("n_tools") == "1" and meta.get("model") == MODEL, f"file metadata {meta}")
            shape = f.get_slice(layer_keys[0]).get_shape()
            check(shape[0] > 1, f"{layer_keys[0]} shape {shape}")

    print()
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} checks). Do not claim runs on this machine.")
        sys.exit(1)
    print("SMOKE TEST PASSED. This machine is ready: ./run_next.sh")


if __name__ == "__main__":
    main()
