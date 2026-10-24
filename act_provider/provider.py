"""Inspect model provider that records residual-stream activations during generation.

Use as `hfact/<hf model id>`, e.g. `--model-role untrusted=hfact/Qwen/Qwen3-8B`.

Model args (Inspect `-M`):
    layers   comma-separated decoder layer indices to record (default: 1/4, 1/2, 3/4 depth)
    out_dir  where activation files go (default: ./activations)

Every model call (one agent turn) writes one file:
    <out_dir>/<eval_id>/<sample_id>_e<epoch>/turn<NNNNN>.safetensors

Each file holds, per recorded layer, `layer<i>` with shape [n_positions, d_model]. Row 0 is the
last prompt token, the point right before the model starts writing its turn. Rows 1.. are the
generated tokens, ending one token before the end of the turn. `token_ids` holds the token at
each row, and `prompt_ids` the full tokenized prompt.

Generation runs one call at a time so each forward pass belongs to exactly one sample.
"""

import asyncio
import os
import re
from typing import Any

import torch
from inspect_ai.log._samples import sample_active
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    modelapi,
)
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.tool import ToolChoice, ToolInfo
from safetensors.torch import save_file


def _decoder_layers(model: Any) -> Any:
    # Llama/Qwen/Mistral-style models keep their decoder blocks at model.model.layers
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"can't find decoder layers on {type(model).__name__}")


def _safe(name: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


class ActivationHFAPI(HuggingFaceAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        layers = model_args.pop("layers", None)
        self.out_dir = str(model_args.pop("out_dir", "activations"))
        super().__init__(model_name, base_url, api_key, config, **model_args)

        blocks = _decoder_layers(self.model)
        if layers is None:
            n = len(blocks)
            self.layers = sorted({n // 4, n // 2, (3 * n) // 4})
        elif isinstance(layers, int):
            self.layers = [layers]
        else:
            self.layers = [int(x) for x in str(layers).split(",")]

        self._lock = asyncio.Lock()
        self._acts: dict[int, list[torch.Tensor]] = {}
        self._ids: list[torch.Tensor] = []
        self._prompt_ids = torch.empty(0, dtype=torch.long)
        self._turns: dict[str, int] = {}

        self.model.register_forward_pre_hook(self._record_ids, with_kwargs=True)
        for i in self.layers:
            blocks[i].register_forward_hook(self._record_layer(i))

    def _reset(self) -> None:
        self._acts = {i: [] for i in self.layers}
        self._ids = []
        self._prompt_ids = torch.empty(0, dtype=torch.long)

    def _record_ids(self, module: Any, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return
        if not self._ids:
            # prefill pass sees the whole prompt
            self._prompt_ids = ids[0].cpu()
        self._ids.append(ids[0, -1:].cpu())

    def _record_layer(self, i: int) -> Any:
        def hook(module: Any, args: tuple, output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            self._acts[i].append(h[0, -1:].detach().to(torch.float16).cpu())

        return hook

    def max_connections(self) -> int:
        # one sample per forward pass, so hook outputs can't mix samples
        return 1

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if config.max_connections is not None and config.max_connections > 1:
            raise ValueError("hfact needs max_connections=1 to keep samples apart")

        async with self._lock:
            self._reset()
            output = await super().generate(input, tools, tool_choice, config)
            self._save()
        return output

    def _save(self) -> None:
        active = sample_active()
        if active is not None:
            run = _safe(active.eval_id)
            sample = f"{_safe(active.sample.id)}_e{active.epoch}"
        else:
            run, sample = "no_eval", "no_sample"
        key = f"{run}/{sample}"
        turn = self._turns.get(key, 0)
        self._turns[key] = turn + 1

        # decode steps run one past the last kept token, so trim every list to the same length
        n = min([len(self._ids)] + [len(v) for v in self._acts.values()])
        tensors = {
            f"layer{i}": torch.cat(self._acts[i][:n]).contiguous() for i in self.layers
        }
        tensors["token_ids"] = torch.cat(self._ids[:n]).contiguous()
        tensors["prompt_ids"] = self._prompt_ids.contiguous()

        path = os.path.join(self.out_dir, run, sample)
        os.makedirs(path, exist_ok=True)
        save_file(tensors, os.path.join(path, f"turn{turn:05d}.safetensors"))


@modelapi(name="hfact")
def hfact() -> type[ActivationHFAPI]:
    return ActivationHFAPI
