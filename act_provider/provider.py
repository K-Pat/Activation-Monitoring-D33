"""Inspect model provider that records residual-stream activations during generation.

Use as `hfact/<hf model id>`, e.g. `--model-role untrusted=hfact/Qwen/Qwen3-8B`.

Model args (Inspect `-M`):
    layers   comma-separated decoder layer indices to record (default: 1/4, 1/2, 3/4 depth)
    out_dir  where activation files go (default: ./activations)

Environment:
    ACT_THINKING=0   turn off Qwen3-style <think> blocks (passes enable_thinking=False to the
                     chat template). Unset leaves the model's default, which is thinking on.
    ACT_HF_REPO=<org>/<dataset>
                     when the eval finishes, upload its activations and eval log to this
                     Hugging Face dataset repo. Unset means files stay on local disk only.

Every model call (one agent turn) writes one file:
    <out_dir>/<eval_id>/<sample_id>_e<epoch>/turn<NNNNN>.safetensors

Each file holds, per recorded layer, `layer<i>` with shape [n_positions, d_model]. Row 0 is the
last prompt token, the point right before the model starts writing its turn. Rows 1.. are the
generated tokens, ending one token before the end of the turn. `token_ids` holds the token at
each row, and `prompt_ids` the full tokenized prompt.

Generation runs one call at a time so each forward pass belongs to exactly one sample.
"""

import asyncio
import json
import os
import re
from typing import Any

import torch
from inspect_ai.hooks import Hooks, TaskEnd, hooks
from inspect_ai.log._samples import sample_active
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    modelapi,
)
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model._call_tools import parse_tool_call
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util._json import JSON_SCHEMA_EXTENDED_FIELDS, json_schema_dump
from safetensors.torch import save_file
from shortuuid import uuid

# Qwen3-Coder writes tool calls as XML, which Inspect's HF provider doesn't parse:
#   <tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>
_XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _uses_xml_tools(model_name: str) -> bool:
    return "qwen3-coder" in model_name.lower()


def parse_xml_tool_calls(text: str, tools: list[ToolInfo]) -> tuple[str, list[Any]]:
    """Split a Qwen3-Coder response into (text before the first call, tool calls)."""
    calls = []
    for m in _XML_CALL.finditer(text):
        name, body = m.group(1), m.group(2)
        info = next((t for t in tools if t.name == name), None)
        props = info.parameters.properties if info else {}
        args: dict[str, Any] = {}
        for p in _XML_PARAM.finditer(body):
            key, val = p.group(1), p.group(2)
            ptype = getattr(props.get(key), "type", "string")
            if ptype not in ("string", None):
                try:
                    val = json.loads(val)
                except ValueError:
                    pass
            args[key] = val
        calls.append(parse_tool_call(f"{name}_{uuid()}", name, json.dumps(args), tools))
    first = _XML_CALL.search(text)
    content = (text[: first.start()] if first else text).strip()
    return content, calls


def render_native_chat(tokenizer: Any, messages: list[ChatMessage], tools: list[ToolInfo]) -> str:
    """Build the prompt with the model's own chat template, keeping tool calls structured.

    Inspect's HF provider rewrites past tool calls into the text as JSON for every Qwen model.
    Qwen3-Coder's template renders them in its own XML format, so pass them through instead.
    """
    hf_messages = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.text}
        if m.role == "assistant" and m.tool_calls:
            d["tool_calls"] = [
                {"type": "function", "function": {"name": tc.function, "arguments": tc.arguments}}
                for tc in m.tool_calls
            ]
        elif m.role == "tool" and getattr(m, "error", None) is not None:
            d["content"] = (m.text + "\n" if m.text else "") + f"Error: {m.error.message}"
        hf_messages.append(d)
    tools_list = [json_schema_dump(t, exclude=JSON_SCHEMA_EXTENDED_FIELDS) for t in tools]
    return tokenizer.apply_chat_template(
        hf_messages, tools=tools_list or None, add_generation_prompt=True, tokenize=False
    )


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


# every out_dir an hfact model has written to in this process, for the upload hook
_OUT_DIRS: set[str] = set()

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
        self.out_dir = os.path.abspath(str(model_args.pop("out_dir", "activations")))
        _OUT_DIRS.add(self.out_dir)
        # ct run eval can't pass -M args, so thinking can also be set from the environment
        thinking = os.environ.get("ACT_THINKING")
        if thinking is not None and "enable_thinking" not in model_args:
            model_args["enable_thinking"] = thinking.strip().lower() not in ("0", "false", "no", "off")
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
        if tools and _uses_xml_tools(self.model_name):
            self._parse_xml_tools(output, tools)
        return output

    def hf_chat(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> str:
        if _uses_xml_tools(self.model_name):
            return render_native_chat(self.tokenizer, messages, tools)
        return super().hf_chat(messages, tools)

    def _parse_xml_tools(self, output: ModelOutput, tools: list[ToolInfo]) -> None:
        for choice in output.choices:
            msg = choice.message
            if msg.tool_calls or "<function=" not in msg.text:
                continue
            content, calls = parse_xml_tool_calls(msg.text, tools)
            if calls:
                choice.message = ChatMessageAssistant(
                    content=content, tool_calls=calls, model=msg.model, source="generate"
                )
                choice.stop_reason = "tool_calls"

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


def _upload(repo: str, run: str, run_dirs: list[str], log_path: str | None) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = []
    for run_dir in run_dirs:
        for root, _, files in os.walk(run_dir):
            for f in files:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, run_dir).replace(os.sep, "/")
                ops.append(CommitOperationAdd(f"activations/{run}/{rel}", local))
    if log_path and os.path.isfile(log_path):
        ops.append(CommitOperationAdd(f"logs/{os.path.basename(log_path)}", log_path))
    else:
        print(f"[hfact] eval log not found at {log_path!r}, uploading activations only")

    HfApi().create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Add eval {run}",
    )
    print(f"[hfact] uploaded {len(ops)} files for eval {run} to {repo}")


@hooks(
    name="hfact_upload",
    description="Upload hfact activations and the eval log to a Hugging Face dataset repo",
)
class UploadToHF(Hooks):
    """Runs when a task finishes. Set ACT_HF_REPO=<org>/<dataset> to turn it on.

    Uploads <out_dir>/<eval_id>/ to activations/<eval_id>/ and the eval log to logs/ in one
    commit. If the upload fails, Inspect logs a warning and the files stay on disk.
    """

    def enabled(self) -> bool:
        return bool(os.environ.get("ACT_HF_REPO"))

    async def on_task_end(self, data: TaskEnd) -> None:
        run = _safe(data.eval_id)
        run_dirs = [os.path.join(d, run) for d in _OUT_DIRS if os.path.isdir(os.path.join(d, run))]
        if not run_dirs:
            return  # this eval didn't use hfact
        await asyncio.to_thread(
            _upload, os.environ["ACT_HF_REPO"], run, run_dirs, data.log.location or None
        )
