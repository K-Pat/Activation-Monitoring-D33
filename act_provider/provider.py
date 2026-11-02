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
    ACT_MAX_TOKENS=<n>
                     max tokens the model can write per turn (default 8192, Inspect's is 2048)
    ACT_RUN_ID=<id>  file the upload under runs/<id>/<eval_id>/ with a manifest.json
    ACT_RUN_META=<json>
                     extra fields (task, condition, versions) copied into the manifest

Each turn file's safetensors metadata records the model, the recorded layers, the run ID and
`n_tools`. A turn with n_tools=0 is a side call such as the attack policy's private planner,
not an action the agent took.

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

    def max_tokens(self) -> int | None:
        # Inspect's HF default is 2048, which truncates large file edits mid tool call, so the
        # call can't be parsed and the agent retries until Control Tower stops it for no progress
        return int(os.environ.get("ACT_MAX_TOKENS", "8192"))

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
            # calls with no tools are side calls, e.g. the attack policy's private planner
            self._save(n_tools=len(tools))
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

    def _save(self, n_tools: int) -> None:
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
        meta = {
            "model": self.model_name,
            "n_tools": str(n_tools),
            "layers": ",".join(str(i) for i in self.layers),
            "run_id": os.environ.get("ACT_RUN_ID", ""),
        }
        save_file(tensors, os.path.join(path, f"turn{turn:05d}.safetensors"), metadata=meta)


@modelapi(name="hfact")
def hfact() -> type[ActivationHFAPI]:
    return ActivationHFAPI


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def build_manifest(log: Any, run_id: str | None, meta: dict[str, Any], n_turns: int) -> dict:
    """One record per attempted trajectory: what ran, how it ended, how it scored."""
    if not log.samples and log.location:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(log.location)  # eval sets can hand hooks a header-only log
    roles = {k: v.model for k, v in (log.eval.model_roles or {}).items()}
    samples = []
    for s in log.samples or []:
        samples.append(
            {
                "id": str(s.id),
                "epoch": s.epoch,
                "scores": {k: _jsonable(v.value) for k, v in (s.scores or {}).items()},
                "error": s.error.message[:2000] if s.error else None,
                "limit": _jsonable(s.limit.model_dump()) if s.limit else None,
            }
        )
    return {
        "run_id": run_id,
        "eval_id": log.eval.eval_id,
        "status": log.status,  # success = the eval finished, whatever the task score
        "error": log.error.message[:2000] if log.error else None,
        "model_roles": roles,
        "started_at": log.stats.started_at if log.stats else None,
        "completed_at": log.stats.completed_at if log.stats else None,
        "n_turns": n_turns,
        "samples": samples,
        "meta": meta,
    }


def upload_eval(
    repo: str,
    eval_id: str,
    run_dirs: list[str],
    log_path: str | None,
    manifest: dict,
    run_id: str | None = None,
) -> None:
    """Upload one eval's activations, log and manifest to the dataset in a single commit.

    With a run ID:  runs/<run_id>/<eval_id>/{manifest.json, log.eval, activations/...}
    Without one:    activations/<eval_id>/... and logs/<log file>
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    base = f"runs/{run_id}/{eval_id}" if run_id else None
    ops = []
    for run_dir in run_dirs:
        for root, _, files in os.walk(run_dir):
            for f in files:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, run_dir).replace(os.sep, "/")
                dest = f"{base}/activations/{rel}" if base else f"activations/{eval_id}/{rel}"
                ops.append(CommitOperationAdd(dest, local))
    if log_path and os.path.isfile(log_path):
        dest = f"{base}/log.eval" if base else f"logs/{os.path.basename(log_path)}"
        ops.append(CommitOperationAdd(dest, log_path))
    else:
        print(f"[hfact] eval log not found at {log_path!r}, uploading without it")
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    for run_dir in run_dirs:  # keep a local copy so a failed upload can be retried
        with open(os.path.join(run_dir, "manifest.json"), "wb") as fh:
            fh.write(manifest_bytes)
    if base:
        ops.append(CommitOperationAdd(f"{base}/manifest.json", manifest_bytes))

    HfApi().create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Add {run_id or 'eval'} ({eval_id}, {manifest['status']})",
    )
    print(f"[hfact] uploaded {len(ops)} files for {run_id or 'eval'} {eval_id} to {repo}")


def find_run_dirs(eval_id: str, out_dirs: set[str] | None = None) -> list[str]:
    run = _safe(eval_id)
    dirs = out_dirs if out_dirs is not None else _OUT_DIRS
    return [os.path.join(d, run) for d in dirs if os.path.isdir(os.path.join(d, run))]


def count_turns(run_dirs: list[str]) -> int:
    return sum(
        f.endswith(".safetensors") for d in run_dirs for _, _, files in os.walk(d) for f in files
    )


@hooks(
    name="hfact_upload",
    description="Upload hfact activations and the eval log to a Hugging Face dataset repo",
)
class UploadToHF(Hooks):
    """Runs when a task finishes. Set ACT_HF_REPO=<org>/<dataset> to turn it on.

    ACT_RUN_ID (e.g. T1_attack_r2) files the upload under runs/<run_id>/, and ACT_RUN_META
    (a JSON object) is copied into the manifest. If the upload fails, Inspect logs a warning
    and the files stay on disk with a local manifest.json.
    """

    def enabled(self) -> bool:
        return bool(os.environ.get("ACT_HF_REPO"))

    async def on_task_end(self, data: TaskEnd) -> None:
        run_dirs = find_run_dirs(data.eval_id)
        if not run_dirs:
            return  # this eval didn't use hfact
        run_id = os.environ.get("ACT_RUN_ID") or None
        meta = json.loads(os.environ.get("ACT_RUN_META") or "{}")
        manifest = await asyncio.to_thread(
            build_manifest, data.log, run_id, meta, count_turns(run_dirs)
        )
        await asyncio.to_thread(
            upload_eval,
            os.environ["ACT_HF_REPO"],
            _safe(data.eval_id),
            run_dirs,
            data.log.location or None,
            manifest,
            run_id,
        )
