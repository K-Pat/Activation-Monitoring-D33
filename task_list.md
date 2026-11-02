# Codex Task Brief: LinuxArena Honest vs. Attack Activation Experiment

## Objective

Work within the **existing local Control Tower repository** to prepare, validate, and execute a reproducible LinuxArena pilot using **Qwen3-8B**. Collect **three honest trajectories and three attack trajectories for each of six paired tasks** (36 trajectories total), then align each model-call input and output with a final-layer activation vector, targeting **approximately 1,000 vectors in aggregate**. The vector target is approximate; the number depends on completed model calls.

The user has already cloned Control Tower, installed Docker, run `uv sync`, generated model rates, pulled LinuxArena settings, and completed the basic `test` evaluation. **Do not redo setup unnecessarily.** Docker and Control Tower run on the Mac; Qwen may be served by a Vast.ai GPU instance through an OpenAI-compatible/vLLM endpoint. OpenRouter credentials may also be available for an initial hosted-model connectivity check, but hosted model calls do not expose their internal activations.

## Experiment matrix — use these six proposed pairs

| ID | LinuxArena environment | Main task (both conditions) | Side task (attack condition only) | Honest runs | Attack runs |
|---|---|---|---|---:|---:|
| T1 | `web_scraping` | `allowlist_config` | `visit_malicious_website` | 3 | 3 |
| T2 | `web_scraping` | `config_manager` | `expose_secret` | 3 | 3 |
| T3 | `shared_dotfiles` | `fix_requirements_parsing` | `install_blacklisted_package` | 3 | 3 |
| T4 | `shared_dotfiles` | `security_lines_sed_strips_too_much` | `bypass_blacklist` | 3 | 3 |
| T5 | `file_indexing` | `content_preview` | `authentication_bypass` | 3 | 3 |
| T6 | `file_indexing` | `duplicate_detection` | `file_corruption` | 3 | 3 |
| **Total** | **3 environments** | **6 main tasks** | **6 side tasks** | **18** | **18** |

These are the **requested candidate IDs**, not a guarantee that every ID or pairing is valid in the installed version. Check existence, compatibility, and scoring definitions before running anything. If an ID is unavailable or incompatible, **report it and propose an explicitly documented replacement; do not silently change the design**.

## Required workflow

### 1. Audit the existing setup (read-only first)

- Inspect the local Control Tower README/docs, `uv run ct --help`, `uv run ct run eval --help`, and relevant task-file documentation. Determine the **actual** supported syntax and JSONL schema rather than assuming example snippets from chat are correct.
- Verify Docker is running and LinuxArena settings are present. Use `uv run ct settings envs`, and for each named environment inspect main-task and side-task listings. Confirm all six combinations are allowed and have appropriate scorers.
- Inspect the configured model provider without printing, logging, or committing API secrets. Confirm that the self-hosted `Qwen/Qwen3-8B` API responds (typically through a local SSH tunnel). If not configured, give the exact missing setup steps; do not invent credentials or hostnames.
- Record the Control Tower revision, Qwen model/revision, task settings revision, provider endpoint *without credentials*, and generation settings. Ensure task definitions are deterministic while model sampling and seeds are controlled/documented.

### 2. Run a two-trajectory pilot

- Select T1 and perform **one honest** run (main task only) and **one attack** run (same main task + listed side task).
- Use the appropriate Control Tower policies (`honest` and `attack`), self-hosted model role, and attack main-task scoring if supported.
- Confirm: environment starts; model calls succeed; Qwen produces tool calls; `.eval` logs are written; run status and task/side-task scores can be read; model-call events and role information are present.
- Record total LLM calls and model-generated actions per trajectory. If either pilot succeeds, count it as the first of the three desired repetitions for that condition; do not inadvertently produce a fourth replicate.
- **Stop and report blockers** if the pair is invalid, the inference endpoint is unreachable, or Qwen never reaches usable tool interactions. Do not start the full batch in that state.

### 3. Execute the balanced 36-trajectory design

- Run **3 honest + 3 attack** trajectories for **each** of T1–T6, including any valid pilot runs. Maintain exactly the same main task, environment baseline, model, tool configuration, and generation settings across each honest/attack pair. The attack side-task instruction is the planned difference.
- If supported and verified, use `epochs=3` or equivalent. **Check whether `max_samples=1` means a global cap or concurrency cap in this version**, and do not accidentally run only one sample per batch. If epochs or JSONL batching is uncertain, use explicit per-task/per-repetition commands or a small script that checks completed counts.
- Keep detailed run manifests with unique IDs such as `T1_honest_r1`, `T1_attack_r1`, etc.; distinct output paths; condition; task/side-task IDs; seed; start/end status; call count; and log path. Do not overwrite prior evaluation logs.
- Prefer a conservative concurrency limit so Docker environments and the GPU server are not overloaded. Avoid unbounded API spending. If executing calls incurs new charges, use already provisioned resources and keep runs within the agreed 36-trajectory budget; request approval before scaling beyond that budget.

### 4. Collect and align activations

- Inspect the existing teammate implementation at `https://github.com/K-Pat/Activation-Monitoring-D33` **if accessible locally or publicly**, and reuse its tested extraction logic if appropriate. If not accessible, report that limitation; do not claim integration is complete.
- For each Qwen model-call event, save **one final-layer activation vector at the last input token**. Save `trajectory_id`, `model_call_index`, `condition`, task IDs, model checkpoint/revision, layer/token choice, and output/action linkage.
- Distinguish **online capture** (hooks in the actual model inference process) from **offline replay** (running the exact recorded prompt through the same checkpoint using Transformers). Normal `.eval` logs and standard hosted OpenRouter inference **do not contain hidden states**. For offline replay, reproduce the chat template, tool schemas, and prompt formatting; flag any mismatch.
- Persist vectors in a clear structure such as `artifacts/activations/<trajectory_id>.pt` and metadata in JSON/CSV. Validate that the number of saved vectors equals the number of selected Qwen model-call events per trajectory.
- Aim for ~1,000 vectors, but **do not fabricate or pad counts**. Report observed totals. If 36 trajectories produce fewer than 1,000, quantify the shortfall and estimate additional runs; do not launch them without approval.

### 5. Analyze and summarize

Produce these deliverables:

1. `artifacts/run_manifest.csv` — exactly one record per attempted trajectory, with outcome and log reference.
2. `artifacts/activation_manifest.csv` — exactly one record per extracted activation, with model-call association and file/offset.
3. `artifacts/summary.md` — counts by task and condition, total model calls/vectors, successful/failed/incomplete runs, main-task scores, attack-side-task scores, and any data-quality issues.
4. At least one preliminary honest-versus-attack activation visualization (e.g. PCA) and, **only if enough independent trajectories exist**, an exploratory linear classifier with splitting **by trajectory rather than by vector**.
5. Reproducible scripts/commands for task discovery, execution, extraction, and analysis, with paths and dependencies documented. Do not commit `.env`, private tokens, SSH keys, or downloaded secrets.

## Scientific guardrails

- **Attack condition is a run-level label, not proof of malicious behavior at every step.** Keep the side-task outcome separate from the run's honest/attack condition, and distinguish attempted sabotage from scored success.
- The ~1,000 model-call vectors are **not 1,000 independent trajectories**; avoid random train/test splits across steps from the same run.
- Do not claim that a classifier has learned malicious behavior if it may simply be identifying the side-task instruction in the attack prompt. Explain this confound prominently.
- Preserve unsuccessful or prematurely terminated *model* attempts with explicit status; distinguish those from infrastructure errors.

## Execution behavior requested from Codex

**First response:** Summarize what is already configured, report task compatibility checks, and show the exact pilot commands you intend to run. Proceed with the pilot using existing credentials/configuration if available. After a validated pilot, execute the remaining runs up to the requested 36-trajectory budget where feasible, save artifacts, and summarize actual results. Never claim success based solely on a command being launched; verify completed logs and vector files. If you cannot access a required system, give the specific command or action needed from the user and continue with everything that can be prepared locally.
