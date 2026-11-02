"""Coordinate the 36 runs in tasks.csv across several machines.

Done comes from the Hugging Face dataset: a run is done when some attempt uploaded a
runs/<run_id>/<eval_id>/manifest.json with status "success" (the eval finished, whatever the
task score). Claimed comes from claims.jsonl in this repo, pushed to GitHub, so two machines
can't take the same run. A push that loses a race pulls and picks again.

Run it through the shell wrappers, which load pipeline/config.env and Control Tower's venv:
    ./run_next.sh --status | --dry-run
Direct use: python pipeline/tracker.py {status,next,claim-next,claim,release,check,reupload}
"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "pipeline" / "tasks.csv"
CLAIMS = ROOT / "claims.jsonl"
HF_REPO = os.environ.get("HF_REPO", "d33-activation-monitoring/agent-activations")
TTL_S = float(os.environ.get("CLAIM_TTL_HOURS", "6")) * 3600


def load_tasks() -> list[dict]:
    with open(TASKS) as fh:
        return list(csv.DictReader(fh))


def task_row(run_id: str) -> dict:
    for row in load_tasks():
        if row["run_id"] == run_id:
            return row
    sys.exit(f"unknown run id {run_id!r}, see pipeline/tasks.csv")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=check)


def who() -> str:
    name = os.environ.get("D33_OPERATOR") or git("config", "user.name", check=False).stdout.strip()
    return name or "unknown"


def now() -> float:
    return time.time()


def fmt_age(ts: float) -> str:
    m = int((now() - ts) // 60)
    return f"{m // 60}h{m % 60:02d}m ago" if m >= 60 else f"{m}m ago"


def hf_manifests() -> dict[str, list[dict]]:
    """run_id -> manifests of every uploaded attempt."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi()
    out: dict[str, list[dict]] = {}
    try:
        entries = api.list_repo_tree(HF_REPO, repo_type="dataset", path_in_repo="runs", recursive=True)
        paths = [e.path for e in entries if e.path.endswith("/manifest.json")]
    except (EntryNotFoundError, RepositoryNotFoundError):
        return out
    except Exception as ex:  # the folder doesn't exist until the first run uploads
        if "404" in str(ex) or "not found" in str(ex).lower():
            return out
        raise
    for p in paths:
        local = hf_hub_download(HF_REPO, p, repo_type="dataset")
        with open(local) as fh:
            out.setdefault(p.split("/")[1], []).append(json.load(fh))
    return out


def claim_events() -> list[dict]:
    if not CLAIMS.exists():
        return []
    return [json.loads(line) for line in CLAIMS.read_text().splitlines() if line.strip()]


def active_claims(events: list[dict]) -> dict[str, dict]:
    last: dict[str, dict] = {}
    for e in events:
        last[e["run_id"]] = e
    return {
        rid: e for rid, e in last.items() if e["event"] == "claim" and now() - e["at"] < TTL_S
    }


def is_done(manifests: list[dict]) -> bool:
    return any(m.get("status") == "success" for m in manifests)


def score_summary(m: dict) -> str:
    parts = []
    for s in m.get("samples", []):
        tasks = s.get("scores", {}).get("tasks")
        if isinstance(tasks, dict):
            parts.append(f"main={tasks.get('main')} side={tasks.get('side')}")
        if s.get("error"):
            parts.append("sample error")
    return ", ".join(parts) or "no scores"


def snapshot() -> tuple[list[dict], dict[str, list[dict]], dict[str, dict]]:
    """Pull first so other machines' claims are visible. A stale copy could never claim a run
    twice (GitHub rejects a push from a copy that's behind), but it would show wrong status."""
    pull = git("pull", "--rebase", "--autostash", "-q", check=False)
    if pull.returncode != 0:
        sys.exit(f"git pull failed, so other machines' claims can't be seen:\n{pull.stderr.strip()}")
    return load_tasks(), hf_manifests(), active_claims(claim_events())


def pick_next(tasks, manifests, claims, want: str | None = None) -> str | None:
    for row in tasks:
        rid = row["run_id"]
        if want and rid != want:
            continue
        if is_done(manifests.get(rid, [])) or rid in claims:
            if want:
                state = "done" if is_done(manifests.get(rid, [])) else f"claimed by {claims[rid]['who']}"
                sys.exit(f"{rid} is {state}")
            continue
        return rid
    return None


def append_event(run_id: str, event: str, **extra) -> None:
    rec = {"run_id": run_id, "event": event, "who": who(), "host": socket.gethostname(), "at": now()}
    rec.update(extra)
    with open(CLAIMS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    git("add", "claims.jsonl")
    git("commit", "-q", "-m", f"{event} {run_id} ({rec['who']}@{rec['host']})", "--", "claims.jsonl")


def push_event(run_id: str | None, event: str, choose=None, **extra) -> str | None:
    """Append an event and push it, retrying on a lost race. `choose` re-picks the run."""
    for _ in range(6):
        tasks, manifests, claims = snapshot()
        rid = choose(tasks, manifests, claims) if choose else run_id
        if rid is None:
            return None
        append_event(rid, event, **extra)
        if git("push", "-q", "origin", "HEAD", check=False).returncode == 0:
            return rid
        git("reset", "-q", "--keep", "HEAD~1", check=False)  # drop our commit, keep other edits
        time.sleep(2)
    sys.exit("could not push to GitHub after 6 tries: check network and repo write access")


def cmd_status(args) -> None:
    tasks, manifests, claims = snapshot()
    counts = {"done": 0, "claimed": 0, "todo": 0}
    if args.json:
        print(json.dumps({"manifests": manifests, "claims": claims}, indent=2, default=str))
        return
    print(f"{'run_id':<16} {'state':<9} detail")
    for row in tasks:
        rid = row["run_id"]
        ms = manifests.get(rid, [])
        failed = sum(m.get("status") != "success" for m in ms)
        tail = f" ({failed} failed attempt{'s' * (failed != 1)})" if failed else ""
        if is_done(ms):
            m = next(m for m in ms if m.get("status") == "success")
            state, detail = "done", f"{m.get('n_turns')} turns, {score_summary(m)}"
            counts["done"] += 1
        elif rid in claims:
            c = claims[rid]
            state, detail = "claimed", f"{c['who']}@{c['host']}, {fmt_age(c['at'])}"
            counts["claimed"] += 1
        else:
            state, detail = "todo", ""
            counts["todo"] += 1
        print(f"{rid:<16} {state:<9} {detail}{tail}")
    print(f"\n{counts['done']} done, {counts['claimed']} claimed, {counts['todo']} to do, of {len(tasks)}")


def cmd_next(args) -> None:
    rid = pick_next(*snapshot())
    print(rid or "")
    if rid is None:
        sys.exit(3)


def cmd_claim_next(args) -> None:
    rid = push_event(None, "claim", choose=pick_next)
    print(rid or "")
    if rid is None:
        sys.exit(3)


def cmd_claim(args) -> None:
    rid = push_event(None, "claim", choose=lambda t, m, c: pick_next(t, m, c, want=args.run_id))
    print(rid)


def cmd_release(args) -> None:
    push_event(args.run_id, "release", reason=args.reason)
    print(f"released {args.run_id}")


def cmd_check(args) -> None:
    ms = hf_manifests().get(args.run_id, [])
    ok = [m for m in ms if not args.eval_id or m.get("eval_id") == args.eval_id]
    if not ok:
        print(f"{args.run_id}: no upload found on {HF_REPO}")
        sys.exit(1)
    for m in ok:
        print(f"{args.run_id}: eval {m['eval_id']} status={m['status']} turns={m['n_turns']} {score_summary(m)}")
    sys.exit(0 if is_done(ok) else 2)


def cmd_reupload(args) -> None:
    """Retry an upload that failed at the end of a run, from the local files."""
    sys.path.insert(0, str(ROOT))
    from inspect_ai.log import read_eval_log

    from act_provider.provider import build_manifest, count_turns, find_run_dirs, upload_eval

    log = read_eval_log(args.log)
    act_root = os.path.join(os.environ.get("CT_DIR", str(Path.home() / "control-tower")), "activations")
    run_dirs = find_run_dirs(log.eval.eval_id, {act_root})
    if not run_dirs:
        sys.exit(f"no activations for eval {log.eval.eval_id} under {act_root}")
    meta = {**task_row(args.run_id), "reuploaded_at": datetime.now(timezone.utc).isoformat()}
    manifest = build_manifest(log, args.run_id, meta, count_turns(run_dirs))
    upload_eval(HF_REPO, log.eval.eval_id, run_dirs, args.log, manifest, args.run_id)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="table of done / claimed / to do")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("next", help="print the run that would be claimed next").set_defaults(fn=cmd_next)
    sub.add_parser("claim-next", help="claim the next free run and print its id").set_defaults(fn=cmd_claim_next)
    s = sub.add_parser("claim", help="claim a specific run")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_claim)
    s = sub.add_parser("release", help="give a claimed run back")
    s.add_argument("run_id")
    s.add_argument("--reason", default="")
    s.set_defaults(fn=cmd_release)
    s = sub.add_parser("check", help="exit 0 if the run has a successful upload")
    s.add_argument("run_id")
    s.add_argument("--eval-id")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("reupload", help="retry a failed upload from local files")
    s.add_argument("run_id")
    s.add_argument("--log", required=True, help="path to the run's .eval log")
    s.set_defaults(fn=cmd_reupload)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
