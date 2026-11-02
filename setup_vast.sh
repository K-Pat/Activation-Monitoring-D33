#!/usr/bin/env bash
# One-time setup of a fresh GPU machine (a Vast.ai VM instance) for collecting runs.
#
#   HF_TOKEN=hf_... GH_TOKEN=github_pat_... GIT_NAME="Your Name" GIT_EMAIL=you@example.com \
#     ./setup_vast.sh [--start] [--skip-smoke-test]
#
# HF_TOKEN   your Hugging Face token with write access to the d33-activation-monitoring org
# GH_TOKEN   GitHub token that can push to K-Pat/Activation-Monitoring-D33 (for run claims)
# GIT_NAME / GIT_EMAIL   who the claim commits are attributed to
# --start    after a passing smoke test, start ./run_next.sh --loop
#
# Safe to re-run: finished steps are skipped or redone harmlessly. It stops at the first
# problem with a message saying what to fix.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/pipeline/config.env"
export PATH="$HOME/.local/bin:$PATH"

START=0 SMOKE=1
for a in "$@"; do
  case "$a" in
    --start) START=1 ;;
    --skip-smoke-test) SMOKE=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

step() { echo; echo "==> $*"; }
die() { echo; echo "SETUP STOPPED: $*" >&2; exit 1; }
SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO="sudo"

# ---------------------------------------------------------------------------
step "1/9 Checking inputs"
: "${HF_TOKEN:?set HF_TOKEN (Hugging Face token with write access to $HF_REPO)}"
: "${GH_TOKEN:?set GH_TOKEN (GitHub token that can push to this repo)}"
: "${GIT_NAME:?set GIT_NAME (your name, used on claim commits)}"
: "${GIT_EMAIL:?set GIT_EMAIL}"
echo "model $MODEL | dataset $HF_REPO | control tower $CT_DIR"

# ---------------------------------------------------------------------------
step "2/9 Preflight: GPU, Docker, disk"
command -v nvidia-smi >/dev/null || die "no nvidia-smi. Rent a GPU machine with NVIDIA drivers."
# Vast's VM image ships driver 535: too old for the pinned torch (needs >= 580) and for Blackwell
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
if [[ -z "$DRIVER_MAJOR" || "$DRIVER_MAJOR" -lt 580 ]]; then
  echo "NVIDIA driver ${DRIVER_MAJOR:-not working} is too old: installing driver 580 (needs one reboot)"
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-headers-$(uname -r)" >/dev/null
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq "nvidia-*535*" "libnvidia-*535*" >/dev/null 2>&1 || true
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-580 >/dev/null \
    || die "driver install failed. On Blackwell GPUs (RTX PRO 6000, B200) try nvidia-driver-580-open."
  die "driver 580 installed. Reboot (sudo reboot), reconnect, and run ./setup_vast.sh again."
fi
GPU="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1)"
IFS=, read -r GPU_NAME GPU_MB GPU_CC <<<"$GPU"
GPU_GB=$(( ${GPU_MB// /} / 1024 ))
GPU_CC="${GPU_CC// /}"
echo "GPU:${GPU_NAME} ${GPU_GB} GB, compute capability ${GPU_CC}"
CC_OK=$(awk -v c="$GPU_CC" 'BEGIN{print (c >= 8.9) ? 1 : 0}')
if [[ "$MODEL" == *FP8* && $CC_OK -eq 0 ]]; then
  die "${GPU_NAME} is too old for FP8 (needs compute capability >= 8.9). Rent an L40S, RTX 6000 Ada, H100 or H200."
elif [[ $GPU_GB -lt $MIN_GPU_GB ]]; then
  die "GPU has ${GPU_GB} GB, need at least 48 GB (weights ~29 GB plus ~0.23 GB per 1k tokens of context). Rent an L40S 48GB, H100 or H200."
fi
command -v docker >/dev/null || die "no docker. On Vast, rent a VM instance (filter vms_enabled=true): regular instances can't run Docker."
$SUDO docker info >/dev/null 2>&1 || die "docker is installed but not running or not accessible. Try: sudo systemctl start docker"
if [[ $(id -u) -ne 0 ]] && ! docker info >/dev/null 2>&1; then
  die "your user can't run docker without sudo. Run: sudo usermod -aG docker \$USER, then log out and back in."
fi
# Ubuntu's docker.io package lacks the compose and buildx plugins that LinuxArena needs
if ! docker compose version >/dev/null 2>&1 || ! docker buildx version >/dev/null 2>&1; then
  echo "installing Docker's compose and buildx plugins"
  $SUDO install -m 0755 -d /etc/apt/keyrings
  $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-plugin docker-buildx-plugin >/dev/null \
    || die "couldn't install docker compose/buildx plugins"
fi
echo "docker $(docker version --format '{{.Server.Version}}'), $(docker compose version --short) compose"
DISK_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
echo "free disk: ${DISK_GB} GB"
[[ $DISK_GB -ge $MIN_DISK_GB ]] || die "need at least ${MIN_DISK_GB} GB free disk (model ~31 GB, Docker images, activations)."

# ---------------------------------------------------------------------------
step "3/9 System tools"
NEED=()
for t in git tmux curl; do command -v "$t" >/dev/null || NEED+=("$t"); done
if [[ ${#NEED[@]} -gt 0 ]]; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${NEED[@]}" ca-certificates
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "git $(git --version | cut -d' ' -f3), tmux $(tmux -V | cut -d' ' -f2), uv $(uv --version | cut -d' ' -f2)"

# ---------------------------------------------------------------------------
step "4/9 GitHub access (for run claims)"
git config --global credential.helper store
umask 077
printf 'https://x-access-token:%s@github.com\n' "$GH_TOKEN" > "$HOME/.git-credentials"
umask 022
git -C "$REPO_DIR" config user.name "$GIT_NAME"
git -C "$REPO_DIR" config user.email "$GIT_EMAIL"
git -C "$REPO_DIR" pull -q --rebase --autostash || die "can't pull this repo. Check GH_TOKEN."
git -C "$REPO_DIR" push --dry-run -q origin HEAD 2>/dev/null \
  || die "GH_TOKEN can't push to this repo. Ask the repo owner for write access, or make a token with Contents: read and write."
echo "can pull and push $(git -C "$REPO_DIR" remote get-url origin | sed 's#https://[^@]*@#https://#')"

# ---------------------------------------------------------------------------
step "5/9 Control Tower at $CT_COMMIT"
if [[ ! -d "$CT_DIR/.git" ]]; then
  git clone -q "$CT_REPO" "$CT_DIR"
fi
git -C "$CT_DIR" fetch -q origin
git -C "$CT_DIR" -c advice.detachedHead=false checkout -q "$CT_COMMIT" \
  || die "can't check out Control Tower $CT_COMMIT. Uncommitted changes in $CT_DIR?"
[[ -f "$CT_DIR/.env" ]] || cp "$CT_DIR/.env.example" "$CT_DIR/.env"
(cd "$CT_DIR" && uv sync -q)
(cd "$CT_DIR" && uv run ct settings pull "$SETTINGS_NAME@$SETTINGS_REF") \
  || die "couldn't pull the $SETTINGS_NAME settings at $SETTINGS_REF"
# ct run eval refuses to start without a model price table
if [[ ! -s "$HOME/.cache/control-tower/model_rates.json" ]]; then
  (cd "$CT_DIR" && uv run python scripts/generate_model_rates.py >/dev/null) || die "couldn't generate model rates"
fi
GOT="$(git -C "$CT_DIR/.settings/$SETTINGS_NAME" rev-parse HEAD)"
[[ "$GOT" == "$SETTINGS_REF" ]] || die "settings are at $GOT, expected $SETTINGS_REF"
echo "Control Tower $(git -C "$CT_DIR" rev-parse --short HEAD), settings $(git -C "$CT_DIR/.settings/$SETTINGS_NAME" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
step "6/9 Activation provider and FP8 kernels"
# after `uv sync` (which removes packages it doesn't know about), never before
(cd "$CT_DIR" && uv pip install -q -e "$REPO_DIR" "$KERNELS_SPEC")
(cd "$CT_DIR" && uv run python -c "
import act_provider, kernels, torch, transformers
print('act_provider ok | kernels', kernels.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'torch cannot see the GPU'
") || die "provider install check failed (see error above)"

# ---------------------------------------------------------------------------
step "7/9 Hugging Face login"
(cd "$CT_DIR" && uv run hf auth login --token "$HF_TOKEN" >/dev/null 2>&1) || die "hf login failed. Check HF_TOKEN."
(cd "$CT_DIR" && HF_REPO="$HF_REPO" uv run python -c "
import os
from huggingface_hub import HfApi
api = HfApi(); me = api.whoami()
org = os.environ['HF_REPO'].split('/')[0]
orgs = [o['name'] for o in me.get('orgs', [])]
role = me.get('auth', {}).get('accessToken', {}).get('role')
print('logged in as', me['name'], '| token role', role, '| orgs', orgs)
assert org in orgs, f'not a member of {org}: ask to be invited on huggingface.co/{org}'
assert role in ('write', 'fineGrained'), 'token is read-only: make a write token'
api.dataset_info(os.environ['HF_REPO'])
print('can see', os.environ['HF_REPO'])
") || die "Hugging Face access check failed (see message above)"

# ---------------------------------------------------------------------------
step "8/9 Downloading $MODEL (~31 GB, skipped if cached)"
(cd "$CT_DIR" && uv run hf download "$MODEL" >/dev/null) || die "model download failed"
echo "model cached"
(cd "$CT_DIR" && SKIP_DOCKER_BUILD=1 uv run ct settings pull-images --only web_scraping --only shared_dotfiles --only file_indexing) \
  || echo "  (pre-pulling Docker images failed. Not fatal: the first run builds them, which takes a few extra minutes.)"

# ---------------------------------------------------------------------------
if [[ $SMOKE -eq 1 ]]; then
  step "9/9 Smoke test (loads the model, a few minutes)"
  (cd "$CT_DIR" && MODEL="$MODEL" uv run python "$REPO_DIR/pipeline/smoke_test.py") || die "smoke test failed. Don't claim runs on this machine."
else
  step "9/9 Smoke test skipped (--skip-smoke-test)"
fi

echo
echo "SETUP COMPLETE on $(hostname): ${GPU_NAME}, Control Tower $(git -C "$CT_DIR" rev-parse --short HEAD)"
echo "Next:"
echo "  ./run_next.sh --status    see what's done and what's left"
echo "  ./run_next.sh             claim and run one task (recommended the first time)"
echo "  ./run_next.sh --loop      keep going until everything is claimed"
if [[ $START -eq 1 ]]; then
  echo; echo "--start given: starting ./run_next.sh --loop"
  exec "$REPO_DIR/run_next.sh" --loop
fi
