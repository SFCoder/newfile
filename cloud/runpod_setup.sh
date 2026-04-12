#!/usr/bin/env bash
# cloud/runpod_setup.sh — Run once on each fresh RunPod pod (or re-run to update).
#
# Usage:
#   bash /workspace/newfile/cloud/runpod_setup.sh [MODEL_ID ...]
#
# Examples:
#   bash /workspace/newfile/cloud/runpod_setup.sh
#   bash /workspace/newfile/cloud/runpod_setup.sh Qwen/Qwen2.5-7B SparseLLM/ReluLLaMA-7B
#
# The script is idempotent — safe to re-run on an existing pod.
# All state that should persist across pod restarts lives on /workspace (the
# network volume): pip wheels, HuggingFace model cache, git credentials.
#
# ── EDIT BEFORE FIRST RUN ────────────────────────────────────────────────────
# Replace the placeholder values below with your own, then commit the change
# once.  After that, re-running the script on any pod picks them up
# automatically.
GIT_USER_EMAIL="YOUR_EMAIL"
GIT_USER_NAME="YOUR_NAME"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_URL="https://github.com/SFCoder/newfile.git"
REPO_DIR="/workspace/newfile"
WORKSPACE="/workspace"
CRED_FILE="/workspace/.git-credentials"
HF_HOME_DIR="/workspace/.cache/huggingface"

# ── helpers ───────────────────────────────────────────────────────────────────
ok()      { echo "  ✓ $*"; }
warn()    { echo "  ⚠ $*"; }
info()    { echo "  → $*"; }
section() { echo ""; echo "── $* ──────────────────────────────────────────────"; }

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RunPod environment setup — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Disk space ─────────────────────────────────────────────────────────────
section "1. Disk space check"

FREE_GB=$(df -BG "$WORKSPACE" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || echo "?")
echo "  $WORKSPACE : ${FREE_GB}GB free"
if [[ "$FREE_GB" =~ ^[0-9]+$ ]] && [ "$FREE_GB" -lt 60 ]; then
    warn "Less than 60 GB free on $WORKSPACE."
    warn "Large model downloads (7B=14GB fp16, 72B=144GB fp16) may fail."
    warn "Consider cleaning /workspace before downloading models."
else
    ok "${FREE_GB}GB free (≥ 60 GB threshold)"
fi

# ── 2. Git identity ───────────────────────────────────────────────────────────
section "2. Git identity"

git config --global user.email "$GIT_USER_EMAIL"
git config --global user.name  "$GIT_USER_NAME"

if [ "$GIT_USER_EMAIL" = "YOUR_EMAIL" ] || [ "$GIT_USER_NAME" = "YOUR_NAME" ]; then
    warn "Git identity is still set to placeholder values."
    warn "Edit GIT_USER_EMAIL / GIT_USER_NAME at the top of this script,"
    warn "then commit the change once so every pod picks it up automatically."
else
    ok "user.name  = $GIT_USER_NAME"
    ok "user.email = $GIT_USER_EMAIL"
fi

# ── 3. Git credential helper ──────────────────────────────────────────────────
section "3. Git credential store"

# Store credentials on the volume so they survive pod restarts.
git config --global credential.helper "store --file $CRED_FILE"
ok "credential.helper → store --file $CRED_FILE"

if [ -s "$CRED_FILE" ]; then
    ok "Credentials file present and non-empty."
else
    info "No credentials stored yet."
    info "They are saved automatically after the first authenticated push/pull."
    info "Or pre-populate with:"
    info "  echo 'https://USER:TOKEN@github.com' > $CRED_FILE"
fi

# ── 4. HuggingFace cache on the network volume ────────────────────────────────
section "4. HuggingFace cache"

export HF_HOME="$HF_HOME_DIR"
ok "HF_HOME=$HF_HOME (exported for this session)"

# Persist into new interactive shells.
for RC in "$HOME/.bashrc" "$HOME/.profile"; do
    if [ -f "$RC" ] && ! grep -qF "HF_HOME" "$RC" 2>/dev/null; then
        echo "export HF_HOME=$HF_HOME_DIR" >> "$RC"
        ok "Added HF_HOME export to $RC"
    fi
done

# System-wide persistence (survives su / non-login shells on RunPod).
if [ -d /etc/profile.d ] && \
   { [ ! -f /etc/profile.d/hf_home.sh ] || ! grep -qF "HF_HOME" /etc/profile.d/hf_home.sh 2>/dev/null; }; then
    echo "export HF_HOME=$HF_HOME_DIR" > /etc/profile.d/hf_home.sh
    ok "Created /etc/profile.d/hf_home.sh"
fi

mkdir -p "$HF_HOME_DIR"

# Symlink /root/.cache/huggingface → /workspace/.cache/huggingface
# This ensures any code that uses the default XDG cache path also lands on
# the volume, not the ephemeral container filesystem.
HF_LINK="$HOME/.cache/huggingface"
mkdir -p "$HOME/.cache"

CURRENT_TARGET="$(readlink "$HF_LINK" 2>/dev/null || true)"
if [ "$CURRENT_TARGET" = "$HF_HOME_DIR" ]; then
    ok "Symlink already correct ($HF_LINK → $HF_HOME_DIR)"
elif [ -L "$HF_LINK" ]; then
    warn "Symlink $HF_LINK points to $CURRENT_TARGET — updating."
    ln -sfn "$HF_HOME_DIR" "$HF_LINK"
    ok "Symlink updated ($HF_LINK → $HF_HOME_DIR)"
elif [ -d "$HF_LINK" ]; then
    warn "$HF_LINK is a real directory — moving contents to volume, then symlinking."
    mv "$HF_LINK" "${HF_LINK}.bak.$(date +%s)"
    ln -s "$HF_HOME_DIR" "$HF_LINK"
    ok "Moved existing cache and symlinked ($HF_LINK → $HF_HOME_DIR)"
else
    ln -s "$HF_HOME_DIR" "$HF_LINK"
    ok "Symlinked ($HF_LINK → $HF_HOME_DIR)"
fi

# ── 5. Python dependencies ────────────────────────────────────────────────────
section "5. Python dependencies (pip install)"

# Install every package required across all experiments.
# pip skips already-up-to-date packages, so this is fast on repeat runs.
pip install -q \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    scipy \
    scikit-learn \
    matplotlib \
    huggingface_hub \
    tiktoken \
    sentencepiece \
    protobuf \
    numpy \
    pydantic \
    tqdm \
    datasets \
    fastapi \
    "uvicorn[standard]" \
    httpx

ok "pip install complete"

# ── 6. Clone or update repository ────────────────────────────────────────────
section "6. Repository"

if [ ! -d "$REPO_DIR/.git" ]; then
    info "Cloning $REPO_URL → $REPO_DIR …"
    git clone "$REPO_URL" "$REPO_DIR"
    ok "Cloned to $REPO_DIR"
else
    info "Repo already present — fetching latest …"
    BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" 2>/dev/null \
        || warn "Fast-forward failed on branch '$BRANCH' — check manually."
    ok "Up to date on branch: $BRANCH"
fi

# ── 7. Register / download models (optional positional args) ──────────────────
if [ "$#" -gt 0 ]; then
    section "7. Model registration"
    cd "$REPO_DIR"

    for MODEL_ID in "$@"; do
        info "Registering $MODEL_ID …"
        python3 - "$MODEL_ID" <<'PYEOF'
import sys
sys.path.insert(0, '.')
from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

model_id = sys.argv[1]
reg = ModelRegistry(DEFAULT_REGISTRY_PATH)
try:
    entry = reg.register_new_model(
        model_id=model_id,
        hf_repo=model_id,
        min_stake=0,
        download_if_missing=True,
    )
    print(f"  ✓ {model_id}  (hash={entry.weight_hash[:16]}…)")
except Exception as e:
    print(f"  [ERROR] {model_id}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    done
fi

# ── 8. Package import verification ───────────────────────────────────────────
section "8. Package verification"

python3 - <<'PYEOF'
import sys

PACKAGES = [
    ("torch",            "torch"),
    ("transformers",     "transformers"),
    ("accelerate",       "accelerate"),
    ("bitsandbytes",     "bitsandbytes"),
    ("scipy",            "scipy"),
    ("scikit-learn",     "sklearn"),
    ("matplotlib",       "matplotlib"),
    ("huggingface_hub",  "huggingface_hub"),
    ("tiktoken",         "tiktoken"),
    ("sentencepiece",    "sentencepiece"),
    ("protobuf",         "google.protobuf"),
    ("numpy",            "numpy"),
    ("pydantic",         "pydantic"),
    ("tqdm",             "tqdm"),
    ("datasets",         "datasets"),
]

all_ok = True
for display, import_name in PACKAGES:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓  {display:<18} {ver}")
    except ImportError as exc:
        print(f"  ✗  {display:<18} MISSING — {exc}")
        all_ok = False

if not all_ok:
    print("\n  Some packages failed to import — re-run this script to retry.", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "  HF_HOME : $HF_HOME_DIR"
echo ""
echo "  Cached models:"
if [ -d "$HF_HOME_DIR/hub" ] && [ -n "$(ls -A "$HF_HOME_DIR/hub" 2>/dev/null)" ]; then
    ls "$HF_HOME_DIR/hub" | sed 's/^/    /'
else
    echo "    (none yet — pass MODEL_IDs as args to download)"
fi

if [ -d "$REPO_DIR" ]; then
    echo ""
    BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    COMMIT=$(git -C "$REPO_DIR" log -1 --format="%h %s" 2>/dev/null || echo "unknown")
    echo "  Repo   : $REPO_DIR"
    echo "  Branch : $BRANCH"
    echo "  Commit : $COMMIT"
fi

echo ""
echo "  Quick-start:"
echo "    python3 adversarial_study.py"
echo "    python3 threshold_study.py --model Qwen/Qwen2.5-7B"
echo "    bash cloud/push_results.sh \"what you ran\""
echo ""
