#!/usr/bin/env bash
# cloud/runpod_setup.sh — Run once on each fresh RunPod pod.
#
# Usage:
#   bash /workspace/newfile/cloud/runpod_setup.sh [MODEL_ID ...]
#
# Examples:
#   bash /workspace/newfile/cloud/runpod_setup.sh
#   bash /workspace/newfile/cloud/runpod_setup.sh Qwen/Qwen2.5-7B Qwen/Qwen2.5-72B
#
# The script is idempotent — safe to run on an existing pod to pull updates.
# Everything that costs time (model weights, pip packages) lives on the
# network volume and is only downloaded once across all pods.

set -euo pipefail

# ── configuration ─────────────────────────────────────────────────────────
# Edit these two lines once, then commit the change.
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="you@example.com"

# GitHub repo — update to match your fork/clone target.
REPO_URL="https://github.com/sfcoder/newfile.git"

REPO_DIR="/workspace/newfile"
WORKSPACE="/workspace"

# ── helpers ───────────────────────────────────────────────────────────────
ok()   { echo "  ✓ $*"; }
warn() { echo "  [WARN] $*"; }
info() { echo "  → $*"; }

echo ""
echo "================================================================"
echo "  RunPod setup — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "================================================================"

# ── 1. Git identity — stored on the volume so it persists across pods ─────
echo ""
echo "── 1. Git identity"

VOLUME_GITCONFIG="$WORKSPACE/.gitconfig"
ROOT_GITCONFIG="$HOME/.gitconfig"

# Symlink /root/.gitconfig → /workspace/.gitconfig so it survives reboots.
if [ ! -e "$VOLUME_GITCONFIG" ]; then
    info "Creating $VOLUME_GITCONFIG with placeholder identity …"
    git config --file "$VOLUME_GITCONFIG" user.name  "$GIT_USER_NAME"
    git config --file "$VOLUME_GITCONFIG" user.email "$GIT_USER_EMAIL"
    info "Edit $VOLUME_GITCONFIG to set your real name/email if needed."
fi

# (Re-)create the symlink on every boot since /root is ephemeral.
if [ -L "$ROOT_GITCONFIG" ]; then
    ok "gitconfig symlink already in place"
elif [ -f "$ROOT_GITCONFIG" ]; then
    warn "$ROOT_GITCONFIG exists as a regular file; backing it up"
    mv "$ROOT_GITCONFIG" "${ROOT_GITCONFIG}.bak"
    ln -s "$VOLUME_GITCONFIG" "$ROOT_GITCONFIG"
    ok "gitconfig symlinked ($ROOT_GITCONFIG → $VOLUME_GITCONFIG)"
else
    ln -s "$VOLUME_GITCONFIG" "$ROOT_GITCONFIG"
    ok "gitconfig symlinked ($ROOT_GITCONFIG → $VOLUME_GITCONFIG)"
fi

GIT_NAME=$(git config --global user.name  2>/dev/null || echo "(not set)")
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "(not set)")
ok "git identity: $GIT_NAME <$GIT_EMAIL>"

# ── 2. Git credential store — credentials live on the volume ──────────────
echo ""
echo "── 2. Git credentials"

CRED_FILE="$WORKSPACE/.git-credentials"
git config --global credential.helper "store --file $CRED_FILE"

if [ -s "$CRED_FILE" ]; then
    ok "credential store in place ($CRED_FILE)"
else
    info "No stored credentials yet."
    info "They will be saved automatically after your first 'git push'."
    info "You can also pre-populate with:"
    info "  echo 'https://USER:TOKEN@github.com' > $CRED_FILE"
fi

# ── 3. Clone or pull the repository ───────────────────────────────────────
echo ""
echo "── 3. Repository"

if [ ! -d "$REPO_DIR/.git" ]; then
    if [ "$REPO_URL" = "https://github.com/sfcoder/newfile.git" ]; then
        warn "REPO_URL is still the placeholder — edit it in this script before cloning."
        warn "Skipping clone; set REPO_URL and re-run."
    else
        info "Cloning $REPO_URL → $REPO_DIR …"
        git clone "$REPO_URL" "$REPO_DIR"
        ok "cloned to $REPO_DIR"
    fi
else
    info "Repo already exists — pulling latest …"
    git -C "$REPO_DIR" fetch origin
    BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" 2>/dev/null \
        || warn "fast-forward failed (diverged?); check manually"
    ok "up to date on branch: $BRANCH"
fi

if [ ! -d "$REPO_DIR" ]; then
    echo ""
    echo "ERROR: $REPO_DIR not found after clone step — cannot continue."
    exit 1
fi

cd "$REPO_DIR"

# ── 4. HuggingFace cache symlink ──────────────────────────────────────────
echo ""
echo "── 4. HuggingFace cache"

HF_VOLUME="$WORKSPACE/.cache/huggingface/hub"
HF_ROOT="$HOME/.cache/huggingface/hub"

mkdir -p "$HF_VOLUME"
mkdir -p "$(dirname "$HF_ROOT")"

if [ -L "$HF_ROOT" ]; then
    ok "symlink already in place ($HF_ROOT → $HF_VOLUME)"
elif [ -d "$HF_ROOT" ]; then
    warn "$HF_ROOT is a real directory — moving contents to volume …"
    mv "$HF_ROOT" "${HF_ROOT}.bak"
    ln -s "$HF_VOLUME" "$HF_ROOT"
    ok "moved existing cache and symlinked ($HF_ROOT → $HF_VOLUME)"
else
    ln -s "$HF_VOLUME" "$HF_ROOT"
    ok "symlinked ($HF_ROOT → $HF_VOLUME)"
fi

# ── 5. Python dependencies ────────────────────────────────────────────────
echo ""
echo "── 5. Python dependencies"

if [ -f "requirements.txt" ]; then
    info "pip install -r requirements.txt …"
    pip install -q -r requirements.txt
    ok "dependencies installed"
else
    warn "requirements.txt not found — skipping"
fi

# ── 6. Register / download models ─────────────────────────────────────────
if [ "$#" -gt 0 ]; then
    echo ""
    echo "── 6. Model registration"

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
    print(f"  ✓ {model_id} (hash={entry.weight_hash[:16]}…)")
except Exception as e:
    print(f"  [ERROR] could not register {model_id}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    done
fi

# ── 7. Summary ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Setup complete"
echo "================================================================"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git log -1 --format="%h %s" 2>/dev/null || echo "unknown")
echo ""
echo "  Repo   : $REPO_DIR"
echo "  Branch : $BRANCH"
echo "  Commit : $COMMIT"

# Cached models — show subdirectory names in HF hub cache
echo ""
echo "  Cached models:"
if [ -d "$HF_VOLUME" ] && [ "$(ls -A "$HF_VOLUME" 2>/dev/null)" ]; then
    ls "$HF_VOLUME" | sed 's/^/    /'
else
    echo "    (none yet)"
fi

# Registered models in registry.json
echo ""
echo "  Registered in registry.json:"
python3 -c "
import sys; sys.path.insert(0,'.')
from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH
try:
    reg = ModelRegistry(DEFAULT_REGISTRY_PATH)
    for m in reg.list_models():
        print(f'    {m}')
except Exception as e:
    print(f'    (could not read registry: {e})')
" 2>/dev/null || echo "    (registry unavailable)"

echo ""
echo "  Quick-start commands:"
echo ""
echo "    # Threshold study"
echo "    python3 threshold_study.py --model Qwen/Qwen2.5-7B"
echo ""
echo "    # Adversarial study"
echo "    python3 adversarial_study.py"
echo ""
echo "    # Max savings experiment"
echo "    python3 tools/max_savings_test.py --model Qwen/Qwen2.5-7B"
echo ""
echo "    # Push results when done"
echo "    bash cloud/push_results.sh \"describe what you ran\""
echo ""
echo "  To register a new model:"
echo "    bash cloud/runpod_setup.sh Qwen/Qwen2.5-72B"
echo ""
