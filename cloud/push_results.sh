#!/usr/bin/env bash
# cloud/push_results.sh — Commit and push experiment results to the remote.
#
# Usage:
#   bash cloud/push_results.sh "describe what you ran"
#
# What it does:
#   1. Runs migrate.py to import any standard_v1 results files
#   2. Warns if results.db is accidentally staged (it should be gitignored)
#   3. Stages everything under analysis_results/
#   4. Commits with the provided message and pushes

set -euo pipefail

# ── argument check ────────────────────────────────────────────────────────
if [ "$#" -eq 0 ]; then
    echo "Usage: bash cloud/push_results.sh \"commit message\""
    echo ""
    echo "Example:"
    echo "  bash cloud/push_results.sh \"72B threshold sweep — 5 thresholds, 10 prompts\""
    exit 1
fi

COMMIT_MSG="$1"

# ── locate repo root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================"
echo "  push_results.sh — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "================================================================"

# ── 1. Import any new standard_v1 results into the database ───────────────
echo ""
echo "── 1. migrate.py"
if [ -f "results.db" ]; then
    python3 migrate.py
else
    echo "  (results.db not found — skipping migration)"
fi

# ── 2. Safety check: results.db must not be staged ────────────────────────
echo ""
echo "── 2. Staging checks"

# Un-stage results.db if somehow tracked
if git ls-files --error-unmatch results.db &>/dev/null 2>&1; then
    echo "  [WARN] results.db is tracked by git — removing from staging area"
    git rm --cached results.db
    echo "  [WARN] Also committing the removal of results.db …"
fi

if git diff --cached --name-only 2>/dev/null | grep -q "^results\.db$"; then
    echo "  [WARN] results.db is in the staging area — unstaging it"
    git restore --staged results.db 2>/dev/null || git reset HEAD results.db 2>/dev/null
fi

echo "  ✓ results.db not staged"

# ── 3. Stage analysis_results/ ────────────────────────────────────────────
echo ""
echo "── 3. Staging analysis_results/"

if [ ! -d "analysis_results" ]; then
    echo "  [WARN] analysis_results/ directory not found — nothing to stage"
    exit 0
fi

git add analysis_results/

STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
if [ "$STAGED" -eq 0 ]; then
    echo "  Nothing new in analysis_results/ — nothing to commit."
    exit 0
fi

echo "  Staged $STAGED file(s):"
git diff --cached --name-only | head -20 | sed 's/^/    /'
if [ "$STAGED" -gt 20 ]; then
    echo "    … ($((STAGED - 20)) more)"
fi

# ── 4. Commit and push ────────────────────────────────────────────────────
echo ""
echo "── 4. Commit"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git commit -m "$COMMIT_MSG

https://github.com/sfcoder/newfile"

echo ""
echo "── 5. Push → origin/$BRANCH"
git push origin "$BRANCH"

echo ""
echo "================================================================"
echo "  Done."
echo "  Branch : $BRANCH"
echo "  Commit : $(git log -1 --format='%h %s')"
echo "================================================================"
echo ""
