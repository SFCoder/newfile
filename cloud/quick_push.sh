#!/bin/bash
set -e
git add -A analysis_results/ tools/ requirements-runpod.txt cloud/ .gitignore 2>/dev/null || true
git reset HEAD -- results.db __pycache__/ 2>/dev/null || true
git commit -m "$1" || echo "Nothing to commit"
git push origin claude/max-savings-experiment-Gktym
