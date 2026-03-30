#!/bin/bash
# =============================================================================
# run_tier1.sh — Run the complete Tier 1 sanity check
#
# Usage:
#   chmod +x run_tier1.sh
#   ./run_tier1.sh
#
# This runs 1B vs 3B with 500 prompts each, then analyzes the results.
# Requires: GPU with 12GB+ VRAM, HuggingFace login completed.
# Expected time: 2-4 hours total.
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "SwiGLU Fingerprint Experiment — Tier 1"
echo "Llama-3.2-1B vs Llama-3.2-3B"
echo "=============================================="
echo ""

NUM_PROMPTS=${1:-500}
MAX_TOKENS=${2:-32}

echo "Configuration:"
echo "  Prompts per model: $NUM_PROMPTS"
echo "  Tokens generated:  $MAX_TOKENS"
echo ""

# Check GPU
echo "--- Checking GPU ---"
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU detected. This experiment requires a GPU.')
    exit(1)
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'GPU: {gpu} ({mem:.1f} GB)')
if mem < 8:
    print('WARNING: Less than 8GB VRAM. The 3B model may not fit.')
    print('Consider using --quantize 4bit or a larger GPU.')
"
echo ""

# Check HuggingFace login
echo "--- Checking HuggingFace access ---"
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.model_info('meta-llama/Llama-3.2-1B')
    print('Access confirmed for Llama-3.2-1B')
except Exception as e:
    print(f'ERROR: Cannot access Llama models: {e}')
    print('Run: huggingface-cli login')
    print('And accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B')
    exit(1)
"
echo ""

# Phase 1: Extract from 1B model
echo "=============================================="
echo "Phase 1/3: Extracting fingerprints from 1B model"
echo "=============================================="
START=$(date +%s)

python3 extract_activations.py \
    --model meta-llama/Llama-3.2-1B \
    --tag 1b \
    --num-prompts "$NUM_PROMPTS" \
    --max-new-tokens "$MAX_TOKENS" \
    --output-dir ./fingerprint_data

END=$(date +%s)
echo "1B extraction took $((END - START)) seconds"
echo ""

# Phase 2: Extract from 3B model
echo "=============================================="
echo "Phase 2/3: Extracting fingerprints from 3B model"
echo "=============================================="
START=$(date +%s)

python3 extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --tag 3b \
    --num-prompts "$NUM_PROMPTS" \
    --max-new-tokens "$MAX_TOKENS" \
    --output-dir ./fingerprint_data

END=$(date +%s)
echo "3B extraction took $((END - START)) seconds"
echo ""

# Phase 3: Analysis
echo "=============================================="
echo "Phase 3/3: Analyzing fingerprint separation"
echo "=============================================="

python3 analyze_fingerprints.py \
    --model-a ./fingerprint_data/1b \
    --model-b ./fingerprint_data/3b \
    --output ./analysis_results/1b_vs_3b

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: ./analysis_results/1b_vs_3b/"
echo ""
echo "Key files:"
echo "  summary.json                          — headline metrics"
echo "  compare_sparsity_ratio.png            — sparsity comparison"
echo "  compare_gini_mean.png                 — activation selectivity"
echo "  compare_energy_concentration_top1pct.png — energy distribution"
echo "  per_layer_accuracy.png                — which layers matter"
echo "  feature_importance.png                — which statistics matter"
echo "  trajectory_curvature.png              — refinement dynamics"
echo ""
echo "Quick summary:"
python3 -c "
import json
with open('./analysis_results/1b_vs_3b/summary.json') as f:
    s = json.load(f)
acc = s['classification']['random_forest_accuracy']
auc = s['classification']['random_forest_auc']
print(f'  Random Forest Accuracy: {acc:.4f}')
print(f'  Random Forest AUC:      {auc:.4f}')
print()
if acc > 0.95:
    print('  VERDICT: STRONGLY DISTINGUISHABLE')
    print('  The fingerprint works. Proceed to Tier 2.')
elif acc > 0.80:
    print('  VERDICT: CLEARLY DISTINGUISHABLE')
    print('  Promising results. Worth investigating further.')
elif acc > 0.60:
    print('  VERDICT: WEAKLY DISTINGUISHABLE')
    print('  Some signal present. May need richer statistics.')
else:
    print('  VERDICT: NOT DISTINGUISHABLE')
    print('  Fingerprint does not separate these models.')
"
