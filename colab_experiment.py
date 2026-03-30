# =============================================================================
# SwiGLU Fingerprint Experiment — Google Colab Version
# =============================================================================
# Paste this entire file into a single Colab cell.
# Runtime: select GPU (T4 free tier works).
# Time: ~2-3 hours.
#
# BEFORE RUNNING: You need to:
# 1. Accept the Llama license at https://huggingface.co/meta-llama/Llama-3.2-1B
#    and https://huggingface.co/meta-llama/Llama-3.2-3B
# 2. Set your HuggingFace token below (line marked with ← EDIT THIS)
# =============================================================================

HF_TOKEN = "hf_your_token_here"  # ← EDIT THIS: paste your HuggingFace token

# --- Setup ---
!pip install -q torch transformers datasets accelerate bitsandbytes scipy scikit-learn matplotlib tqdm huggingface_hub

from huggingface_hub import login
login(token=HF_TOKEN)

import os
os.makedirs("fingerprint_data", exist_ok=True)
os.makedirs("analysis_results", exist_ok=True)

# --- Download experiment scripts ---
# Write extract_activations.py
# (In practice, you'd upload the files or clone a repo. For Colab,
#  the simplest approach is to upload the .py files using the sidebar
#  file browser, or clone from a git repo:)
#
# !git clone https://github.com/YOUR_USERNAME/swiglu-fingerprint.git
# %cd swiglu-fingerprint

# If you've uploaded the files manually, just make sure they're in
# the current directory:
!ls *.py

# --- Run extraction on 1B model ---
print("=" * 60)
print("Extracting fingerprints from Llama-3.2-1B")
print("=" * 60)

!python extract_activations.py \
    --model meta-llama/Llama-3.2-1B \
    --tag 1b \
    --num-prompts 300 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data

# --- Run extraction on 3B model ---
print("=" * 60)
print("Extracting fingerprints from Llama-3.2-3B")
print("=" * 60)

!python extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --tag 3b \
    --num-prompts 300 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data

# --- Run analysis ---
print("=" * 60)
print("Analyzing fingerprint separation")
print("=" * 60)

!python analyze_fingerprints.py \
    --model-a ./fingerprint_data/1b \
    --model-b ./fingerprint_data/3b \
    --output ./analysis_results/1b_vs_3b

# --- Display results ---
import json
from IPython.display import Image, display

with open("./analysis_results/1b_vs_3b/summary.json") as f:
    summary = json.load(f)

acc = summary["classification"]["random_forest_accuracy"]
auc = summary["classification"]["random_forest_auc"]
print(f"\n{'='*60}")
print(f"RESULTS: Random Forest Accuracy = {acc:.4f}, AUC = {auc:.4f}")
print(f"{'='*60}\n")

# Show the plots inline
for plot in [
    "compare_sparsity_ratio.png",
    "compare_gini_mean.png",
    "compare_energy_concentration_top1pct.png",
    "per_layer_accuracy.png",
    "feature_importance.png",
    "trajectory_curvature.png",
]:
    path = f"./analysis_results/1b_vs_3b/{plot}"
    if os.path.exists(path):
        print(f"\n--- {plot} ---")
        display(Image(filename=path))

# --- Download results ---
# Zip everything for download
!tar czf swiglu_results.tar.gz analysis_results/ fingerprint_data/*/metadata.json fingerprint_data/*/summary.json 2>/dev/null || true
print("\nDownload swiglu_results.tar.gz from the file browser (left sidebar)")
