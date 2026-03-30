# Implementation Guide: SwiGLU Activation Fingerprinting Experiment

## Before You Start

You need three things: a computer with a GPU, a HuggingFace account with Llama access, and about 4 hours for the first meaningful result.

---

## Step 1: Get Hardware

Pick one of these options.

### Option A: Your Own GPU (cheapest, simplest)

Any NVIDIA GPU with 16GB+ VRAM works for Tier 1 (1B vs 3B models). This includes RTX 3090, RTX 4080, RTX 4090, or any A-series card. Check what you have:

```bash
nvidia-smi
```

Look at the "MiB" column for total memory. You need at least 8GB for the 1B model, 12GB for the 3B model. If you have 24GB, both fit comfortably.

### Option B: Google Colab (free, but limited)

Go to https://colab.research.google.com, create a new notebook, and select Runtime > Change runtime type > T4 GPU (free tier) or A100 (Colab Pro). A T4 has 16GB VRAM — enough for Tier 1. Run all the commands below as notebook cells prefixed with `!`.

### Option C: Cloud GPU rental (for Tier 2/3)

For the full 70B experiment, rent from one of these:

- **RunPod** (https://runpod.io): A100 80GB at ~$1.50/hr. Most straightforward.
- **Lambda Labs** (https://lambdalabs.com): A100 at ~$1.10/hr. Often waitlisted.
- **Vast.ai** (https://vast.ai): Cheapest, but more setup required.

Rent a machine with 1x A100 80GB. Budget $30-80 for the full Tier 3 experiment.

---

## Step 2: Set Up the Environment

SSH into your machine (or open your terminal / Colab notebook) and run these commands one at a time.

```bash
# 1. Create a working directory
mkdir -p ~/swiglu_experiment
cd ~/swiglu_experiment

# 2. Install Python 3.10+ if not present (skip on Colab / most cloud instances)
# Most GPU instances already have this. Check with:
python3 --version
# You need 3.10 or higher.

# 3. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install PyTorch (with CUDA support)
# Check your CUDA version first:
nvidia-smi | head -3
# Look for "CUDA Version: XX.X"

# For CUDA 12.x (most modern setups):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older machines):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install the remaining dependencies
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.27.0 \
    bitsandbytes>=0.43.0 scipy>=1.12.0 scikit-learn>=1.4.0 \
    matplotlib>=3.8.0 numpy>=1.26.0 tqdm>=4.66.0

# 6. Verify CUDA works with PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
```

That last command should print your GPU name. If it says "CUDA available: False", your PyTorch installation doesn't see the GPU — go back and check the CUDA version match.

---

## Step 3: Get Model Access

Llama models are gated. You need to accept Meta's license.

```bash
# 1. Go to EACH of these URLs in a browser and click "Accept":
#    https://huggingface.co/meta-llama/Llama-3.2-1B
#    https://huggingface.co/meta-llama/Llama-3.2-3B
#
#    (For Tier 2/3, also accept:)
#    https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
#    https://huggingface.co/meta-llama/Meta-Llama-3.1-70B

# 2. Create a HuggingFace access token:
#    Go to https://huggingface.co/settings/tokens
#    Click "New token" > give it a name > select "Read" access > Create
#    Copy the token (starts with "hf_...")

# 3. Log in from the command line:
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted. Say "yes" to saving it.

# 4. Verify access works:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-1B', max_workers=1, allow_patterns='config.json')"
# If this downloads without error, you have access.
```

If step 4 gives a 403 error, you either haven't accepted the license or the token is wrong. The license approval is usually instant but occasionally takes a few minutes.

---

## Step 4: Get the Experiment Code

Create the three files. You can either download them from where I provided them, or create them manually:

```bash
cd ~/swiglu_experiment

# If you have the files from the previous step, copy them here.
# Otherwise, create them with the content from the artifacts.
# The three files you need are:
#   extract_activations.py
#   analyze_fingerprints.py
#   requirements.txt
```

Verify the files are present:

```bash
ls -la *.py
# Should show extract_activations.py and analyze_fingerprints.py
```

---

## Step 5: Run Tier 1 — The Sanity Check

This is the critical experiment. You are running 500 identical prompts through a 1B and a 3B model, capturing what the SwiGLU gate does at every layer, and then checking if the two models are distinguishable from those statistics alone.

### 5a: Extract fingerprints from the 1B model

```bash
python3 extract_activations.py \
    --model meta-llama/Llama-3.2-1B \
    --tag 1b \
    --num-prompts 500 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data
```

What this does:
- Downloads the 1B model (~2GB, first run only)
- Loads 500 prompts from the C4 dataset
- For each prompt, runs greedy generation (32 new tokens)
- At each layer, captures the SwiGLU activation vector and computes
  sparsity, Gini coefficient, magnitude statistics, etc.
- Saves everything to `./fingerprint_data/1b/fingerprints.json`

What to expect:
- Download time: 2-5 minutes (first run)
- Processing time: 20-60 minutes depending on GPU
- You'll see a progress bar. If it's moving, everything is working.
- Checkpoint files save every 100 prompts so you don't lose progress.

What can go wrong:
- "CUDA out of memory": reduce `--max-new-tokens` to 16, or `--num-prompts` to 200.
- "HTTP 403": model access not approved. Go back to Step 3.
- "Connection timeout": HuggingFace is slow. Try again, or set
  `HF_HUB_DOWNLOAD_TIMEOUT=300` environment variable.

### 5b: Extract fingerprints from the 3B model

```bash
python3 extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --tag 3b \
    --num-prompts 500 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data
```

Same process, larger model. Takes roughly 2x longer than the 1B.

### 5c: Compare them

```bash
python3 analyze_fingerprints.py \
    --model-a ./fingerprint_data/1b \
    --model-b ./fingerprint_data/3b \
    --output ./analysis_results/1b_vs_3b
```

This takes 1-2 minutes (no GPU needed, it's just statistics).

### 5d: Read your results

```bash
# The headline number:
cat ./analysis_results/1b_vs_3b/summary.json | python3 -m json.tool | grep -A2 "random_forest"

# The full text output was printed to the terminal during the run.
# Look for the "VERDICT" line at the very end.
```

Look at the plots:

```bash
ls ./analysis_results/1b_vs_3b/*.png
```

These are the files that matter:

| File | What it shows |
|------|---------------|
| `compare_sparsity_ratio.png` | Left: sparsity by layer depth. Right: distribution overlap. |
| `compare_gini_mean.png` | Same for Gini coefficient (activation selectivity). |
| `compare_energy_concentration_top1pct.png` | How concentrated is the energy in top 1% of neurons? |
| `per_layer_accuracy.png` | Which layers are best at telling the models apart? |
| `feature_importance.png` | Which specific measurements matter most? |
| `trajectory_curvature.png` | Layer-to-layer refinement rate. |

If you're on a remote machine, download the plots:

```bash
# From your LOCAL machine:
scp -r user@remote:~/swiglu_experiment/analysis_results ./local_results/

# Or zip them on the remote and download:
cd ~/swiglu_experiment
tar czf results.tar.gz analysis_results/
# Then download results.tar.gz
```

---

## Step 6: Interpret Results

### The go/no-go decision

Open `summary.json` and look at `random_forest_accuracy`:

- **Above 0.95**: Strong signal. The fingerprint clearly distinguishes the models.
  Proceed to Tier 2. You likely have a viable proof-of-inference mechanism.

- **0.80 to 0.95**: Good signal. The models are distinguishable but there's some
  overlap. Look at `feature_importance.png` to see which statistics are doing the
  heavy lifting. Consider whether adding more statistics (cross-layer correlations,
  activation histogram shapes) would push accuracy higher.

- **0.60 to 0.80**: Weak signal. Something is there but it's not strong enough
  to be practically useful. Check whether the issue is that one or two statistics
  carry all the signal (and the rest are noise), or whether everything is marginal.

- **Below 0.60**: No meaningful signal at this level. But don't give up entirely —
  1B vs 3B may simply be too similar. The 8B vs 70B gap is much larger and might
  show separation that 1B vs 3B doesn't.

### What the plots tell you

**compare_sparsity_ratio.png**: If the curves are clearly separated (blue above
or below red with minimal overlap), sparsity alone can distinguish the models.
If they overlap, sparsity isn't enough and you need the other statistics.

**per_layer_accuracy.png**: Look at the shape. If deep layers (right side) are
more discriminative, that matches the theory — deep layers are where model
capacity matters most. If early layers are more discriminative, something
unexpected is happening worth investigating.

**feature_importance.png**: The top features should be structural statistics
(gini, sparsity, energy concentration) at deep layers. If raw magnitudes
(mag_mean, mag_std) dominate, the fingerprint might be sensitive to scaling
rather than structure — that's a robustness concern.

**trajectory_curvature.png**: The larger model should show higher displacement
in the mid-to-deep layers (it's doing more per-layer refinement). If the
trajectories look identical, layer-to-layer dynamics aren't useful.

---

## Step 7: Tier 2 — Quantization Robustness (if Tier 1 passes)

This answers: does the fingerprint see quantization as noise (good) or as a
different model (bad)?

```bash
# 8B at full fp16 precision
python3 extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --tag 8b-fp16 \
    --num-prompts 500 \
    --output-dir ./fingerprint_data

# Same 8B at 4-bit quantization
python3 extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --tag 8b-4bit \
    --num-prompts 500 \
    --quantize 4bit \
    --output-dir ./fingerprint_data

# Compare
python3 analyze_fingerprints.py \
    --model-a ./fingerprint_data/8b-fp16 \
    --model-b ./fingerprint_data/8b-4bit \
    --output ./analysis_results/8b_fp16_vs_4bit
```

**What you want**: classification accuracy BELOW 0.70. You want the system
to see fp16 and 4-bit as "the same model." If accuracy is above 0.85, your
fingerprint is too sensitive — you'll need to coarsen the statistics.

Also run 8B vs 3B to establish the inter-model separation:

```bash
python3 analyze_fingerprints.py \
    --model-a ./fingerprint_data/8b-fp16 \
    --model-b ./fingerprint_data/3b \
    --output ./analysis_results/8b_vs_3b
```

**Ideal outcome**: 8B-fp16 vs 8B-4bit accuracy ~0.55 (indistinguishable),
8B vs 3B accuracy >0.90 (clearly different). That gap between intra-model
noise and inter-model signal is your operating margin.

---

## Step 8: Tier 3 — The Real Test (if Tier 2 passes)

This requires serious GPU. Rent a machine with 80GB+ VRAM or use 4-bit
quantization on a 48GB card.

```bash
# 70B at 4-bit (the "honest provider" scenario)
python3 extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-70B \
    --tag 70b-4bit \
    --num-prompts 1000 \
    --quantize 4bit \
    --output-dir ./fingerprint_data

# Compare against the 8B (the "attacker substitution" scenario)
python3 analyze_fingerprints.py \
    --model-a ./fingerprint_data/70b-4bit \
    --model-b ./fingerprint_data/8b-fp16 \
    --output ./analysis_results/70b_vs_8b
```

This is the test that matters: can you detect when someone runs 8B and claims
it was 70B? You want >0.95 accuracy.

---

## Troubleshooting

### "torch.cuda.OutOfMemoryError"

The model doesn't fit in your GPU. Options:

```bash
# Try 4-bit quantization:
python3 extract_activations.py --model ... --quantize 4bit ...

# Or reduce batch processing (the script processes one prompt at a time,
# so this shouldn't usually be an issue. If it is, reduce max tokens):
python3 extract_activations.py --model ... --max-new-tokens 16 ...
```

### Slow extraction

The hook captures activations at every layer for every forward pass during
generation. Each generated token triggers a full forward pass. To speed up:

```bash
# Generate fewer tokens (16 instead of 64)
python3 extract_activations.py --model ... --max-new-tokens 16 ...

# Use fewer prompts for initial testing
python3 extract_activations.py --model ... --num-prompts 100 ...
```

### "KeyError" in the analysis script

The two models have different architectures (different number of layers).
The analysis script handles this by normalizing to relative depth, but edge
cases can break. Check that both fingerprint JSON files have data:

```bash
python3 -c "
import json
with open('./fingerprint_data/1b/fingerprints.json') as f:
    data = json.load(f)
print(f'Prompts: {len(data)}')
print(f'Layers per prompt: {len(data[0][\"layers\"])}')
print(f'Stats per layer: {list(data[0][\"layers\"][\"0\"].keys())}')
"
```

### The C4 dataset won't load

The script falls back to synthetic prompts if C4 is unavailable. This is
fine for the experiment — the prompts don't need to be naturalistic, they
just need to be diverse enough to exercise different parts of the model.

### Model download is very slow

Set these environment variables:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer  # faster downloads
```

---

## What to Do With Your Results

### If the experiment works (accuracy > 0.95 for different-size models)

You have empirical evidence that SwiGLU activation statistics can distinguish
model sizes. This is a publishable result. Your next steps:

1. Write it up. The framing is: "We demonstrate that post-gating activation
   statistics in SwiGLU-based transformers provide a compact, model-specific
   fingerprint suitable for lightweight inference verification."

2. Test adversarial robustness. Fine-tune a small model to minimize
   fingerprint divergence from a large model. See if it can fool the
   classifier without matching the large model's output quality.

3. Contact the teams building proof-of-inference systems (Inference Labs,
   Ritual, OpenGradient). Show them the results. They are actively looking
   for cheaper alternatives to full ZK proving.

4. Design the verification protocol. Based on which statistics are most
   discriminative (from `feature_importance.png`), specify the minimal
   fingerprint that achieves >0.95 detection rate.

### If the experiment partially works (accuracy 0.70-0.95)

The signal exists but needs amplification. Try:

- Richer statistics: full activation histograms, cross-layer correlation
  matrices, PCA components of the activation trajectory.
- More prompts: 5000 instead of 500. Some statistics may need more
  samples to stabilize.
- Token-position analysis: compare fingerprints at specific token positions
  (first token, last token, mid-sequence) separately.

### If the experiment doesn't work (accuracy < 0.70)

The summary statistics may be too lossy. Try:
- Random projections of the raw activation vector (higher dimensional
  fingerprint, more expensive but more informative).
- The full binary sparsity mask (which neurons are active/inactive) —
  this is a much richer signal than the aggregate sparsity ratio.
- Hash-based fingerprints: quantize the top-k activation values and
  hash them, creating a bitstring fingerprint per layer.

If none of these work for 1B vs 3B, try 3B vs 8B or 8B vs 70B. The
fingerprinting approach may only work when the model size gap is large enough.

---

## Cost and Time Summary

| What | Hardware | Time | Cost |
|------|----------|------|------|
| Tier 1: 1B vs 3B, 500 prompts each | RTX 4090 or free Colab T4 | 2-4 hours | $0 |
| Tier 2: 8B fp16 vs 4-bit, 500 prompts | A100 40GB | 4-8 hours | $5-12 |
| Tier 3: 70B vs 8B, 1000 prompts | A100 80GB | 12-24 hours | $30-80 |
| Analysis (all tiers) | Any CPU | 5 minutes | $0 |

Start with Tier 1. If it works, you've spent zero dollars and half a day
to validate a novel approach to proof of inference. That's a good trade.
