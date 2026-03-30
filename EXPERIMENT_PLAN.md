# SwiGLU Activation Fingerprinting: Experiment Plan

## Hypothesis

The GELU/SwiGLU activation pattern at each layer of an LLM constitutes a
compact fingerprint that reliably distinguishes which model performed inference.
A smaller (distilled/cheaper) model cannot reproduce the activation statistics
of a larger model without computational cost converging to that of running the
larger model directly. If confirmed, this property can serve as a lightweight
proof-of-inference mechanism for blockchain verification.

## What This Experiment Tests

We run identical prompts through model pairs of different sizes from the same
family and measure whether their SwiGLU activation statistics are easily
distinguishable. If a simple classifier achieves near-perfect accuracy at
telling them apart from activation statistics alone, the fingerprint has
discriminative power. If that power holds across diverse inputs and is robust
to quantization, it has practical viability.

---

## Phase 0: Environment Setup

### Hardware Requirements

You have three tiers of experiment you can run depending on available compute:

**Tier 1 — Minimal (single consumer GPU, 16-24GB VRAM)**
- Models: Llama-3.2-1B vs Llama-3.2-3B
- This is your sanity check. These are small enough to run on a single RTX 4090
  or even a free Colab T4. If the fingerprint doesn't distinguish 1B from 3B,
  it won't distinguish 8B from 70B, and the whole approach is dead.
- Expected runtime: ~2-4 hours for 1000 prompts per model.

**Tier 2 — Mid-range (single A100 80GB or equivalent)**
- Models: Llama-3.1-8B (fp16) vs Llama-3.1-8B (4-bit quantized)
- This tests a different dimension: same architecture, same weights, different
  numerical precision. If quantization alone changes the fingerprint beyond
  the tolerance band, that's a problem the system needs to handle.
- Also: Llama-3.1-8B vs Llama-3.2-3B (different sizes, same family).
- Expected runtime: ~4-8 hours for 1000 prompts per model.

**Tier 3 — Full experiment (4x A100 80GB or cloud equivalent)**
- Models: Llama-3.1-70B (4-bit quantized) vs Llama-3.1-8B (fp16)
- This is the real test. The 70B-vs-8B comparison is the actual threat model:
  a provider claims to run 70B but substitutes 8B.
- Expected runtime: ~12-24 hours for 1000 prompts on the 70B model.
- Cloud cost estimate: ~$30-80 on Lambda Labs or RunPod.

**Recommendation: Start with Tier 1.** It costs nothing and gives you a
go/no-go signal within a day. Only proceed to Tier 2/3 if Tier 1 shows
strong separation.

### Software Setup

```bash
# Clone the experiment code
git clone <your-repo> swiglu_fingerprint
cd swiglu_fingerprint

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (needed for Llama model access)
# You need to have accepted the Llama license at https://huggingface.co/meta-llama
huggingface-cli login
```

### Model Access

All Llama models require accepting Meta's license on HuggingFace:
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://huggingface.co/meta-llama/Llama-3.2-3B
- https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- https://huggingface.co/meta-llama/Meta-Llama-3.1-70B

Accept the license, then your HuggingFace token will grant download access.

---

## Phase 1: Tier 1 Sanity Check (1B vs 3B)

### Step 1: Extract fingerprints from the small model

```bash
python extract_activations.py \
    --model meta-llama/Llama-3.2-1B \
    --tag 1b \
    --num-prompts 500 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data
```

### Step 2: Extract fingerprints from the larger model

```bash
python extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --tag 3b \
    --num-prompts 500 \
    --max-new-tokens 32 \
    --output-dir ./fingerprint_data
```

### Step 3: Run the comparison analysis

```bash
python analyze_fingerprints.py \
    --model-a ./fingerprint_data/1b \
    --model-b ./fingerprint_data/3b \
    --output ./analysis_results/1b_vs_3b
```

### What to look for

Open `./analysis_results/1b_vs_3b/` and examine:

1. **summary.json** — The headline number is `random_forest_accuracy`. You want
   this above 0.90 to proceed. Above 0.95 is excellent.

2. **compare_sparsity_ratio.png** — Do the two models have visibly different
   sparsity profiles across layers? The blue and red curves should be clearly
   separated, not overlapping.

3. **compare_gini_mean.png** — Same question for Gini coefficient. This measures
   how "spiky" the activation pattern is. The larger model should show higher
   Gini (sparser, more selective activations).

4. **per_layer_accuracy.png** — Which layers are most discriminative? If deep
   layers are more discriminative than shallow layers, that's consistent with
   the theory (deep layers show the most divergence between model capacities).

5. **feature_importance.png** — Which specific statistics at which layers matter
   most? This tells you what to focus on for the fingerprint specification.

6. **trajectory_curvature.png** — Does the larger model show higher trajectory
   curvature (more refinement per layer)? If yes, this supports the trajectory-
   based verification approach.

### Decision gate

- Classification accuracy > 0.95 → **Strong signal. Proceed to Tier 2.**
- Classification accuracy 0.80-0.95 → **Promising. Proceed, but note which
  statistics drive the separation and consider adding more features.**
- Classification accuracy 0.55-0.80 → **Weak. Investigate why. Consider
  capturing more detailed statistics (full activation histograms, cross-layer
  correlations). May need to rethink the approach.**
- Classification accuracy < 0.55 → **No signal. The fingerprint doesn't work
  at this level. Consider whether the models are too similar (1B vs 3B may
  not be different enough) or whether the statistics are too lossy.**

---

## Phase 2: Quantization Robustness (same model, different precision)

This phase tests whether honest implementation differences (quantization)
produce fingerprint variation that could be confused with model substitution.

### Run the same model at fp16 and 4-bit

```bash
# Full precision (fp16)
python extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --tag 8b-fp16 \
    --num-prompts 500 \
    --output-dir ./fingerprint_data

# 4-bit quantized
python extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --tag 8b-4bit \
    --num-prompts 500 \
    --quantize 4bit \
    --output-dir ./fingerprint_data
```

### Analyze

```bash
python analyze_fingerprints.py \
    --model-a ./fingerprint_data/8b-fp16 \
    --model-b ./fingerprint_data/8b-4bit \
    --output ./analysis_results/8b_fp16_vs_4bit
```

### What you want to see

You want the classifier to have LOW accuracy here (ideally below 0.70). This
means the fingerprint is robust to quantization — it sees both as "the same
model." If the classifier easily distinguishes fp16 from 4-bit, your
fingerprint is too sensitive to implementation details and will produce false
positives in production.

If quantization sensitivity is too high, you'll need to:
- Use coarser-grained statistics (fewer histogram bins, wider thresholds)
- Normalize activations before computing statistics
- Focus on rank-order statistics (which neurons are most active) rather
  than magnitude-based statistics

---

## Phase 3: The Real Test (8B vs 70B)

```bash
# 8B at full precision
python extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --tag 8b \
    --num-prompts 1000 \
    --output-dir ./fingerprint_data

# 70B at 4-bit (most realistic attacker scenario: they'd quantize to save cost)
python extract_activations.py \
    --model meta-llama/Meta-Llama-3.1-70B \
    --tag 70b-4bit \
    --num-prompts 1000 \
    --quantize 4bit \
    --output-dir ./fingerprint_data
```

### The key comparison matrix

Run analysis on all relevant pairs:

```bash
# The threat model: can we catch 8B pretending to be 70B?
python analyze_fingerprints.py \
    --model-a ./fingerprint_data/70b-4bit \
    --model-b ./fingerprint_data/8b \
    --output ./analysis_results/70b_vs_8b

# Sanity: does the same model match itself?
# (Run 8B twice on different prompts and compare)
python analyze_fingerprints.py \
    --model-a ./fingerprint_data/8b \
    --model-b ./fingerprint_data/8b \
    --output ./analysis_results/8b_vs_8b_self
```

### Success criteria

The experiment is a success if ALL of these hold:

1. **70B vs 8B classification accuracy > 0.95** — models are clearly
   distinguishable from activation statistics alone.

2. **8B-fp16 vs 8B-4bit classification accuracy < 0.70** — quantization
   doesn't break the fingerprint (low false positive rate).

3. **8B vs 8B-self classification accuracy ≈ 0.50** — the same model's
   fingerprint is consistent across different inputs (no self-confusion).

4. **The most discriminative features are structural, not superficial** —
   features like energy_concentration, gini, and deep-layer sparsity should
   dominate, not raw magnitudes (which are sensitive to scaling).

---

## Phase 4: Extended Analysis (if Phase 3 succeeds)

### Cross-family comparison

Test with non-Llama models to see if the fingerprint generalizes:

```bash
# Mistral 7B (similar size to Llama 8B, different architecture details)
python extract_activations.py \
    --model mistralai/Mistral-7B-v0.3 \
    --tag mistral-7b \
    --num-prompts 1000 \
    --output-dir ./fingerprint_data
```

Can the fingerprint tell Llama-8B from Mistral-7B? (It should, easily,
since they have different weight matrices.)

### Distilled model comparison

The hardest test: compare a model against its own distillation.

If you can find or create a distilled version of one of these models (e.g.,
using the `distilbert`-style approach of dropping alternate layers), compare
the teacher and student. This is the closest simulation of the actual attack.

### Adversarial fine-tuning test

Fine-tune a small model to minimize activation fingerprint divergence from
the large model. This requires:
1. Collecting activation statistics from the large model on a training set
2. Adding a "fingerprint matching" loss term to the small model's training
3. Seeing if the fine-tuned small model can fool the classifier

This is the hardest test and the most informative. If it fails (the small
model can't match the fingerprint even with adversarial training), the
approach is fundamentally sound.

---

## Interpreting Results for the Proof-of-Inference Application

Once you have results, map them to the verification system design:

| Experimental Result | Implication for Verification |
|---|---|
| High classification accuracy (>0.95) | Fingerprint is discriminative enough for fraud detection |
| Low quantization sensitivity (<0.70) | System can tolerate honest implementation differences |
| Deep layers most discriminative | Focus verification on deep-layer statistics (cheaper) |
| Gini/sparsity most important features | These are cheap to compute (minimal inference overhead) |
| Trajectory curvature differs | Layer-to-layer dynamics carry signal (supports TaT approach) |
| Top-k neuron overlap is low | Neuron identity is model-specific (hard to forge) |

## Output Artifacts

After running all phases, you should have:

1. A set of plots showing clear separation (or lack thereof) between models
2. A `summary.json` with quantitative separation metrics
3. Feature importance rankings telling you exactly what to measure
4. A clear go/no-go answer on whether the approach warrants further investment

This is enough to write a workshop paper or technical report, which is the
right format for socializing the idea before building the full system.
