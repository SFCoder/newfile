"""
Per-layer attention skip vulnerability profile for ReluLLaMA-70B.

Measures how much attention computation can be skipped (zeroed out) at each
layer — and across combinations of layers — before output quality degrades.
This is the ReluLLaMA-70B equivalent of the max_savings_test.py experiment
run on Qwen 7B.

Experiment design
-----------------
Phase 1 – Baselines:
    Generate 30 tokens (greedy) per prompt with full attention, storing
    token IDs and the prefill hidden state.

Phase 2 – Single-layer skips:
    Zero attention output at each of 9 sampled layers (0, 10, 20, 30, 40,
    50, 60, 70, 79), run all 9 prompts at each layer.  81 results total.

Phase 3 – Multi-layer skips:
    For skip counts [2, 3, 5, 7, 10, 14, 28] × 7 strategies:
      best_case         – skip the N "safest" layers (highest single-skip
                          token-match-rate from Phase 2; greedy ranking)
      random_seed_{0,1,2}
      positional_first  – skip layers 0..N-1
      positional_middle – skip N layers centred on layer 40
      positional_last   – skip layers 80-N..79
    9 prompts per combination.  441 results total.

    Combined total ≈ 522 results.

Metrics per result
------------------
- token_match_rate   fraction of generated tokens matching the baseline
- cosine_similarity  cosine of prefill hidden state vs baseline
- savings_pct        skip_count / 80 × 100
- coherence          "coherent" (≥0.8), "degraded" (0.3–0.8), "garbage" (<0.3)

Output
------
analysis_results/max_savings_relu70b/<HARDWARE>_<TIMESTAMP>.json
Partial snapshots written every 50 results to …<TIMESTAMP>.partial.json.

Run on RunPod with 48 GB+ VRAM.  Do not run locally.
"""

import json
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "SparseLLM/ReluLLaMA-70B"
OUTPUT_DIR = "analysis_results/max_savings_relu70b"
NUM_LAYERS = 80
MAX_NEW_TOKENS = 30
PRINT_EVERY = 10
SAVE_EVERY = 50

# 9-prompt set (3 simple / 3 moderate / 3 complex) shared with max_savings_test.py.
_PROMPT_TABLE = [
    # (text, complexity_tier)
    ("The capital of France is",                                              "simple"),
    ("Water boils at",                                                        "simple"),
    ("The sun rises in the",                                                  "simple"),
    ("Explain what a neural network is:",                                     "moderate"),
    ("The difference between supervised and unsupervised learning is",        "moderate"),
    ("In physics, Newton's second law states that",                           "moderate"),
    ("Analyze the key factors that led to the fall of the Roman Empire:",     "complex"),
    ("Describe the relationship between entropy and information theory:",     "complex"),
    ("Compare the economic theories of Keynes and Hayek:",                   "complex"),
]
PROMPT_TEXTS = [p for p, _ in _PROMPT_TABLE]
COMPLEXITY_TIERS = [t for _, t in _PROMPT_TABLE]

# Single-layer sample: 9 evenly-spaced positions across the 80-layer model.
SINGLE_SKIP_LAYERS = [0, 10, 20, 30, 40, 50, 60, 70, 79]

MULTI_SKIP_COUNTS = [2, 3, 5, 7, 10, 14, 28]
MULTI_STRATEGIES = [
    "best_case",
    "random_seed_0",
    "random_seed_1",
    "random_seed_2",
    "positional_first",
    "positional_middle",
    "positional_last",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "_")
    return "CPU"


def coherence_label(token_match: float) -> str:
    if token_match >= 0.8:
        return "coherent"
    if token_match >= 0.3:
        return "degraded"
    return "garbage"


def format_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    secs_remaining = (elapsed / done) * (total - done)
    h, rem = divmod(int(secs_remaining), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_token_match(attack_ids: list, baseline_ids: list) -> float:
    if not baseline_ids:
        return 0.0
    n = min(len(attack_ids), len(baseline_ids))
    matches = sum(a == b for a, b in zip(attack_ids[:n], baseline_ids[:n]))
    return matches / len(baseline_ids)


def compute_cosine(attack_h, baseline_h) -> float:
    if attack_h is None or baseline_h is None:
        return 0.0
    return F.cosine_similarity(
        attack_h.flatten().unsqueeze(0),
        baseline_h.flatten().unsqueeze(0),
    ).item()


# ── Hook machinery ────────────────────────────────────────────────────────────

def make_attn_skip_hook():
    """Forward hook that zeroes the attention output for a single layer."""
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


class PrefillHiddenCapture:
    """
    Captures the last-token hidden state from the *first* forward call only
    (the prefill step).  Subsequent decode-step calls are ignored.

    Hook target: model.model.norm  (output shape [bsz, seq, hidden]).
    """

    def __init__(self):
        self.value = None
        self._captured = False

    def reset(self):
        self.value = None
        self._captured = False

    def __call__(self, module, inp, output):
        if self._captured:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        # Take the last-position hidden state of the first (prefill) call.
        self.value = tensor[:, -1, :].detach().float().cpu()
        self._captured = True


def apply_skip_hooks(model, layer_indices: list) -> list:
    handles = []
    for idx in layer_indices:
        h = model.model.layers[idx].self_attn.register_forward_hook(
            make_attn_skip_hook()
        )
        handles.append(h)
    return handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_generate(model, inputs: dict, hidden_capture: PrefillHiddenCapture):
    """Run model.generate (greedy, max_new_tokens=30).  Return (token_ids, hidden)."""
    hidden_capture.reset()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    token_ids = out[0, input_len:].tolist()
    return token_ids, hidden_capture.value


def run_one(model, tokenizer, prompt: str, skip_layers: list,
            hidden_capture: PrefillHiddenCapture):
    """
    Install skip hooks, run generate, remove hooks.
    Retries once on CUDA OOM.  Returns (token_ids, hidden) or (None, None).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for attempt in range(2):
        handles = apply_skip_hooks(model, skip_layers)
        try:
            token_ids, hidden = run_generate(model, inputs, hidden_capture)
            return token_ids, hidden
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM (attempt {attempt + 1}), clearing cache...")
            torch.cuda.empty_cache()
            if attempt == 1:
                print(f"  Giving up on '{prompt[:40]}'")
                return None, None
        finally:
            remove_hooks(handles)

    return None, None  # unreachable but satisfies linters


# ── Strategy layer selection ──────────────────────────────────────────────────

def strategy_layers(strategy: str, skip_count: int,
                    safest_first: list = None) -> list:
    """
    Return the sorted list of layer indices to skip for a given strategy.

    safest_first: layers ranked from most to least safe to skip, derived from
    Phase-2 single-layer token-match-rates.  Used only by best_case.
    """
    n = skip_count
    mid = NUM_LAYERS // 2  # 40

    if strategy == "best_case":
        if safest_first is not None:
            return sorted(safest_first[:n])
        # Fallback if Phase-2 data is unavailable.
        rng = random.Random(99)
        return sorted(rng.sample(range(NUM_LAYERS), n))

    if strategy == "random_seed_0":
        return sorted(random.Random(0).sample(range(NUM_LAYERS), n))

    if strategy == "random_seed_1":
        return sorted(random.Random(1).sample(range(NUM_LAYERS), n))

    if strategy == "random_seed_2":
        return sorted(random.Random(2).sample(range(NUM_LAYERS), n))

    if strategy == "positional_first":
        return list(range(n))

    if strategy == "positional_middle":
        start = max(0, mid - n // 2)
        end = min(NUM_LAYERS, start + n)
        # If clamped at the high end, shift start back.
        if end - start < n:
            start = max(0, end - n)
        return list(range(start, end))

    if strategy == "positional_last":
        return list(range(NUM_LAYERS - n, NUM_LAYERS))

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Partial-save helper ───────────────────────────────────────────────────────

def save_partial(results: list, hardware: str, timestamp: str,
                 suffix: str = ".partial.json") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{hardware}_{timestamp}{suffix}")
    with open(path, "w") as fh:
        json.dump(results, fh)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer for {MODEL_ID} (use_fast=False) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    print(f"Loading {MODEL_ID} in 4-bit NF4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    hardware = get_hardware_name()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    partial_path = os.path.join(OUTPUT_DIR, f"{hardware}_{timestamp}.partial.json")

    # Hook on the final layer norm to capture prefill hidden states.
    hidden_capture = PrefillHiddenCapture()
    norm_handle = model.model.norm.register_forward_hook(hidden_capture)

    # Total inference calls for progress tracking (excludes the 9 baselines).
    total_runs = (
        len(SINGLE_SKIP_LAYERS) * len(PROMPT_TEXTS)
        + len(MULTI_SKIP_COUNTS) * len(MULTI_STRATEGIES) * len(PROMPT_TEXTS)
    )

    all_results = []
    done = 0
    start_time = time.time()

    try:
        # ── Phase 1: Baselines ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Phase 1 — Generating baselines ({len(PROMPT_TEXTS)} prompts)")
        print(f"{'='*60}")
        baseline_ids = {}
        baseline_hidden = {}

        for p_idx, prompt in enumerate(PROMPT_TEXTS):
            ids, hidden = run_one(model, tokenizer, prompt, [], hidden_capture)
            baseline_ids[p_idx] = ids if ids is not None else []
            baseline_hidden[p_idx] = hidden
            n_tok = len(baseline_ids[p_idx])
            print(f"  [{p_idx}] ({COMPLEXITY_TIERS[p_idx]:8s}) "
                  f"'{prompt[:42]}' → {n_tok} tokens")

        # ── Phase 2: Single-layer skips ───────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Phase 2 — Single-layer skips "
              f"({len(SINGLE_SKIP_LAYERS)} layers × {len(PROMPT_TEXTS)} prompts = "
              f"{len(SINGLE_SKIP_LAYERS) * len(PROMPT_TEXTS)} runs)")
        print(f"{'='*60}")

        # Track per-layer match rates for best_case ranking.
        single_layer_match: dict[int, list] = {l: [] for l in SINGLE_SKIP_LAYERS}

        for layer_idx in SINGLE_SKIP_LAYERS:
            for p_idx, prompt in enumerate(PROMPT_TEXTS):
                attack_ids, attack_hidden = run_one(
                    model, tokenizer, prompt, [layer_idx], hidden_capture
                )
                if attack_ids is None:
                    attack_ids, attack_hidden = [], None

                tmr = compute_token_match(attack_ids, baseline_ids[p_idx])
                cosine = compute_cosine(attack_hidden, baseline_hidden[p_idx])
                savings = 1.0 / NUM_LAYERS * 100.0
                single_layer_match[layer_idx].append(tmr)

                all_results.append({
                    "experiment_type": "single_layer",
                    "prompt": prompt,
                    "complexity_tier": COMPLEXITY_TIERS[p_idx],
                    "layers_skipped": [layer_idx],
                    "skip_strategy": "single",
                    "skip_count": 1,
                    "token_match_rate": tmr,
                    "cosine_similarity": cosine,
                    "savings_pct": savings,
                    "coherence": coherence_label(tmr),
                    "timestamp": timestamp,
                })
                done += 1

                if done % PRINT_EVERY == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{done:3d}/{total_runs}] "
                          f"layer={layer_idx:2d} p={p_idx} "
                          f"match={tmr:.3f} cosine={cosine:.4f} "
                          f"elapsed={elapsed:.0f}s ETA={format_eta(elapsed, done, total_runs)}")
                if done % SAVE_EVERY == 0:
                    save_partial(all_results, hardware, timestamp)
                    print(f"  ↳ partial save ({len(all_results)} results)")

        # ── Build best_case layer ranking from Phase-2 results ────────────────
        # For layers in SINGLE_SKIP_LAYERS: use actual mean token-match-rate.
        # For un-tested layers: interpolate from neighbours or use global mean.
        layer_mean: dict[int, float] = {}
        for l in SINGLE_SKIP_LAYERS:
            vals = single_layer_match[l]
            layer_mean[l] = sum(vals) / len(vals) if vals else 0.0

        # Interpolate for untested layers using nearest tested neighbour.
        tested_sorted = sorted(SINGLE_SKIP_LAYERS)
        for l in range(NUM_LAYERS):
            if l in layer_mean:
                continue
            # Find nearest tested layer.
            nearest = min(tested_sorted, key=lambda t: abs(t - l))
            layer_mean[l] = layer_mean[nearest]

        # safest_first: layers sorted by descending mean token-match
        # (highest match = skipping that layer costs the least).
        safest_first = sorted(range(NUM_LAYERS),
                              key=lambda l: layer_mean[l], reverse=True)

        print(f"\nTop-10 safest layers to skip (by Phase-2 mean token-match): "
              f"{safest_first[:10]}")

        # ── Phase 3: Multi-layer skips ────────────────────────────────────────
        n_multi = len(MULTI_SKIP_COUNTS) * len(MULTI_STRATEGIES) * len(PROMPT_TEXTS)
        print(f"\n{'='*60}")
        print(f"Phase 3 — Multi-layer skips "
              f"({len(MULTI_SKIP_COUNTS)} counts × {len(MULTI_STRATEGIES)} strategies "
              f"× {len(PROMPT_TEXTS)} prompts = {n_multi} runs)")
        print(f"{'='*60}")

        for skip_count in MULTI_SKIP_COUNTS:
            for strategy in MULTI_STRATEGIES:
                layers = strategy_layers(strategy, skip_count, safest_first)
                savings = skip_count / NUM_LAYERS * 100.0

                for p_idx, prompt in enumerate(PROMPT_TEXTS):
                    attack_ids, attack_hidden = run_one(
                        model, tokenizer, prompt, layers, hidden_capture
                    )
                    if attack_ids is None:
                        attack_ids, attack_hidden = [], None

                    tmr = compute_token_match(attack_ids, baseline_ids[p_idx])
                    cosine = compute_cosine(attack_hidden, baseline_hidden[p_idx])

                    all_results.append({
                        "experiment_type": "multi_layer",
                        "prompt": prompt,
                        "complexity_tier": COMPLEXITY_TIERS[p_idx],
                        "layers_skipped": layers,
                        "skip_strategy": strategy,
                        "skip_count": skip_count,
                        "token_match_rate": tmr,
                        "cosine_similarity": cosine,
                        "savings_pct": savings,
                        "coherence": coherence_label(tmr),
                        "timestamp": timestamp,
                    })
                    done += 1

                    if done % PRINT_EVERY == 0:
                        elapsed = time.time() - start_time
                        print(f"  [{done:3d}/{total_runs}] "
                              f"skip={skip_count:2d} {strategy:20s} p={p_idx} "
                              f"match={tmr:.3f} cosine={cosine:.4f} "
                              f"elapsed={elapsed:.0f}s "
                              f"ETA={format_eta(elapsed, done, total_runs)}")
                    if done % SAVE_EVERY == 0:
                        save_partial(all_results, hardware, timestamp)
                        print(f"  ↳ partial save ({len(all_results)} results)")

    except KeyboardInterrupt:
        print(f"\n--- Interrupted after {done} runs ---")
        path = save_partial(all_results, hardware, timestamp)
        print(f"Partial results saved to {path}")
        return

    finally:
        norm_handle.remove()

    # ── Summaries ─────────────────────────────────────────────────────────────

    # Per-layer mean token-match (single-skip).
    per_layer_mean = {}
    for l in SINGLE_SKIP_LAYERS:
        vals = [r["token_match_rate"] for r in all_results
                if r["experiment_type"] == "single_layer"
                and r["layers_skipped"] == [l]]
        per_layer_mean[str(l)] = round(sum(vals) / len(vals), 6) if vals else 0.0

    # Per-skip-count mean token-match (multi-skip, all strategies combined).
    per_skip_count_mean = {}
    for sc in MULTI_SKIP_COUNTS:
        vals = [r["token_match_rate"] for r in all_results
                if r["experiment_type"] == "multi_layer"
                and r["skip_count"] == sc]
        per_skip_count_mean[str(sc)] = round(sum(vals) / len(vals), 6) if vals else 0.0

    # Maximum savings_pct while output remained coherent (token_match >= 0.8).
    coherent = [r for r in all_results if r["coherence"] == "coherent"]
    max_savings_coherent = max((r["savings_pct"] for r in coherent), default=0.0)

    output = {
        "hardware": hardware,
        "timestamp": timestamp,
        "model": MODEL_ID,
        "experiment": "max_savings_relu70b",
        "total_results": len(all_results),
        "results": all_results,
        "summary": {
            "per_layer_mean_token_match_rate": per_layer_mean,
            "per_skip_count_mean_token_match_rate": per_skip_count_mean,
            "max_savings_pct_coherent": round(max_savings_coherent, 4),
        },
    }

    fname = f"{hardware}_{timestamp}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w") as fh:
        json.dump(output, fh, indent=2)

    # Remove partial file if present.
    if os.path.exists(partial_path):
        os.remove(partial_path)

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"Done in {h:02d}:{m:02d}:{s:02d}")
    print(f"Saved   : {fpath}")
    print(f"Results : {len(all_results)}")
    print(f"Max coherent savings: {max_savings_coherent:.2f}%")
    print(f"Per-layer summary   : {per_layer_mean}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
