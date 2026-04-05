#!/usr/bin/env python3
"""
Per-Inference Inactive Neuron Gate Value Profiling at 72B
==========================================================
Profiles the gate value distribution of per-inference inactive neurons for
Qwen 2.5 72B. Unlike Experiment 9, which sampled gate values at fixed token
positions, this experiment tracks the MAXIMUM absolute gate value each neuron
achieves across the ENTIRE inference — prompt processing and all generated
tokens — and classifies only neurons that never exceed the threshold as
per-inference inactive.

These are the neurons that appear in provider.py's union mask and are the
actual encoding candidates for the blockchain consensus mechanism.

Usage:
    python3 tools/gate_value_distribution_v2.py
    python3 tools/gate_value_distribution_v2.py --model-id /path/to/model
    python3 tools/gate_value_distribution_v2.py --threshold 0.05 --num-tokens 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
ACTIVITY_THRESHOLD = 0.05   # primary; matches provider.py threshold sweep
NUM_TOKENS = 10             # generated tokens per prompt (>=10 as required)
EXPECTED_INTERMEDIATE_SIZE = 29568
EXPECTED_NUM_LAYERS = 80

OUTPUT_DIR = _ROOT / "analysis_results" / "gate_value_distribution_v2"
VECTORS_DIR = OUTPUT_DIR / "vectors"

# ---------------------------------------------------------------------------
# Prompt set — identical to tools/gate_value_distribution.py
# ---------------------------------------------------------------------------

PROMPTS = {
    "simple_factual": [
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "What year did World War II end?",
        "Define the word 'entropy'.",
    ],
    "complex_reasoning": [
        "Explain why the sky is blue using physics principles.",
        "Compare and contrast democracy and authoritarianism.",
        "If a train leaves Chicago at 60mph and another leaves New York at 80mph, when do they meet? The distance is 790 miles.",
        "What are the second-order effects of raising the minimum wage?",
    ],
    "creative_writing": [
        "Write the opening paragraph of a mystery novel set in Tokyo.",
        "Compose a metaphor comparing the internet to an ocean.",
        "Describe a sunset from the perspective of a blind person who just regained sight.",
        "Write a dialogue between two AIs debating consciousness.",
    ],
    "code_technical": [
        "Write a Python function to find the longest palindromic substring.",
        "Explain how a B-tree index works in a database.",
        "What is the difference between TCP and UDP?",
        "Describe the CAP theorem and its implications.",
    ],
    "long_context": [
        (
            "The following is an excerpt from a technical report on climate change impacts: "
            "Rising global temperatures have led to significant changes in precipitation patterns worldwide. "
            "Coastal regions face increasing flood risks while inland areas experience prolonged droughts. "
            "Agricultural systems are particularly vulnerable, with crop yields declining in tropical regions. "
            "Adaptation strategies include improved irrigation, drought-resistant crop varieties, and revised "
            "planting schedules. However, the pace of climate change may outstrip the capacity for adaptation "
            "in the most vulnerable regions. Economic models suggest that without intervention, climate-related "
            "damages could reach 2-4% of global GDP annually by 2050. Summarize the key arguments."
        ),
        (
            "The software specification states: The system shall process incoming API requests through a "
            "three-stage pipeline. Stage one validates authentication tokens against the identity provider. "
            "Stage two applies rate limiting based on the caller's subscription tier, with burst allowances "
            "of 2x the base rate for premium tiers. Stage three routes the request to the appropriate "
            "microservice based on the URL path prefix. All stages must complete within 50ms for the 99th "
            "percentile latency target. Failures at any stage return appropriate HTTP status codes with "
            "structured error bodies. The system must handle graceful degradation when downstream services "
            "are unavailable, falling back to cached responses where possible. Identify potential issues in "
            "this specification."
        ),
        (
            "The story begins: Marcus stood at the edge of the pier, watching the last ferry disappear into "
            "the morning fog. The letter in his pocket felt heavier than paper should. Three years of silence, "
            "and now this — a single page, handwritten, from someone he had convinced himself no longer "
            "existed. The harbor smelled of diesel and salt, the same as it always had, but everything else "
            "had changed. The coffee shop where they used to meet was a bank now. The bookstore was luxury "
            "apartments. Even the lighthouse had been automated, its keeper's cottage converted into an "
            "Airbnb. He unfolded the letter again, though he had already memorized every word. Continue this "
            "story for two paragraphs."
        ),
        (
            "The legal clause reads: Notwithstanding any provision to the contrary contained herein, the "
            "indemnifying party shall defend, indemnify, and hold harmless the indemnified party and its "
            "officers, directors, employees, agents, successors, and assigns from and against any and all "
            "claims, damages, losses, costs, and expenses (including reasonable attorneys' fees) arising out "
            "of or relating to any breach of representation, warranty, or obligation under this Agreement, "
            "provided that the indemnified party gives prompt written notice of any such claim and provides "
            "reasonable cooperation in the defense thereof. The indemnifying party shall have sole control "
            "of the defense and settlement of any such claim, provided that no settlement shall require any "
            "admission of liability or payment by the indemnified party without its prior written consent. "
            "Explain this in plain English."
        ),
    ],
    "adversarial_edge": [
        "Hi",
        "the the the the the the the the the the",
        "flurbo gaxnip tremolo wistful quazar blenching",
        "3.14159 2.71828 1.41421 1.61803",
        "Hello, comment allez-vous today? Ich bin gut, gracias.",
        ".",
    ],
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_id: str = MODEL_ID):
    """Load model in NF4 4-bit quantization via bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_id} with NF4 4-bit quantization ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Resolve snapshot path via registry if model_id is a HF repo name.
    model_path: str = model_id
    registry = None
    if not Path(model_id).exists():
        try:
            from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH
            registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
            if model_id in registry.list_models():
                entry = registry.get_entry(model_id)
                snapshot = ModelRegistry._snapshot_path(entry.hf_repo)
                model_path = str(snapshot)
                print(f"  Registry snapshot: {model_path}")
            else:
                print(f"  {model_id!r} not in registry — loading from HF cache")
        except Exception as exc:
            print(f"  Registry lookup failed ({exc}); loading from HF cache")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    print(f"  Loaded: {num_layers} layers, intermediate_size={intermediate_size}")
    if intermediate_size != EXPECTED_INTERMEDIATE_SIZE:
        print(f"  WARNING: expected {EXPECTED_INTERMEDIATE_SIZE}, got {intermediate_size}")
    return model, tokenizer, registry


# ---------------------------------------------------------------------------
# Per-inference max gate accumulation
# ---------------------------------------------------------------------------


def profile_prompt(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int,
    threshold: float,
) -> tuple[list[np.ndarray], int, int]:
    """
    Run full inference (prompt + generation) and return per-layer max-gate
    accumulation vectors.

    For each layer, the returned array is the element-wise maximum absolute
    gate value each neuron achieved across EVERY token (prompt and generated).
    Only neurons whose maximum never exceeded `threshold` are per-inference
    inactive.

    Returns
    -------
    max_accum : list of np.ndarray, shape (intermediate_size,), float32
        One array per layer. max_accum[l][n] = max |gate| neuron n saw.
    prompt_token_count : int
    generated_token_count : int
    """
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size

    # Running max per neuron per layer, initialised to zero on CPU.
    max_accum = [
        np.zeros(intermediate_size, dtype=np.float32)
        for _ in range(num_layers)
    ]

    def make_hook(layer_idx: int):
        accum = max_accum[layer_idx]

        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]  # [batch, seq_len, hidden_size]
            with torch.no_grad():
                gate = module.act_fn(module.gate_proj(x)).abs()
                # gate: [batch, seq_len, intermediate_size]
                if gate.dim() == 3:
                    # Collapse batch and seq dimensions — we want the max over
                    # all token positions in this forward pass.
                    gate_max = gate[0].max(dim=0).values
                elif gate.dim() == 2:
                    gate_max = gate.max(dim=0).values
                else:
                    gate_max = gate.abs()
                # Update running max in-place (CPU numpy).
                gate_np = gate_max.float().cpu().numpy()
                np.maximum(accum, gate_np, out=accum)
                del gate, gate_max, gate_np
            return output  # pass through unmodified

        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

    # Determine input device (embedding layer).
    try:
        input_device = model.model.embed_tokens.weight.device
    except AttributeError:
        input_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    try:
        enc = tokenizer(prompt, return_tensors="pt").to(input_device)
        prompt_len = int(enc["input_ids"].shape[1])

        with torch.no_grad():
            out = model.generate(
                enc["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_len = int(out.shape[1]) - prompt_len
    finally:
        for h in hooks:
            h.remove()

    return max_accum, prompt_len, generated_len


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_layer_stats(max_gate: np.ndarray, threshold: float) -> dict:
    """
    Compute per-inference inactive/active statistics from the max-gate vector.

    Parameters
    ----------
    max_gate : 1-D float32 array, shape (intermediate_size,)
        Element i = maximum absolute gate value neuron i achieved across the
        entire inference.
    threshold : float
        Activity threshold; neurons with max_gate < threshold are inactive.
    """
    inactive_mask = max_gate < threshold
    active_mask   = ~inactive_mask

    inactive_vals = max_gate[inactive_mask]
    active_vals   = max_gate[active_mask]

    inactive_count = int(inactive_mask.sum())
    active_count   = int(active_mask.sum())

    def _stats(arr: np.ndarray, include_std: bool = True) -> dict:
        if len(arr) == 0:
            d = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
            if include_std:
                d["std"] = 0.0
            return d
        d = {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
        }
        if include_std:
            d["std"] = float(arr.std())
        return d

    # Inactive histogram: bins over [0, threshold)
    inactive_histogram = {
        "0.000_to_0.001": int(((max_gate >= 0.0)   & (max_gate < 0.001)).sum()),
        "0.001_to_0.005": int(((max_gate >= 0.001) & (max_gate < 0.005)).sum()),
        "0.005_to_0.010": int(((max_gate >= 0.005) & (max_gate < 0.010)).sum()),
        "0.010_to_0.020": int(((max_gate >= 0.010) & (max_gate < 0.020)).sum()),
        "0.020_to_0.050": int(((max_gate >= 0.020) & (max_gate < threshold)).sum()),
    }

    encoding_capacity = {
        "below_0.001": int((max_gate < 0.001).sum()),
        "below_0.005": int((max_gate < 0.005).sum()),
        "below_0.010": int((max_gate < 0.010).sum()),
        "below_0.020": int((max_gate < 0.020).sum()),
    }

    # Gap: how cleanly separated are the two populations?
    gap = float(active_vals.min()) - float(inactive_vals.max()) if (
        len(active_vals) > 0 and len(inactive_vals) > 0
    ) else 0.0

    return {
        "per_inference_inactive_count": inactive_count,
        "per_inference_active_count":   active_count,
        "inactive_max_gate_stats":      _stats(inactive_vals, include_std=True),
        "active_max_gate_stats":        _stats(active_vals,   include_std=False),
        "gap":                          gap,
        "inactive_histogram":           inactive_histogram,
        "encoding_capacity":            encoding_capacity,
    }


# ---------------------------------------------------------------------------
# Prompt list builder
# ---------------------------------------------------------------------------


def build_prompt_list() -> list[tuple[str, str, str]]:
    """Return [(prompt_id, category, prompt_text), ...]."""
    out = []
    for category, texts in PROMPTS.items():
        for i, text in enumerate(texts, start=1):
            out.append((f"{category}_{i:02d}", category, text))
    return out


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    model,
    tokenizer,
    registry,
    threshold: float,
    num_tokens: int,
) -> tuple[list, int, int]:
    """
    Profile per-inference inactive neurons for all 26 prompts.

    Returns (all_results, num_layers, intermediate_size).
    all_results is a list of (prompt_id, category, layers_stats_list).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    num_layers        = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    prompt_list       = build_prompt_list()

    print(f"\nModel layers: {num_layers}  |  intermediate_size: {intermediate_size}")
    print(f"Threshold: {threshold}  |  Generated tokens: {num_tokens}")
    print(f"Prompts: {len(prompt_list)}\n")

    # ---- optional DB writer ----
    writer = None
    try:
        from adversarial_suite.db.writer import ResultsWriter
        writer = ResultsWriter(
            "gate_value_distribution_v2",
            script_path="tools/gate_value_distribution_v2.py",
            notes=f"threshold={threshold} num_tokens={num_tokens}",
        )
        writer.__enter__()
        db_model_id = writer.ensure_model(
            MODEL_ID,
            registry=registry,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
        )
        print("  DB writer active.")
    except Exception as exc:
        print(f"  DB writer unavailable ({exc}); continuing without DB.")
        writer = None
        db_model_id = None

    all_results = []

    for prompt_idx, (prompt_id, category, prompt_text) in enumerate(prompt_list):
        print(
            f"[{prompt_idx + 1:2d}/{len(prompt_list)}] {prompt_id}: "
            f"{prompt_text[:60]}..."
        )
        t0 = time.time()

        max_accum, prompt_len, gen_len = profile_prompt(
            model, tokenizer, prompt_text, num_tokens, threshold
        )
        total_tokens = prompt_len + gen_len

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  ({prompt_len} prompt + {gen_len} gen tokens)")

        # ---- per-layer stats ----
        layers_stats = []
        for layer_idx, max_gate in enumerate(max_accum):
            stats = compute_layer_stats(max_gate, threshold)

            # Save numpy vector for this layer.
            vec_path = VECTORS_DIR / f"{prompt_id}_layer{layer_idx:02d}.npy"
            np.save(str(vec_path), max_gate)

            layers_stats.append({"layer": layer_idx, **stats})

            # Write to DB (one row per prompt x layer).
            if writer is not None:
                try:
                    db_prompt_id = writer.ensure_prompt(
                        prompt_text,
                        category=category,
                    )
                    writer.add_result(
                        model_id=db_model_id,
                        prompt_id=db_prompt_id,
                        attack_type="gate_value_profiling",
                        layer=layer_idx,
                        raw_data={
                            "inactive_count":              stats["per_inference_inactive_count"],
                            "encoding_capacity_below_0.01": stats["encoding_capacity"]["below_0.010"],
                            "inactive_max":                stats["inactive_max_gate_stats"]["max"],
                            "active_min":                  stats["active_max_gate_stats"]["min"],
                            "gap":                         stats["gap"],
                        },
                    )
                except Exception as db_exc:
                    print(f"  DB write error (layer {layer_idx}): {db_exc}")

        # ---- save JSON ----
        result_json = {
            "experiment":            "gate_value_distribution_v2",
            "model":                 "Qwen2.5-72B",
            "quantization":          "nf4",
            "prompt_id":             prompt_id,
            "prompt_category":       category,
            "prompt_text":           prompt_text,
            "activity_threshold":    threshold,
            "total_neurons":         intermediate_size,
            "num_layers":            num_layers,
            "total_tokens_processed": total_tokens,
            "prompt_tokens":         prompt_len,
            "generated_tokens":      gen_len,
            "layers":                layers_stats,
        }
        json_path = OUTPUT_DIR / f"{prompt_id}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result_json, fh, indent=2)

        all_results.append((prompt_id, category, layers_stats))
        print(f"  Saved {json_path.name}")

        # Free GPU memory.
        del max_accum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if writer is not None:
        try:
            writer.__exit__(None, None, None)
        except Exception as exc:
            print(f"  DB close error: {exc}")

    return all_results, num_layers, intermediate_size


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def compute_summary(
    all_results: list,
    num_layers: int,
    intermediate_size: int,
    threshold: float,
) -> dict:
    """Aggregate per-prompt per-layer stats into cross-prompt summary."""

    # per-layer accumulators (indexed by layer_idx)
    layer_inactive_counts   = {l: [] for l in range(num_layers)}
    layer_enc_below_001     = {l: [] for l in range(num_layers)}

    # global accumulators
    all_inactive_maxes = []
    all_active_mins    = []
    all_enc_001        = []   # encoding capacity at 0.001, per layer per prompt

    for _pid, _cat, layers_stats in all_results:
        for ls in layers_stats:
            li  = ls["layer"]
            ic  = ls["per_inference_inactive_count"]
            enc = ls["encoding_capacity"]
            layer_inactive_counts[li].append(ic)
            layer_enc_below_001[li].append(enc["below_0.010"])
            all_enc_001.append(enc["below_0.010"])
            imax = ls["inactive_max_gate_stats"]["max"]
            amin = ls["active_max_gate_stats"]["min"]
            if imax > 0:
                all_inactive_maxes.append(imax)
            if amin > 0:
                all_active_mins.append(amin)

    # Encoding capacity minimums across all prompts+layers
    enc_0001_vals = [ls["encoding_capacity"]["below_0.001"] for _, _, lss in all_results for ls in lss]
    enc_0005_vals = [ls["encoding_capacity"]["below_0.005"] for _, _, lss in all_results for ls in lss]
    enc_001_vals  = [ls["encoding_capacity"]["below_0.010"] for _, _, lss in all_results for ls in lss]
    enc_002_vals  = [ls["encoding_capacity"]["below_0.020"] for _, _, lss in all_results for ls in lss]

    def kb(n: int) -> float:
        return (n / 8) * num_layers / 1024

    min_enc_001 = int(min(enc_001_vals)) if enc_001_vals else 0
    max_enc_001 = int(max(enc_001_vals)) if enc_001_vals else 0
    mean_enc_001 = float(np.mean(enc_001_vals)) if enc_001_vals else 0.0
    cv_enc_001   = float(np.std(enc_001_vals) / np.mean(enc_001_vals)) if np.mean(enc_001_vals) > 0 else 0.0

    max_inactive = float(max(all_inactive_maxes)) if all_inactive_maxes else 0.0
    min_active   = float(min(all_active_mins))    if all_active_mins    else 0.0
    gap          = min_active - max_inactive

    # Worst-case prompt (minimum encoding capacity at 0.01 across all layers)
    worst_prompt_id = "n/a"
    worst_category  = "n/a"
    worst_cap       = min_enc_001
    for pid, cat, lss in all_results:
        prompt_min = min(ls["encoding_capacity"]["below_0.010"] for ls in lss)
        if prompt_min <= worst_cap:
            worst_cap       = prompt_min
            worst_prompt_id = pid
            worst_category  = cat

    # Per-layer averages (for display)
    layer_avg_inactive = {
        li: float(np.mean(counts)) for li, counts in layer_inactive_counts.items()
    }
    layer_avg_enc_001 = {
        li: float(np.mean(counts)) for li, counts in layer_enc_below_001.items()
    }

    return {
        "min_enc_0001": int(min(enc_0001_vals)) if enc_0001_vals else 0,
        "min_enc_0005": int(min(enc_0005_vals)) if enc_0005_vals else 0,
        "min_enc_001":  min_enc_001,
        "min_enc_002":  int(min(enc_002_vals)) if enc_002_vals else 0,
        "kb_0001":      kb(int(min(enc_0001_vals)) if enc_0001_vals else 0),
        "kb_0005":      kb(int(min(enc_0005_vals)) if enc_0005_vals else 0),
        "kb_001":       kb(min_enc_001),
        "kb_002":       kb(int(min(enc_002_vals)) if enc_002_vals else 0),
        "max_inactive": max_inactive,
        "min_active":   min_active,
        "gap":          gap,
        "cv_enc_001":   cv_enc_001,
        "min_enc_001_val":  min_enc_001,
        "max_enc_001_val":  max_enc_001,
        "mean_enc_001_val": mean_enc_001,
        "worst_prompt_id":  worst_prompt_id,
        "worst_category":   worst_category,
        "worst_cap":        worst_cap,
        "layer_avg_inactive": layer_avg_inactive,
        "layer_avg_enc_001":  layer_avg_enc_001,
    }


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

DISPLAY_LAYERS = [0, 10, 20, 30, 40, 50, 60, 70, 79]


def print_summary(s: dict, num_layers: int, intermediate_size: int) -> None:
    print("\n" + "=" * 60)
    print("=== Per-Inference Gate Value Distribution Summary ===")
    print("=" * 60)

    print("\nPer-inference inactive neurons (never fire across entire inference):")
    avg_inactive = s["layer_avg_inactive"]
    avg_enc_001  = s["layer_avg_enc_001"]
    for li in DISPLAY_LAYERS:
        if li >= num_layers:
            continue
        ic  = avg_inactive.get(li, 0)
        enc = avg_enc_001.get(li, 0)
        pct = 100.0 * ic / intermediate_size if intermediate_size > 0 else 0.0
        print(
            f"  Layer {li:2d}: {ic:6.0f} inactive ({pct:4.1f}%) "
            f"— encoding below 0.01: {enc:6.0f}"
        )

    print("\nEncoding Capacity (neurons with max gate < threshold, MINIMUM across all prompts):")
    print(f"  Threshold 0.001: {s['min_enc_0001']:5d} neurons/layer ({s['kb_0001']:6.1f} KB/inference)")
    print(f"  Threshold 0.005: {s['min_enc_0005']:5d} neurons/layer ({s['kb_0005']:6.1f} KB/inference)")
    print(f"  Threshold 0.010: {s['min_enc_001']:5d} neurons/layer ({s['kb_001']:6.1f} KB/inference)")
    print(f"  Threshold 0.020: {s['min_enc_002']:5d} neurons/layer ({s['kb_002']:6.1f} KB/inference)")

    print("\nDistribution Gap (between per-inference inactive and active):")
    print(f"  Max gate value among inactive neurons (at 0.01 threshold): {s['max_inactive']:.5f}")
    print(f"  Min gate value among active neurons:                       {s['min_active']:.5f}")
    print(f"  Gap:                                                       {s['gap']:.5f}")
    print(f"  Cross-hardware tolerance (from prior experiments):         0.008")
    safety_margin = s["gap"] - 0.008
    print(f"  Safety margin:                                             {safety_margin:.5f}")

    print("\nCross-prompt consistency (encoding capacity at 0.01):")
    print(
        f"  Min:  {s['min_enc_001_val']:5d}  "
        f"Max: {s['max_enc_001_val']:5d}  "
        f"Mean: {s['mean_enc_001_val']:.0f}  "
        f"CV: {s['cv_enc_001'] * 100:.1f}%"
    )

    worst_total = s["worst_cap"] * num_layers
    worst_kb    = worst_total / 8 / 1024
    print(f"\nWorst-case prompt: {s['worst_prompt_id']} ({s['worst_category']})")
    print(f"  Encoding capacity at 0.01: {s['worst_cap']} neurons/layer")
    print(f"  Total across {num_layers} layers: {worst_total} neurons ({worst_kb:.1f} KB)")

    cap_pass  = s["min_enc_001"]  >= 1000
    gap_pass  = s["gap"]          >= 0.03
    cv_pass   = s["cv_enc_001"]   <  0.30
    all_pass  = cap_pass and gap_pass and cv_pass

    print(f"\nVERDICT: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Encoding capacity >= 1000 at 0.01: {'YES' if cap_pass else 'NO'}")
    print(f"  Distribution gap >= 0.03:           {'YES' if gap_pass else 'NO'}")
    print(f"  Cross-prompt variation < 30%:       {'YES' if cv_pass else 'NO'}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-inference inactive neuron gate value profiling for Qwen 2.5 72B"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HF repo or local snapshot path (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--threshold", type=float, default=ACTIVITY_THRESHOLD,
        help=f"Activity threshold (default: {ACTIVITY_THRESHOLD})"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=NUM_TOKENS,
        help=f"Tokens to generate per prompt (default: {NUM_TOKENS})"
    )
    parser.add_argument(
        "--output-dir", default="analysis_results/gate_value_distribution_v2",
        help="Output directory"
    )
    args = parser.parse_args()

    global OUTPUT_DIR, VECTORS_DIR
    OUTPUT_DIR  = Path(args.output_dir)
    VECTORS_DIR = OUTPUT_DIR / "vectors"

    model, tokenizer, registry = load_model(args.model_id)

    all_results, num_layers, intermediate_size = run_experiment(
        model, tokenizer, registry,
        threshold=args.threshold,
        num_tokens=args.num_tokens,
    )

    print("\nComputing summary ...")
    summary = compute_summary(all_results, num_layers, intermediate_size, args.threshold)
    print_summary(summary, num_layers, intermediate_size)

    # Persist summary.
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment":         "gate_value_distribution_v2",
                "model":              args.model_id,
                "quantization":       "nf4",
                "activity_threshold": args.threshold,
                "num_prompts":        len(all_results),
                "num_layers":         num_layers,
                "intermediate_size":  intermediate_size,
                "summary":            summary,
            },
            fh, indent=2,
        )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
