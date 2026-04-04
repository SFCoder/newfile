#!/usr/bin/env python3
"""
Gate Value Distribution Profiling at 72B
=========================================
Profiles the full gate value distribution of inactive neurons across diverse
prompts, layers, and token positions for Qwen 2.5 72B.

This experiment answers: Does Qwen 2.5 72B reliably produce enough deeply
inactive neurons (gate values far below the activity threshold) to carry
transaction data in a blockchain consensus mechanism?

Usage:
    python3 tools/gate_value_distribution.py
    python3 tools/gate_value_distribution.py --threshold 0.1 --num-tokens 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
ACTIVITY_THRESHOLD = 0.05
NUM_TOKENS = 12        # generate 12 new tokens; 11 seq_len=1 decode steps
CAPTURE_POSITIONS = [0, 5, 10]   # generation step indices to capture
EXPECTED_INTERMEDIATE_SIZE = 29568
EXPECTED_NUM_LAYERS = 80

OUTPUT_DIR = Path("analysis_results/gate_value_distribution")
VECTORS_DIR = OUTPUT_DIR / "vectors"

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
    """
    Load model with 4-bit NF4 quantization via bitsandbytes.

    Tries the model_registry snapshot path first (avoids redundant hash
    verification for the 72B which takes many minutes). Falls back to loading
    directly from the HuggingFace cache.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_id} with NF4 4-bit quantization ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Try to resolve the snapshot path via the registry (skip weight
    # verification for large models — the registry check is advisory here).
    model_path: str = model_id
    try:
        repo_root = Path(__file__).parent.parent
        sys.path.insert(0, str(repo_root))
        from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        if model_id in registry.list_models():
            entry = registry.get_entry(model_id)
            snapshot = registry._snapshot_path(entry.hf_repo)
            model_path = str(snapshot)
            print(f"  Registry snapshot: {model_path}")
        else:
            print(f"  {model_id!r} not in registry — loading from HF cache")
    except Exception as exc:
        print(f"  Registry lookup failed ({exc}), loading from HF cache")

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
    print(
        f"  Loaded: {num_layers} layers, "
        f"intermediate_size={intermediate_size}"
    )
    if intermediate_size != EXPECTED_INTERMEDIATE_SIZE:
        print(
            f"  WARNING: expected intermediate_size={EXPECTED_INTERMEDIATE_SIZE}, "
            f"got {intermediate_size}"
        )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Gate value capture
# ---------------------------------------------------------------------------

def capture_gate_values(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = NUM_TOKENS,
    capture_positions: list[int] = CAPTURE_POSITIONS,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Run greedy generation, capturing post-activation gate values at specific
    generation steps.

    Only captures during the generation phase (single-token decode steps where
    the MLP input has seq_len == 1). Skips the prompt-processing prefill pass.

    Returns
    -------
    dict mapping (layer_idx, capture_idx) -> float32 numpy array of shape
    (intermediate_size,), where capture_idx is the index into capture_positions.
    """
    num_layers = len(model.model.layers)
    max_capture_step = max(capture_positions)

    # Mutable state shared across all layer hooks within one generation call.
    generation_step = [0]  # incremented after every complete decode step
    captured: dict[tuple[int, int], np.ndarray] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            # x shape: [batch, seq_len, hidden_size]
            seq_len = x.shape[-2]

            # Skip the prompt-processing prefill pass.
            if seq_len != 1:
                return output

            step = generation_step[0]

            if step in capture_positions:
                cap_idx = capture_positions.index(step)
                with torch.no_grad():
                    # Post-activation gate values — shape [1, 1, intermediate_size]
                    gate = module.act_fn(module.gate_proj(x))
                    gate_np = gate.squeeze().float().cpu().numpy().copy()
                captured[(layer_idx, cap_idx)] = gate_np
                del gate

            # Advance the step counter after the last layer of each decode step.
            if layer_idx == num_layers - 1:
                generation_step[0] += 1

            return output

        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

    # Determine input device (first embedding parameter).
    try:
        input_device = model.model.embed_tokens.weight.device
    except AttributeError:
        input_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    return captured


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_gate_stats(
    gate_values: np.ndarray,
    threshold: float = ACTIVITY_THRESHOLD,
) -> dict:
    """
    Compute summary statistics from a 1-D float32 gate value array.
    Uses absolute values throughout — SiLU can produce negatives, but
    magnitude determines neuron contribution.
    """
    abs_vals = np.abs(gate_values)
    total = int(len(abs_vals))

    inactive_mask = abs_vals < threshold
    active_mask = ~inactive_mask
    inactive_vals = abs_vals[inactive_mask]
    active_vals = abs_vals[active_mask]

    inactive_count = int(inactive_mask.sum())
    active_count = int(active_mask.sum())

    # Inactive histogram bins: [0, 0.001), [0.001, 0.005), [0.005, 0.01),
    #                           [0.01, 0.02), [0.02, threshold)
    inactive_histogram = {
        "0.000_to_0.001": int(((abs_vals >= 0.0)    & (abs_vals < 0.001)).sum()),
        "0.001_to_0.005": int(((abs_vals >= 0.001)  & (abs_vals < 0.005)).sum()),
        "0.005_to_0.010": int(((abs_vals >= 0.005)  & (abs_vals < 0.010)).sum()),
        "0.010_to_0.020": int(((abs_vals >= 0.010)  & (abs_vals < 0.020)).sum()),
        "0.020_to_0.050": int(((abs_vals >= 0.020)  & (abs_vals < threshold)).sum()),
    }

    # Active histogram bins: [0.05, 0.1), [0.1, 0.5), [0.5, 1.0), [1.0, inf)
    active_histogram = {
        "0.050_to_0.100": int(((abs_vals >= 0.050) & (abs_vals < 0.100)).sum()),
        "0.100_to_0.500": int(((abs_vals >= 0.100) & (abs_vals < 0.500)).sum()),
        "0.500_to_1.000": int(((abs_vals >= 0.500) & (abs_vals < 1.000)).sum()),
        "1.000_plus":     int( (abs_vals >= 1.000).sum()),
    }

    # Encoding capacity: neurons strictly below sub-thresholds
    encoding_capacity = {
        "below_0.001": int((abs_vals < 0.001).sum()),
        "below_0.005": int((abs_vals < 0.005).sum()),
        "below_0.010": int((abs_vals < 0.010).sum()),
        "below_0.020": int((abs_vals < 0.020).sum()),
    }

    def _stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
        }

    return {
        "active_count":        active_count,
        "inactive_count":      inactive_count,
        "inactive_gate_stats": _stats(inactive_vals),
        "active_gate_stats":   _stats(active_vals),
        "inactive_histogram":  inactive_histogram,
        "active_histogram":    active_histogram,
        "encoding_capacity":   encoding_capacity,
    }


# ---------------------------------------------------------------------------
# Prompt list builder
# ---------------------------------------------------------------------------

def build_prompt_list() -> list[tuple[str, str, str]]:
    """Return [(prompt_id, category, prompt_text), ...] for all 26 prompts."""
    result = []
    for category, texts in PROMPTS.items():
        for i, text in enumerate(texts, start=1):
            result.append((f"{category}_{i:02d}", category, text))
    return result


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    model,
    tokenizer,
    threshold: float,
    num_tokens: int,
) -> tuple[list, int, int]:
    """
    Run gate value profiling for all prompts.

    Returns (all_results, num_layers, intermediate_size) where all_results is
    a list of (prompt_id, category, layers_data) tuples.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    prompt_list = build_prompt_list()

    print(f"\nModel layers: {num_layers}  |  intermediate_size: {intermediate_size}")
    print(f"Activity threshold: {threshold}  |  Tokens per prompt: {num_tokens}")
    print(f"Capture positions: {CAPTURE_POSITIONS}  |  Prompts: {len(prompt_list)}\n")

    all_results = []

    for prompt_idx, (prompt_id, category, prompt_text) in enumerate(prompt_list):
        print(
            f"[{prompt_idx + 1:2d}/{len(prompt_list)}] {prompt_id}: "
            f"{prompt_text[:60]}..."
        )
        t0 = time.time()

        captured = capture_gate_values(
            model, tokenizer, prompt_text,
            num_tokens=num_tokens,
            capture_positions=CAPTURE_POSITIONS,
        )

        elapsed = time.time() - t0
        n_captured = len(captured)
        expected = num_layers * len(CAPTURE_POSITIONS)
        if n_captured < expected:
            print(f"  WARNING: captured {n_captured}/{expected} (layer, pos) pairs")
        else:
            print(f"  Captured {n_captured} gate tensors in {elapsed:.1f}s")

        # Build per-layer data; compute overlap across token positions in-memory.
        layers_data = []
        for layer_idx in range(num_layers):
            token_positions_data = []
            layer_gate_vecs = []   # kept for overlap computation

            for cap_idx, token_pos in enumerate(CAPTURE_POSITIONS):
                key = (layer_idx, cap_idx)
                if key in captured:
                    gate_vals = captured[key].astype(np.float32)
                else:
                    gate_vals = np.zeros(intermediate_size, dtype=np.float32)

                # Save numpy vector
                vec_path = VECTORS_DIR / f"{prompt_id}_layer{layer_idx:02d}_token{cap_idx}.npy"
                np.save(str(vec_path), gate_vals)

                stats = compute_gate_stats(gate_vals, threshold=threshold)
                token_positions_data.append({
                    "token_position": token_pos,
                    **stats,
                })
                layer_gate_vecs.append(gate_vals)

            layers_data.append({
                "layer": layer_idx,
                "token_positions": token_positions_data,
                # Store gate vecs temporarily for overlap; removed before JSON save.
                "_gate_vecs": layer_gate_vecs,
            })

        # Compute per-layer token-position overlap (Jaccard of inactive sets).
        layer_overlaps = []
        for ld in layers_data:
            vecs = ld["_gate_vecs"]
            inactive_sets = [np.abs(v) < threshold for v in vecs]
            pairs = [
                (inactive_sets[i], inactive_sets[j])
                for i in range(len(inactive_sets))
                for j in range(i + 1, len(inactive_sets))
            ]
            for a, b in pairs:
                union = int((a | b).sum())
                if union > 0:
                    layer_overlaps.append(int((a & b).sum()) / union)

        # Strip the temporary gate vecs before JSON serialisation.
        for ld in layers_data:
            del ld["_gate_vecs"]

        # Save JSON result for this prompt.
        result = {
            "experiment":            "gate_value_distribution",
            "model":                 "Qwen2.5-72B",
            "quantization":          "nf4",
            "prompt_id":             prompt_id,
            "prompt_category":       category,
            "prompt_text":           prompt_text,
            "activity_threshold":    threshold,
            "total_neurons":         intermediate_size,
            "num_layers":            num_layers,
            "token_positions_captured": CAPTURE_POSITIONS,
            "layers":                layers_data,
        }
        json_path = OUTPUT_DIR / f"{prompt_id}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        all_results.append((prompt_id, category, layers_data, layer_overlaps))

        # Free GPU memory between prompts.
        del captured
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    """Aggregate statistics across all prompts, layers, and token positions."""

    enc_below_0001 = []
    enc_below_0005 = []
    enc_below_001  = []
    enc_below_002  = []
    inactive_maxes = []
    active_mins    = []
    all_overlaps   = []

    for prompt_id, category, layers_data, layer_overlaps in all_results:
        all_overlaps.extend(layer_overlaps)
        for ld in layers_data:
            for pd in ld["token_positions"]:
                enc = pd["encoding_capacity"]
                enc_below_0001.append(enc["below_0.001"])
                enc_below_0005.append(enc["below_0.005"])
                enc_below_001.append( enc["below_0.010"])
                enc_below_002.append( enc["below_0.020"])
                inactive_maxes.append(pd["inactive_gate_stats"]["max"])
                am = pd["active_gate_stats"]["min"]
                if am > 0:
                    active_mins.append(am)

    min_enc_0001 = int(min(enc_below_0001)) if enc_below_0001 else 0
    min_enc_0005 = int(min(enc_below_0005)) if enc_below_0005 else 0
    min_enc_001  = int(min(enc_below_001))  if enc_below_001  else 0
    min_enc_002  = int(min(enc_below_002))  if enc_below_002  else 0

    max_inactive = float(max(inactive_maxes)) if inactive_maxes else 0.0
    min_active   = float(min(active_mins))    if active_mins   else 0.0
    gap          = min_active - max_inactive

    def kb_per_inference(neurons: int) -> float:
        return (neurons / 8) * num_layers / 1024

    # Cross-prompt variation: coefficient of variation of enc_below_001.
    arr = np.array(enc_below_001, dtype=float)
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0

    # Worst-case prompt at threshold 0.01.
    worst_prompt_id  = "n/a"
    worst_category   = "n/a"
    worst_cap        = int(min_enc_001)

    for prompt_id, category, layers_data, _ in all_results:
        prompt_min = min(
            pd["encoding_capacity"]["below_0.010"]
            for ld in layers_data
            for pd in ld["token_positions"]
        )
        if prompt_min <= worst_cap:
            worst_cap       = prompt_min
            worst_prompt_id = prompt_id
            worst_category  = category

    mean_overlap = float(np.mean(all_overlaps) * 100) if all_overlaps else 0.0

    return {
        "min_enc_0001":    min_enc_0001,
        "min_enc_0005":    min_enc_0005,
        "min_enc_001":     min_enc_001,
        "min_enc_002":     min_enc_002,
        "kb_0001":         kb_per_inference(min_enc_0001),
        "kb_0005":         kb_per_inference(min_enc_0005),
        "kb_001":          kb_per_inference(min_enc_001),
        "kb_002":          kb_per_inference(min_enc_002),
        "max_inactive":    max_inactive,
        "min_active":      min_active,
        "gap":             gap,
        "cv":              cv,
        "worst_prompt_id": worst_prompt_id,
        "worst_category":  worst_category,
        "worst_cap":       worst_cap,
        "mean_overlap":    mean_overlap,
    }


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_summary(s: dict, num_layers: int) -> None:
    print("\n" + "=" * 45)
    print("=== Gate Value Distribution Summary ===")
    print("=" * 45)

    print("\nEncoding Capacity (neurons below threshold, minimum across all prompts):")
    print(f"  Threshold 0.001: {s['min_enc_0001']:5d} neurons/layer ({s['kb_0001']:6.1f} KB/inference)")
    print(f"  Threshold 0.005: {s['min_enc_0005']:5d} neurons/layer ({s['kb_0005']:6.1f} KB/inference)")
    print(f"  Threshold 0.010: {s['min_enc_001']:5d} neurons/layer ({s['kb_001']:6.1f} KB/inference)")
    print(f"  Threshold 0.020: {s['min_enc_002']:5d} neurons/layer ({s['kb_002']:6.1f} KB/inference)")

    print("\nDistribution Gap:")
    print(f"  Max inactive gate value (at 0.01 threshold): {s['max_inactive']:.5f}")
    print(f"  Min active gate value:                       {s['min_active']:.5f}")
    print(f"  Gap:                                         {s['gap']:.5f}")
    print(f"  Cross-hardware tolerance (from prior experiments): 0.008")
    safety_margin = s["gap"] - 0.008
    print(f"  Safety margin:                               {safety_margin:.5f}")

    print("\nToken Position Stability:")
    print(f"  Mean overlap between token positions: {s['mean_overlap']:.1f}%")

    print(f"\nWorst-case prompt: {s['worst_prompt_id']} ({s['worst_category']})")
    print(f"  Encoding capacity at 0.01: {s['worst_cap']} neurons/layer")

    cap_pass     = s["min_enc_001"]  >= 1000
    gap_pass     = s["gap"]          >= 0.03
    cv_pass      = s["cv"]           <  0.30
    overlap_pass = s["mean_overlap"] >  90.0
    all_pass     = cap_pass and gap_pass and cv_pass and overlap_pass

    print(f"\nVERDICT: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Encoding capacity >= 1000 at 0.01: {'YES' if cap_pass else 'NO'}")
    print(f"  Distribution gap >= 0.03:           {'YES' if gap_pass else 'NO'}")
    print(f"  Cross-prompt variation < 30%:       {'YES' if cv_pass else 'NO'}")
    print(f"  Token position overlap > 90%:       {'YES' if overlap_pass else 'NO'}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate value distribution profiling for Qwen 2.5 72B"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--threshold", type=float, default=ACTIVITY_THRESHOLD,
        help=f"Activity threshold for gate values (default: {ACTIVITY_THRESHOLD})"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=NUM_TOKENS,
        help=f"New tokens to generate per prompt (default: {NUM_TOKENS})"
    )
    parser.add_argument(
        "--output-dir", default="analysis_results/gate_value_distribution",
        help="Output directory for JSON results"
    )
    args = parser.parse_args()

    # Apply overrides.
    global OUTPUT_DIR, VECTORS_DIR  # noqa
    OUTPUT_DIR  = Path(args.output_dir)
    VECTORS_DIR = OUTPUT_DIR / "vectors"

    model, tokenizer = load_model(args.model_id)

    all_results, num_layers, intermediate_size = run_experiment(
        model, tokenizer,
        threshold=args.threshold,
        num_tokens=args.num_tokens,
    )

    print("\nComputing summary statistics ...")
    summary = compute_summary(all_results, num_layers, intermediate_size, args.threshold)
    print_summary(summary, num_layers)

    # Persist summary JSON for downstream analysis.
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment":        "gate_value_distribution",
                "model":             args.model_id,
                "quantization":      "nf4",
                "activity_threshold": args.threshold,
                "num_prompts":       len(all_results),
                "num_layers":        num_layers,
                "intermediate_size": intermediate_size,
                "capture_positions": CAPTURE_POSITIONS,
                "summary":           summary,
            },
            fh, indent=2,
        )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
