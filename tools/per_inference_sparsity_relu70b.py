#!/usr/bin/env python3
"""
Experiment 3 — Per-Inference Sparsity Profiling on ReluLLaMA-70B
================================================================
For each prompt and each layer, record the maximum gate activation each
neuron achieves across ALL token steps (prefill + generation).  A neuron
that never exceeds 0.0 is per-inference inactive.

Usage:
    python tools/per_inference_sparsity_relu70b.py [--num-tokens N] [--out-dir DIR]

Outputs:
    analysis_results/per_inference_sparsity_relu70b/{gpu}_{timestamp}.json
    analysis_results/per_inference_sparsity_relu70b/vectors/{prompt_idx}_layer{l}.npy  (max-gate vectors)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "In a world where AI systems can",
    "The mitochondria is the powerhouse of the",
    "Once upon a time in a land far away",
    "The fundamental theorem of calculus states",
    "Large language models work by",
    "The best way to learn programming is",
    "Climate change is caused by",
    "The history of the Roman Empire began",
    "Quantum computing differs from classical computing because",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load ReluLLaMA-70B with 4-bit NF4 quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Per-inference sparsity for one prompt
# ---------------------------------------------------------------------------

def run_prompt(model, tokenizer, prompt: str, num_tokens: int) -> dict[str, Any]:
    """
    Run inference on `prompt`, generating `num_tokens` new tokens.
    Returns per-layer stats dict.
    """
    import numpy as np
    import torch

    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size  # 11008

    # max_gate[layer_idx] accumulates the maximum gate value each neuron
    # achieves across ALL forward passes (prefill + every generation step).
    max_gate: list[np.ndarray] = [
        np.zeros(intermediate_size, dtype=np.float32) for _ in range(num_layers)
    ]

    # per_token_zero_fractions[layer_idx] is a list of per-step zero fractions
    per_token_zero_fracs: list[list[float]] = [[] for _ in range(num_layers)]

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]  # [batch, seq_len, hidden_size]
            with torch.no_grad():
                gate = torch.relu(module.gate_proj(x))  # ReLU activation
                # Collapse to per-neuron max across batch & seq_len
                if gate.dim() == 3:
                    step_max = gate[0].max(dim=0).values   # [inter_size]
                    zero_frac = (gate[0] == 0.0).float().mean().item()
                elif gate.dim() == 2:
                    step_max = gate.max(dim=0).values
                    zero_frac = (gate == 0.0).float().mean().item()
                else:
                    step_max = gate.abs()
                    zero_frac = (gate == 0.0).float().mean().item()

                gate_np = step_max.float().cpu().numpy()
                np.maximum(max_gate[layer_idx], gate_np, out=max_gate[layer_idx])
                per_token_zero_fracs[layer_idx].append(zero_frac)
                del gate, step_max, gate_np
            return output
        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            inputs["input_ids"],
            max_new_tokens=num_tokens,
            do_sample=False,
        )

    for h in hooks:
        h.remove()

    # Compute per-layer statistics
    layer_stats = []
    for layer_idx in range(num_layers):
        mg = max_gate[layer_idx]
        n_inactive = int((mg == 0.0).sum())
        n_total = intermediate_size
        sparsity = n_inactive / n_total

        nonzero_vals = mg[mg > 0.0]
        min_nonzero = float(nonzero_vals.min()) if len(nonzero_vals) > 0 else None
        mean_nonzero = float(nonzero_vals.mean()) if len(nonzero_vals) > 0 else None

        avg_per_token_zero = float(np.mean(per_token_zero_fracs[layer_idx])) if per_token_zero_fracs[layer_idx] else 0.0

        layer_stats.append({
            "layer": layer_idx,
            "n_inactive": n_inactive,
            "n_active": n_total - n_inactive,
            "per_inference_sparsity": round(sparsity, 6),
            "avg_per_token_zero_frac": round(avg_per_token_zero, 6),
            "min_nonzero_gate": round(min_nonzero, 8) if min_nonzero is not None else None,
            "mean_nonzero_gate": round(mean_nonzero, 6) if mean_nonzero is not None else None,
        })

    return {
        "layer_stats": layer_stats,
        "max_gate_arrays": max_gate,  # numpy arrays, returned for optional saving
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Per-inference sparsity profiling for ReluLLaMA-70B"
    )
    parser.add_argument("--model-id", default="SparseLLM/ReluLLaMA-70B",
                        help="HuggingFace model ID (default: SparseLLM/ReluLLaMA-70B)")
    parser.add_argument("--num-tokens", type=int, default=20,
                        help="Number of tokens to generate per prompt (default: 20)")
    parser.add_argument("--out-dir", default="analysis_results/per_inference_sparsity_relu70b",
                        help="Output directory (default: analysis_results/per_inference_sparsity_relu70b)")
    parser.add_argument("--save-vectors", action="store_true",
                        help="Save per-layer max-gate numpy vectors alongside JSON")
    parser.add_argument("--prompts", nargs="+", type=int,
                        help="Prompt indices to run (default: all 10)")
    args = parser.parse_args()

    import torch
    import numpy as np

    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"
    if args.save_vectors:
        vec_dir.mkdir(parents=True, exist_ok=True)

    # Determine which prompts to run
    prompt_indices = args.prompts if args.prompts else list(range(len(PROMPTS)))

    print(f"Loading {args.model_id} …")
    t0 = time.time()
    model, tokenizer = load_model(args.model_id)
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.1f}s")

    # GPU info
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []

    total_inactive_all = []
    total_sparsity_all = []

    for p_idx in prompt_indices:
        prompt = PROMPTS[p_idx]
        print(f"\n[{p_idx+1}/{len(prompt_indices)}] Prompt {p_idx}: {prompt[:60]!r}")
        t1 = time.time()
        out = run_prompt(model, tokenizer, prompt, args.num_tokens)
        elapsed = time.time() - t1
        print(f"  done in {elapsed:.1f}s")

        layer_stats = out["layer_stats"]
        max_gate_arrays = out["max_gate_arrays"]

        # Aggregate
        per_inf_sparsities = [ls["per_inference_sparsity"] for ls in layer_stats]
        avg_sparsity = float(np.mean(per_inf_sparsities))
        total_inactive = sum(ls["n_inactive"] for ls in layer_stats)
        total_inactive_all.append(total_inactive)
        total_sparsity_all.append(avg_sparsity)

        print(f"  avg per-inference sparsity: {avg_sparsity:.2%}")
        print(f"  total inactive neurons (all layers): {total_inactive:,}")

        if args.save_vectors:
            for layer_idx, mg in enumerate(max_gate_arrays):
                np.save(vec_dir / f"p{p_idx}_layer{layer_idx}.npy", mg)

        results.append({
            "prompt_idx": p_idx,
            "prompt": prompt,
            "num_tokens": args.num_tokens,
            "avg_per_inference_sparsity": round(avg_sparsity, 6),
            "total_inactive_neurons": total_inactive,
            "layer_stats": layer_stats,
        })

    # Summary
    overall_avg_sparsity = float(np.mean(total_sparsity_all))
    summary = {
        "model": args.model_id,
        "num_prompts": len(prompt_indices),
        "num_tokens": args.num_tokens,
        "overall_avg_sparsity": round(overall_avg_sparsity, 6),
        "gpu": gpu_name,
        "timestamp": timestamp,
        "results": results,
    }

    out_path = out_dir / f"{gpu_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nResults saved to {out_path}")

    # Stdout summary
    print("\n" + "="*60)
    print("EXPERIMENT 3 — PER-INFERENCE SPARSITY SUMMARY")
    print("="*60)
    print(f"Model:                  {args.model_id}")
    print(f"Prompts run:            {len(prompt_indices)}")
    print(f"Tokens per prompt:      {args.num_tokens}")
    print(f"Overall avg sparsity:   {overall_avg_sparsity:.2%}")
    verdict = "PASS" if overall_avg_sparsity >= 0.30 else "FAIL"
    print(f"Sparsity >=30% target:  {verdict}")
    print("="*60)

    # Optional DB write
    try:
        from adversarial_suite.db.writer import ResultsWriter
        script_path = str(Path(__file__).resolve())
        with ResultsWriter("per_inference_sparsity_relu70b", script_path=script_path) as writer:
            db_model = writer.ensure_model(args.model_id)
            for res in results:
                db_prompt = writer.ensure_prompt(res["prompt"], category="sparsity")
                for ls in res["layer_stats"]:
                    writer.add_result(
                        model_id=db_model,
                        prompt_id=db_prompt,
                        attack_type="per_inference_sparsity",
                        layer=ls["layer"],
                        position=0,
                        raw_data=ls,
                    )
        print("DB write succeeded.")
    except Exception as exc:
        print(f"DB write failed ({exc}); continuing.")


if __name__ == "__main__":
    main()
