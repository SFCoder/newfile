#!/usr/bin/env python3
"""
Experiment 7 — Equivocation Quality-of-Service Test on ReluLLaMA-70B
======================================================================
Tests how output quality degrades when Gaussian noise is injected into the
hidden states at selected MLP input sites during the generation phase only.

For each (prompt, noise_level, target_layer) combination:
  - Generate the clean reference output (no noise).
  - Generate the noisy output (noise injected at target_layer during generation).
  - Compare token-by-token: token_match_rate, exact_match (all tokens match).

Noise model (per generation step, target layer):
    x_noisy = x + randn_like(x) * noise_fraction * x.norm(dim=-1, keepdim=True)

This perturbs the MLP input proportionally to the local hidden state norm,
so the perturbation scales naturally with activation magnitude.

Usage:
    python tools/equivocation_qos_relu70b.py [--num-tokens N] [--out-dir DIR]

Outputs:
    analysis_results/equivocation_qos_relu70b/{gpu}_{timestamp}.json
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
# Prompts (5 for this experiment)
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "The mitochondria is the powerhouse of the",
    "Large language models work by",
    "The best way to learn programming is",
    "Quantum computing differs from classical computing because",
]

NOISE_LEVELS = [0.001, 0.01, 0.05]
TARGET_LAYERS = [20, 40, 60]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
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
# Generation helpers
# ---------------------------------------------------------------------------

def generate_tokens(model, tokenizer, prompt: str, num_tokens: int) -> list[int]:
    """Clean generation — no hooks."""
    import torch
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=num_tokens,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return out[0][prompt_len:].tolist()


def generate_tokens_noisy(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int,
    target_layer: int,
    noise_fraction: float,
) -> list[int]:
    """
    Generation with noise injected into the MLP input at `target_layer`
    on every generation step (seq_len == 1 only; prefill is clean).
    """
    import torch

    def noise_pre_hook(module, input_tuple):
        x = input_tuple[0]
        if x.shape[1] != 1:
            return input_tuple  # prefill — skip
        with torch.no_grad():
            norm = x.norm(dim=-1, keepdim=True)
            noise = torch.randn_like(x) * noise_fraction * norm
            x_noisy = x + noise
        return (x_noisy,) + input_tuple[1:]

    layer = model.model.layers[target_layer]
    hook = layer.mlp.register_forward_pre_hook(noise_pre_hook)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=num_tokens,
            do_sample=False,
        )
    hook.remove()

    prompt_len = inputs["input_ids"].shape[1]
    return out[0][prompt_len:].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 7: Equivocation QoS test for ReluLLaMA-70B"
    )
    parser.add_argument("--model-id", default="SparseLLM/ReluLLaMA-70B",
                        help="HuggingFace model ID (default: SparseLLM/ReluLLaMA-70B)")
    parser.add_argument("--num-tokens", type=int, default=20,
                        help="Number of tokens to generate per prompt (default: 20)")
    parser.add_argument("--out-dir", default="analysis_results/equivocation_qos_relu70b",
                        help="Output directory (default: analysis_results/equivocation_qos_relu70b)")
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        help="Noise fractions to test (default: 0.001 0.01 0.05)")
    parser.add_argument("--target-layers", nargs="+", type=int,
                        help="Layer indices to inject noise at (default: 20 40 60)")
    parser.add_argument("--prompts", nargs="+", type=int,
                        help="Prompt indices to run (default: all 5)")
    args = parser.parse_args()

    import torch
    import numpy as np

    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = args.noise_levels if args.noise_levels else NOISE_LEVELS
    target_layers = args.target_layers if args.target_layers else TARGET_LAYERS
    prompt_indices = args.prompts if args.prompts else list(range(len(PROMPTS)))

    print(f"Loading {args.model_id} …")
    t0 = time.time()
    model, tokenizer = load_model(args.model_id)
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.1f}s")

    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results = []

    # Track aggregate match rates per noise level
    agg: dict[float, list[float]] = {nl: [] for nl in noise_levels}

    for p_idx in prompt_indices:
        prompt = PROMPTS[p_idx]
        print(f"\nPrompt {p_idx}: {prompt[:60]!r}")

        # Clean reference
        ref_tokens = generate_tokens(model, tokenizer, prompt, args.num_tokens)
        ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        print(f"  reference: {ref_text[:80]!r}")

        prompt_result: dict[str, Any] = {
            "prompt_idx": p_idx,
            "prompt": prompt,
            "reference_tokens": ref_tokens,
            "reference_text": ref_text,
            "noise_experiments": [],
        }

        for noise_frac in noise_levels:
            for tgt_layer in target_layers:
                t1 = time.time()
                noisy_tokens = generate_tokens_noisy(
                    model, tokenizer, prompt, args.num_tokens,
                    tgt_layer, noise_frac,
                )
                elapsed = time.time() - t1
                noisy_text = tokenizer.decode(noisy_tokens, skip_special_tokens=True)

                n = min(len(ref_tokens), len(noisy_tokens))
                matches = sum(r == n_ for r, n_ in zip(ref_tokens[:n], noisy_tokens[:n]))
                match_rate = matches / n if n > 0 else 0.0
                exact_match = (ref_tokens == noisy_tokens)
                agg[noise_frac].append(match_rate)

                print(f"  noise={noise_frac:.3f} layer={tgt_layer:3d}: "
                      f"match={match_rate:.2%}  exact={exact_match}  ({elapsed:.1f}s)")

                prompt_result["noise_experiments"].append({
                    "noise_fraction": noise_frac,
                    "target_layer": tgt_layer,
                    "noisy_tokens": noisy_tokens,
                    "noisy_text": noisy_text,
                    "token_match_rate": round(match_rate, 4),
                    "exact_match": exact_match,
                    "elapsed_s": round(elapsed, 2),
                })

        results.append(prompt_result)

    # Per-noise-level aggregate
    noise_summary = {}
    for nl in noise_levels:
        vals = agg[nl]
        avg_match = float(np.mean(vals)) if vals else 0.0
        noise_summary[str(nl)] = {
            "avg_token_match_rate": round(avg_match, 4),
            "min_token_match_rate": round(float(min(vals)), 4) if vals else 0.0,
        }

    summary = {
        "model": args.model_id,
        "num_prompts": len(prompt_indices),
        "num_tokens": args.num_tokens,
        "noise_levels": noise_levels,
        "target_layers": target_layers,
        "noise_summary": noise_summary,
        "gpu": gpu_name,
        "timestamp": timestamp,
        "results": results,
    }

    out_path = out_dir / f"{gpu_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nResults saved to {out_path}")

    print("\n" + "="*60)
    print("EXPERIMENT 7 — EQUIVOCATION QOS SUMMARY")
    print("="*60)
    print(f"Model:          {args.model_id}")
    print(f"Noise levels:   {noise_levels}")
    print(f"Target layers:  {target_layers}")
    print()
    for nl in noise_levels:
        ns = noise_summary[str(nl)]
        print(f"  noise={nl:.3f}: avg_match={ns['avg_token_match_rate']:.2%}  "
              f"min_match={ns['min_token_match_rate']:.2%}")
    # Quality threshold: at noise=0.001, avg match should be >=80%
    low_noise_match = noise_summary[str(0.001)]["avg_token_match_rate"] if 0.001 in noise_levels else None
    if low_noise_match is not None:
        verdict = "PASS" if low_noise_match >= 0.80 else "FAIL"
        print(f"\n0.1% noise avg match >=80%: {verdict}")
    print("="*60)

    # Optional DB write
    try:
        from adversarial_suite.db.writer import ResultsWriter
        script_path = str(Path(__file__).resolve())
        with ResultsWriter("equivocation_qos_relu70b", script_path=script_path) as writer:
            db_model = writer.ensure_model(args.model_id)
            for res in results:
                db_prompt = writer.ensure_prompt(res["prompt"], category="equivocation_qos")
                for exp in res["noise_experiments"]:
                    writer.add_result(
                        model_id=db_model,
                        prompt_id=db_prompt,
                        attack_type=f"equivocation_noise_{exp['noise_fraction']}",
                        layer=exp["target_layer"],
                        position=0,
                        raw_data=exp,
                    )
        print("DB write succeeded.")
    except Exception as exc:
        print(f"DB write failed ({exc}); continuing.")


if __name__ == "__main__":
    main()
