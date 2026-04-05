#!/usr/bin/env python3
"""
Experiment 4 — Attention Head Split Verification on ReluLLaMA-70B
==================================================================
Verifies that the output projection (o_proj) of each attention layer is
exactly additive by query head.  For each layer, we capture the pre-projection
attention output (shape [1, 1, num_heads * head_dim]), then analytically
compute the sum of per-head contributions using the dequantized o_proj weight.

LLaMA-2 70B architecture:
  num_attention_heads = 64  (query heads)
  head_dim            = 128
  hidden_size         = 8192  (= 64 * 128)
  num_key_value_heads = 8   (GQA — irrelevant for o_proj splitting)

Split verification:
  ref   = attn_out @ o_proj.weight.T         (full matmul)
  parts = [attn_out[:, h*hd:(h+1)*hd] @ o_proj.weight[:, h*hd:(h+1)*hd].T
           for h in range(num_heads)]
  reconstructed = sum(parts)
  max_abs_diff  = (reconstructed - ref).abs().max()
  cosine_sim    = F.cosine_similarity(reconstructed, ref, dim=-1).item()

Usage:
    python tools/attention_split_relu70b.py [--num-tokens N] [--out-dir DIR]

Outputs:
    analysis_results/attention_split_relu70b/{gpu}_{timestamp}.json
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
# Weight dequantization
# ---------------------------------------------------------------------------

def dequantize_weight(module) -> "torch.Tensor":
    """Return the float32 weight matrix for a (possibly 4-bit) linear layer."""
    import torch
    w = module.weight
    if hasattr(w, "dequantize"):
        return w.dequantize().float()
    return w.float()


# ---------------------------------------------------------------------------
# Head split verification for one attention layer / one generation step
# ---------------------------------------------------------------------------

def verify_head_split(
    attn_out_1d: "torch.Tensor",   # [num_heads * head_dim] float32 on device
    o_proj_module,
    num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """
    Analytically verify that summing per-head contributions to o_proj equals
    the full projection.  Returns a dict of stats.
    """
    import torch
    import torch.nn.functional as F

    device = attn_out_1d.device
    W = dequantize_weight(o_proj_module).to(device)  # [hidden_size, num_heads*head_dim]

    x = attn_out_1d.unsqueeze(0).float()  # [1, num_heads*head_dim]
    ref = (x @ W.t()).squeeze(0)           # [hidden_size]

    sum_parts = torch.zeros_like(ref)
    per_head_max_diff = []

    for h in range(num_heads):
        s = h * head_dim
        e = (h + 1) * head_dim
        partial = (x[:, s:e] @ W[:, s:e].t()).squeeze(0)  # [hidden_size]
        diff = (partial - (ref * (1.0 / num_heads))).abs().max().item()
        per_head_max_diff.append(round(float(diff), 8))
        sum_parts = sum_parts + partial

    max_abs_diff = (sum_parts - ref).abs().max().item()
    cosine_sim = F.cosine_similarity(
        sum_parts.unsqueeze(0), ref.unsqueeze(0), dim=-1
    ).item()

    del W, x, ref, sum_parts
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "max_abs_diff": round(float(max_abs_diff), 8),
        "cosine_similarity": round(float(cosine_sim), 8),
        "per_head_max_diff": per_head_max_diff,
        "additive_ok": bool(max_abs_diff < 1e-3),
    }


# ---------------------------------------------------------------------------
# Run one prompt — hook on o_proj input, verify first generation step
# ---------------------------------------------------------------------------

def run_prompt(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int,
    layers_to_verify: list[int],
) -> list[dict[str, Any]]:
    import torch

    num_heads = model.config.num_attention_heads   # 64
    head_dim  = model.config.hidden_size // num_heads  # 128

    # step counter and captured data
    gen_step = [0]
    captured: dict[int, torch.Tensor] = {}  # layer_idx -> attn_out snapshot

    def make_pre_hook(layer_idx: int):
        def pre_hook_fn(module, input_tuple):
            # Only capture first generation step (seq_len == 1)
            x = input_tuple[0]
            if x.shape[1] != 1:
                return  # prefill
            if gen_step[0] != 0:
                return  # only first gen step
            captured[layer_idx] = x[0, 0, :].detach().float()  # [num_heads*head_dim]
        return pre_hook_fn

    hooks = []
    for layer_idx in layers_to_verify:
        layer = model.model.layers[layer_idx]
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            make_pre_hook(layer_idx)
        ))

    # Hook to advance gen_step counter using layer 0 mlp (always present)
    def step_counter_hook(module, input_tuple, output):
        x = input_tuple[0]
        if x.shape[1] == 1:
            gen_step[0] += 1
        return output
    step_hook = model.model.layers[0].mlp.register_forward_hook(step_counter_hook)

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
    step_hook.remove()

    # Compute split verification for each captured layer
    layer_results = []
    for layer_idx in layers_to_verify:
        if layer_idx not in captured:
            layer_results.append({"layer": layer_idx, "error": "not captured"})
            continue
        attn_out = captured[layer_idx]
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        stats = verify_head_split(attn_out, o_proj, num_heads, head_dim)
        stats["layer"] = layer_idx
        layer_results.append(stats)

    return layer_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Attention head split verification for ReluLLaMA-70B"
    )
    parser.add_argument("--model-id", default="SparseLLM/ReluLLaMA-70B",
                        help="HuggingFace model ID (default: SparseLLM/ReluLLaMA-70B)")
    parser.add_argument("--num-tokens", type=int, default=20,
                        help="Number of tokens to generate per prompt (default: 20)")
    parser.add_argument("--out-dir", default="analysis_results/attention_split_relu70b",
                        help="Output directory (default: analysis_results/attention_split_relu70b)")
    parser.add_argument("--layers", nargs="+", type=int,
                        help="Layer indices to verify (default: 0 20 40 60 79)")
    parser.add_argument("--prompts", nargs="+", type=int,
                        help="Prompt indices to run (default: all 10)")
    args = parser.parse_args()

    import torch
    import numpy as np

    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_to_verify = args.layers if args.layers else [0, 20, 40, 60, 79]
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
    all_additive_ok = []

    for p_idx in prompt_indices:
        prompt = PROMPTS[p_idx]
        print(f"\n[{p_idx+1}/{len(prompt_indices)}] Prompt {p_idx}: {prompt[:60]!r}")
        t1 = time.time()
        layer_results = run_prompt(model, tokenizer, prompt, args.num_tokens, layers_to_verify)
        elapsed = time.time() - t1
        print(f"  done in {elapsed:.1f}s")

        for lr in layer_results:
            if "error" not in lr:
                ok = lr["additive_ok"]
                all_additive_ok.append(ok)
                print(f"  layer {lr['layer']:3d}: max_diff={lr['max_abs_diff']:.2e}  "
                      f"cos_sim={lr['cosine_similarity']:.8f}  {'OK' if ok else 'FAIL'}")

        results.append({
            "prompt_idx": p_idx,
            "prompt": prompt,
            "layer_results": layer_results,
        })

    all_pass = all(all_additive_ok)
    pass_rate = sum(all_additive_ok) / len(all_additive_ok) if all_additive_ok else 0.0

    summary = {
        "model": args.model_id,
        "num_prompts": len(prompt_indices),
        "num_tokens": args.num_tokens,
        "layers_verified": layers_to_verify,
        "all_additive_ok": all_pass,
        "pass_rate": round(pass_rate, 4),
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
    print("EXPERIMENT 4 — ATTENTION HEAD SPLIT VERIFICATION SUMMARY")
    print("="*60)
    print(f"Model:            {args.model_id}")
    print(f"Layers verified:  {layers_to_verify}")
    print(f"Pass rate:        {pass_rate:.2%}")
    verdict = "PASS" if all_pass else "FAIL"
    print(f"All additive OK:  {verdict}")
    print("="*60)

    # Optional DB write
    try:
        from adversarial_suite.db.writer import ResultsWriter
        script_path = str(Path(__file__).resolve())
        with ResultsWriter("attention_split_relu70b", script_path=script_path) as writer:
            db_model = writer.ensure_model(args.model_id)
            for res in results:
                db_prompt = writer.ensure_prompt(res["prompt"], category="attention_split")
                for lr in res["layer_results"]:
                    if "error" not in lr:
                        writer.add_result(
                            model_id=db_model,
                            prompt_id=db_prompt,
                            attack_type="attention_head_split",
                            layer=lr["layer"],
                            position=0,
                            raw_data=lr,
                        )
        print("DB write succeeded.")
    except Exception as exc:
        print(f"DB write failed ({exc}); continuing.")


if __name__ == "__main__":
    main()
