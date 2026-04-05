#!/usr/bin/env python3
"""
Split Verification on ReluLLaMA-70B  (Experiment 10)
======================================================
Measures whether FFN neuron-group splitting (the core of our split-
verification protocol) holds for ReLU-activated models, and captures
ReLU sparsity + economic data as side effects.

Cross-hardware component: run this script on two different GPU types;
use tools/compare_cross_hardware.py to compare the two JSON result files.

Usage:
    python3 tools/split_verification_relu70b.py
    python3 tools/split_verification_relu70b.py --model-id SparseLLM/ReluLLaMA-2-70B
    python3 tools/split_verification_relu70b.py --model-id /path/to/local/snapshot

Dependencies: torch, transformers>=4.40.0, accelerate, bitsandbytes>=0.41.0,
              numpy, scipy, protobuf
If tokenizer errors occur: pip install transformers==4.46.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Heavy dependencies (torch, numpy) are imported lazily inside functions so
# that `--help` works even when packages are not yet installed.

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Prompts — not used as argparse defaults
# ---------------------------------------------------------------------------

PROMPTS = [
    "What is the capital of France?",
    "Explain why the sky is blue using physics principles.",
    "Write the opening paragraph of a mystery novel set in Tokyo.",
    "Write a Python function to find the longest palindromic substring.",
    "Compare and contrast democracy and authoritarianism.",
    "What are the second-order effects of raising the minimum wage?",
    "Describe the CAP theorem and its implications.",
    "Hi",
    "the the the the the the the the the the",
    "3.14159 2.71828 1.41421 1.61803",
]

DISPLAY_LAYERS = [0, 10, 20, 30, 40, 50, 60, 70]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def get_gpu_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"name": "CPU", "vram_gb": 0.0, "count": 0}
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / (1024 ** 3), 1)
    return {"name": name, "vram_gb": vram_gb, "count": torch.cuda.device_count()}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_id: str):
    """
    Load the model with NF4 4-bit quantization.

    Tries to resolve model_id via model_registry snapshot path first
    (avoids full weight re-verification for large models).
    Falls back to loading directly from the HuggingFace cache.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    resolved = model_id
    if not Path(model_id).exists():
        try:
            from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH
            reg = ModelRegistry(DEFAULT_REGISTRY_PATH)
            if model_id in reg.list_models():
                entry = reg.get_entry(model_id)
                snap = ModelRegistry._snapshot_path(entry.hf_repo)
                resolved = str(snap)
                print(f"  Registry snapshot: {resolved}")
        except Exception as exc:
            print(f"  Registry not used ({exc})")

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers        = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    hidden_size       = model.config.hidden_size
    hidden_act        = getattr(model.config, "hidden_act", "unknown")

    print(
        f"  Layers: {num_layers}  |  hidden_size: {hidden_size}  "
        f"|  intermediate_size: {intermediate_size}"
    )
    print(f"  Activation function (config): {hidden_act}")

    if num_layers > 0:
        mlp0 = model.model.layers[0].mlp
        if hasattr(mlp0, "act_fn"):
            print(f"  MLP act_fn type: {type(mlp0.act_fn).__name__}")

    if hidden_act.lower() not in ("relu",):
        print(f"  WARNING: expected 'relu' for ReluLLaMA, got '{hidden_act}'.")
        print(f"  Model may not be ReluLLaMA — ReLU sparsity results will be incorrect.")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dequantization helper
# ---------------------------------------------------------------------------


def dequantize_weight(linear_module):
    """Return the weight matrix as float32, dequantizing 4-bit if needed."""
    import torch
    w = linear_module.weight
    try:
        return w.dequantize().float()
    except AttributeError:
        pass
    try:
        import bitsandbytes.functional as bnbF
        if hasattr(w, "quant_state"):
            return bnbF.dequantize_4bit(w.data, w.quant_state).float()
    except Exception:
        pass
    return w.data.float()


# ---------------------------------------------------------------------------
# Split verification
# ---------------------------------------------------------------------------


def compute_splits(
    intermediate_1d,        # torch.Tensor, shape (intermediate_size,)
    down_proj_module,
    split_counts: list[int],
) -> dict:
    """
    Verify that partitioning the FFN intermediate vector into N contiguous
    groups and summing each group's down_proj contribution equals the full
    down_proj output (linear superposition).

    Uses the dequantized weight matrix so the comparison is exact up to
    float32 rounding — hardware-independent.

    Returns dict {str(N): {max_abs_diff, mean_abs_diff, cosine_similarity,
                           token_match}}
    """
    import torch

    inter_size = intermediate_1d.shape[0]
    device     = intermediate_1d.device
    results: dict[str, dict] = {}

    try:
        W = dequantize_weight(down_proj_module).to(device)  # [hidden_size, inter_size]
    except Exception as exc:
        return {"error": str(exc)}

    bias: Optional = None
    if down_proj_module.bias is not None:
        bias = down_proj_module.bias.float().to(device)

    inter = intermediate_1d.float()   # [inter_size]

    # Reference: full computation with dequantized float32 weights
    ref = (inter.unsqueeze(0) @ W.t()).squeeze(0)   # [hidden_size]
    if bias is not None:
        ref = ref + bias

    hidden_size = W.shape[0]

    for N in split_counts:
        actual_N = min(N, inter_size)

        # Balanced split: first `r` groups get one extra element
        q, r = divmod(inter_size, actual_N)
        groups: list[tuple[int, int]] = []
        s = 0
        for i in range(actual_N):
            e = s + q + (1 if i < r else 0)
            groups.append((s, e))
            s = e

        # Sum partial contributions on the same device (no CPU round-trip)
        sum_out = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        for gs, ge in groups:
            partial = (inter[gs:ge].unsqueeze(0) @ W[:, gs:ge].t()).squeeze(0)
            sum_out = sum_out + partial
        if bias is not None:
            sum_out = sum_out + bias

        diff      = (sum_out - ref).abs()
        max_diff  = float(diff.max())
        mean_diff = float(diff.mean())

        # Cosine similarity (float32)
        dot    = float((sum_out * ref).sum())
        norm_s = float(torch.norm(sum_out))
        norm_r = float(torch.norm(ref))
        denom  = max(norm_s * norm_r, 1e-12)
        cos_sim = float(max(-1.0, min(1.0, dot / denom)))

        # Argmax agreement as a proxy "token match"
        token_match = bool(int(sum_out.argmax()) == int(ref.argmax()))

        results[str(N)] = {
            "max_abs_diff":      max_diff,
            "mean_abs_diff":     mean_diff,
            "cosine_similarity": cos_sim,
            "token_match":       token_match,
        }

    del W, ref, inter
    return results


# ---------------------------------------------------------------------------
# ReLU stats
# ---------------------------------------------------------------------------


def compute_relu_stats(gate_1d) -> dict:   # gate_1d: torch.Tensor, 1-D post-ReLU
    """
    Sparsity statistics from a single-token post-ReLU gate vector.
    ReLU(x) = max(0, x): zeros are EXACT in IEEE 754, not near-zero.
    """
    g     = gate_1d.float().cpu()
    total = int(g.numel())

    zero_count = int((g == 0.0).sum())
    near_001   = int((g < 0.001).sum())
    near_005   = int((g < 0.005).sum())
    near_010   = int((g < 0.010).sum())

    import torch
    nonzero     = g[g > 0.0]
    min_nonzero = float(nonzero.min()) if nonzero.numel() > 0 else 0.0

    return {
        "total_neurons":          total,
        "relu_zero_count":        zero_count,
        "relu_zero_fraction":     zero_count / max(total, 1),
        "relu_near_zero_0.001":   near_001,
        "relu_near_zero_0.005":   near_005,
        "relu_near_zero_0.010":   near_010,
        "min_nonzero_gate_value": min_nonzero,
        "gate_value_gap":         min_nonzero,
    }


# ---------------------------------------------------------------------------
# Per-prompt inference
# ---------------------------------------------------------------------------


def run_prompt(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int,
    split_counts: list[int],
) -> tuple[list[dict], float, int]:
    """
    Run one prompt and return (layer_entries, elapsed_s, tokens_generated).

    layer_entries: one dict per (layer, token_step) covering all generation
    steps. The first generation step (step == 0) also includes split results.
    """
    import torch

    num_layers      = len(model.model.layers)
    layer_entries: list[dict] = []
    generation_step = [0]   # mutable so the closure can mutate it

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tuple, output):
            x       = input_tuple[0]    # [batch, seq_len, hidden_size]
            seq_len = x.shape[-2]
            is_gen  = (seq_len == 1)
            step    = generation_step[0]

            with torch.no_grad():
                gate_raw     = module.act_fn(module.gate_proj(x))
                up           = module.up_proj(x)
                intermediate = gate_raw * up

            if is_gen:
                gate_1d   = gate_raw[0, 0, :]      # [intermediate_size]
                relu_stat = compute_relu_stats(gate_1d)

                entry: dict = {"layer": layer_idx, "token_step": step, **relu_stat}

                # Split verification: first generation step only
                if step == 0:
                    inter_1d = intermediate[0, 0, :].float()
                    try:
                        split_res = compute_splits(inter_1d, module, split_counts)
                        entry["splits"] = split_res
                    except Exception as exc:
                        entry["splits_error"] = str(exc)
                    finally:
                        del inter_1d
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                layer_entries.append(entry)

            # Advance step counter after the last layer of each decode step
            if is_gen and layer_idx == num_layers - 1:
                generation_step[0] += 1

            del gate_raw, up, intermediate
            return output

        return hook_fn

    hooks = [
        layer.mlp.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    try:
        input_device = model.model.embed_tokens.weight.device
    except AttributeError:
        input_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    try:
        t0 = time.time()
        enc = tokenizer(prompt, return_tensors="pt").to(input_device)
        prompt_len = int(enc["input_ids"].shape[1])

        with torch.no_grad():
            out = model.generate(
                enc["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0
        tokens_generated = int(out.shape[1]) - prompt_len
    finally:
        for h in hooks:
            h.remove()

    return layer_entries, elapsed, tokens_generated


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    model,
    tokenizer,
    num_tokens: int,
    split_counts: list[int],
    output_dir: Path,
) -> dict:
    """Run all prompts, save results JSON, return the result dict."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu   = get_gpu_info()
    ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{safe_filename(gpu['name'])}_{ts}.json"

    print(f"\nGPU: {gpu['name']} ({gpu['vram_gb']} GB, {gpu['count']} device(s))")
    print(f"Prompts: {len(PROMPTS)}  |  Tokens: {num_tokens}  |  Splits: {split_counts}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts_data: list[dict] = []
    total_start  = time.time()
    total_tokens = 0

    for pidx, prompt_text in enumerate(PROMPTS):
        print(f"[{pidx + 1:2d}/{len(PROMPTS)}] {prompt_text[:60]}...")

        layer_entries, elapsed, tokens_gen = run_prompt(
            model, tokenizer, prompt_text, num_tokens, split_counts
        )
        total_tokens += tokens_gen

        tok_per_s = tokens_gen / max(elapsed, 0.001)
        print(f"  {tokens_gen} tokens  {elapsed:.1f}s  ({tok_per_s:.1f} tok/s)")

        prompts_data.append({
            "prompt_id":              pidx,
            "prompt_text":            prompt_text,
            "tokens_generated":       tokens_gen,
            "inference_time_seconds": round(elapsed, 3),
            "layers":                 layer_entries,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    mem_gb = 0.0
    if torch.cuda.is_available():
        mem_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    result = {
        "experiment":                   "split_verification_relu70b",
        "model":                        "ReluLLaMA-70B",
        "quantization":                 "nf4",
        "gpu_name":                     gpu["name"],
        "gpu_vram_gb":                  gpu["vram_gb"],
        "timestamp":                    utcnow(),
        "total_inference_time_seconds": round(total_elapsed, 2),
        "tokens_per_second":            round(total_tokens / max(total_elapsed, 0.001), 2),
        "gpu_memory_used_gb":           mem_gb,
        "split_counts_tested":          split_counts,
        "prompts":                      prompts_data,
    }

    out_path = output_dir / fname
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nSaved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def compute_summary(result: dict, split_counts: list[int]) -> dict:
    import numpy as np

    split_worst: dict[str, dict] = {
        str(N): {"max_abs_diff": 0.0, "min_cosine": 1.0, "match": 0, "total": 0}
        for N in split_counts
    }
    layer_relu: dict[int, list] = {}

    for pdata in result["prompts"]:
        for entry in pdata["layers"]:
            li = entry["layer"]
            layer_relu.setdefault(li, []).append((
                entry.get("relu_zero_fraction", 0.0),
                entry.get("min_nonzero_gate_value", 0.0),
            ))
            if "splits" in entry:
                for N in split_counts:
                    key = str(N)
                    if key in entry["splits"]:
                        s  = entry["splits"][key]
                        sw = split_worst[key]
                        sw["max_abs_diff"] = max(sw["max_abs_diff"], s.get("max_abs_diff", 0.0))
                        sw["min_cosine"]   = min(sw["min_cosine"],   s.get("cosine_similarity", 1.0))
                        sw["match"]       += int(s.get("token_match", False))
                        sw["total"]       += 1

    layer_avg: dict[int, dict] = {}
    for li, stats in layer_relu.items():
        fracs = [x[0] for x in stats]
        mins  = [x[1] for x in stats if x[1] > 0]
        layer_avg[li] = {
            "avg_relu_zero_fraction": float(np.mean(fracs)),
            "avg_min_nonzero":        float(np.mean(mins)) if mins else 0.0,
        }

    return {"split_worst": split_worst, "layer_avg": layer_avg}


def print_summary(
    result: dict,
    summary: dict,
    split_counts: list[int],
    num_layers: int,
) -> None:
    import numpy as np

    sw = summary["split_worst"]
    la = summary["layer_avg"]

    print("\n" + "=" * 55)
    print("=== Split Verification on ReluLLaMA-70B ===")
    print("=" * 55)
    print(f"\nGPU: {result['gpu_name']}")
    print(f"Model: ReluLLaMA-70B (NF4 4-bit)")

    print("\nSplit Verification Results (worst case across all prompts and layers):")
    for N in split_counts:
        key = str(N)
        if key in sw:
            d = sw[key]
            match_s = f"{d['match']:3d}/{d['total']:3d}"
            print(
                f"  {N:5d} splits: "
                f"max_diff={d['max_abs_diff']:.6f}  "
                f"cosine={d['min_cosine']:.6f}  "
                f"token_match={match_s}"
            )

    print("\nReLU Sparsity (averaged across prompts and token steps):")
    for li in sorted(li for li in DISPLAY_LAYERS if li in la):
        s   = la[li]
        pct = s["avg_relu_zero_fraction"] * 100
        mnz = s["avg_min_nonzero"]
        print(
            f"  Layer {li:2d}: {pct:5.1f}% exactly zero  |  "
            f"min nonzero gate: {mnz:.5f}  |  gap: {mnz:.5f}"
        )

    n_prompts = max(len(result["prompts"]), 1)
    avg_time  = result["total_inference_time_seconds"] / n_prompts
    gpu_cost  = 3.00   # $/hr rough A100 estimate
    cost_inf  = avg_time / 3600 * gpu_cost

    print("\nEconomic Data:")
    print(f"  Avg inference time:  {avg_time:.1f} seconds")
    print(f"  Avg tokens/second:   {result['tokens_per_second']:.1f}")
    print(f"  GPU memory used:     {result['gpu_memory_used_gb']:.1f} GB")
    print(f"  Est. cost/inference: ${cost_inf:.5f} (at ${gpu_cost:.2f}/hr)")

    worst_max_diff = max((sw[str(N)]["max_abs_diff"] for N in split_counts if str(N) in sw), default=1.0)
    worst_cosine   = min((sw[str(N)]["min_cosine"]   for N in split_counts if str(N) in sw), default=0.0)

    mid_fracs    = [la[li]["avg_relu_zero_fraction"] for li in range(20, 61) if li in la]
    mid_sparsity = float(np.mean(mid_fracs)) if mid_fracs else 0.0

    all_min_nz  = [la[li]["avg_min_nonzero"] for li in la if la[li]["avg_min_nonzero"] > 0]
    relu_exact  = bool(all_min_nz)

    split_exact    = worst_max_diff < 1e-6
    split_accurate = worst_cosine   > 0.9999
    sparsity_ok    = mid_sparsity   > 0.40
    all_pass       = split_exact and split_accurate and relu_exact and sparsity_ok

    print(f"\nVERDICT: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Split verification exact (max_diff < 1e-6):     {'YES' if split_exact else 'NO'}")
    print(f"  Split verification accurate (cosine > 0.9999):  {'YES' if split_accurate else 'NO'}")
    print(f"  ReLU zeros are exact (gap > 0):                 {'YES' if relu_exact else 'NO'}")
    print(f"  Per-token sparsity > 40% (layers 20-60):        {'YES' if sparsity_ok else 'NO'}")
    print()


# ---------------------------------------------------------------------------
# Optional database write
# ---------------------------------------------------------------------------


def write_to_db(result: dict, split_counts: list[int]) -> None:
    try:
        from adversarial_suite.db.writer import ResultsWriter
    except ImportError:
        print("  DB writer not available; skipping.")
        return

    try:
        with ResultsWriter(
            "split_verification_relu70b",
            script_path="tools/split_verification_relu70b.py",
            notes=f"gpu={result['gpu_name']} nf4",
        ) as writer:
            db_model = writer.ensure_model(result["model"])
            for pdata in result["prompts"]:
                db_prompt = writer.ensure_prompt(pdata["prompt_text"])
                for entry in pdata["layers"]:
                    if "splits" not in entry:
                        continue
                    splits_100 = entry["splits"].get("100", {})
                    writer.add_result(
                        model_id=db_model,
                        prompt_id=db_prompt,
                        attack_type="split_verification",
                        layer=entry["layer"],
                        position=entry.get("token_step", 0),
                        raw_data={
                            "relu_zero_count":       entry.get("relu_zero_count"),
                            "cosine_similarity_100": splits_100.get("cosine_similarity"),
                            "max_abs_diff_100":      splits_100.get("max_abs_diff"),
                            "gate_value_gap":        entry.get("gate_value_gap"),
                        },
                    )
        print("  Results written to database.")
    except Exception as exc:
        print(f"  DB write failed ({exc}); continuing.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # IMPORTANT: argparse is set up before any heavy imports so that
    # `--help` works even when torch/transformers are not installed.
    # Do NOT use module-level constants as argparse defaults.
    parser = argparse.ArgumentParser(
        description="Split verification on ReluLLaMA-70B with cross-hardware component"
    )
    parser.add_argument(
        "--model-id",
        default="SparseLLM/ReluLLaMA-70B",
        help="HuggingFace model ID or local snapshot path "
             "(alternative: SparseLLM/ReluLLaMA-2-70B)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10,
        help="New tokens to generate per prompt (default: 10)",
    )
    parser.add_argument(
        "--splits",
        default="2,10,100,1000",
        help="Comma-separated split counts to test (default: 2,10,100,1000)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results/split_verification_relu70b",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database write",
    )
    args = parser.parse_args()

    # Parse split counts here (after argparse, before heavy imports)
    split_counts = [int(s.strip()) for s in args.splits.split(",")]
    output_dir   = _ROOT / args.output_dir

    # Heavy imports only happen after --help is handled
    model, tokenizer = load_model(args.model_id)
    num_layers = len(model.model.layers)

    result = run_experiment(
        model, tokenizer,
        num_tokens=args.num_tokens,
        split_counts=split_counts,
        output_dir=output_dir,
    )

    summary = compute_summary(result, split_counts)
    print_summary(result, summary, split_counts, num_layers)

    if not args.skip_db:
        write_to_db(result, split_counts)


if __name__ == "__main__":
    main()
