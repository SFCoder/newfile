#!/usr/bin/env python3
"""
threshold_study.py — sweep neuron-mask activation thresholds across registered models.

For each (model, threshold, prompt) triple the tool:
  1. Runs autoregressive generation to get claimed tokens and records the per-layer
     maximum |gate*up| activation across every position (prompt prefill + generation).
  2. Runs ONE single forward pass of [prompt + generated tokens] without masks to
     get the full-model logit distribution at every generated position.
  3. For each threshold value, builds a union mask from the recorded max activations,
     runs ONE single forward pass with those masks applied, and computes per-position
     metrics against the full-model baseline.

The single-pass verification (steps 2–3) is mathematically equivalent to autoregressive
masked replay because causal attention ensures position i only sees positions 0..i
regardless of what follows in the input sequence.

Outputs
-------
  analysis_results/threshold_study/results.json          full per-position data
  analysis_results/threshold_study/summary.csv           per (model, threshold) row
  analysis_results/threshold_study/threshold_vs_cosine.png
  analysis_results/threshold_study/threshold_vs_compression.png

Usage
-----
  python3 threshold_study.py
  python3 threshold_study.py --match-cosine 0.999 --top-k 5
  python3 threshold_study.py --neuron-threshold 0.01 --model Qwen/Qwen2.5-7B
  python3 threshold_study.py --models all --summary-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import logging

import torch
import torch.nn.functional as F

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

# Suppress the verbose "pad_token == eos_token / attention_mask" messages that
# transformers emits for every generate() / forward() call.  These come through
# Python's logging subsystem (not warnings), so we filter them there.
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# Also filter via warnings module for any code that uses warnings.warn()
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token id.*")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]

MODEL_ORDER: list[str] = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

# 10 diverse prompts: factual, creative, code, reasoning, mixed
PROMPTS: list[str] = [
    # Factual
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    # Creative writing
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
    # Code generation
    'def fibonacci(n):\n    """Return the nth Fibonacci number.\"\"\"\n',
    "SELECT name, email FROM users WHERE last_login <",
    # Reasoning
    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins are",
    "The key difference between supervised and unsupervised machine learning is that",
    # Mixed
    "In July 1969, humans first landed on the Moon. The mission was called Apollo",
]

NUM_TOKENS: int = 20

OUTPUT_DIR = Path("analysis_results/threshold_study")

# Display-only memory estimates
_KNOWN_MEM_GB: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B": 1.0,
    "Qwen/Qwen2.5-3B":   6.2,
    "Qwen/Qwen2.5-7B":  14.7,
}

# Approximate seconds for one generation run and one single forward pass
_APPROX_GEN_SEC: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B": 0.4,
    "Qwen/Qwen2.5-3B":   1.2,
    "Qwen/Qwen2.5-7B":   2.5,
}
_APPROX_PASS_SEC: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B": 0.06,
    "Qwen/Qwen2.5-3B":   0.18,
    "Qwen/Qwen2.5-7B":   0.50,
}

# ─────────────────────────────────────────────────────────────────────────────
# StudyModelContext — load / unload via registry
# ─────────────────────────────────────────────────────────────────────────────

class StudyModelContext:
    """
    Lightweight ModelContext for threshold_study.  Loads via registry (which
    verifies weight hash) and frees all GPU/MPS memory on exit.
    """

    def __init__(self, model_id: str, registry: ModelRegistry):
        self.model_id = model_id
        self._registry = registry
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "StudyModelContext":
        mem_gb = _KNOWN_MEM_GB.get(self.model_id, "?")
        _hdr(f"Loading {self.model_id}  (~{mem_gb} GB)")
        t0 = time.perf_counter()
        self.model, self.tokenizer = self._registry.load_verified_model(self.model_id)
        elapsed = time.perf_counter() - t0
        _info(f"Model loaded in {elapsed:.1f}s")
        return self

    def __exit__(self, *_):
        _info(f"Unloading {self.model_id} and freeing memory…")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement — provider phase
# ─────────────────────────────────────────────────────────────────────────────

def record_activation_maxima(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = NUM_TOKENS,
) -> tuple[list[int], list[torch.Tensor]]:
    """
    Run autoregressive generation.  For every MLP layer, track the maximum
    absolute value of (gate * up) across all positions and all generation steps.

    Returns
    -------
    output_token_ids : list[int]   — generated token IDs (prompt excluded)
    max_abs          : list[Tensor] — one float32 CPU tensor [intermediate_size]
                       per layer.  max_abs[l][j] = max |activation[l, :, j]|
                       seen at any position during this run.
    """
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size

    # Start with zeros; OR-accumulate via torch.maximum
    max_abs: list[torch.Tensor] = [
        torch.zeros(intermediate_size, dtype=torch.float32)
        for _ in range(num_layers)
    ]

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            with torch.no_grad():
                # Recompute intermediate to get gate * up values
                gate = module.act_fn(module.gate_proj(x))
                up   = module.up_proj(x)
                # Cast to float32 before taking abs to avoid float16 precision loss
                inter_abs = (gate * up).float().abs()  # [batch, seq, intermediate_size]
            # Collapse batch and seq: result is [intermediate_size]
            max_vals = inter_abs.amax(dim=(0, 1)).cpu()
            max_abs[layer_idx] = torch.maximum(max_abs[layer_idx], max_vals)
            return output  # pass through unmodified
        return hook_fn

    hooks = [
        layer.mlp.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        output_token_ids = output[0][prompt_len:].tolist()
    finally:
        for h in hooks:
            h.remove()

    return output_token_ids, max_abs


# ─────────────────────────────────────────────────────────────────────────────
# Mask construction and compression metric
# ─────────────────────────────────────────────────────────────────────────────

def build_masks(
    max_abs: list[torch.Tensor],
    threshold: float,
) -> list[torch.Tensor]:
    """Return one bool mask per layer: True where max |activation| > threshold."""
    return [m > threshold for m in max_abs]


def compression_pct(masks: list[torch.Tensor]) -> float:
    """Percentage of neurons that are False (zeroed out) in the mask set."""
    total = sum(m.numel() for m in masks)
    if total == 0:
        return 0.0
    active = sum(int(m.sum().item()) for m in masks)
    return 100.0 * (total - active) / total


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement — verifier phase (single forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_pass(
    model,
    tokenizer,
    prompt: str,
    output_token_ids: list[int],
    masks: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Feed [prompt_tokens + generated_tokens] through the model in ONE forward pass.

    If masks is not None, forward hooks zero out neurons outside the mask at
    every MLP layer.  Because the model uses causal attention, logit at position
    i depends only on tokens 0..i regardless of what comes after — so this
    single-pass result is identical to step-by-step masked autoregressive replay.

    Returns
    -------
    Float32 CPU tensor of shape [T, vocab_size] where T = len(output_token_ids).
    Row i is the logit distribution that predicts output_token_ids[i].
    """
    device = next(model.parameters()).device
    T = len(output_token_ids)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    gen_ids = torch.tensor(output_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # [1, L+T]
    prompt_len = prompt_ids.shape[1]

    hooks: list = []
    if masks is not None:
        # Pre-move masks to device once (reused across all positions in this pass)
        device_masks = [m.to(device=device) for m in masks]

        def make_hook(bool_mask: torch.Tensor):
            def hook_fn(module, input_tuple, _output):
                x = input_tuple[0]
                gate  = module.act_fn(module.gate_proj(x))
                up    = module.up_proj(x)
                inter = gate * up
                return module.down_proj(inter * bool_mask.to(dtype=inter.dtype))
            return hook_fn

        hooks = [
            model.model.layers[i].mlp.register_forward_hook(make_hook(dm))
            for i, dm in enumerate(device_masks)
        ]

    try:
        with torch.no_grad():
            out = model(full_ids)
        # out.logits: [1, L+T, vocab_size]
        # logits[j] is the distribution predicting token at position j+1.
        # Generated token i is at full-sequence position L+i, so predicted by
        # logits[L-1+i].  Slice: logits[L-1 : L-1+T].
        gen_logits = out.logits[0, prompt_len - 1 : prompt_len - 1 + T, :].float()
    finally:
        for h in hooks:
            h.remove()

    return gen_logits.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-position metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_metrics(
    full_logits:   torch.Tensor,   # [T, vocab_size] float32
    sparse_logits: torch.Tensor,   # [T, vocab_size] float32
    output_token_ids: list[int],
    match_cosine: float,
    top_k: int,
) -> list[dict]:
    """
    Compute per-position comparison metrics between full-model and sparse logits.

    Match criterion
    ---------------
    When match_cosine >= 1.0 (default):  passed = rank <= top_k
    When match_cosine <  1.0:            passed = (rank <= top_k) AND (cos_sim >= match_cosine)
    """
    T = len(output_token_ids)
    results: list[dict] = []

    for i in range(T):
        fl = full_logits[i]    # [vocab_size]
        sl = sparse_logits[i]  # [vocab_size]
        correct_token = output_token_ids[i]

        # Cosine similarity between full-model and sparse logit distributions
        cos_sim = float(
            F.cosine_similarity(fl.unsqueeze(0), sl.unsqueeze(0)).item()
        )
        cos_sim = max(-1.0, min(1.0, cos_sim))  # numerical clamp

        # Rank of correct token in SPARSE predictions (1-indexed, 1 = top)
        # count(sparse_logits > sparse_logits[correct]) + 1
        correct_score = sl[correct_token].item()
        rank = int((sl > correct_score).sum().item()) + 1

        # L2 distance between the two logit vectors
        logit_diff = float((fl - sl).norm(p=2).item())

        # Match criterion
        if match_cosine >= 1.0:
            passed = rank <= top_k
        else:
            passed = (rank <= top_k) and (cos_sim >= match_cosine)

        results.append({
            "position":           i,
            "correct_token_id":   correct_token,
            "cosine_sim":         round(cos_sim, 8),
            "rank":               rank,
            "logit_diff_magnitude": round(logit_diff, 6),
            "passed":             passed,
        })

    return results


def compute_aggregates(
    positions: list[dict],
    masks: list[torch.Tensor],
) -> dict:
    """Summarise per-position metrics into per-prompt scalars."""
    comp = round(compression_pct(masks), 4)

    if not positions:
        return {
            "compression_pct":  comp,
            "pass_count":       0,
            "total_count":      0,
            "pass_rate":        0.0,
            "avg_cosine_sim":   None,
            "min_cosine_sim":   None,
            "avg_rank":         None,
            "max_rank":         None,
            "first_fail_position": None,
        }

    cos_sims = [p["cosine_sim"] for p in positions]
    ranks    = [p["rank"]       for p in positions]
    pass_count = sum(1 for p in positions if p["passed"])
    first_fail = next((p["position"] for p in positions if not p["passed"]), None)

    return {
        "compression_pct":   comp,
        "pass_count":        pass_count,
        "total_count":       len(positions),
        "pass_rate":         round(100.0 * pass_count / len(positions), 2),
        "avg_cosine_sim":    round(sum(cos_sims) / len(cos_sims), 8),
        "min_cosine_sim":    round(min(cos_sims), 8),
        "avg_rank":          round(sum(ranks) / len(ranks), 4),
        "max_rank":          max(ranks),
        "first_fail_position": first_fail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep orchestration for a single model
# ─────────────────────────────────────────────────────────────────────────────

def sweep_model(
    model_id: str,
    model,
    tokenizer,
    prompts: list[str],
    thresholds: list[float],
    match_cosine: float,
    top_k: int,
    save_positions: bool = True,
) -> list[dict]:
    """
    Run all prompts at all thresholds for an already-loaded model.

    Returns a list of result dicts, one per (prompt, threshold) pair.
    Each dict contains: model, threshold, prompt_idx, prompt,
    output_token_ids, aggregates, and (if save_positions) positions.
    """
    results: list[dict] = []
    sorted_thresholds = sorted(thresholds, reverse=True)  # high → low

    for prompt_idx, prompt in enumerate(prompts):
        short = prompt[:60].replace("\n", "↵")
        print(f"\n  [{prompt_idx + 1:>2}/{len(prompts)}] \"{short}\"")

        # ── Provider: autoregressive generation + record max |activations| ──
        t0 = time.perf_counter()
        output_token_ids, max_abs = record_activation_maxima(
            model, tokenizer, prompt, NUM_TOKENS
        )
        gen_sec = time.perf_counter() - t0

        if not output_token_ids:
            print(f"         → 0 tokens (EOS immediately), skipping prompt")
            continue

        print(f"         → {len(output_token_ids)} tokens in {gen_sec:.2f}s")

        # ── Full-model single pass (no masks) — baseline logit distributions ──
        full_logits = run_single_pass(
            model, tokenizer, prompt, output_token_ids, masks=None
        )

        # ── Verifier: one sparse pass per threshold ──────────────────────────
        for threshold in sorted_thresholds:
            masks = build_masks(max_abs, threshold)

            sparse_logits = run_single_pass(
                model, tokenizer, prompt, output_token_ids, masks=masks
            )

            positions = compute_position_metrics(
                full_logits, sparse_logits, output_token_ids, match_cosine, top_k
            )
            agg = compute_aggregates(positions, masks)

            # Progress line
            print(
                f"         thresh={threshold:<9.5f}  "
                f"compress={agg['compression_pct']:5.1f}%  "
                f"cosine={agg['avg_cosine_sim'] or 0:.6f}  "
                f"rank_avg={agg['avg_rank'] or 0:.1f}  "
                f"pass={agg['pass_count']:>3}/{agg['total_count']}"
                f" ({agg['pass_rate']:.0f}%)"
            )

            entry: dict = {
                "model":            model_id,
                "threshold":        threshold,
                "prompt_idx":       prompt_idx,
                "prompt":           prompt,
                "output_token_ids": output_token_ids,
                "aggregates":       agg,
            }
            if save_positions:
                entry["positions"] = positions

            results.append(entry)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _table_rows(
    all_results: list[dict],
    models: list[str],
    thresholds: list[float],
) -> list[dict]:
    """Aggregate per-(model, threshold) summary rows for the printed table and CSV."""
    from collections import defaultdict

    # group by (model, threshold)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["threshold"])].append(r["aggregates"])

    rows: list[dict] = []
    for model_id in models:
        for threshold in sorted(thresholds, reverse=True):
            aggs = groups.get((model_id, threshold), [])
            valid = [a for a in aggs if a.get("avg_cosine_sim") is not None]
            if not valid:
                continue

            total_pos  = sum(a["total_count"] for a in aggs)
            total_pass = sum(a["pass_count"]  for a in aggs)

            rows.append({
                "model":       model_id,
                "threshold":   threshold,
                "avg_compress": round(sum(a["compression_pct"] for a in valid) / len(valid), 2),
                "avg_cosine":  round(sum(a["avg_cosine_sim"]   for a in valid) / len(valid), 8),
                "min_cosine":  round(min(a["min_cosine_sim"]   for a in valid), 8),
                "avg_rank":    round(sum(a["avg_rank"]         for a in valid) / len(valid), 4),
                "max_rank":    max(a["max_rank"] for a in valid),
                "pass_rate":   round(100.0 * total_pass / total_pos, 2) if total_pos else 0.0,
                "total_positions": total_pos,
                "total_pass":      total_pass,
            })

    return rows


def print_summary_table(rows: list[dict]) -> None:
    """Print a fixed-width aligned table to stdout."""
    if not rows:
        print("  (no results to display)")
        return

    # column widths (right-justified values, left-justified model names)
    C = [22, 10, 12, 11, 11, 10, 10, 9]
    headers = ["Model", "Threshold", "Compress%", "Avg Cosine", "Min Cosine",
               "Avg Rank", "Max Rank", "Pass%"]

    def fmt(vals: list) -> str:
        parts = []
        for v, w in zip(vals, C):
            s = str(v)
            parts.append(s.ljust(w) if len(parts) == 0 else s.rjust(w))
        return "  " + "  ".join(parts)

    sep = "  " + "-" * (sum(C) + 2 * (len(C) - 1))

    print()
    print(fmt(headers))
    print(sep)

    prev_model = None
    for r in rows:
        if r["model"] != prev_model and prev_model is not None:
            print(sep)
        model_disp = r["model"].split("/")[-1] if r["model"] != prev_model else ""
        prev_model = r["model"]

        print(fmt([
            model_disp,
            f"{r['threshold']:.5f}",
            f"{r['avg_compress']:.1f}%",
            f"{r['avg_cosine']:.6f}",
            f"{r['min_cosine']:.6f}",
            f"{r['avg_rank']:.2f}",
            str(r["max_rank"]),
            f"{r['pass_rate']:.1f}%",
        ]))

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# File output
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results: list[dict], output_dir: Path) -> None:
    """Write full per-position data to results.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=True)
    print(f"  Saved per-position data → {path}  ({path.stat().st_size // 1024} KB)")


def save_csv(rows: list[dict], output_dir: Path) -> None:
    """Write summary rows to summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    fieldnames = [
        "model", "threshold", "avg_compress", "avg_cosine", "min_cosine",
        "avg_rank", "max_rank", "pass_rate", "total_positions", "total_pass",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved summary CSV      → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_STYLES: dict[str, dict] = {
    "Qwen/Qwen2.5-0.5B": {"color": "#2196F3", "marker": "o", "label": "Qwen2.5-0.5B"},
    "Qwen/Qwen2.5-3B":   {"color": "#FF9800", "marker": "s", "label": "Qwen2.5-3B"},
    "Qwen/Qwen2.5-7B":   {"color": "#4CAF50", "marker": "^", "label": "Qwen2.5-7B"},
}


def _group_by_model(rows: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    # sort each group by threshold ascending
    for model_id in g:
        g[model_id].sort(key=lambda r: r["threshold"])
    return g


def plot_results(rows: list[dict], output_dir: Path) -> None:
    """Generate threshold_vs_cosine.png and threshold_vs_compression.png."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available — skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_by_model(rows)

    # ── Plot 1: threshold vs average cosine similarity ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_id, model_rows in grouped.items():
        style = _MODEL_STYLES.get(model_id, {"color": "grey", "marker": "x", "label": model_id})
        xs = [r["threshold"]   for r in model_rows]
        ys = [r["avg_cosine"]  for r in model_rows]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Neuron activation threshold", fontsize=12)
    ax.set_ylabel("Average cosine similarity\n(sparse logits vs full-model logits)", fontsize=11)
    ax.set_title("Sparse replay fidelity vs activation threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()   # left=high threshold, right=low threshold (→ more accurate)
    fig.tight_layout()
    path1 = output_dir / "threshold_vs_cosine.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved plot             → {path1}")

    # ── Plot 2: threshold vs compression percentage ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_id, model_rows in grouped.items():
        style = _MODEL_STYLES.get(model_id, {"color": "grey", "marker": "x", "label": model_id})
        xs = [r["threshold"]    for r in model_rows]
        ys = [r["avg_compress"] for r in model_rows]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Neuron activation threshold", fontsize=12)
    ax.set_ylabel("Neurons zeroed out (%)", fontsize=11)
    ax.set_title("Mask compression vs activation threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path2 = output_dir / "threshold_vs_compression.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved plot             → {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Runtime estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_runtime(
    models: list[str],
    thresholds: list[float],
    num_prompts: int,
) -> None:
    """Print an estimated runtime breakdown before the sweep starts."""
    print("\n  Estimated runtime:")
    total = 0.0
    for model_id in models:
        gen_sec  = _APPROX_GEN_SEC.get(model_id, 2.0)
        pass_sec = _APPROX_PASS_SEC.get(model_id, 0.5)
        # 1 generation + 1 full pass + len(thresholds) sparse passes
        per_prompt = gen_sec + (1 + len(thresholds)) * pass_sec
        model_sec  = num_prompts * per_prompt + 10  # +10s for load/unload
        label      = model_id.split("/")[-1]
        print(f"    {label:<18} {num_prompts} prompts × "
              f"({gen_sec:.1f}s gen + {1 + len(thresholds)} passes × {pass_sec:.2f}s)"
              f" + 10s load  ≈  {model_sec:.0f}s")
        total += model_sec
    print(f"    {'TOTAL':<18} ≈ {total:.0f}s  (~{total / 60:.1f} min)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Interrupt handler
# ─────────────────────────────────────────────────────────────────────────────

_accumulated_results: list[dict] = []
_interrupt_output_dir: Path = OUTPUT_DIR


def _on_interrupt(sig, frame) -> None:  # noqa: ANN001
    print("\n\n  [!] Interrupted — saving partial results…")
    if _accumulated_results:
        rows = _table_rows(
            _accumulated_results,
            list({r["model"] for r in _accumulated_results}),
            list({r["threshold"] for r in _accumulated_results}),
        )
        print_summary_table(rows)
        _interrupt_output_dir.mkdir(parents=True, exist_ok=True)
        save_results(_accumulated_results, _interrupt_output_dir)
        save_csv(rows, _interrupt_output_dir)
        plot_results(rows, _interrupt_output_dir)
    else:
        print("  No results accumulated yet.")
    sys.exit(0)


signal.signal(signal.SIGINT, _on_interrupt)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(msg: str) -> None:
    width = 68
    print()
    print("  " + "═" * width)
    print(f"  {msg}")
    print("  " + "═" * width)


def _info(msg: str) -> None:
    print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep neuron-mask activation thresholds across registered models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--neuron-threshold",
        type=float,
        nargs="+",
        metavar="T",
        default=None,
        help=(
            "One or more activation thresholds to test.  "
            "Default: sweep [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]."
        ),
    )
    p.add_argument(
        "--match-cosine",
        type=float,
        default=1.0,
        metavar="C",
        help=(
            "Minimum cosine similarity (sparse vs full logits) for a position "
            "to count as matched.  Default 1.0 uses top-k rank only."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Maximum rank of the correct token in sparse predictions for the "
            "position to count as matched.  Default 1 (must be top prediction)."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        action="append",
        dest="model_flags",
        metavar="MODEL_ID",
        help=(
            "Model to include.  May be repeated for multiple models.  "
            "Use 'all' to include every model in the registry."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        dest="models_shorthand",
        metavar="all",
        help="Shorthand: --models all includes every registered model.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip saving per-position data to results.json; print table and save CSV only.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory for output files.  Default: {OUTPUT_DIR}",
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        metavar="PATH",
        help="Path to registry.json.  Default: registry.json next to this script.",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _accumulated_results, _interrupt_output_dir

    args = parse_args()

    # ── Validate arguments ───────────────────────────────────────────────────
    if not (0.0 <= args.match_cosine <= 1.0):
        print(f"Error: --match-cosine must be in [0, 1], got {args.match_cosine}")
        sys.exit(1)
    if args.top_k < 1:
        print(f"Error: --top-k must be >= 1, got {args.top_k}")
        sys.exit(1)

    # ── Resolve thresholds ───────────────────────────────────────────────────
    thresholds: list[float] = (
        args.neuron_threshold if args.neuron_threshold else DEFAULT_THRESHOLDS
    )
    thresholds = sorted(set(thresholds), reverse=True)  # deduplicate, high→low

    # ── Resolve model list ───────────────────────────────────────────────────
    registry = ModelRegistry(args.registry)
    all_registered = registry.list_models()

    # Combine --model (repeatable) and --models (shorthand) into one list.
    raw_flags: list[str] = list(args.model_flags or [])
    if args.models_shorthand:
        raw_flags.append(args.models_shorthand)

    if not raw_flags or "all" in [f.lower() for f in raw_flags]:
        # No filter → use all registered models in preferred order
        selected_models = [m for m in MODEL_ORDER if m in all_registered]
        for m in all_registered:
            if m not in selected_models:
                selected_models.append(m)
    else:
        selected_models = []
        for m in raw_flags:
            if m not in all_registered:
                print(f"Error: model {m!r} is not in the registry.")
                print(f"Registered models: {all_registered}")
                sys.exit(1)
            if m not in selected_models:
                selected_models.append(m)
        # Honour MODEL_ORDER within the selection
        ordered    = [m for m in MODEL_ORDER if m in selected_models]
        remainder  = [m for m in selected_models if m not in ordered]
        selected_models = ordered + remainder

    _interrupt_output_dir = args.output_dir
    save_positions = not args.summary_only

    # ── Banner ───────────────────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║           NEURON MASK THRESHOLD STUDY                       ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Models:       {', '.join(m.split('/')[-1] for m in selected_models)}")
    print(f"  Thresholds:   {thresholds}")
    print(f"  Prompts:      {len(PROMPTS)}")
    print(f"  Tokens/prompt:{NUM_TOKENS}")
    print(f"  Match cosine: {args.match_cosine}  |  Top-k: {args.top_k}")
    print(f"  Output dir:   {args.output_dir}")
    if args.summary_only:
        print("  Mode:         summary-only (no per-position JSON)")

    estimate_runtime(selected_models, thresholds, len(PROMPTS))

    # ── Sweep ────────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    wall_start = time.perf_counter()

    for model_id in selected_models:
        with StudyModelContext(model_id, registry) as ctx:
            model_results = sweep_model(
                model_id      = model_id,
                model         = ctx.model,
                tokenizer     = ctx.tokenizer,
                prompts       = PROMPTS,
                thresholds    = thresholds,
                match_cosine  = args.match_cosine,
                top_k         = args.top_k,
                save_positions= save_positions,
            )
        all_results.extend(model_results)
        _accumulated_results = all_results  # keep interrupt handler in sync

    wall_sec = time.perf_counter() - wall_start

    # ── Output ───────────────────────────────────────────────────────────────
    rows = _table_rows(all_results, selected_models, thresholds)

    _hdr("RESULTS SUMMARY")
    print_summary_table(rows)

    print(f"  Total wall time: {wall_sec:.1f}s  ({wall_sec / 60:.1f} min)")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        save_results(all_results, args.output_dir)
    save_csv(rows, args.output_dir)
    plot_results(rows, args.output_dir)

    print()
    print("  Done.")


if __name__ == "__main__":
    main()
