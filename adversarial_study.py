#!/usr/bin/env python3
"""
adversarial_study.py — measure the security margin between honest and fraudulent
verification in the sparse-replay proof-of-inference system.

Three attack categories are tested:

  wrong-model   A provider runs a cheap model but claims an expensive one.
                The verifier loads the claimed (expensive) model and checks
                whether it would have predicted the attacker's tokens.
                Test pairs: 0.5B→3B, 0.5B→7B, 3B→7B.

  random-masks  A provider generates honest tokens but submits random neuron
                masks.  The verifier replays with those masks; wrong neurons
                produce divergent logits that don't match the claimed tokens.

  token-swap    A provider takes honest output and replaces 1, 3, 5, or 10
                tokens with random vocabulary tokens, then submits the
                corrupted sequence for verification.

For every attack the matching honest baseline is measured with identical
infrastructure so the security gap can be quantified directly.

Primary metric: cos_sim_perfect — cosine similarity between the verifier's
softmax probability distribution and the one-hot distribution for the claimed
token.  Ranges [0, 1].  Honest ≈ 1.0, fraud ≈ 0.  The security margin is
    honest_worst_case_cos - fraud_best_case_cos

Usage
-----
  python3 adversarial_study.py                         # all attacks
  python3 adversarial_study.py --attack wrong-model
  python3 adversarial_study.py --attack random-masks
  python3 adversarial_study.py --attack token-swap
  python3 adversarial_study.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import random as pyrandom
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

# ── Suppress transformers verbosity ──────────────────────────────────────────
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
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
# Constants — shared with threshold_study.py
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS: list[str] = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
    'def fibonacci(n):\n    """Return the nth Fibonacci number.\"\"\"\n',
    "SELECT name, email FROM users WHERE last_login <",
    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins are",
    "The key difference between supervised and unsupervised machine learning is that",
    "In July 1969, humans first landed on the Moon. The mission was called Apollo",
]

NUM_TOKENS = 20

# Cosine thresholds for per-threshold pass rates in the security table
COSINE_THRESHOLDS: list[float] = [0.99, 0.98, 0.95, 0.90, 0.85, 0.80]

# Threshold used to build union masks for the random-mask attack
MASK_THRESHOLD: float = 0.01

# Token substitution levels for the token-swap attack
SWAP_LEVELS: list[int] = [1, 3, 5, 10]

OUTPUT_DIR = Path("analysis_results/adversarial_study")

_KNOWN_MEM_GB: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B": 1.0,
    "Qwen/Qwen2.5-3B":   6.2,
    "Qwen/Qwen2.5-7B":  14.7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Model context
# ─────────────────────────────────────────────────────────────────────────────

class AdvModelContext:
    """Load a model via the registry and free memory on exit."""

    def __init__(self, model_id: str, registry: ModelRegistry):
        self.model_id = model_id
        self._registry = registry
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "AdvModelContext":
        _hdr(f"Loading {self.model_id}  (~{_KNOWN_MEM_GB.get(self.model_id, '?')} GB)")
        t0 = time.perf_counter()
        self.model, self.tokenizer = self._registry.load_verified_model(self.model_id)
        _info(f"Loaded in {time.perf_counter() - t0:.1f}s")
        return self

    def __exit__(self, *_):
        _info(f"Unloading {self.model_id}…")
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Core inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_tokens(model, tokenizer, prompt: str, num_tokens: int = NUM_TOKENS) -> list[int]:
    """Autoregressive greedy generation. Returns generated token IDs only."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=num_tokens,
            do_sample=False,
        )
    return out[0][inputs["input_ids"].shape[1]:].tolist()


def generate_tokens_with_maxabs(
    model, tokenizer, prompt: str, num_tokens: int = NUM_TOKENS
) -> tuple[list[int], list[torch.Tensor]]:
    """
    Autoregressive generation + per-layer max |gate*up| recording.
    Returns (token_ids, max_abs) where max_abs[l] is [intermediate_size] float32 CPU.
    """
    device = next(model.parameters()).device
    intermediate_size = model.config.intermediate_size
    num_layers = len(model.model.layers)

    max_abs = [torch.zeros(intermediate_size, dtype=torch.float32) for _ in range(num_layers)]

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            x = inp[0]
            with torch.no_grad():
                inter_abs = (module.act_fn(module.gate_proj(x)) * module.up_proj(x)).float().abs()
            max_abs[layer_idx] = torch.maximum(max_abs[layer_idx], inter_abs.amax(dim=(0, 1)).cpu())
            return out
        return hook_fn

    hooks = [l.mlp.register_forward_hook(make_hook(i)) for i, l in enumerate(model.model.layers)]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(inputs["input_ids"], max_new_tokens=num_tokens, do_sample=False)
        token_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    finally:
        for h in hooks: h.remove()

    return token_ids, max_abs


def forward_pass_logits(
    model,
    tokenizer,
    prompt: str,
    token_ids: list[int],
    masks: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Single forward pass of [prompt + token_ids].

    If masks is not None, applies them as MLP hooks (sparse replay).
    Returns float32 CPU tensor [T, vocab_size] — row i predicts token_ids[i].
    """
    device = next(model.parameters()).device
    T = len(token_ids)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    gen_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    L = prompt_ids.shape[1]

    hooks: list = []
    if masks is not None:
        dmasks = [m.to(device) for m in masks]

        def make_hook(bm: torch.Tensor):
            def hook_fn(module, inp, _out):
                x = inp[0]
                inter = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
                return module.down_proj(inter * bm.to(dtype=inter.dtype))
            return hook_fn

        hooks = [model.model.layers[i].mlp.register_forward_hook(make_hook(dm))
                 for i, dm in enumerate(dmasks)]

    try:
        with torch.no_grad():
            out = model(full_ids)
        # logit at position j predicts token j+1; first generated token is at index L
        logits = out.logits[0, L - 1 : L - 1 + T, :].float()
    finally:
        for h in hooks: h.remove()

    return logits.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────────────

def honest_masks_from_maxabs(max_abs: list[torch.Tensor], threshold: float = MASK_THRESHOLD):
    """Build union masks: True where max |activation| > threshold."""
    return [m > threshold for m in max_abs]


def random_masks_same_density(
    max_abs: list[torch.Tensor],
    threshold: float = MASK_THRESHOLD,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Build masks with the same number of active neurons per layer as the honest
    masks at `threshold`, but choosing random neurons instead of the ones that
    actually fired.
    """
    rng = pyrandom.Random(seed)
    result: list[torch.Tensor] = []
    for m in max_abs:
        n_active = int((m > threshold).sum().item())
        dim = m.shape[0]
        if n_active == 0:
            result.append(torch.zeros(dim, dtype=torch.bool))
        elif n_active >= dim:
            result.append(torch.ones(dim, dtype=torch.bool))
        else:
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[torch.tensor(rng.sample(range(dim), n_active))] = True
            result.append(mask)
    return result


def swap_tokens(
    token_ids: list[int],
    n_swaps: int,
    vocab_size: int,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Replace `n_swaps` randomly chosen positions with random vocabulary tokens.
    Returns (corrupted_ids, swapped_positions).
    """
    rng = pyrandom.Random(seed)
    ids = list(token_ids)
    positions = sorted(rng.sample(range(len(ids)), min(n_swaps, len(ids))))
    for pos in positions:
        # pick a different token
        replacement = rng.randint(0, vocab_size - 1)
        while replacement == ids[pos] and vocab_size > 1:
            replacement = rng.randint(0, vocab_size - 1)
        ids[pos] = replacement
    return ids, positions


# ─────────────────────────────────────────────────────────────────────────────
# Per-position metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_metrics(
    logits: torch.Tensor,       # [T, vocab_size] float32
    claimed_token_ids: list[int],
    swapped_positions: Optional[list[int]] = None,
) -> list[dict]:
    """
    For each generated position, compute:

      cos_sim_perfect  — cosine similarity between the verifier's softmax
                         probability vector and the one-hot for the claimed token.
                         Equivalent to  prob[claimed] / ||prob||.
                         Honest ≈ 1.0 (model strongly predicts claimed token).
                         Fraud ≈ 0   (model predicts a completely different token).

      token_prob       — softmax probability assigned to the claimed token.
      rank             — rank of claimed token in descending logit order (1=top).
      pass_XX          — whether cos_sim_perfect >= 0.XX for each threshold.
    """
    T = len(claimed_token_ids)
    swapped_set = set(swapped_positions or [])
    results: list[dict] = []

    for i in range(T):
        v = logits[i]             # [V] float32
        claimed = claimed_token_ids[i]

        probs = torch.softmax(v, dim=-1)           # [V]
        token_prob = float(probs[claimed].item())
        norm_p = float(probs.norm(p=2).item())
        cos_sim = token_prob / norm_p if norm_p > 0 else 0.0
        cos_sim = max(0.0, min(1.0, cos_sim))

        rank = int((v > v[claimed]).sum().item()) + 1

        entry: dict = {
            "position":       i,
            "claimed_token":  claimed,
            "cos_sim_perfect": round(cos_sim, 8),
            "token_prob":     round(token_prob, 8),
            "rank":           rank,
        }
        if swapped_positions is not None:
            entry["was_swapped"] = i in swapped_set
        for t in COSINE_THRESHOLDS:
            entry[f"pass_{int(t * 100)}"] = cos_sim >= t

        results.append(entry)

    return results


def aggregate_metrics(positions: list[dict]) -> dict:
    """Compute per-scenario aggregate statistics from per-position data."""
    if not positions:
        return {}

    cos_sims = [p["cos_sim_perfect"] for p in positions]
    ranks    = [p["rank"]            for p in positions]

    agg: dict = {
        "n_positions":   len(positions),
        "avg_cos":       round(sum(cos_sims) / len(cos_sims), 8),
        "worst_cos":     round(min(cos_sims), 8),   # lowest fidelity = worst case for honest
        "best_cos":      round(max(cos_sims), 8),   # highest fidelity = best case for fraud
        "avg_rank":      round(sum(ranks) / len(ranks), 4),
        "best_rank":     min(ranks),                # best attacker outcome
        "worst_rank":    max(ranks),
    }
    for t in COSINE_THRESHOLDS:
        key = f"pass_{int(t * 100)}"
        agg[key] = round(sum(p[key] for p in positions) / len(positions), 4)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Wrong-model attack orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_wrong_model_attacks(
    registry: ModelRegistry,
    prompts: list[str],
    save_positions: bool = True,
) -> list[dict]:
    """
    Three substitution pairs.  Memory-optimal sequence:
      1. Load 0.5B → generate all attacker outputs → unload
      2. Load 3B   → generate honest tokens + verify 0.5B fraud → unload
      3. Load 7B   → generate honest tokens + verify 0.5B & 3B fraud + honest → unload
    """
    results: list[dict] = []

    # ── Step 1: attacker generates with 0.5B ─────────────────────────────────
    _section("Generating attacker tokens: 0.5B")
    with AdvModelContext("Qwen/Qwen2.5-0.5B", registry) as ctx:
        tokens_05B = []
        for i, prompt in enumerate(prompts):
            toks = generate_tokens(ctx.model, ctx.tokenizer, prompt)
            tokens_05B.append(toks)
            _info(f"  [{i+1:>2}/{len(prompts)}] {len(toks)} tokens")

    # ── Step 2: 3B verifier ───────────────────────────────────────────────────
    _section("Verifier: 3B  (honest baseline + fraud from 0.5B)")
    with AdvModelContext("Qwen/Qwen2.5-3B", registry) as ctx:
        tokens_3B = []
        for i, prompt in enumerate(prompts):
            toks = generate_tokens(ctx.model, ctx.tokenizer, prompt)
            tokens_3B.append(toks)

        for i, prompt in enumerate(prompts):
            # Honest: 3B verifier sees 3B tokens
            if tokens_3B[i]:
                logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, tokens_3B[i])
                positions = compute_position_metrics(logits, tokens_3B[i])
                results.append(_make_result(
                    scenario="honest_3B",
                    is_honest=True,
                    attacker="Qwen/Qwen2.5-3B",
                    verifier="Qwen/Qwen2.5-3B",
                    prompt_idx=i, prompt=prompt,
                    token_ids=tokens_3B[i],
                    positions=positions,
                    save_positions=save_positions,
                ))

            # Fraud: 3B verifier sees 0.5B tokens
            if tokens_05B[i]:
                logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, tokens_05B[i])
                positions = compute_position_metrics(logits, tokens_05B[i])
                results.append(_make_result(
                    scenario="wrong_model_0.5B_claims_3B",
                    is_honest=False,
                    attacker="Qwen/Qwen2.5-0.5B",
                    verifier="Qwen/Qwen2.5-3B",
                    prompt_idx=i, prompt=prompt,
                    token_ids=tokens_05B[i],
                    positions=positions,
                    save_positions=save_positions,
                ))

        _print_inline_pair(
            results, "honest_3B", "wrong_model_0.5B_claims_3B",
            "Pair 2: 0.5B→3B"
        )

    # ── Step 3: 7B verifier ───────────────────────────────────────────────────
    _section("Verifier: 7B  (honest baseline + fraud from 0.5B + fraud from 3B)")
    with AdvModelContext("Qwen/Qwen2.5-7B", registry) as ctx:
        tokens_7B = []
        for i, prompt in enumerate(prompts):
            toks = generate_tokens(ctx.model, ctx.tokenizer, prompt)
            tokens_7B.append(toks)

        for i, prompt in enumerate(prompts):
            # Honest: 7B verifier sees 7B tokens
            if tokens_7B[i]:
                logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, tokens_7B[i])
                positions = compute_position_metrics(logits, tokens_7B[i])
                results.append(_make_result(
                    scenario="honest_7B",
                    is_honest=True,
                    attacker="Qwen/Qwen2.5-7B",
                    verifier="Qwen/Qwen2.5-7B",
                    prompt_idx=i, prompt=prompt,
                    token_ids=tokens_7B[i],
                    positions=positions,
                    save_positions=save_positions,
                ))

            # Fraud: 7B verifier sees 0.5B tokens (Pair 1)
            if tokens_05B[i]:
                logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, tokens_05B[i])
                positions = compute_position_metrics(logits, tokens_05B[i])
                results.append(_make_result(
                    scenario="wrong_model_0.5B_claims_7B",
                    is_honest=False,
                    attacker="Qwen/Qwen2.5-0.5B",
                    verifier="Qwen/Qwen2.5-7B",
                    prompt_idx=i, prompt=prompt,
                    token_ids=tokens_05B[i],
                    positions=positions,
                    save_positions=save_positions,
                ))

            # Fraud: 7B verifier sees 3B tokens (Pair 3)
            if tokens_3B[i]:
                logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, tokens_3B[i])
                positions = compute_position_metrics(logits, tokens_3B[i])
                results.append(_make_result(
                    scenario="wrong_model_3B_claims_7B",
                    is_honest=False,
                    attacker="Qwen/Qwen2.5-3B",
                    verifier="Qwen/Qwen2.5-7B",
                    prompt_idx=i, prompt=prompt,
                    token_ids=tokens_3B[i],
                    positions=positions,
                    save_positions=save_positions,
                ))

        _print_inline_pair(results, "honest_7B", "wrong_model_0.5B_claims_7B", "Pair 1: 0.5B→7B")
        _print_inline_pair(results, "honest_7B", "wrong_model_3B_claims_7B",   "Pair 3: 3B→7B")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Random-mask attack orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_random_mask_attacks(
    registry: ModelRegistry,
    prompts: list[str],
    save_positions: bool = True,
) -> list[dict]:
    """
    For each prompt, compare honest-mask sparse replay vs random-mask sparse replay.
    Uses the 7B model at MASK_THRESHOLD.
    """
    results: list[dict] = []
    model_id = "Qwen/Qwen2.5-7B"

    _section(f"Random-mask attack on {model_id}  (threshold={MASK_THRESHOLD})")
    with AdvModelContext(model_id, registry) as ctx:
        for i, prompt in enumerate(prompts):
            short = prompt[:55].replace("\n", "↵")
            print(f"\n  [{i+1:>2}/{len(prompts)}] \"{short}\"")

            token_ids, max_abs = generate_tokens_with_maxabs(ctx.model, ctx.tokenizer, prompt)
            if not token_ids:
                print("         → 0 tokens, skip")
                continue
            print(f"         → {len(token_ids)} tokens")

            h_masks = honest_masks_from_maxabs(max_abs, MASK_THRESHOLD)
            r_masks = random_masks_same_density(max_abs, MASK_THRESHOLD, seed=i * 100)

            total_n  = sum(m.numel() for m in h_masks)
            active_h = sum(int(m.sum().item()) for m in h_masks)
            active_r = sum(int(m.sum().item()) for m in r_masks)
            compress_h = 100.0 * (total_n - active_h) / total_n
            compress_r = 100.0 * (total_n - active_r) / total_n
            print(f"         honest mask: {100 - compress_h:.1f}% density  |  "
                  f"random mask: {100 - compress_r:.1f}% density  (same)")
            # Per-layer zeroed neuron counts (first prompt only to avoid spam)
            if i == 0:
                print(f"         Per-layer zeroed neurons (out of {h_masks[0].numel()}):")
                for li, (hm, rm) in enumerate(zip(h_masks, r_masks)):
                    dim = hm.numel()
                    h_zeroed = dim - int(hm.sum().item())
                    r_zeroed = dim - int(rm.sum().item())
                    print(f"           layer {li:02d}: honest zeroed={h_zeroed:6d}  random zeroed={r_zeroed:6d}")

            # Honest mask replay
            h_logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, token_ids, h_masks)
            h_positions = compute_position_metrics(h_logits, token_ids)
            results.append(_make_result(
                scenario="random_masks_honest",
                is_honest=True,
                attacker=model_id, verifier=model_id,
                prompt_idx=i, prompt=prompt,
                token_ids=token_ids,
                positions=h_positions,
                save_positions=save_positions,
                extra={"mask_threshold": MASK_THRESHOLD},
            ))

            # Random mask replay
            r_logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, token_ids, r_masks)
            r_positions = compute_position_metrics(r_logits, token_ids)
            results.append(_make_result(
                scenario="random_masks_fraud",
                is_honest=False,
                attacker=model_id, verifier=model_id,
                prompt_idx=i, prompt=prompt,
                token_ids=token_ids,
                positions=r_positions,
                save_positions=save_positions,
                extra={"mask_threshold": MASK_THRESHOLD},
            ))

            h_agg = aggregate_metrics(h_positions)
            r_agg = aggregate_metrics(r_positions)
            print(f"         honest:  avg_cos={h_agg['avg_cos']:.6f}  rank_avg={h_agg['avg_rank']:.1f}")
            print(f"         random:  avg_cos={r_agg['avg_cos']:.6f}  rank_avg={r_agg['avg_rank']:.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Token-swap attack orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_token_swap_attacks(
    registry: ModelRegistry,
    prompts: list[str],
    save_positions: bool = True,
) -> list[dict]:
    """
    Measure how cosine_sim_perfect degrades as the number of corrupted tokens
    increases from 1 to 10.  Uses 7B model, full (unmasked) forward pass.
    """
    results: list[dict] = []
    model_id = "Qwen/Qwen2.5-7B"

    _section(f"Token-swap attack on {model_id}")
    with AdvModelContext(model_id, registry) as ctx:
        vocab_size = ctx.model.config.vocab_size

        # Generate honest tokens once
        honest_tokens_list: list[list[int]] = []
        for prompt in prompts:
            honest_tokens_list.append(generate_tokens(ctx.model, ctx.tokenizer, prompt))

        for i, (prompt, token_ids) in enumerate(zip(prompts, honest_tokens_list)):
            if not token_ids:
                continue
            short = prompt[:55].replace("\n", "↵")
            print(f"\n  [{i+1:>2}/{len(prompts)}] \"{short}\"  ({len(token_ids)} tokens)")

            # Honest baseline for this prompt
            logits = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, token_ids)
            h_positions = compute_position_metrics(logits, token_ids)
            results.append(_make_result(
                scenario="token_swap_honest",
                is_honest=True,
                attacker=model_id, verifier=model_id,
                prompt_idx=i, prompt=prompt,
                token_ids=token_ids,
                positions=h_positions,
                save_positions=save_positions,
                extra={"n_swaps": 0},
            ))
            h_agg = aggregate_metrics(h_positions)
            print(f"         honest:   avg_cos={h_agg['avg_cos']:.6f}  pass_99={h_agg['pass_99']:.0%}")

            for n_swaps in SWAP_LEVELS:
                if n_swaps > len(token_ids):
                    continue
                corrupted, swapped_pos = swap_tokens(
                    token_ids, n_swaps, vocab_size, seed=i * 1000 + n_swaps
                )
                logits_c = forward_pass_logits(ctx.model, ctx.tokenizer, prompt, corrupted)
                c_positions = compute_position_metrics(logits_c, corrupted, swapped_pos)
                results.append(_make_result(
                    scenario=f"token_swap_{n_swaps}",
                    is_honest=False,
                    attacker=model_id, verifier=model_id,
                    prompt_idx=i, prompt=prompt,
                    token_ids=corrupted,
                    positions=c_positions,
                    save_positions=save_positions,
                    extra={"n_swaps": n_swaps, "swapped_positions": swapped_pos},
                ))
                c_agg = aggregate_metrics(c_positions)
                print(f"         swap={n_swaps:>2}:   avg_cos={c_agg['avg_cos']:.6f}  "
                      f"pass_99={c_agg['pass_99']:.0%}  (at swapped: "
                      f"avg={_avg_swapped_cos(c_positions):.4f})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Security margin computation
# ─────────────────────────────────────────────────────────────────────────────

_HONEST_FOR_FRAUD: dict[str, str] = {
    "wrong_model_0.5B_claims_3B": "honest_3B",
    "wrong_model_0.5B_claims_7B": "honest_7B",
    "wrong_model_3B_claims_7B":   "honest_7B",
    "random_masks_fraud":          "random_masks_honest",
    **{f"token_swap_{n}": "token_swap_honest" for n in SWAP_LEVELS},
}


def compute_security_margins(all_results: list[dict]) -> list[dict]:
    """
    For each (fraud_scenario, honest_scenario) pair compute the security gap:
      honest_worst_cos — worst (min) cosine across all honest positions
      fraud_best_cos   — best  (max) cosine across all fraud  positions
      gap              — honest_worst - fraud_best  (positive = separable)
    """
    from collections import defaultdict
    # Collect all cos_sim_perfect values per scenario key
    cos_by_scenario: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        key = r["scenario"]
        for pos in r.get("positions") or []:
            cos_by_scenario[key].append(pos["cos_sim_perfect"])

    margins: list[dict] = []
    seen_honest: set[str] = set()

    for fraud_key, honest_key in _HONEST_FOR_FRAUD.items():
        if fraud_key not in cos_by_scenario:
            continue
        fraud_cosines  = cos_by_scenario[fraud_key]
        honest_cosines = cos_by_scenario.get(honest_key, [])
        if not fraud_cosines or not honest_cosines:
            continue

        honest_worst = min(honest_cosines)
        fraud_best   = max(fraud_cosines)
        gap          = honest_worst - fraud_best

        margins.append({
            "fraud_scenario":  fraud_key,
            "honest_scenario": honest_key,
            "honest_n":        len(honest_cosines),
            "fraud_n":         len(fraud_cosines),
            "honest_avg":      round(sum(honest_cosines) / len(honest_cosines), 6),
            "honest_worst":    round(honest_worst, 6),
            "fraud_avg":       round(sum(fraud_cosines) / len(fraud_cosines), 6),
            "fraud_best":      round(fraud_best, 6),
            "gap":             round(gap, 6),
            "secure":          gap > 0,
            "honest_cosines":  honest_cosines,
            "fraud_cosines":   fraud_cosines,
        })

    return margins


# ─────────────────────────────────────────────────────────────────────────────
# Security margin table
# ─────────────────────────────────────────────────────────────────────────────

def print_security_margin_table(margins: list[dict], all_results: list[dict]) -> None:
    """
    Print two-part table: per-attack cos summary + per-threshold pass rates.
    Annotates each fraud row with the gap to the corresponding honest baseline.
    """
    if not margins:
        print("  (no results)")
        return

    print()
    _rule()
    print("  SECURITY MARGIN SUMMARY")
    _rule()

    # Part 1: aggregated metrics
    hdrs = ["Scenario", "n", "Avg cos", "Worst/Best cos", "Avg rank", "Best rank"]
    cw   = [36, 6, 10, 16, 10, 10]

    def fmt(vals): return "  " + "  ".join(str(v).rjust(w) for v, w in zip(vals, cw))
    sep = "  " + "-" * (sum(cw) + 2 * (len(cw) - 1))

    from collections import defaultdict
    agg_pool: dict[str, list] = defaultdict(list)
    for r in all_results:
        agg_pool[r["scenario"]].append(r["aggregates"])

    def pool_agg(scenario: str) -> Optional[dict]:
        pool = agg_pool.get(scenario, [])
        if not pool: return None
        all_cos = []
        all_ranks = []
        for a in pool:
            if a:
                all_cos.append(a.get("avg_cos", 0))
                all_ranks.append(a.get("avg_rank", 0))
        return {
            "avg_cos": round(sum(all_cos)/len(all_cos), 6) if all_cos else 0,
            "avg_rank": round(sum(all_ranks)/len(all_ranks), 2) if all_ranks else 0,
        }

    print()
    print(fmt(hdrs))
    print(sep)

    printed_honest: set[str] = set()
    for m in margins:
        hs = m["honest_scenario"]
        fs = m["fraud_scenario"]

        # Print honest baseline once per unique scenario
        if hs not in printed_honest:
            printed_honest.add(hs)
            a = pool_agg(hs)
            cos_label = f"{m['honest_worst']:.6f} (worst)"
            if a:
                print(fmt([f"  HONEST: {hs}", m["honest_n"],
                           f"{m['honest_avg']:.6f}", cos_label,
                           f"{a['avg_rank']:.2f}", "1"]))

        # Fraud row
        a = pool_agg(fs)
        cos_label = f"{m['fraud_best']:.6f} (best)"
        gap_label = f"  ← gap {m['gap']:+.4f} {'✓ SECURE' if m['secure'] else '✗ OVERLAP'}"
        if a:
            print(fmt([f"  FRAUD:  {fs}", m["fraud_n"],
                       f"{m['fraud_avg']:.6f}", cos_label,
                       f"{a['avg_rank']:.2f}",
                       str(min((p.get("best_rank",1)) for p in agg_pool[fs] if p))]))
        print(f"  {'':36}  {gap_label}")
        print()

    print(sep)

    # Part 2: threshold pass rates
    print()
    print("  PASS RATES AT EACH COSINE THRESHOLD")
    print()
    thr_labels = [f"≥{t:.2f}" for t in COSINE_THRESHOLDS]
    cw2 = [36] + [8] * len(COSINE_THRESHOLDS)

    def fmt2(vals): return "  " + "  ".join(str(v).rjust(w) for v, w in zip(vals, cw2))
    sep2 = "  " + "-" * (sum(cw2) + 2 * (len(cw2) - 1))

    print(fmt2(["Scenario"] + thr_labels))
    print(sep2)

    all_scenarios = []
    for m in margins:
        if m["honest_scenario"] not in all_scenarios:
            all_scenarios.append(m["honest_scenario"])
        all_scenarios.append(m["fraud_scenario"])

    for scenario in all_scenarios:
        pool = agg_pool.get(scenario, [])
        if not pool: continue
        pass_rates = []
        for t in COSINE_THRESHOLDS:
            key = f"pass_{int(t * 100)}"
            vals = [a[key] for a in pool if a and key in a]
            pass_rates.append(f"{sum(vals)/len(vals):.1%}" if vals else "  —")
        prefix = "HONEST" if any(scenario == m["honest_scenario"] for m in margins) else "FRAUD "
        print(fmt2([f"  {prefix}: {scenario}"] + pass_rates))

    print(sep2)

    # Part 3: explicit security margin announcements
    print()
    for m in margins:
        secure_str = "SECURE — distributions are separable" if m["secure"] else \
                     "⚠ OVERLAP — not separable at any single threshold"
        print(f"  SECURITY MARGIN  {m['fraud_scenario']!s:<38}")
        print(f"    Honest worst case:  {m['honest_worst']:.6f}")
        print(f"    Fraud  best  case:  {m['fraud_best']:.6f}")
        print(f"    Gap:                {m['gap']:+.6f}  →  {secure_str}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# File output
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results: list[dict], output_dir: Path) -> None:
    """
    Write results to JSON.  For each entry, respect its _save_positions flag:
    if False, strip the positions list before writing to keep the file small.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"

    def _strip(r: dict) -> dict:
        out = {k: v for k, v in r.items() if k != "_save_positions"}
        if not r.get("_save_positions", True):
            out.pop("positions", None)
        return out

    with open(path, "w", encoding="utf-8") as fh:
        json.dump([_strip(r) for r in all_results], fh, indent=2, ensure_ascii=True)
    _info(f"Saved results.json  ({path.stat().st_size // 1024} KB)  → {path}")


def save_csv(all_results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for r in all_results:
        agg = r.get("aggregates") or {}
        base = {
            "scenario": r["scenario"],
            "is_honest": r["is_honest"],
            "attacker": r.get("attacker_model", ""),
            "verifier": r.get("verifier_model", ""),
            "prompt_idx": r["prompt_idx"],
            "n_tokens": len(r.get("claimed_token_ids") or []),
        }
        base.update(agg)
        rows.append(base)

    if not rows: return
    path = output_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _info(f"Saved summary.csv              → {path}")


def plot_security_margin(margins: list[dict], output_dir: Path) -> None:
    """
    Overlapping histograms of cos_sim_perfect for each honest/fraud pair.
    If the distributions don't overlap, the system is secure at any threshold
    between the two histograms.
    """
    if not HAS_MATPLOTLIB:
        _info("matplotlib not available — skipping plots")
        return

    n = len(margins)
    if n == 0: return

    output_dir.mkdir(parents=True, exist_ok=True)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, m in enumerate(margins):
        ax = axes[idx // ncols][idx % ncols]
        hc = m["honest_cosines"]
        fc = m["fraud_cosines"]

        bins = 40
        ax.hist(hc, bins=bins, alpha=0.55, color="#2196F3", label=f"Honest  (n={len(hc)})")
        ax.hist(fc, bins=bins, alpha=0.55, color="#F44336", label=f"Fraud   (n={len(fc)})")

        # Mark worst/best with vertical lines
        ax.axvline(m["honest_worst"], color="#0D47A1", linestyle="--", linewidth=1.4,
                   label=f"honest worst={m['honest_worst']:.4f}")
        ax.axvline(m["fraud_best"],   color="#B71C1C", linestyle="--", linewidth=1.4,
                   label=f"fraud best={m['fraud_best']:.4f}")

        if m["gap"] > 0:
            ax.axvspan(m["fraud_best"], m["honest_worst"], alpha=0.12, color="green",
                       label=f"gap={m['gap']:.4f}")

        title = m["fraud_scenario"].replace("_", "\n").replace("wrong\nmodel", "wrong-model")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("cos_sim_perfect", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Security Margin: Honest vs Fraudulent Verification Distributions",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    path = output_dir / "security_margin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _info(f"Saved security_margin.png     → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(
    scenario: str,
    is_honest: bool,
    attacker: str,
    verifier: str,
    prompt_idx: int,
    prompt: str,
    token_ids: list[int],
    positions: list[dict],
    save_positions: bool,
    extra: Optional[dict] = None,
) -> dict:
    agg = aggregate_metrics(positions)
    r: dict = {
        "scenario":          scenario,
        "is_honest":         is_honest,
        "attacker_model":    attacker,
        "verifier_model":    verifier,
        "prompt_idx":        prompt_idx,
        "prompt":            prompt,
        "claimed_token_ids": token_ids,
        "aggregates":        agg,
        # Always keep positions in memory for security margin computation.
        # save_positions controls whether they are written to disk.
        "positions":         positions,
        "_save_positions":   save_positions,
    }
    if extra:
        r.update(extra)
    return r


def _avg_swapped_cos(positions: list[dict]) -> float:
    swapped = [p["cos_sim_perfect"] for p in positions if p.get("was_swapped")]
    return sum(swapped) / len(swapped) if swapped else 0.0


def _print_inline_pair(results: list[dict], honest_key: str, fraud_key: str, label: str):
    honest_pool = [r["aggregates"] for r in results if r["scenario"] == honest_key and r["aggregates"]]
    fraud_pool  = [r["aggregates"] for r in results if r["scenario"] == fraud_key  and r["aggregates"]]
    if not honest_pool or not fraud_pool: return
    ha = sum(a["avg_cos"] for a in honest_pool) / len(honest_pool)
    fa = sum(a["avg_cos"] for a in fraud_pool)  / len(fraud_pool)
    hw = min(a["worst_cos"] for a in honest_pool)
    fb = max(a["best_cos"]  for a in fraud_pool)
    print(f"\n  {label}")
    print(f"    Honest avg cos={ha:.6f}  worst={hw:.6f}")
    print(f"    Fraud  avg cos={fa:.6f}  best ={fb:.6f}")
    print(f"    Gap: {hw - fb:+.6f}  {'✓ separable' if hw > fb else '⚠ overlap'}")


def _hdr(msg: str):   print(f"\n  ══{'═' * 62}\n  {msg}\n  ══{'═' * 62}")
def _section(msg: str): print(f"\n  ── {msg} {'─' * max(0, 60 - len(msg))}")
def _rule():           print("  " + "─" * 68)
def _info(msg: str):   print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Interrupt handling
# ─────────────────────────────────────────────────────────────────────────────

_accumulated: list[dict] = []
_output_dir_global: Path = OUTPUT_DIR


def _on_interrupt(sig, frame):
    print("\n\n  [!] Interrupted — saving partial results…")
    if _accumulated:
        margins = compute_security_margins(_accumulated)
        print_security_margin_table(margins, _accumulated)
        _output_dir_global.mkdir(parents=True, exist_ok=True)
        save_results(_accumulated, _output_dir_global)
        save_csv(_accumulated, _output_dir_global)
        plot_security_margin(margins, _output_dir_global)
    sys.exit(0)


signal.signal(signal.SIGINT, _on_interrupt)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure security margin between honest and fraudulent verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--attack",
        choices=["wrong-model", "random-masks", "token-swap"],
        action="append",
        dest="attacks",
        metavar="TYPE",
        help="Attack type(s) to run.  May be repeated.  Default: all three.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-position data in JSON output.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        metavar="PATH",
    )
    return p.parse_args()


def main() -> None:
    global _accumulated, _output_dir_global

    args = parse_args()
    attacks = set(args.attacks or ["wrong-model", "random-masks", "token-swap"])
    _output_dir_global = args.output_dir
    save_positions = not args.summary_only

    registry = ModelRegistry(args.registry)

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║           ADVERSARIAL VERIFICATION STUDY                    ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Attacks:    {', '.join(sorted(attacks))}")
    print(f"  Prompts:    {len(PROMPTS)}")
    print(f"  Tokens:     {NUM_TOKENS} per prompt")
    print(f"  Output dir: {args.output_dir}")
    if args.summary_only:
        print("  Mode:       summary-only (no per-position JSON)")
    print()

    all_results: list[dict] = []

    if "wrong-model" in attacks:
        results = run_wrong_model_attacks(registry, PROMPTS, save_positions)
        all_results.extend(results)
        _accumulated = all_results

    if "random-masks" in attacks:
        results = run_random_mask_attacks(registry, PROMPTS, save_positions)
        all_results.extend(results)
        _accumulated = all_results

    if "token-swap" in attacks:
        results = run_token_swap_attacks(registry, PROMPTS, save_positions)
        all_results.extend(results)
        _accumulated = all_results

    # ── Output ────────────────────────────────────────────────────────────────
    margins = compute_security_margins(all_results)
    print_security_margin_table(margins, all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_results(all_results, args.output_dir)   # respects per-entry _save_positions flag
    save_csv(all_results, args.output_dir)
    plot_security_margin(margins, args.output_dir)

    print()
    _info("Done.")


if __name__ == "__main__":
    main()
