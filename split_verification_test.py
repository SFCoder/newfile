#!/usr/bin/env python3
"""
split_verification_test.py

Tests whether FFN neuron computation can be split across multiple workers
and summed to produce the exact same result as a single full computation.

Hypothesis
----------
down_proj is a linear map over the intermediate activation (gate * up).
Splitting the intermediate into disjoint groups, applying down_proj to each
group's contribution (with all other neurons zeroed), and summing the partial
results must equal the full down_proj output:

    sum_g  down_proj(mask_g  ⊙  inter)  =  down_proj(inter)

This follows from linearity: sum_g mask_g = 1  →  sum_g (mask_g ⊙ inter) = inter.

Two implementations are verified against each other
---------------------------------------------------
zero-multiply   Zero non-group neurons in a full-size buffer, apply down_proj.
                Tests the conceptual property directly.

column-slice    Select group columns of W and the matching elements of inter,
                multiply.  This is the efficient implementation.

Both must produce the same result.  Any difference is a float-point
order-of-operations artefact from different summation paths in BLAS/MPS.

Split configurations tested
---------------------------
Uniform (all neurons, contiguous equal-sized groups):  [2, 3, 5, 10, 50, 100, 1000]
Masked  (active neurons only, from a real neuron mask): [2, 3, 10]

Outputs
-------
  analysis_results/split_verification/results.json
  analysis_results/split_verification/summary.csv
"""

from __future__ import annotations

import csv
import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token id.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B"

# Same 5 prompts as threshold_study for consistency
PROMPTS: list[str] = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
]

# Uniform split sizes: 2 through 1000 groups of contiguous neurons
N_SPLITS: list[int] = [2, 3, 5, 10, 50, 100, 1000]

# Masked test only needs a subset (active-neuron groups scale differently)
MASKED_N_SPLITS: list[int] = [2, 3, 10]

# Threshold for "active neuron" in the masked test.
# Using 0.01 (same as adversarial_study) to get meaningful sparsity.
# provider.py uses 1e-6 which yields ~100% density; we compute the mask
# directly from captured activations here to demonstrate a sparse case.
MASK_THRESHOLD: float = 0.01

OUTPUT_DIR = Path("analysis_results/split_verification")


# ─────────────────────────────────────────────────────────────────────────────
# Capture: single forward pass → per-layer intermediate + MLP output
# ─────────────────────────────────────────────────────────────────────────────

def capture_prompt(model, tokenizer, prompt: str) -> dict:
    """
    Run a single forward pass and record, for every MLP layer:
      intermediate  — the gate * up activation, stored float32 on CPU
      mlp_output    — the down_proj result, stored float32 on CPU

    Float32 storage avoids dtype noise when computing differences later.
    The model itself still runs in float16 internally.

    Returns
    -------
    {
      'predicted_token': int,
      'layers': {
        layer_idx: {
          'intermediate': Tensor[1, seq, I]  float32 CPU,
          'mlp_output':   Tensor[1, seq, H]  float32 CPU,
        }
      }
    }
    """
    device = next(model.parameters()).device
    layer_data: dict[int, dict] = {}

    def make_hook(li: int):
        def hook_fn(module, inp, out):
            x = inp[0]
            with torch.no_grad():
                gate = module.act_fn(module.gate_proj(x))
                up   = module.up_proj(x)
                inter = (gate * up).detach().float().cpu()
            layer_data[li] = {
                "intermediate": inter,
                "mlp_output":   out.detach().float().cpu(),
            }
            return out
        return hook_fn

    hooks = [
        model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        for i in range(len(model.model.layers))
    ]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()

    return {
        "predicted_token": int(out.logits[0, -1, :].argmax().item()),
        "layers": layer_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Split utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_equal_splits(n: int, total: int) -> list[tuple[int, int]]:
    """Partition [0, total) into n contiguous ranges of equal (±1) size."""
    base, rem = divmod(total, n)
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        groups.append((start, start + size))
        start += size
    return groups


def split_index_list(n: int, indices: list[int]) -> list[list[int]]:
    """Partition a list of indices into n approximately equal sub-lists."""
    m = len(indices)
    if m == 0:
        return [[] for _ in range(n)]
    base, rem = divmod(m, n)
    groups: list[list[int]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        groups.append(indices[start: start + size])
        start += size
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Core split computation  (float32 throughout — cast W and inter before matmul)
# ─────────────────────────────────────────────────────────────────────────────
#
# All four functions cast the weight matrix to float32 on device before any
# multiplication.  The intermediate activation is already float32 (captured
# that way in the hook).  Accumulation is also float32.  This eliminates
# float16 rounding as the source of error so that the only remaining
# differences between the split sum and the full output come from floating-
# point summation order — which should be at most a few ULPs (< 1e-5).

def zero_multiply_sum(
    inter: torch.Tensor,           # [1, seq, I] float32 CPU
    groups: list[tuple[int, int]], # contiguous index ranges
    W: torch.Tensor,               # [H, I] float16 on device
    bias: Optional[torch.Tensor],  # [H] float16 on device | None
    device,
) -> torch.Tensor:
    """
    Zero-and-multiply: for each group, slice the group columns of W (f16),
    cast that slice to float32, multiply by the matching inter elements,
    accumulate into a float32 running sum, then discard the float32 tensors.

    This directly instantiates the split-verification identity:
        sum_g  W @ (mask_g ⊙ inter)  =  W @ inter
    """
    seq = inter.shape[1]
    H   = W.shape[0]

    inter_dev = inter.to(device=device, dtype=torch.float16)  # keep f16 on device
    total = torch.zeros(1, seq, H, dtype=torch.float32)       # accumulator on CPU

    for start, end in groups:
        gi_f32 = inter_dev[0:1, :, start:end].float()         # [1, seq, g] f32
        gw_f32 = W[:, start:end].float()                      # [H, g] f32
        total += (gi_f32 @ gw_f32.T).cpu()
        del gi_f32, gw_f32

    if bias is not None:
        total += bias.float().cpu()
    return total  # [1, seq, H] float32 CPU


def column_slice_sum(
    inter: torch.Tensor,
    groups: list[tuple[int, int]],
    W: torch.Tensor,
    bias: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """
    Column-slice: for each group, slice the group's columns from W in f16,
    cast the slice (and matching inter elements) to float32, then multiply.

    Efficient implementation of the same identity:
        sum_g  inter[:, g]  @  W[:, g].T  =  inter  @  W.T
    """
    seq = inter.shape[1]
    H   = W.shape[0]

    inter_dev = inter.to(device=device, dtype=torch.float16)  # keep f16 on device
    total = torch.zeros(1, seq, H, dtype=torch.float32)

    for start, end in groups:
        gi = inter_dev[0:1, :, start:end].float()  # [1, seq, group_size] f32
        gw = W[:, start:end].float()               # [H, group_size] f32
        total += (gi @ gw.T).cpu()
        del gi, gw

    if bias is not None:
        total += bias.float().cpu()
    return total


def zero_multiply_sum_indexed(
    inter: torch.Tensor,
    index_groups: list[list[int]],
    W: torch.Tensor,
    bias: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """Zero-and-multiply for arbitrary (non-contiguous) index groups, float32."""
    seq = inter.shape[1]
    H   = W.shape[0]

    inter_dev = inter.to(device=device, dtype=torch.float16)  # keep f16 on device
    total = torch.zeros(1, seq, H, dtype=torch.float32)

    for idx_list in index_groups:
        if not idx_list:
            continue
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        gi_f32 = inter_dev[0:1, :, idx].float()  # [1, seq, g] f32
        gw_f32 = W[:, idx].float()               # [H, g] f32
        total += (gi_f32 @ gw_f32.T).cpu()
        del gi_f32, gw_f32

    if bias is not None:
        total += bias.float().cpu()
    return total


def column_slice_sum_indexed(
    inter: torch.Tensor,
    index_groups: list[list[int]],
    W: torch.Tensor,
    bias: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """Column-slice for arbitrary index groups, float32."""
    seq = inter.shape[1]
    H   = W.shape[0]

    inter_dev = inter.to(device=device, dtype=torch.float16)  # keep f16 on device
    total = torch.zeros(1, seq, H, dtype=torch.float32)

    for idx_list in index_groups:
        if not idx_list:
            continue
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        gi = inter_dev[0:1, :, idx].float()  # [1, seq, g] f32
        gw = W[:, idx].float()               # [H, g] f32
        total += (gi @ gw.T).cpu()
        del gi, gw

    if bias is not None:
        total += bias.float().cpu()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(full: torch.Tensor, approx: torch.Tensor) -> dict:
    """
    Compare two [1, seq, H] float32 tensors element-wise.

    Returns max_abs_diff, rel_error, cos_sim.
    """
    diff     = (full - approx).abs()
    max_abs  = float(diff.max().item())
    full_max = float(full.abs().max().item())
    rel_err  = max_abs / full_max if full_max > 0 else 0.0
    cos      = float(
        F.cosine_similarity(full.flatten().unsqueeze(0),
                            approx.flatten().unsqueeze(0)).item()
    )
    cos = max(-1.0, min(1.0, cos))
    return {"max_abs_diff": max_abs, "rel_error": rel_err, "cos_sim": cos}


# ─────────────────────────────────────────────────────────────────────────────
# Token injection test
# ─────────────────────────────────────────────────────────────────────────────

def run_injection_pass(
    model,
    tokenizer,
    prompt: str,
    partial_sums: dict,   # layer_idx → [1, seq, H] float32 CPU
) -> int:
    """
    Run a forward pass with every MLP's output replaced by a precomputed
    partial sum.  If the split identity holds, the predicted token is unchanged.
    """
    device = next(model.parameters()).device

    def make_hook(ps: torch.Tensor):
        def fn(module, inp, out):
            # Cast back to model dtype on device for the residual stream
            return ps.to(dtype=out.dtype, device=out.device)
        return fn

    hooks = [
        model.model.layers[li].mlp.register_forward_hook(make_hook(ps))
        for li, ps in partial_sums.items()
    ]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()

    return int(out.logits[0, -1, :].argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# Uniform split test (all neurons, contiguous groups)
# ─────────────────────────────────────────────────────────────────────────────

def run_uniform_split_tests(
    model,
    tokenizer,
    all_captures: list[dict],
) -> list[dict]:
    """
    For each (n_splits, prompt, layer) triple:
      - compute zero-multiply-sum and column-slice-sum
      - compare each against the captured full MLP output
      - compare the two approaches against each other

    Iterates layers in the outer loop to amortize weight transfers.

    Returns a flat list of records (one per triple), each carrying
    zm, cs, and zm_vs_cs metric dicts, plus token_match_rate for its n_splits.
    """
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size

    # Pre-allocate injection sum storage: injection_sums[n][prompt_idx][layer_idx]
    injection_sums: dict[int, list[dict[int, torch.Tensor]]] = {
        n: [{} for _ in all_captures]
        for n in N_SPLITS
    }

    records: list[dict] = []

    print(f"\n  Phase 1/2: computing split metrics ({num_layers} layers × "
          f"{len(all_captures)} prompts × {len(N_SPLITS)} split configs)")

    for li in range(num_layers):
        down_proj = model.model.layers[li].mlp.down_proj
        W    = down_proj.weight           # float16 on device
        bias = down_proj.bias             # None for Qwen2.5-7B

        for pi, cap in enumerate(all_captures):
            inter    = cap["layers"][li]["intermediate"]  # [1, seq, I] f32 CPU
            # Recompute reference in float32 by summing column slices of W (f16→f32)
            # This avoids materialising a full float32 copy of W (136 MB → 272 MB).
            full_ref = column_slice_sum(
                inter, make_equal_splits(1, I), W, bias, device
            )  # single "group" = all columns; same float32 path as the split functions

            for n in N_SPLITS:
                groups = make_equal_splits(n, I)

                zm = zero_multiply_sum(inter, groups, W, bias, device)
                cs = column_slice_sum(inter, groups, W, bias, device)

                records.append({
                    "n_splits":    n,
                    "prompt_idx":  pi,
                    "layer_idx":   li,
                    "zero_multiply": compute_metrics(full_ref, zm),
                    "column_slice":  compute_metrics(full_ref, cs),
                    "zm_vs_cs":      compute_metrics(zm, cs),
                    # token_match_rate filled in phase 2
                })

                injection_sums[n][pi][li] = zm   # zm used for injection

        if (li + 1) % 7 == 0 or li == num_layers - 1:
            print(f"    layer {li + 1:>2}/{num_layers} done")

    # Phase 2: token injection tests — one forward pass per (n, prompt)
    print(f"\n  Phase 2/2: token injection tests "
          f"({len(N_SPLITS)} splits × {len(all_captures)} prompts = "
          f"{len(N_SPLITS) * len(all_captures)} passes)")

    token_match_by_n: dict[int, float] = {}
    for n in N_SPLITS:
        matches = 0
        for pi, (prompt, cap) in enumerate(zip(PROMPTS, all_captures)):
            inj_token = run_injection_pass(
                model, tokenizer, prompt, injection_sums[n][pi]
            )
            if inj_token == cap["predicted_token"]:
                matches += 1
        rate = matches / len(all_captures)
        token_match_by_n[n] = rate
        print(f"    n={n:>4}: {matches}/{len(all_captures)} token matches  "
              f"({rate:.0%})")

    for rec in records:
        rec["token_match_rate"] = token_match_by_n[rec["n_splits"]]

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Masked split test (active neurons only)
# ─────────────────────────────────────────────────────────────────────────────

def run_masked_split_tests(
    model,
    tokenizer,
    all_captures: list[dict],
) -> list[dict]:
    """
    Build a neuron mask from prompt 0's captured activations at MASK_THRESHOLD.
    For each layer, identify active neurons and split only those into groups.

    The reference output is down_proj applied to the FULLY MASKED intermediate
    (all non-active neurons zeroed).  The split sum should reproduce this exactly.

    This mirrors the real verification scenario: validators are assigned disjoint
    subsets of the claimed active neurons, each computes its partial contribution,
    and the sum is compared to the verifier's own masked replay.
    """
    device    = next(model.parameters()).device
    num_layers = len(model.model.layers)

    # Build per-layer active-neuron index lists from prompt 0's intermediates
    print(f"\n  Building neuron mask from prompt 0  (threshold={MASK_THRESHOLD})")
    active_per_layer: dict[int, list[int]] = {}
    for li in range(num_layers):
        inter    = all_captures[0]["layers"][li]["intermediate"]  # [1, seq, I] f32 CPU
        max_abs  = inter.abs().amax(dim=(0, 1))                   # [I]
        active   = max_abs.gt(MASK_THRESHOLD).nonzero(as_tuple=False).squeeze(-1).tolist()
        active_per_layer[li] = active

    total_active  = sum(len(v) for v in active_per_layer.values())
    total_neurons = num_layers * model.config.intermediate_size
    density       = 100.0 * total_active / total_neurons
    print(f"  Mask density: {density:.1f}%  ({total_active:,} / {total_neurons:,} neurons)")
    print(f"  Running masked split tests on {len(all_captures)} prompts, "
          f"n_splits={MASKED_N_SPLITS}")

    records: list[dict] = []

    for li in range(num_layers):
        down_proj  = model.model.layers[li].mlp.down_proj
        W    = down_proj.weight
        bias = down_proj.bias
        active_idx = active_per_layer[li]

        # Pre-compute the reference masked output on device (reused for each prompt)
        if active_idx:
            idx_t = torch.tensor(active_idx, dtype=torch.long, device=device)

        for pi, cap in enumerate(all_captures):
            inter    = cap["layers"][li]["intermediate"]  # [1, seq, I] f32 CPU
            # Reference: apply full mask → down_proj (float32 via column slices)
            if active_idx:
                inter_dev = inter.to(device=device, dtype=torch.float16)
                masked_inter = torch.zeros(1, inter.shape[1], inter.shape[2],
                                           dtype=torch.float16, device=device)
                masked_inter[0, :, idx_t] = inter_dev[0, :, idx_t]
                # Use column_slice_sum with one group spanning all columns so W
                # is never materialised as a full float32 matrix
                masked_inter_cpu = masked_inter.float().cpu()
                del masked_inter, inter_dev
                ref = column_slice_sum(
                    masked_inter_cpu, make_equal_splits(1, inter.shape[2]),
                    W, bias, device
                )
            else:
                seq = inter.shape[1]
                H   = W.shape[0]
                ref = torch.zeros(1, seq, H, dtype=torch.float32)

            for n in MASKED_N_SPLITS:
                if n > max(1, len(active_idx)):
                    continue
                groups = split_index_list(n, active_idx)

                zm = zero_multiply_sum_indexed(inter, groups, W, bias, device)
                cs = column_slice_sum_indexed(inter, groups, W, bias, device)

                records.append({
                    "n_splits":        n,
                    "prompt_idx":      pi,
                    "layer_idx":       li,
                    "n_active_neurons": len(active_idx),
                    "zero_multiply":   compute_metrics(ref, zm),
                    "column_slice":    compute_metrics(ref, cs),
                    "zm_vs_cs":        compute_metrics(zm, cs),
                })

        if (li + 1) % 7 == 0 or li == num_layers - 1:
            print(f"    layer {li + 1:>2}/{num_layers} done")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict], approach: str) -> list[dict]:
    """Aggregate metrics for `approach` across all (prompt, layer) pairs, grouped by n_splits."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_n[r["n_splits"]].append(r[approach])

    rows = []
    for n in sorted(by_n):
        ms = by_n[n]
        rows.append({
            "n_splits":          n,
            "approach":          approach,
            "avg_max_abs_diff":  sum(m["max_abs_diff"] for m in ms) / len(ms),
            "worst_max_abs_diff": max(m["max_abs_diff"] for m in ms),
            "avg_cos_sim":       sum(m["cos_sim"] for m in ms) / len(ms),
            "worst_cos_sim":     min(m["cos_sim"] for m in ms),
            "avg_rel_error":     sum(m["rel_error"] for m in ms) / len(ms),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(records: list[dict], title: str, with_token_match: bool) -> None:
    if not records:
        return
    print()
    print(f"  {'═' * 72}")
    print(f"  {title}")
    print(f"  {'═' * 72}")

    # One row per n_splits: show zm, cs, zm_vs_cs side by side
    by_n: dict[int, dict] = {}
    token_match: dict[int, float] = {}
    for r in records:
        n = r["n_splits"]
        if n not in by_n:
            by_n[n] = {"zero_multiply": [], "column_slice": [], "zm_vs_cs": []}
        for ap in ("zero_multiply", "column_slice", "zm_vs_cs"):
            by_n[n][ap].append(r[ap])
        if with_token_match:
            token_match[n] = r.get("token_match_rate", 0.0)

    def worst(ms, key): return max(m[key] for m in ms)
    def best_cos(ms):   return min(m["cos_sim"] for m in ms)

    print()
    print(f"  {'n':>6}  {'approach':>13}  {'worst_abs_diff':>16}  "
          f"{'avg_abs_diff':>14}  {'worst_cos':>11}  {'token_match':>12}")
    print(f"  {'-' * 78}")

    for n in sorted(by_n):
        first = True
        for ap in ("zero_multiply", "column_slice", "zm_vs_cs"):
            ms = by_n[n][ap]
            w_abs = worst(ms, "max_abs_diff")
            a_abs = sum(m["max_abs_diff"] for m in ms) / len(ms)
            w_cos = best_cos(ms)
            if ap == "zero_multiply" and with_token_match:
                tm = f"{token_match.get(n, 0):.0%}"
            else:
                tm = "—"
            label = f"{n:>6}" if first else f"{'':>6}"
            first = False
            print(f"  {label}  {ap:>13}  {w_abs:>16.4e}  {a_abs:>14.4e}  "
                  f"{w_cos:>11.8f}  {tm:>12}")
        print(f"  {'-' * 78}")

    print()


def print_summary_table(uniform_records: list[dict], masked_records: list[dict]) -> None:
    _print_section(
        uniform_records,
        "UNIFORM SPLIT — all neurons, contiguous equal groups",
        with_token_match=True,
    )
    _print_section(
        masked_records,
        f"MASKED SPLIT — active neurons only (threshold={MASK_THRESHOLD})",
        with_token_match=False,
    )


def save_results(
    uniform_records: list[dict],
    masked_records: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json — full per-(n, prompt, layer) data
    path_json = output_dir / "results.json"
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"uniform_splits": uniform_records, "masked_splits": masked_records},
            fh, indent=2, ensure_ascii=True,
        )
    print(f"  Saved → {path_json}  ({path_json.stat().st_size // 1024} KB)")

    # summary.csv — one row per (test_type, n_splits, approach)
    csv_rows: list[dict] = []
    for test_type, records in [("uniform", uniform_records), ("masked", masked_records)]:
        if not records:
            continue
        for approach in ("zero_multiply", "column_slice", "zm_vs_cs"):
            for row in aggregate(records, approach):
                row["test_type"] = test_type
                # Add token_match_rate for uniform/zero_multiply rows
                if test_type == "uniform" and approach == "zero_multiply":
                    n = row["n_splits"]
                    sample = next((r for r in records if r["n_splits"] == n), None)
                    row["token_match_rate"] = (
                        sample["token_match_rate"] if sample else None
                    )
                csv_rows.append(row)

    if csv_rows:
        path_csv = output_dir / "summary.csv"
        fieldnames = list(csv_rows[0].keys())
        with open(path_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  Saved → {path_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         FFN SPLIT VERIFICATION TEST                         ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Hypothesis: sum of partial down_proj outputs = full output")
    print(f"  Model:      {MODEL_ID}")
    print(f"  Prompts:    {len(PROMPTS)}")
    print(f"  N_splits:   {N_SPLITS}")
    print(f"  Masked n:   {MASKED_N_SPLITS}  (threshold={MASK_THRESHOLD})")
    print()

    registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
    t0 = time.perf_counter()
    model, tokenizer = registry.load_verified_model(MODEL_ID)
    print(f"\n  Model loaded in {time.perf_counter() - t0:.1f}s")

    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    H = model.config.hidden_size
    print(f"  Device: {device}  |  dtype: {model.dtype}")
    print(f"  Layers: {num_layers}  |  intermediate_size: {I}  |  hidden_size: {H}")

    # ── Phase 1: Capture activations ──────────────────────────────────────────
    print(f"\n  Capturing MLP activations for {len(PROMPTS)} prompts…")
    all_captures: list[dict] = []
    for pi, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        cap = capture_prompt(model, tokenizer, prompt)
        seq_len = cap["layers"][0]["intermediate"].shape[1]
        print(f"    [{pi+1}/{len(PROMPTS)}]  seq={seq_len:>3}  "
              f"predicted_token={cap['predicted_token']:>6}  "
              f"({time.perf_counter()-t0:.1f}s)  "
              f"{prompt[:50]!r}")
        all_captures.append(cap)

    # ── Phase 2: Uniform split tests ──────────────────────────────────────────
    t0 = time.perf_counter()
    uniform_records = run_uniform_split_tests(model, tokenizer, all_captures)
    print(f"\n  Uniform split tests complete  ({time.perf_counter()-t0:.1f}s)")

    # ── Phase 3: Masked split tests ───────────────────────────────────────────
    t0 = time.perf_counter()
    masked_records = run_masked_split_tests(model, tokenizer, all_captures)
    print(f"\n  Masked split tests complete  ({time.perf_counter()-t0:.1f}s)")

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary_table(uniform_records, masked_records)
    save_results(uniform_records, masked_records, OUTPUT_DIR)

    print()
    print("  Done.")


if __name__ == "__main__":
    main()
