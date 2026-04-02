#!/usr/bin/env python3
"""
cross_hardware_test.py
======================
Test whether the same FFN computation on two different hardware backends
(MacBook MPS vs NVIDIA CUDA) produces identical numbers.

Usage
-----
  # On each machine, capture results:
  python3 cross_hardware_test.py --capture --output hardware_results_mac.json
  python3 cross_hardware_test.py --capture --output hardware_results_cuda.json

  # Transfer both files to the same machine, then compare:
  python3 cross_hardware_test.py --compare hardware_results_mac.json hardware_results_cuda.json

Capture output
--------------
For each of 5 prompts × 3 layers (0, 14, 27), the capture file stores:
  - token_ids                 : list[int]  — tokenization must match across machines
  - predicted_token           : int
  - logits                    : list[float] float32, at the final-token position
  - mlp_input                 : list[float] float32  — residual stream entering FFN
  - mlp_output                : list[float] float32  — down_proj output
  - intermediate              : list[float] float32  — gate * up activation
  - partial_results           : list of 5 lists, one per neuron group (column-slice,
                                float32)

All tensors flattened to 1-D before JSON serialisation.
Shape metadata saved alongside so the compare side can reconstruct.

Comparison output
-----------------
Per-tensor stats between file A and file B:
  max_abs_diff, mean_abs_diff, cosine_similarity
Plus a summary table and a pass/fail verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token id.*")

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants — must be identical on both machines
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B"

# Same 5 prompts used in split_verification_test.py
PROMPTS: list[str] = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
]

# Three representative layers: early / middle / late
CAPTURE_LAYERS: list[int] = [0, 14, 27]

# 5-way equal split of 18,944 intermediate neurons
N_GROUPS = 5
INTERMEDIATE_SIZE = 18_944   # Qwen2.5-7B

# Absolute difference thresholds for pass/fail
PASS_THRESHOLD_MAX  = 1e-3   # max abs diff below this → pass
PASS_THRESHOLD_MEAN = 1e-5   # mean abs diff below this → pass


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_equal_splits(n: int, total: int) -> list[tuple[int, int]]:
    """Partition [0, total) into n contiguous equal (±1) ranges."""
    base, rem = divmod(total, n)
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        groups.append((start, start + size))
        start += size
    return groups


def column_slice_partial(
    inter_f16: torch.Tensor,   # [1, seq, I] float16 on device
    W: torch.Tensor,           # [H, I] float16 on device
    start: int,
    end: int,
) -> torch.Tensor:
    """
    Compute down_proj contribution from neurons [start:end] in float32.
    Slices W columns in f16, casts the slice to f32, multiplies.
    Returns [1, seq, H] float32 on CPU.
    """
    gi = inter_f16[0:1, :, start:end].float()   # [1, seq, g] f32
    gw = W[:, start:end].float()               # [H, g] f32
    result = (gi @ gw.T).cpu()
    del gi, gw
    return result


def tensor_to_list(t: torch.Tensor) -> tuple[list, list[int]]:
    """Return (flat list of floats, shape) for JSON storage."""
    return t.reshape(-1).tolist(), list(t.shape)


def list_to_tensor(data: list, shape: list[int]) -> torch.Tensor:
    """Reconstruct tensor from flat list and shape."""
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Capture mode
# ─────────────────────────────────────────────────────────────────────────────

def load_model(registry_path: Path):
    """Load Qwen2.5-7B through the model registry (verifies weight hash)."""
    registry = ModelRegistry(registry_path)
    print(f"Loading {MODEL_ID} via model registry …")
    model, tokenizer = registry.load_verified_model(MODEL_ID, dtype=torch.float16)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def capture_one_prompt(
    model,
    tokenizer,
    prompt: str,
    capture_layers: list[int],
    device,
) -> dict:
    """
    Run one greedy forward pass and record per-layer data for capture_layers.

    For each captured layer:
      mlp_input    — residual stream entering the FFN  [1, seq, H]  float32
      intermediate — gate * up                         [1, seq, I]  float32
      mlp_output   — down_proj output                  [1, seq, H]  float32
      partial_results — list of N_GROUPS tensors        [1, seq, H]  float32 each

    Also records:
      token_ids      — list[int]
      predicted_token — int  (argmax of final logits)
      logits         — list[float]  (full vocab logit vector at final position)
    """
    layer_captures: dict[int, dict] = {}
    groups = make_equal_splits(N_GROUPS, INTERMEDIATE_SIZE)

    def make_mlp_hook(li: int):
        def hook_fn(module, inp, out):
            x = inp[0]  # residual stream entering this MLP, float16 on device
            with torch.no_grad():
                gate  = module.act_fn(module.gate_proj(x))
                up    = module.up_proj(x)
                inter = gate * up                    # [1, seq, I] float16 on device

                # Partial results — column slice in f32, keep on CPU
                W     = module.down_proj.weight      # [H, I] float16 on device
                inter_f16 = inter                    # already float16 on device
                partials: list[list] = []
                partial_shapes: list[list[int]] = []
                for start, end in groups:
                    pr = column_slice_partial(inter_f16, W, start, end)
                    flat, shape = tensor_to_list(pr)
                    partials.append(flat)
                    partial_shapes.append(shape)
                    del pr

                layer_captures[li] = {
                    "mlp_input":        tensor_to_list(x.detach().float().cpu()),
                    "intermediate":     tensor_to_list(inter.detach().float().cpu()),
                    "mlp_output":       tensor_to_list(out.detach().float().cpu()),
                    "partial_results":  partials,
                    "partial_shapes":   partial_shapes,
                }
            return out
        return hook_fn

    hooks = [
        model.model.layers[li].mlp.register_forward_hook(make_mlp_hook(li))
        for li in capture_layers
    ]

    inputs   = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()

    with torch.no_grad():
        output = model(**inputs)

    for h in hooks:
        h.remove()

    logits_last = output.logits[0, -1, :].float().cpu()  # [vocab] float32
    predicted   = int(logits_last.argmax().item())

    return {
        "token_ids":       token_ids,
        "predicted_token": predicted,
        "logits":          logits_last.tolist(),
        "logits_shape":    [int(logits_last.shape[0])],
        "layers":          layer_captures,
    }


def run_capture(output_path: Path, registry_path: Path) -> None:
    """Capture mode: run all prompts, save results to JSON."""
    model, tokenizer, device = load_model(registry_path)

    results: dict = {
        "model_id":      MODEL_ID,
        "device":        str(device),
        "capture_layers": CAPTURE_LAYERS,
        "n_groups":      N_GROUPS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "group_ranges":  make_equal_splits(N_GROUPS, INTERMEDIATE_SIZE),
        "prompts":       PROMPTS,
        "captures":      [],
    }

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi + 1}/{len(PROMPTS)}: {prompt[:60]!r}")
        cap = capture_one_prompt(model, tokenizer, prompt, CAPTURE_LAYERS, device)
        results["captures"].append(cap)
        print(f"    predicted token: {cap['predicted_token']} "
              f"({tokenizer.decode([cap['predicted_token']])!r})")
        for li in CAPTURE_LAYERS:
            ldata = cap["layers"][li]
            inter_shape = ldata["intermediate"][1]
            print(f"    layer {li:>2}: inter shape {inter_shape}, "
                  f"{len(ldata['partial_results'])} partial groups")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving capture to {output_path} …")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done — {size_mb:.1f} MB written.")


# ─────────────────────────────────────────────────────────────────────────────
# Compare mode
# ─────────────────────────────────────────────────────────────────────────────

def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison metrics between two float32 tensors of the same shape."""
    diff   = (a - b).abs()
    cos    = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return {
        "max_abs_diff":  float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "cosine_sim":    float(cos),
        "shape":         list(a.shape),
    }


def load_capture(path: Path) -> dict:
    print(f"Loading {path} …")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"  device={data['device']!r}  prompts={len(data['captures'])}")
    return data


def verify_metadata(a: dict, b: dict) -> None:
    """Abort with a clear message if the two captures are structurally incompatible."""
    errors: list[str] = []
    if a["model_id"] != b["model_id"]:
        errors.append(f"model_id mismatch: {a['model_id']!r} vs {b['model_id']!r}")
    if a["capture_layers"] != b["capture_layers"]:
        errors.append(f"capture_layers mismatch: {a['capture_layers']} vs {b['capture_layers']}")
    if a["n_groups"] != b["n_groups"]:
        errors.append(f"n_groups mismatch: {a['n_groups']} vs {b['n_groups']}")
    if a["intermediate_size"] != b["intermediate_size"]:
        errors.append(f"intermediate_size mismatch")
    if a["prompts"] != b["prompts"]:
        errors.append("prompts differ — results are not comparable")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


FIELD_LABELS = {
    "mlp_input":   "MLP input (residual)",
    "intermediate": "intermediate (gate*up)",
    "mlp_output":  "MLP output (down_proj)",
    "logits":      "logits (final position)",
}


def run_compare(path_a: Path, path_b: Path) -> None:
    """Compare mode: load two capture files and print a detailed diff report."""
    data_a = load_capture(path_a)
    data_b = load_capture(path_b)

    verify_metadata(data_a, data_b)

    label_a = path_a.stem
    label_b = path_b.stem
    capture_layers: list[int] = data_a["capture_layers"]
    n_prompts = len(data_a["captures"])
    n_groups  = data_a["n_groups"]

    print(f"\n{'='*70}")
    print(f"Cross-hardware comparison")
    print(f"  A: {label_a}  (device={data_a['device']!r})")
    print(f"  B: {label_b}  (device={data_b['device']!r})")
    print(f"  model:   {data_a['model_id']}")
    print(f"  layers:  {capture_layers}")
    print(f"  prompts: {n_prompts}")
    print(f"{'='*70}")

    all_stats: list[dict] = []   # flat list for summary table

    token_matches = 0

    for pi in range(n_prompts):
        cap_a = data_a["captures"][pi]
        cap_b = data_b["captures"][pi]

        # ── token ID check ──────────────────────────────────────────────────
        if cap_a["token_ids"] != cap_b["token_ids"]:
            print(f"\n  [prompt {pi}] WARNING: token_ids differ — "
                  f"tokenization is inconsistent across machines!")
        else:
            tok_match = "OK"

        pred_a = cap_a["predicted_token"]
        pred_b = cap_b["predicted_token"]
        tok_match_flag = pred_a == pred_b
        if tok_match_flag:
            token_matches += 1

        print(f"\n── Prompt {pi}: {PROMPTS[pi][:55]!r}")
        print(f"   predicted token  A={pred_a}  B={pred_b}  "
              f"{'MATCH' if tok_match_flag else '*** MISMATCH ***'}")

        # ── logits ──────────────────────────────────────────────────────────
        logits_a = torch.tensor(cap_a["logits"], dtype=torch.float32)
        logits_b = torch.tensor(cap_b["logits"], dtype=torch.float32)
        stats = tensor_stats(logits_a, logits_b)
        stats["prompt"] = pi
        stats["layer"]  = "—"
        stats["field"]  = "logits"
        all_stats.append(stats)
        print(f"   logits           max_diff={stats['max_abs_diff']:.2e}  "
              f"mean_diff={stats['mean_abs_diff']:.2e}  "
              f"cos={stats['cosine_sim']:.8f}")

        # ── per-layer tensors ────────────────────────────────────────────────
        for li in capture_layers:
            la = cap_a["layers"][str(li)]   # JSON keys are strings
            lb = cap_b["layers"][str(li)]

            for field, label in FIELD_LABELS.items():
                if field == "logits":
                    continue
                ta = list_to_tensor(la[field][0], la[field][1])
                tb = list_to_tensor(lb[field][0], lb[field][1])
                if ta.shape != tb.shape:
                    print(f"   layer {li:>2} {label:30s}  SHAPE MISMATCH "
                          f"{list(ta.shape)} vs {list(tb.shape)}")
                    continue
                stats = tensor_stats(ta, tb)
                stats["prompt"] = pi
                stats["layer"]  = li
                stats["field"]  = field
                all_stats.append(stats)
                print(f"   layer {li:>2} {label:30s}  "
                      f"max={stats['max_abs_diff']:.2e}  "
                      f"mean={stats['mean_abs_diff']:.2e}  "
                      f"cos={stats['cosine_sim']:.8f}")

            # ── partial results ──────────────────────────────────────────────
            for g in range(n_groups):
                ta = list_to_tensor(la["partial_results"][g], la["partial_shapes"][g])
                tb = list_to_tensor(lb["partial_results"][g], lb["partial_shapes"][g])
                if ta.shape != tb.shape:
                    print(f"   layer {li:>2} group {g}  SHAPE MISMATCH")
                    continue
                stats = tensor_stats(ta, tb)
                stats["prompt"] = pi
                stats["layer"]  = li
                stats["field"]  = f"partial_group_{g}"
                all_stats.append(stats)
                print(f"   layer {li:>2} partial group {g:>2}              "
                      f"  max={stats['max_abs_diff']:.2e}  "
                      f"mean={stats['mean_abs_diff']:.2e}  "
                      f"cos={stats['cosine_sim']:.8f}")

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Token match rate: {token_matches}/{n_prompts} prompts "
          f"({100*token_matches/n_prompts:.0f}%)")

    if all_stats:
        max_of_maxes  = max(s["max_abs_diff"]  for s in all_stats)
        mean_of_means = sum(s["mean_abs_diff"] for s in all_stats) / len(all_stats)
        min_cos       = min(s["cosine_sim"]    for s in all_stats)

        print(f"  Worst max_abs_diff  (any tensor): {max_of_maxes:.4e}")
        print(f"  Avg   mean_abs_diff (all tensors): {mean_of_means:.4e}")
        print(f"  Min   cosine_sim    (any tensor): {min_cos:.8f}")

        # per-field breakdown
        fields_seen = dict.fromkeys(s["field"] for s in all_stats)
        print(f"\n  Per-field worst max_abs_diff:")
        for field in fields_seen:
            worst = max(s["max_abs_diff"] for s in all_stats if s["field"] == field)
            label = f"    {field:35s}"
            verdict = ("PASS" if worst < PASS_THRESHOLD_MAX else "FAIL")
            print(f"{label}  {worst:.4e}  [{verdict}]")

        # overall verdict
        overall_pass = (
            max_of_maxes < PASS_THRESHOLD_MAX
            and mean_of_means < PASS_THRESHOLD_MEAN
            and token_matches == n_prompts
        )
        print(f"\n{'='*70}")
        if overall_pass:
            print("VERDICT: PASS — hardware backends produce consistent results.")
            print("         Distributed verification across these machines should work.")
        else:
            print("VERDICT: FAIL — hardware backends diverge beyond acceptable tolerance.")
            failures = []
            if max_of_maxes >= PASS_THRESHOLD_MAX:
                failures.append(f"max_abs_diff {max_of_maxes:.4e} >= {PASS_THRESHOLD_MAX:.0e}")
            if mean_of_means >= PASS_THRESHOLD_MEAN:
                failures.append(f"mean_abs_diff {mean_of_means:.4e} >= {PASS_THRESHOLD_MEAN:.0e}")
            if token_matches < n_prompts:
                failures.append(f"token mismatch on {n_prompts - token_matches}/{n_prompts} prompts")
            for f in failures:
                print(f"         • {f}")
        print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-hardware FFN consistency test for distributed verification."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Run capture mode: load model, run prompts, save tensors to JSON.",
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two capture files produced on different hardware.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("hardware_results.json"),
        help="Output file path for --capture (default: hardware_results.json).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to registry.json (default: next to model_registry.py).",
    )
    args = parser.parse_args()

    if args.capture:
        run_capture(args.output, args.registry)
    else:
        run_compare(Path(args.compare[0]), Path(args.compare[1]))


if __name__ == "__main__":
    main()
