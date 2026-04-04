#!/usr/bin/env python3
"""
Cross-Hardware Comparison Script
=================================
Compare split verification results from two different GPU runs to assess
cross-hardware consistency of ReluLLaMA-70B computations.

Usage:
    python3 tools/compare_cross_hardware.py results_a100.json results_4090.json

What it checks:
  1. Zero classification: are the same neurons exactly zero on both GPUs?
  2. Split partial results: are cosine similarities consistent across GPUs?
  3. Overall verdict: is cross-hardware variation within acceptable bounds?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_index(result: dict) -> dict:
    """Build {(prompt_idx, layer_idx, token_step) -> entry_dict} index."""
    idx: dict = {}
    for pdata in result["prompts"]:
        pidx = pdata["prompt_id"]
        for entry in pdata["layers"]:
            key = (pidx, entry["layer"], entry.get("token_step", -1))
            idx[key] = entry
    return idx


def compare(path_a: str, path_b: str) -> None:
    ra = load_results(path_a)
    rb = load_results(path_b)

    print("=" * 65)
    print("=== Cross-Hardware Comparison ===")
    print("=" * 65)
    print(f"\nFile A: {Path(path_a).name}")
    print(f"  GPU: {ra.get('gpu_name', 'unknown')}  |  timestamp: {ra.get('timestamp', '?')}")
    print(f"File B: {Path(path_b).name}")
    print(f"  GPU: {rb.get('gpu_name', 'unknown')}  |  timestamp: {rb.get('timestamp', '?')}")

    idx_a = build_index(ra)
    idx_b = build_index(rb)

    common = sorted(set(idx_a) & set(idx_b))
    print(f"\nCommon (prompt, layer, token_step) entries: {len(common)}")
    if not common:
        print("  No common entries found — check that prompts match between files.")
        return

    # Accumulators
    zero_count_diffs:    list[int]   = []
    zero_fraction_diffs: list[float] = []
    cosine_diffs:  dict[str, list[float]] = {}
    maxdiff_diffs: dict[str, list[float]] = {}
    token_match_agree: list[bool] = []

    for key in common:
        a = idx_a[key]
        b = idx_b[key]

        # Zero count comparison
        za = a.get("relu_zero_count", 0)
        zb = b.get("relu_zero_count", 0)
        zero_count_diffs.append(abs(za - zb))

        fa = a.get("relu_zero_fraction", 0.0)
        fb = b.get("relu_zero_fraction", 0.0)
        zero_fraction_diffs.append(abs(fa - fb))

        # Split metrics comparison
        splits_a = a.get("splits", {})
        splits_b = b.get("splits", {})
        for n_str in splits_a:
            if n_str not in splits_b:
                continue
            sa = splits_a[n_str]
            sb = splits_b[n_str]
            cosine_diffs.setdefault(n_str, []).append(
                abs(sa.get("cosine_similarity", 1.0) - sb.get("cosine_similarity", 1.0))
            )
            maxdiff_diffs.setdefault(n_str, []).append(
                abs(sa.get("max_abs_diff", 0.0) - sb.get("max_abs_diff", 0.0))
            )
            # Token match agreement: do both GPUs agree on whether argmax matches?
            tm_a = sa.get("token_match", None)
            tm_b = sb.get("token_match", None)
            if tm_a is not None and tm_b is not None:
                token_match_agree.append(tm_a == tm_b)

    # ---- Zero classification ----
    print("\n--- Zero Classification (ReLU exact zeros) ---")
    exact_matches = sum(1 for d in zero_count_diffs if d == 0)
    print(f"  Entries with identical zero count: {exact_matches}/{len(zero_count_diffs)}")
    if zero_count_diffs:
        print(f"  Max count difference:  {max(zero_count_diffs)}")
        print(f"  Mean count difference: {np.mean(zero_count_diffs):.2f}")
        frac_exact = sum(1 for d in zero_fraction_diffs if d < 1e-6)
        print(f"  Fraction exact (diff < 1e-6): {frac_exact}/{len(zero_fraction_diffs)}")

    # ---- Split verification consistency ----
    print("\n--- Split Verification Cosine Similarity Difference ---")
    for n_str in sorted(cosine_diffs.keys(), key=lambda s: int(s)):
        diffs = cosine_diffs[n_str]
        print(
            f"  {int(n_str):>5} splits: "
            f"max={max(diffs):.8f}  mean={np.mean(diffs):.8f}  n={len(diffs)}"
        )

    print("\n--- Max Absolute Difference in Partial Results (cross-GPU delta) ---")
    for n_str in sorted(maxdiff_diffs.keys(), key=lambda s: int(s)):
        diffs = maxdiff_diffs[n_str]
        print(
            f"  {int(n_str):>5} splits: "
            f"max_delta={max(diffs):.8f}  mean_delta={np.mean(diffs):.8f}"
        )

    if token_match_agree:
        agree_rate = sum(token_match_agree) / len(token_match_agree)
        print(f"\n  Token match agreement across GPUs: {agree_rate * 100:.1f}%")

    # ---- Verdict ----
    print("\n--- Cross-Hardware Verdict ---")
    zero_exact_rate = exact_matches / max(len(zero_count_diffs), 1)
    print(f"  Zero classification identical: {zero_exact_rate * 100:.1f}%  "
          f"({'PASS' if zero_exact_rate > 0.99 else 'FAIL'})")

    if cosine_diffs.get("100"):
        worst = max(cosine_diffs["100"])
        consistent = worst < 1e-4
        print(f"  Cosine diff 100-splits worst: {worst:.8f}  "
              f"({'PASS' if consistent else 'FAIL'} — threshold 1e-4)")

    if cosine_diffs.get("1000"):
        worst = max(cosine_diffs["1000"])
        consistent = worst < 1e-4
        print(f"  Cosine diff 1000-splits worst: {worst:.8f}  "
              f"({'PASS' if consistent else 'FAIL'} — threshold 1e-4)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare split verification results from two different GPU runs"
    )
    parser.add_argument("file_a", help="JSON result file from first GPU")
    parser.add_argument("file_b", help="JSON result file from second GPU")
    args = parser.parse_args()
    compare(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
