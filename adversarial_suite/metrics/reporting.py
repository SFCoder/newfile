"""
Reporting functions — summary tables, CSV files, and plots.

All functions are pure with respect to their inputs: they read from a
list of result dicts and write to paths or return strings.  No model,
tokenizer, or framework dependency.

Functions
---------
print_summary_table         — grouped by model + tier, one row per strategy
print_attacker_optimal      — max safe skip count per model + tier
save_results_json           — full results to JSON
save_summary_csv            — summary table rows to CSV
save_attacker_optimal_csv   — optimal strategy rows to CSV
plot_model_comparison        — savings vs model size, one line per tier
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Summary table (grouped by model + tier)
# ---------------------------------------------------------------------------


def print_summary_table(results: list) -> None:
    """
    Print a summary table grouped by model and prompt complexity tier.

    Columns: model | tier | skip_count | strategy | token_match | coherence | savings%

    For each (model, tier, skip_count, strategy) combination, the row
    shown is the one with the highest token match rate across all random
    seeds and prompts (best observed result for that configuration).
    """
    # Aggregate: keyed by (model_id, tier, skip_count, strategy_family)
    # strategy_family merges "random_seed_0/1/2" into "random"
    agg: dict = defaultdict(list)
    for r in results:
        fam = _strategy_family(r["strategy"])
        key = (r["model_id"], r["tier"], r["skip_count"], fam)
        agg[key].append(r)

    # Print header
    col = [14, 10, 6, 16, 10, 10, 9]
    header = _row(
        ["Model", "Tier", "Skip", "Strategy", "MatchRate", "Coherence", "Savings%"],
        col,
    )
    sep = "-" * sum(col + [3 * len(col)])
    print()
    print("  SUMMARY TABLE — token match rate and attacker savings by configuration")
    print("  " + sep)
    print("  " + header)
    print("  " + sep)

    prev_model = None
    for key in sorted(agg.keys()):
        model_id, tier, skip_count, strategy = key
        rows = agg[key]
        # Best match rate across prompts / seeds
        best = max(rows, key=lambda x: x["token_match_rate"])
        match_rate = best["token_match_rate"]
        coherence = best["coherence"]
        savings = best["savings_pct"]
        short_model = model_id.split("/")[-1]
        if model_id != prev_model:
            print()
            prev_model = model_id
        print(
            "  "
            + _row(
                [
                    short_model,
                    tier,
                    str(skip_count),
                    strategy,
                    f"{match_rate:.1%}",
                    coherence,
                    f"{savings:.1f}%",
                ],
                col,
            )
        )
    print("  " + sep)
    print()


# ---------------------------------------------------------------------------
# Attacker optimal summary
# ---------------------------------------------------------------------------


def print_attacker_optimal(results: list) -> None:
    """
    For each (model, tier), find the maximum skip count at which ANY
    strategy achieves ≥ 80% token match rate (coherent threshold).

    Prints a table showing: model | tier | max_safe_skip | strategy | savings%
    """
    # Group by (model_id, tier)
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["model_id"], r["tier"])].append(r)

    col = [14, 10, 10, 18, 9]
    header = _row(
        ["Model", "Tier", "MaxSkip", "BestStrategy", "Savings%"],
        col,
    )
    sep = "-" * sum(col + [3 * len(col)])

    print()
    print("  ATTACKER OPTIMAL STRATEGY — max safe skip count per model + complexity tier")
    print("  " + sep)
    print("  " + header)
    print("  " + sep)

    prev_model = None
    for key in sorted(groups.keys()):
        model_id, tier = key
        rows = groups[key]
        coherent_rows = [r for r in rows if r["token_match_rate"] >= 0.80]
        short_model = model_id.split("/")[-1]
        if model_id != prev_model:
            print()
            prev_model = model_id
        if not coherent_rows:
            print(
                "  "
                + _row(
                    [short_model, tier, "0", "(none — all fail)", "0.0%"],
                    col,
                )
            )
            continue
        best = max(coherent_rows, key=lambda x: (x["skip_count"], x["token_match_rate"]))
        print(
            "  "
            + _row(
                [
                    short_model,
                    tier,
                    str(best["skip_count"]),
                    _strategy_family(best["strategy"]),
                    f"{best['savings_pct']:.1f}%",
                ],
                col,
            )
        )
    print("  " + sep)
    print()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_results_json(results: list, path: Path) -> None:
    """Write the full results list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"results": results},
            fh,
            indent=2,
            default=_json_default,
        )
    print(f"  Saved full results → {path}")


def save_summary_csv(results: list, path: Path) -> None:
    """
    Write a summary CSV with one row per (model, tier, skip_count,
    strategy_family), showing the best token match rate observed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    agg: dict = defaultdict(list)
    for r in results:
        fam = _strategy_family(r["strategy"])
        key = (r["model_id"], r["tier"], r["skip_count"], fam)
        agg[key].append(r)

    fieldnames = [
        "model_id", "tier", "skip_count", "strategy",
        "token_match_rate", "coherence", "savings_pct",
        "mean_cosine_similarity", "perplexity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(agg.keys()):
            model_id, tier, skip_count, strategy = key
            rows = agg[key]
            best = max(rows, key=lambda x: x["token_match_rate"])
            writer.writerow({
                "model_id": model_id,
                "tier": tier,
                "skip_count": skip_count,
                "strategy": strategy,
                "token_match_rate": round(best["token_match_rate"], 4),
                "coherence": best["coherence"],
                "savings_pct": round(best["savings_pct"], 2),
                "mean_cosine_similarity": _round_or_none(best.get("mean_cosine_similarity"), 4),
                "perplexity": _round_or_none(best.get("perplexity"), 2),
            })
    print(f"  Saved summary CSV  → {path}")


def save_attacker_optimal_csv(results: list, path: Path) -> None:
    """
    Write a CSV with one row per (model, tier), showing the max safe
    skip count and corresponding savings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["model_id"], r["tier"])].append(r)

    fieldnames = [
        "model_id", "tier", "max_safe_skip_count",
        "best_strategy", "savings_pct", "token_match_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups.keys()):
            model_id, tier = key
            rows = groups[key]
            coherent_rows = [r for r in rows if r["token_match_rate"] >= 0.80]
            if not coherent_rows:
                writer.writerow({
                    "model_id": model_id,
                    "tier": tier,
                    "max_safe_skip_count": 0,
                    "best_strategy": "none",
                    "savings_pct": 0.0,
                    "token_match_rate": 0.0,
                })
                continue
            best = max(
                coherent_rows,
                key=lambda x: (x["skip_count"], x["token_match_rate"]),
            )
            writer.writerow({
                "model_id": model_id,
                "tier": tier,
                "max_safe_skip_count": best["skip_count"],
                "best_strategy": _strategy_family(best["strategy"]),
                "savings_pct": round(best["savings_pct"], 2),
                "token_match_rate": round(best["token_match_rate"], 4),
            })
    print(f"  Saved optimal CSV  → {path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_model_comparison(results: list, path: Path) -> None:
    """
    Plot max-safe attacker savings vs model size (hidden_size as proxy),
    with one line per prompt complexity tier.

    X-axis: model hidden_size (numeric proxy for model size).
    Y-axis: maximum safe attacker savings percentage (coherent threshold).

    Saves the plot to *path*.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping comparison plot.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect max safe savings per (model, tier)
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["model_id"], r["tier"])].append(r)

    # Also need model size; extract from metadata if present
    model_sizes: dict = {}
    for r in results:
        model_id = r["model_id"]
        if model_id not in model_sizes and "hidden_size" in r.get("metadata", {}):
            model_sizes[model_id] = r["metadata"]["hidden_size"]

    # Fallback size ordering from name
    def _size_key(model_id: str) -> int:
        return model_sizes.get(model_id, _infer_size_from_name(model_id))

    tiers = ["simple", "moderate", "complex"]
    tier_colors = {"simple": "tab:blue", "moderate": "tab:orange", "complex": "tab:red"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for tier in tiers:
        points = []
        for (model_id, t), rows in groups.items():
            if t != tier:
                continue
            coherent = [r for r in rows if r["token_match_rate"] >= 0.80]
            if not coherent:
                max_savings = 0.0
            else:
                max_savings = max(r["savings_pct"] for r in coherent)
            size = _size_key(model_id)
            label = model_id.split("/")[-1]
            points.append((size, max_savings, label))
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        labels = [p[2] for p in points]
        ax.plot(xs, ys, marker="o", label=tier, color=tier_colors.get(tier))
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Model hidden size (proxy for model size)", fontsize=11)
    ax.set_ylabel("Max safe attacker savings (%)", fontsize=11)
    ax.set_title(
        "Maximum safe attention-skip savings by model size and prompt complexity",
        fontsize=11,
    )
    ax.legend(title="Prompt complexity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison plot → {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strategy_family(strategy: str) -> str:
    """Collapse random_seed_0/1/2 variants into a single 'random' label."""
    if strategy.startswith("random"):
        return "random"
    return strategy


def _row(values: list, widths: list) -> str:
    """Format a row of values padded to given column widths."""
    return "  ".join(str(v).ljust(w)[:w] for v, w in zip(values, widths))


def _round_or_none(value, decimals: int):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), decimals)


def _json_default(obj):
    """JSON serialisation fallback for non-standard types."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


def _infer_size_from_name(model_id: str) -> int:
    """Rough hidden_size proxy from model name for sorting purposes."""
    name = model_id.lower()
    if "72b" in name:
        return 8192
    if "7b" in name:
        return 3584
    if "3b" in name:
        return 2048
    if "0.5b" in name or "500m" in name:
        return 896
    return 1
