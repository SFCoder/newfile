#!/usr/bin/env python3
"""
migrate.py — one-time migration of existing JSON experiment results into results.db.

Idempotent: running twice does not create duplicate records.
Each JSON file is imported under an experiment whose notes field contains
a migration marker ("migrated_from:<filename>"); if that experiment row
already exists the file is skipped.

Source files
------------
analysis_results/threshold_study/results.json   → attack_type="threshold_sweep"
analysis_results/adversarial_study/results.json → attack_type derived from scenario
analysis_results/sparse_replay/results.json     → attack_type="honest" (sparse masks)
analysis_results/max_savings/results.json       → attack_type="attention_skip"

Scenario → attack_type mapping for adversarial_study
-----------------------------------------------------
  random_masks_honest  → attack_type="honest"
  random_masks_fraud   → attack_type="random_mask"
  wrong_model_*        → attack_type="model_substitution"
  token_swap_*         → attack_type="token_tamper"

Usage
-----
    python3 migrate.py              # import all available JSON files
    python3 migrate.py --dry-run    # count rows without writing
    python3 migrate.py --db /path/to/results.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from adversarial_suite.db.schema import DEFAULT_DB_PATH, init_db  # noqa: E402
from adversarial_suite.db.writer import ResultsWriter               # noqa: E402

ANALYSIS_DIR = _ROOT / "analysis_results"

# Complexity tier lookup for known prompts used across experiments
_KNOWN_COMPLEXITY: dict = {
    "The capital of France is": "simple",
    "Water boils at 100 degrees Celsius": "simple",
    "The color of the sky": "simple",
    "Explain how a car engine": "moderate",
    "Describe the process by which plants": "moderate",
    "Explain why the moon": "moderate",
}


def _infer_complexity(prompt: str) -> Optional[str]:
    for prefix, tier in _KNOWN_COMPLEXITY.items():
        if prompt.startswith(prefix):
            return tier
    return None


def _already_migrated(conn, marker: str) -> bool:
    """Return True if an experiment with this migration marker exists."""
    row = conn.execute(
        "SELECT id FROM experiments WHERE notes LIKE ?",
        (f"%migrated_from:{marker}%",),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Per-file migration functions
# ---------------------------------------------------------------------------


def migrate_threshold_study(writer: ResultsWriter, data: list, dry_run: bool) -> int:
    """
    Import threshold_study results.

    Each record is one (model, threshold, prompt) triple with per-position
    detail.  We write:
      - one aggregate result row per triple (layer=null, position=null)
      - one result row per position (layer=null, position=<i>)

    attack_type = "threshold_sweep"
    attack_params = {"threshold": <value>}
    """
    n = 0
    for rec in data:
        model_id = writer.ensure_model(rec["model"])
        prompt_id = writer.ensure_prompt(
            rec["prompt"], complexity=_infer_complexity(rec["prompt"])
        )
        agg = rec.get("aggregates", {})
        ap = {"threshold": rec["threshold"]}

        if not dry_run:
            # Aggregate row
            writer.add_result(
                model_id=model_id,
                prompt_id=prompt_id,
                attack_type="threshold_sweep",
                attack_params=ap,
                token_match_rate=agg.get("pass_rate", 0) / 100.0
                    if agg.get("pass_rate") is not None else None,
                cosine_similarity=agg.get("avg_cosine_sim"),
                rank=agg.get("avg_rank"),
                compression_pct=agg.get("compression_pct"),
                pass_fail=agg.get("pass_rate") == 100.0,
                verification_target="local",
                raw_data={
                    "min_cosine_sim": agg.get("min_cosine_sim"),
                    "max_rank": agg.get("max_rank"),
                    "first_fail_position": agg.get("first_fail_position"),
                    "output_token_ids": rec.get("output_token_ids"),
                },
            )
            n += 1

            # Per-position rows
            for pos_rec in rec.get("positions", []):
                writer.add_result(
                    model_id=model_id,
                    prompt_id=prompt_id,
                    attack_type="threshold_sweep",
                    attack_params=ap,
                    position=pos_rec["position"],
                    cosine_similarity=pos_rec.get("cosine_sim"),
                    rank=pos_rec.get("rank"),
                    pass_fail=pos_rec.get("passed"),
                    verification_target="local",
                    raw_data={
                        "correct_token_id": pos_rec.get("correct_token_id"),
                        "logit_diff_magnitude": pos_rec.get("logit_diff_magnitude"),
                    },
                )
                n += 1
        else:
            n += 1 + len(rec.get("positions", []))

    return n


def migrate_adversarial_study(writer: ResultsWriter, data: list, dry_run: bool) -> int:
    """
    Import adversarial_study results.

    scenario → attack_type mapping:
      random_masks_honest  → "honest"
      random_masks_fraud   → "random_mask"
      wrong_model_*        → "model_substitution"
      token_swap_*         → "token_tamper"
    """
    def _attack_type(scenario: str, is_honest: bool) -> str:
        if is_honest:
            return "honest"
        s = scenario.lower()
        if "wrong_model" in s or "substitute" in s:
            return "model_substitution"
        if "token_swap" in s or "token_tamper" in s:
            return "token_tamper"
        if "random_mask" in s or "rand_mask" in s:
            return "random_mask"
        return "honest" if is_honest else "random_mask"

    n = 0
    for rec in data:
        scenario = rec.get("scenario", "")
        is_honest = rec.get("is_honest", True)
        attacker_model = rec.get("attacker_model", "Qwen/Qwen2.5-7B")
        verifier_model = rec.get("verifier_model", "Qwen/Qwen2.5-7B")

        model_id = writer.ensure_model(attacker_model)
        prompt_id = writer.ensure_prompt(
            rec["prompt"], complexity=_infer_complexity(rec["prompt"])
        )
        agg = rec.get("aggregates", {})
        at = _attack_type(scenario, is_honest)
        ap: dict = {"mask_threshold": rec.get("mask_threshold")}
        if at == "model_substitution":
            ap["attacker_model"] = attacker_model
            ap["verifier_model"] = verifier_model

        if not dry_run:
            writer.add_result(
                model_id=model_id,
                prompt_id=prompt_id,
                attack_type=at,
                attack_params=ap,
                cosine_similarity=agg.get("avg_cos"),
                rank=agg.get("avg_rank"),
                pass_fail=is_honest,
                verification_target="local",
                raw_data={
                    "scenario": scenario,
                    "n_positions": agg.get("n_positions"),
                    "worst_cos": agg.get("worst_cos"),
                    "best_cos": agg.get("best_cos"),
                    "pass_99": agg.get("pass_99"),
                    "pass_98": agg.get("pass_98"),
                    "pass_95": agg.get("pass_95"),
                    "pass_90": agg.get("pass_90"),
                    "pass_85": agg.get("pass_85"),
                    "pass_80": agg.get("pass_80"),
                    "claimed_token_ids": rec.get("claimed_token_ids"),
                    "verifier_model": verifier_model,
                },
            )
            n += 1
        else:
            n += 1

    return n


def migrate_sparse_replay(writer: ResultsWriter, data: list, dry_run: bool) -> int:
    """
    Import sparse_replay results.

    These are honest runs testing that sparse masks reproduce exact output.
    attack_type = "honest", compression and cosine similarity recorded.
    """
    n = 0
    for rec in data:
        model_id = writer.ensure_model("Qwen/Qwen2.5-7B")
        prompt_id = writer.ensure_prompt(
            rec["prompt"], complexity=_infer_complexity(rec["prompt"])
        )

        if not dry_run:
            writer.add_result(
                model_id=model_id,
                prompt_id=prompt_id,
                attack_type="honest",
                attack_params={"experiment": "sparse_replay"},
                token_match_rate=rec.get("token_match_rate"),
                cosine_similarity=rec.get("cosine_similarity"),
                compression_pct=(rec.get("compression") or 0) * 100,
                pass_fail=rec.get("exact_text_match"),
                verification_target="local",
                raw_data={
                    "sparsity": rec.get("sparsity"),
                    "max_logit_diff": rec.get("max_logit_diff"),
                    "mean_logit_diff": rec.get("mean_logit_diff"),
                    "same_next_token": rec.get("same_next_token"),
                    "top5_overlap": rec.get("top5_overlap"),
                    "full_text": rec.get("full_text"),
                    "sparse_text": rec.get("sparse_text"),
                },
            )
            n += 1
        else:
            n += 1

    return n


def migrate_max_savings(writer: ResultsWriter, data: list, dry_run: bool) -> int:
    """
    Import max_savings results.

    These are attention-skip attack results.
    attack_type = "attention_skip"
    """
    records = data if isinstance(data, list) else data.get("results", [])
    n = 0
    for rec in records:
        model_id = writer.ensure_model(rec.get("model_id", "unknown"))
        prompt_id = writer.ensure_prompt(
            rec["prompt"],
            complexity=rec.get("tier") or _infer_complexity(rec["prompt"]),
        )
        at = "attention_skip"
        ap = {
            "layers_skipped": rec.get("layers_skipped", []),
            "skip_strategy": rec.get("strategy", ""),
            "skip_count": rec.get("skip_count"),
        }

        if not dry_run:
            writer.add_result(
                model_id=model_id,
                prompt_id=prompt_id,
                attack_type=at,
                attack_params=ap,
                token_match_rate=rec.get("token_match_rate"),
                cosine_similarity=rec.get("mean_cosine_similarity"),
                perplexity=rec.get("perplexity"),
                coherence=rec.get("coherence"),
                savings_pct=rec.get("savings_pct"),
                pass_fail=rec.get("verification_passed"),
                verification_target="local",
                raw_data={
                    "honest_tokens": rec.get("honest_tokens"),
                    "fraudulent_tokens": rec.get("fraudulent_tokens"),
                    "verification_match_rate": rec.get("verification_match_rate"),
                    "elapsed_s": rec.get("elapsed_s"),
                },
            )
            n += 1
        else:
            n += 1

    return n


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_MIGRATIONS = [
    {
        "marker": "threshold_study/results.json",
        "path": ANALYSIS_DIR / "threshold_study" / "results.json",
        "experiment_name": "threshold_study",
        "script_path": "threshold_study.py",
        "fn": migrate_threshold_study,
    },
    {
        "marker": "adversarial_study/results.json",
        "path": ANALYSIS_DIR / "adversarial_study" / "results.json",
        "experiment_name": "adversarial_study",
        "script_path": "adversarial_study.py",
        "fn": migrate_adversarial_study,
    },
    {
        "marker": "sparse_replay/results.json",
        "path": ANALYSIS_DIR / "sparse_replay" / "results.json",
        "experiment_name": "sparse_replay",
        "script_path": "sparse_replay.py",
        "fn": migrate_sparse_replay,
    },
    {
        "marker": "max_savings/results.json",
        "path": ANALYSIS_DIR / "max_savings" / "results.json",
        "experiment_name": "max_savings",
        "script_path": "tools/max_savings_test.py",
        "fn": migrate_max_savings,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate existing JSON results into results.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Path to results.db.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count rows without writing anything.")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)

    total_experiments = 0
    total_results = 0
    skipped = []
    errors = []

    print()
    print("=" * 66)
    print("  MIGRATION" + ("  (DRY RUN — no writes)" if args.dry_run else ""))
    print("=" * 66)

    for mig in _MIGRATIONS:
        path: Path = mig["path"]
        marker: str = mig["marker"]

        if not path.exists():
            print(f"  SKIP  {marker}  (file not found)")
            skipped.append(marker)
            continue

        if not args.dry_run and _already_migrated(conn, marker):
            print(f"  SKIP  {marker}  (already imported)")
            skipped.append(marker)
            continue

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  ERROR {marker}: {exc}")
            errors.append(marker)
            continue

        notes = f"migrated_from:{marker}"

        if args.dry_run:
            # Dry run: create a temporary writer that does no DB work
            class _DryWriter:
                _model_cache: dict = {}
                _prompt_cache: dict = {}
                def ensure_model(self, name, **_): return 0
                def ensure_prompt(self, text, **_): return 0
                def add_result(self, **_): pass
            dw = _DryWriter()
            n = mig["fn"](dw, data if isinstance(data, list) else data.get("results", data), True)
            print(f"  DRY   {marker}  →  {n} rows would be written")
            total_results += n
            total_experiments += 1
        else:
            with ResultsWriter(
                mig["experiment_name"],
                script_path=mig["script_path"],
                config_name="default",
                notes=notes,
                db_path=db_path,
            ) as writer:
                records = data if isinstance(data, list) else data.get("results", data)
                n = mig["fn"](writer, records, False)
            print(f"  OK    {marker}  →  {n} rows written")
            total_experiments += 1
            total_results += n

    conn.close()

    print()
    print("-" * 66)
    print(f"  Experiments imported : {total_experiments}")
    print(f"  Results written      : {total_results:,}")
    if skipped:
        print(f"  Skipped              : {len(skipped)} ({', '.join(skipped[:3])})")
    if errors:
        print(f"  Errors               : {len(errors)} ({', '.join(errors)})")
    print("=" * 66)
    print()

    if not args.dry_run and total_results > 0:
        print("  Run  python3 analyze.py --summary  to view results.")
        print()


if __name__ == "__main__":
    main()
