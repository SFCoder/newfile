"""
standard_format.py
==================
Write and read experiment results in the universal standard_v1 JSON format.

Any experiment that produces a file in this format is auto-imported by
migrate.py without requiring a custom parser.  This means new experiment
scripts never need a migrate.py update — they just call write_standard_results()
and the data flows into the database automatically.

File layout
-----------
::

    {
      "format": "standard_v1",
      "experiment_name": "my_experiment",
      "script_path": "tools/my_experiment.py",
      "written_at": "2024-01-01T00:00:00Z",
      "results": [
        {
          // ── required ──────────────────────────────────────────────
          "model_name":  "Qwen/Qwen2.5-7B",
          "prompt_text": "The capital of France is",
          "attack_type": "attention_skip",

          // ── optional model metadata (used by ensure_model) ────────
          "model_parameter_count_b": 7.0,
          "model_num_layers":        28,
          "model_intermediate_size": 18944,
          "model_hidden_size":       3584,
          "model_weight_hash":       "b21b4b…",

          // ── optional prompt metadata ──────────────────────────────
          "prompt_complexity": "simple",    // "simple"|"moderate"|"complex"
          "prompt_category":   "factual",

          // ── result columns (all nullable) ─────────────────────────
          "attack_params":      {"layers_skipped": [5, 14], "skip_strategy": "best_case"},
          "layer":              null,
          "position":           null,
          "token_match_rate":   0.85,
          "cosine_similarity":  0.9923,
          "rank":               1.0,
          "perplexity":         2.1,
          "coherence":          "coherent",
          "compression_pct":    null,
          "savings_pct":        3.5,
          "absolute_error":     null,
          "pass_fail":          true,
          "verification_target": "local",
          "raw_data":           {}
        }
      ]
    }

Required keys per result: ``model_name``, ``prompt_text``, ``attack_type``.
All other keys are optional — missing values are stored as NULL.

Usage
-----
::

    from adversarial_suite.db.standard_format import write_standard_results

    write_standard_results(
        results=[
            {
                "model_name":       "Qwen/Qwen2.5-7B",
                "model_num_layers": 28,
                "prompt_text":      "The capital of France is",
                "prompt_complexity": "simple",
                "attack_type":      "attention_skip",
                "attack_params":    {"layers_skipped": [5, 14]},
                "token_match_rate": 0.85,
                "savings_pct":      3.5,
                "pass_fail":        True,
            }
        ],
        filepath=Path("analysis_results/my_experiment/results_standard.json"),
        experiment_name="my_experiment",
        script_path="tools/my_experiment.py",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

FORMAT_KEY = "standard_v1"

# All recognised per-result column names (excluding model_* and prompt_* prefixed keys)
_RESULT_COLUMNS = {
    "attack_params", "layer", "position",
    "token_match_rate", "cosine_similarity", "rank", "perplexity",
    "coherence", "compression_pct", "savings_pct", "absolute_error",
    "pass_fail", "verification_target", "raw_data",
}

# Model metadata keys carried inside each result record
_MODEL_META_KEYS = {
    "model_parameter_count_b", "model_num_layers", "model_intermediate_size",
    "model_hidden_size", "model_weight_hash",
}

# Prompt metadata keys carried inside each result record
_PROMPT_META_KEYS = {"prompt_complexity", "prompt_category", "prompt_token_count"}


def write_standard_results(
    results: list,
    filepath: Path,
    experiment_name: str = "",
    script_path: str = "",
) -> None:
    """
    Write *results* to *filepath* in the standard_v1 JSON format.

    Parameters
    ----------
    results
        List of dicts.  Each dict must contain ``model_name``,
        ``prompt_text``, and ``attack_type``.  All other keys listed in
        the module docstring are optional.
    filepath
        Destination path.  Parent directories are created automatically.
    experiment_name
        Human-readable experiment identifier stored in the file header.
        Used as the experiment name when migrate.py imports the file.
    script_path
        Relative path to the script that produced the file (informational).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": FORMAT_KEY,
        "experiment_name": experiment_name,
        "script_path": script_path,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": [_normalise(r) for r in results],
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"  Saved standard results ({len(results)} rows) → {filepath}")


def is_standard_format(data: dict) -> bool:
    """Return True if *data* is a standard_v1 results file."""
    return isinstance(data, dict) and data.get("format") == FORMAT_KEY


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(rec: dict) -> dict:
    """
    Normalise a result dict to the canonical standard_v1 shape.

    Unknown keys are folded into ``raw_data`` so no information is lost.
    """
    out: dict = {}

    # Required fields
    out["model_name"]  = rec.get("model_name", "")
    out["prompt_text"] = rec.get("prompt_text", "")
    out["attack_type"] = rec.get("attack_type", "")

    # Model metadata
    for k in _MODEL_META_KEYS:
        if k in rec:
            out[k] = rec[k]

    # Prompt metadata
    for k in _PROMPT_META_KEYS:
        if k in rec:
            out[k] = rec[k]

    # Standard result columns
    for col in _RESULT_COLUMNS:
        if col in rec:
            out[col] = rec[col]

    # Fold unrecognised keys into raw_data
    known = (
        {"model_name", "prompt_text", "attack_type"}
        | _MODEL_META_KEYS
        | _PROMPT_META_KEYS
        | _RESULT_COLUMNS
    )
    overflow = {k: v for k, v in rec.items() if k not in known}
    if overflow:
        existing = out.get("raw_data") or {}
        if isinstance(existing, str):
            try:
                existing = json.loads(existing)
            except Exception:
                existing = {"_raw": existing}
        existing.update(overflow)
        out["raw_data"] = existing

    return out


def _json_default(obj):
    """Fallback JSON serialiser for non-standard types."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)
