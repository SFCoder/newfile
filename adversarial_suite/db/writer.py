"""
ResultsWriter
=============
Context-manager class for writing experiment results to results.db.

All experiment scripts import this class and use it as shown below.
The writer handles: hardware auto-detection, git metadata capture,
model/prompt/config upserts (no duplicates), batched inserts, and
automatic experiment start/end timestamps.

Usage::

    from adversarial_suite.db.writer import ResultsWriter

    with ResultsWriter("max_savings", config_name="default") as writer:
        model_id  = writer.ensure_model("Qwen/Qwen2.5-7B", registry=registry)
        prompt_id = writer.ensure_prompt("The capital of France is",
                                         complexity="simple")
        writer.add_result(
            model_id=model_id,
            prompt_id=prompt_id,
            attack_type="attention_skip",
            attack_params={"layers_skipped": [5, 14], "skip_strategy": "best_case"},
            token_match_rate=0.85,
            cosine_similarity=0.9923,
            perplexity=2.1,
            coherence="coherent",
            savings_pct=3.5,
            pass_fail=True,
        )
    # finished_at written on __exit__

To swap from local to CosmosVerification in the future, change the
verification_target parameter to "cosmos" — the writer accepts it as a
plain string.

Deduplication
-------------
Models and prompts are identified by natural key (model_name and
prompt text, respectively).  ensure_model / ensure_prompt return the
existing row ID if the record already exists, so running the same
experiment twice does not create duplicates.  Experiment records are
always new rows (each run is a distinct event).

Batching
--------
Results are buffered in memory and flushed to the database every
FLUSH_BATCH_SIZE inserts, or when the context manager exits.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from adversarial_suite.db.schema import DEFAULT_DB_PATH, init_db

FLUSH_BATCH_SIZE = 100

_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_info() -> tuple:
    """Return (branch, commit_hash) from the repo at _ROOT."""
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=str(_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def _detect_hardware() -> dict:
    """
    Detect the current machine's hardware and return a dict ready for
    INSERT into the hardware table.
    """
    import platform

    hostname = socket.gethostname()
    device_type = "cpu"
    device_name = platform.processor() or platform.machine() or "unknown"
    gpu_memory_gb = None

    try:
        import torch

        if torch.cuda.is_available():
            device_type = "cuda"
            props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = round(props.total_memory / 1e9, 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon (MPS)"
    except ImportError:
        pass

    return {
        "device_type": device_type,
        "device_name": device_name,
        "gpu_memory_gb": gpu_memory_gb,
        "hostname": hostname,
    }


# ---------------------------------------------------------------------------
# ResultsWriter
# ---------------------------------------------------------------------------


class ResultsWriter:
    """
    Context-manager that writes experiment results to results.db.

    Parameters
    ----------
    experiment_name
        Short identifier for this experiment run, e.g. "max_savings".
    script_path
        Relative path to the script that created this experiment.
        Auto-detected from __file__ if not provided.
    config_name
        Name of the configuration record to associate with this run.
        Must exist in the configurations table (seeded by init_db).
        Defaults to "default".
    notes
        Optional free-text annotation.
    db_path
        Path to the SQLite database file.  Defaults to project root.
    """

    def __init__(
        self,
        experiment_name: str,
        script_path: str = "",
        config_name: str = "default",
        notes: str = "",
        db_path: Path = DEFAULT_DB_PATH,
    ):
        self.experiment_name = experiment_name
        self.script_path = script_path
        self.config_name = config_name
        self.notes = notes
        self.db_path = db_path

        self._conn = None
        self._experiment_id: Optional[int] = None
        self._result_buffer: list = []

        # Caches to avoid redundant DB lookups within a session
        self._model_cache: dict = {}    # model_name -> id
        self._prompt_cache: dict = {}   # prompt_text -> id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResultsWriter":
        self._conn = init_db(self.db_path)

        hw = _detect_hardware()
        machine_id = self._upsert_hardware(hw)
        config_id = self._find_config_id(self.config_name)
        git_branch, git_commit = _git_info()

        cur = self._conn.execute(
            """
            INSERT INTO experiments
                (name, script_path, started_at, machine_id,
                 git_branch, git_commit, notes, config_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment_name,
                self.script_path,
                _utcnow(),
                machine_id,
                git_branch,
                git_commit,
                self.notes,
                config_id,
            ),
        )
        self._conn.commit()
        self._experiment_id = cur.lastrowid
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush()
        if self._conn:
            self._conn.execute(
                "UPDATE experiments SET finished_at = ? WHERE id = ?",
                (_utcnow(), self._experiment_id),
            )
            self._conn.commit()
            self._conn.close()
            self._conn = None
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def experiment_id(self) -> Optional[int]:
        return self._experiment_id

    def ensure_model(
        self,
        model_name: str,
        registry=None,
        parameter_count_b: Optional[float] = None,
        num_layers: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weight_hash: Optional[str] = None,
    ) -> int:
        """
        Return the models.id for model_name, creating the record if needed.

        If a ModelRegistry instance is supplied, metadata is pulled from
        the registry entry.  Manual keyword arguments supplement or
        override registry data.

        Parameters
        ----------
        model_name
            HuggingFace model ID, e.g. "Qwen/Qwen2.5-7B".
        registry
            Optional ModelRegistry instance from model_registry.py.
        parameter_count_b, num_layers, intermediate_size, hidden_size, weight_hash
            Manual values used when no registry is provided, or to override
            registry values.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        # Attempt registry lookup
        reg_json: Optional[str] = None
        if registry is not None:
            try:
                entry = registry.get_entry(model_name)
                num_layers = num_layers or entry.num_layers
                intermediate_size = intermediate_size or entry.intermediate_size
                hidden_size = hidden_size or entry.hidden_size
                weight_hash = weight_hash or entry.weight_hash
                reg_json = json.dumps(
                    {
                        "model_id": entry.model_id,
                        "hf_repo": entry.hf_repo,
                        "weight_hash": entry.weight_hash,
                        "num_layers": entry.num_layers,
                        "intermediate_size": entry.intermediate_size,
                        "hidden_size": entry.hidden_size,
                        "min_stake": entry.min_stake,
                    }
                )
                # Infer parameter_count_b from hidden_size if not given
                if parameter_count_b is None:
                    parameter_count_b = _infer_param_count(model_name, hidden_size)
            except Exception:
                pass

        row_id = self._upsert_model(
            model_name=model_name,
            parameter_count_b=parameter_count_b,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            weight_hash=weight_hash,
            registry_entry_json=reg_json,
        )
        self._model_cache[model_name] = row_id
        return row_id

    def ensure_prompt(
        self,
        text: str,
        complexity: Optional[str] = None,
        category: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """
        Return the prompts.id for text, creating the record if needed.

        If a prompt with the same text already exists, the existing ID is
        returned and the metadata is NOT updated (to avoid inconsistencies
        across experiments).

        Parameters
        ----------
        text
            The verbatim prompt string.
        complexity
            "simple" | "moderate" | "complex" | None.
        category
            Optional tag: "factual" | "reasoning" | "creative" | etc.
        token_count
            Pre-computed token count (informational only).
        """
        if text in self._prompt_cache:
            return self._prompt_cache[text]

        row_id = self._upsert_prompt(text, complexity, category, token_count)
        self._prompt_cache[text] = row_id
        return row_id

    def add_result(
        self,
        model_id: int,
        prompt_id: int,
        attack_type: str,
        attack_params: Optional[dict] = None,
        layer: Optional[int] = None,
        position: Optional[int] = None,
        token_match_rate: Optional[float] = None,
        cosine_similarity: Optional[float] = None,
        rank: Optional[float] = None,
        perplexity: Optional[float] = None,
        coherence: Optional[str] = None,
        compression_pct: Optional[float] = None,
        savings_pct: Optional[float] = None,
        absolute_error: Optional[float] = None,
        pass_fail: Optional[bool] = None,
        verification_target: str = "local",
        raw_data: Optional[dict] = None,
    ) -> None:
        """
        Buffer a single result row.  Flushed every FLUSH_BATCH_SIZE rows
        or when the context manager exits.

        Parameters mirror the results table columns.  See schema.py for
        column descriptions.
        """
        pf = None if pass_fail is None else (1 if pass_fail else 0)
        self._result_buffer.append(
            (
                self._experiment_id,
                model_id,
                prompt_id,
                attack_type,
                json.dumps(attack_params) if attack_params else None,
                layer,
                position,
                token_match_rate,
                cosine_similarity,
                rank,
                perplexity,
                coherence,
                compression_pct,
                savings_pct,
                absolute_error,
                pf,
                verification_target,
                json.dumps(raw_data) if raw_data else None,
            )
        )
        if len(self._result_buffer) >= FLUSH_BATCH_SIZE:
            self._flush()

    def add_result_from_dict(self, row: dict) -> None:
        """
        Convenience wrapper: accepts a dict whose keys match add_result
        keyword parameters.  Unknown keys are silently forwarded to
        raw_data.
        """
        known = {
            "model_id", "prompt_id", "attack_type", "attack_params",
            "layer", "position", "token_match_rate", "cosine_similarity",
            "rank", "perplexity", "coherence", "compression_pct",
            "savings_pct", "absolute_error", "pass_fail",
            "verification_target", "raw_data",
        }
        kwargs = {k: v for k, v in row.items() if k in known}
        overflow = {k: v for k, v in row.items() if k not in known}
        if overflow:
            existing = kwargs.get("raw_data") or {}
            if isinstance(existing, str):
                try:
                    existing = json.loads(existing)
                except Exception:
                    existing = {}
            existing.update(overflow)
            kwargs["raw_data"] = existing
        self.add_result(**kwargs)

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _upsert_hardware(self, hw: dict) -> int:
        """Find or insert a hardware row matching device_type + hostname."""
        cur = self._conn.execute(
            "SELECT id FROM hardware WHERE device_type = ? AND hostname = ?",
            (hw["device_type"], hw["hostname"]),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self._conn.execute(
            """
            INSERT INTO hardware (device_type, device_name, gpu_memory_gb, hostname)
            VALUES (?, ?, ?, ?)
            """,
            (hw["device_type"], hw["device_name"], hw["gpu_memory_gb"], hw["hostname"]),
        )
        self._conn.commit()
        return cur.lastrowid

    def _find_config_id(self, name: str) -> Optional[int]:
        cur = self._conn.execute(
            "SELECT id FROM configurations WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        return row["id"] if row else None

    def _upsert_model(self, model_name: str, **kwargs) -> int:
        cur = self._conn.execute(
            "SELECT id FROM models WHERE model_name = ?", (model_name,)
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self._conn.execute(
            """
            INSERT INTO models
                (model_name, parameter_count_b, num_layers, intermediate_size,
                 hidden_size, weight_hash, registry_entry_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                kwargs.get("parameter_count_b"),
                kwargs.get("num_layers"),
                kwargs.get("intermediate_size"),
                kwargs.get("hidden_size"),
                kwargs.get("weight_hash"),
                kwargs.get("registry_entry_json"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def _upsert_prompt(
        self,
        text: str,
        complexity: Optional[str],
        category: Optional[str],
        token_count: Optional[int],
    ) -> int:
        cur = self._conn.execute(
            "SELECT id FROM prompts WHERE text = ?", (text,)
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self._conn.execute(
            "INSERT INTO prompts (text, complexity_tier, category, token_count) VALUES (?, ?, ?, ?)",
            (text, complexity, category, token_count),
        )
        self._conn.commit()
        return cur.lastrowid

    def _flush(self) -> None:
        if not self._result_buffer or not self._conn:
            return
        self._conn.executemany(
            """
            INSERT INTO results (
                experiment_id, model_id, prompt_id, attack_type, attack_params,
                layer, position, token_match_rate, cosine_similarity, rank,
                perplexity, coherence, compression_pct, savings_pct,
                absolute_error, pass_fail, verification_target, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._result_buffer,
        )
        self._conn.commit()
        self._result_buffer.clear()


# ---------------------------------------------------------------------------
# Helper: infer approximate parameter count from model name / hidden_size
# ---------------------------------------------------------------------------


def _infer_param_count(model_name: str, hidden_size: Optional[int]) -> Optional[float]:
    name = model_name.lower()
    for tag, val in [("72b", 72.0), ("7b", 7.0), ("3b", 3.0), ("0.5b", 0.5),
                     ("500m", 0.5)]:
        if tag in name:
            return val
    if hidden_size:
        # Very rough estimate: ~12 * hidden_size^2 parameters
        return round(12 * (hidden_size ** 2) / 1e9, 1)
    return None
