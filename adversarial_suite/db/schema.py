"""
Database schema for the adversarial testing suite.

Creates and manages results.db (SQLite) with the following tables:

  hardware        — machine where an experiment ran
  configurations  — tunable verification parameters (future chain params)
  experiments     — one row per experiment run
  models          — registered model metadata
  prompts         — deduplicated prompt texts
  results         — one row per (experiment, model, prompt, attack config)

The schema is forward-compatible: the results.raw_data TEXT column holds
arbitrary JSON for measurements that don't fit the standard columns, so
new experiment types never require schema alterations.

WAL journal mode is enabled for better concurrent read performance.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Default location: project root
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "results.db"

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hardware (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    device_type     TEXT    NOT NULL,           -- "mps" | "cuda" | "cpu"
    device_name     TEXT,                        -- e.g. "Apple M2 Pro", "NVIDIA A100 80GB"
    gpu_memory_gb   REAL,                        -- null for CPU/MPS
    hostname        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS configurations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT    NOT NULL UNIQUE,
    neuron_threshold        REAL,               -- activation magnitude cutoff
    match_cosine            REAL,               -- cosine similarity pass threshold
    top_k                   INTEGER,            -- top-k logit comparison window
    verification_sample_rate REAL,              -- fraction of layers verified (0–1)
    notes                   TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,               -- e.g. "max_savings", "threshold_study"
    script_path TEXT,                           -- relative path to the producing script
    started_at  TEXT,                           -- ISO-8601 datetime
    finished_at TEXT,
    machine_id  INTEGER REFERENCES hardware(id),
    git_branch  TEXT,
    git_commit  TEXT,
    notes       TEXT,
    config_id   INTEGER REFERENCES configurations(id)
);

CREATE TABLE IF NOT EXISTS models (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT    NOT NULL UNIQUE,    -- HuggingFace repo ID
    parameter_count_b   REAL,                        -- e.g. 7.0, 72.0
    num_layers          INTEGER,
    intermediate_size   INTEGER,
    hidden_size         INTEGER,
    weight_hash         TEXT,
    registry_entry_json TEXT                         -- full registry entry JSON
);

CREATE TABLE IF NOT EXISTS prompts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT    NOT NULL UNIQUE,
    complexity_tier TEXT,                        -- "simple" | "moderate" | "complex" | null
    token_count     INTEGER,
    category        TEXT                         -- "factual" | "reasoning" | "creative" | null
);

CREATE TABLE IF NOT EXISTS results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       INTEGER REFERENCES experiments(id),
    model_id            INTEGER REFERENCES models(id),
    prompt_id           INTEGER REFERENCES prompts(id),
    attack_type         TEXT,    -- "honest" | "attention_skip" | "random_mask" |
                                 -- "model_substitution" | "token_tamper" |
                                 -- "split_verification" | "cross_hardware" |
                                 -- "threshold_sweep" | "fake_ffn"
    attack_params       TEXT,    -- JSON dict: {"layers_skipped": [5,14], ...}
    layer               INTEGER, -- null = whole-inference; specific layer index
    position            INTEGER, -- null = aggregate; specific token position
    token_match_rate    REAL,
    cosine_similarity   REAL,
    rank                REAL,
    perplexity          REAL,
    coherence           TEXT,    -- "coherent" | "degraded" | "garbage" | null
    compression_pct     REAL,
    savings_pct         REAL,
    absolute_error      REAL,
    pass_fail           INTEGER, -- 1 = pass, 0 = fail, null = not applicable
    verification_target TEXT,    -- "local" | "cosmos"
    raw_data            TEXT     -- JSON for any overflow measurements
);

CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_results_model      ON results(model_id);
CREATE INDEX IF NOT EXISTS idx_results_attack     ON results(attack_type);
CREATE INDEX IF NOT EXISTS idx_results_prompt     ON results(prompt_id);
CREATE INDEX IF NOT EXISTS idx_results_model_attack ON results(model_id, attack_type);
"""

# Default configurations inserted on first creation
_DEFAULT_CONFIGS = [
    {
        "name": "default",
        "neuron_threshold": 1e-6,
        "match_cosine": 0.99,
        "top_k": None,
        "verification_sample_rate": 1.0,
        "notes": "Standard configuration used in all baseline experiments.",
    },
    {
        "name": "aggressive_threshold",
        "neuron_threshold": 0.01,
        "match_cosine": 0.95,
        "top_k": None,
        "verification_sample_rate": 1.0,
        "notes": "Higher neuron threshold — more compression, slightly lower fidelity.",
    },
    {
        "name": "relaxed_matching",
        "neuron_threshold": 1e-6,
        "match_cosine": 0.85,
        "top_k": None,
        "verification_sample_rate": 0.5,
        "notes": "Relaxed pass threshold and 50% layer sampling — faster verification.",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at db_path.

    Applies WAL mode, foreign-key enforcement, and row_factory on every
    new connection.  Callers are responsible for closing the connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create tables and seed default configurations if the database is new.

    Safe to call on an existing database — all CREATE statements use
    IF NOT EXISTS.

    Returns an open connection; caller must close it.
    """
    conn = get_connection(db_path)
    conn.executescript(_DDL)

    # Seed default configurations (INSERT OR IGNORE to stay idempotent)
    for cfg in _DEFAULT_CONFIGS:
        conn.execute(
            """
            INSERT OR IGNORE INTO configurations
                (name, neuron_threshold, match_cosine, top_k,
                 verification_sample_rate, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cfg["name"],
                cfg["neuron_threshold"],
                cfg["match_cosine"],
                cfg["top_k"],
                cfg["verification_sample_rate"],
                cfg["notes"],
            ),
        )
    conn.commit()
    return conn
