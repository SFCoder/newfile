#!/usr/bin/env python3
"""
analyze.py — command-line analysis and query tool for results.db.

Usage
-----
    # Summary dashboard
    python3 analyze.py --summary

    # Group-by queries
    python3 analyze.py --metric token_match_rate --group-by model_name,attack_type
    python3 analyze.py --metric cosine_similarity --group-by attack_type \\
                        --where "model_name='Qwen/Qwen2.5-7B'"
    python3 analyze.py --metric savings_pct --group-by model_name,complexity_tier \\
                        --where "attack_type='attention_skip'"

    # Cross-dimension comparison (optionally with plot)
    python3 analyze.py --compare model_name --metric token_match_rate \\
                        --where "attack_type='attention_skip'" --plot
    python3 analyze.py --compare attack_type --metric pass_fail \\
                        --where "model_name='Qwen/Qwen2.5-7B'"

    # Metadata listing
    python3 analyze.py --list-experiments
    python3 analyze.py --configs

    # Export
    python3 analyze.py --export results.csv --where "experiment_name='max_savings'"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from adversarial_suite.db.schema import DEFAULT_DB_PATH, get_connection  # noqa: E402

PLOTS_DIR = _ROOT / "analysis_results" / "plots"

# ---------------------------------------------------------------------------
# Dimension → SQL fragment mapping
# ---------------------------------------------------------------------------
# Each entry: column_alias -> (SELECT expression, JOIN clause or "")
# The JOIN clauses are accumulated and de-duplicated per query.

_DIM_MAP = {
    "model_name":       ("m.model_name",         "JOIN models m ON r.model_id = m.id"),
    "attack_type":      ("r.attack_type",         ""),
    "complexity_tier":  ("p.complexity_tier",     "JOIN prompts p ON r.prompt_id = p.id"),
    "experiment_name":  ("e.name",                "JOIN experiments e ON r.experiment_id = e.id"),
    "layer":            ("r.layer",               ""),
    "position":         ("r.position",            ""),
    "device_type":      ("hw.device_type",        "JOIN experiments e ON r.experiment_id = e.id "
                                                   "JOIN hardware hw ON e.machine_id = hw.id"),
    "config_name":      ("cfg.name",              "JOIN experiments e ON r.experiment_id = e.id "
                                                   "JOIN configurations cfg ON e.config_id = cfg.id"),
    "coherence":        ("r.coherence",           ""),
    "verification_target": ("r.verification_target", ""),
}

# Metric column → SQL column in results table
_METRIC_MAP = {
    "token_match_rate":   "r.token_match_rate",
    "cosine_similarity":  "r.cosine_similarity",
    "rank":               "r.rank",
    "perplexity":         "r.perplexity",
    "compression_pct":    "r.compression_pct",
    "savings_pct":        "r.savings_pct",
    "absolute_error":     "r.absolute_error",
    "pass_fail":          "r.pass_fail",
}

# WHERE clause alias rewrites (user-facing -> SQL expression)
_WHERE_ALIASES = {
    "model_name":       "m.model_name",
    "attack_type":      "r.attack_type",
    "complexity_tier":  "p.complexity_tier",
    "experiment_name":  "e.name",
    "device_type":      "hw.device_type",
    "config_name":      "cfg.name",
    "coherence":        "r.coherence",
    "layer":            "r.layer",
    "pass_fail":        "r.pass_fail",
    "verification_target": "r.verification_target",
}


# ---------------------------------------------------------------------------
# Query builder helpers
# ---------------------------------------------------------------------------


def _build_base_query(
    select_extras: list,
    dimensions: list,
    where_clause: str = "",
    extra_joins: list = None,
) -> tuple:
    """
    Build a SELECT … FROM results r … query with the required joins.

    Returns (sql, needed_joins_set).
    """
    joins_needed: set = set()

    # Collect joins from dimensions
    for dim in dimensions:
        if dim in _DIM_MAP:
            _, jclause = _DIM_MAP[dim]
            if jclause:
                # A single dimension may require multiple joins (e.g. device_type)
                for part in jclause.split(" JOIN "):
                    part = part.strip()
                    if part:
                        joins_needed.add("JOIN " + part if not part.upper().startswith("JOIN") else part)

    if extra_joins:
        for j in extra_joins:
            joins_needed.add(j)

    # Rewrite WHERE aliases
    sql_where = _rewrite_where(where_clause, dimensions) if where_clause else ""

    join_sql = "\n    ".join(sorted(joins_needed))
    select_sql = ", ".join(select_extras)
    where_sql = f"WHERE {sql_where}" if sql_where else ""

    return join_sql, select_sql, where_sql


def _rewrite_where(clause: str, active_dims: list) -> str:
    """
    Replace user-friendly dimension names with their SQL equivalents.
    Also injects required joins implicitly for any alias used in WHERE.
    """
    result = clause
    for alias, sql_expr in _WHERE_ALIASES.items():
        result = result.replace(alias + "=", sql_expr + "=")
        result = result.replace(alias + " =", sql_expr + " =")
        result = result.replace(alias + " LIKE", sql_expr + " LIKE")
        result = result.replace(alias + " IN", sql_expr + " IN")
    return result


def _ensure_where_joins(where_clause: str) -> list:
    """Return JOIN fragments required by any alias used in the WHERE clause."""
    joins = []
    for alias, (_, jclause) in _DIM_MAP.items():
        if alias in where_clause and jclause:
            for part in jclause.split(" JOIN "):
                part = part.strip()
                if part:
                    j = "JOIN " + part if not part.upper().startswith("JOIN") else part
                    if j not in joins:
                        joins.append(j)
    return joins


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_summary(conn) -> None:
    """Print a high-level dashboard of all experiments and key findings."""

    def q(sql):
        return conn.execute(sql).fetchone()

    def qa(sql):
        return conn.execute(sql).fetchall()

    n_experiments = q("SELECT COUNT(*) FROM experiments")[0]
    n_models = q("SELECT COUNT(*) FROM models")[0]
    n_prompts = q("SELECT COUNT(*) FROM prompts")[0]
    n_results = q("SELECT COUNT(*) FROM results")[0]

    _header("DATABASE SUMMARY")
    print(f"  Experiments : {n_experiments}")
    print(f"  Models      : {n_models}")
    print(f"  Prompts     : {n_prompts}")
    print(f"  Results     : {n_results:,}")

    if n_results == 0:
        print("\n  (no results yet — run migrate.py or an experiment script)")
        return

    # Per-attack-type summary
    rows = qa(
        """
        SELECT
            r.attack_type,
            COUNT(*) AS n,
            AVG(r.token_match_rate) AS avg_match,
            AVG(r.cosine_similarity) AS avg_cos,
            SUM(CASE WHEN r.pass_fail = 1 THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN r.pass_fail IS NOT NULL THEN 1 ELSE 0 END), 0)
                AS pass_rate
        FROM results r
        WHERE r.attack_type IS NOT NULL
        GROUP BY r.attack_type
        ORDER BY n DESC
        """
    )

    _header("PER-ATTACK-TYPE SUMMARY")
    _print_table(
        ["Attack Type", "N", "Avg Match", "Avg Cos Sim", "Pass Rate"],
        [
            [
                r["attack_type"],
                r["n"],
                _fmt_pct(r["avg_match"]),
                _fmt_f(r["avg_cos"]),
                _fmt_pct(r["pass_rate"]),
            ]
            for r in rows
        ],
        [22, 7, 10, 12, 10],
    )

    # Security highlights
    _header("SECURITY HIGHLIGHTS")

    # Maximum attacker savings with coherent output
    max_sav = q(
        """
        SELECT MAX(r.savings_pct) FROM results r
        WHERE r.attack_type = 'attention_skip'
          AND r.coherence = 'coherent'
          AND r.savings_pct IS NOT NULL
        """
    )
    print(f"  Max attacker savings (coherent output) : {_fmt_f(max_sav[0])}%")

    # Worst-case detection rate (fraud pass_fail = 1 means passed — attacker wins)
    worst_det = q(
        """
        SELECT 1.0 - AVG(r.pass_fail) FROM results r
        WHERE r.attack_type != 'honest' AND r.pass_fail IS NOT NULL
        """
    )
    print(f"  Worst-case fraud detection rate        : {_fmt_pct(worst_det[0])}")

    # Cross-hardware divergence (if multiple hardware entries)
    n_hw = q("SELECT COUNT(DISTINCT machine_id) FROM experiments")[0]
    if n_hw > 1:
        div = q(
            """
            SELECT AVG(r.absolute_error) FROM results r
            WHERE r.attack_type = 'cross_hardware'
              AND r.absolute_error IS NOT NULL
            """
        )
        print(f"  Avg cross-hardware divergence          : {_fmt_f(div[0])}")

    # Per-model coherence threshold
    model_rows = qa(
        """
        SELECT m.model_name,
               MAX(r.savings_pct) AS max_sav
        FROM results r
        JOIN models m ON r.model_id = m.id
        WHERE r.attack_type = 'attention_skip'
          AND r.coherence = 'coherent'
          AND r.savings_pct IS NOT NULL
        GROUP BY m.model_name
        ORDER BY m.hidden_size
        """
    )
    if model_rows:
        _header("MAX SAFE SAVINGS BY MODEL")
        _print_table(
            ["Model", "Max Safe Savings"],
            [[r["model_name"], _fmt_f(r["max_sav"]) + "%"] for r in model_rows],
            [30, 18],
        )


def cmd_group_by(conn, metric: str, group_by: str, where: str = "") -> None:
    """Group metric by one or more dimensions, printing mean/min/max/count."""
    dims = [d.strip() for d in group_by.split(",")]

    unknown = [d for d in dims if d not in _DIM_MAP]
    if unknown:
        print(f"  Unknown dimension(s): {unknown}")
        print(f"  Valid dimensions: {sorted(_DIM_MAP)}")
        return

    if metric not in _METRIC_MAP:
        print(f"  Unknown metric: {metric}")
        print(f"  Valid metrics: {sorted(_METRIC_MAP)}")
        return

    metric_col = _METRIC_MAP[metric]
    dim_exprs = [_DIM_MAP[d][0] for d in dims]
    dim_labels = " || ' / ' || ".join(
        [f"COALESCE(CAST({e} AS TEXT), 'null')" for e in dim_exprs]
    )

    extra_joins = _ensure_where_joins(where)
    join_sql, _, where_sql = _build_base_query([], dims, where, extra_joins)

    sql = f"""
        SELECT
            {", ".join(dim_exprs)},
            COUNT({metric_col})          AS n,
            AVG({metric_col})            AS mean_val,
            MIN({metric_col})            AS min_val,
            MAX({metric_col})            AS max_val
        FROM results r
        {join_sql}
        {"WHERE " + _rewrite_where(where, dims) if where else ""}
        GROUP BY {", ".join(dim_exprs)}
        ORDER BY {", ".join(dim_exprs)}
    """

    rows = conn.execute(sql).fetchall()
    if not rows:
        print("  (no rows match)")
        return

    col_w = [max(len(str(d)), 20) for d in dims] + [6, 12, 12, 12]
    _header(f"{metric.upper()} grouped by {group_by}")
    headers = dims + ["N", "Mean", "Min", "Max"]
    _print_table(
        headers,
        [[r[i] for i in range(len(dims))] + [r["n"], _fmt_f(r["mean_val"]),
          _fmt_f(r["min_val"]), _fmt_f(r["max_val"])]
         for r in rows],
        col_w,
    )


def cmd_compare(conn, compare_dim: str, metric: str, where: str = "", plot: bool = False) -> None:
    """Print a comparison table for compare_dim; optionally generate a plot."""
    if compare_dim not in _DIM_MAP:
        print(f"  Unknown dimension: {compare_dim}")
        return
    if metric not in _METRIC_MAP:
        print(f"  Unknown metric: {metric}")
        return

    dim_expr = _DIM_MAP[compare_dim][0]
    metric_col = _METRIC_MAP[metric]
    extra_joins = _ensure_where_joins(where)
    join_sql, _, _ = _build_base_query([], [compare_dim], where, extra_joins)

    sql = f"""
        SELECT
            {dim_expr} AS dim_val,
            COUNT({metric_col}) AS n,
            AVG({metric_col}) AS mean_val,
            MIN({metric_col}) AS min_val,
            MAX({metric_col}) AS max_val
        FROM results r
        {join_sql}
        {"WHERE " + _rewrite_where(where, [compare_dim]) if where else ""}
        GROUP BY {dim_expr}
        ORDER BY mean_val DESC
    """

    rows = conn.execute(sql).fetchall()
    if not rows:
        print("  (no rows match)")
        return

    _header(f"COMPARE {compare_dim.upper()} — metric: {metric}")
    _print_table(
        [compare_dim, "N", "Mean", "Min", "Max"],
        [[r["dim_val"], r["n"], _fmt_f(r["mean_val"]),
          _fmt_f(r["min_val"]), _fmt_f(r["max_val"])]
         for r in rows],
        [28, 6, 10, 10, 10],
    )

    if plot:
        _make_bar_plot(
            labels=[r["dim_val"] for r in rows],
            values=[r["mean_val"] for r in rows],
            xlabel=compare_dim,
            ylabel=f"Mean {metric}",
            title=f"{metric} by {compare_dim}" + (f" — {where}" if where else ""),
            filename=f"compare_{compare_dim}_{metric}.png",
        )


def cmd_list_experiments(conn) -> None:
    rows = conn.execute(
        """
        SELECT e.id, e.name, e.started_at, e.finished_at,
               e.git_branch, e.git_commit,
               hw.device_type, hw.hostname,
               (SELECT COUNT(*) FROM results r WHERE r.experiment_id = e.id) AS n_results
        FROM experiments e
        LEFT JOIN hardware hw ON e.machine_id = hw.id
        ORDER BY e.started_at
        """
    ).fetchall()

    _header("EXPERIMENTS")
    _print_table(
        ["ID", "Name", "Started", "Device", "Host", "Branch", "Results"],
        [
            [
                r["id"],
                r["name"],
                (r["started_at"] or "")[:16],
                r["device_type"] or "?",
                r["hostname"] or "?",
                (r["git_branch"] or "")[:20],
                r["n_results"],
            ]
            for r in rows
        ],
        [4, 22, 16, 6, 16, 20, 8],
    )


def cmd_configs(conn) -> None:
    rows = conn.execute("SELECT * FROM configurations ORDER BY id").fetchall()
    _header("CONFIGURATIONS")
    _print_table(
        ["ID", "Name", "Threshold", "Cos Match", "Top-K", "Sample Rate", "Notes"],
        [
            [
                r["id"],
                r["name"],
                r["neuron_threshold"],
                r["match_cosine"],
                r["top_k"],
                r["verification_sample_rate"],
                (r["notes"] or "")[:40],
            ]
            for r in rows
        ],
        [4, 22, 12, 10, 6, 12, 40],
    )


def cmd_export(conn, output_path: str, where: str = "") -> None:
    """Export query results to a CSV file."""
    all_joins = (
        "JOIN models m ON r.model_id = m.id "
        "JOIN prompts p ON r.prompt_id = p.id "
        "JOIN experiments e ON r.experiment_id = e.id "
        "LEFT JOIN hardware hw ON e.machine_id = hw.id "
        "LEFT JOIN configurations cfg ON e.config_id = cfg.id"
    )
    sql = f"""
        SELECT
            e.name AS experiment_name,
            m.model_name,
            p.text AS prompt,
            p.complexity_tier,
            r.attack_type,
            r.attack_params,
            r.layer,
            r.position,
            r.token_match_rate,
            r.cosine_similarity,
            r.rank,
            r.perplexity,
            r.coherence,
            r.compression_pct,
            r.savings_pct,
            r.absolute_error,
            r.pass_fail,
            r.verification_target,
            hw.device_type,
            hw.hostname,
            e.git_commit,
            e.started_at
        FROM results r
        {all_joins}
        {"WHERE " + _rewrite_where(where, list(_DIM_MAP)) if where else ""}
        ORDER BY e.started_at, r.id
    """
    rows = conn.execute(sql).fetchall()
    if not rows:
        print("  (no rows match — nothing exported)")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(list(r))
    print(f"  Exported {len(rows):,} rows to {out}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _make_bar_plot(
    labels: list,
    values: list,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    clean = [str(l) if l is not None else "null" for l in labels]
    ax.bar(clean, [v if v is not None else 0 for v in values])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def _make_line_plot(
    x_values: list,
    y_values: list,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    w = 66
    print()
    print("  " + "=" * w)
    print(f"  {title}")
    print("  " + "=" * w)


def _print_table(headers: list, rows: list, widths: list) -> None:
    def _row(vals):
        return "  ".join(
            str(v if v is not None else "")[:w].ljust(w)
            for v, w in zip(vals, widths)
        )

    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("  " + sep)
    print("  " + _row(headers))
    print("  " + sep)
    for r in rows:
        print("  " + _row(r))
    print("  " + sep)


def _fmt_f(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query and visualise results.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Path to results.db (default: project root).")
    parser.add_argument("--summary", action="store_true",
                        help="Print overview dashboard.")
    parser.add_argument("--metric",
                        help="Metric column to aggregate.")
    parser.add_argument("--group-by",
                        help="Comma-separated dimensions to group by.")
    parser.add_argument("--compare",
                        help="Dimension to compare values across.")
    parser.add_argument("--where", default="",
                        help="SQL WHERE clause (use column aliases).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a matplotlib plot for --compare.")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List all experiment runs.")
    parser.add_argument("--configs", action="store_true",
                        help="Show configuration table.")
    parser.add_argument("--export",
                        help="Export filtered results to this CSV file.")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"  Database not found: {db_path}")
        print("  Run migrate.py to populate from existing JSON files, or run")
        print("  an experiment script to start collecting results.")
        sys.exit(1)

    conn = get_connection(db_path)

    try:
        if args.summary:
            cmd_summary(conn)
        elif args.group_by and args.metric:
            cmd_group_by(conn, args.metric, args.group_by, args.where)
        elif args.compare and args.metric:
            cmd_compare(conn, args.compare, args.metric, args.where, args.plot)
        elif args.list_experiments:
            cmd_list_experiments(conn)
        elif args.configs:
            cmd_configs(conn)
        elif args.export:
            cmd_export(conn, args.export, args.where)
        else:
            parser.print_help()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
