#!/usr/bin/env python3
"""
tools/export_dashboard.py — Generate analysis_results/dashboard.html from results.db.

Embeds all experiment data as inline JSON and opens the dashboard in the
default browser.  The output file is self-contained — no server, no CDN.

Usage
-----
    python3 tools/export_dashboard.py
    python3 tools/export_dashboard.py --db /path/to/results.db
    python3 tools/export_dashboard.py --out /path/to/dashboard.html
    python3 tools/export_dashboard.py --no-open
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DEFAULT_DB  = _ROOT / "results.db"
DEFAULT_OUT = _ROOT / "analysis_results" / "dashboard.html"


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def query_data(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # --- meta counts ---
    def count(table: str) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    meta = {
        "total_experiments": count("experiments"),
        "total_models":      count("models"),
        "total_results":     count("results"),
        "total_prompts":     count("prompts"),
        "generated_at":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    # --- attack type stats (for bar chart) ---
    attack_stats = [
        dict(r) for r in conn.execute("""
            SELECT attack_type,
                   ROUND(AVG(token_match_rate), 4)  AS avg_tmr,
                   COUNT(*)                          AS n
            FROM results
            WHERE token_match_rate IS NOT NULL
              AND layer    IS NULL
              AND position IS NULL
            GROUP BY attack_type
            ORDER BY avg_tmr DESC
        """)
    ]

    # --- results for explorer table ---
    results = []
    for r in conn.execute("""
        SELECT r.id,
               e.name          AS experiment_name,
               m.model_name,
               p.text          AS prompt_text,
               p.complexity_tier,
               r.attack_type,
               ROUND(r.token_match_rate,  4) AS token_match_rate,
               ROUND(r.cosine_similarity, 4) AS cosine_similarity,
               ROUND(r.rank,              2) AS rank,
               ROUND(r.perplexity,        3) AS perplexity,
               r.coherence,
               ROUND(r.savings_pct, 2)       AS savings_pct,
               r.pass_fail,
               r.layer,
               r.position
        FROM   results r
        JOIN   models      m ON m.id = r.model_id
        JOIN   prompts     p ON p.id = r.prompt_id
        JOIN   experiments e ON e.id = r.experiment_id
        ORDER BY r.id
    """):
        row = dict(r)
        results.append(row)

    conn.close()
    return {
        "meta":         meta,
        "attackStats":  attack_stats,
        "results":      results,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proof-of-Inference Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #1a1a2e;
  --surface:  #16213e;
  --surface2: #0f3460;
  --border:   #2a2a5a;
  --text:     #e0e0f0;
  --dim:      #8080b0;
  --accent:   #7c5cbf;
  --green:    #4ade80;
  --red:      #f87171;
  --yellow:   #fbbf24;
  --blue:     #60a5fa;

  --c-05b:  #4ecdc4;
  --c-3b:   #45b7d1;
  --c-7b:   #7ed680;
  --c-72b:  #f7c948;

  --c-honest:       #4ade80;
  --c-threshold:    #60a5fa;
  --c-attn:         #f87171;
  --c-fake-ffn:     #fb923c;
  --c-sub:          #c084fc;
  --c-tamper:       #f472b6;
  --c-rand:         #a78bfa;
  --c-split:        #67e8f9;
  --c-fingerprint:  #fde68a;
}

html { font-size: 14px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
  line-height: 1.5;
  min-height: 100vh;
}

a { color: var(--accent); }

/* ---------- layout ---------- */
.page { max-width: 1400px; margin: 0 auto; padding: 24px 20px 60px; }

header {
  display: flex;
  align-items: baseline;
  gap: 16px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}
header h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -.3px; }
header .ts { font-size: .8rem; color: var(--dim); margin-left: auto; }

section { margin-bottom: 32px; }
section > h2 {
  font-size: .95rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--dim);
  margin-bottom: 14px;
}

/* ---------- cards ---------- */
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 800px) { .cards { grid-template-columns: repeat(2, 1fr); } }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 20px;
}
.card .label { font-size: .78rem; color: var(--dim); text-transform: uppercase; letter-spacing: .06em; }
.card .value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }

/* ---------- filters ---------- */
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 24px;
}
.filters label { font-size: .8rem; color: var(--dim); display: flex; flex-direction: column; gap: 4px; }
.filters select {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 5px 10px;
  font-size: .85rem;
  min-width: 150px;
  cursor: pointer;
}
.filters select:focus { outline: 2px solid var(--accent); }
.btn-reset {
  margin-left: auto;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--dim);
  padding: 6px 14px;
  font-size: .82rem;
  cursor: pointer;
  transition: border-color .15s, color .15s;
}
.btn-reset:hover { border-color: var(--accent); color: var(--text); }

/* ---------- chart panel ---------- */
.chart-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px;
}
.chart-panel h3 { font-size: .85rem; color: var(--dim); font-weight: 500; margin-bottom: 14px; }
canvas { display: block; width: 100%; }

/* ---------- table ---------- */
.tbl-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: .82rem;
}
th {
  background: var(--surface2);
  color: var(--dim);
  text-align: left;
  padding: 9px 12px;
  font-weight: 500;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border);
}
th:hover { color: var(--text); }
th .sort-arrow { margin-left: 4px; opacity: .5; }
td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
}
tr.row-honest  td { background: rgba(74,222,128,.05); }
tr.row-fraud   td { background: rgba(248,113,113,.08); }
tr.row-ambig   td { background: rgba(251,191,36,.04); }

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: .75rem;
  font-weight: 600;
}
.badge-pass  { background: rgba(74,222,128,.15); color: var(--green); }
.badge-fail  { background: rgba(248,113,113,.15); color: var(--red); }
.badge-null  { background: rgba(128,128,176,.1);  color: var(--dim); }

/* ---------- pagination ---------- */
.pagination {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  font-size: .82rem;
  color: var(--dim);
}
.pagination button {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  padding: 4px 12px;
  cursor: pointer;
  font-size: .82rem;
}
.pagination button:disabled { opacity: .35; cursor: default; }
.pagination button:not(:disabled):hover { border-color: var(--accent); }
.pg-info { margin: 0 8px; }
</style>
</head>
<body>
<div class="page">

<header>
  <h1>Proof-of-Inference Experiment Dashboard</h1>
  <span class="ts" id="ts"></span>
</header>

<!-- Overview cards -->
<section>
  <h2>Overview</h2>
  <div class="cards">
    <div class="card"><div class="label">Experiments</div><div class="value" id="card-exp">—</div></div>
    <div class="card"><div class="label">Models Tested</div><div class="value" id="card-models">—</div></div>
    <div class="card"><div class="label">Total Results</div><div class="value" id="card-results">—</div></div>
    <div class="card"><div class="label">Prompts</div><div class="value" id="card-prompts">—</div></div>
  </div>
</section>

<!-- Filters -->
<div class="filters">
  <label>Model
    <select id="f-model"><option value="">All models</option></select>
  </label>
  <label>Attack type
    <select id="f-attack"><option value="">All attacks</option></select>
  </label>
  <label>Complexity
    <select id="f-complexity"><option value="">All tiers</option></select>
  </label>
  <button class="btn-reset" id="btn-reset">Reset filters</button>
</div>

<!-- Attack type bar chart -->
<section>
  <h2>Attack Type Comparison</h2>
  <div class="chart-panel">
    <h3>Average token match rate by attack type (aggregate rows only)</h3>
    <canvas id="chart-attacks" height="300"></canvas>
  </div>
</section>

<!-- Results table -->
<section>
  <h2>Results Explorer</h2>
  <div class="chart-panel">
    <div class="tbl-wrap">
      <table id="tbl">
        <thead>
          <tr>
            <th data-col="experiment_name">Experiment<span class="sort-arrow">↕</span></th>
            <th data-col="model_name">Model<span class="sort-arrow">↕</span></th>
            <th data-col="attack_type">Attack type<span class="sort-arrow">↕</span></th>
            <th data-col="complexity_tier">Complexity<span class="sort-arrow">↕</span></th>
            <th data-col="token_match_rate">Token match<span class="sort-arrow">↕</span></th>
            <th data-col="cosine_similarity">Cosine sim<span class="sort-arrow">↕</span></th>
            <th data-col="coherence">Coherence<span class="sort-arrow">↕</span></th>
            <th data-col="savings_pct">Savings %<span class="sort-arrow">↕</span></th>
            <th data-col="pass_fail">Pass/Fail<span class="sort-arrow">↕</span></th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
    <div class="pagination">
      <button id="pg-prev">← Prev</button>
      <span class="pg-info" id="pg-info"></span>
      <button id="pg-next">Next →</button>
    </div>
  </div>
</section>

</div><!-- .page -->

<script id="raw-data" type="application/json">
{{DATA_JSON}}
</script>

<script>
'use strict';

// ── data ──────────────────────────────────────────────────────────────────
const DB = JSON.parse(document.getElementById('raw-data').textContent);

const ATTACK_COLORS = {
  honest:             '#4ade80',
  threshold_sweep:    '#60a5fa',
  attention_skip:     '#f87171',
  fake_ffn:           '#fb923c',
  model_substitution: '#c084fc',
  token_tamper:       '#f472b6',
  random_mask:        '#a78bfa',
  split_verification: '#67e8f9',
  fingerprint:        '#fde68a',
};
function attackColor(t) { return ATTACK_COLORS[t] || '#8080b0'; }

// ── state ──────────────────────────────────────────────────────────────────
const state = {
  filterModel:     '',
  filterAttack:    '',
  filterComplexity:'',
  sortCol:         'id',
  sortDir:         1,   // 1 = asc, -1 = desc
  page:            0,
  PER_PAGE:        50,
};

// ── filtering ──────────────────────────────────────────────────────────────
function getFiltered() {
  return DB.results.filter(r =>
    (!state.filterModel      || r.model_name      === state.filterModel)     &&
    (!state.filterAttack     || r.attack_type     === state.filterAttack)    &&
    (!state.filterComplexity || r.complexity_tier === state.filterComplexity)
  );
}

function getAggFiltered() {
  // Aggregate rows only (no per-layer, no per-position)
  return getFiltered().filter(r => r.layer == null && r.position == null);
}

// ── canvas helper ──────────────────────────────────────────────────────────
function setupCanvas(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const dpr = window.devicePixelRatio || 1;
  const W   = el.clientWidth;
  const H   = el.clientHeight || parseInt(el.getAttribute('height'), 10) || 300;
  el.width  = W * dpr;
  el.height = H * dpr;
  const ctx = el.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, W, H };
}

// ── attack bar chart ───────────────────────────────────────────────────────
function renderAttackChart() {
  const rows = getAggFiltered().filter(r => r.token_match_rate != null);

  // Aggregate by attack_type
  const map = {};
  rows.forEach(r => {
    if (!map[r.attack_type]) map[r.attack_type] = { sum: 0, n: 0 };
    map[r.attack_type].sum += r.token_match_rate;
    map[r.attack_type].n  += 1;
  });

  // Fall back to pre-computed if no filtered data
  let items;
  if (Object.keys(map).length === 0) {
    items = DB.attackStats.map(a => ({
      label: a.attack_type,
      value: a.avg_tmr,
      color: attackColor(a.attack_type),
    }));
  } else {
    items = Object.entries(map)
      .map(([at, v]) => ({ label: at, value: v.sum / v.n, color: attackColor(at) }))
      .sort((a, b) => b.value - a.value);
  }

  if (!items.length) return;

  const c = setupCanvas('chart-attacks');
  if (!c) return;
  const { ctx, W, H } = c;

  const pad = { t: 20, r: 80, b: 36, l: 200 };
  const plotW = W - pad.l - pad.r;
  const rowH  = Math.max(20, Math.min(34, (H - pad.t - pad.b) / items.length));
  const totalH = rowH * items.length;

  // BG
  ctx.fillStyle = '#16213e';
  ctx.fillRect(0, 0, W, H);

  // Grid lines at 0, 0.25, 0.5, 0.75, 1.0
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const x = pad.l + v * plotW;
    ctx.strokeStyle = '#2a2a5a';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + totalH); ctx.stroke();
    ctx.fillStyle = '#8080b0';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText((v * 100).toFixed(0) + '%', x, pad.t + totalH + 16);
  });

  // Bars
  items.forEach((item, i) => {
    const y  = pad.t + i * rowH;
    const bw = item.value * plotW;
    const bh = rowH * 0.68;
    const by = y + (rowH - bh) / 2;

    // Track bg
    ctx.fillStyle = '#0f3460';
    ctx.fillRect(pad.l, by, plotW, bh);

    // Bar fill
    ctx.fillStyle = item.color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(pad.l, by, bw, bh);
    ctx.globalAlpha = 1;

    // Label
    ctx.fillStyle = '#e0e0f0';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText(item.label, pad.l - 8, y + rowH / 2 + 4);

    // Value
    ctx.fillStyle = '#e0e0f0';
    ctx.textAlign = 'left';
    ctx.fillText((item.value * 100).toFixed(1) + '%', pad.l + bw + 6, y + rowH / 2 + 4);
  });

  // X-axis label
  ctx.fillStyle = '#8080b0';
  ctx.font = '11px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Avg token match rate', pad.l + plotW / 2, H - 4);
}

// ── results table ──────────────────────────────────────────────────────────
function fmt(v, digits) {
  if (v == null) return '<span style="color:var(--dim)">—</span>';
  if (typeof v === 'number') return v.toFixed(digits ?? 3);
  return String(v);
}

function passBadge(pf) {
  if (pf === 1  || pf === true)  return '<span class="badge badge-pass">PASS</span>';
  if (pf === 0  || pf === false) return '<span class="badge badge-fail">FAIL</span>';
  return '<span class="badge badge-null">—</span>';
}

function rowClass(r) {
  if (r.attack_type === 'honest' || r.pass_fail === 1) return 'row-honest';
  if (r.pass_fail === 0) return 'row-fraud';
  return 'row-ambig';
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

function renderTable() {
  let rows = getFiltered();

  // Sort
  const col = state.sortCol;
  rows = [...rows].sort((a, b) => {
    const av = a[col] ?? '', bv = b[col] ?? '';
    if (av < bv) return -state.sortDir;
    if (av > bv) return  state.sortDir;
    return 0;
  });

  const total   = rows.length;
  const pages   = Math.max(1, Math.ceil(total / state.PER_PAGE));
  state.page    = Math.min(state.page, pages - 1);
  const start   = state.page * state.PER_PAGE;
  const slice   = rows.slice(start, start + state.PER_PAGE);

  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = slice.map(r => `
    <tr class="${rowClass(r)}">
      <td title="${r.experiment_name}">${truncate(r.experiment_name, 22)}</td>
      <td title="${r.model_name}">${truncate(r.model_name, 22)}</td>
      <td><span style="color:${attackColor(r.attack_type)}">${r.attack_type}</span></td>
      <td>${r.complexity_tier ?? '<span style="color:var(--dim)">—</span>'}</td>
      <td>${fmt(r.token_match_rate, 3)}</td>
      <td>${fmt(r.cosine_similarity, 4)}</td>
      <td>${r.coherence ?? '<span style="color:var(--dim)">—</span>'}</td>
      <td>${fmt(r.savings_pct, 2)}</td>
      <td>${passBadge(r.pass_fail)}</td>
    </tr>
  `).join('');

  document.getElementById('pg-info').textContent =
    `${start + 1}–${Math.min(start + state.PER_PAGE, total)} of ${total.toLocaleString()}`;
  document.getElementById('pg-prev').disabled = state.page === 0;
  document.getElementById('pg-next').disabled = state.page >= pages - 1;
}

// ── populate filters ───────────────────────────────────────────────────────
function populateFilters() {
  function opts(sel, values) {
    const el = document.getElementById(sel);
    const first = el.options[0];
    el.innerHTML = '';
    el.appendChild(first);
    values.forEach(v => {
      const o = document.createElement('option');
      o.value = o.textContent = v;
      el.appendChild(o);
    });
  }

  const models     = [...new Set(DB.results.map(r => r.model_name))].sort();
  const attacks    = [...new Set(DB.results.map(r => r.attack_type))].sort();
  const complexity = [...new Set(DB.results.map(r => r.complexity_tier).filter(Boolean))].sort();

  opts('f-model',      models);
  opts('f-attack',     attacks);
  opts('f-complexity', complexity);
}

// ── update sort arrows ─────────────────────────────────────────────────────
function updateSortArrows() {
  document.querySelectorAll('#tbl th').forEach(th => {
    const arrow = th.querySelector('.sort-arrow');
    if (!arrow) return;
    if (th.dataset.col === state.sortCol) {
      arrow.textContent = state.sortDir === 1 ? ' ↑' : ' ↓';
      arrow.style.opacity = '1';
    } else {
      arrow.textContent = ' ↕';
      arrow.style.opacity = '0.4';
    }
  });
}

// ── render all ────────────────────────────────────────────────────────────
function renderAll() {
  renderAttackChart();
  renderTable();
  updateSortArrows();
}

// ── events ─────────────────────────────────────────────────────────────────
document.getElementById('f-model').addEventListener('change', e => {
  state.filterModel = e.target.value; state.page = 0; renderAll();
});
document.getElementById('f-attack').addEventListener('change', e => {
  state.filterAttack = e.target.value; state.page = 0; renderAll();
});
document.getElementById('f-complexity').addEventListener('change', e => {
  state.filterComplexity = e.target.value; state.page = 0; renderAll();
});
document.getElementById('btn-reset').addEventListener('click', () => {
  state.filterModel = state.filterAttack = state.filterComplexity = '';
  state.page = 0;
  document.getElementById('f-model').value      = '';
  document.getElementById('f-attack').value     = '';
  document.getElementById('f-complexity').value = '';
  renderAll();
});
document.getElementById('pg-prev').addEventListener('click', () => {
  state.page--; renderTable();
});
document.getElementById('pg-next').addEventListener('click', () => {
  state.page++; renderTable();
});

// Column sort
document.querySelectorAll('#tbl th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    if (state.sortCol === th.dataset.col) {
      state.sortDir *= -1;
    } else {
      state.sortCol = th.dataset.col;
      state.sortDir = 1;
    }
    state.page = 0;
    renderAll();
  });
});

// Redraw chart on resize
let _rsz;
window.addEventListener('resize', () => {
  clearTimeout(_rsz);
  _rsz = setTimeout(renderAttackChart, 120);
});

// ── init ──────────────────────────────────────────────────────────────────
(function init() {
  const m = DB.meta;
  document.getElementById('card-exp').textContent     = m.total_experiments.toLocaleString();
  document.getElementById('card-models').textContent  = m.total_models.toLocaleString();
  document.getElementById('card-results').textContent = m.total_results.toLocaleString();
  document.getElementById('card-prompts').textContent = m.total_prompts.toLocaleString();
  document.getElementById('ts').textContent = 'Generated ' + m.generated_at;

  populateFilters();
  renderAll();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_html(data: dict) -> str:
    data_json = json.dumps(data, separators=(",", ":"), default=str, ensure_ascii=False)
    return _TEMPLATE.replace("{{DATA_JSON}}", data_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate analysis_results/dashboard.html from results.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--db",      default=str(DEFAULT_DB),  help="Path to results.db.")
    p.add_argument("--out",     default=str(DEFAULT_OUT), help="Output HTML path.")
    p.add_argument("--no-open", action="store_true",      help="Do not open browser.")
    args = p.parse_args()

    db_path  = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Querying {db_path} …", end=" ", flush=True)
    data = query_data(db_path)
    print(f"{data['meta']['total_results']:,} results loaded.")

    print(f"Writing {out_path} …", end=" ", flush=True)
    html = build_html(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"done ({size_kb:,} KB).")

    if not args.no_open:
        uri = out_path.resolve().as_uri()
        print(f"Opening {uri}")
        webbrowser.open(uri)


if __name__ == "__main__":
    main()
