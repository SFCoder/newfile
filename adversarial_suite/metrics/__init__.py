"""
metrics/ — pure metric computation and reporting functions.

Functions have no dependencies on how attacks were performed or how
verification was done.  They operate on token lists, logit lists, and
result dicts.

Submodules
----------
compute    — token_match_rate, cosine_similarity, perplexity, coherence,
             attacker_savings_pct, summarise
reporting  — print_summary_table, print_attacker_optimal, save_results_json,
             save_summary_csv, save_attacker_optimal_csv, plot_model_comparison
"""

from adversarial_suite.metrics import compute, reporting

__all__ = ["compute", "reporting"]
