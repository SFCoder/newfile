"""
Pure metric computation functions.

All functions in this module are pure: they depend only on their
arguments and produce no side effects.  They have no knowledge of how
the attack was performed, which model was used, or how verification was
carried out.

Functions
---------
token_match_rate          — fraction of positions where tokens agree
cosine_similarity_scores  — per-position cosine similarity of logit vectors
mean_cosine_similarity    — mean of cosine_similarity_scores
perplexity_from_log_probs — exponentiated mean negative log probability
classify_coherence        — "coherent" / "degraded" / "garbage"
attacker_savings_pct      — estimated FLOP savings from skipping N layers
summarise                 — convenience wrapper returning a metrics dict
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------


def token_match_rate(honest_tokens: list, fraudulent_tokens: list) -> float:
    """
    Fraction of token positions where honest and fraudulent tokens agree.

    Positions beyond the shorter sequence are counted as mismatches.

    Parameters
    ----------
    honest_tokens, fraudulent_tokens
        Lists of integer token IDs (same length expected but not required).

    Returns
    -------
    float in [0, 1].  1.0 means the outputs are identical.
    """
    if not honest_tokens:
        return 0.0
    n = len(honest_tokens)
    matches = sum(
        1 for i in range(n)
        if i < len(fraudulent_tokens) and honest_tokens[i] == fraudulent_tokens[i]
    )
    return matches / n


def classify_coherence(match_rate: float) -> str:
    """
    Classify attack output quality based on token match rate.

    Thresholds (per spec)
    ---------------------
    ≥ 0.80  → "coherent"   (attack is nearly undetectable)
    0.20–0.80 → "degraded"  (output is noticeably wrong)
    < 0.20  → "garbage"    (output is obviously incoherent)

    Parameters
    ----------
    match_rate
        Output of token_match_rate().

    Returns
    -------
    One of "coherent", "degraded", "garbage".
    """
    if match_rate >= 0.80:
        return "coherent"
    elif match_rate >= 0.20:
        return "degraded"
    else:
        return "garbage"


# ---------------------------------------------------------------------------
# Logit-based metrics
# ---------------------------------------------------------------------------


def cosine_similarity_scores(
    honest_logits: list,
    attack_logits: list,
) -> list:
    """
    Per-position cosine similarity between honest and attack logit vectors.

    Parameters
    ----------
    honest_logits, attack_logits
        Each is a list of N vectors (one per generated token position).
        Each vector is a 1-D sequence of floats with length vocab_size.
        Accepts both plain Python lists and torch tensors.

    Returns
    -------
    List of N floats in [-1, 1], one per position.
    """
    import torch

    results = []
    for h, a in zip(honest_logits, attack_logits):
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        h = h.float()
        a = a.float()
        norm_h = torch.norm(h)
        norm_a = torch.norm(a)
        if norm_h == 0 or norm_a == 0:
            results.append(0.0)
        else:
            results.append((torch.dot(h, a) / (norm_h * norm_a)).item())
    return results


def mean_cosine_similarity(
    honest_logits: list,
    attack_logits: list,
) -> float:
    """
    Mean cosine similarity across all generated token positions.

    Returns 0.0 if either list is empty.
    """
    scores = cosine_similarity_scores(honest_logits, attack_logits)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def perplexity_from_log_probs(log_probs: list) -> float:
    """
    Compute perplexity from a list of per-token log probabilities.

    Perplexity = exp(− mean(log_probs)).

    Parameters
    ----------
    log_probs
        List of natural-log probabilities assigned by the model to each
        generated token.  Typically negative floats.

    Returns
    -------
    float ≥ 1.0.  Lower perplexity = model considers the sequence more
    likely = output is more coherent.  Returns inf if log_probs is empty.
    """
    if not log_probs:
        return float("inf")
    mean_nll = -sum(log_probs) / len(log_probs)
    return math.exp(mean_nll)


def compute_log_probs_from_logits(
    logits: list,
    token_ids: list,
) -> list:
    """
    Compute per-token log probabilities given a list of logit vectors.

    Parameters
    ----------
    logits
        List of N vectors (one per position).  logits[i] is the logit
        vector that predicts token_ids[i].
    token_ids
        List of N integer token IDs whose probabilities we want.

    Returns
    -------
    List of N log-probability floats.
    """
    import torch
    import torch.nn.functional as F

    log_probs = []
    for logit_vec, tok_id in zip(logits, token_ids):
        if not isinstance(logit_vec, torch.Tensor):
            logit_vec = torch.tensor(logit_vec, dtype=torch.float32)
        lp = F.log_softmax(logit_vec.float(), dim=-1)
        log_probs.append(lp[tok_id].item())
    return log_probs


# ---------------------------------------------------------------------------
# Computation savings
# ---------------------------------------------------------------------------


def attacker_savings_pct(num_skipped: int, total_layers: int) -> float:
    """
    Estimated percentage of total inference FLOPs saved by skipping
    attention at *num_skipped* out of *total_layers* layers.

    Assumption: attention accounts for roughly 1/3 of per-layer compute,
    so skipping attention at N layers saves N/total_layers × 33 % of
    total compute.

    Parameters
    ----------
    num_skipped
        Number of layers at which attention is zeroed.
    total_layers
        Total number of transformer layers in the model.

    Returns
    -------
    Float in [0, 100].
    """
    if total_layers == 0:
        return 0.0
    return (num_skipped / total_layers) * 33.0


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def summarise(
    honest_tokens: list,
    fraudulent_tokens: list,
    num_skipped: int,
    total_layers: int,
    honest_logits: Optional[list] = None,
    attack_logits: Optional[list] = None,
    honest_logits_for_ppl: Optional[list] = None,
) -> dict:
    """
    Compute all standard metrics and return them as a flat dict.

    Parameters
    ----------
    honest_tokens, fraudulent_tokens
        Token ID lists.
    num_skipped
        Number of attention layers skipped.
    total_layers
        Total model layers.
    honest_logits, attack_logits
        Optional per-position logit vectors (for cosine similarity).
        If either is None, cosine similarity is reported as None.
    honest_logits_for_ppl
        Optional per-position logit vectors from the *honest* model scored
        against the fraudulent token sequence (for perplexity).
        If None, perplexity is reported as None.

    Returns
    -------
    dict with keys: token_match_rate, coherence, savings_pct,
    mean_cosine_similarity (or None), perplexity (or None).
    """
    match_rate = token_match_rate(honest_tokens, fraudulent_tokens)

    cos_sim: Optional[float] = None
    if honest_logits is not None and attack_logits is not None:
        cos_sim = mean_cosine_similarity(honest_logits, attack_logits)

    ppl: Optional[float] = None
    if honest_logits_for_ppl is not None:
        lps = compute_log_probs_from_logits(honest_logits_for_ppl, fraudulent_tokens)
        ppl = perplexity_from_log_probs(lps)

    return {
        "token_match_rate": match_rate,
        "coherence": classify_coherence(match_rate),
        "savings_pct": attacker_savings_pct(num_skipped, total_layers),
        "mean_cosine_similarity": cos_sim,
        "perplexity": ppl,
    }
