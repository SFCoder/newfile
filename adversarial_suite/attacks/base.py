"""
Base dataclass shared by all attack implementations.

Every attack module must return an AttackResult so that the verification
and metrics layers can process results without knowing which specific
attack was used.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root on sys.path so verifier / provider can be imported.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifier import VerificationBundle  # noqa: E402


@dataclass
class AttackResult:
    """
    Standardised result returned by every attack implementation.

    Fields
    ------
    honest_tokens
        Token IDs produced by unmodified honest inference.
    fraudulent_tokens
        Token IDs produced by the attack (the output the provider claims).
    fraudulent_neuron_masks
        Per-layer neuron masks recorded *during the attack run*.
        Keys are zero-padded layer indices (same convention as
        VerificationBundle.layer_key).  None for attacks that do not
        intercept FFN activations.
    metadata
        Attack-specific information: which layers were skipped, what
        substitution was used, seeds, strategies, etc.
    ground_truth_bundle
        A VerificationBundle built from the honest run.  Stored so that
        callers can compare the fraudulent bundle against the honest one
        without re-running honest inference.
    honest_logits
        Logits captured at each generation step during the honest run,
        shape (num_tokens, vocab_size) as a CPU float tensor or list of
        per-step logit lists.  None if not captured.
    fraudulent_logits
        Logits captured at each generation step during the attack run,
        same format as honest_logits.  None if not captured.
    """

    honest_tokens: list
    fraudulent_tokens: list
    fraudulent_neuron_masks: Optional[dict]
    metadata: dict
    ground_truth_bundle: Optional[VerificationBundle]
    honest_logits: Optional[list] = field(default=None)
    fraudulent_logits: Optional[list] = field(default=None)
