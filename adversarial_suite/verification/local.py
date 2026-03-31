"""
LocalVerification
=================
Verification backend that runs the verifier engine directly in-process,
using whatever model and tokenizer are passed in.

This is the backend used for all local experiments.  To run the same
experiment against a live Cosmos testnet, swap LocalVerification for
CosmosVerification — the interface is identical.

Usage::

    from adversarial_suite.verification.local import LocalVerification

    target  = LocalVerification(model, tokenizer)
    result  = target.verify(bundle)
    print(result.verified, result.token_match_rate)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifier import VerificationBundle, VerificationResult, verify  # noqa: E402
from adversarial_suite.verification.base import VerificationTarget     # noqa: E402


class LocalVerification(VerificationTarget):
    """
    Runs verify() from verifier.py directly with a pre-loaded model.

    The model and tokenizer must already be loaded; LocalVerification
    does not manage model lifecycle.  Use ModelContext from demo.py (or
    your own context manager) to load and unload models safely.

    Parameters
    ----------
    model
        A loaded HuggingFace CausalLM model, placed on the desired device.
    tokenizer
        The corresponding tokenizer.
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def verify(self, bundle: VerificationBundle) -> VerificationResult:
        """
        Replay the bundle's claimed computation using the loaded model and
        compare replayed tokens against the bundle's claimed tokens.

        Returns a VerificationResult with verified=True iff the replayed
        tokens exactly match bundle.output_token_ids.
        """
        return verify(bundle, self._model, self._tokenizer)
