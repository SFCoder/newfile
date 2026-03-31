"""
VerificationTarget — abstract base class for all verification backends.

Every verification backend must implement verify(bundle) -> VerificationResult.
The interface is designed so that swapping from local verification to
on-chain Cosmos verification requires changing exactly one line in the
calling code.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifier import VerificationBundle, VerificationResult  # noqa: E402


class VerificationTarget(ABC):
    """
    Abstract verification backend.

    Implementations
    ---------------
    LocalVerification   — loads the model and calls verify() directly.
    CosmosVerification  — submits the bundle to a Cosmos testnet via gRPC
                          and returns the on-chain result (stub).

    To swap backends in an experiment, replace::

        target = LocalVerification(model, tokenizer)

    with::

        target = CosmosVerification(node_url="http://...", chain_id="...")
    """

    @abstractmethod
    def verify(self, bundle: VerificationBundle) -> VerificationResult:
        """
        Verify a VerificationBundle and return the result.

        Parameters
        ----------
        bundle
            The bundle to verify.  Typically constructed by the attack
            module from fraudulent tokens and FFN masks.

        Returns
        -------
        VerificationResult
            Contains verified (bool), token_match_rate, replayed tokens,
            and other details.
        """
        ...
