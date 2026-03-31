"""
verification/ — verification backends.

All backends implement VerificationTarget.verify(bundle) -> VerificationResult.
Swapping from local to on-chain verification requires changing one line.

Available backends
------------------
LocalVerification   — runs verify() directly in-process (implemented)
CosmosVerification  — submits to a Cosmos testnet via gRPC (stub)
"""

from adversarial_suite.verification.base import VerificationTarget
from adversarial_suite.verification.local import LocalVerification
from adversarial_suite.verification.cosmos import CosmosVerification

__all__ = [
    "VerificationTarget",
    "LocalVerification",
    "CosmosVerification",
]
