"""
CosmosVerification  (stub)
===========================
Verification backend that submits a VerificationBundle to a Cosmos-SDK
testnet via gRPC and returns the on-chain verification result.

This stub defines the interface so that experiments written against
LocalVerification can be trivially upgraded to on-chain verification by
swapping the backend.  The implementation requires a deployed proof-of-
inference Cosmos module with a SubmitBundle transaction type and a
QueryVerificationResult query method.

Planned implementation
----------------------
1. Serialise the bundle to JSON (bundle.canonical_json()).
2. Build a MsgSubmitBundle protobuf message with:
     - submitter:  the caller's bech32 address
     - bundle_json: the canonical JSON bytes
     - bundle_hash: bundle.content_hash()
3. Sign and broadcast the transaction via the Cosmos gRPC endpoint.
4. Poll QueryVerificationResult(bundle_hash) until the on-chain
   verifier returns a result (or a timeout elapses).
5. Decode the protobuf response into a VerificationResult and return it.

The on-chain verifier runs the same replay_with_masks() logic as the
local verifier, but the model weights are loaded from an on-chain
content-addressed store (IPFS CID anchored by the model registry
smart contract).

Usage (once implemented)::

    from adversarial_suite.verification.cosmos import CosmosVerification

    target = CosmosVerification(
        node_url="http://testnet.example.com:9090",
        chain_id="proof-of-inference-testnet-1",
        wallet_mnemonic="...",
    )
    result = target.verify(bundle)

To swap from local to on-chain, change one line in max_savings_test.py::

    # Before (local):
    target = LocalVerification(model, tokenizer)

    # After (on-chain):
    target = CosmosVerification(node_url=NODE_URL, chain_id=CHAIN_ID, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifier import VerificationBundle, VerificationResult  # noqa: E402
from adversarial_suite.verification.base import VerificationTarget  # noqa: E402


class CosmosVerification(VerificationTarget):
    """
    Stub for Cosmos-testnet verification backend.

    Submits a VerificationBundle to the on-chain verifier via gRPC and
    returns the VerificationResult produced by the chain.

    Not yet implemented.  See module docstring for the planned design.
    """

    def __init__(
        self,
        node_url: str,
        chain_id: str,
        wallet_mnemonic: str = "",
        timeout_seconds: int = 120,
    ):
        self.node_url = node_url
        self.chain_id = chain_id
        self.wallet_mnemonic = wallet_mnemonic
        self.timeout_seconds = timeout_seconds

    def verify(self, bundle: VerificationBundle) -> VerificationResult:
        raise NotImplementedError(
            "CosmosVerification is not yet implemented.  "
            "Use LocalVerification for local experiments.  "
            "See module docstring for the planned on-chain interface."
        )
