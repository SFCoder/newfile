"""
Test suite for the sparse-replay verification system.

Loads Qwen/Qwen2.5-7B once per session (session-scoped fixture) and runs all
tests against the already-loaded singleton.  Core verify() calls go directly to
verifier.py — no HTTP round-trip.  The HTTP layer is tested separately via
TestClient in TestHttpLayer.

Test matrix
-----------
TestNeuronMask (7 tests)
  - round-trip: from_indices → to_indices preserves the set exactly
  - empty() produces a mask with active_count() == 0
  - bitset encoding produces the expected base64 string for a known input
  - bits length matches ceil(intermediate_size / 8)
  - density() reports the correct fraction
  - invalid index raises ValueError
  - mismatched bits length raises ValidationError

TestBundleSchema (3 tests)
  - canonical_json() is deterministic across two calls on the same object
  - content_hash() changes when any field changes
  - layer_key() zero-pads so lexicographic sort == numeric sort

TestHonestBundlesVerify (4 tests)
  - honest bundles return verified=True and token_match_rate == 1.0
  - replayed_token_ids exactly equals output_token_ids
  - first_mismatch_position is None
  - bundle_hash in result matches bundle.content_hash()

TestTamperedTokensFail (6 tests)
  - 1-token shift → verified=False
  - 5-token shift → verified=False, first_mismatch_position is not None
  - match_rate upper-bounded by (n - num_changes) / n

TestTamperedMasksFail (6 tests)
  - zero masks        → verified=False
  - random masks      → verified=False
  - 5%-sparse masks   → verified=False

TestEdgeCases (2 tests)
  - empty output_token_ids → verified=False, no crash
  - content_hash is a 64-character hex string

TestHttpLayer (4 tests)
  - GET /health → 200 with expected keys
  - POST /verify with honest bundle → verified=True
  - POST /verify with wrong model_name → 400
  - POST /verify with tampered tokens → verified=False
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from verifier import NeuronMask, VerificationBundle, VerificationResult, verify
from verification_api import SUPPORTED_MODEL, load_model, app
from provider import (
    generate_honest_bundle,
    tamper_masks_random,
    tamper_masks_sparse,
    tamper_masks_zero,
    tamper_tokens,
)

# ---------------------------------------------------------------------------
# Prompts shared across tests
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
]
NUM_TOKENS = 15


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def loaded_model():
    return load_model(SUPPORTED_MODEL)


@pytest.fixture(scope="session")
def honest_bundles(loaded_model):
    model, tokenizer = loaded_model
    return {
        prompt: generate_honest_bundle(
            prompt, num_tokens=NUM_TOKENS, model=model, tokenizer=tokenizer
        )
        for prompt in TEST_PROMPTS
    }


# ---------------------------------------------------------------------------
# TestNeuronMask
# ---------------------------------------------------------------------------


class TestNeuronMask:
    def test_round_trip(self):
        indices = [0, 1, 100, 999, 18943]
        mask = NeuronMask.from_indices(indices, intermediate_size=18944)
        assert sorted(mask.to_indices()) == sorted(indices)

    def test_empty_mask(self):
        mask = NeuronMask.empty(intermediate_size=18944)
        assert mask.active_count() == 0
        assert mask.to_indices() == []

    def test_bits_length(self):
        import base64
        for size in [8, 9, 100, 18944]:
            mask = NeuronMask.empty(size)
            decoded = base64.b64decode(mask.bits)
            assert len(decoded) == (size + 7) // 8

    def test_known_encoding(self):
        # Neuron 0 set → byte 0 = 0b00000001 = 1 → base64 of b'\x01\x00'
        mask = NeuronMask.from_indices([0], intermediate_size=16)
        import base64
        buf = base64.b64decode(mask.bits)
        assert buf[0] == 0b00000001
        assert buf[1] == 0x00

    def test_density(self):
        mask = NeuronMask.from_indices([0, 1, 2, 3], intermediate_size=8)
        assert mask.density() == 0.5

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            NeuronMask.from_indices([18944], intermediate_size=18944)

    def test_mismatched_bits_length_raises(self):
        import base64
        bad_bits = base64.b64encode(b"\x00").decode()  # 1 byte, but size=18944 needs 2368
        with pytest.raises(ValidationError):
            NeuronMask(intermediate_size=18944, bits=bad_bits)


# ---------------------------------------------------------------------------
# TestBundleSchema
# ---------------------------------------------------------------------------


class TestBundleSchema:
    def _minimal_bundle(self) -> VerificationBundle:
        mask = NeuronMask.from_indices([0, 1, 2], intermediate_size=16)
        return VerificationBundle(
            model_name=SUPPORTED_MODEL,
            prompt="Hello",
            output_token_ids=[1, 2, 3],
            neuron_masks={VerificationBundle.layer_key(0): mask},
        )

    def test_canonical_json_is_deterministic(self):
        bundle = self._minimal_bundle()
        assert bundle.canonical_json() == bundle.canonical_json()

    def test_content_hash_changes_with_tokens(self):
        bundle = self._minimal_bundle()
        tampered = tamper_tokens(bundle, num_changes=1)
        assert bundle.content_hash() != tampered.content_hash()

    def test_layer_key_sort_order(self):
        keys = [VerificationBundle.layer_key(i) for i in range(100)]
        assert keys == sorted(keys)  # lexicographic == numeric for zero-padded keys


# ---------------------------------------------------------------------------
# TestHonestBundlesVerify
# ---------------------------------------------------------------------------


class TestHonestBundlesVerify:
    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_verified_true(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(honest_bundles[prompt], model, tokenizer)
        assert result.verified, (
            f"Honest bundle failed for {prompt!r}\n"
            f"Claimed:  {honest_bundles[prompt].output_token_ids}\n"
            f"Replayed: {result.replayed_token_ids}"
        )

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_full_token_match(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(honest_bundles[prompt], model, tokenizer)
        assert result.token_match_rate == 1.0
        assert result.replayed_token_ids == honest_bundles[prompt].output_token_ids
        assert result.first_mismatch_position is None

    def test_bundle_hash_in_result(self, loaded_model, honest_bundles):
        model, tokenizer = loaded_model
        bundle = honest_bundles[TEST_PROMPTS[0]]
        result = verify(bundle, model, tokenizer)
        assert result.bundle_hash == bundle.content_hash()


# ---------------------------------------------------------------------------
# TestTamperedTokensFail
# ---------------------------------------------------------------------------


class TestTamperedTokensFail:
    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_single_token_change_fails(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(tamper_tokens(honest_bundles[prompt], num_changes=1), model, tokenizer)
        assert not result.verified
        assert result.token_match_rate < 1.0

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_multiple_token_changes_fail(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(tamper_tokens(honest_bundles[prompt], num_changes=5), model, tokenizer)
        assert not result.verified
        assert result.first_mismatch_position is not None

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_match_rate_upper_bound(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        num_changes = 3
        n = len(honest_bundles[prompt].output_token_ids)
        result = verify(
            tamper_tokens(honest_bundles[prompt], num_changes=num_changes), model, tokenizer
        )
        assert result.token_match_rate <= (n - num_changes) / n + 1e-6


# ---------------------------------------------------------------------------
# TestTamperedMasksFail
# ---------------------------------------------------------------------------


class TestTamperedMasksFail:
    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_zero_masks_fail(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(tamper_masks_zero(honest_bundles[prompt]), model, tokenizer)
        assert not result.verified

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_random_masks_fail(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(tamper_masks_random(honest_bundles[prompt], seed=42), model, tokenizer)
        assert not result.verified

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_sparse_masks_fail(self, loaded_model, honest_bundles, prompt):
        model, tokenizer = loaded_model
        result = verify(tamper_masks_sparse(honest_bundles[prompt], keep_fraction=0.05), model, tokenizer)
        assert not result.verified


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_output_tokens(self, loaded_model):
        model, tokenizer = loaded_model
        bundle = VerificationBundle(
            model_name=SUPPORTED_MODEL,
            prompt="Hello",
            output_token_ids=[],
            neuron_masks={},
        )
        result = verify(bundle, model, tokenizer)
        assert not result.verified
        assert result.token_match_rate == 0.0

    def test_content_hash_is_hex_sha256(self, honest_bundles):
        h = honest_bundles[TEST_PROMPTS[0]].content_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# TestHttpLayer
# ---------------------------------------------------------------------------


class TestHttpLayer:
    @pytest.fixture(scope="class")
    def client(self, loaded_model):
        with TestClient(app) as c:
            yield c

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "device" in body
        assert "num_layers" in body
        assert "intermediate_size" in body

    def test_honest_bundle_verifies_over_http(self, client, honest_bundles):
        bundle = honest_bundles[TEST_PROMPTS[0]]
        resp = client.post("/verify", json=bundle.model_dump())
        assert resp.status_code == 200
        body = resp.json()
        assert body["verified"] is True
        assert body["token_match_rate"] == 1.0

    def test_wrong_model_name_returns_400(self, client):
        mask = NeuronMask.from_indices([0, 1], intermediate_size=16)
        bundle = VerificationBundle(
            model_name="openai/gpt-4",
            prompt="Hello",
            output_token_ids=[1, 2, 3],
            neuron_masks={VerificationBundle.layer_key(0): mask},
        )
        resp = client.post("/verify", json=bundle.model_dump())
        assert resp.status_code == 400
        assert "only supports" in resp.json()["detail"].lower()

    def test_tampered_tokens_fail_over_http(self, client, honest_bundles):
        tampered = tamper_tokens(honest_bundles[TEST_PROMPTS[0]], num_changes=3)
        resp = client.post("/verify", json=tampered.model_dump())
        assert resp.status_code == 200
        assert resp.json()["verified"] is False
