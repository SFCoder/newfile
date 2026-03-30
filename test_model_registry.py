"""
Tests for model_registry.py

Covers:
  TestRegistryJSON          — registry.json is valid and all entries loadable
  TestListAndGet            — list_models(), get_entry(), unknown model error
  TestWeightHash            — compute_weight_hash() matches stored values
  TestVerifyWeights         — verify_weights() passes for real models
  TestLoadVerifiedModel     — load succeeds for registered model; tampered hash raises
  TestRegisterNewModel      — register_new_model() computes and persists correct hash
  TestBackendAbstraction    — _load_registry / _save_registry round-trip cleanly

Design notes
------------
- Tests that actually load model weights (TestLoadVerifiedModel) are marked
  @pytest.mark.slow and skipped unless -m slow is passed, because loading a
  7B model in every CI run is impractical.
- Hash and registration tests run without loading model weights (fast).
- The tampered-hash test patches the registry entry in-memory so it never
  touches the real registry.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from model_registry import (
    ModelEntry,
    ModelRegistry,
    DEFAULT_REGISTRY_PATH,
    get_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REAL_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]

# Known-good hashes matching the values computed when registry.json was built.
# If these change, the weights on disk have changed — that's the point.
EXPECTED_HASHES = {
    "Qwen/Qwen2.5-0.5B": "d69088004383b22aee2d56a7d88238eb39f7add7c999f92da046fda0c96557dd",
    "Qwen/Qwen2.5-3B":   "71fd74069d690809c5a2788e96e3a6d35ee020c128b59a6d774534931058d408",
    "Qwen/Qwen2.5-7B":   "b21b4b47cdb2bab55317204504a5b1651789462e5d9d5d649d05890df335f504",
}

EXPECTED_ARCH = {
    "Qwen/Qwen2.5-0.5B": {"num_layers": 24, "intermediate_size": 4864,  "hidden_size": 896},
    "Qwen/Qwen2.5-3B":   {"num_layers": 36, "intermediate_size": 11008, "hidden_size": 2048},
    "Qwen/Qwen2.5-7B":   {"num_layers": 28, "intermediate_size": 18944, "hidden_size": 3584},
}


# ---------------------------------------------------------------------------
# TestRegistryJSON
# ---------------------------------------------------------------------------


class TestRegistryJSON:
    """registry.json must be valid, complete, and self-consistent."""

    def test_file_exists(self):
        assert DEFAULT_REGISTRY_PATH.exists(), (
            f"registry.json not found at {DEFAULT_REGISTRY_PATH}"
        )

    def test_valid_json(self):
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        assert "schema_version" in data
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_all_three_models_present(self):
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        for model_id in REAL_MODELS:
            assert model_id in data["models"], (
                f"{model_id!r} missing from registry.json"
            )

    def test_required_fields_present(self):
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        required = {"model_id", "hf_repo", "weight_hash", "num_layers",
                    "intermediate_size", "hidden_size", "min_stake"}
        for model_id, entry in data["models"].items():
            missing = required - set(entry)
            assert not missing, (
                f"{model_id!r} is missing fields: {missing}"
            )

    def test_weight_hash_format(self):
        """Every weight_hash must be a 64-char lowercase hex string."""
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        for model_id, entry in data["models"].items():
            h = entry["weight_hash"]
            assert len(h) == 64, f"{model_id}: hash length {len(h)} != 64"
            assert all(c in "0123456789abcdef" for c in h), (
                f"{model_id}: hash contains non-hex characters"
            )

    def test_stored_hashes_match_expected(self):
        """Hashes in registry.json must match the values computed when it was built."""
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        for model_id, expected_hash in EXPECTED_HASHES.items():
            actual = data["models"][model_id]["weight_hash"]
            assert actual == expected_hash, (
                f"{model_id}: stored hash {actual!r} != expected {expected_hash!r}"
            )

    def test_architecture_metadata_correct(self):
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        for model_id, arch in EXPECTED_ARCH.items():
            entry = data["models"][model_id]
            for field, value in arch.items():
                assert entry[field] == value, (
                    f"{model_id}.{field}: got {entry[field]}, expected {value}"
                )

    def test_min_stake_is_positive_int(self):
        with open(DEFAULT_REGISTRY_PATH) as fh:
            data = json.load(fh)
        for model_id, entry in data["models"].items():
            stake = entry["min_stake"]
            assert isinstance(stake, int) and stake >= 0, (
                f"{model_id}: min_stake must be a non-negative int, got {stake!r}"
            )


# ---------------------------------------------------------------------------
# TestListAndGet
# ---------------------------------------------------------------------------


class TestListAndGet:
    @pytest.fixture
    def registry(self):
        return ModelRegistry(DEFAULT_REGISTRY_PATH)

    def test_list_models_returns_all_three(self, registry):
        models = registry.list_models()
        for m in REAL_MODELS:
            assert m in models

    def test_get_entry_returns_model_entry(self, registry):
        entry = registry.get_entry("Qwen/Qwen2.5-7B")
        assert isinstance(entry, ModelEntry)
        assert entry.model_id == "Qwen/Qwen2.5-7B"

    def test_get_entry_fields(self, registry):
        for model_id, arch in EXPECTED_ARCH.items():
            entry = registry.get_entry(model_id)
            assert entry.num_layers == arch["num_layers"]
            assert entry.intermediate_size == arch["intermediate_size"]
            assert entry.hidden_size == arch["hidden_size"]
            assert entry.weight_hash == EXPECTED_HASHES[model_id]

    def test_unknown_model_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get_entry("openai/gpt-4")

    def test_key_error_message_lists_known_models(self, registry):
        with pytest.raises(KeyError) as exc_info:
            registry.get_entry("mystery/model")
        assert "Qwen" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestWeightHash
# ---------------------------------------------------------------------------


class TestWeightHash:
    """
    compute_weight_hash() must be deterministic and match registry.json values.
    These tests read the local cache but do NOT load model weights into memory.
    """

    @pytest.mark.parametrize("model_id", REAL_MODELS)
    def test_hash_matches_registry(self, model_id):
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        computed = ModelRegistry.compute_weight_hash(
            registry.get_entry(model_id).hf_repo
        )
        assert computed == EXPECTED_HASHES[model_id], (
            f"{model_id}: computed {computed!r} != expected {EXPECTED_HASHES[model_id]!r}"
        )

    @pytest.mark.parametrize("model_id", REAL_MODELS)
    def test_hash_is_deterministic(self, model_id):
        """Two calls on the same files must return the same hash."""
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        hf_repo = registry.get_entry(model_id).hf_repo
        h1 = ModelRegistry.compute_weight_hash(hf_repo)
        h2 = ModelRegistry.compute_weight_hash(hf_repo)
        assert h1 == h2

    def test_unknown_repo_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ModelRegistry.compute_weight_hash("not-a-real/model")


# ---------------------------------------------------------------------------
# TestVerifyWeights
# ---------------------------------------------------------------------------


class TestVerifyWeights:
    @pytest.mark.parametrize("model_id", REAL_MODELS)
    def test_real_weights_verify(self, model_id):
        """verify_weights() must return True for all registered models."""
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        assert registry.verify_weights(model_id) is True

    def test_tampered_hash_fails_verify(self):
        """A registry entry with a wrong hash must cause verify_weights() to return False."""
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        # Patch the in-memory entry only — never touches registry.json
        real_entry = registry.get_entry("Qwen/Qwen2.5-0.5B")
        bad_entry = ModelEntry(
            model_id=real_entry.model_id,
            hf_repo=real_entry.hf_repo,
            weight_hash="a" * 64,  # obviously wrong
            num_layers=real_entry.num_layers,
            intermediate_size=real_entry.intermediate_size,
            hidden_size=real_entry.hidden_size,
            min_stake=real_entry.min_stake,
        )
        registry._entries["Qwen/Qwen2.5-0.5B"] = bad_entry
        assert registry.verify_weights("Qwen/Qwen2.5-0.5B") is False

    def test_unknown_model_raises(self):
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        with pytest.raises(KeyError):
            registry.verify_weights("fake/model")


# ---------------------------------------------------------------------------
# TestLoadVerifiedModel
# ---------------------------------------------------------------------------


class TestLoadVerifiedModel:
    """
    Tests that actually load model weights are slow — run with -m slow.
    The tampered-hash test is fast because it raises before touching weights.
    """

    def test_tampered_hash_raises_before_loading(self):
        """
        load_verified_model() must raise ValueError if the registered hash
        doesn't match the on-disk hash.  The model must NOT be loaded.
        """
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        real_entry = registry.get_entry("Qwen/Qwen2.5-0.5B")
        bad_entry = ModelEntry(
            model_id=real_entry.model_id,
            hf_repo=real_entry.hf_repo,
            weight_hash="b" * 64,
            num_layers=real_entry.num_layers,
            intermediate_size=real_entry.intermediate_size,
            hidden_size=real_entry.hidden_size,
            min_stake=real_entry.min_stake,
        )
        registry._entries["Qwen/Qwen2.5-0.5B"] = bad_entry

        with pytest.raises(ValueError, match="Weight hash mismatch"):
            registry.load_verified_model("Qwen/Qwen2.5-0.5B")

    def test_unregistered_model_raises_key_error(self):
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        with pytest.raises(KeyError, match="not registered"):
            registry.load_verified_model("fantasy/model-1B")

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-0.5B"])
    def test_load_verified_model_succeeds(self, model_id):
        """Hash-verified load returns a working (model, tokenizer) pair."""
        import torch
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        model, tokenizer = registry.load_verified_model(model_id)

        assert model is not None
        assert tokenizer is not None
        # Smoke-test: forward pass doesn't crash
        inputs = tokenizer("Hello", return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits.shape[-1] > 0

    @pytest.mark.slow
    def test_error_message_contains_both_hashes(self):
        """The ValueError message must show both the registered and actual hash."""
        registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
        real_entry = registry.get_entry("Qwen/Qwen2.5-0.5B")
        fake_hash = "c" * 64
        bad_entry = ModelEntry(
            model_id=real_entry.model_id,
            hf_repo=real_entry.hf_repo,
            weight_hash=fake_hash,
            num_layers=real_entry.num_layers,
            intermediate_size=real_entry.intermediate_size,
            hidden_size=real_entry.hidden_size,
            min_stake=real_entry.min_stake,
        )
        registry._entries["Qwen/Qwen2.5-0.5B"] = bad_entry

        with pytest.raises(ValueError) as exc_info:
            registry.load_verified_model("Qwen/Qwen2.5-0.5B")

        msg = str(exc_info.value)
        assert fake_hash in msg
        assert EXPECTED_HASHES["Qwen/Qwen2.5-0.5B"] in msg


# ---------------------------------------------------------------------------
# TestRegisterNewModel
# ---------------------------------------------------------------------------


class TestRegisterNewModel:
    """
    register_new_model() must compute the correct hash and write it to the
    registry JSON without modifying existing entries.

    Uses a temporary registry file so the real registry.json is never touched.
    """

    @pytest.fixture
    def tmp_registry(self, tmp_path):
        """Copy registry.json to a temp dir and return a Registry pointing at it."""
        tmp_file = tmp_path / "registry.json"
        shutil.copy(DEFAULT_REGISTRY_PATH, tmp_file)
        return ModelRegistry(tmp_file)

    def test_register_existing_model_returns_same_entry(self, tmp_registry):
        """
        Re-registering a model that's already present with the same hash is
        a no-op — returns the existing entry without error.
        """
        entry = tmp_registry.register_new_model(
            model_id="Qwen/Qwen2.5-0.5B",
            hf_repo="Qwen/Qwen2.5-0.5B",
            download_if_missing=False,
        )
        assert entry.weight_hash == EXPECTED_HASHES["Qwen/Qwen2.5-0.5B"]

    def test_register_new_model_writes_correct_hash(self, tmp_path):
        """
        Register a model into an empty registry and confirm the written hash
        matches the value computed directly.
        """
        empty_registry_path = tmp_path / "empty_registry.json"
        empty_registry_path.write_text(
            json.dumps({"schema_version": "1", "models": {}})
        )
        registry = ModelRegistry(empty_registry_path)

        entry = registry.register_new_model(
            model_id="Qwen/Qwen2.5-0.5B",
            hf_repo="Qwen/Qwen2.5-0.5B",
            min_stake=100,
            download_if_missing=False,
        )

        assert entry.weight_hash == EXPECTED_HASHES["Qwen/Qwen2.5-0.5B"]

        # Confirm it was actually written to disk
        with open(empty_registry_path) as fh:
            data = json.load(fh)
        assert "Qwen/Qwen2.5-0.5B" in data["models"]
        assert data["models"]["Qwen/Qwen2.5-0.5B"]["weight_hash"] == EXPECTED_HASHES["Qwen/Qwen2.5-0.5B"]

    def test_register_new_model_persists_architecture(self, tmp_path):
        """register_new_model() must read and store architecture from config."""
        empty_registry_path = tmp_path / "registry.json"
        empty_registry_path.write_text(
            json.dumps({"schema_version": "1", "models": {}})
        )
        registry = ModelRegistry(empty_registry_path)
        entry = registry.register_new_model(
            "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",
            download_if_missing=False,
        )
        assert entry.num_layers == 24
        assert entry.intermediate_size == 4864
        assert entry.hidden_size == 896

    def test_register_conflicting_hash_raises(self, tmp_path):
        """
        If the registry already has model_id with a *different* hash, raise ValueError.
        """
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps({
            "schema_version": "1",
            "models": {
                "Qwen/Qwen2.5-0.5B": {
                    "model_id": "Qwen/Qwen2.5-0.5B",
                    "hf_repo": "Qwen/Qwen2.5-0.5B",
                    "weight_hash": "d" * 64,   # wrong hash
                    "num_layers": 24,
                    "intermediate_size": 4864,
                    "hidden_size": 896,
                    "min_stake": 0,
                }
            }
        }))
        registry = ModelRegistry(registry_path)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_new_model(
                "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",
                download_if_missing=False,
            )

    def test_does_not_overwrite_other_entries(self, tmp_path):
        """Registering a new model must not disturb existing entries."""
        tmp_file = tmp_path / "registry.json"
        shutil.copy(DEFAULT_REGISTRY_PATH, tmp_file)
        registry = ModelRegistry(tmp_file)

        # Re-register 0.5B (no-op because hash matches)
        registry.register_new_model(
            "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",
            download_if_missing=False,
        )

        with open(tmp_file) as fh:
            data = json.load(fh)

        # 7B entry must be untouched
        assert data["models"]["Qwen/Qwen2.5-7B"]["weight_hash"] == EXPECTED_HASHES["Qwen/Qwen2.5-7B"]


# ---------------------------------------------------------------------------
# TestBackendAbstraction
# ---------------------------------------------------------------------------


class TestBackendAbstraction:
    """
    The _load_registry / _save_registry methods form the only coupling to the
    storage backend.  This test confirms they round-trip cleanly, which means
    swapping the backend only requires replacing those two methods.
    """

    def test_load_save_round_trip(self, tmp_path):
        """_load_registry → modify → _save_registry → _load_registry must preserve data."""
        tmp_file = tmp_path / "registry.json"
        shutil.copy(DEFAULT_REGISTRY_PATH, tmp_file)
        registry = ModelRegistry(tmp_file)

        original = registry._load_registry()
        # Add a sentinel field
        original["_test_sentinel"] = "hello"
        registry._save_registry(original)

        reloaded = registry._load_registry()
        assert reloaded["_test_sentinel"] == "hello"
        assert reloaded["models"] == original["models"]

    def test_empty_registry_returns_empty_models(self, tmp_path):
        """A registry file with no models must produce an empty list."""
        empty_path = tmp_path / "empty.json"
        empty_path.write_text(json.dumps({"schema_version": "1", "models": {}}))
        registry = ModelRegistry(empty_path)
        assert registry.list_models() == []

    def test_missing_registry_file_returns_empty(self, tmp_path):
        """If registry.json doesn't exist yet, the registry starts empty."""
        registry = ModelRegistry(tmp_path / "nonexistent.json")
        assert registry.list_models() == []

    def test_get_registry_returns_singleton(self):
        """get_registry() must return the same object on repeated calls."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
