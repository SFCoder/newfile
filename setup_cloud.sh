#!/usr/bin/env bash
set -euo pipefail
echo "Writing project files..."

cat > registry.json << 'HEREDOC_END'
{
  "schema_version": "1",
  "models": {
    "Qwen/Qwen2.5-0.5B": {
      "model_id": "Qwen/Qwen2.5-0.5B",
      "hf_repo": "Qwen/Qwen2.5-0.5B",
      "weight_hash": "d69088004383b22aee2d56a7d88238eb39f7add7c999f92da046fda0c96557dd",
      "num_layers": 24,
      "intermediate_size": 4864,
      "hidden_size": 896,
      "min_stake": 100
    },
    "Qwen/Qwen2.5-3B": {
      "model_id": "Qwen/Qwen2.5-3B",
      "hf_repo": "Qwen/Qwen2.5-3B",
      "weight_hash": "71fd74069d690809c5a2788e96e3a6d35ee020c128b59a6d774534931058d408",
      "num_layers": 36,
      "intermediate_size": 11008,
      "hidden_size": 2048,
      "min_stake": 500
    },
    "Qwen/Qwen2.5-7B": {
      "model_id": "Qwen/Qwen2.5-7B",
      "hf_repo": "Qwen/Qwen2.5-7B",
      "weight_hash": "b21b4b47cdb2bab55317204504a5b1651789462e5d9d5d649d05890df335f504",
      "num_layers": 28,
      "intermediate_size": 18944,
      "hidden_size": 3584,
      "min_stake": 1000
    }
  }
}
HEREDOC_END

cat > model_registry.py << 'HEREDOC_END'
"""
Model Registry
==============
Local library used by both providers and verifiers to load models with
weight integrity guarantees.

The registry maps model IDs to:
  - HuggingFace repo identifier (used for downloading / caching)
  - Expected weight hash (SHA-256 over all safetensors shards, sorted by name)
  - Architecture metadata (num_layers, intermediate_size, hidden_size)
  - Minimum stake placeholder (integer, for future on-chain enforcement)

Current backend: registry.json on disk.
Future backend:  smart contract read — swap out _load_registry() only.

Public surface
--------------
  ModelEntry                — dataclass describing one registered model
  ModelRegistry             — main class; all functionality lives here
    .list_models()          → list[str]
    .get_entry(model_id)    → ModelEntry
    .compute_weight_hash()  → str
    .verify_weights()       → bool
    .load_verified_model()  → (model, tokenizer)   ← the only path to a live model
    .register_new_model()   → ModelEntry           ← propose + hash + persist

The only public way to obtain a model instance is load_verified_model().
It always checks the hash before loading. Unverified weights are never loaded.

Weight hash algorithm
---------------------
SHA-256 over the concatenation of per-shard hex digests, shards sorted by
filename (ascending).  Formally:

    shard_digests = [sha256(shard_bytes) for shard in sorted(weight_files)]
    weight_hash   = sha256(  "".join(shard_digests).encode("ascii")  ).hexdigest()

This is deterministic across machines and Python versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default path sits next to this file so both provider and verifier machines
# can drop the repo into the same directory and share the same registry.
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    model_id: str
    hf_repo: str
    weight_hash: str          # deterministic SHA-256 (see module docstring)
    num_layers: int
    intermediate_size: int
    hidden_size: int
    min_stake: int            # placeholder; will become on-chain requirement


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Local model registry.  Acts as a drop-in stand-in for an on-chain registry:
    the only method that touches the backend store is _load_registry() / _save_registry().
    Everything else operates on the in-memory dict produced by those two methods.
    """

    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH):
        self._path = registry_path
        self._entries: dict[str, ModelEntry] = {}
        self._refresh()

    # ------------------------------------------------------------------
    # Backend abstraction — swap these two methods to move on-chain
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        """
        Read the registry from the local JSON file.

        On-chain replacement: call the smart contract's getRegistry() view
        function and return the decoded dict in the same shape.
        """
        if not self._path.exists():
            return {"schema_version": "1", "models": {}}
        with open(self._path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_registry(self, data: dict) -> None:
        """
        Persist the registry dict to the local JSON file.

        On-chain replacement: submit a registerModel() transaction.
        """
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _refresh(self) -> None:
        """Reload from the backend into the in-memory cache."""
        data = self._load_registry()
        self._entries = {
            model_id: ModelEntry(**entry)
            for model_id, entry in data.get("models", {}).items()
        }

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return all registered model IDs."""
        return list(self._entries)

    def get_entry(self, model_id: str) -> ModelEntry:
        """Return the ModelEntry for model_id, or raise KeyError."""
        if model_id not in self._entries:
            raise KeyError(
                f"Model {model_id!r} is not registered. "
                f"Known models: {self.list_models()}"
            )
        return self._entries[model_id]

    # ------------------------------------------------------------------
    # Weight hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_path(hf_repo: str) -> Path:
        """
        Resolve the local HuggingFace cache snapshot directory for hf_repo.
        Uses the commit hash recorded in refs/main so the path is pinned to
        the exact checkpoint that was downloaded.
        """
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + hf_repo.replace("/", "--")
        refs_main = cache_root / slug / "refs" / "main"
        if not refs_main.exists():
            raise FileNotFoundError(
                f"No local cache found for {hf_repo!r}. "
                f"Expected: {refs_main}. "
                f"Download the model first with huggingface_hub.snapshot_download()."
            )
        commit = refs_main.read_text().strip()
        return cache_root / slug / "snapshots" / commit

    @staticmethod
    def _weight_files(snapshot: Path) -> list[Path]:
        """
        Return the list of weight shard files in a snapshot directory,
        sorted by filename for deterministic ordering.

        Resolves symlinks so we always hash the actual blob content.
        """
        files = sorted(
            f for f in snapshot.iterdir()
            if f.suffix == ".safetensors" and not f.name.endswith(".index.json")
        )
        if not files:
            raise FileNotFoundError(
                f"No .safetensors weight files found in {snapshot}"
            )
        return [f.resolve() for f in files]

    @classmethod
    def compute_weight_hash(cls, hf_repo: str) -> str:
        """
        Compute the deterministic weight hash for the locally cached weights
        of hf_repo.

        Algorithm:
          1. Locate the snapshot directory via refs/main.
          2. Collect all .safetensors shards, sorted by filename.
          3. SHA-256 each shard (8 MB streaming reads).
          4. SHA-256 the ASCII string formed by concatenating the hex digests.

        Returns the final hex digest (64 lowercase hex characters).
        """
        snapshot = cls._snapshot_path(hf_repo)
        weight_files = cls._weight_files(snapshot)

        shard_digests: list[str] = []
        for wf in weight_files:
            h = hashlib.sha256()
            with open(wf, "rb") as fh:
                while chunk := fh.read(8 * 1024 * 1024):
                    h.update(chunk)
            shard_digests.append(h.hexdigest())
            logger.debug("  shard %s: %s", wf.name, shard_digests[-1])

        combined = hashlib.sha256(
            "".join(shard_digests).encode("ascii")
        ).hexdigest()
        logger.debug("combined weight hash for %s: %s", hf_repo, combined)
        return combined

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_weights(self, model_id: str) -> bool:
        """
        Return True iff the locally cached weights for model_id match the
        registered hash.  Does not load the model.

        Raises KeyError if model_id is not in the registry.
        Raises FileNotFoundError if the weights are not cached locally.
        """
        entry = self.get_entry(model_id)
        actual = self.compute_weight_hash(entry.hf_repo)
        return actual == entry.weight_hash

    # ------------------------------------------------------------------
    # Model loading  — the only path to a live model instance
    # ------------------------------------------------------------------

    def load_verified_model(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Verify weights, then load and return (model, tokenizer).

        Args:
            model_id: Must be present in the registry.
            device:   "cuda" | "mps" | "cpu" | None (auto-detect).
            dtype:    torch dtype for model weights.  Default: float16.

        Raises:
            KeyError:           model_id not registered.
            FileNotFoundError:  weights not cached locally.
            ValueError:         weight hash mismatch — weights may be corrupt
                                or tampered; the model is NOT loaded.
        """
        entry = self.get_entry(model_id)

        logger.info("Verifying weights for %s …", model_id)
        actual_hash = self.compute_weight_hash(entry.hf_repo)
        if actual_hash != entry.weight_hash:
            raise ValueError(
                f"Weight hash mismatch for {model_id!r}.\n"
                f"  registered: {entry.weight_hash}\n"
                f"  on disk:    {actual_hash}\n"
                f"Refusing to load — weights may be corrupt or tampered."
            )
        logger.info("Weight hash verified: %s", actual_hash)

        if device is None:
            device = _detect_device()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = self._snapshot_path(entry.hf_repo)

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            dtype=dtype,
        )
        model = model.to(device)
        model.eval()

        logger.info(
            "Loaded %s on %s (%d layers, intermediate_size=%d)",
            model_id, device, entry.num_layers, entry.intermediate_size,
        )
        return model, tokenizer

    # ------------------------------------------------------------------
    # Registration flow
    # ------------------------------------------------------------------

    def register_new_model(
        self,
        model_id: str,
        hf_repo: str,
        min_stake: int = 0,
        download_if_missing: bool = True,
    ) -> ModelEntry:
        """
        Propose and register a new model.

        Steps:
          1. Optionally download weights (snapshot_download with weights_only).
          2. Compute the weight hash from the cached files.
          3. Read architecture metadata from config.json.
          4. Write the entry to registry.json.
          5. Return the new ModelEntry.

        Raises:
            ValueError: model_id already registered with a different hash.
        """
        if model_id in self._entries:
            existing = self._entries[model_id]
            logger.warning(
                "%s is already registered (hash=%s). Re-computing to confirm.",
                model_id, existing.weight_hash,
            )

        if download_if_missing:
            try:
                self._snapshot_path(hf_repo)
            except FileNotFoundError:
                logger.info("Downloading %s …", hf_repo)
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=hf_repo)

        weight_hash = self.compute_weight_hash(hf_repo)

        # Validate against existing entry if present
        if model_id in self._entries:
            if self._entries[model_id].weight_hash != weight_hash:
                raise ValueError(
                    f"{model_id!r} is already registered with hash "
                    f"{self._entries[model_id].weight_hash!r} but the local "
                    f"weights hash to {weight_hash!r}. Remove the entry first "
                    f"if you intend to update it."
                )
            return self._entries[model_id]

        # Read architecture from config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(self._snapshot_path(hf_repo)))

        entry = ModelEntry(
            model_id=model_id,
            hf_repo=hf_repo,
            weight_hash=weight_hash,
            num_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            hidden_size=config.hidden_size,
            min_stake=min_stake,
        )

        # Persist
        data = self._load_registry()
        data.setdefault("models", {})[model_id] = asdict(entry)
        self._save_registry(data)
        self._refresh()

        logger.info("Registered %s (hash=%s)", model_id, weight_hash)
        return entry


# ---------------------------------------------------------------------------
# Device detection  (module-level helper, used by load_verified_model)
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Module-level default instance  (convenience for single-machine use)
# ---------------------------------------------------------------------------

_default_registry: Optional[ModelRegistry] = None


def get_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    """Return the module-level default ModelRegistry, creating it if needed."""
    global _default_registry
    if _default_registry is None or _default_registry._path != registry_path:
        _default_registry = ModelRegistry(registry_path)
    return _default_registry
HEREDOC_END

cat > verifier.py << 'HEREDOC_END'
"""
Pure verification core — no FastAPI, no server state, no model singleton.

Defines:
  NeuronMask         — compact bitset encoding of active neuron indices
  VerificationBundle — the canonical record that gets hashed and stored (IPFS, DB, chain)
  VerificationResult — output of verify()
  replay_with_masks()— runs greedy generation under claimed neuron masks
  verify()           — compares replayed tokens against bundle's claimed tokens

Nothing in this module starts servers, loads models, or holds global state.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Optional

import torch
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# NeuronMask — one per MLP layer per bundle
# ---------------------------------------------------------------------------


class NeuronMask(BaseModel):
    """
    Compact bitset representation of which neurons were active (|activation| >
    threshold) during at least one generation step.

    Encoding
    --------
    `bits` is a standard base64-encoded byte string.  The underlying buffer is
    ceil(intermediate_size / 8) bytes.  Bit i of the buffer (byte i//8, bit
    position i%8, LSB-first within each byte) is 1 iff neuron i was active.

    This is ~28× smaller than a JSON integer list for Qwen2.5-7B's 18,944-
    dimensional intermediate layer:
        index list: ~2.5 MB per bundle (28 layers × ~92 KB each)
        bitset:     ~66 KB per bundle  (28 layers × ~2.4 KB each)
    """

    intermediate_size: int = Field(description="Total neurons in this MLP layer.")
    bits: str = Field(
        description=(
            "Base64-encoded bitset.  Length in bytes = ceil(intermediate_size / 8).  "
            "Bit i (LSB-first) is 1 iff neuron i was active."
        )
    )

    @model_validator(mode="after")
    def _check_bits_length(self) -> "NeuronMask":
        expected = (self.intermediate_size + 7) // 8
        actual = len(base64.b64decode(self.bits))
        if actual != expected:
            raise ValueError(
                f"bits decodes to {actual} bytes; expected {expected} "
                f"for intermediate_size={self.intermediate_size}"
            )
        return self

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_indices(cls, indices: list[int], intermediate_size: int) -> "NeuronMask":
        """Build a NeuronMask from a list of active neuron indices."""
        buf = bytearray((intermediate_size + 7) // 8)
        for idx in indices:
            if idx < 0 or idx >= intermediate_size:
                raise ValueError(f"Index {idx} out of range [0, {intermediate_size})")
            buf[idx >> 3] |= 1 << (idx & 7)
        return cls(
            intermediate_size=intermediate_size,
            bits=base64.b64encode(bytes(buf)).decode("ascii"),
        )

    @classmethod
    def empty(cls, intermediate_size: int) -> "NeuronMask":
        """All-zero mask (no neurons active)."""
        buf = bytes((intermediate_size + 7) // 8)
        return cls(
            intermediate_size=intermediate_size,
            bits=base64.b64encode(buf).decode("ascii"),
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def to_indices(self) -> list[int]:
        """Return sorted list of active neuron indices."""
        buf = base64.b64decode(self.bits)
        return [
            byte_i * 8 + bit_i
            for byte_i, byte_val in enumerate(buf)
            for bit_i in range(8)
            if (byte_val >> bit_i) & 1
            if byte_i * 8 + bit_i < self.intermediate_size
        ]

    def active_count(self) -> int:
        """Number of active neurons (popcount)."""
        return sum(bin(b).count("1") for b in base64.b64decode(self.bits))

    def density(self) -> float:
        """Fraction of neurons that are active."""
        return self.active_count() / self.intermediate_size


# ---------------------------------------------------------------------------
# VerificationBundle — the canonical, hashable claim
# ---------------------------------------------------------------------------


class VerificationBundle(BaseModel):
    """
    The complete, self-contained record of a single provider inference run.

    This is the object that gets:
    - Serialised to JSON and pinned to IPFS
    - Hashed via content_hash() to produce a CID-equivalent for on-chain anchoring
    - Submitted to POST /verify for replay verification

    Fields
    ------
    schema_version        Bumped when the encoding changes; verifiers can reject
                          bundles they don't understand.
    model_name            HuggingFace model ID, e.g. "Qwen/Qwen2.5-7B".
    prompt                Verbatim input string.
    output_token_ids      Greedy-decoded token IDs (prompt tokens excluded).
    neuron_masks          Per-layer active-neuron bitsets.  Keys are zero-padded
                          decimal layer indices ("00", "01", ...) so that
                          lexicographic and numeric sort orders agree.
    activation_threshold  The magnitude threshold used to decide "active".
    """

    schema_version: str = Field(default="1", description="Bundle encoding version.")
    model_name: str
    prompt: str
    output_token_ids: list[int]
    neuron_masks: dict[str, NeuronMask] = Field(
        description=(
            "Keys are zero-padded decimal layer indices.  "
            "Use VerificationBundle.layer_key(i) to build them."
        )
    )
    activation_threshold: float = Field(default=1e-6)

    # ------------------------------------------------------------------
    # Layer key convention — zero-padded to 3 digits so JSON key sort
    # is identical to numeric sort for up to 999-layer models.
    # ------------------------------------------------------------------

    @staticmethod
    def layer_key(layer_idx: int) -> str:
        return f"{layer_idx:03d}"

    # ------------------------------------------------------------------
    # Canonical serialisation & deterministic hash
    # ------------------------------------------------------------------

    def canonical_json(self) -> str:
        """
        Produce a deterministic, whitespace-free JSON string suitable for
        hashing.  All dict keys are sorted at every nesting level.

        The output is stable across Python versions because:
        - pydantic's model_dump() produces plain dicts/lists/scalars
        - json.dumps with sort_keys=True and no whitespace is spec-defined
        - base64 encoding of the bitset buffers is deterministic by construction
        """
        return json.dumps(
            self.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def content_hash(self) -> str:
        """
        SHA-256 hex digest of canonical_json().

        This is the value to store on-chain or use as an IPFS CID input.
        To derive a real CIDv1: wrap the canonical_json bytes with the
        dag-json multicodec and SHA-256 multihash per the IPFS spec.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# VerificationResult — returned by verify()
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    verified: bool = Field(
        description="True iff replayed tokens exactly match the bundle's claimed tokens."
    )
    bundle_hash: str = Field(description="content_hash() of the bundle that was verified.")
    replayed_token_ids: list[int]
    token_match_rate: float = Field(description="Fraction of token positions that agree (0–1).")
    first_mismatch_position: Optional[int] = Field(
        default=None,
        description="Index of the first mismatching token, or null if fully matching.",
    )
    details: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure replay and verify functions
# ---------------------------------------------------------------------------


def _build_layer_mask_tensor(
    mask: NeuronMask,
    device: torch.device,
) -> torch.Tensor:
    """Convert a NeuronMask to a bool tensor of shape [intermediate_size] on device."""
    indices = mask.to_indices()
    buf = torch.zeros(mask.intermediate_size, dtype=torch.bool, device=device)
    if indices:
        buf[torch.tensor(indices, dtype=torch.long, device=device)] = True
    return buf


def replay_with_masks(
    bundle: VerificationBundle,
    model,
    tokenizer,
) -> list[int]:
    """
    Run greedy generation with each MLP layer's intermediate activations gated
    by the claimed neuron mask from the bundle.

    For every registered layer, a forward hook rewrites the MLP as:
        gate  = act_fn(gate_proj(x))
        up    = up_proj(x)
        inter = gate * up
        inter = inter * claimed_mask   ← zero out unclaimed neurons
        return down_proj(inter)

    For any layer absent from bundle.neuron_masks the MLP runs unmodified.

    Returns the generated token IDs excluding the prompt tokens.
    """
    device = next(model.parameters()).device
    num_tokens = len(bundle.output_token_ids)

    # Pre-build bool tensors on device (one allocation per layer, reused each step)
    layer_masks: dict[int, torch.Tensor] = {}
    for key, mask in bundle.neuron_masks.items():
        layer_masks[int(key)] = _build_layer_mask_tensor(mask, device)

    def make_hook(layer_idx: int):
        if layer_idx not in layer_masks:
            return None
        bool_mask = layer_masks[layer_idx]

        def hook_fn(module, input_tuple, _output):
            x = input_tuple[0]
            gate = module.act_fn(module.gate_proj(x))
            up = module.up_proj(x)
            inter = gate * up
            return module.down_proj(inter * bool_mask.to(inter.dtype))

        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        fn = make_hook(layer_idx)
        if fn is not None:
            hooks.append(layer.mlp.register_forward_hook(fn))

    try:
        inputs = tokenizer(bundle.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return output[0][prompt_len:].tolist()
    finally:
        for h in hooks:
            h.remove()


def verify(
    bundle: VerificationBundle,
    model,
    tokenizer,
) -> VerificationResult:
    """
    Replay the bundle's claimed computation and check whether the output matches.

    This is a pure function: given the same bundle and model weights it always
    produces the same result.  It does not read or write any global state.
    """
    claimed = bundle.output_token_ids
    bundle_hash = bundle.content_hash()

    if not claimed:
        return VerificationResult(
            verified=False,
            bundle_hash=bundle_hash,
            replayed_token_ids=[],
            token_match_rate=0.0,
            details={"error": "output_token_ids is empty"},
        )

    replayed = replay_with_masks(bundle, model, tokenizer)

    n = min(len(claimed), len(replayed))
    matches = sum(1 for a, b in zip(claimed[:n], replayed[:n]) if a == b)
    match_rate = matches / len(claimed)

    first_mismatch: Optional[int] = None
    for i, (a, b) in enumerate(zip(claimed, replayed)):
        if a != b:
            first_mismatch = i
            break
    if first_mismatch is None and len(replayed) != len(claimed):
        first_mismatch = n

    return VerificationResult(
        verified=(replayed == claimed),
        bundle_hash=bundle_hash,
        replayed_token_ids=replayed,
        token_match_rate=match_rate,
        first_mismatch_position=first_mismatch,
        details={
            "claimed_length": len(claimed),
            "replayed_length": len(replayed),
            "matched_tokens": matches,
        },
    )
HEREDOC_END

cat > provider.py << 'HEREDOC_END'
"""
Provider helper — generates honest VerificationBundles from real model
inference and provides tampered variants for test coverage.

This module has no dependency on verification_api.py.  It imports only from
verifier.py (pure schemas and logic) and, lazily, from the model singleton
only when generate_honest_bundle() is called without explicit model/tokenizer
arguments.
"""

from __future__ import annotations

import random

import torch

from verifier import NeuronMask, VerificationBundle

# Neurons with |gate*up| <= this value are excluded from the union mask.
# Must be low enough that zeroing excluded neurons doesn't shift any greedy
# argmax — empirically 0.01 fails for 3B (90% density, token divergence at
# position 9); 1e-6 yields ~100% density and exact replay for all model sizes.
ACTIVATION_THRESHOLD: float = 1e-6

# Qwen2.5-7B vocabulary size — used only in tamper helpers to ensure shifted
# token IDs stay within the valid range
QWEN_VOCAB_SIZE = 151936


# ---------------------------------------------------------------------------
# Honest bundle generation
# ---------------------------------------------------------------------------


def generate_honest_bundle(
    prompt: str,
    num_tokens: int = 20,
    model=None,
    tokenizer=None,
    model_id: str = "Qwen/Qwen2.5-7B",
) -> VerificationBundle:
    """
    Run honest greedy inference, record per-layer union neuron masks, and
    return a VerificationBundle that the verifier will accept.

    The "union mask" for layer l is the set of neurons whose intermediate
    activation magnitude exceeded ACTIVATION_THRESHOLD on at least one token
    position across the full generation trajectory.  This single fixed mask
    is sufficient to reproduce the output exactly (verified empirically:
    5/5 exact matches on Qwen2.5-7B, 30 tokens each).

    Args:
        prompt:     Input text.
        num_tokens: Number of new tokens to generate.
        model:      If None, uses the singleton from verification_api.get_model().
        tokenizer:  If None, uses the singleton from verification_api.get_model().
    """
    if model is None or tokenizer is None:
        from model_registry import get_registry
        model, tokenizer = get_registry().load_verified_model("Qwen/Qwen2.5-7B")

    device = next(model.parameters()).device
    intermediate_size: int = model.config.intermediate_size
    num_layers = len(model.model.layers)

    # layer_idx -> running union bitset as a bytearray on CPU
    n_bytes = (intermediate_size + 7) // 8
    union_bufs: list[bytearray] = [bytearray(n_bytes) for _ in range(num_layers)]

    def make_recording_hook(layer_idx: int):
        buf = union_bufs[layer_idx]

        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            gate = module.act_fn(module.gate_proj(x))
            up = module.up_proj(x)
            intermediate = gate * up  # [batch, seq, intermediate_size]

            # Collapse to a 1-D bool mask: True where any position was active
            active = (intermediate.abs() > ACTIVATION_THRESHOLD).any(dim=0).any(dim=0)
            # active: [intermediate_size] bool on device — pull to CPU and OR into buf
            active_cpu = active.cpu()
            for idx in active_cpu.nonzero(as_tuple=False).squeeze(-1).tolist():
                buf[idx >> 3] |= 1 << (idx & 7)

            return output  # pass through unmodified

        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(make_recording_hook(layer_idx)))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=num_tokens,
            do_sample=False,
        )

    for h in hooks:
        h.remove()

    prompt_len = inputs["input_ids"].shape[1]
    output_token_ids = output[0][prompt_len:].tolist()

    import base64
    neuron_masks = {
        VerificationBundle.layer_key(i): NeuronMask(
            intermediate_size=intermediate_size,
            bits=base64.b64encode(bytes(buf)).decode("ascii"),
        )
        for i, buf in enumerate(union_bufs)
    }

    return VerificationBundle(
        model_name=model_id,
        prompt=prompt,
        output_token_ids=output_token_ids,
        neuron_masks=neuron_masks,
        activation_threshold=ACTIVATION_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Tamper helpers — return new VerificationBundle instances, never mutate
# ---------------------------------------------------------------------------


def tamper_tokens(bundle: VerificationBundle, num_changes: int = 3) -> VerificationBundle:
    """
    Return a bundle with `num_changes` output token IDs shifted by a large prime.
    The neuron masks are left intact, so the verifier replay produces the real
    tokens and the comparison fails.
    """
    ids = list(bundle.output_token_ids)
    positions = random.sample(range(len(ids)), min(num_changes, len(ids)))
    for pos in positions:
        shifted = (ids[pos] + 7919) % QWEN_VOCAB_SIZE
        if shifted == ids[pos]:
            shifted = (ids[pos] + 1) % QWEN_VOCAB_SIZE
        ids[pos] = shifted
    return bundle.model_copy(update={"output_token_ids": ids})


def tamper_masks_zero(bundle: VerificationBundle) -> VerificationBundle:
    """
    Return a bundle where every layer's mask is empty (no neurons active).
    Replaying with zero MLP contribution produces garbage logits.
    """
    empty_masks = {
        key: NeuronMask.empty(mask.intermediate_size)
        for key, mask in bundle.neuron_masks.items()
    }
    return bundle.model_copy(update={"neuron_masks": empty_masks})


def tamper_masks_random(bundle: VerificationBundle, seed: int = 42) -> VerificationBundle:
    """
    Return a bundle where each layer's active-neuron set is replaced by a
    random set of the same cardinality.  The random masks activate wrong
    neurons, causing the replayed computation to diverge.
    """
    rng = random.Random(seed)
    new_masks: dict[str, NeuronMask] = {}
    for key, mask in bundle.neuron_masks.items():
        n_active = mask.active_count()
        dim = mask.intermediate_size
        if n_active == 0 or n_active >= dim:
            new_masks[key] = NeuronMask.empty(dim)
        else:
            random_indices = rng.sample(range(dim), n_active)
            new_masks[key] = NeuronMask.from_indices(random_indices, dim)
    return bundle.model_copy(update={"neuron_masks": new_masks})


def tamper_masks_sparse(
    bundle: VerificationBundle,
    keep_fraction: float = 0.05,
    seed: int = 99,
) -> VerificationBundle:
    """
    Return a bundle retaining only `keep_fraction` of each layer's active
    neurons.  Severe pruning causes the replayed output to diverge.
    """
    rng = random.Random(seed)
    new_masks: dict[str, NeuronMask] = {}
    for key, mask in bundle.neuron_masks.items():
        indices = mask.to_indices()
        n_keep = max(1, int(len(indices) * keep_fraction))
        kept = rng.sample(indices, min(n_keep, len(indices)))
        new_masks[key] = NeuronMask.from_indices(kept, mask.intermediate_size)
    return bundle.model_copy(update={"neuron_masks": new_masks})
HEREDOC_END

echo "  registry.json      ✓"
echo "  model_registry.py  ✓"
echo "  verifier.py        ✓"
echo "  provider.py        ✓"
echo ""
echo "Writing threshold_study.py and adversarial_study.py..."
echo "(Run setup_cloud_part2.sh next)"

# ===========================================================================
# Environment setup (idempotent — safe to run multiple times)
# ===========================================================================

echo ""
echo "=== Environment setup ==="

# --- Install Python dependencies -------------------------------------------
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt …"
    pip install -q -r requirements.txt
    echo "  ✓ dependencies installed"
else
    echo "  [WARN] requirements.txt not found — skipping pip install"
fi

# --- HuggingFace cache symlink ---------------------------------------------
# On cloud instances, /workspace has persistent storage; /root/.cache does not.
# If the workspace cache exists and the root cache does not, create a symlink.
WORKSPACE_HF_CACHE="/workspace/.cache/huggingface/hub"
ROOT_HF_CACHE="/root/.cache/huggingface/hub"

if [ -d "$WORKSPACE_HF_CACHE" ] && [ ! -e "$ROOT_HF_CACHE" ]; then
    mkdir -p "$(dirname "$ROOT_HF_CACHE")"
    ln -s "$WORKSPACE_HF_CACHE" "$ROOT_HF_CACHE"
    echo "  ✓ HuggingFace cache symlink: $ROOT_HF_CACHE -> $WORKSPACE_HF_CACHE"
elif [ -L "$ROOT_HF_CACHE" ]; then
    echo "  ✓ HuggingFace cache symlink already exists: $ROOT_HF_CACHE"
else
    echo "  (no workspace HF cache found; using default cache location)"
fi

# --- Register models passed as CLI arguments --------------------------------
# Usage: bash setup_cloud.sh Qwen/Qwen2.5-7B Qwen/Qwen2.5-72B
# Each argument is treated as a model ID (and HF repo) to register.
if [ "$#" -gt 0 ]; then
    echo ""
    echo "Registering models: $*"
    for MODEL_ID in "$@"; do
        echo "  Registering $MODEL_ID …"
        python3 - <<PYEOF
import sys
sys.path.insert(0, '.')
from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH
reg = ModelRegistry(DEFAULT_REGISTRY_PATH)
try:
    entry = reg.register_new_model(
        model_id="$MODEL_ID",
        hf_repo="$MODEL_ID",
        min_stake=0,
        download_if_missing=True,
    )
    print(f"  ✓ registered $MODEL_ID (hash={entry.weight_hash[:16]}…)")
except Exception as e:
    print(f"  [ERROR] could not register $MODEL_ID: {e}")
PYEOF
    done
fi

echo ""
echo "=== setup_cloud.sh complete ==="
