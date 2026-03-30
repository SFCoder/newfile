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
