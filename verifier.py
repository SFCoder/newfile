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
