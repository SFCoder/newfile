#!/usr/bin/env python3
"""
demo.py — Interactive demonstration of the sparse-replay proof-of-inference system.

Simulates a marketplace where a provider and a verifier are on separate machines.
They share no model instances and communicate only through serialised
VerificationBundle JSON — exactly as they would across a network.

Usage
-----
    python3 demo.py --scenario honest
    python3 demo.py --scenario all
    python3 demo.py --scenario all --max-memory
    python3 demo.py --prompt "What is gravity?" --model Qwen/Qwen2.5-7B --scenario honest
    python3 demo.py --list-scenarios
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from model_registry import ModelRegistry, ModelEntry, DEFAULT_REGISTRY_PATH
from verifier import VerificationBundle, VerificationResult, NeuronMask, verify
from provider import (
    generate_honest_bundle,
    tamper_tokens,
    tamper_masks_random,
    ACTIVATION_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Display constants
# ═══════════════════════════════════════════════════════════════════════════════

_W = 68  # column width for headers and boxes

# Approximate peak RAM (float16, MPS) for known models — display only, not measured.
_KNOWN_MEMORY_GB: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B": 1.0,
    "Qwen/Qwen2.5-3B":   6.2,
    "Qwen/Qwen2.5-7B":  14.7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ModelContext — the ONLY way a model enters or leaves memory
# ═══════════════════════════════════════════════════════════════════════════════

class ModelContext:
    """
    Context manager that loads a model through the registry on entry and
    explicitly frees all memory on exit.

    Usage::

        with ModelContext("Qwen/Qwen2.5-7B", registry) as ctx:
            # ctx.model and ctx.tokenizer are available here
            bundle = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
        # model is gone, memory freed

    The registry verifies the weight hash before loading.  If verification
    fails, __enter__ raises ValueError and __exit__ is never called — no
    partial state is left behind.
    """

    def __init__(self, model_id: str, registry: ModelRegistry):
        self.model_id = model_id
        self._registry = registry
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "ModelContext":
        mem_gb = _KNOWN_MEMORY_GB.get(self.model_id, "?")
        _log("REGISTRY", f"Verifying weight hash for {self.model_id} …")
        entry = self._registry.get_entry(self.model_id)
        actual_hash = ModelRegistry.compute_weight_hash(entry.hf_repo)
        if actual_hash != entry.weight_hash:
            raise ValueError(
                f"Weight hash mismatch for {self.model_id!r}.\n"
                f"  registered: {entry.weight_hash}\n"
                f"  on disk:    {actual_hash}"
            )
        _log("REGISTRY", f"✓ Hash verified: {actual_hash[:16]}…")
        _log("REGISTRY", f"Loading model (~{mem_gb} GB on {_device_name()}) …")
        t0 = time.perf_counter()
        self.model, self.tokenizer = self._registry.load_verified_model(self.model_id)
        elapsed = time.perf_counter() - t0
        _log("REGISTRY", f"Model loaded in {elapsed:.1f}s.")
        return self

    def __exit__(self, *_):
        _log("REGISTRY", f"Unloading {self.model_id} and freeing memory …")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log("REGISTRY", "Memory freed.")


# ═══════════════════════════════════════════════════════════════════════════════
# Narration helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log(role: str, msg: str = "") -> None:
    print(f"  {('[' + role + ']'):<14} {msg}")


def _blank() -> None:
    print()


def _divider() -> None:
    print("  " + "─" * (_W - 2))


def _header(title: str) -> None:
    print()
    print("  " + "═" * (_W - 2))
    print(f"  {title}")
    print("  " + "═" * (_W - 2))


def _verdict_box(lines: list[tuple[str, str]]) -> None:
    """Print a framed verdict box.  Each line is (label, value)."""
    _blank()
    inner_w = _W - 4  # 2-char indent + 1 for each border
    print("  ┌" + "─" * inner_w + "┐")
    for label, value in lines:
        content = f"{label}{value}"
        padding = inner_w - len(content)
        print(f"  │ {content}{' ' * max(0, padding - 1)}│")
    print("  └" + "─" * inner_w + "┘")
    _blank()


def _device_name() -> str:
    if torch.cuda.is_available():
        return "CUDA"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"


def _decode_tokens(token_ids: list[int], tokenizer, max_chars: int = 80) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _bundle_stats(bundle: VerificationBundle) -> dict:
    """Compute display-friendly bundle statistics."""
    canon_bytes = len(bundle.canonical_json().encode())
    total_active = sum(m.active_count() for m in bundle.neuron_masks.values())
    total_neurons = sum(m.intermediate_size for m in bundle.neuron_masks.values())
    num_layers = len(bundle.neuron_masks)
    return {
        "size_kb": canon_bytes / 1024,
        "total_active": total_active,
        "total_neurons": total_neurons,
        "density_pct": 100 * total_active / total_neurons if total_neurons else 0,
        "num_layers": num_layers,
    }


def _print_result_detail(
    result: VerificationResult,
    bundle: VerificationBundle,
    tokenizer,
) -> None:
    claimed_text  = _decode_tokens(bundle.output_token_ids, tokenizer)
    replayed_text = _decode_tokens(result.replayed_token_ids, tokenizer)
    _log("VERIFIER", f"Claimed output:  \"{claimed_text}\"")
    _log("VERIFIER", f"Replayed output: \"{replayed_text}\"")
    _divider()
    match_n = result.details.get("matched_tokens", 0)
    total   = result.details.get("claimed_length", len(bundle.output_token_ids))
    _log("VERIFIER", f"Token match: {match_n}/{total} ({result.token_match_rate*100:.1f}%)")
    if result.first_mismatch_position is not None:
        _log("VERIFIER", f"First mismatch at token position {result.first_mismatch_position}")


def _print_verdict(result: VerificationResult | None, *, extra_lines: list = None) -> None:
    if result is None:
        # Used for non-verify outcomes (e.g. registration rejected)
        lines = [("✗  VERDICT: ", "REGISTRATION REJECTED")]
    elif result.verified:
        lines = [
            ("✓  VERDICT: ", "VERIFIED"),
            ("   Bundle hash: ", result.bundle_hash[:32] + "…"),
            ("   Token match: ", f"{result.token_match_rate*100:.1f}%"),
        ]
    else:
        lines = [
            ("✗  VERDICT: ", "FRAUD DETECTED"),
            ("   Bundle hash: ", result.bundle_hash[:32] + "…"),
            ("   Token match: ", f"{result.token_match_rate*100:.1f}%"),
        ]
    if extra_lines:
        lines.extend(extra_lines)
    _verdict_box(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DemoProvider — the claiming side
# ═══════════════════════════════════════════════════════════════════════════════

class DemoProvider:
    """
    Simulates a provider node in the marketplace.

    Contract: holds NO model references between calls.  Receives model and
    tokenizer as arguments so the caller (scenario function) controls the
    object lifetime via ModelContext.  Communicates with DemoVerifier only
    through a serialised JSON string.
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def generate(
        self,
        prompt: str,
        model_id: str,
        num_tokens: int,
        model,
        tokenizer,
    ) -> str:
        """
        Run honest inference, record neuron masks, return a serialised
        VerificationBundle JSON string.
        """
        _log("PROVIDER", f"Running inference on: \"{prompt[:60]}\"")
        _log("PROVIDER", f"Model: {model_id}  |  Tokens to generate: {num_tokens}")

        t0 = time.perf_counter()
        bundle = generate_honest_bundle(
            prompt,
            num_tokens=num_tokens,
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
        )
        elapsed = time.perf_counter() - t0

        stats = _bundle_stats(bundle)
        output_text = _decode_tokens(bundle.output_token_ids, tokenizer)

        _log("PROVIDER", f"Generated in {elapsed:.1f}s: \"{output_text}\"")
        _divider()
        _log("PROVIDER", f"Recorded masks for {stats['num_layers']} MLP layers")
        _log("PROVIDER", (
            f"Active neurons (union): {stats['total_active']:,} / {stats['total_neurons']:,} "
            f"({stats['density_pct']:.1f}%)"
        ))
        _log("PROVIDER", f"Bundle size: {stats['size_kb']:.1f} KB (bitset-encoded)")

        bundle_json = bundle.model_dump_json()
        bundle_hash = bundle.content_hash()
        _log("PROVIDER", f"Bundle hash: {bundle_hash}")
        _blank()
        _log("PROVIDER", "Serialising bundle and transmitting to verifier …")
        _log("PROVIDER", "(In production this travels over the network / gets pinned to IPFS)")

        return bundle_json

    def generate_lying_about_model(
        self,
        prompt: str,
        actual_model_id: str,
        claimed_model_id: str,
        num_tokens: int,
        model,
        tokenizer,
    ) -> str:
        """
        Dishonest variant: generate with actual_model_id but label the bundle
        as claimed_model_id.  Used by the wrong-model scenario.
        """
        _log("PROVIDER", f"Running inference on: \"{prompt[:60]}\"")
        _log("PROVIDER", f"Actual model:  {actual_model_id}")
        _log("PROVIDER", f"Claimed model: {claimed_model_id}  ← THE LIE")

        t0 = time.perf_counter()
        bundle = generate_honest_bundle(
            prompt,
            num_tokens=num_tokens,
            model=model,
            tokenizer=tokenizer,
            model_id=actual_model_id,   # honest masks for the actual model
        )
        elapsed = time.perf_counter() - t0

        # Tamper: relabel the bundle as the larger model
        lying_bundle = bundle.model_copy(update={"model_name": claimed_model_id})

        stats = _bundle_stats(lying_bundle)
        entry = self._registry.get_entry(actual_model_id)
        output_text = _decode_tokens(lying_bundle.output_token_ids, tokenizer)

        _log("PROVIDER", f"Generated in {elapsed:.1f}s: \"{output_text}\"")
        _divider()
        _log("PROVIDER", (
            f"Mask intermediate_size = {entry.intermediate_size} "
            f"({actual_model_id} architecture)"
        ))
        _log("PROVIDER", (
            f"Bundle mislabelled as {claimed_model_id} "
            f"(intermediate_size should be "
            f"{self._registry.get_entry(claimed_model_id).intermediate_size})"
        ))
        _log("PROVIDER", f"Bundle hash: {lying_bundle.content_hash()}")
        _blank()
        _log("PROVIDER", "Transmitting lying bundle to verifier …")

        return lying_bundle.model_dump_json()


# ═══════════════════════════════════════════════════════════════════════════════
# DemoVerifier — the checking side
# ═══════════════════════════════════════════════════════════════════════════════

class DemoVerifier:
    """
    Simulates a verifier node in the marketplace.

    Contract: holds NO model references between calls.  Receives a raw JSON
    string (as if from the network), deserialises it, and replays the
    computation against weights it loaded independently.
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def receive_and_verify(
        self,
        bundle_json: str,
        model,
        tokenizer,
    ) -> VerificationResult:
        """
        Deserialise a bundle, run the sparse replay, and return the result.
        Raises no exceptions — all failures are captured in VerificationResult.
        """
        bundle = VerificationBundle.model_validate_json(bundle_json)
        size_kb = len(bundle_json.encode()) / 1024

        _log("VERIFIER", f"Received bundle ({size_kb:.1f} KB)")
        _log("VERIFIER", f"Bundle hash: {bundle.content_hash()}")
        _log("VERIFIER", f"Claimed model: {bundle.model_name}")
        _log("VERIFIER", f"Claimed output: {len(bundle.output_token_ids)} tokens")
        _log("VERIFIER", f"Neuron masks: {len(bundle.neuron_masks)} layers")

        # Pre-check: mask dimensions must match this model's architecture
        compat_ok, compat_msg = self._check_mask_compatibility(bundle, model)
        if not compat_ok:
            _blank()
            _log("VERIFIER", "✗ Architecture mismatch detected!")
            _log("VERIFIER", compat_msg)
            # Return a synthetic failed result rather than crashing
            return VerificationResult(
                verified=False,
                bundle_hash=bundle.content_hash(),
                replayed_token_ids=[],
                token_match_rate=0.0,
                first_mismatch_position=0,
                details={"error": "architecture_mismatch", "detail": compat_msg},
            )

        _blank()
        _log("VERIFIER", f"Replaying {len(bundle.output_token_ids)} tokens under claimed masks …")
        t0 = time.perf_counter()
        result = verify(bundle, model, tokenizer)
        elapsed = time.perf_counter() - t0
        _log("VERIFIER", f"Replay complete in {elapsed:.1f}s.")

        _print_result_detail(result, bundle, tokenizer)
        return result

    @staticmethod
    def _check_mask_compatibility(
        bundle: VerificationBundle,
        model,
    ) -> tuple[bool, str]:
        """
        Return (True, "") if all mask intermediate_sizes match the model,
        or (False, reason) if there is a mismatch.
        """
        expected = model.config.intermediate_size
        for key, mask in bundle.neuron_masks.items():
            if mask.intermediate_size != expected:
                return False, (
                    f"Layer {key}: mask intermediate_size={mask.intermediate_size} "
                    f"but model has intermediate_size={expected}.  "
                    f"These masks were created with a different model architecture."
                )
        return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario system
#
# HOW TO ADD A NEW SCENARIO
# ─────────────────────────
# 1. Write a function with this exact signature:
#
#      def scenario_my_name(
#          prompt: str,
#          model_id: str,
#          max_memory: bool,
#          provider: DemoProvider,
#          verifier: DemoVerifier,
#          registry: ModelRegistry,
#      ) -> None:
#
# 2. Add one entry to the SCENARIOS dict at the bottom of this section:
#
#      "my-name": ScenarioEntry(
#          fn=scenario_my_name,
#          description="One-line description shown in --list-scenarios",
#          default_prompt="A suitable default prompt for this scenario",
#          default_model="Qwen/Qwen2.5-7B",
#          approx_minutes=5.0,
#      ),
#
# That's it.  The CLI picks up new scenarios automatically.
#
# Planned future scenarios to add here:
#   "adversarial-finetune"  — provider runs a fine-tuned model with same base mask pattern
#   "quantization-mismatch" — provider uses INT4, verifier expects float16
#   "cross-hardware"        — replay on different hardware to test fp determinism
#   "partial-masks"         — provider omits some layer masks, verifier fills with defaults
#   "replay-attack"         — provider submits a bundle from a previous run against a new prompt
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScenarioEntry:
    fn: Callable
    description: str
    default_prompt: str
    default_model: str
    approx_minutes: float  # rough estimate for --scenario all preamble


# ──────────────────────────────────────────────────────────────────────────────

def scenario_honest(
    prompt: str,
    model_id: str,
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    The golden path: provider runs the real model, records real masks,
    submits an honest bundle.  Verifier replays and confirms every token.
    Expected: VERIFIED.
    """
    result = None
    if max_memory:
        # Load → provider work → unload → load → verifier work → unload
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
        with ModelContext(model_id, registry) as ctx:
            result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)
    else:
        # Single load shared across both roles (same model, same machine)
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
            _blank()
            _divider()
            result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)
    _print_verdict(result)


def scenario_tampered_output(
    prompt: str,
    model_id: str,
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    Provider generates an honest bundle, then alters 3 output token IDs before
    submitting.  The neuron masks are still correct — but the claimed tokens
    don't match what the masks produce.  Expected: FRAUD DETECTED.
    """
    result = None
    if max_memory:
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
        bundle = VerificationBundle.model_validate_json(bundle_json)
        tampered = tamper_tokens(bundle, num_changes=3)
        bundle_json = tampered.model_dump_json()
        _blank()
        _log("PROVIDER", "Altering 3 output token IDs before submission …")
        _log("PROVIDER", f"New bundle hash: {tampered.content_hash()}")
        _blank()
        with ModelContext(model_id, registry) as ctx:
            result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)
    else:
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
            bundle = VerificationBundle.model_validate_json(bundle_json)
            tampered = tamper_tokens(bundle, num_changes=3)
            _blank()
            _log("PROVIDER", "Altering 3 output token IDs before submission …")
            _log("PROVIDER", f"New bundle hash: {tampered.content_hash()}")
            _blank()
            _divider()
            result = verifier.receive_and_verify(tampered.model_dump_json(), ctx.model, ctx.tokenizer)
    _print_verdict(result)


def scenario_wrong_model(
    prompt: str,
    model_id: str,  # used as the claimed (large) model
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    Provider runs the cheap 0.5B model but labels the bundle as 7B.
    Verifier loads 7B, finds that the mask intermediate_size (4864) doesn't
    match 7B's architecture (18944) — immediate fraud detection.
    Expected: FRAUD DETECTED.

    Always loads models sequentially (provider 0.5B first, then verifier 7B)
    regardless of --max-memory, because two different models are involved.
    """
    actual_model_id = "Qwen/Qwen2.5-0.5B"
    claimed_model_id = model_id  # the lie

    _log("PROVIDER", f"(Running {actual_model_id} while claiming to be {claimed_model_id})")
    _blank()

    # Step 1: provider loads the small model
    with ModelContext(actual_model_id, registry) as ctx:
        bundle_json = provider.generate_lying_about_model(
            prompt, actual_model_id, claimed_model_id, 20, ctx.model, ctx.tokenizer
        )
    # Small model is now freed — load the large model for verifier

    _blank()
    _divider()
    # Step 2: verifier loads the claimed (large) model
    with ModelContext(claimed_model_id, registry) as ctx:
        result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)

    _print_verdict(
        result,
        extra_lines=[(
            "   Reason: ",
            f"Masks sized for {actual_model_id} don't fit {claimed_model_id}",
        )],
    )


def scenario_fake_masks(
    prompt: str,
    model_id: str,
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    Provider generates real output from the real model, then replaces all
    neuron masks with random binary masks of the same density.
    The claimed tokens are real, but the masks don't reproduce them.
    Expected: FRAUD DETECTED.
    """
    result = None
    if max_memory:
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
        bundle = VerificationBundle.model_validate_json(bundle_json)
        faked = tamper_masks_random(bundle, seed=12345)
        _blank()
        _log("PROVIDER", "Replacing all neuron masks with random masks (same density) …")
        _log("PROVIDER", f"New bundle hash: {faked.content_hash()}")
        _blank()
        with ModelContext(model_id, registry) as ctx:
            result = verifier.receive_and_verify(faked.model_dump_json(), ctx.model, ctx.tokenizer)
    else:
        with ModelContext(model_id, registry) as ctx:
            bundle_json = provider.generate(prompt, model_id, 20, ctx.model, ctx.tokenizer)
            bundle = VerificationBundle.model_validate_json(bundle_json)
            faked = tamper_masks_random(bundle, seed=12345)
            _blank()
            _log("PROVIDER", "Replacing all neuron masks with random masks (same density) …")
            _log("PROVIDER", f"New bundle hash: {faked.content_hash()}")
            _blank()
            _divider()
            result = verifier.receive_and_verify(faked.model_dump_json(), ctx.model, ctx.tokenizer)
    _print_verdict(result)


def scenario_tampered_weights(
    prompt: str,
    model_id: str,
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    Attempt to register a model with a deliberately incorrect weight hash.
    The registry detects the mismatch before any weights are loaded.
    No model inference is run at all — the attack is blocked at registration.
    Expected: REGISTRATION REJECTED.
    """
    import tempfile, shutil

    _log("PROVIDER", "Attempting to register a model with a tampered hash …")
    _blank()

    # Use a temp registry so we never corrupt the real registry.json
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "registry.json"
        shutil.copy(DEFAULT_REGISTRY_PATH, tmp_path)
        tmp_registry = ModelRegistry(tmp_path)

        # Manually inject a wrong hash for 0.5B
        entry = tmp_registry.get_entry("Qwen/Qwen2.5-0.5B")
        wrong_hash = "a" * 64
        _log("PROVIDER", f"Real weight hash:   {entry.weight_hash[:32]}…")
        _log("PROVIDER", f"Submitted hash:     {wrong_hash[:32]}…  ← TAMPERED")
        _blank()

        # Patch the in-memory entry to simulate a provider submitting a wrong hash,
        # then attempt to load — which internally recomputes the real hash and compares.
        from model_registry import ModelEntry
        bad_entry = ModelEntry(
            model_id=entry.model_id, hf_repo=entry.hf_repo, weight_hash=wrong_hash,
            num_layers=entry.num_layers, intermediate_size=entry.intermediate_size,
            hidden_size=entry.hidden_size, min_stake=entry.min_stake,
        )
        tmp_registry._entries["Qwen/Qwen2.5-0.5B"] = bad_entry

        _log("REGISTRY", "Verifying submitted hash against on-disk weights …")
        rejected = False
        rejection_reason = ""
        try:
            # This triggers hash recomputation and comparison
            tmp_registry.load_verified_model("Qwen/Qwen2.5-0.5B")
        except ValueError as exc:
            rejected = True
            rejection_reason = str(exc).split("\n")[0]
            _log("REGISTRY", f"✗ REJECTED: {rejection_reason}")

    _blank()
    if rejected:
        _print_verdict(
            None,
            extra_lines=[("   Reason: ", "On-disk hash doesn't match submitted hash")],
        )
    else:
        _log("ERROR", "Registry accepted tampered hash — this should not happen!")


def scenario_different_model(
    prompt: str,
    model_id: str,  # ignored; this scenario always uses 3B
    max_memory: bool,
    provider: DemoProvider,
    verifier: DemoVerifier,
    registry: ModelRegistry,
) -> None:
    """
    Honest end-to-end transaction using Qwen/Qwen2.5-3B to demonstrate that
    the system works across model sizes, not just 7B.
    Expected: VERIFIED.
    """
    use_model = "Qwen/Qwen2.5-3B"
    _log("PROVIDER", f"Using {use_model} for this scenario.")
    _blank()

    result = None
    if max_memory:
        with ModelContext(use_model, registry) as ctx:
            bundle_json = provider.generate(prompt, use_model, 20, ctx.model, ctx.tokenizer)
        with ModelContext(use_model, registry) as ctx:
            result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)
    else:
        with ModelContext(use_model, registry) as ctx:
            bundle_json = provider.generate(prompt, use_model, 20, ctx.model, ctx.tokenizer)
            _blank()
            _divider()
            result = verifier.receive_and_verify(bundle_json, ctx.model, ctx.tokenizer)
    _print_verdict(result)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, ScenarioEntry] = {
    "different-model": ScenarioEntry(
        fn=scenario_different_model,
        description="Honest transaction using Qwen2.5-3B — shows the system works across sizes",
        default_prompt="Explain what machine learning is in one sentence:",
        default_model="Qwen/Qwen2.5-7B",  # ignored by this scenario
        approx_minutes=2.0,
    ),
    "honest": ScenarioEntry(
        fn=scenario_honest,
        description="Honest provider: bundle passes verification end-to-end",
        default_prompt="The capital of France is",
        default_model="Qwen/Qwen2.5-7B",
        approx_minutes=4.0,
    ),
    "tampered-output": ScenarioEntry(
        fn=scenario_tampered_output,
        description="Provider alters 3 output tokens after generation — caught by replay",
        default_prompt="The three primary colors are",
        default_model="Qwen/Qwen2.5-7B",
        approx_minutes=4.0,
    ),
    "fake-masks": ScenarioEntry(
        fn=scenario_fake_masks,
        description="Provider submits real output with randomised neuron masks — caught",
        default_prompt="Write a Python function that adds two numbers:",
        default_model="Qwen/Qwen2.5-7B",
        approx_minutes=4.0,
    ),
    "wrong-model": ScenarioEntry(
        fn=scenario_wrong_model,
        description="Provider runs 0.5B but claims 7B — caught by architecture mismatch",
        default_prompt="In a parallel universe,",
        default_model="Qwen/Qwen2.5-7B",
        approx_minutes=3.0,
    ),
    "tampered-weights": ScenarioEntry(
        fn=scenario_tampered_weights,
        description="Attempt to register a model with a wrong hash — registry rejects it",
        default_prompt="",  # no inference needed
        default_model="Qwen/Qwen2.5-0.5B",
        approx_minutes=0.5,
    ),
}

# Order for --scenario all: minimises total model-load time by grouping by model.
# 3B first, then 7B group (honest/tampered/fake-masks), then wrong-model
# (0.5B→7B, 7B warm in filesystem cache), then no-load scenario.
_ALL_ORDER = [
    "different-model",
    "honest",
    "tampered-output",
    "fake-masks",
    "wrong-model",
    "tampered-weights",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario(
    name: str,
    prompt: str,
    model_id: str,
    max_memory: bool,
    registry: ModelRegistry,
) -> None:
    entry = SCENARIOS[name]
    provider = DemoProvider(registry)
    verifier = DemoVerifier(registry)

    mem_gb = _KNOWN_MEMORY_GB.get(model_id, "?")
    _header(
        f"SCENARIO: {name.upper()}  "
        f"(~{entry.approx_minutes:.0f} min | peak ~{mem_gb} GB)"
    )
    _log("DEMO", entry.description)
    _blank()

    t0 = time.perf_counter()
    entry.fn(prompt, model_id, max_memory, provider, verifier, registry)
    elapsed = time.perf_counter() - t0

    _log("DEMO", f"Scenario completed in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demo.py",
        description="Interactive proof-of-inference demonstration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 demo.py --scenario honest\n"
            "  python3 demo.py --scenario all\n"
            "  python3 demo.py --scenario all --max-memory\n"
            "  python3 demo.py --prompt 'What is gravity?' --scenario honest\n"
            "  python3 demo.py --list-scenarios\n"
        ),
    )
    p.add_argument(
        "--scenario",
        choices=list(SCENARIOS) + ["all"],
        default="honest",
        metavar="NAME",
        help="Scenario to run, or 'all' to run every scenario in sequence.",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Override the default prompt for the chosen scenario.",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Override the default model for the chosen scenario.",
    )
    p.add_argument(
        "--max-memory",
        action="store_true",
        help=(
            "Always unload the provider's model before the verifier loads, "
            "even when both use the same model.  Slower but safe on 16 GB machines."
        ),
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenarios and exit.",
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        metavar="PATH",
        help="Path to registry.json (default: registry.json next to this file).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:\n")
        for name, entry in SCENARIOS.items():
            print(f"  {name:<20}  {entry.description}")
            print(f"  {'':20}  default model:  {entry.default_model}")
            print(f"  {'':20}  default prompt: \"{entry.default_prompt[:55]}\"")
            print()
        sys.exit(0)

    registry = ModelRegistry(Path(args.registry))

    if args.scenario == "all":
        total_mins = sum(SCENARIOS[n].approx_minutes for n in _ALL_ORDER)
        mem_gb = _KNOWN_MEMORY_GB.get("Qwen/Qwen2.5-7B", "?")
        print()
        print("  " + "═" * (_W - 2))
        print(f"  PROOF-OF-INFERENCE DEMO — running all {len(_ALL_ORDER)} scenarios")
        print(f"  Estimated total time: ~{total_mins:.0f} minutes")
        print(f"  Peak memory:         ~{mem_gb} GB (7B model, float16, MPS)")
        print(f"  Memory mode:         {'--max-memory (always sequential)' if args.max_memory else 'shared (same model reused within scenario)'}")
        print("  " + "═" * (_W - 2))

        for i, name in enumerate(_ALL_ORDER, 1):
            entry = SCENARIOS[name]
            prompt  = args.prompt  or entry.default_prompt
            model_id = args.model  or entry.default_model
            print(f"\n  [{i}/{len(_ALL_ORDER)}] ", end="", flush=True)
            run_scenario(name, prompt, model_id, args.max_memory, registry)
    else:
        entry    = SCENARIOS[args.scenario]
        prompt   = args.prompt   or entry.default_prompt
        model_id = args.model    or entry.default_model
        run_scenario(args.scenario, prompt, model_id, args.max_memory, registry)

    print()


if __name__ == "__main__":
    main()
