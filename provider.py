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
