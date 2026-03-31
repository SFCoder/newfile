"""
AttentionSkipAttack
===================
Attack: zero out the self-attention output at a selected subset of
transformer layers while computing every FFN layer honestly.

The verification protocol checks the FFN (via neuron masks) but does
not verify attention computations.  This attack exploits that gap.

The provider runs inference with zeroed attention at N layers, records
the honest FFN neuron masks from that pass, and submits a bundle with
the fraudulent output tokens + those FFN masks.  The verifier replays
using the claimed masks (with full attention); if the outputs match,
the attack passes verification.

The experiment measures: how many layers can be skipped before the
output degrades below an acceptable token-match threshold?

Usage::

    from adversarial_suite.attacks.attention_skip import AttentionSkipAttack

    attack = AttentionSkipAttack()
    result = attack.run(
        model, tokenizer, prompt,
        layers_to_skip=[0, 1, 4],
        num_tokens=30,
        model_id="Qwen/Qwen2.5-7B",
    )
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifier import VerificationBundle, NeuronMask  # noqa: E402
from provider import ACTIVATION_THRESHOLD             # noqa: E402
from adversarial_suite.attacks.base import AttackResult  # noqa: E402


class AttentionSkipAttack:
    """
    Zero-attention skip attack.

    For each layer in `layers_to_skip`, a forward hook replaces the
    self-attention output tensor with zeros before the residual addition.
    The FFN at every layer runs honestly and its neuron masks are
    recorded.  The attack output tokens are those produced by this
    modified forward pass.

    Computation saved: approximately (len(layers_to_skip) / total_layers)
    × 33% of total inference FLOPs, since attention accounts for roughly
    one third of per-layer compute.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        model,
        tokenizer,
        prompt: str,
        *,
        layers_to_skip: list,
        num_tokens: int = 30,
        model_id: str = "unknown",
        honest_tokens: Optional[list] = None,
        capture_logits: bool = True,
    ) -> AttackResult:
        """
        Execute the attack and return an AttackResult.

        Parameters
        ----------
        model, tokenizer
            Loaded model and tokenizer.
        prompt
            Input text.
        layers_to_skip
            Layer indices at which to zero the attention output.
        num_tokens
            Number of new tokens to generate.
        model_id
            Model identifier written into bundle metadata.
        honest_tokens
            Pre-computed honest token IDs.  If None, honest inference is
            run here.  Pass pre-computed tokens to avoid redundant
            inference when calling this method repeatedly for the same
            prompt.
        capture_logits
            If True, capture per-step logits via an lm_head hook during
            both honest and attack generation.  Required for cosine
            similarity metrics.
        """
        layers_to_skip_set = set(layers_to_skip)
        num_layers = len(model.model.layers)

        # --- honest run -------------------------------------------------
        if honest_tokens is None:
            honest_tokens, h_logits = self._generate(
                model, tokenizer, prompt, num_tokens,
                zeroed_layers=set(), capture_logits=capture_logits,
            )
        else:
            h_logits = None  # caller pre-computed; logits not available here

        # Build honest bundle (records honest FFN masks)
        honest_bundle = self._build_bundle(
            model, tokenizer, prompt, num_tokens, honest_tokens, model_id,
            zeroed_layers=set(),
        )

        # --- attack run -------------------------------------------------
        fraudulent_tokens, a_logits = self._generate(
            model, tokenizer, prompt, num_tokens,
            zeroed_layers=layers_to_skip_set, capture_logits=capture_logits,
        )

        # Record FFN masks from the attack run
        attack_masks = self._record_ffn_masks(
            model, tokenizer, prompt, num_tokens,
            zeroed_layers=layers_to_skip_set,
        )

        return AttackResult(
            honest_tokens=honest_tokens,
            fraudulent_tokens=fraudulent_tokens,
            fraudulent_neuron_masks=attack_masks,
            metadata={
                "attack_type": "attention_skip",
                "layers_skipped": sorted(layers_to_skip),
                "num_layers_skipped": len(layers_to_skip),
                "total_layers": num_layers,
                "model_id": model_id,
                "prompt": prompt,
                "num_tokens": num_tokens,
            },
            ground_truth_bundle=honest_bundle,
            honest_logits=h_logits,
            fraudulent_logits=a_logits,
        )

    def build_fraudulent_bundle(
        self,
        result: AttackResult,
        model_id: str,
        activation_threshold: float = ACTIVATION_THRESHOLD,
    ) -> VerificationBundle:
        """
        Construct the VerificationBundle that the attacker submits.

        Contains the fraudulent output tokens and the FFN masks recorded
        during the attack run (honest FFN under zeroed attention).
        """
        return VerificationBundle(
            model_name=model_id,
            prompt=result.metadata["prompt"],
            output_token_ids=result.fraudulent_tokens,
            neuron_masks=result.fraudulent_neuron_masks,
            activation_threshold=activation_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_attention_hooks(model, zeroed_layers: set) -> list:
        """Register forward hooks that zero the attention output at each
        layer in *zeroed_layers*.  Returns the list of hook handles."""
        hooks = []
        if not zeroed_layers:
            return hooks

        def make_hook():
            def hook_fn(module, inp, output):
                # Qwen2.5 self_attn returns (attn_out, past_kv) or
                # (attn_out, past_kv, attn_weights).  Zero only attn_out.
                if isinstance(output, tuple):
                    zeros = torch.zeros_like(output[0])
                    return (zeros,) + output[1:]
                return torch.zeros_like(output)
            return hook_fn

        for idx, layer in enumerate(model.model.layers):
            if idx in zeroed_layers:
                hooks.append(
                    layer.self_attn.register_forward_hook(make_hook())
                )
        return hooks

    @staticmethod
    def _logit_capture_hooks(model) -> tuple:
        """
        Register a hook on model.lm_head that captures the last-position
        logit at every generation step.

        Returns (hooks_list, logits_container).  After generation,
        logits_container is a list of length num_tokens, each element a
        1-D CPU tensor of shape (vocab_size,).
        """
        captured: list = []

        def hook_fn(module, inp, output):
            # output: [batch, seq_len, vocab_size]
            # During autoregressive generation with KV cache, seq_len==1
            # after the first prefill; always take the last position.
            captured.append(output[0, -1].detach().cpu())

        hook = model.lm_head.register_forward_hook(hook_fn)
        return [hook], captured

    def _generate(
        self,
        model,
        tokenizer,
        prompt: str,
        num_tokens: int,
        zeroed_layers: set,
        capture_logits: bool,
    ) -> tuple:
        """
        Run model.generate() with optional zeroed-attention hooks and
        optional logit capture.

        Returns (token_ids, logits_or_None).
        """
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        attn_hooks = self._zero_attention_hooks(model, zeroed_layers)
        if capture_logits:
            logit_hooks, captured_logits = self._logit_capture_hooks(model)
        else:
            logit_hooks, captured_logits = [], None

        all_hooks = attn_hooks + logit_hooks
        try:
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=num_tokens,
                    do_sample=False,
                )
        finally:
            for h in all_hooks:
                h.remove()

        tokens = output[0][prompt_len:].tolist()
        logits = captured_logits if capture_logits else None
        return tokens, logits

    def _record_ffn_masks(
        self,
        model,
        tokenizer,
        prompt: str,
        num_tokens: int,
        zeroed_layers: set,
    ) -> dict:
        """
        Run inference with zeroed attention (attack conditions) and record
        the union FFN neuron masks for every layer.

        Returns dict mapping layer_key -> NeuronMask.
        """
        device = next(model.parameters()).device
        intermediate_size = model.config.intermediate_size
        num_layers = len(model.model.layers)
        n_bytes = (intermediate_size + 7) // 8
        union_bufs: list = [bytearray(n_bytes) for _ in range(num_layers)]

        def make_ffn_hook(layer_idx: int):
            buf = union_bufs[layer_idx]

            def hook_fn(module, inp, output):
                x = inp[0]
                gate = module.act_fn(module.gate_proj(x))
                up = module.up_proj(x)
                intermediate = gate * up
                # Union of active neurons across all positions in this step
                active = (
                    (intermediate.abs() > ACTIVATION_THRESHOLD)
                    .any(dim=0)
                    .any(dim=0)
                    .cpu()
                )
                for idx in active.nonzero(as_tuple=False).squeeze(-1).tolist():
                    buf[idx >> 3] |= 1 << (idx & 7)
                return output  # pass through unmodified

            return hook_fn

        attn_hooks = self._zero_attention_hooks(model, zeroed_layers)
        ffn_hooks = [
            model.model.layers[i].mlp.register_forward_hook(
                make_ffn_hook(i)
            )
            for i in range(num_layers)
        ]
        all_hooks = attn_hooks + ffn_hooks

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                model.generate(
                    inputs["input_ids"],
                    max_new_tokens=num_tokens,
                    do_sample=False,
                )
        finally:
            for h in all_hooks:
                h.remove()

        return {
            VerificationBundle.layer_key(i): NeuronMask(
                intermediate_size=intermediate_size,
                bits=base64.b64encode(bytes(buf)).decode("ascii"),
            )
            for i, buf in enumerate(union_bufs)
        }

    def _build_bundle(
        self,
        model,
        tokenizer,
        prompt: str,
        num_tokens: int,
        tokens: list,
        model_id: str,
        zeroed_layers: set,
    ) -> VerificationBundle:
        """Build a VerificationBundle from a completed generation run."""
        masks = self._record_ffn_masks(
            model, tokenizer, prompt, num_tokens, zeroed_layers
        )
        return VerificationBundle(
            model_name=model_id,
            prompt=prompt,
            output_token_ids=tokens,
            neuron_masks=masks,
            activation_threshold=ACTIVATION_THRESHOLD,
        )
