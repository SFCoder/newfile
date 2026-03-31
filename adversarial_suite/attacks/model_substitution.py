"""
ModelSubstitutionAttack  (stub)
================================
Attack: the provider secretly runs a smaller / cheaper model and submits
neuron masks fabricated to look like they came from the registered model.

This attack is more sophisticated than attention skipping because it
requires either:
  (a) a surrogate model whose activation patterns can be mapped to the
      registered model's neuron space, or
  (b) fabricated masks that happen to reproduce plausible-looking output.

The verification protocol will catch naive fabrication because
replay_with_masks() gates the registered model's FFN with the submitted
masks — if the masks came from a different model, the gated computation
diverges and the token comparison fails.

A realistic implementation would need to solve the mask-transfer problem:
find a mapping from surrogate-model activation indices to registered-model
activation indices such that the masked replay still produces the claimed
tokens.  This is an open research question and the subject of a future
experiment.

Interface (to be implemented)
------------------------------
    attack = ModelSubstitutionAttack(surrogate_model_id="Qwen/Qwen2.5-0.5B")
    result = attack.run(
        registered_model, registered_tokenizer, prompt,
        surrogate_model=surrogate_model,
        surrogate_tokenizer=surrogate_tokenizer,
        num_tokens=30,
    )
"""

from __future__ import annotations

from adversarial_suite.attacks.base import AttackResult  # noqa: F401


class ModelSubstitutionAttack:
    """
    Stub for the model-substitution attack.

    The provider runs a cheaper surrogate model and constructs fake FFN
    masks that pass the registered model's replay verification.

    Not yet implemented.  Will be added in a future experiment focused on
    cross-model mask transfer.
    """

    def __init__(self, surrogate_model_id: str = ""):
        self.surrogate_model_id = surrogate_model_id

    def run(self, model, tokenizer, prompt: str, **kwargs) -> AttackResult:
        raise NotImplementedError(
            "ModelSubstitutionAttack is not yet implemented.  "
            "See module docstring for the planned interface."
        )
