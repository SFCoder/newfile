"""
TokenTamperAttack  (stub)
==========================
Attack: the provider runs honest inference but submits subtly altered
output token IDs while keeping the neuron masks intact.

Because replay_with_masks() reproduces the honest tokens, the token
comparison fails and the tamper is detected — this attack is expected to
be *caught* by the verifier and serves as a sanity check / lower bound.

Interface (to be implemented)
------------------------------
    attack = TokenTamperAttack(num_changes=3)
    result = attack.run(model, tokenizer, prompt, num_tokens=30)
"""

from __future__ import annotations

from adversarial_suite.attacks.base import AttackResult  # noqa: F401


class TokenTamperAttack:
    """
    Stub for the token-tampering attack.

    The provider shifts a small number of output token IDs after honest
    inference.  The masks remain honest, so the verifier replay produces
    the real tokens and the comparison fails.

    This is the "always detected" baseline — useful for verifying that
    the verification protocol is working correctly.

    Not yet implemented.  Will be added alongside the random-mask
    baseline experiments.
    """

    def __init__(self, num_changes: int = 3, shift: int = 7919):
        self.num_changes = num_changes
        self.shift = shift

    def run(self, model, tokenizer, prompt: str, **kwargs) -> AttackResult:
        raise NotImplementedError(
            "TokenTamperAttack is not yet implemented.  "
            "See module docstring for the planned interface."
        )
