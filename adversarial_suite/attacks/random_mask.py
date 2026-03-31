"""
RandomMaskAttack  (stub)
=========================
Attack: the provider submits randomly generated neuron masks instead of
masks derived from real inference.

If the verifier's replay happens to produce the same tokens as the
provider's claimed output (highly unlikely for large random masks), the
bundle passes.  More practically, this attack is used as a baseline to
quantify how much the mask structure constrains the output — if random
masks produce the same tokens as honest masks at some rate, the protocol
has a false-positive problem.

Interface (to be implemented)
------------------------------
    attack = RandomMaskAttack(seed=42, density=0.95)
    result = attack.run(model, tokenizer, prompt, num_tokens=30)
"""

from __future__ import annotations

from adversarial_suite.attacks.base import AttackResult  # noqa: F401


class RandomMaskAttack:
    """
    Stub for the random-mask fabrication attack.

    The provider submits randomly generated neuron masks.  The output
    tokens are from honest inference (so they look plausible), but the
    masks do not correspond to the actual computation.

    Not yet implemented.  Will be added in a future experiment on mask
    fabrication detection.
    """

    def __init__(self, seed: int = 42, density: float = 0.95):
        self.seed = seed
        self.density = density

    def run(self, model, tokenizer, prompt: str, **kwargs) -> AttackResult:
        raise NotImplementedError(
            "RandomMaskAttack is not yet implemented.  "
            "See module docstring for the planned interface."
        )
