"""
attacks/ — standardised attack implementations.

Every attack returns an AttackResult dataclass so that the verification
and metrics layers can process results without knowing the attack type.

Available attacks
-----------------
AttentionSkipAttack   — zero attention output at selected layers (implemented)
ModelSubstitutionAttack — run a cheaper model, fake FFN masks (stub)
RandomMaskAttack      — submit random neuron masks (stub)
TokenTamperAttack     — alter output token IDs (stub)
"""

from adversarial_suite.attacks.base import AttackResult
from adversarial_suite.attacks.attention_skip import AttentionSkipAttack
from adversarial_suite.attacks.model_substitution import ModelSubstitutionAttack
from adversarial_suite.attacks.random_mask import RandomMaskAttack
from adversarial_suite.attacks.token_tamper import TokenTamperAttack

__all__ = [
    "AttackResult",
    "AttentionSkipAttack",
    "ModelSubstitutionAttack",
    "RandomMaskAttack",
    "TokenTamperAttack",
]
