"""
Adversarial testing suite for proof-of-inference blockchain systems.

Provides a structured framework for testing attacks against LLM inference
verification protocols.  Designed to be used both for local experiments
and as a harness against live blockchain verification targets.

Structure
---------
attacks/       — Attack implementations with standardized interfaces.
                 Each attack returns an AttackResult dataclass.
verification/  — Verification targets.  LocalVerification runs the
                 verifier engine directly; a future CosmosVerification
                 will submit bundles to a Cosmos testnet via gRPC.
metrics/       — Pure functions for computing and reporting metrics.
                 No dependency on how the attack was performed or how
                 verification was done.

Typical usage::

    from adversarial_suite.attacks.attention_skip import AttentionSkipAttack
    from adversarial_suite.verification.local import LocalVerification
    from adversarial_suite.metrics import compute, reporting

    attack  = AttentionSkipAttack()
    result  = attack.run(model, tokenizer, prompt, layers_to_skip=[0, 1, 2])
    target  = LocalVerification(model, tokenizer)
    vresult = target.verify(result.fraudulent_bundle)
    metrics = compute.summarise(result, vresult)
"""
