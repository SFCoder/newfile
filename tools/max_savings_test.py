#!/usr/bin/env python3
"""
max_savings_test.py
===================
Quantifies the maximum computation an attacker can skip while still
producing output that passes verification and remains coherent.

Attack model
------------
The provider zeroes the self-attention output at N selected transformer
layers (replacing it with zeros, which is equivalent to skipping that
layer's attention).  The FFN at every layer is computed honestly.  The
verification protocol checks the FFN (via neuron masks) but not the
attention.  This experiment measures how many layers of attention can be
skipped before the output degrades or becomes detectably wrong.

Usage
-----
    # Full sweep on default model (Qwen/Qwen2.5-7B)
    python3 tools/max_savings_test.py

    # All registered models
    python3 tools/max_savings_test.py --models all

    # Specific model with 4-bit quantization
    python3 tools/max_savings_test.py --model Qwen/Qwen2.5-72B --quantize

    # Specific model
    python3 tools/max_savings_test.py --model Qwen/Qwen2.5-3B

    # Quick targeted test
    python3 tools/max_savings_test.py --skip-counts 1,5,10 --complexity simple
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH  # noqa: E402
from verifier import VerificationBundle                           # noqa: E402
from provider import ACTIVATION_THRESHOLD                         # noqa: E402
from adversarial_suite.attacks.attention_skip import AttentionSkipAttack  # noqa: E402
from adversarial_suite.verification.local import LocalVerification        # noqa: E402
from adversarial_suite.metrics import compute as metrics_compute          # noqa: E402
from adversarial_suite.metrics import reporting                           # noqa: E402

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SIMPLE_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius because",
    "The color of the sky is blue due to",
]

MODERATE_PROMPTS = [
    "Explain how a car engine converts fuel into motion",
    "Describe the process by which plants convert sunlight into energy",
    "Explain why the moon appears to change shape throughout the month",
]

COMPLEX_PROMPTS = [
    (
        "Alice is taller than Bob. Bob is taller than Carol. "
        "Carol is taller than David. David is taller than Eve. "
        "Who is the third tallest person and why?"
    ),
    (
        "If a train leaves Chicago at 9am traveling east at 60mph, "
        "and another train leaves New York at 10am traveling west at 80mph, "
        "and the cities are 790 miles apart, at what time do the trains meet?"
    ),
    (
        "A farmer has 100 feet of fencing and wants to enclose a rectangular "
        "area against a river (no fence needed on the river side). "
        "What dimensions maximize the enclosed area?"
    ),
]

PROMPT_TIERS = {
    "simple": SIMPLE_PROMPTS,
    "moderate": MODERATE_PROMPTS,
    "complex": COMPLEX_PROMPTS,
}

ALL_PROMPTS = SIMPLE_PROMPTS + MODERATE_PROMPTS + COMPLEX_PROMPTS

# ---------------------------------------------------------------------------
# Default skip counts
# ---------------------------------------------------------------------------

BASE_SKIP_COUNTS = [1, 2, 3, 5, 7, 10, 14]  # "all" appended dynamically

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_DIR = _ROOT / "analysis_results" / "max_savings"
ATTN_TRUST_DIR = _ROOT / "analysis_results" / "attention_trust"

NUM_TOKENS = 30
RANDOM_SEEDS = [0, 1, 2]


# ===========================================================================
# ModelContext (mirrors demo.py)
# ===========================================================================


class ModelContext:
    """Load a model through the registry; free memory on exit."""

    def __init__(self, model_id: str, registry: ModelRegistry, quantize: bool = False):
        self.model_id = model_id
        self._registry = registry
        self._quantize = quantize
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "ModelContext":
        print(f"  [LOAD] {self.model_id} …")
        if self._quantize:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            entry = self._registry.get_entry(self.model_id)
            snap = ModelRegistry._snapshot_path(entry.hf_repo)
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(snap))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                str(snap), quantization_config=bnb_cfg, device_map="auto"
            )
            self.model.eval()
        else:
            self.model, self.tokenizer = self._registry.load_verified_model(
                self.model_id
            )
        print(f"  [LOAD] ✓ loaded {self.model_id}")
        return self

    def __exit__(self, *_):
        print(f"  [UNLOAD] freeing {self.model_id} …")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [UNLOAD] done.")


# ===========================================================================
# Layer selection strategies
# ===========================================================================


def load_attention_trust_data(model_id: str) -> Optional[dict]:
    """
    Load per-layer attention tolerance data from a previous
    attention_trust_test run (if available).

    Returns a dict mapping layer_index (int) -> token_match_rate (float),
    or None if no data is available for this model.

    Expected file format::

        {
          "results": {
            "Qwen/Qwen2.5-7B": {
              "layer_tolerances": [0.95, 0.88, ...]
            }
          }
        }
    """
    results_path = ATTN_TRUST_DIR / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path) as fh:
            data = json.load(fh)
        # Support both top-level model key and nested "results" key
        model_data = (
            data.get("results", {}).get(model_id)
            or data.get(model_id)
        )
        if model_data is None:
            return None
        tolerances = model_data.get("layer_tolerances", [])
        if not tolerances:
            return None
        return {i: float(v) for i, v in enumerate(tolerances)}
    except Exception:
        return None


def select_layers_best_case(
    skip_count: int,
    num_layers: int,
    layer_tolerances: Optional[dict],
    fallback_seed: int = 42,
) -> tuple:
    """
    Return (layers, used_best_case_data: bool).

    If layer_tolerances is available, skip the most tolerant layers first
    (highest token match with zeroed attention = most tolerant).
    Otherwise fall back to a fixed random draw and return used_data=False.
    """
    if layer_tolerances:
        sorted_layers = sorted(
            range(num_layers),
            key=lambda l: -layer_tolerances.get(l, 0.0),
        )
        return sorted_layers[:skip_count], True
    rng = random.Random(fallback_seed)
    return rng.sample(range(num_layers), min(skip_count, num_layers)), False


def select_layers_random(skip_count: int, num_layers: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(range(num_layers), min(skip_count, num_layers))


def select_layers_positional_first(skip_count: int, num_layers: int) -> list:
    return list(range(min(skip_count, num_layers)))


def select_layers_positional_middle(skip_count: int, num_layers: int) -> list:
    n = min(skip_count, num_layers)
    start = max(0, (num_layers - n) // 2)
    return list(range(start, start + n))


def select_layers_positional_last(skip_count: int, num_layers: int) -> list:
    n = min(skip_count, num_layers)
    return list(range(num_layers - n, num_layers))


def build_strategy_runs(
    skip_count: int,
    num_layers: int,
    layer_tolerances: Optional[dict],
) -> list:
    """
    Return a list of (strategy_name, layers_to_skip) tuples for all
    strategies at this skip_count.

    Strategies:
      - best_case       (1 run; falls back to random if no prior data)
      - random_seed_0/1/2  (3 runs)
      - positional_first / positional_middle / positional_last  (3 runs)
    """
    runs = []

    # best_case
    layers, used_data = select_layers_best_case(
        skip_count, num_layers, layer_tolerances
    )
    name = "best_case" if used_data else "best_case(fallback_random)"
    runs.append((name, sorted(layers)))

    # random
    for seed in RANDOM_SEEDS:
        layers = select_layers_random(skip_count, num_layers, seed)
        runs.append((f"random_seed_{seed}", sorted(layers)))

    # positional
    runs.append(
        ("positional_first", select_layers_positional_first(skip_count, num_layers))
    )
    runs.append(
        ("positional_middle", select_layers_positional_middle(skip_count, num_layers))
    )
    runs.append(
        ("positional_last", select_layers_positional_last(skip_count, num_layers))
    )

    return runs


# ===========================================================================
# Per-prompt helpers (honest inference + logit capture)
# ===========================================================================


def generate_honest(model, tokenizer, prompt: str, num_tokens: int = NUM_TOKENS):
    """
    Run honest greedy generation and capture logits via lm_head hook.

    Returns (token_ids, logits_list) where logits_list has num_tokens
    entries, each a CPU float tensor of shape (vocab_size,).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    captured: list = []

    def logit_hook(module, inp, output):
        captured.append(output[0, -1].detach().cpu())

    hook = model.lm_head.register_forward_hook(logit_hook)
    try:
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=num_tokens,
                do_sample=False,
            )
    finally:
        hook.remove()

    tokens = output[0][prompt_len:].tolist()
    return tokens, captured  # captured has num_tokens logit vectors


def compute_forward_logits(
    model,
    tokenizer,
    prompt: str,
    generated_tokens: list,
    zeroed_layers: Optional[set] = None,
) -> list:
    """
    Run a single forward pass on (prompt + generated_tokens) and return
    the logit vectors for each generated token position.

    If zeroed_layers is provided, attention at those layers is zeroed
    during the forward pass.

    Returns a list of num_tokens CPU float tensors of shape (vocab_size,).
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    all_ids = input_ids + generated_tokens
    all_tensor = torch.tensor([all_ids], dtype=torch.long).to(device)
    prompt_len = len(input_ids)
    num_tokens = len(generated_tokens)

    hooks = []
    if zeroed_layers:
        for idx in zeroed_layers:
            if idx < len(model.model.layers):
                def _make_zero_hook():
                    def h(module, inp, out):
                        if isinstance(out, tuple):
                            return (torch.zeros_like(out[0]),) + out[1:]
                        return torch.zeros_like(out)
                    return h
                hooks.append(
                    model.model.layers[idx].self_attn.register_forward_hook(
                        _make_zero_hook()
                    )
                )

    try:
        with torch.no_grad():
            out = model(all_tensor)
        # logits shape: [1, seq_len, vocab_size]
        # Position p predicts token at p+1; for generated_tokens[i]
        # we want logits at position prompt_len-1+i
        logits_slice = out.logits[0, prompt_len - 1: prompt_len - 1 + num_tokens]
        return [logits_slice[i].cpu() for i in range(logits_slice.shape[0])]
    finally:
        for h in hooks:
            h.remove()


def compute_perplexity(
    honest_logits_for_fraudulent: list,
    fraudulent_tokens: list,
) -> float:
    """
    Compute perplexity of the fraudulent token sequence scored by the
    honest model.

    Parameters
    ----------
    honest_logits_for_fraudulent
        Logits from the honest model run on (prompt + fraudulent_tokens).
        List of num_tokens logit vectors.
    fraudulent_tokens
        The token IDs to score.
    """
    log_probs = metrics_compute.compute_log_probs_from_logits(
        honest_logits_for_fraudulent, fraudulent_tokens
    )
    return metrics_compute.perplexity_from_log_probs(log_probs)


# ===========================================================================
# Core experiment loop for one model
# ===========================================================================


def run_model_experiment(
    model_id: str,
    model,
    tokenizer,
    skip_counts: list,
    complexity_filter: Optional[str],
) -> list:
    """
    Run the full attention-skip sweep for one loaded model.

    Returns a list of result dicts, one per (skip_count, strategy, prompt).
    """
    entry_info = {
        "model_id": model_id,
        "num_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }
    num_layers = entry_info["num_layers"]
    hidden_size = entry_info["hidden_size"]

    # Resolve prompt tiers
    if complexity_filter:
        tiers_to_run = {
            k: v for k, v in PROMPT_TIERS.items() if k == complexity_filter
        }
    else:
        tiers_to_run = PROMPT_TIERS

    # Add "all layers" to skip counts
    all_skip_counts = sorted(set(skip_counts + [num_layers]))
    all_skip_counts = [s for s in all_skip_counts if s <= num_layers]

    # Load per-layer tolerance data if available
    layer_tolerances = load_attention_trust_data(model_id)
    if layer_tolerances:
        print(f"  [INFO] Loaded attention trust data for {model_id}")
    else:
        print(f"  [INFO] No attention trust data for {model_id}; best_case falls back to random")

    attack = AttentionSkipAttack()
    verifier_target = LocalVerification(model, tokenizer)

    # Pre-compute honest tokens and logits for all prompts
    print(f"  [INFO] Pre-computing honest inference for all prompts …")
    honest_cache: dict = {}  # prompt -> (tokens, logits)
    for tier, prompts in tiers_to_run.items():
        for prompt in prompts:
            t0 = time.perf_counter()
            tokens, logits = generate_honest(model, tokenizer, prompt, NUM_TOKENS)
            elapsed = time.perf_counter() - t0
            honest_cache[prompt] = (tokens, logits)
            short_p = prompt[:50].replace("\n", " ")
            print(f"    ✓ {short_p!r}  ({elapsed:.1f}s)")

    results: list = []
    total_configs = len(all_skip_counts) * len(tiers_to_run) * sum(
        len(p) for p in tiers_to_run.values()
    )
    run_idx = 0

    for skip_count in all_skip_counts:
        strategy_runs = build_strategy_runs(skip_count, num_layers, layer_tolerances)

        for strategy_name, layers_to_skip in strategy_runs:
            for tier, prompts in tiers_to_run.items():
                for prompt in prompts:
                    run_idx += 1
                    honest_tokens, honest_logits = honest_cache[prompt]

                    t0 = time.perf_counter()

                    # --- Run attack -----------------------------------------
                    try:
                        attack_result = attack.run(
                            model, tokenizer, prompt,
                            layers_to_skip=layers_to_skip,
                            num_tokens=NUM_TOKENS,
                            model_id=model_id,
                            honest_tokens=honest_tokens,
                            capture_logits=True,
                        )
                    except Exception as exc:
                        print(f"    [ERROR] attack failed: {exc}")
                        continue

                    fraudulent_tokens = attack_result.fraudulent_tokens
                    fraudulent_logits = attack_result.fraudulent_logits  # from lm_head hook

                    # --- Build fraudulent bundle ----------------------------
                    fraud_bundle = attack.build_fraudulent_bundle(
                        attack_result, model_id
                    )

                    # --- Verify ---------------------------------------------
                    try:
                        v_result = verifier_target.verify(fraud_bundle)
                    except Exception as exc:
                        print(f"    [ERROR] verification failed: {exc}")
                        v_result = None

                    # --- Cosine similarity (honest logits vs attack logits) -
                    cos_sim: Optional[float] = None
                    if honest_logits and fraudulent_logits:
                        try:
                            cos_sim = metrics_compute.mean_cosine_similarity(
                                honest_logits, fraudulent_logits
                            )
                        except Exception:
                            pass

                    # --- Perplexity (honest model scoring fraudulent tokens) -
                    perplexity: Optional[float] = None
                    try:
                        honest_for_ppl = compute_forward_logits(
                            model, tokenizer, prompt, fraudulent_tokens,
                            zeroed_layers=None,
                        )
                        perplexity = compute_perplexity(honest_for_ppl, fraudulent_tokens)
                    except Exception:
                        pass

                    # --- Metrics -------------------------------------------
                    match_rate = metrics_compute.token_match_rate(
                        honest_tokens, fraudulent_tokens
                    )
                    coherence = metrics_compute.classify_coherence(match_rate)
                    savings_pct = metrics_compute.attacker_savings_pct(
                        skip_count, num_layers
                    )

                    elapsed = time.perf_counter() - t0

                    result_row = {
                        "model_id": model_id,
                        "tier": tier,
                        "prompt": prompt,
                        "skip_count": skip_count,
                        "strategy": strategy_name,
                        "layers_skipped": layers_to_skip,
                        "honest_tokens": honest_tokens,
                        "fraudulent_tokens": fraudulent_tokens,
                        "token_match_rate": round(match_rate, 4),
                        "coherence": coherence,
                        "savings_pct": round(savings_pct, 2),
                        "mean_cosine_similarity": (
                            round(cos_sim, 4) if cos_sim is not None else None
                        ),
                        "perplexity": (
                            round(perplexity, 2) if perplexity is not None else None
                        ),
                        "verification_passed": (
                            v_result.verified if v_result else None
                        ),
                        "verification_match_rate": (
                            round(v_result.token_match_rate, 4)
                            if v_result else None
                        ),
                        "elapsed_s": round(elapsed, 1),
                        "metadata": {
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                        },
                    }
                    results.append(result_row)

                    # Progress line
                    short_p = prompt[:35].replace("\n", " ")
                    ver_sym = (
                        "✓" if (v_result and v_result.verified) else "✗"
                        if v_result else "?"
                    )
                    print(
                        f"    skip={skip_count:2d} | {strategy_name:<28} | "
                        f"{tier:<8} | match={match_rate:.1%} | {coherence:<9} | "
                        f"savings={savings_pct:.1f}% | ver={ver_sym} | "
                        f"{elapsed:.1f}s"
                    )

    return results


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify maximum attacker savings from attention skipping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID to run (default: Qwen/Qwen2.5-7B).",
    )
    parser.add_argument(
        "--models",
        metavar="all",
        help="Pass 'all' to run every model in the registry.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load the model with 4-bit quantization (bitsandbytes).",
    )
    parser.add_argument(
        "--skip-counts",
        default=None,
        help="Comma-separated skip counts, e.g. '1,5,10' (default: full sweep).",
    )
    parser.add_argument(
        "--complexity",
        choices=["simple", "moderate", "complex"],
        default=None,
        help="Restrict to a single prompt complexity tier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    registry = ModelRegistry(DEFAULT_REGISTRY_PATH)

    # Determine which models to run
    if args.models == "all":
        all_ids = registry.list_models()
        # Sort smallest to largest by hidden_size
        def _hs(mid):
            try:
                return registry.get_entry(mid).hidden_size
            except Exception:
                return 0
        model_ids = sorted(all_ids, key=_hs)
        print(f"Running on all {len(model_ids)} models: {model_ids}")
    else:
        model_ids = [args.model]

    # Skip counts
    if args.skip_counts:
        skip_counts = [int(x.strip()) for x in args.skip_counts.split(",")]
    else:
        skip_counts = list(BASE_SKIP_COUNTS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list = []

    for model_id in model_ids:
        print()
        print("=" * 70)
        print(f"  MODEL: {model_id}")
        print("=" * 70)

        try:
            with ModelContext(model_id, registry, quantize=args.quantize) as ctx:
                model_results = run_model_experiment(
                    model_id=model_id,
                    model=ctx.model,
                    tokenizer=ctx.tokenizer,
                    skip_counts=skip_counts,
                    complexity_filter=args.complexity,
                )
        except KeyError:
            print(f"  [SKIP] {model_id!r} not in registry.")
            continue
        except Exception as exc:
            print(f"  [ERROR] Failed to run {model_id}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        all_results.extend(model_results)

        # Checkpoint after each model
        reporting.save_results_json(
            all_results, OUT_DIR / "results.json"
        )

    if not all_results:
        print("No results collected.  Exiting.")
        return

    # --- Save outputs -------------------------------------------------------
    reporting.save_results_json(all_results, OUT_DIR / "results.json")
    reporting.save_summary_csv(all_results, OUT_DIR / "summary.csv")
    reporting.save_attacker_optimal_csv(all_results, OUT_DIR / "attacker_optimal.csv")

    # --- Print tables -------------------------------------------------------
    reporting.print_summary_table(all_results)
    reporting.print_attacker_optimal(all_results)

    # --- Cross-model comparison plot (only meaningful with multiple models) --
    unique_models = {r["model_id"] for r in all_results}
    if len(unique_models) > 1:
        reporting.plot_model_comparison(
            all_results, OUT_DIR / "model_comparison.png"
        )

    print(f"\n  All outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
