#!/usr/bin/env python3
"""
attention_trust_test.py
=======================
Tests whether the FFN produces coherent output when given fake attention inputs.

Hypothesis
----------
Attention is where the model builds context.  If you feed fake attention output
into the real FFN, the FFN should produce garbage because it is transforming a
meaningless representation.  If that's true, an attacker who skips real
attention and injects fake values gets caught automatically.

The reverse question is also tested: is fake FFN output equally destructive, or
is attention uniquely critical?

Fake attention types
--------------------
  wrong_prompt  — real attention, but computed on a different prompt
  small_model   — attention from Qwen2.5-0.5B (skipped if hidden_size differs)
  random        — Gaussian noise matched to the real attention's mean / std
  zeros         — all zeros

Fake FFN
--------
  random        — Gaussian noise matched to real MLP output statistics

What is injected
----------------
The hook replaces the self-attention module's output tensor (shape [1, seq, H])
before it is added to the residual stream.  The residual up to the target layer
is always the real value; only the attention contribution at that layer is fake.

CLI
---
  python3 attention_trust_test.py
  python3 attention_trust_test.py --model Qwen/Qwen2.5-72B --quantize
  python3 attention_trust_test.py --layer 14 --fake-type random

Outputs
-------
  Printed generated text for each scenario
  Summary table: prompt × layer × fake_type → token_match, cosine_sim, coherence
  analysis_results/attention_trust/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token id.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS: list[str] = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
]

# Default target layers for 7B (28 total, indices 0-27)
DEFAULT_LAYERS_7B: list[int] = [5, 14, 23]
# For 72B (80 total, indices 0-79)
DEFAULT_LAYERS_72B: list[int] = [10, 40, 70]

N_GEN_TOKENS = 20

SMALL_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B"

ALL_FAKE_TYPES  = ["wrong_prompt", "small_model", "random", "zeros"]
ALL_INTERVENTION = ["fake_attn", "fake_ffn"]

OUTPUT_DIR = Path("analysis_results/attention_trust")

COHERENCE_THRESHOLDS = (0.50, 0.10)   # above 50% → coherent, 10-50% → degraded


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    tokens: list[int]
    text: str
    # logits[step] = float32 CPU tensor [vocab], one per generated token
    logits: list[torch.Tensor] = field(default_factory=list)
    perplexity: float = float("nan")


@dataclass
class TestRecord:
    prompt_idx: int
    prompt: str
    target_layer: int
    intervention: str      # "fake_attn" | "fake_ffn"
    fake_type: str         # "wrong_prompt" | "small_model" | "random" | "zeros" | "n/a"
    skipped: bool
    skip_reason: str
    token_match_rate: float
    avg_cosine_sim: float
    perplexity: float
    generated_text: str
    coherence: str         # "coherent" | "degraded" | "garbage"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_main_model(model_id: str, quantize: bool, registry_path: Path):
    """
    Load the target model through the registry (hash-verified).
    If --quantize is set, load in 4-bit using bitsandbytes (CUDA only).
    """
    registry = ModelRegistry(registry_path)
    print(f"Loading {model_id} …")
    if quantize:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit("bitsandbytes not installed — run: pip install bitsandbytes")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model, tokenizer = registry.load_verified_model(
            model_id, dtype=torch.float16
        )
        # quantize flag requires manual quantization config; just warn
        print("NOTE: --quantize passed but registry.load_verified_model does not "
              "accept quantization config directly.  "
              "Edit load_verified_model or load manually if 4-bit is needed.")
    else:
        model, tokenizer = registry.load_verified_model(model_id, dtype=torch.float16)
    model.eval()
    device = next(model.parameters()).device
    print(f"  loaded on {device}  "
          f"layers={len(model.model.layers)}  "
          f"hidden={model.config.hidden_size}  "
          f"intermediate={model.config.intermediate_size}")
    return model, tokenizer, device


def load_small_model(registry_path: Path):
    """
    Load Qwen2.5-0.5B for fake-type 'small_model'.
    Does NOT go through the registry (may not be registered); uses HF directly.
    Returns (model, tokenizer, device) or raises.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {SMALL_MODEL_ID} for small-model fake-type …")
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, torch_dtype=torch.float16)
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) \
             else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  small model on {device}  hidden={model.config.hidden_size}")
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# Manual greedy generation (returns tokens + per-step logits)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,         # [1, seq] on device
    n_tokens: int,
    attn_hook_fn: Optional[Callable] = None,
    ffn_hook_fn: Optional[Callable]  = None,
    target_layer: Optional[int]       = None,
) -> tuple[list[int], list[torch.Tensor]]:
    """
    Autoregressive greedy generation with optional hooks on one layer.

    attn_hook_fn(module, inp, out, step) → replacement out tuple
    ffn_hook_fn(module, inp, out, step) → replacement tensor

    Both hooks receive the current step number (0 = prefill / first token).
    Hooks are installed on model.model.layers[target_layer].self_attn or .mlp.

    Returns (list of token IDs, list of [vocab] float32 logit tensors).
    """
    device      = next(model.parameters()).device
    input_ids   = prompt_ids.to(device)
    attn_mask   = torch.ones_like(input_ids)
    past_kv     = None
    gen_tokens: list[int]          = []
    gen_logits: list[torch.Tensor] = []

    step_counter = [0]

    def make_attn_hook(user_fn):
        def _hook(module, inp, out):
            result = user_fn(module, inp, out, step_counter[0])
            return result
        return _hook

    def make_ffn_hook(user_fn):
        def _hook(module, inp, out):
            result = user_fn(module, inp, out, step_counter[0])
            return result
        return _hook

    hooks = []
    if attn_hook_fn is not None and target_layer is not None:
        hooks.append(
            model.model.layers[target_layer].self_attn.register_forward_hook(
                make_attn_hook(attn_hook_fn)
            )
        )
    if ffn_hook_fn is not None and target_layer is not None:
        hooks.append(
            model.model.layers[target_layer].mlp.register_forward_hook(
                make_ffn_hook(ffn_hook_fn)
            )
        )

    try:
        for _ in range(n_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            logits_last = out.logits[0, -1, :].float().cpu()
            next_tok    = int(logits_last.argmax().item())
            gen_tokens.append(next_tok)
            gen_logits.append(logits_last)

            past_kv   = out.past_key_values
            input_ids = torch.tensor([[next_tok]], device=device, dtype=torch.long)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1
            )
            step_counter[0] += 1
    finally:
        for h in hooks:
            h.remove()

    return gen_tokens, gen_logits


# ─────────────────────────────────────────────────────────────────────────────
# Capture helpers
# ─────────────────────────────────────────────────────────────────────────────

def capture_attn_output(model, prompt_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Single prefill pass; returns the self_attn output at layer_idx.
    Shape: [1, seq, hidden_size]  float32 CPU.
    """
    device   = next(model.parameters()).device
    captured = [None]

    def hook(module, inp, out):
        captured[0] = out[0].detach().float().cpu()
        return out

    h = model.model.layers[layer_idx].self_attn.register_forward_hook(hook)
    with torch.no_grad():
        model(
            input_ids=prompt_ids.to(device),
            attention_mask=torch.ones_like(prompt_ids.to(device)),
        )
    h.remove()
    return captured[0]   # [1, seq, H]


def capture_ffn_output(model, prompt_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Single prefill pass; returns the MLP output at layer_idx.
    Shape: [1, seq, hidden_size]  float32 CPU.
    """
    device   = next(model.parameters()).device
    captured = [None]

    def hook(module, inp, out):
        captured[0] = out.detach().float().cpu()
        return out

    h = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(
            input_ids=prompt_ids.to(device),
            attention_mask=torch.ones_like(prompt_ids.to(device)),
        )
    h.remove()
    return captured[0]   # [1, seq, H]


# ─────────────────────────────────────────────────────────────────────────────
# Hook factories for each fake type
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_seq(fake: torch.Tensor, real_seq: int, device, dtype) -> torch.Tensor:
    """
    Adjust fake tensor's sequence dimension to match real_seq by truncating or
    zero-padding, then cast to (device, dtype).
    fake: [1, fake_seq, H]
    """
    H        = fake.shape[2]
    fake_seq = fake.shape[1]
    if fake_seq == real_seq:
        return fake.to(device=device, dtype=dtype)
    elif fake_seq > real_seq:
        return fake[:, :real_seq, :].to(device=device, dtype=dtype)
    else:
        pad = torch.zeros(1, real_seq - fake_seq, H, dtype=dtype, device=device)
        return torch.cat([fake.to(device=device, dtype=dtype), pad], dim=1)


def make_zeros_attn_hook():
    """Replace attention output with zeros at every step."""
    def hook_fn(module, inp, out, step):
        real = out[0]
        return (torch.zeros_like(real),) + out[1:]
    return hook_fn


def make_zeros_ffn_hook():
    def hook_fn(module, inp, out, step):
        return torch.zeros_like(out)
    return hook_fn


def make_random_attn_hook(real_attn_prefill: torch.Tensor):
    """
    Replace attention output with Gaussian noise matched to the prefill
    statistics (mean / std computed across the full prefill tensor).
    At step 0 (prefill): shape [1, seq, H]; at step k>0: shape [1, 1, H].
    """
    mean = float(real_attn_prefill.mean().item())
    std  = float(real_attn_prefill.std().item())

    def hook_fn(module, inp, out, step):
        real = out[0]
        noise = torch.randn_like(real) * std + mean
        return (noise,) + out[1:]
    return hook_fn


def make_random_ffn_hook(real_ffn_prefill: torch.Tensor):
    mean = float(real_ffn_prefill.mean().item())
    std  = float(real_ffn_prefill.std().item())

    def hook_fn(module, inp, out, step):
        noise = torch.randn_like(out) * std + mean
        return noise
    return hook_fn


def make_wrong_prompt_attn_hook(wrong_attn_prefill: torch.Tensor):
    """
    Step 0 (prefill): inject wrong_attn_prefill (seq-adjusted).
    Step k>0: inject zeros (wrong prompt's per-step dynamics are irrelevant).
    """
    def hook_fn(module, inp, out, step):
        real = out[0]
        if step == 0:
            fake = _adjust_seq(wrong_attn_prefill, real.shape[1],
                               real.device, real.dtype)
        else:
            fake = torch.zeros_like(real)
        return (fake,) + out[1:]
    return hook_fn


def make_small_model_attn_hook(small_attn_prefill: torch.Tensor):
    """
    Inject small model's attention output.  Only called when hidden sizes match.
    Same step-0 / later logic as wrong_prompt.
    """
    def hook_fn(module, inp, out, step):
        real = out[0]
        if step == 0:
            fake = _adjust_seq(small_attn_prefill, real.shape[1],
                               real.device, real.dtype)
        else:
            fake = torch.zeros_like(real)
        return (fake,) + out[1:]
    return hook_fn


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    prompt_ids: torch.Tensor,   # [1, prompt_seq]  on CPU
    gen_token_ids: list[int],
) -> float:
    """
    Perplexity of generated tokens conditioned on the prompt.
    Prompt positions are masked out (label = -100).
    """
    if not gen_token_ids:
        return float("nan")
    device  = next(model.parameters()).device
    gen_t   = torch.tensor([gen_token_ids], dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids.to(device), gen_t], dim=1)
    labels  = torch.cat([
        torch.full((1, prompt_ids.shape[1]), -100, dtype=torch.long, device=device),
        gen_t,
    ], dim=1)
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    loss = out.loss.item()
    return math.exp(min(loss, 20))   # cap at exp(20) to avoid inf for pure garbage


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    gt: GenerationResult,
    test_tokens: list[int],
    test_logits: list[torch.Tensor],
) -> tuple[float, float, str]:
    """
    Returns (token_match_rate, avg_cosine_sim, coherence_label).
    """
    n = min(len(gt.tokens), len(test_tokens), len(test_logits), len(gt.logits))
    if n == 0:
        return 0.0, 0.0, "garbage"

    # Token match rate
    matches = sum(1 for a, b in zip(gt.tokens[:n], test_tokens[:n]) if a == b)
    token_match = matches / n

    # Average cosine similarity of logit vectors
    cos_sims: list[float] = []
    for gl, tl in zip(gt.logits[:n], test_logits[:n]):
        cos = F.cosine_similarity(gl.unsqueeze(0), tl.unsqueeze(0)).item()
        cos_sims.append(cos)
    avg_cos = sum(cos_sims) / len(cos_sims)

    # Coherence label
    hi, lo = COHERENCE_THRESHOLDS
    if token_match >= hi:
        coherence = "coherent"
    elif token_match >= lo:
        coherence = "degraded"
    else:
        coherence = "garbage"

    return token_match, avg_cos, coherence


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth generation
# ─────────────────────────────────────────────────────────────────────────────

def get_ground_truth(
    model, tokenizer, prompt: str, n_tokens: int, device
) -> GenerationResult:
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    tokens, logits = greedy_generate(
        model, tokenizer, prompt_ids, n_tokens,
        attn_hook_fn=None, ffn_hook_fn=None, target_layer=None,
    )
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    ppl  = compute_perplexity(model, prompt_ids.cpu(), tokens)

    return GenerationResult(tokens=tokens, text=text, logits=logits, perplexity=ppl)


# ─────────────────────────────────────────────────────────────────────────────
# Core test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fake_attn_test(
    model,
    tokenizer,
    prompt_idx: int,
    prompt: str,
    target_layer: int,
    fake_type: str,
    gt: GenerationResult,
    # pre-captured data (may be None if unavailable)
    wrong_attn: Optional[torch.Tensor],
    small_attn: Optional[torch.Tensor],
    real_attn_prefill: Optional[torch.Tensor],
    device,
) -> TestRecord:
    """Run one fake-attention injection experiment, return a TestRecord."""
    skipped     = False
    skip_reason = ""
    hook_fn     = None

    if fake_type == "zeros":
        hook_fn = make_zeros_attn_hook()

    elif fake_type == "random":
        if real_attn_prefill is None:
            skipped     = True
            skip_reason = "real_attn_prefill not captured"
        else:
            hook_fn = make_random_attn_hook(real_attn_prefill)

    elif fake_type == "wrong_prompt":
        if wrong_attn is None:
            skipped     = True
            skip_reason = "wrong_attn not captured"
        else:
            hook_fn = make_wrong_prompt_attn_hook(wrong_attn)

    elif fake_type == "small_model":
        if small_attn is None:
            skipped     = True
            skip_reason = "small_model hidden_size incompatible or model unavailable"
        else:
            hook_fn = make_small_model_attn_hook(small_attn)

    if skipped:
        return TestRecord(
            prompt_idx=prompt_idx, prompt=prompt, target_layer=target_layer,
            intervention="fake_attn", fake_type=fake_type,
            skipped=True, skip_reason=skip_reason,
            token_match_rate=float("nan"), avg_cosine_sim=float("nan"),
            perplexity=float("nan"), generated_text="",
            coherence="n/a",
        )

    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    tokens, logits = greedy_generate(
        model, tokenizer, prompt_ids, N_GEN_TOKENS,
        attn_hook_fn=hook_fn, target_layer=target_layer,
    )
    text  = tokenizer.decode(tokens, skip_special_tokens=True)
    ppl   = compute_perplexity(model, prompt_ids.cpu(), tokens)
    tmr, cos, coherence = compute_metrics(gt, tokens, logits)

    return TestRecord(
        prompt_idx=prompt_idx, prompt=prompt, target_layer=target_layer,
        intervention="fake_attn", fake_type=fake_type,
        skipped=False, skip_reason="",
        token_match_rate=tmr, avg_cosine_sim=cos,
        perplexity=ppl, generated_text=text,
        coherence=coherence,
    )


def run_fake_ffn_test(
    model,
    tokenizer,
    prompt_idx: int,
    prompt: str,
    target_layer: int,
    gt: GenerationResult,
    real_ffn_prefill: Optional[torch.Tensor],
    device,
) -> TestRecord:
    """Run one fake-FFN injection experiment (random noise only)."""
    if real_ffn_prefill is None:
        return TestRecord(
            prompt_idx=prompt_idx, prompt=prompt, target_layer=target_layer,
            intervention="fake_ffn", fake_type="random",
            skipped=True, skip_reason="real_ffn_prefill not captured",
            token_match_rate=float("nan"), avg_cosine_sim=float("nan"),
            perplexity=float("nan"), generated_text="",
            coherence="n/a",
        )

    hook_fn   = make_random_ffn_hook(real_ffn_prefill)
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    tokens, logits = greedy_generate(
        model, tokenizer, prompt_ids, N_GEN_TOKENS,
        ffn_hook_fn=hook_fn, target_layer=target_layer,
    )
    text  = tokenizer.decode(tokens, skip_special_tokens=True)
    ppl   = compute_perplexity(model, prompt_ids.cpu(), tokens)
    tmr, cos, coherence = compute_metrics(gt, tokens, logits)

    return TestRecord(
        prompt_idx=prompt_idx, prompt=prompt, target_layer=target_layer,
        intervention="fake_ffn", fake_type="random",
        skipped=False, skip_reason="",
        token_match_rate=tmr, avg_cosine_sim=cos,
        perplexity=ppl, generated_text=text,
        coherence=coherence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def determine_target_layers(model, args_layers: Optional[list[int]]) -> list[int]:
    if args_layers:
        return args_layers
    n = len(model.model.layers)
    if n >= 70:
        return DEFAULT_LAYERS_72B
    return DEFAULT_LAYERS_7B


def run_all(
    model,
    tokenizer,
    device,
    target_layers: list[int],
    fake_types: list[str],
    small_model_info: Optional[tuple],  # (small_model, small_tokenizer, small_device, small_layer_map)
) -> list[TestRecord]:
    records: list[TestRecord] = []

    # Determine small model compatibility once
    main_hidden  = model.config.hidden_size
    small_hidden = small_model_info[0].config.hidden_size if small_model_info else None
    small_compat = (small_hidden == main_hidden) if small_hidden else False
    if "small_model" in fake_types and not small_compat:
        print(f"\n  Skipping fake_type=small_model: "
              f"main hidden_size={main_hidden}, "
              f"small model hidden_size={small_hidden or 'n/a'} — incompatible.\n")

    n_layers_main  = len(model.model.layers)
    n_layers_small = small_model_info[0].config.num_hidden_layers if small_model_info else None

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'━'*70}")
        print(f"Prompt {pi}: {prompt!r}")
        print(f"{'━'*70}")

        inputs    = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"].cpu()

        # Ground truth
        print("  [ground truth] generating …")
        gt = get_ground_truth(model, tokenizer, prompt, N_GEN_TOKENS, device)
        print(f"    → {gt.text!r}")
        print(f"    ppl={gt.perplexity:.1f}")

        # Wrong-prompt capture (use the *next* prompt cyclically)
        wrong_pi   = (pi + 1) % len(PROMPTS)
        wrong_ids  = tokenizer(PROMPTS[wrong_pi], return_tensors="pt")["input_ids"].cpu()

        for li in target_layers:
            print(f"\n  ── layer {li} ──")

            # Capture real attention and FFN outputs at this layer (for stats)
            real_attn_prefill = capture_attn_output(
                model, prompt_ids.to(device), li
            )
            real_ffn_prefill = capture_ffn_output(
                model, prompt_ids.to(device), li
            )

            # Wrong-prompt attention capture
            wrong_attn_prefill = capture_attn_output(
                model, wrong_ids.to(device), li
            )

            # Small model attention capture (if compatible)
            small_attn_prefill: Optional[torch.Tensor] = None
            if small_compat and small_model_info is not None:
                sm, st, sd, _ = small_model_info
                # Map target layer index proportionally to small model depth
                small_li = round(li * (n_layers_small - 1) / (n_layers_main - 1))
                small_input_ids = st(prompt, return_tensors="pt")["input_ids"].cpu()
                small_attn_prefill = capture_attn_output(sm, small_input_ids.to(sd), small_li)

            # ── Fake attention tests ─────────────────────────────────────────
            for ft in fake_types:
                print(f"    [fake_attn / {ft}] injecting at layer {li} …")
                rec = run_fake_attn_test(
                    model=model, tokenizer=tokenizer,
                    prompt_idx=pi, prompt=prompt,
                    target_layer=li, fake_type=ft, gt=gt,
                    wrong_attn=wrong_attn_prefill,
                    small_attn=small_attn_prefill,
                    real_attn_prefill=real_attn_prefill,
                    device=device,
                )
                records.append(rec)
                if rec.skipped:
                    print(f"      SKIPPED: {rec.skip_reason}")
                else:
                    print(f"      → {rec.generated_text!r}")
                    print(f"        token_match={rec.token_match_rate:.0%}  "
                          f"cos={rec.avg_cosine_sim:.4f}  "
                          f"ppl={rec.perplexity:.1f}  "
                          f"[{rec.coherence}]")

            # ── Fake FFN test ────────────────────────────────────────────────
            print(f"    [fake_ffn / random] injecting at layer {li} …")
            rec = run_fake_ffn_test(
                model=model, tokenizer=tokenizer,
                prompt_idx=pi, prompt=prompt,
                target_layer=li, gt=gt,
                real_ffn_prefill=real_ffn_prefill,
                device=device,
            )
            records.append(rec)
            if not rec.skipped:
                print(f"      → {rec.generated_text!r}")
                print(f"        token_match={rec.token_match_rate:.0%}  "
                      f"cos={rec.avg_cosine_sim:.4f}  "
                      f"ppl={rec.perplexity:.1f}  "
                      f"[{rec.coherence}]")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(records: list[TestRecord]) -> None:
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = (
        f"{'Prompt':>6}  {'Layer':>5}  {'Intervention':>12}  {'FakeType':>14}  "
        f"{'TokMatch':>8}  {'CosSim':>8}  {'PPL':>8}  {'Coherence':>10}"
    )
    print(header)
    print("-" * 100)

    for r in records:
        if r.skipped:
            row = (
                f"{r.prompt_idx:>6}  {r.target_layer:>5}  {r.intervention:>12}  "
                f"{r.fake_type:>14}  {'SKIPPED':>8}  {'':>8}  {'':>8}  {r.skip_reason[:10]:>10}"
            )
        else:
            row = (
                f"{r.prompt_idx:>6}  {r.target_layer:>5}  {r.intervention:>12}  "
                f"{r.fake_type:>14}  "
                f"{r.token_match_rate:>8.0%}  "
                f"{r.avg_cosine_sim:>8.4f}  "
                f"{r.perplexity:>8.1f}  "
                f"{r.coherence:>10}"
            )
        print(row)

    print(f"{'='*100}")

    # Aggregate by intervention + fake_type
    from collections import defaultdict
    groups: dict[tuple, list[TestRecord]] = defaultdict(list)
    for r in records:
        if not r.skipped:
            groups[(r.intervention, r.fake_type)].append(r)

    if groups:
        print("\nAggregate (mean across prompts & layers, non-skipped only):")
        print(f"  {'Intervention':>12}  {'FakeType':>14}  {'TokMatch':>8}  "
              f"{'CosSim':>8}  {'PPL':>8}  {'Coherence %':>11}")
        print("  " + "-" * 70)
        for (interv, ft), recs in sorted(groups.items()):
            tmr   = sum(r.token_match_rate for r in recs) / len(recs)
            cos   = sum(r.avg_cosine_sim   for r in recs) / len(recs)
            ppl   = sum(r.perplexity       for r in recs if not math.isnan(r.perplexity))
            ppl   = ppl / max(1, sum(1 for r in recs if not math.isnan(r.perplexity)))
            n_coh = sum(1 for r in recs if r.coherence == "coherent")
            print(f"  {interv:>12}  {ft:>14}  {tmr:>8.0%}  {cos:>8.4f}  "
                  f"{ppl:>8.1f}  {n_coh}/{len(recs)} coherent")
    print(f"{'='*100}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(records: list[TestRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    data = [asdict(r) for r in records]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Test FFN coherence under fake attention / FFN inputs."
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--quantize", action="store_true",
        help="Load in 4-bit (requires bitsandbytes, CUDA only)",
    )
    p.add_argument(
        "--layer", type=int, nargs="+",
        help="Target layer(s) to test (default: [5,14,23] for 7B, [10,40,70] for 72B)",
    )
    p.add_argument(
        "--fake-type", dest="fake_type", nargs="+",
        choices=ALL_FAKE_TYPES,
        help="Fake attention type(s) to test (default: all)",
    )
    p.add_argument(
        "--no-small-model", dest="no_small_model", action="store_true",
        help="Skip loading the 0.5B model even if small_model fake type is selected",
    )
    p.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
        help="Path to registry.json",
    )
    p.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for results.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer, device = load_main_model(args.model, args.quantize, args.registry)
    target_layers = determine_target_layers(model, args.layer)
    fake_types    = args.fake_type or ALL_FAKE_TYPES

    print(f"\nTest configuration:")
    print(f"  model:        {args.model}")
    print(f"  target layers:{target_layers}")
    print(f"  fake types:   {fake_types}")
    print(f"  n_gen_tokens: {N_GEN_TOKENS}")
    print(f"  prompts:      {len(PROMPTS)}")

    # Attempt to load small model if needed
    small_model_info = None
    if "small_model" in fake_types and not args.no_small_model:
        try:
            sm, st, sd = load_small_model(args.registry)
            small_model_info = (sm, st, sd, None)
        except Exception as e:
            print(f"  WARNING: Could not load small model ({e}). "
                  f"small_model fake type will be skipped.")

    records = run_all(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_layers=target_layers,
        fake_types=fake_types,
        small_model_info=small_model_info,
    )

    print_summary_table(records)
    save_results(records, args.output_dir)


if __name__ == "__main__":
    main()
