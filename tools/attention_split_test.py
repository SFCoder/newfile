#!/usr/bin/env python3
"""
attention_split_test.py
=======================
Tests whether attention head computation can be split across validators.

Key insight: the o_proj output equals the sum of per-head contributions:
    o_proj(x) = sum_n( x[n*hd:(n+1)*hd] @ O_weight[:, n*hd:(n+1)*hd].T ) + O_bias

This means each validator can compute a subset of heads independently and
the results can be summed to recover the full attention output.

Architecture defaults (Qwen2.5-7B):
    28 layers, 28 attention heads, head_dim=128, hidden_size=3584

Modes
-----
Capture mode (run on each hardware target):
    python3 tools/attention_split_test.py --capture --output attn_split_mac.json

Compare mode (cross-hardware analysis):
    python3 tools/attention_split_test.py --compare attn_split_mac.json attn_split_cuda.json

With quantization:
    python3 tools/attention_split_test.py --model Qwen/Qwen2.5-72B --quantize --capture --output attn_split_72B.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH          # noqa: E402
from adversarial_suite.db.writer import ResultsWriter                    # noqa: E402
from adversarial_suite.db.standard_format import write_standard_results  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Split groupings: (num_validators, heads_per_validator)
SPLIT_GROUPINGS = [(2, 14), (4, 7), (7, 4), (14, 2), (28, 1)]

CAPTURE_PROMPTS = [
    ("The capital of France is", "simple"),
    ("Water boils at 100 degrees Celsius because", "simple"),
    ("Explain how a neural network learns", "moderate"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is shortest?", "moderate"),
    ("Derive the formula for the sum of a geometric series step by step", "complex"),
]

OUT_DIR = _ROOT / "analysis_results" / "attention_split"

# ---------------------------------------------------------------------------
# Model loading (mirrors max_savings_test.py ModelContext)
# ---------------------------------------------------------------------------


class ModelContext:
    """Load a model through the registry; free memory on exit."""

    def __init__(self, model_id: str, registry: ModelRegistry, quantize: bool = False):
        self.model_id = model_id
        self._registry = registry
        self._quantize = quantize
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "ModelContext":
        print(f"  [LOAD] {self.model_id} ...")
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
            self.model, self.tokenizer = self._registry.load_verified_model(self.model_id)
        print(f"  [LOAD] loaded {self.model_id}")
        return self

    def __exit__(self, *_):
        print(f"  [UNLOAD] freeing {self.model_id} ...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [UNLOAD] done.")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_hardware() -> dict:
    """Return a dict describing the current compute device."""
    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        device_name = "Apple Silicon MPS"
    else:
        device_type = "cpu"
        device_name = "CPU"
    return {
        "device_type": device_type,
        "device_name": device_name,
        "hostname": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# Per-layer data capture
# ---------------------------------------------------------------------------


def capture_layer_data(
    layer_idx: int,
    o_proj_input: torch.Tensor,
    o_proj_output: torch.Tensor,
    O_weight: torch.Tensor,
    O_bias: Optional[torch.Tensor],
    num_heads: int,
    head_dim: int,
    timing_ms: float,
) -> dict:
    """
    At the last token position, compute per-head contributions and verify
    the o_proj decomposition.

    Parameters
    ----------
    o_proj_input  : (1, seq_len, num_heads * head_dim)  — float32
    o_proj_output : (1, seq_len, hidden_size)           — float32
    O_weight      : (hidden_size, num_heads * head_dim) — float32
    O_bias        : (hidden_size,) or None
    """
    # Extract last token position
    last_in = o_proj_input[0, -1, :].to(torch.float32)   # (num_heads * head_dim,)
    full_out = o_proj_output[0, -1, :].to(torch.float32)  # (hidden_size,)

    O_weight_f32 = O_weight.to(torch.float32)
    O_bias_f32 = O_bias.to(torch.float32) if O_bias is not None else None

    hidden_size = O_weight_f32.shape[0]

    # Compute per-head contributions: h_n @ O_weight[:, n*hd:(n+1)*hd].T
    head_contribs: list[torch.Tensor] = []
    for n in range(num_heads):
        h_n = last_in[n * head_dim : (n + 1) * head_dim]          # (head_dim,)
        O_n = O_weight_f32[:, n * head_dim : (n + 1) * head_dim]   # (hidden_size, head_dim)
        contrib = h_n @ O_n.T                                       # (hidden_size,)
        head_contribs.append(contrib)

    # Reconstruct full output from per-head contributions
    reconstructed = torch.stack(head_contribs).sum(dim=0)          # (hidden_size,)
    if O_bias_f32 is not None:
        reconstructed = reconstructed + O_bias_f32

    # Decomposition error
    diff = (reconstructed - full_out).abs()
    decomp_max_err = diff.max().item()
    decomp_mean_err = diff.mean().item()
    # Cosine similarity
    cos_num = (reconstructed * full_out).sum().item()
    cos_den = (reconstructed.norm() * full_out.norm()).item()
    decomp_cos = cos_num / (cos_den + 1e-12)

    # Split grouping errors
    split_errors: dict = {}
    for n_validators, heads_per_v in SPLIT_GROUPINGS:
        if n_validators * heads_per_v != num_heads:
            # Skip invalid groupings for this model config
            continue
        group_sums: list[torch.Tensor] = []
        for v in range(n_validators):
            start = v * heads_per_v
            group_sum = torch.stack(head_contribs[start : start + heads_per_v]).sum(dim=0)
            group_sums.append(group_sum)
        total = torch.stack(group_sums).sum(dim=0)
        if O_bias_f32 is not None:
            total = total + O_bias_f32
        gdiff = (total - full_out).abs()
        gcos_num = (total * full_out).sum().item()
        gcos_den = (total.norm() * full_out.norm()).item()
        key = f"{n_validators}x{heads_per_v}"
        split_errors[key] = {
            "max_abs": gdiff.max().item(),
            "mean_abs": gdiff.mean().item(),
            "cosine_sim": gcos_num / (gcos_den + 1e-12),
        }

    return {
        "attn_input_norm": last_in.norm().item(),
        "full_attn_output": full_out.tolist(),
        "head_contributions": [c.tolist() for c in head_contribs],
        "decomposition_check": {
            "max_abs_error": decomp_max_err,
            "mean_abs_error": decomp_mean_err,
            "cosine_sim": decomp_cos,
        },
        "split_errors": split_errors,
        "head_timing_ms": timing_ms,
    }


# ---------------------------------------------------------------------------
# Capture mode
# ---------------------------------------------------------------------------


def run_capture(args) -> None:
    """Load model, run 5 prompts, capture per-layer attention data, save JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path

    registry = ModelRegistry(DEFAULT_REGISTRY_PATH)

    print(f"[CAPTURE] model={args.model}  output={output_path}")

    with ModelContext(args.model, registry, quantize=args.quantize) as ctx:
        model = ctx.model
        tokenizer = ctx.tokenizer

        # Read architecture from config
        cfg = model.config
        num_layers: int = cfg.num_hidden_layers
        num_heads: int = cfg.num_attention_heads
        hidden_size: int = cfg.hidden_size
        head_dim: int = hidden_size // num_heads

        # Adaptive capture layers: first, middle, last
        capture_layers = [0, num_layers // 2, num_layers - 1]

        print(f"  num_layers={num_layers}  num_heads={num_heads}  "
              f"head_dim={head_dim}  hidden_size={hidden_size}")
        print(f"  capture_layers={capture_layers}")

        hardware = detect_hardware()
        device = next(model.parameters()).device

        prompt_results = []

        for prompt_text, prompt_complexity in CAPTURE_PROMPTS:
            print(f"\n  [PROMPT] {prompt_text!r} ({prompt_complexity})")

            # Storage for hook captures: {layer_idx: {"input": ..., "output": ...}}
            hook_data: dict[int, dict] = {li: {} for li in capture_layers}
            hooks = []

            # Register o_proj forward hooks for captured layers
            for li in capture_layers:
                layer = model.model.layers[li]

                def make_proj_hook(layer_idx):
                    def proj_hook(module, inp, out):
                        # inp is a tuple; inp[0] is the actual input tensor
                        hook_data[layer_idx]["o_proj_input"] = inp[0].detach().cpu()
                        hook_data[layer_idx]["o_proj_output"] = out.detach().cpu()
                    return proj_hook

                h = layer.self_attn.o_proj.register_forward_hook(make_proj_hook(li))
                hooks.append(h)

            t_start = time.perf_counter()
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            gen_time = time.perf_counter() - t_start

            # Remove hooks immediately after forward pass
            for h in hooks:
                h.remove()

            # Predicted token
            logits = out.logits[0, -1, :].float()
            top5 = torch.topk(logits, 5)
            predicted_id = int(top5.indices[0].item())
            predicted_token = tokenizer.decode([predicted_id])
            top5_tokens = [
                {
                    "id": int(top5.indices[i].item()),
                    "token": tokenizer.decode([int(top5.indices[i].item())]),
                    "logit": float(top5.values[i].item()),
                }
                for i in range(5)
            ]

            # Build per-layer capture
            layers_data: dict = {}
            for li in capture_layers:
                hd = hook_data[li]
                if "o_proj_input" not in hd or "o_proj_output" not in hd:
                    print(f"    [WARN] layer {li}: hook data missing, skipping")
                    continue

                layer = model.model.layers[li]
                O_weight = layer.self_attn.o_proj.weight.detach().cpu().to(torch.float32)
                O_bias_param = layer.self_attn.o_proj.bias
                O_bias = (
                    O_bias_param.detach().cpu().to(torch.float32)
                    if O_bias_param is not None
                    else None
                )

                t_head_start = time.perf_counter()
                layer_capture = capture_layer_data(
                    layer_idx=li,
                    o_proj_input=hd["o_proj_input"],
                    o_proj_output=hd["o_proj_output"],
                    O_weight=O_weight,
                    O_bias=O_bias,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    timing_ms=(time.perf_counter() - t_head_start) * 1000.0,
                )
                layers_data[str(li)] = layer_capture

                decomp = layer_capture["decomposition_check"]
                print(
                    f"    layer {li:2d}: decomp_max_err={decomp['max_abs_error']:.2e}  "
                    f"cos={decomp['cosine_sim']:.8f}"
                )

            prompt_result = {
                "prompt_text": prompt_text,
                "prompt_complexity": prompt_complexity,
                "predicted_token_id": predicted_id,
                "predicted_token": predicted_token,
                "top5_tokens": top5_tokens,
                "generation_time_s": round(gen_time, 4),
                "layers": layers_data,
            }
            prompt_results.append(prompt_result)

        # Assemble capture document
        capture_doc = {
            "format": "attention_split_v1",
            "model_name": args.model,
            "model_num_layers": num_layers,
            "model_num_heads": num_heads,
            "model_head_dim": head_dim,
            "model_hidden_size": hidden_size,
            "hardware": hardware,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "capture_layers": capture_layers,
            "prompts": prompt_results,
        }

    # Write JSON capture file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(capture_doc, fh, indent=2)

    print(f"\n[CAPTURE] Written to {output_path}")

    # Print validator cost estimate
    _print_validator_cost_estimate(hidden_size, head_dim)


def _print_validator_cost_estimate(hidden_size: int, head_dim: int) -> None:
    """Print combined validator cost estimates for single-head verification."""
    # Q, K, V weight slices + O_proj columns (float32 = 4 bytes each)
    single_head_bytes = (
        head_dim * hidden_size   # Q weight slice
        + head_dim * hidden_size  # K weight slice
        + head_dim * hidden_size  # V weight slice
        + hidden_size * head_dim  # O_proj columns
    ) * 4  # float32
    single_head_mb = single_head_bytes / (1024 ** 2)

    # Combined validator: 1 attention head + 100 FFN neurons
    ffn_neurons_bytes = 100 * hidden_size * 2 * 4  # up + down projection slices
    combined_bytes = single_head_bytes + ffn_neurons_bytes
    combined_mb = combined_bytes / (1024 ** 2)

    print(f"\n[COST ESTIMATE]")
    print(f"  Single head (Q+K+V+O slices): {single_head_mb:.2f} MB per layer")
    print(f"  100 FFN neurons (up+down):    {ffn_neurons_bytes / (1024**2):.2f} MB per layer")
    print(f"  Combined validator:            {combined_mb:.2f} MB per layer")


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def cosine_sim_lists(a: list, b: list) -> float:
    """Compute cosine similarity between two equal-length float lists."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    dot = (ta * tb).sum().item()
    denom = (ta.norm() * tb.norm()).item()
    return dot / (denom + 1e-12)


def mean_abs_diff_lists(a: list, b: list) -> float:
    """Mean absolute difference between two float lists."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return (ta - tb).abs().mean().item()


def max_abs_diff_lists(a: list, b: list) -> float:
    """Max absolute difference between two float lists."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return (ta - tb).abs().max().item()


def run_compare(args) -> None:
    """Load two capture files, compute cross-hardware comparison metrics."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path_a = Path(args.file_a)
    path_b = Path(args.file_b)
    if not path_a.is_absolute():
        path_a = _ROOT / path_a
    if not path_b.is_absolute():
        path_b = _ROOT / path_b

    print(f"[COMPARE] {path_a.name}  vs  {path_b.name}")

    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)

    # Validate format
    for label, doc in [("A", a), ("B", b)]:
        if doc.get("format") != "attention_split_v1":
            print(f"  [WARN] File {label} has unexpected format: {doc.get('format')}")

    # Check architecture compatibility
    arch_keys = ["model_name", "model_num_layers", "model_num_heads", "model_head_dim", "model_hidden_size"]
    arch_match = all(a.get(k) == b.get(k) for k in arch_keys)
    if not arch_match:
        print("  [WARN] Architecture mismatch between capture files:")
        for k in arch_keys:
            if a.get(k) != b.get(k):
                print(f"    {k}: {a.get(k)} vs {b.get(k)}")

    hw_a = a.get("hardware", {})
    hw_b = b.get("hardware", {})
    capture_layers = a.get("capture_layers", [])
    num_heads = a.get("model_num_heads", 28)

    prompts_a = a.get("prompts", [])
    prompts_b = b.get("prompts", [])
    n_prompts = min(len(prompts_a), len(prompts_b))

    # Per-prompt metrics
    token_matches = []
    # {layer_str: [cosine_sims across prompts]}
    layer_full_cos: dict[str, list] = {str(li): [] for li in capture_layers}
    layer_full_mean_abs: dict[str, list] = {str(li): [] for li in capture_layers}
    # {layer_str: {head_idx: [cosine_sims across prompts]}}
    layer_head_cos: dict[str, dict[int, list]] = {
        str(li): {n: [] for n in range(num_heads)} for li in capture_layers
    }
    # Split error comparison: {layer_str: {split_key: {"a": [...], "b": [...]}}}
    split_comparison: dict[str, dict[str, dict]] = {str(li): {} for li in capture_layers}

    for i in range(n_prompts):
        pa = prompts_a[i]
        pb = prompts_b[i]

        # Token match: compare predicted token
        token_match = int(pa.get("predicted_token_id") == pb.get("predicted_token_id"))
        token_matches.append(token_match)

        for li_str in [str(li) for li in capture_layers]:
            la = pa.get("layers", {}).get(li_str)
            lb = pb.get("layers", {}).get(li_str)
            if la is None or lb is None:
                continue

            # Full attention output cosine similarity
            full_a = la.get("full_attn_output")
            full_b = lb.get("full_attn_output")
            if full_a and full_b:
                layer_full_cos[li_str].append(cosine_sim_lists(full_a, full_b))
                layer_full_mean_abs[li_str].append(mean_abs_diff_lists(full_a, full_b))

            # Per-head cosine similarity
            heads_a = la.get("head_contributions", [])
            heads_b = lb.get("head_contributions", [])
            for n in range(min(len(heads_a), len(heads_b), num_heads)):
                layer_head_cos[li_str][n].append(cosine_sim_lists(heads_a[n], heads_b[n]))

            # Split error comparison
            se_a = la.get("split_errors", {})
            se_b = lb.get("split_errors", {})
            all_split_keys = set(se_a.keys()) | set(se_b.keys())
            for sk in all_split_keys:
                if sk not in split_comparison[li_str]:
                    split_comparison[li_str][sk] = {"max_abs_a": [], "max_abs_b": [], "cos_a": [], "cos_b": []}
                if sk in se_a:
                    split_comparison[li_str][sk]["max_abs_a"].append(se_a[sk].get("max_abs", float("nan")))
                    split_comparison[li_str][sk]["cos_a"].append(se_a[sk].get("cosine_sim", float("nan")))
                if sk in se_b:
                    split_comparison[li_str][sk]["max_abs_b"].append(se_b[sk].get("max_abs", float("nan")))
                    split_comparison[li_str][sk]["cos_b"].append(se_b[sk].get("cosine_sim", float("nan")))

    token_match_rate = sum(token_matches) / max(len(token_matches), 1)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    # Build report text
    lines = []
    lines.append("=" * 70)
    lines.append("ATTENTION SPLIT CROSS-HARDWARE COMPARISON")
    lines.append("=" * 70)
    lines.append(f"File A:  {path_a}")
    lines.append(f"  Model:    {a.get('model_name')}")
    lines.append(f"  Hardware: {hw_a.get('device_type')} / {hw_a.get('device_name')} @ {hw_a.get('hostname')}")
    lines.append(f"  Captured: {a.get('captured_at')}")
    lines.append("")
    lines.append(f"File B:  {path_b}")
    lines.append(f"  Model:    {b.get('model_name')}")
    lines.append(f"  Hardware: {hw_b.get('device_type')} / {hw_b.get('device_name')} @ {hw_b.get('hostname')}")
    lines.append(f"  Captured: {b.get('captured_at')}")
    lines.append("")
    lines.append(f"Architecture match: {arch_match}")
    lines.append(f"Prompts compared:   {n_prompts}")
    lines.append(f"Token match rate:   {token_match_rate:.1%}  ({sum(token_matches)}/{n_prompts} matching predicted tokens)")
    lines.append("")

    lines.append("PER-LAYER FULL ATTENTION OUTPUT SIMILARITY")
    lines.append("-" * 50)
    for li_str in [str(li) for li in capture_layers]:
        cos_vals = layer_full_cos[li_str]
        mae_vals = layer_full_mean_abs[li_str]
        if cos_vals:
            lines.append(
                f"  Layer {li_str:>2}: cos_sim={_mean(cos_vals):.8f}  "
                f"mean_abs_diff={_mean(mae_vals):.4e}  "
                f"(min_cos={min(cos_vals):.8f}  max_cos={max(cos_vals):.8f})"
            )
        else:
            lines.append(f"  Layer {li_str:>2}: no data")
    lines.append("")

    lines.append("PER-HEAD COSINE SIMILARITY (mean across prompts)")
    lines.append("-" * 50)
    for li_str in [str(li) for li in capture_layers]:
        head_dict = layer_head_cos[li_str]
        head_means = []
        for n in range(num_heads):
            vals = head_dict.get(n, [])
            if vals:
                head_means.append(_mean(vals))
        if head_means:
            overall = _mean(head_means)
            min_hcos = min(head_means)
            max_hcos = max(head_means)
            lines.append(
                f"  Layer {li_str:>2}: mean_head_cos={overall:.8f}  "
                f"min={min_hcos:.8f}  max={max_hcos:.8f}"
            )
            # Identify most/least consistent heads
            min_idx = head_means.index(min_hcos)
            max_idx = head_means.index(max_hcos)
            lines.append(
                f"           least consistent: head {min_idx} (cos={min_hcos:.8f})  "
                f"most consistent: head {max_idx} (cos={max_hcos:.8f})"
            )
        else:
            lines.append(f"  Layer {li_str:>2}: no data")
    lines.append("")

    lines.append("SPLIT GROUPING DECOMPOSITION ERRORS (mean across prompts)")
    lines.append("-" * 50)
    for li_str in [str(li) for li in capture_layers]:
        lines.append(f"  Layer {li_str}:")
        sc = split_comparison.get(li_str, {})
        for sk in sorted(sc.keys()):
            sd = sc[sk]
            mean_max_a = _mean(sd["max_abs_a"])
            mean_max_b = _mean(sd["max_abs_b"])
            mean_cos_a = _mean(sd["cos_a"])
            mean_cos_b = _mean(sd["cos_b"])
            lines.append(
                f"    {sk}: A max_abs={mean_max_a:.2e} cos={mean_cos_a:.8f} | "
                f"B max_abs={mean_max_b:.2e} cos={mean_cos_b:.8f}"
            )
        if not sc:
            lines.append("    (no split data)")
    lines.append("")

    lines.append("FEASIBILITY ASSESSMENT")
    lines.append("-" * 50)
    # Assess cross-hardware consistency
    all_cos = [v for vals in layer_full_cos.values() for v in vals]
    mean_cos_overall = _mean(all_cos)
    if mean_cos_overall >= 0.9999:
        assessment = "EXCELLENT — outputs are effectively identical across hardware"
    elif mean_cos_overall >= 0.999:
        assessment = "GOOD — outputs are highly consistent across hardware"
    elif mean_cos_overall >= 0.99:
        assessment = "MODERATE — minor numerical differences across hardware"
    else:
        assessment = "POOR — significant numerical differences across hardware"
    lines.append(f"  Overall mean cosine similarity: {mean_cos_overall:.8f}")
    lines.append(f"  Assessment: {assessment}")
    lines.append(f"  Token match rate: {token_match_rate:.1%}")
    if token_match_rate >= 1.0:
        lines.append("  Predicted tokens: IDENTICAL across hardware")
    elif token_match_rate >= 0.8:
        lines.append("  Predicted tokens: MOSTLY IDENTICAL across hardware")
    else:
        lines.append("  Predicted tokens: DIVERGE across hardware — splitting may be problematic")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    # Write text report
    report_path = OUT_DIR / "cross_hardware_comparison.txt"
    with open(report_path, "w") as fh:
        fh.write(report_text)
    print(f"\n[COMPARE] Report written to {report_path}")

    # Build standard results records
    std_records = []
    for i in range(n_prompts):
        pa = prompts_a[i]
        pb = prompts_b[i]
        for li_str in [str(li) for li in capture_layers]:
            la = pa.get("layers", {}).get(li_str)
            lb = pb.get("layers", {}).get(li_str)
            if la is None or lb is None:
                continue
            full_a = la.get("full_attn_output")
            full_b = lb.get("full_attn_output")
            cos = cosine_sim_lists(full_a, full_b) if full_a and full_b else None
            mae = mean_abs_diff_lists(full_a, full_b) if full_a and full_b else None
            std_records.append({
                "model_name": a.get("model_name", ""),
                "prompt_text": pa.get("prompt_text", ""),
                "attack_type": "attention_split",
                "model_num_layers": a.get("model_num_layers"),
                "model_num_heads": a.get("model_num_heads"),
                "model_head_dim": a.get("model_head_dim"),
                "model_hidden_size": a.get("model_hidden_size"),
                "prompt_complexity": pa.get("prompt_complexity"),
                "attack_params": {
                    "layer": int(li_str),
                    "hardware_a": hw_a.get("device_type"),
                    "hardware_b": hw_b.get("device_type"),
                    "hostname_a": hw_a.get("hostname"),
                    "hostname_b": hw_b.get("hostname"),
                },
                "layer": int(li_str),
                "cosine_similarity": round(cos, 8) if cos is not None else None,
                "absolute_error": round(mae, 8) if mae is not None else None,
                "token_match_rate": token_match_rate,
                "pass_fail": token_match_rate >= 0.8,
                "verification_target": "cross_hardware",
                "raw_data": {
                    "decomp_check_a": la.get("decomposition_check"),
                    "decomp_check_b": lb.get("decomposition_check"),
                    "split_errors_a": la.get("split_errors"),
                    "split_errors_b": lb.get("split_errors"),
                },
            })

    # Write standard results JSON
    std_path = OUT_DIR / "results_standard.json"
    write_standard_results(
        std_records,
        std_path,
        experiment_name="attention_split",
        script_path="tools/attention_split_test.py",
    )
    print(f"[COMPARE] Standard results written to {std_path}")

    # Write DB results
    try:
        with ResultsWriter(
            "attention_split",
            script_path="tools/attention_split_test.py",
            config_name="cross_hardware",
        ) as db_writer:
            for rec in std_records:
                try:
                    model_id_db = db_writer.ensure_model(rec["model_name"])
                    prompt_id_db = db_writer.ensure_prompt(
                        rec["prompt_text"], complexity=rec.get("prompt_complexity")
                    )
                    db_writer.add_result(
                        model_id=model_id_db,
                        prompt_id=prompt_id_db,
                        attack_type="attention_split",
                        attack_params=rec.get("attack_params"),
                        token_match_rate=rec.get("token_match_rate"),
                        cosine_similarity=rec.get("cosine_similarity"),
                        absolute_error=rec.get("absolute_error"),
                        pass_fail=rec.get("pass_fail"),
                        verification_target="cross_hardware",
                    )
                except Exception as exc:
                    print(f"  [WARN] DB write failed for one record: {exc}")
        print(f"[COMPARE] DB results written.")
    except Exception as exc:
        print(f"  [WARN] DB writer failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test attention head computation splitting across validators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model with 4-bit quantization (bitsandbytes).",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--capture",
        action="store_true",
        help="Run capture mode: run prompts and save attention data to JSON.",
    )
    mode_group.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Run compare mode: compare two capture files.",
    )

    parser.add_argument(
        "--output",
        default="attn_split_capture.json",
        help="Output file path for --capture mode (default: attn_split_capture.json).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.capture:
        run_capture(args)
    elif args.compare:
        args.file_a = args.compare[0]
        args.file_b = args.compare[1]
        run_compare(args)
    else:
        print("[ERROR] Must specify --capture or --compare.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
