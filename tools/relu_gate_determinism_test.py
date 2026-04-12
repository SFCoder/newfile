#!/usr/bin/env python3
"""
relu_gate_determinism_test.py
==============================
Captures pre-ReLU gate values (preReluGateValues_n) at every layer during
inference on ReluLLaMA-7B and tests whether they are bit-identical across
different hardware.

Hypothesis: Inferences run on ReluLLaMA-7B with common inputs produce 100%
identical preReluGateValues_n vectors across different hardware 100% of the
time.

The captured vector: gate_proj(normedState2_n) — the output of the FFN's
gate_proj linear layer BEFORE ReLU (act_fn) is applied.  Shape per layer:
[seq_len, intermediate_size].

Mode 1 — capture:
    python3 tools/relu_gate_determinism_test.py --capture --output relu_gate_mac.json

Mode 2 — compare:
    python3 tools/relu_gate_determinism_test.py --compare file_a.json file_b.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo root on sys.path so we can import model_registry
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_registry import ModelRegistry, DEFAULT_REGISTRY_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "SparseLLM/ReluLLaMA-7B"
HF_REPO = "SparseLLM/ReluLLaMA-7B"

# 5 prompts — shared with cross_hardware_test.py for consistency
PROMPTS: list[str] = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at high altitude it boils at",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a land far away, there lived a young inventor who",
    "The old detective walked slowly toward the dimly lit warehouse and",
]

RESULTS_DIR = REPO_ROOT / "analysis_results" / "relu_gate_determinism"


# ---------------------------------------------------------------------------
# ResultsWriter — writes experiment records to a JSON database file
# ---------------------------------------------------------------------------


class ResultsWriter:
    """
    Appends result records to analysis_results/{experiment_name}/results.json.

    Each call to write() appends one JSON object to the top-level list in that
    file.  The file is created on first write if it does not already exist.
    """

    def __init__(self, experiment_name: str, attack_type: str) -> None:
        self.experiment_name = experiment_name
        self.attack_type = attack_type
        self._dir = REPO_ROOT / "analysis_results" / experiment_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "results.json"

    def write(self, data: dict) -> None:
        """Append a result record that includes the experiment metadata."""
        record = {
            "experiment": self.experiment_name,
            "attack_type": self.attack_type,
            **data,
        }
        existing: list = []
        if self._path.exists():
            try:
                with open(self._path) as fh:
                    existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, ValueError):
                existing = []
        existing.append(record)
        with open(self._path, "w") as fh:
            json.dump(existing, fh, indent=2)


# ---------------------------------------------------------------------------
# Model registration and loading
# ---------------------------------------------------------------------------


def ensure_registered(registry: ModelRegistry) -> None:
    """Register SparseLLM/ReluLLaMA-7B in the registry if not already present."""
    if MODEL_ID not in registry.list_models():
        print(f"[registry] {MODEL_ID} not found — registering on first use.")
        print("           Downloading weights and computing SHA-256 hash;")
        print("           this may take several minutes on first run.")
        registry.register_new_model(
            model_id=MODEL_ID,
            hf_repo=HF_REPO,
            min_stake=0,
            download_if_missing=True,
        )
        print(f"[registry] {MODEL_ID} registered successfully.")
    else:
        print(f"[registry] {MODEL_ID} already registered.")


def load_model(registry: ModelRegistry):
    """
    Load ReluLLaMA-7B at float16 via the registry.

    The registry verifies the SHA-256 weight hash before loading.
    Returns (model, tokenizer).
    """
    ensure_registered(registry)
    print(f"[model] Verifying weights and loading {MODEL_ID} (float16) …")
    model, tokenizer = registry.load_verified_model(
        MODEL_ID,
        device=None,   # auto-detect: cuda → mps → cpu
        dtype=torch.float16,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pre-ReLU gate value hooks
# ---------------------------------------------------------------------------


class PreReluGateHook:
    """
    Captures gate_proj(normedState2_n) — the pre-ReLU gate values — at every
    MLP layer by registering forward hooks on each layer's gate_proj linear.

    In a LLaMA-family FFN:
        pre_relu = gate_proj(x)       ← we capture this
        gated    = act_fn(pre_relu)   ← ReLU in ReluLLaMA
        output   = down_proj(gated * up_proj(x))

    Captured shape per layer: [batch, seq_len, intermediate_size] → stored as
    float32 (upcast from float16 to avoid precision loss in serialisation).
    """

    def __init__(self) -> None:
        self._hooks: list = []
        self.captures: dict[int, torch.Tensor] = {}

    def attach(self, model) -> int:
        """Attach a hook to gate_proj of every MLP layer.  Returns layer count."""
        layers = _get_layers(model)
        for layer_idx, layer in enumerate(layers):
            gate_proj = layer.mlp.gate_proj

            def _make_hook(idx: int):
                def _hook(module, input_tuple, output):
                    # output: [batch, seq_len, intermediate_size]
                    # Store as float32 for lossless serialisation
                    self.captures[idx] = output.detach().float()
                return _hook

            self._hooks.append(gate_proj.register_forward_hook(_make_hook(layer_idx)))
        print(f"[hooks] Attached pre-ReLU gate hooks to {len(layers)} layers.")
        return len(layers)

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captures)
        self.captures.clear()
        return result

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _get_layers(model):
    """Return the transformer layer list for a LLaMA-family model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(
        f"Cannot find model.model.layers on {type(model).__name__}. "
        "Is this a LLaMA-family (AutoModelForCausalLM) model?"
    )


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------


def get_hardware_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_type"] = "mps"
    else:
        info["device_type"] = "cpu"
    return info


# ---------------------------------------------------------------------------
# Mode 1: Capture
# ---------------------------------------------------------------------------


def run_capture(output_path: str) -> None:
    """
    Load ReluLLaMA-7B, run 5 prompts through it, capture gate_proj outputs
    (pre-ReLU) at every layer for every token position, and save to JSON.
    """
    out = Path(output_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(DEFAULT_REGISTRY_PATH)
    model, tokenizer = load_model(registry)
    device = next(model.parameters()).device

    layers = _get_layers(model)
    num_layers = len(layers)
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    print(f"\n[capture] Model         : {MODEL_ID}")
    print(f"[capture] Layers        : {num_layers}")
    print(f"[capture] hidden_size   : {hidden_size}")
    print(f"[capture] intermediate  : {intermediate_size}")
    print(f"[capture] Device        : {device}")
    print(f"[capture] Prompts       : {len(PROMPTS)}")
    print()

    hook = PreReluGateHook()
    hook.attach(model)

    prompt_records: list[dict] = []

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"  [{prompt_idx + 1}/{len(PROMPTS)}] {prompt[:70]!r}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Predicted next token: greedy argmax over last-position logits
        predicted_next_token_id = int(outputs.logits[0, -1, :].argmax().item())

        layer_captures = hook.get_and_clear()

        # Serialise each layer's tensor: [1, seq_len, intermediate_size] → list
        # We squeeze the batch dim (always 1) and call .tolist() which converts
        # float32 → float64 at full precision, preserving the exact float32 bit
        # pattern for round-trip fidelity.
        layers_data: dict[str, list] = {}
        for layer_idx in sorted(layer_captures.keys()):
            tensor = layer_captures[layer_idx]       # [1, seq_len, intermediate_size]
            layers_data[str(layer_idx)] = tensor[0].cpu().tolist()  # [seq_len, intermediate_size]

        prompt_records.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "seq_len": int(inputs["input_ids"].shape[1]),
            "predicted_next_token_id": predicted_next_token_id,
            "layers": layers_data,
        })

        del layer_captures, outputs

    hook.detach()

    hw_info = get_hardware_info()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    doc = {
        "schema_version": "1",
        "experiment": "relu_gate_determinism",
        "attack_type": "honest",
        "model_id": MODEL_ID,
        "hf_repo": HF_REPO,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "dtype": "float16",
        "timestamp": timestamp,
        "hardware": hw_info,
        "prompts": prompt_records,
    }

    with open(out, "w") as fh:
        json.dump(doc, fh)

    print(f"\n[capture] Saved to {out}")

    ResultsWriter(
        experiment_name="relu_gate_determinism",
        attack_type="honest",
    ).write({
        "mode": "capture",
        "output_file": str(out.resolve()),
        "model_id": MODEL_ID,
        "num_prompts": len(PROMPTS),
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "hardware": hw_info,
        "timestamp": timestamp,
    })


# ---------------------------------------------------------------------------
# Mode 2: Compare
# ---------------------------------------------------------------------------


def _ulp_size_float32(a: np.ndarray) -> np.ndarray:
    """
    Return the ULP (unit in the last place) of each element in `a`, interpreted
    as float32.  Uses numpy.spacing on the float64 representation of each
    float32 value, which gives the correct float32 ULP.
    """
    a32 = a.astype(np.float32)
    ulp = np.abs(np.spacing(a32.astype(np.float64)))
    # For exact zeros the ULP is the smallest positive float32 (subnormal)
    ulp = np.where(ulp == 0.0, float(np.finfo(np.float32).tiny), ulp)
    return ulp


def run_compare(file_a: str, file_b: str) -> None:
    """
    Load two capture files produced by --capture and compare them element-wise.

    For every layer × prompt × token position × intermediate dimension, reports:
      - Fraction of elements that are bit-identical (diff exactly 0.0)
      - Fraction within 1 ULP (float32)
      - Maximum and mean absolute difference
      - Whether predicted next-tokens match on all prompts
      - Per-layer breakdown of the same statistics
    """
    with open(file_a) as fh:
        data_a = json.load(fh)
    with open(file_b) as fh:
        data_b = json.load(fh)

    def hw_label(d: dict) -> str:
        hw = d.get("hardware", {})
        return hw.get("gpu_name", hw.get("device_type", "unknown"))

    print(f"\n[compare] File A : {file_a}")
    print(f"          model  : {data_a.get('model_id', '?')}")
    print(f"          hw     : {hw_label(data_a)}")
    print(f"          time   : {data_a.get('timestamp', '?')}")
    print(f"\n[compare] File B : {file_b}")
    print(f"          model  : {data_b.get('model_id', '?')}")
    print(f"          hw     : {hw_label(data_b)}")
    print(f"          time   : {data_b.get('timestamp', '?')}")

    num_layers = min(data_a["num_layers"], data_b["num_layers"])
    if data_a["num_layers"] != data_b["num_layers"]:
        print(f"\n[compare] WARNING: layer count mismatch "
              f"({data_a['num_layers']} vs {data_b['num_layers']}); "
              f"comparing first {num_layers}.")

    prompts_a = {p["prompt_idx"]: p for p in data_a["prompts"]}
    prompts_b = {p["prompt_idx"]: p for p in data_b["prompts"]}
    common_idxs = sorted(set(prompts_a) & set(prompts_b))

    print(f"\n[compare] Comparing {len(common_idxs)} prompt(s) × "
          f"{num_layers} layer(s) …\n")

    # ---- Accumulators -------------------------------------------------------
    # Global (all prompts, all layers, all positions, all dims)
    total_elements = 0
    identical_count = 0
    within_1ulp_count = 0
    sum_abs_diff = 0.0
    max_abs_diff = 0.0

    # Per-layer
    layer_total: dict[int, int] = {i: 0 for i in range(num_layers)}
    layer_identical: dict[int, int] = {i: 0 for i in range(num_layers)}
    layer_within_1ulp: dict[int, int] = {i: 0 for i in range(num_layers)}
    layer_sum_diff: dict[int, float] = {i: 0.0 for i in range(num_layers)}
    layer_max_diff: dict[int, float] = {i: 0.0 for i in range(num_layers)}

    token_match_count = 0
    per_prompt_results: list[dict] = []

    for pidx in common_idxs:
        pa = prompts_a[pidx]
        pb = prompts_b[pidx]

        tok_match = (pa["predicted_next_token_id"] == pb["predicted_next_token_id"])
        token_match_count += int(tok_match)

        prompt_layer_stats: dict[str, dict] = {}

        for layer_idx in range(num_layers):
            key = str(layer_idx)
            if key not in pa["layers"] or key not in pb["layers"]:
                continue

            # Load as float32; the JSON values were written from float32 tensors
            arr_a = np.array(pa["layers"][key], dtype=np.float32)
            arr_b = np.array(pb["layers"][key], dtype=np.float32)

            if arr_a.shape != arr_b.shape:
                print(f"  WARNING: prompt {pidx} layer {layer_idx} shape mismatch "
                      f"{arr_a.shape} vs {arr_b.shape} — skipped.")
                continue

            # Compute differences in float64 to avoid cancellation artefacts
            diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
            ident_mask = diff == 0.0
            ulp = _ulp_size_float32(arr_a)
            ulp1_mask = diff <= ulp

            n = diff.size
            n_ident = int(ident_mask.sum())
            n_ulp1 = int(ulp1_mask.sum())
            layer_max = float(diff.max())
            layer_sum = float(diff.sum())

            # Accumulate global
            total_elements += n
            identical_count += n_ident
            within_1ulp_count += n_ulp1
            sum_abs_diff += layer_sum
            if layer_max > max_abs_diff:
                max_abs_diff = layer_max

            # Accumulate per-layer
            layer_total[layer_idx] += n
            layer_identical[layer_idx] += n_ident
            layer_within_1ulp[layer_idx] += n_ulp1
            layer_sum_diff[layer_idx] += layer_sum
            if layer_max > layer_max_diff[layer_idx]:
                layer_max_diff[layer_idx] = layer_max

            prompt_layer_stats[key] = {
                "bit_identical_frac": round(n_ident / n * 100.0, 6),
                "within_1ulp_frac": round(n_ulp1 / n * 100.0, 6),
                "max_abs_diff": layer_max,
                "mean_abs_diff": layer_sum / n,
            }

        per_prompt_results.append({
            "prompt_idx": pidx,
            "prompt": pa["prompt"],
            "token_match": tok_match,
            "token_a": pa["predicted_next_token_id"],
            "token_b": pb["predicted_next_token_id"],
            "layer_stats": prompt_layer_stats,
        })

    # ---- Compute global summary ---------------------------------------------
    if total_elements == 0:
        print("[compare] ERROR: no elements were compared (empty captures?).")
        return

    token_total = len(common_idxs)
    global_ident_pct = identical_count / total_elements * 100.0
    global_ulp1_pct = within_1ulp_count / total_elements * 100.0
    global_mean_diff = sum_abs_diff / total_elements
    tokens_all_match = (token_match_count == token_total)

    # ---- Print report -------------------------------------------------------
    SEP = "=" * 72
    print(SEP)
    print("COMPARISON REPORT — preReluGateValues_n (gate_proj output, pre-ReLU)")
    print(SEP)

    print(f"\nGlobal statistics  ({total_elements:,} elements, "
          f"{token_total} prompt(s), {num_layers} layer(s)):")
    print(f"  Bit-identical fraction  : {global_ident_pct:>12.6f} %")
    print(f"  Within 1 ULP fraction   : {global_ulp1_pct:>12.6f} %")
    print(f"  Maximum absolute diff   : {max_abs_diff:>12.4e}")
    print(f"  Mean absolute diff      : {global_mean_diff:>12.4e}")

    print(f"\nPredicted-token match     : "
          f"{token_match_count}/{token_total} "
          f"({'PASS ✓' if tokens_all_match else 'FAIL ✗'})")

    print(f"\nPer-layer statistics:")
    hdr = f"  {'Layer':>5}  {'Bit-ident %':>12}  {'≤1 ULP %':>10}  "
    hdr += f"{'Max diff':>12}  {'Mean diff':>12}"
    print(hdr)
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}")
    for li in range(num_layers):
        n = layer_total[li]
        if n == 0:
            continue
        ident_pct = layer_identical[li] / n * 100.0
        ulp1_pct = layer_within_1ulp[li] / n * 100.0
        mean_d = layer_sum_diff[li] / n
        flag = "  ← DIFFERS" if ident_pct < 100.0 else ""
        print(f"  {li:>5}  {ident_pct:>11.6f}%  {ulp1_pct:>9.4f}%  "
              f"{layer_max_diff[li]:>12.4e}  {mean_d:>12.4e}{flag}")

    # ---- Verdict ------------------------------------------------------------
    hypothesis_confirmed = (global_ident_pct == 100.0) and tokens_all_match
    print(f"\n{SEP}")
    if hypothesis_confirmed:
        verdict = "HYPOTHESIS CONFIRMED"
        detail = (
            "  All preReluGateValues_n elements are bit-identical across hardware.\n"
            "  All predicted next tokens match."
        )
    else:
        verdict = "HYPOTHESIS DISPROVED"
        lines = []
        if global_ident_pct < 100.0:
            deviant_pct = 100.0 - global_ident_pct
            lines.append(f"  {deviant_pct:.6f}% of elements are NOT bit-identical.")
        if not tokens_all_match:
            mismatches = token_total - token_match_count
            lines.append(f"  Predicted-token mismatch on {mismatches}/{token_total} prompt(s).")
        detail = "\n".join(lines)
    print(f"VERDICT: {verdict}")
    print(detail)
    print(SEP)

    # ---- Write to database --------------------------------------------------
    per_layer_report = {
        str(li): {
            "bit_identical_frac": round(layer_identical[li] / layer_total[li] * 100.0, 6),
            "within_1ulp_frac": round(layer_within_1ulp[li] / layer_total[li] * 100.0, 6),
            "max_abs_diff": layer_max_diff[li],
            "mean_abs_diff": layer_sum_diff[li] / layer_total[li],
            "total_elements": layer_total[li],
        }
        for li in range(num_layers) if layer_total[li] > 0
    }

    writer = ResultsWriter(
        experiment_name="relu_gate_determinism",
        attack_type="honest",
    )
    writer.write({
        "mode": "compare",
        "file_a": str(Path(file_a).resolve()),
        "file_b": str(Path(file_b).resolve()),
        "hw_a": hw_label(data_a),
        "hw_b": hw_label(data_b),
        "total_elements": total_elements,
        "global_bit_identical_frac": global_ident_pct,
        "global_within_1ulp_frac": global_ulp1_pct,
        "global_max_abs_diff": max_abs_diff,
        "global_mean_abs_diff": global_mean_diff,
        "token_match_count": token_match_count,
        "token_total": token_total,
        "tokens_all_match": tokens_all_match,
        "hypothesis_confirmed": hypothesis_confirmed,
        "per_layer": per_layer_report,
        "per_prompt": per_prompt_results,
    })
    print(f"\n[compare] Results appended to {writer._path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether ReluLLaMA-7B pre-ReLU gate values are bit-identical "
            "across different hardware."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture",
        action="store_true",
        help=(
            "Load ReluLLaMA-7B, run 5 prompts, capture gate_proj outputs "
            "(pre-ReLU) at every layer, and save to --output."
        ),
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two capture files and report per-element statistics.",
    )

    parser.add_argument(
        "--output",
        default="relu_gate_capture.json",
        help="Output JSON path for --capture mode.  (default: relu_gate_capture.json)",
    )

    args = parser.parse_args()

    if args.capture:
        run_capture(args.output)
    else:
        run_compare(args.compare[0], args.compare[1])


if __name__ == "__main__":
    main()
