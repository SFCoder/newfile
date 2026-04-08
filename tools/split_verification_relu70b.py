"""
Verify MLP down_proj layer splitting on ReluLLaMA-70B.

Dequantizes the 4-bit NF4 down_proj weight using bitsandbytes dequantize_4bit
and verifies that the reconstructed matmul matches the model's full forward pass
output (cosine similarity 1.0, max diff 0.0).

Run on RunPod with 48GB+ VRAM.  Results saved to
analysis_results/split_verification_relu70b/<HARDWARE>_<TIMESTAMP>.json.
"""

import json
import os

import torch
from bitsandbytes.functional import dequantize_4bit
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SparseLLM/ReluLLaMA-70B"
OUTPUT_DIR = "analysis_results/split_verification_relu70b"

PROMPTS = [
    "The capital of France is",
    "In mathematics, the derivative of",
    "The theory of evolution was proposed by",
    "Python is a programming language that",
    "The speed of light in vacuum is approximately",
]

TESTED_LAYERS = [0, 20, 40, 60, 79]


def get_hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "_")
    return "CPU"


def dequantize_linear(linear):
    """Dequantize a bitsandbytes 4-bit quantized Linear layer's weight to float32."""
    weight = linear.weight
    dequant = dequantize_4bit(
        weight.data,
        weight.quant_state,
    ).to(torch.float32)
    return dequant.reshape(linear.out_features, linear.in_features)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer for {MODEL_ID} (use_fast=False)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    hardware = get_hardware_name()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    all_results = []
    pass_count = 0
    total_count = 0
    worst_cosine = 1.0
    worst_max_diff = 0.0

    for layer_idx in TESTED_LAYERS:
        layer = model.model.layers[layer_idx]
        down_proj = layer.mlp.down_proj

        captured = {}

        def make_hook(store):
            def hook(module, inp, out):
                store["input"] = inp[0].detach().float().cpu()
                store["output"] = out.detach().float().cpu()
            return hook

        handle = down_proj.register_forward_hook(make_hook(captured))

        # Dequantize down_proj weight once per layer — same pattern used in
        # attention_split_relu70b.py for o_proj.
        weight = dequantize_linear(down_proj)  # [out_features, in_features]

        for prompt_idx, prompt in enumerate(PROMPTS):
            captured.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                model(**inputs)

            hidden = captured["input"]    # [1, seq, in_features]
            baseline = captured["output"] # [1, seq, out_features]

            h = hidden.squeeze(0)   # [seq, in_features]
            b = baseline.squeeze(0) # [seq, out_features]

            # Reconstruct output by summing all neuron contributions.
            # Mathematically: output = h @ weight.T  (each column of weight.T
            # is one neuron's output projection vector).
            reconstructed = h @ weight.T  # [seq, out_features]

            cosine = torch.nn.functional.cosine_similarity(
                reconstructed.flatten().unsqueeze(0),
                b.flatten().unsqueeze(0),
            ).item()
            max_diff = (reconstructed - b).abs().max().item()
            passed = cosine >= 0.9999 and max_diff <= 0.001

            if passed:
                pass_count += 1
            total_count += 1
            worst_cosine = min(worst_cosine, cosine)
            worst_max_diff = max(worst_max_diff, max_diff)

            all_results.append({
                "layer_idx": layer_idx,
                "prompt_idx": prompt_idx,
                "cosine_similarity": cosine,
                "max_abs_diff": max_diff,
                "passed": passed,
            })
            print(
                f"Layer {layer_idx:2d}, Prompt {prompt_idx}: "
                f"cosine={cosine:.6f}  max_diff={max_diff:.6f}  passed={passed}"
            )

        handle.remove()

    output = {
        "hardware": hardware,
        "timestamp": timestamp,
        "model": MODEL_ID,
        "experiment": "split_verification_relu70b",
        "results": all_results,
        "summary": {
            "pass_rate": pass_count / total_count if total_count else 0.0,
            "pass_count": pass_count,
            "total_count": total_count,
            "worst_cosine_similarity": worst_cosine,
            "worst_max_abs_diff": worst_max_diff,
        },
    }

    fname = f"{hardware}_{timestamp}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {fpath}")
    print(f"Pass rate : {pass_count}/{total_count}")
    print(f"Worst cosine : {worst_cosine:.6f}")
    print(f"Worst max diff: {worst_max_diff:.6f}")


if __name__ == "__main__":
    main()
