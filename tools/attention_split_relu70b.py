"""
Verify attention head splitting on ReluLLaMA-70B.

For each tested layer the script:
  1. Captures the hidden-state input to self_attn via a forward hook.
  2. Captures the concatenated per-head attention output (input to o_proj)
     via a forward hook on the o_proj sub-module.
  3. Captures the full attention output (output of self_attn) as the baseline.
  4. Dequantizes the o_proj weight from 4-bit NF4 using bitsandbytes
     dequantize_4bit — this is the fix for the raw-packed-weight bug.
  5. For each of the 64 attention heads independently, extracts that head's
     slice of the concatenated attention output and multiplies it by the
     corresponding columns of the dequantized o_proj weight.
  6. Sums the 64 head contributions and compares the result to the baseline
     using cosine similarity and maximum absolute difference.

Expected result: cosine 1.000000, max_diff 0.000000 for all layers/prompts,
confirming that attention head splitting works on ReluLLaMA-70B.

Run on RunPod with 48GB+ VRAM.  Results saved to
analysis_results/attention_split_relu70b/<HARDWARE>_<TIMESTAMP>.json.
"""

import json
import os

import torch
from bitsandbytes.functional import dequantize_4bit
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SparseLLM/ReluLLaMA-70B"
OUTPUT_DIR = "analysis_results/attention_split_relu70b"

# Standard prompt set shared with other ReluLLaMA experiments.
PROMPTS = [
    "The capital of France is",
    "In mathematics, the derivative of",
    "The theory of evolution was proposed by",
    "Python is a programming language that",
    "The speed of light in vacuum is approximately",
]

# Sample 5 layers spread evenly across the 80-layer model.
TESTED_LAYERS = [0, 20, 40, 60, 79]


def get_hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "_")
    return "CPU"


def dequantize_linear(linear):
    """Dequantize a bitsandbytes 4-bit quantized Linear layer's weight to float32.

    Uses the same pattern as split_verification_relu70b.py uses for down_proj.
    The weight.data tensor holds the raw 4-bit packed bytes; dequantize_4bit
    converts it back to the original dtype (float16) before we cast to float32
    and reshape to [out_features, in_features].
    """
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

    # Derive attention geometry from model config.
    cfg = model.config
    num_heads = cfg.num_attention_heads          # 64
    num_kv_heads = cfg.num_key_value_heads       # 8
    hidden_size = cfg.hidden_size                # 8192
    head_dim = hidden_size // num_heads          # 128

    hardware = get_hardware_name()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    all_results = []
    pass_count = 0
    total_count = 0
    worst_cosine = 1.0
    worst_max_diff = 0.0

    for layer_idx in TESTED_LAYERS:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        o_proj = attn.o_proj

        # Three hooks per layer:
        #   attn_store   – captures self_attn input (hidden states) and output (baseline)
        #   oproj_store  – captures the concatenated per-head output fed into o_proj
        attn_store = {}
        oproj_store = {}

        def make_attn_hook(astore):
            def hook(module, inp, out):
                # inp[0]: hidden states going into self_attn  [1, seq, hidden]
                # out[0]: attention output returned by self_attn  [1, seq, hidden]
                astore["hidden"] = inp[0].detach().float().cpu()
                out_tensor = out[0] if isinstance(out, tuple) else out
                astore["baseline"] = out_tensor.detach().float().cpu()
            return hook

        def make_oproj_hook(ostore):
            def hook(module, inp, out):
                # inp[0]: concatenated per-head attention outputs  [1, seq, num_heads*head_dim]
                ostore["attn_ctx"] = inp[0].detach().float().cpu()
            return hook

        h_attn = attn.register_forward_hook(make_attn_hook(attn_store))
        h_oproj = o_proj.register_forward_hook(make_oproj_hook(oproj_store))

        # Dequantize o_proj weight once per layer.
        # Previously the code used o_proj.weight.data directly, which is the
        # raw 4-bit packed byte tensor (shape [1, out*in/2]) — causing the
        # "mat1 and mat2 shapes cannot be multiplied" error.
        # Fix: call dequantize_4bit to recover the actual weight matrix.
        o_proj_weight = dequantize_linear(o_proj)  # [hidden_size, num_heads*head_dim]

        for prompt_idx, prompt in enumerate(PROMPTS):
            attn_store.clear()
            oproj_store.clear()

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                model(**inputs)

            # attn_ctx: [1, seq, num_heads * head_dim]
            # baseline: [1, seq, hidden_size]
            attn_ctx = oproj_store["attn_ctx"]
            baseline = attn_store["baseline"]

            seq_len = attn_ctx.shape[1]

            # Sum contributions from each of the 64 attention heads.
            # For head h the contribution is:
            #   attn_ctx_h  [seq, head_dim]  @  o_proj_weight[:, cols_h].T  [head_dim, hidden]
            #             = [seq, hidden]
            # where cols_h = h*head_dim : (h+1)*head_dim.
            reconstructed = torch.zeros(seq_len, hidden_size, dtype=torch.float32)

            for h in range(num_heads):
                col_start = h * head_dim
                col_end = col_start + head_dim

                # Head h's portion of the concatenated attention output.
                attn_ctx_h = attn_ctx[0, :, col_start:col_end]  # [seq, head_dim]

                # Columns of o_proj that correspond to head h's input dimensions.
                # o_proj_weight shape: [out_features=hidden, in_features=num_heads*head_dim]
                # We want the columns (input-dimension slice) for head h.
                o_proj_h = o_proj_weight[:, col_start:col_end]  # [hidden, head_dim]

                # Contribution: [seq, head_dim] @ [head_dim, hidden] = [seq, hidden]
                reconstructed += attn_ctx_h @ o_proj_h.T

            b = baseline.squeeze(0)  # [seq, hidden]

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

        h_attn.remove()
        h_oproj.remove()

    output = {
        "hardware": hardware,
        "timestamp": timestamp,
        "model": MODEL_ID,
        "experiment": "attention_split_relu70b",
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
