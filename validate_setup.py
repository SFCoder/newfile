"""
Quick validation — using Qwen2.5 (ungated, SwiGLU architecture)
"""

import sys
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    errors = []
    device = get_device()

    print("1. Checking compute device...")
    if device == "mps":
        print("   OK: Apple Silicon GPU (MPS)")
    elif device == "cuda":
        print(f"   OK: NVIDIA GPU — {torch.cuda.get_device_name(0)}")
    else:
        print("   WARNING: CPU only (will be slow)")

    print("2. Checking dependencies...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import numpy as np
        from scipy import stats
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib
        print("   OK: All packages available")
    except ImportError as e:
        print(f"   FAIL: {e}")
        sys.exit(1)

    print("3. Checking model access...")
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        print(f"   OK: Qwen2.5-0.5B accessible ({c.num_hidden_layers} layers, {c.hidden_act} activation)")
    except Exception as e:
        print(f"   FAIL: {e}")
        sys.exit(1)

    print("4. Loading model (may take 1-2 minutes on first run)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device == "mps" else torch.float16

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=dtype)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    num_layers = len(model.model.layers)
    print(f"   OK: Model loaded ({num_layers} layers)")

    print("5. Testing SwiGLU hook...")
    from extract_activations import SwiGLUHook, compute_layer_stats

    hook = SwiGLUHook()
    hook.attach(model)

    prompt = "The theory of general relativity states that"
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=8, do_sample=False)

    activations = hook.get_and_clear()
    if len(activations) == 0:
        print("   FAIL: No activations captured.")
        sys.exit(1)
    print(f"   OK: Captured activations from {len(activations)} layers")

    print("6. Testing fingerprint statistics...")
    first_layer = list(activations.keys())[0]
    act = activations[first_layer]
    print(f"   Activation shape: {act.shape}")

    stats_result = compute_layer_stats(act)
    print(f"   Sparsity ratio:      {stats_result['sparsity_ratio']:.4f}")
    print(f"   Gini coefficient:    {stats_result['gini_mean']:.4f}")
    print(f"   L1/L2 ratio:         {stats_result['l1_l2_ratio']:.4f}")
    print(f"   Energy conc (top1%): {stats_result['energy_concentration_top1pct']:.4f}")
    print(f"   Active magnitude:    {stats_result['mag_mean']:.4f} +/- {stats_result['mag_std']:.4f}")
    print(f"   Top neuron indices:  {stats_result['top_100_neuron_indices'][:10]}...")

    if not (0 <= stats_result['sparsity_ratio'] <= 1):
        errors.append("Sparsity ratio outside [0,1]")
    if not (0 <= stats_result['gini_mean'] <= 1):
        errors.append("Gini coefficient outside [0,1]")

    hook.detach()

    print("7. Testing hook reuse...")
    hook2 = SwiGLUHook()
    hook2.attach(model)
    inputs2 = tokenizer("Photosynthesis is the process by which", return_tensors="pt").to(model_device)
    with torch.no_grad():
        model.generate(inputs2["input_ids"], max_new_tokens=8, do_sample=False)
    act2 = hook2.get_and_clear()
    if len(act2) != len(activations):
        errors.append(f"Inconsistent layer count: {len(act2)} vs {len(activations)}")
    else:
        print(f"   OK: Consistent capture ({len(act2)} layers)")
    hook2.detach()

    print()
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("=" * 50)
        print("ALL CHECKS PASSED")
        print("=" * 50)
        print()
        print("Run the experiment with:")
        print()
        print("  python3 extract_activations.py --model Qwen/Qwen2.5-0.5B --tag small --num-prompts 500")
        print("  python3 extract_activations.py --model Qwen/Qwen2.5-3B --tag large --num-prompts 500")
        print("  python3 analyze_fingerprints.py --model-a ./fingerprint_data/small --model-b ./fingerprint_data/large --output ./analysis_results/small_vs_large")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
