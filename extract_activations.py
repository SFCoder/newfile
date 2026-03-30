"""
SwiGLU Activation Fingerprint Extraction
=========================================
Hooks into the MLP forward pass of Llama-family models to capture
post-gating activation statistics at every layer during inference.

Usage:
    # Small-scale sanity check (8B vs 1B on CPU/single GPU)
    python extract_activations.py --model meta-llama/Llama-3.2-1B --tag 1b --num-prompts 100
    python extract_activations.py --model meta-llama/Llama-3.2-3B --tag 3b --num-prompts 100

    # Full experiment (8B vs 70B, requires significant GPU)
    python extract_activations.py --model meta-llama/Meta-Llama-3.1-8B --tag 8b --num-prompts 1000
    python extract_activations.py --model meta-llama/Meta-Llama-3.1-70B --tag 70b --num-prompts 1000 --quantize 4bit
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fingerprint statistics computed on-the-fly per layer per token position
# ---------------------------------------------------------------------------

def gini_coefficient(x: torch.Tensor) -> float:
    """Gini coefficient of absolute activation values. 1.0 = maximally sparse."""
    abs_x = x.abs().float()
    if abs_x.sum() == 0:
        return 0.0
    sorted_x = torch.sort(abs_x)[0]
    n = len(sorted_x)
    index = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
    return (2 * (index * sorted_x).sum() / (n * sorted_x.sum()) - (n + 1) / n).item()


def compute_layer_stats(activation: torch.Tensor, threshold: float = 0.01) -> dict:
    """
    Compute fingerprint statistics for a single layer's post-gate activation.

    Args:
        activation: shape [batch, seq_len, intermediate_dim] — the SwiGLU output
                    BEFORE the down-projection.
        threshold: absolute value below which a neuron is considered "inactive"

    Returns:
        dict of summary statistics (all JSON-serializable)
    """
    # Work in float32 for numerical stability
    act = activation.detach().float()

    # Flatten across batch and sequence for whole-layer statistics
    # shape: [num_tokens, intermediate_dim]
    flat = act.reshape(-1, act.shape[-1])
    num_tokens = flat.shape[0]
    dim = flat.shape[1]

    # --- Per-neuron statistics (aggregated across tokens) ---
    abs_flat = flat.abs()

    # Sparsity: fraction of (token, neuron) pairs below threshold
    sparsity_ratio = (abs_flat < threshold).float().mean().item()

    # Per-neuron activation rate: for each neuron, what fraction of tokens activate it?
    neuron_activation_rate = (abs_flat >= threshold).float().mean(dim=0)  # [dim]

    # Global magnitude statistics
    magnitudes = abs_flat[abs_flat >= threshold]
    if len(magnitudes) == 0:
        mag_mean, mag_std, mag_skew, mag_kurt = 0.0, 0.0, 0.0, 0.0
    else:
        mag_mean = magnitudes.mean().item()
        mag_std = magnitudes.std().item()
        centered = magnitudes - mag_mean
        if mag_std > 1e-8:
            normalized = centered / mag_std
            mag_skew = normalized.pow(3).mean().item()
            mag_kurt = normalized.pow(4).mean().item() - 3.0  # excess kurtosis
        else:
            mag_skew, mag_kurt = 0.0, 0.0

    # Gini coefficient (sample a subset of tokens for speed)
    sample_size = min(64, num_tokens)
    sample_indices = torch.randperm(num_tokens)[:sample_size]
    gini_values = [gini_coefficient(flat[i]) for i in sample_indices]
    gini_mean = float(np.mean(gini_values))
    gini_std = float(np.std(gini_values))

    # L1/L2 ratio (indicator of sparsity — lower = sparser)
    l1 = abs_flat.sum(dim=-1)    # [num_tokens]
    l2 = flat.pow(2).sum(dim=-1).sqrt()  # [num_tokens]
    l1_l2_ratio = (l1 / (l2 + 1e-8)).mean().item()

    # Top-k neuron indices (which neurons fire most often across tokens)
    top_k = min(100, dim)
    _, topk_indices = neuron_activation_rate.topk(top_k)
    topk_indices_list = topk_indices.cpu().tolist()

    # Activation magnitude histogram (50 bins over the active neurons)
    if len(magnitudes) > 0:
        hist_counts, hist_edges = torch.histogram(magnitudes.cpu(), bins=50)
        hist_counts = hist_counts.tolist()
        hist_edges = hist_edges.tolist()
    else:
        hist_counts, hist_edges = [], []

    # Energy concentration: what fraction of total L2 energy is in top 1% of neurons?
    top_1pct = max(1, dim // 100)
    energy_per_neuron = flat.pow(2).mean(dim=0)  # [dim]
    sorted_energy, _ = energy_per_neuron.sort(descending=True)
    total_energy = sorted_energy.sum().item()
    top_1pct_energy = sorted_energy[:top_1pct].sum().item()
    energy_concentration = top_1pct_energy / (total_energy + 1e-8)

    return {
        "sparsity_ratio": sparsity_ratio,
        "gini_mean": gini_mean,
        "gini_std": gini_std,
        "l1_l2_ratio": l1_l2_ratio,
        "mag_mean": mag_mean,
        "mag_std": mag_std,
        "mag_skew": mag_skew,
        "mag_kurtosis": mag_kurt,
        "energy_concentration_top1pct": energy_concentration,
        "top_100_neuron_indices": topk_indices_list,
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
        "num_tokens": num_tokens,
        "intermediate_dim": dim,
    }


# ---------------------------------------------------------------------------
# Hook system for capturing SwiGLU intermediate activations
# ---------------------------------------------------------------------------

class SwiGLUHook:
    """
    Monkey-patches the MLP forward to capture the post-gate activation
    (after SwiGLU gating, before down-projection).

    In HuggingFace Llama:
        down_proj( act_fn(gate_proj(x)) * up_proj(x) )

    We capture: act_fn(gate_proj(x)) * up_proj(x)
    """

    def __init__(self):
        self.layer_activations = {}  # layer_idx -> activation tensor
        self._hooks = []
        self._patched_forwards = []

    def attach(self, model):
        """Attach hooks to all MLP layers in the model."""
        for layer_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            original_forward = mlp.forward

            # Create closure with correct layer_idx binding
            def make_patched_forward(orig_fwd, idx):
                def patched_forward(x):
                    # Compute SwiGLU components
                    gate = mlp.act_fn(mlp.gate_proj(x))
                    up = mlp.up_proj(x)
                    intermediate = gate * up  # This is what we want to capture

                    # Store it
                    self.layer_activations[idx] = intermediate

                    # Complete the forward pass
                    return mlp.down_proj(intermediate)

                return patched_forward

            mlp.forward = make_patched_forward(original_forward, layer_idx)
            self._patched_forwards.append((mlp, original_forward))

        print(f"  Attached SwiGLU hooks to {len(model.model.layers)} layers")

    def get_and_clear(self) -> dict:
        """Return captured activations and clear the buffer."""
        result = dict(self.layer_activations)
        self.layer_activations = {}
        return result

    def detach(self):
        """Restore original forward methods."""
        for mlp, original_forward in self._patched_forwards:
            mlp.forward = original_forward
        self._patched_forwards.clear()


# ---------------------------------------------------------------------------
# Prompt dataset
# ---------------------------------------------------------------------------

def load_prompts(num_prompts: int, max_length: int = 256) -> list[str]:
    """
    Load diverse prompts from a standard dataset.
    Uses OpenWebText for diversity. Falls back to simple prompts if unavailable.
    """
    print(f"Loading {num_prompts} prompts...")

    try:
        # Use a subset of C4 (readily available, diverse)
        from datasets import load_dataset
        from datasets import load_dataset; dataset = load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )
        prompts = []
        for i, example in enumerate(dataset):
            if i >= num_prompts:
                break
            # Truncate to get a prompt-like prefix
            text = example["text"]
            # Take first ~max_length characters as the "prompt"
            words = text.split()[:max_length]
            if len(words) >= 20:  # Skip very short texts
                prompts.append(" ".join(words))

        print(f"  Loaded {len(prompts)} prompts from C4")
        return prompts

    except Exception as e:
        print(f"  Could not load C4 ({e}), falling back to synthetic prompts")
        # Fallback: generate diverse synthetic prompts
        topics = [
            "Explain the theory of",
            "Write a detailed analysis of",
            "Describe the historical significance of",
            "Compare and contrast the approaches to",
            "What are the key challenges in",
            "Provide a technical overview of",
            "Discuss the ethical implications of",
            "How does the process of",
            "What role does technology play in",
            "Analyze the economic impact of",
        ]
        subjects = [
            "quantum computing", "climate change", "neural networks",
            "ancient Rome", "modern architecture", "genetic engineering",
            "renewable energy", "international trade", "machine learning",
            "space exploration", "democratic governance", "protein folding",
            "urban planning", "cryptocurrency", "natural language processing",
        ]
        prompts = []
        for i in range(num_prompts):
            topic = topics[i % len(topics)]
            subject = subjects[i % len(subjects)]
            prompts.append(f"{topic} {subject}")

        return prompts


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(
    model_name: str,
    tag: str,
    num_prompts: int,
    output_dir: str,
    quantize: str = None,
    max_new_tokens: int = 64,
    device: str = "auto",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir) / tag
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"\nLoading model: {model_name}")
    print(f"  Quantization: {quantize or 'none'}")

    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # --- Record model metadata ---
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_heads = model.config.num_attention_heads

    metadata = {
        "model_name": model_name,
        "tag": tag,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_attention_heads": num_heads,
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", num_heads),
        "quantization": quantize,
        "torch_dtype": str(load_kwargs["torch_dtype"]),
        "max_new_tokens": max_new_tokens,
        "num_prompts": num_prompts,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Architecture: {num_layers} layers, {hidden_dim}d hidden, "
          f"{intermediate_dim}d intermediate, {num_heads} heads")

    # --- Attach hooks ---
    hook = SwiGLUHook()
    hook.attach(model)

    # --- Load prompts ---
    prompts = load_prompts(num_prompts)

    # --- Run inference and collect fingerprints ---
    all_fingerprints = []

    print(f"\nRunning inference on {len(prompts)} prompts...")
    start_time = time.time()

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting")):
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(model.device)

        # Run forward pass (generation to exercise the full model)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # Greedy for reproducibility
                temperature=1.0,
                return_dict_in_generate=True,
            )

        # Collect the activations captured during the LAST forward pass
        # (the last token generation step)
        layer_activations = hook.get_and_clear()

        # Compute per-layer fingerprint statistics
        prompt_fingerprint = {
            "prompt_idx": prompt_idx,
            "prompt_length": input_ids.shape[1],
            "generated_length": outputs.sequences.shape[1] - input_ids.shape[1],
            "layers": {},
        }

        for layer_idx in sorted(layer_activations.keys()):
            act = layer_activations[layer_idx]
            stats = compute_layer_stats(act)
            prompt_fingerprint["layers"][layer_idx] = stats

            # Free memory
            del act
        del layer_activations
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_fingerprints.append(prompt_fingerprint)

        # Save intermediate results every 100 prompts
        if (prompt_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (prompt_idx + 1) / elapsed
            print(f"\n  Processed {prompt_idx + 1}/{len(prompts)} "
                  f"({rate:.1f} prompts/sec, "
                  f"est. {(len(prompts) - prompt_idx - 1) / rate / 60:.1f} min remaining)")

            checkpoint_file = output_path / f"fingerprints_checkpoint_{prompt_idx + 1}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(all_fingerprints, f)

    # --- Save final results ---
    elapsed = time.time() - start_time
    print(f"\nExtraction complete: {len(prompts)} prompts in {elapsed:.1f}s "
          f"({len(prompts)/elapsed:.2f} prompts/sec)")

    final_file = output_path / "fingerprints.json"
    with open(final_file, "w") as f:
        json.dump(all_fingerprints, f)

    print(f"Results saved to {final_file}")

    # --- Cleanup ---
    hook.detach()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SwiGLU activation fingerprints from LLM inference"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. meta-llama/Meta-Llama-3.1-8B)"
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Short tag for this run (e.g. '8b', '70b-4bit')"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000,
        help="Number of prompts to process"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./fingerprint_data",
        help="Directory to store results"
    )
    parser.add_argument(
        "--quantize", type=str, choices=["4bit", "8bit"], default=None,
        help="Quantization method (for large models)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Tokens to generate per prompt"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device map for model loading"
    )

    args = parser.parse_args()

    run_extraction(
        model_name=args.model,
        tag=args.tag,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        quantize=args.quantize,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
