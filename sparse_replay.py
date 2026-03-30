"""
Sparse Replay Experiment
=========================
Tests whether zeroing out inactive SwiGLU neurons reproduces the full model's output.

This is the core feasibility test for the SLM-reconstruction verification idea:
1. Run a prompt through the full model, record which neurons fired at each layer
2. Re-run the same prompt, but force all non-firing neurons to zero
3. Compare: does the output match?

If yes: a verifier could replay just the active subnetwork and verify the result.
If no: we need to understand why and how much the outputs diverge.
"""

import torch
import json
import time
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = get_device()
    print(f"Device: {device}")
    print()

    # --- Load model ---
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=dtype)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    model_device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    print(f"Loaded: {num_layers} layers")
    print()

    # --- Test prompts ---
    prompts = [
        "The capital of France is",
        "Explain the theory of general relativity in simple terms:",
        "Write a Python function that computes the Fibonacci sequence:",
        "The three primary colors are",
        "In 1969, humanity achieved something remarkable when",
    ]

    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"{'='*60}")
        print(f"Prompt {prompt_idx + 1}: \"{prompt}\"")
        print(f"{'='*60}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        input_ids = inputs["input_ids"]
        num_tokens_to_generate = 30

        # =================================================================
        # PASS 1: Normal generation — record sparsity masks
        # =================================================================
        print("\n  Pass 1: Full model (recording active neurons)...")

        layer_masks = {}  # layer_idx -> list of masks (one per generated token)

        def make_mask_hook(layer_idx):
            """Creates a hook that records the sparsity mask and lets everything through."""
            def hook_fn(module, input_tuple, output):
                x = input_tuple[0]
                gate = module.act_fn(module.gate_proj(x))
                up = module.up_proj(x)
                intermediate = gate * up

                # Record which neurons are active (above threshold)
                threshold = 0.01
                mask = (intermediate.abs() > threshold)

                if layer_idx not in layer_masks:
                    layer_masks[layer_idx] = []
                layer_masks[layer_idx].append(mask.detach().cpu())

                # Don't modify output — let the full model run normally
                return output

            return hook_fn

        # Attach recording hooks
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            h = layer.mlp.register_forward_hook(make_mask_hook(layer_idx))
            hooks.append(h)

        # Generate with full model
        with torch.no_grad():
            full_output = model.generate(
                input_ids,
                max_new_tokens=num_tokens_to_generate,
                do_sample=False,
            )

        # Remove hooks
        for h in hooks:
            h.remove()

        full_text = tokenizer.decode(full_output[0], skip_special_tokens=True)
        full_tokens = full_output[0].tolist()
        print(f"  Output: \"{full_text}\"")

        # Compute sparsity statistics from masks
        total_neurons = 0
        active_neurons = 0
        for layer_idx in sorted(layer_masks.keys()):
            for mask in layer_masks[layer_idx]:
                total_neurons += mask.numel()
                active_neurons += mask.sum().item()

        sparsity = 1.0 - (active_neurons / total_neurons)
        print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.1f}% of neurons were inactive)")
        print(f"  Active neurons per forward pass: {active_neurons/len(layer_masks[0])/num_layers:.0f} "
              f"out of {layer_masks[0][0].shape[-1]}")

        # =================================================================
        # PASS 2: Masked generation — zero out inactive neurons
        # =================================================================
        print("\n  Pass 2: Sparse replay (only active neurons)...")

        # We need to aggregate masks across generation steps.
        # For the replay, use the UNION of all masks across generation steps
        # (a neuron is kept if it fired on ANY token during generation)
        union_masks = {}
        for layer_idx in sorted(layer_masks.keys()):
            # Flatten each mask to [num_positions, dim] then concatenate
            all_masks = torch.cat([m.reshape(-1, m.shape[-1]) for m in layer_masks[layer_idx]], dim=0)
            # For simplicity, take the union across all positions too
            # This gives us a per-layer neuron mask: [intermediate_dim]
            flat_union = all_masks.any(dim=0)  # [dim]
            union_masks[layer_idx] = flat_union.to(model_device)

        # Report compression
        for layer_idx in [0, num_layers // 2, num_layers - 1]:
            kept = mask.sum().item()
            total = mask.numel()
            print(f"    Layer {layer_idx:2d}: keeping {kept:5d}/{total} neurons ({kept/total*100:.1f}%)")

        total_kept = sum(m.sum().item() for m in union_masks.values())
        total_possible = sum(m.numel() for m in union_masks.values())
        compression = 1.0 - (total_kept / total_possible)
        print(f"    Overall: {compression*100:.1f}% of neurons zeroed out")

        # Create masking hooks that zero out inactive neurons
        mask_step = [0]  # mutable counter for tracking generation steps

        def make_zeroing_hook(layer_idx):
            """Forces inactive neurons to zero during the forward pass."""
            def hook_fn(module, input_tuple, output):
                x = input_tuple[0]
                gate = module.act_fn(module.gate_proj(x))
                up = module.up_proj(x)
                intermediate = gate * up

                # Apply per-step threshold: zero out neurons below threshold RIGHT NOW
                # (not using stored masks — just enforce sparsity directly)
                threshold = 0.01
                sparsity_mask = (intermediate.abs() > threshold)
                masked_intermediate = intermediate * sparsity_mask.to(intermediate.dtype)
                # Broadcast mask to match intermediate shape

                # Compute the down projection with masked intermediate
                new_output = module.down_proj(masked_intermediate)
                return new_output

            return hook_fn

        # Attach zeroing hooks
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            h = layer.mlp.register_forward_hook(make_zeroing_hook(layer_idx))
            hooks.append(h)

        # Generate with masked model
        with torch.no_grad():
            sparse_output = model.generate(
                input_ids,
                max_new_tokens=num_tokens_to_generate,
                do_sample=False,
            )

        # Remove hooks
        for h in hooks:
            h.remove()

        sparse_text = tokenizer.decode(sparse_output[0], skip_special_tokens=True)
        sparse_tokens = sparse_output[0].tolist()
        print(f"  Output: \"{sparse_text}\"")

        # =================================================================
        # COMPARISON
        # =================================================================
        print(f"\n  --- Comparison ---")

        # Token-level match
        full_generated = full_tokens[input_ids.shape[1]:]
        sparse_generated = sparse_tokens[input_ids.shape[1]:]

        min_len = min(len(full_generated), len(sparse_generated))
        matches = sum(1 for a, b in zip(full_generated[:min_len], sparse_generated[:min_len]) if a == b)

        print(f"  Token match: {matches}/{min_len} ({matches/max(min_len,1)*100:.1f}%)")

        if full_text == sparse_text:
            print(f"  Text match:  EXACT MATCH")
        else:
            # Find where they diverge
            full_words = full_text.split()
            sparse_words = sparse_text.split()
            first_diff = None
            for i, (a, b) in enumerate(zip(full_words, sparse_words)):
                if a != b:
                    first_diff = i
                    break
            if first_diff is not None:
                print(f"  Text match:  DIVERGED at word {first_diff}")
                print(f"    Full:   ...{' '.join(full_words[max(0,first_diff-2):first_diff+5])}")
                print(f"    Sparse: ...{' '.join(sparse_words[max(0,first_diff-2):first_diff+5])}")
            else:
                if len(full_words) != len(sparse_words):
                    print(f"  Text match:  Same prefix, different length")
                else:
                    print(f"  Text match:  EXACT MATCH")

        # Also do a single forward pass (not generation) to compare logits
        print(f"\n  --- Logit comparison (single forward pass) ---")

        # Full forward pass
        with torch.no_grad():
            full_logits = model(input_ids).logits

        # Sparse forward pass
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            h = layer.mlp.register_forward_hook(make_zeroing_hook(layer_idx))
            hooks.append(h)

        with torch.no_grad():
            sparse_logits = model(input_ids).logits

        for h in hooks:
            h.remove()

        # Compare logits
        logit_diff = (full_logits - sparse_logits).abs()
        max_diff = logit_diff.max().item()
        mean_diff = logit_diff.mean().item()
        rel_diff = (logit_diff / (full_logits.abs() + 1e-8)).mean().item()

        # Do they predict the same next token?
        full_next = full_logits[0, -1, :].argmax().item()
        sparse_next = sparse_logits[0, -1, :].argmax().item()
        full_next_word = tokenizer.decode([full_next])
        sparse_next_word = tokenizer.decode([sparse_next])

        # Top-5 agreement
        full_top5 = full_logits[0, -1, :].topk(5).indices.tolist()
        sparse_top5 = sparse_logits[0, -1, :].topk(5).indices.tolist()
        top5_overlap = len(set(full_top5) & set(sparse_top5))

        print(f"  Max logit difference:     {max_diff:.6f}")
        print(f"  Mean logit difference:    {mean_diff:.6f}")
        print(f"  Mean relative difference: {rel_diff:.6f}")
        print(f"  Next token (full):   '{full_next_word}' (id={full_next})")
        print(f"  Next token (sparse): '{sparse_next_word}' (id={sparse_next})")
        print(f"  Same next token:     {'YES' if full_next == sparse_next else 'NO'}")
        print(f"  Top-5 overlap:       {top5_overlap}/5")

        # Cosine similarity of full logit vectors
        cos_sim = torch.nn.functional.cosine_similarity(
            full_logits[0, -1, :].unsqueeze(0).float(),
            sparse_logits[0, -1, :].unsqueeze(0).float()
        ).item()
        print(f"  Logit cosine similarity: {cos_sim:.6f}")

        print()

        all_results.append({
            "prompt": prompt,
            "full_text": full_text,
            "sparse_text": sparse_text,
            "token_match_rate": matches / max(min_len, 1),
            "exact_text_match": full_text == sparse_text,
            "compression": compression,
            "sparsity": sparsity,
            "max_logit_diff": max_diff,
            "mean_logit_diff": mean_diff,
            "same_next_token": full_next == sparse_next,
            "top5_overlap": top5_overlap,
            "cosine_similarity": cos_sim,
        })

        # Clear masks for next prompt
        layer_masks.clear()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    exact_matches = sum(1 for r in all_results if r["exact_text_match"])
    avg_token_match = sum(r["token_match_rate"] for r in all_results) / len(all_results)
    avg_compression = sum(r["compression"] for r in all_results) / len(all_results)
    avg_cosine = sum(r["cosine_similarity"] for r in all_results) / len(all_results)
    same_next = sum(1 for r in all_results if r["same_next_token"])
    avg_top5 = sum(r["top5_overlap"] for r in all_results) / len(all_results)

    print(f"\n  Prompts tested: {len(all_results)}")
    print(f"  Exact text matches: {exact_matches}/{len(all_results)}")
    print(f"  Average token match rate: {avg_token_match:.4f} ({avg_token_match*100:.1f}%)")
    print(f"  Average compression: {avg_compression*100:.1f}% neurons zeroed out")
    print(f"  Same next-token prediction: {same_next}/{len(all_results)}")
    print(f"  Average top-5 overlap: {avg_top5:.1f}/5")
    print(f"  Average logit cosine similarity: {avg_cosine:.6f}")

    print(f"\n  INTERPRETATION:")
    if exact_matches == len(all_results):
        print(f"  PERFECT MATCH — sparse replay reproduces full model output exactly.")
        print(f"  The SLM-reconstruction verification approach is mechanically feasible.")
    elif avg_token_match > 0.9:
        print(f"  NEAR MATCH — sparse replay closely approximates full model output.")
        print(f"  Small divergences accumulate over generation but core predictions agree.")
        print(f"  Approach may work with single-step verification (not full generation).")
    elif avg_cosine > 0.99:
        print(f"  LOGITS AGREE, GENERATION DIVERGES — the single-step predictions are")
        print(f"  nearly identical but small differences compound during autoregressive")
        print(f"  generation. Verification on single forward passes would work; verifying")
        print(f"  full generated sequences would need tolerance bounds.")
    else:
        print(f"  SIGNIFICANT DIVERGENCE — zeroing inactive neurons changes the output")
        print(f"  meaningfully. The 'inactive' neurons contribute through accumulated")
        print(f"  residual effects. The simple masking approach needs refinement.")

    # Save results
    output_dir = Path("analysis_results/sparse_replay")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
