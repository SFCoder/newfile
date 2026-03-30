lines = open('sparse_replay.py').readlines()
with open('sparse_replay.py', 'w') as f:
    for line in lines:
        if "# Apply mask: zero out neurons that weren't active in Pass 1" in line:
            f.write('                # Apply per-step threshold: zero out neurons below threshold RIGHT NOW\n')
            f.write('                # (not using stored masks — just enforce sparsity directly)\n')
            f.write('                threshold = 0.01\n')
            f.write('                sparsity_mask = (intermediate.abs() > threshold)\n')
            f.write('                masked_intermediate = intermediate * sparsity_mask.to(intermediate.dtype)\n')
        elif 'mask = union_masks[layer_idx]' in line:
            pass  # skip
        elif 'masked_intermediate = intermediate * mask.unsqueeze' in line:
            pass  # skip, replaced above
        else:
            f.write(line)
print("Patched for per-step masking.")
