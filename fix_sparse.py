lines = open('sparse_replay.py').readlines()
with open('sparse_replay.py', 'w') as f:
    for line in lines:
        if 'stacked = torch.stack(layer_masks[layer_idx])' in line:
            f.write('            # Flatten each mask to [num_positions, dim] then concatenate\n')
            f.write('            all_masks = torch.cat([m.reshape(-1, m.shape[-1]) for m in layer_masks[layer_idx]], dim=0)\n')
        elif 'union_mask = stacked.any(dim=0)' in line:
            pass  # skip this line
        elif 'flat_union = stacked.reshape(-1, stacked.shape[-1]).any(dim=0)' in line:
            f.write('            flat_union = all_masks.any(dim=0)  # [dim]\n')
        else:
            f.write(line)
print("Fixed.")
