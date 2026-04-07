"""Analyze layer-to-layer weight deltas for low-rank structure.

For each consecutive pair of layers, compute the delta and its SVD
to see how much of the delta is captured by low-rank approximations.
"""
import torch
import sys
from collections import defaultdict

def analyze(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)

    # Group weights by layer index and sub-module
    layers = defaultdict(dict)
    for name, tensor in sd.items():
        # Match patterns like "global_blocks.0.attn.qkv.weight" or "local_blocks.2.mlp.gate_up.weight"
        for prefix in ("global_blocks.", "local_blocks."):
            if prefix in name:
                rest = name.split(prefix)[1]
                idx = int(rest.split(".")[0])
                subname = ".".join(rest.split(".")[1:])
                layers[(prefix.rstrip("."), idx)][subname] = tensor.float()

    # Analyze deltas between consecutive layers within each block type
    for block_type in ("global_blocks", "local_blocks"):
        block_layers = {idx: weights for (bt, idx), weights in layers.items() if bt == block_type}
        if len(block_layers) < 2:
            continue

        indices = sorted(block_layers.keys())
        print(f"\n{'='*70}")
        print(f"{block_type}: {len(indices)} layers")
        print(f"{'='*70}")

        for i in range(len(indices) - 1):
            idx_a, idx_b = indices[i], indices[i+1]
            print(f"\n  Delta: layer {idx_a} → layer {idx_b}")

            common_keys = set(block_layers[idx_a].keys()) & set(block_layers[idx_b].keys())

            for key in sorted(common_keys):
                wa = block_layers[idx_a][key]
                wb = block_layers[idx_b][key]

                if wa.ndim < 2 or wa.numel() <= 1024:
                    continue
                if wa.shape != wb.shape:
                    continue

                # Reshape to 2D if needed
                orig_shape = wa.shape
                if wa.ndim > 2:
                    wa = wa.reshape(wa.shape[0], -1)
                    wb = wb.reshape(wb.shape[0], -1)

                delta = wb - wa

                # SVD of delta
                U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
                total_energy = (S ** 2).sum().item()

                if total_energy < 1e-12:
                    print(f"    {key:40s} {list(orig_shape)} — identical (zero delta)")
                    continue

                # How much energy captured at various ranks
                cumulative = torch.cumsum(S ** 2, dim=0) / total_energy

                # Also compute: ||delta||_F / ||W_b||_F (relative size of delta)
                delta_norm = delta.norm().item()
                wb_norm = wb.norm().item()
                relative = delta_norm / max(wb_norm, 1e-8)

                ranks_90 = (cumulative < 0.90).sum().item() + 1
                ranks_95 = (cumulative < 0.95).sum().item() + 1
                ranks_99 = (cumulative < 0.99).sum().item() + 1
                max_rank = min(wa.shape)

                print(f"    {key:40s} {str(list(orig_shape)):20s} "
                      f"||Δ||/||W||={relative:.3f}  "
                      f"rank@90%={ranks_90:3d}  @95%={ranks_95:3d}  @99%={ranks_99:3d}  "
                      f"(max={max_rank})")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "saved_weights/float/hier_mlp4_wd_a100_float.pt"
    print(f"Analyzing: {path}")
    analyze(path)
