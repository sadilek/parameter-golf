"""Analyze layer-to-layer weight deltas for standard GPT model."""
import torch
import sys
from collections import defaultdict

def analyze(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)

    # Group by block index
    layers = defaultdict(dict)
    for name, tensor in sd.items():
        if "blocks." in name:
            parts = name.split("blocks.")[1]
            idx = int(parts.split(".")[0])
            subname = ".".join(parts.split(".")[1:])
            layers[idx][subname] = tensor.float()

    indices = sorted(layers.keys())
    print(f"Found {len(indices)} layers")

    for i in range(len(indices) - 1):
        idx_a, idx_b = indices[i], indices[i+1]
        common = set(layers[idx_a].keys()) & set(layers[idx_b].keys())

        print(f"\n  Delta: layer {idx_a} → {idx_b}")
        for key in sorted(common):
            wa, wb = layers[idx_a][key], layers[idx_b][key]
            if wa.ndim < 2 or wa.numel() <= 1024 or wa.shape != wb.shape:
                continue
            orig_shape = wa.shape
            if wa.ndim > 2:
                wa = wa.reshape(wa.shape[0], -1)
                wb = wb.reshape(wb.shape[0], -1)
            delta = wb - wa
            U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
            total = (S**2).sum().item()
            if total < 1e-12:
                continue
            cum = torch.cumsum(S**2, 0) / total
            rel = delta.norm().item() / max(wb.norm().item(), 1e-8)
            r90 = (cum < 0.90).sum().item() + 1
            r95 = (cum < 0.95).sum().item() + 1
            r99 = (cum < 0.99).sum().item() + 1
            mx = min(wa.shape)
            print(f"    {key:40s} {str(list(orig_shape)):20s} "
                  f"||Δ||/||W||={rel:.3f}  r@90%={r90:3d}  @95%={r95:3d}  @99%={r99:3d}  (max={mx})")

if __name__ == "__main__":
    # Try to find a standard GPT float weight file
    import glob
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        candidates = glob.glob("saved_weights/float/*13L*") + glob.glob("saved_weights/float/*fullstack*") + glob.glob("saved_weights/float/*gpt*")
        if candidates:
            path = candidates[0]
    if not path:
        print("No GPT weights found. Pass path as argument.")
        sys.exit(1)
    print(f"Analyzing: {path}")
    analyze(path)
