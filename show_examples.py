"""Generate example predictions from a trained model checkpoint.

Usage: python3 show_examples.py [--model final_model.ptz] [--num-examples 5]
"""
import io, os, sys, torch, lzma, argparse
import sentencepiece as spm
import numpy as np

# Reuse model definition from train_combined.py
sys.path.insert(0, os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.ptz")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--val-data", default="./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--predict-len", type=int, default=32)
    parser.add_argument("--quant-mode", default="int6")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load val data
    header = np.fromfile(args.val_data, dtype="<i4", count=256)
    n_tokens = int(header[2])
    tokens = np.fromfile(args.val_data, dtype="<u2", count=n_tokens, offset=256*4)
    tokens = torch.from_numpy(tokens.astype(np.int64))

    # Load model
    with open(args.model, "rb") as f:
        raw = f.read()
    try:
        decompressed = lzma.decompress(raw)
    except Exception:
        import zstandard
        decompressed = zstandard.ZstdDecompressor().decompress(raw)
    loaded = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)

    # Determine model config from checkpoint keys
    n_layers = max(int(k.split(".")[1]) for k in loaded if k.startswith("blocks.")) + 1
    # Detect if int6 or ternary
    has_int6 = any(k.endswith(".q") for k in loaded)

    if has_int6:
        from train_combined import deq_sd_int6
        sd = deq_sd_int6(loaded, target_dtype=torch.bfloat16)
    else:
        from train_combined import deq_sd
        sd = deq_sd(loaded)

    # Infer model dim from tok_emb
    embed_weight = sd.get("tok_emb.weight", sd.get("tok_emb.fp_weight"))
    vocab_size_eff, model_dim = embed_weight.shape
    vocab_size = 1024
    n_channels = vocab_size_eff - vocab_size if vocab_size_eff > vocab_size else 0
    if n_channels > 0:
        n_channels = 8  # assume 8ch if mask token present

    # Check for XSA
    has_xsa = any("xsa_gate" in k for k in sd)
    xsa_layers = sum(1 for k in sd if "xsa_gate" in k)

    # Check activation type from MLP structure
    has_gate_up = any("gate_up" in k for k in sd)
    activation = "swiglu" if has_gate_up else "leaky_relu2"

    # Infer mlp_mult
    if has_gate_up:
        gu_key = [k for k in sd if "blocks.0.mlp.gate_up.weight" in k][0]
        hidden2 = sd[gu_key].shape[0]
        mlp_mult = hidden2 // (2 * model_dim)
    else:
        fc_key = [k for k in sd if "blocks.0.mlp.fc.weight" in k][0]
        hidden = sd[fc_key].shape[0]
        mlp_mult = hidden // model_dim

    print(f"Model: {n_layers}L/{model_dim}d, {activation}, mlp_mult={mlp_mult}, xsa={xsa_layers} layers, channels={n_channels}")

    from train_combined import GPT
    model = GPT(
        vocab_size=vocab_size, num_layers=n_layers, model_dim=model_dim,
        num_heads=8, num_kv_heads=4, mlp_mult=mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        activation=activation, n_channels=max(n_channels, 1),
        xsa_layers=xsa_layers,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).bfloat16().eval()

    # Generate examples
    seq_len = args.context_len + args.predict_len
    print(f"\n{'='*80}")
    print(f"Showing {args.num_examples} examples: {args.context_len} context tokens → {args.predict_len} predicted tokens")
    print(f"{'='*80}\n")

    rng = np.random.RandomState(42)
    starts = rng.randint(0, len(tokens) - seq_len - 1, size=args.num_examples)

    for idx, start in enumerate(starts):
        chunk = tokens[start:start + seq_len].to(device).unsqueeze(0)
        context = chunk[:, :args.context_len]
        target = chunk[:, args.context_len:args.context_len + args.predict_len]

        # Get model predictions (greedy)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Teacher-forced: feed all tokens, get predictions
            x = chunk[:, :args.context_len + args.predict_len - 1]
            y = chunk[:, 1:args.context_len + args.predict_len]

            # Get logits for the prediction region
            emb = model._embed(x)
            if model.n_channels > 1:
                emb = emb + model.channel_in[0]
            hidden = model._run_blocks(emb, emb, causal=True)
            if model.n_channels > 1:
                hidden = hidden + model.channel_out[0]
            logits = model._softcap(model._compute_logits(hidden))
            preds = logits[:, args.context_len - 1:].argmax(dim=-1)  # greedy predictions for target region

        # Decode
        context_ids = context[0].cpu().tolist()
        target_ids = target[0].cpu().tolist()
        pred_ids = preds[0].cpu().tolist()[:args.predict_len]

        context_text = sp.decode(context_ids)
        target_text = sp.decode(target_ids)
        pred_text = sp.decode(pred_ids)

        # Count correct tokens
        correct = sum(1 for t, p in zip(target_ids, pred_ids) if t == p)

        print(f"--- Example {idx+1} (start={start}, {correct}/{args.predict_len} tokens correct) ---")
        print(f"CONTEXT: ...{context_text[-200:]}")
        print(f"EXPECTED: {target_text[:200]}")
        print(f"PREDICTED: {pred_text[:200]}")
        print()


if __name__ == "__main__":
    main()
