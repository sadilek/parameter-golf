"""Show top-5 predictions with probabilities for each position."""
import io, torch, lzma, numpy as np, sentencepiece as spm, sys
sys.path.insert(0, '.')
from train_combined import GPT, deq_sd_int6

sp = spm.SentencePieceProcessor('./data/tokenizers/fineweb_1024_bpe.model')
device = torch.device('cuda')

header = np.fromfile('./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin', dtype='<i4', count=256)
tokens = np.fromfile('./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin', dtype='<u2', count=int(header[2]), offset=256*4)
tokens = torch.from_numpy(tokens.astype(np.int64))

with open('final_model.ptz', 'rb') as f:
    raw = f.read()
try:
    dec = lzma.decompress(raw)
except Exception:
    import zstandard
    dec = zstandard.ZstdDecompressor().decompress(raw)
loaded = torch.load(io.BytesIO(dec), map_location='cpu', weights_only=False)
sd = deq_sd_int6(loaded, target_dtype=torch.bfloat16)

model = GPT(vocab_size=1024, num_layers=13, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
            qk_gain_init=1.5, activation='swiglu', n_channels=8, xsa_layers=4)
model.load_state_dict(sd, strict=False)
model = model.to(device).bfloat16().eval()

# Two examples at different positions
for example_idx, start in enumerate([1000, 50000, 200000]):
    chunk = tokens[start:start+80].to(device).unsqueeze(0)
    context_len = 48

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        emb = model._embed(chunk[:, :-1])
        emb = emb + model.channel_in[0]
        hidden = model._run_blocks(emb, emb, causal=True)
        hidden = hidden + model.channel_out[0]
        logits = model._softcap(model._compute_logits(hidden))
        probs = torch.softmax(logits.float(), dim=-1)

    context_text = sp.decode(chunk[0, :context_len].cpu().tolist())
    target_text = sp.decode(chunk[0, context_len:context_len+16].cpu().tolist())
    print(f"\n{'='*90}")
    print(f"Example {example_idx+1} (start={start})")
    print(f"CONTEXT: ...{context_text[-150:]}")
    print(f"TARGET:  {target_text[:150]}")
    print(f"{'='*90}")
    print(f"{'pos':>5} {'correct':>7} {'target':>12} {'p(target)':>10} {'top-5 predictions (token: probability)':}")
    print(f"{'-'*90}")

    for pos in range(context_len-1, min(context_len+15, 79)):
        target_id = chunk[0, pos+1].item()
        target_piece = sp.id_to_piece(target_id).replace('\u2581', '_')
        prob_target = probs[0, pos, target_id].item()

        top5_probs, top5_ids = probs[0, pos].topk(5)
        preds_str = "  ".join(
            f"{'*' if tid.item()==target_id else ' '}{sp.id_to_piece(tid.item()).replace(chr(0x2581),'_')}:{tp.item():.3f}"
            for tid, tp in zip(top5_ids, top5_probs)
        )

        correct = "YES" if top5_ids[0].item() == target_id else ""
        rank = 1 + (probs[0, pos] > prob_target).sum().item()
        print(f"  {pos+1:3d} {correct:>7} {target_piece:>12} {prob_target:>9.3f}  (rank {rank:>3d})  {preds_str}")
