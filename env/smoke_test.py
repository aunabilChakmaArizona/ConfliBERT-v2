"""Smoke test: ModernBERT + ConfliBERT load, tokenize, and run MLM on the GPU."""
import time, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "| bf16 supported:", torch.cuda.is_bf16_supported() if DEV == "cuda" else False)

SENT = "Militants detonated an IED near the cantonment and opened fire on the convoy."

# 1) ConfliBERT tokenizer (incumbent) - just tokenizer, for fertility comparison
print("\n[ConfliBERT tokenizer]")
cb_tok = AutoTokenizer.from_pretrained("eventdata-utd/ConfliBERT-scr-uncased")
cb_ids = cb_tok(SENT)["input_ids"]
print("  vocab:", cb_tok.vocab_size, "| model_max_len:", cb_tok.model_max_length)
print("  tokens:", cb_tok.convert_ids_to_tokens(cb_ids))

# 2) ModernBERT: tokenizer + model + MLM forward
print("\n[ModernBERT-base]")
mb_tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
mb_ids = mb_tok(SENT)["input_ids"]
print("  vocab:", mb_tok.vocab_size, "| model_max_len:", mb_tok.model_max_length)
print("  tokens:", mb_tok.convert_ids_to_tokens(mb_ids))

t0 = time.time()
model = AutoModelForMaskedLM.from_pretrained(
    "answerdotai/ModernBERT-base", torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float32,
    attn_implementation="sdpa",
).to(DEV).eval()
print(f"  model loaded in {time.time()-t0:.1f}s | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

masked = SENT.replace("IED", mb_tok.mask_token)
enc = mb_tok(masked, return_tensors="pt").to(DEV)
with torch.no_grad():
    out = model(**enc).logits
mask_pos = (enc["input_ids"] == mb_tok.mask_token_id).nonzero()[0, 1]
top5 = out[0, mask_pos].topk(5).indices.tolist()
print("  MLM fill for '[MASK]' (was IED):", [mb_tok.decode([t]).strip() for t in top5])
print("  peak VRAM (MB):", round(torch.cuda.max_memory_allocated()/1e6, 1) if DEV == "cuda" else "n/a")
print("\nSMOKE_TEST_OK")
