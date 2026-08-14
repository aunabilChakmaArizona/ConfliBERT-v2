#!/usr/bin/env python
"""
Convert a ModernBERT Composer pre-decay stable-phase checkpoint (.pt) into a
HuggingFace ModernBertForMaskedLM, WITHOUT installing composer.

The stable checkpoint stores the model weights under state['model'] with FlexBERT
naming (model.bert.encoder.layers.N...). The official convert_to_hf.py unpacks that
via composer, strips the leading 'model.', then applies a two-rule regex remap to
HF names. We replicate exactly that remap, then load the weights into the base
release's architecture (config is identical: base is context-extended, only the
weights differ), with strict validation so any mismatch fails loudly.

Usage:
  python convert_stable_ckpt.py --ckpt /root/ckpt_stable/context-extension/ep0-ba52988-rank0.pt \
     --base answerdotai/ModernBERT-base --out /root/models/modernbert-stable-hf
"""
import argparse, re, sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

VAR_MAP = (
    (re.compile(r"encoder\.layers\.(.*)"), r"layers.\1"),
    (re.compile(r"^bert\.(.*)"), r"model.\1"),
)


def remap(name):
    # composer step: strip the leading 'model.' that wraps the inner HF model
    if name.startswith("model."):
        name = name[len("model."):]
    for pat, repl in VAR_MAP:
        name = re.sub(pat, repl, name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[convert] loading composer checkpoint {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    src = ck["state"]["model"]
    remapped = {remap(k): v.float() for k, v in src.items()}
    print(f"[convert] {len(remapped)} tensors remapped; sample:",
          list(remapped.keys())[:3], flush=True)

    print(f"[convert] instantiating architecture from {args.base}", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.base, dtype=torch.float32, attn_implementation="sdpa")

    tie = getattr(model.config, "tie_word_embeddings", False)
    # HF ties decoder.weight to embeddings, so it is absent from a saved state dict.
    remapped.pop("decoder.weight", None)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # The ONLY acceptable missing key is the tied decoder.weight.
    missing = [m for m in missing if m != "decoder.weight"]
    if unexpected:
        print(f"[FATAL] unexpected keys (remap wrong): {unexpected[:10]}", file=sys.stderr)
        sys.exit(2)
    if missing:
        print(f"[FATAL] missing keys (remap wrong): {missing[:10]}", file=sys.stderr)
        sys.exit(2)
    print(f"[convert] load_state_dict clean (tie_word_embeddings={tie}, decoder tied)", flush=True)

    # re-tie so decoder.weight points at the freshly loaded embeddings
    model.tie_weights()

    # ---- validation gate: no NaN on a real forward pass ----
    tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    model.eval()
    with torch.no_grad():
        enc = tok(["Armed clashes were reported near the border on Tuesday.",
                   "The ceasefire agreement collapsed after renewed shelling."],
                  return_tensors="pt", padding=True)
        out = model(**enc)
    logits = out.logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("[FATAL] forward pass produced NaN/Inf logits", file=sys.stderr)
        sys.exit(3)
    print(f"[convert] forward OK: logits {tuple(logits.shape)} "
          f"finite, range [{logits.min():.2f}, {logits.max():.2f}]", flush=True)

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"[convert] saved HF model -> {args.out}", flush=True)
    print("CONVERT_DONE")


if __name__ == "__main__":
    main()
