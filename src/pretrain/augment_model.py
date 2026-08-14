#!/usr/bin/env python
"""
Augment ModernBERT with a domain vocabulary and FVT-initialize the new embeddings.

For each new whole-word token, its input embedding is initialized as the mean of
the ORIGINAL subword embeddings it used to be split into (Fast Vocabulary Transfer,
Gee et al. 2022), so it starts meaningful rather than random. Handles tied/untied
output embeddings. Verifies the fertility improvement on sample sentences, then
saves an augmented model + tokenizer ready for DAPT.

Usage:
  python augment_model.py --base answerdotai/ModernBERT-base \
    --vocab outputs/vocab/domain_vocab.txt --out outputs/models/modernbert-conflivocab-init
"""
from __future__ import annotations
import argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AddedToken

PROBE = [
    "The militants detonated an IED near the cantonment and opened fire on the convoy.",
    "Peshmerga forces clashed with insurgents during the counterinsurgency operation.",
    "A ceasefire collapsed as paramilitary shelling resumed near the checkpoint.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescale-norms", action="store_true",
                    help="rescale each FVT vector to the mean norm of its subword "
                         "constituents; a mean of k vectors is systematically short "
                         "(measured 0.67x here), and with tied embeddings short rows "
                         "get low logits, are never predicted, and so never train")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    orig_tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    aug_tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    model = AutoModelForMaskedLM.from_pretrained(args.base, dtype=torch.float32)
    base_vocab = orig_tok.vocab_size            # real token count (embeddings may be padded larger)

    words = [w.strip() for w in open(args.vocab, encoding="utf-8") if w.strip()]
    pre_added = set(aug_tok.get_added_vocab().keys())
    # whole-word tokens; lstrip absorbs the preceding space so we do not emit a stray space token
    aug_tok.add_tokens([AddedToken(w, single_word=True, lstrip=True, normalized=False) for w in words])
    new_added = {t: i for t, i in aug_tok.get_added_vocab().items() if t not in pre_added}
    print(f"[augment] requested {len(words)} words; added {len(new_added)} new tokens "
          f"-> len(tokenizer) {len(aug_tok)}", flush=True)

    # snapshot original embeddings BEFORE resize for FVT init
    old_emb = model.get_input_embeddings().weight.data.clone()
    tied = bool(getattr(model.config, "tie_word_embeddings", True))
    out_emb_old = (model.get_output_embeddings().weight.data.clone()
                   if (not tied and model.get_output_embeddings() is not None) else None)

    model.resize_token_embeddings(len(aug_tok))
    new_in = model.get_input_embeddings().weight.data
    new_out = (model.get_output_embeddings().weight.data
               if (not tied and model.get_output_embeddings() is not None) else None)

    # target the ACTUAL new token ids; FVT = mean of the original subwords of " <word>"
    fvt, fallback = 0, 0
    for tok_str, new_id in new_added.items():
        text = tok_str if tok_str.startswith(" ") else " " + tok_str
        sub_ids = [s for s in orig_tok.encode(text, add_special_tokens=False) if s < base_vocab]
        if sub_ids:
            subs = old_emb[torch.tensor(sub_ids)]
            vec = subs.mean(0)
            if args.rescale_norms:
                vec = vec * (subs.norm(dim=1).mean() / vec.norm().clamp_min(1e-8))
            new_in[new_id] = vec
            if new_out is not None and out_emb_old is not None:
                osubs = out_emb_old[torch.tensor(sub_ids)]
                ovec = osubs.mean(0)
                if args.rescale_norms:
                    ovec = ovec * (osubs.norm(dim=1).mean() / ovec.norm().clamp_min(1e-8))
                new_out[new_id] = ovec
            fvt += 1
        else:
            fallback += 1
    print(f"[augment] FVT-initialized {fvt} tokens ({fallback} fell back to default init)", flush=True)

    # ---- verify fertility improvement ----
    print("\n[augment] fertility check (content tokens, no specials):")
    tot_o, tot_a = 0, 0
    for s in PROBE:
        no = len(orig_tok(s, add_special_tokens=False)["input_ids"])
        na = len(aug_tok(s, add_special_tokens=False)["input_ids"])
        tot_o += no; tot_a += na
        print(f"  {na:>3} vs {no:>3}  ({100*(no-na)/no:4.1f}% fewer)  {s[:60]}")
    print(f"  TOTAL {tot_a} vs {tot_o}  ({100*(tot_o-tot_a)/tot_o:.1f}% fewer tokens)")

    aug_tok.save_pretrained(args.out)
    model.save_pretrained(args.out)
    print(f"\n[augment] saved augmented model + tokenizer -> {args.out}")
    print("AUGMENT_DONE")


if __name__ == "__main__":
    main()
