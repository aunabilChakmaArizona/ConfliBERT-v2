#!/usr/bin/env python
"""
Intrinsic evaluation: masked-LM pseudo-perplexity on held-out conflict text.

Applies a FIXED (seeded) mask to each eval block so every model is scored on
identical masked positions, then reports mean masked-token cross-entropy and
pseudo-perplexity = exp(loss).

NOTE: this metric is only comparable WITHIN a tokenizer family (ModernBERT-base
vs ConfliBERT-v2 vs the ablation checkpoints, all sharing ModernBERT's vocab).
It is NOT comparable to the original ConfliBERT, which has a different vocab and
baseline entropy; that comparison is made extrinsically on downstream tasks.

Usage:
  python pseudo_ppl.py --eval .../outputs/packed/eval_random_1024 \
    --models base=answerdotai/ModernBERT-base v2=.../outputs/models/conflibert-v2 \
    --out .../analysis/data/pseudo_ppl.csv --mlm-prob 0.30 --max-blocks 2000
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODERNBERT = "answerdotai/ModernBERT-base"


def load_blocks(packed_dir, max_blocks):
    meta = json.load(open(os.path.join(packed_dir, "meta.json")))
    n, seqlen = meta["n_blocks"], meta["seqlen"]
    arr = np.memmap(os.path.join(packed_dir, meta["path"]), dtype=np.uint16, mode="r",
                    shape=(n, seqlen))
    if max_blocks and n > max_blocks:
        idx = np.linspace(0, n - 1, max_blocks).astype(int)
        arr = np.asarray(arr[idx])
    else:
        arr = np.asarray(arr)
    return arr.astype(np.int64)


def build_fixed_masks(blocks, special_ids, mlm_prob, mask_id, vocab_size, seed=1234):
    """Deterministic mask positions shared across all models."""
    rng = np.random.default_rng(seed)
    special = np.array(sorted(special_ids))
    inputs = blocks.copy()
    labels = np.full_like(blocks, -100)
    for i in range(blocks.shape[0]):
        row = blocks[i]
        maskable = ~np.isin(row, special)
        probs = rng.random(row.shape) < mlm_prob
        sel = probs & maskable
        labels[i, sel] = row[sel]
        # 80% [MASK], 10% random, 10% keep (standard)
        r = rng.random(row.shape)
        inputs[i, sel & (r < 0.8)] = mask_id
        rand_sel = sel & (r >= 0.9)
        inputs[i, rand_sel] = rng.integers(0, vocab_size, rand_sel.sum())
    return inputs, labels


@torch.no_grad()
def score(model_path, inputs, labels, device, bsz=16):
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    total_loss, total_tok = 0.0, 0
    lossf = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    for s in range(0, inputs.shape[0], bsz):
        ii = torch.from_numpy(inputs[s:s+bsz]).to(device)
        ll = torch.from_numpy(labels[s:s+bsz]).to(device)
        logits = model(input_ids=ii).logits.float()
        loss = lossf(logits.view(-1, logits.size(-1)), ll.view(-1))
        ntok = (ll != -100).sum().item()
        total_loss += loss.item(); total_tok += ntok
    del model
    torch.cuda.empty_cache()
    return total_loss / max(total_tok, 1), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlm-prob", type=float, default=0.30)
    ap.add_argument("--max-blocks", type=int, default=2000)
    ap.add_argument("--tokenizer", default=MODERNBERT,
                    help="must match the tokenizer the eval blocks were packed with")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    blocks = load_blocks(args.eval, args.max_blocks)
    inputs, labels = build_fixed_masks(blocks, tok.all_special_ids, args.mlm_prob,
                                       tok.mask_token_id, len(tok))
    print(f"[ppl] eval blocks={blocks.shape[0]} seqlen={blocks.shape[1]} "
          f"masked tokens={(labels!=-100).sum():,}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        loss, ntok = score(path, inputs, labels, device)
        ppl = float(np.exp(min(loss, 20)))
        print(f"  {name:10s} loss={loss:.4f}  pseudo_ppl={ppl:.3f}", flush=True)
        rows.append([name, path, round(loss, 5), round(ppl, 4), ntok])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "path", "mlm_loss", "pseudo_ppl", "n_masked_tokens"])
        w.writerows(rows)
    print("PPL_DONE")


if __name__ == "__main__":
    main()
