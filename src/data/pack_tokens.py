#!/usr/bin/env python
"""
Stage 2: tokenize + pack a corpus split into fixed-length MLM blocks.

Streams the Parquet corpus, tokenizes with the ModernBERT tokenizer (no special
tokens), concatenates documents separated by [SEP], and chunks the stream into
blocks of exactly `--seqlen` tokens. Blocks are written as a uint16 memmap
(vocab < 65536) plus a small JSON meta file. This is the training input for DAPT.

Supports the data-selection ablation via `--source-weights` (keep fraction per
source) and a hard `--max-tokens` budget so runs are budget-comparable.

Usage:
  python pack_tokens.py --corpus .../outputs/corpus/parquet \
    --split train --seqlen 1024 --max-tokens 1_500_000_000 \
    --out .../outputs/packed/train_1024
"""
from __future__ import annotations
import argparse, glob, json, os, random
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODERNBERT = "answerdotai/ModernBERT-base"


def parse_weights(s: str):
    if not s:
        return None
    w = {}
    for part in s.split(","):
        k, v = part.split("=")
        w[k.strip()] = float(v)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="train", choices=["train", "eval_random", "eval_temporal"])
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = no cap")
    ap.add_argument("--source-weights", default="", help="e.g. News=1.0,Gigaword=0.5,Wikipedia=0.3")
    ap.add_argument("--tokenizer", default=MODERNBERT)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dedup", action="store_true", help="exact-dedup by text_hash")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seen = set() if args.dedup else None
    n_dup = 0
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    weights = parse_weights(args.source_weights)
    rng = random.Random(args.seed)

    files = sorted(glob.glob(os.path.join(args.corpus, "**", "*.parquet"), recursive=True))
    rng.shuffle(files)  # mix sources/time so packed order is not sorted by shard

    memmap_path = os.path.join(args.out, "tokens.u16")
    seqlen = args.seqlen
    # write incrementally to a growable list of blocks flushed to disk
    fout = open(memmap_path, "wb")
    carry = []                 # leftover token ids that have not filled a block
    n_blocks = 0
    n_tokens_written = 0
    n_docs = 0
    stop = False

    def flush_blocks():
        nonlocal carry, n_blocks, n_tokens_written
        while len(carry) >= seqlen:
            block = np.asarray(carry[:seqlen], dtype=np.uint16)
            fout.write(block.tobytes())
            carry = carry[seqlen:]
            n_blocks += 1
            n_tokens_written += seqlen

    for f in files:
        if stop:
            break
        src = os.path.basename(os.path.dirname(f))
        keep = 1.0 if weights is None else weights.get(src, 0.0)
        if keep <= 0.0:
            continue
        cols = ["source", "text", "split"] + (["text_hash"] if seen is not None else [])
        t = pq.read_table(f, columns=cols)
        texts = t.column("text").to_pylist()
        splits = t.column("split").to_pylist()
        hashes = t.column("text_hash").to_pylist() if seen is not None else None
        batch = []
        for i in range(len(texts)):
            if splits[i] != args.split:
                continue
            if keep < 1.0 and rng.random() > keep:
                continue
            if seen is not None:
                if hashes[i] in seen:
                    n_dup += 1
                    continue
                seen.add(hashes[i])
            batch.append(texts[i])
            if len(batch) >= args.batch:
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    carry.extend(ids); carry.append(sep); n_docs += 1
                flush_blocks(); batch = []
                if args.max_tokens and n_tokens_written >= args.max_tokens:
                    stop = True; break
        if batch and not stop:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                carry.extend(ids); carry.append(sep); n_docs += 1
            flush_blocks()
            if args.max_tokens and n_tokens_written >= args.max_tokens:
                stop = True
        print(f"  {src:12s} {os.path.basename(f):32s} blocks={n_blocks:,} "
              f"tokens={n_tokens_written/1e6:.1f}M docs={n_docs:,}", flush=True)

    fout.close()
    meta = {
        "path": os.path.basename(memmap_path), "dtype": "uint16", "seqlen": seqlen,
        "n_blocks": n_blocks, "n_tokens": n_tokens_written, "n_docs": n_docs,
        "split": args.split, "tokenizer": args.tokenizer,
        "source_weights": weights, "max_tokens": args.max_tokens,
        "dedup": bool(seen is not None), "n_duplicates_skipped": n_dup,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nPACK_DONE blocks={n_blocks:,} tokens={n_tokens_written:,} "
          f"({n_tokens_written/1e9:.2f}B) docs={n_docs:,} -> {args.out}")


if __name__ == "__main__":
    main()
