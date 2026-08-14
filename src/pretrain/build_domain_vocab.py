#!/usr/bin/env python
"""
Mine the corpus for high-value domain tokens to add to ModernBERT's tokenizer.

A candidate is a frequent word that ModernBERT's tokenizer currently splits into
>= 2 subwords. We rank candidates by TOKENS SAVED = frequency * (fertility - 1),
i.e. how much total subword fragmentation adding this whole-word token removes.
Writes:
  domain_vocab.txt        one selected word per line (surface form, cased)
  domain_vocab_stats.csv  word, frequency, base_fertility, tokens_saved
"""
from __future__ import annotations
import argparse, csv, glob, os, random, re
from collections import Counter
import pyarrow.parquet as pq
from transformers import AutoTokenizer

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
MODERNBERT = "answerdotai/ModernBERT-base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-docs", type=int, default=400000)
    ap.add_argument("--min-freq", type=int, default=80, help="min corpus count to consider")
    ap.add_argument("--min-len", type=int, default=3, help="min word length")
    ap.add_argument("--add-tokens", type=int, default=4000, help="how many to select")
    ap.add_argument("--tokenizer", default=MODERNBERT)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    existing = set(tok.get_vocab().keys())
    rng = random.Random(args.seed)

    files = glob.glob(os.path.join(args.corpus, "**", "*.parquet"), recursive=True)
    rng.shuffle(files)
    counts = Counter()
    seen_docs = 0
    for f in files:
        if seen_docs >= args.sample_docs:
            break
        t = pq.read_table(f, columns=["text", "split"])
        texts = t.column("text").to_pylist()
        splits = t.column("split").to_pylist()
        for i in range(len(texts)):
            if splits[i] != "train":
                continue
            counts.update(WORD_RE.findall(texts[i]))
            seen_docs += 1
            if seen_docs >= args.sample_docs:
                break
        if seen_docs % 40000 < 2000:
            print(f"  scanned {seen_docs:,} docs, {len(counts):,} unique words", flush=True)

    # consider only words above min-freq; compute base fertility.
    # exclude apostrophe forms (contractions/possessives) -- generic, not domain vocab.
    cands = [(w, c) for w, c in counts.items()
             if c >= args.min_freq and len(w) >= args.min_len and "'" not in w]
    print(f"[vocab] {len(cands):,} candidate words above freq {args.min_freq}", flush=True)

    scored = []
    for w, c in cands:
        # fertility of the word as it appears after a space (typical context)
        ntok = len(tok.tokenize(" " + w))
        if ntok >= 2:                      # only words we actually fragment
            scored.append((w, c, ntok, c * (ntok - 1)))
    scored.sort(key=lambda x: x[3], reverse=True)
    selected = scored[:args.add_tokens]

    with open(os.path.join(args.out, "domain_vocab.txt"), "w", encoding="utf-8") as fh:
        for w, _, _, _ in selected:
            fh.write(w + "\n")
    with open(os.path.join(args.out, "domain_vocab_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh); w_.writerow(["word", "frequency", "base_fertility", "tokens_saved"])
        for w, c, nt, sv in scored:
            w_.writerow([w, c, nt, sv])

    tot_saved = sum(s[3] for s in selected)
    print(f"[vocab] selected {len(selected):,} tokens | est tokens saved in sample: {tot_saved:,}")
    print("  top 20:", [s[0] for s in selected[:20]])
    print("VOCAB_DONE")


if __name__ == "__main__":
    main()
