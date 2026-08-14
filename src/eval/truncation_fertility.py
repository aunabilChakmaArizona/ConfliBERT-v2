#!/usr/bin/env python
"""
Flagship analysis: the truncation gap + tokenizer fertility.

Reads the extracted corpus (Parquet), draws a source-stratified sample, and
tokenizes each article with the ModernBERT tokenizer (v2) and the ConfliBERT
tokenizer (incumbent). Produces CSVs consumed by R:

  1. token_lengths.csv    per-article token counts + words, per tokenizer.
                          -> distribution of article lengths, % exceeding 512 /
                             1024 / 2048 / 4096, and mean fraction of the article
                             that a 512-token model never sees (the "truncation gap").
  2. fertility_source.csv fertility (subword tokens / whitespace word) per source
                          per tokenizer.
  3. fertility_terms.csv  fertility on a curated conflict-jargon lexicon:
                          subword count per term under each tokenizer.

No GPU needed. Deterministic sampling.

Usage:
  python truncation_fertility.py \
    --corpus /mnt/f/Confli_2/Corpus/conflibert-v2/outputs/corpus/parquet \
    --out    /mnt/f/Confli_2/Corpus/conflibert-v2/analysis/data \
    --per-source 60000
"""
from __future__ import annotations
import argparse, csv, glob, os, random
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODERNBERT = "answerdotai/ModernBERT-base"
CONFLIBERT = "eventdata-utd/ConfliBERT-scr-uncased"
THRESHOLDS = [512, 1024, 2048, 4096]

# Curated conflict / political-violence lexicon for the fertility probe.
CONFLICT_TERMS = [
    "militants", "insurgents", "paramilitary", "cantonment", "counterinsurgency",
    "improvised explosive device", "IED", "VBIED", "RPG", "mortar", "airstrike",
    "ceasefire", "insurgency", "jihadist", "caliphate", "peacekeepers",
    "displacement", "refugees", "atrocities", "genocide", "ethnic cleansing",
    "assassination", "kidnapping", "hostage", "beheading", "suicide bomber",
    "Hezbollah", "Hamas", "Taliban", "al-Qaeda", "Boko Haram", "Al-Shabaab",
    "Jaish-e-Mohammed", "Lashkar-e-Taiba", "peshmerga", "junta", "coup",
    "secessionist", "belligerents", "combatants", "demobilization", "disarmament",
    "reconciliation", "war crimes", "shelling", "artillery", "checkpoint",
    "curfew", "martyrdom", "cadre", "exfiltration", "reconnaissance",
]


def load_tokenizers():
    mb = AutoTokenizer.from_pretrained(MODERNBERT, clean_up_tokenization_spaces=False)
    cb = AutoTokenizer.from_pretrained(CONFLIBERT, clean_up_tokenization_spaces=False)
    return mb, cb


def n_tokens(tok, text: str) -> int:
    # content tokens only, no special tokens; cap encode work at a generous length
    return len(tok(text, add_special_tokens=False, truncation=True,
                   max_length=16384)["input_ids"])


def sample_articles(corpus_dir: str, per_source: int, seed: int):
    files = glob.glob(os.path.join(corpus_dir, "**", "*.parquet"), recursive=True)
    by_source = {}
    for f in files:
        src = os.path.basename(os.path.dirname(f))
        by_source.setdefault(src, []).append(f)
    rng = random.Random(seed)
    out = []  # (source, words, text)
    for src, fs in sorted(by_source.items()):
        rng.shuffle(fs)
        picked = 0
        for f in fs:
            if picked >= per_source:
                break
            t = pq.read_table(f, columns=["source", "n_words", "text", "split"])
            words = t.column("n_words").to_pylist()
            texts = t.column("text").to_pylist()
            splits = t.column("split").to_pylist()
            idx = list(range(len(texts)))
            rng.shuffle(idx)
            for i in idx:
                if picked >= per_source:
                    break
                if splits[i] == "eval_temporal":   # keep the future holdout untouched
                    continue
                out.append((src, words[i], texts[i]))
                picked += 1
        print(f"  sampled {picked:,} from {src}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[fertility] loading tokenizers", flush=True)
    mb, cb = load_tokenizers()

    # ---- 3. lexicon fertility (cheap, do first) ----
    with open(os.path.join(args.out, "fertility_terms.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["term", "words", "modernbert_tokens", "conflibert_tokens"])
        for term in CONFLICT_TERMS:
            w.writerow([term, term.count(" ") + 1, n_tokens(mb, term), n_tokens(cb, term)])
    print("[fertility] wrote fertility_terms.csv", flush=True)

    # ---- sample corpus ----
    print(f"[fertility] sampling up to {args.per_source:,}/source", flush=True)
    sample = sample_articles(args.corpus, args.per_source, args.seed)
    print(f"[fertility] total sampled: {len(sample):,}", flush=True)

    # ---- 1 & 2. token lengths + per-source fertility ----
    agg = {}  # source -> [words, mb_tok, cb_tok]
    tl_path = os.path.join(args.out, "token_lengths.csv")
    with open(tl_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "words", "modernbert_tokens", "conflibert_tokens"])
        for n, (src, words, text) in enumerate(sample):
            mbn = n_tokens(mb, text); cbn = n_tokens(cb, text)
            w.writerow([src, words, mbn, cbn])
            a = agg.setdefault(src, [0, 0, 0])
            a[0] += words; a[1] += mbn; a[2] += cbn
            if (n + 1) % 20000 == 0:
                print(f"  tokenized {n+1:,}/{len(sample):,}", flush=True)

    with open(os.path.join(args.out, "fertility_source.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "words", "modernbert_tokens", "conflibert_tokens",
                    "modernbert_fertility", "conflibert_fertility"])
        for src, (ww, mbn, cbn) in sorted(agg.items()):
            w.writerow([src, ww, mbn, cbn, round(mbn / max(ww, 1), 4), round(cbn / max(ww, 1), 4)])

    # ---- console summary of the truncation gap ----
    print("\n=== truncation gap (ModernBERT token counts) ===")
    import statistics
    mb_lens = []
    with open(tl_path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            mb_lens.append(int(row["modernbert_tokens"]))
    mb_lens.sort()
    n = len(mb_lens)
    def pct_over(k): return 100.0 * sum(1 for x in mb_lens if x > k) / n
    print(f"  articles: {n:,} | median tok: {mb_lens[n//2]} | mean: {sum(mb_lens)/n:.0f} | "
          f"p95: {mb_lens[int(0.95*n)]} | max: {mb_lens[-1]}")
    for k in THRESHOLDS:
        print(f"  % articles > {k:>4} tokens (truncated by a {k}-ctx model): {pct_over(k):5.1f}%")
    # mean fraction of content a 512-ctx model never sees
    lost = sum(max(0, x - 512) for x in mb_lens) / sum(mb_lens)
    print(f"  fraction of ALL corpus tokens beyond position 512 (lost to truncation): {100*lost:.1f}%")
    print("FERTILITY_DONE")


if __name__ == "__main__":
    main()
