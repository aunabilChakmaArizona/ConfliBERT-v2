#!/usr/bin/env python
"""Authoritative corpus statistics from the extracted Parquet (metadata only, no text load).

Writes for the paper / R:
  corpus_stats.csv     source x split: n_articles, n_chars, n_words
  corpus_by_year.csv   source x year : n_articles
and prints a summary table.
"""
from __future__ import annotations
import argparse, csv, glob, os
from collections import defaultdict
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/f/Confli_2/Corpus/conflibert-v2/outputs/corpus/parquet")
    ap.add_argument("--out", default="/mnt/f/Confli_2/Corpus/conflibert-v2/analysis/data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.corpus, "**", "*.parquet"), recursive=True))
    ss = defaultdict(lambda: [0, 0, 0])          # (source,split) -> [n, chars, words]
    sy = defaultdict(int)                        # (source,year) -> n
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=["source", "split", "year", "n_chars", "n_words"])
        src = t.column("source").to_pylist()
        spl = t.column("split").to_pylist()
        yr = t.column("year").to_pylist()
        nc = t.column("n_chars").to_pylist()
        nw = t.column("n_words").to_pylist()
        for j in range(len(src)):
            a = ss[(src[j], spl[j])]; a[0] += 1; a[1] += nc[j]; a[2] += nw[j]
            sy[(src[j], yr[j])] += 1
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(files)} shards", flush=True)

    with open(os.path.join(args.out, "corpus_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "split", "n_articles", "n_chars", "n_words"])
        for (src, spl), (n, c, wd) in sorted(ss.items()):
            w.writerow([src, spl, n, c, wd])
    with open(os.path.join(args.out, "corpus_by_year.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "year", "n_articles"])
        for (src, yr), n in sorted(sy.items()):
            w.writerow([src, yr, n])

    # console summary
    by_source = defaultdict(lambda: [0, 0, 0])
    by_split = defaultdict(int)
    for (src, spl), (n, c, wd) in ss.items():
        a = by_source[src]; a[0] += n; a[1] += c; a[2] += wd
        by_split[spl] += n
    tot_n = sum(a[0] for a in by_source.values())
    tot_w = sum(a[2] for a in by_source.values())
    print("\n=== per source ===")
    for src, (n, c, wd) in sorted(by_source.items()):
        print(f"  {src:14s} {n:>11,} articles  {c/1e9:6.2f}G chars  {wd/1e6:8.1f}M words")
    print("\n=== per split ===")
    for spl, n in sorted(by_split.items()):
        print(f"  {spl:14s} {n:>11,}")
    print(f"\ntotal: {tot_n:,} articles | {tot_w/1e6:.1f}M words | "
          f"~{tot_w*1.3/1e9:.2f}B ModernBERT tokens (rough 1.3x words)")
    print("CORPUS_STATS_DONE")


if __name__ == "__main__":
    main()
