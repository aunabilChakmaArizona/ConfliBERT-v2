#!/usr/bin/env python
"""
Stage 1 (generic): ingest ANY team corpus source into the Parquet contract.

extract_corpus.py handles the original EN-Politics .json.tar.gz archives; this
script handles everything else the team brings (journal dumps, dissertations,
FineWeb-style crawls, scraped text) in the common interchange formats:

    .jsonl / .jsonl.gz   one JSON object per line (--text-field, --date-field)
    .json                a single JSON array of objects
    .csv / .tsv          header row required (--text-field, --date-field)
    .txt / .txt.gz       one DOCUMENT per file (the whole file is the text)

PDFs and other binary formats are out of scope on purpose: convert them to
JSONL first (src/data/crisiswatch_parse.py is a worked PDF example), then
ingest the JSONL.

Cleaning, hashing, and split assignment are IMPORTED from extract_corpus.py,
so every source obeys the identical corpus contract:

    id, source, date, year, n_chars, n_words, text, text_hash, split

Usage:
    python src/data/ingest_source.py --source ArizonaJournals \
        --input /path/to/dump --out /scratch/.../corpus \
        [--text-field body] [--date-field published]

Then validate: python src/data/corpus_stats.py --corpus <out>/parquet --out analysis/data
"""
from __future__ import annotations
import argparse, csv, glob, gzip, hashlib, io, os, sys

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_corpus import SCHEMA, assign_split, clean_text, parse_year

TEXT_EXTS = (".jsonl", ".jsonl.gz", ".json", ".csv", ".tsv", ".txt", ".txt.gz")
csv.field_size_limit(2**31 - 1)


def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_records(path: str, text_field: str, date_field: str, id_field: str):
    """Yield (record_id_suffix, raw_text, raw_date) per document in one input file."""
    low = path.lower()
    if low.endswith((".jsonl", ".jsonl.gz")):
        with open_maybe_gz(path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                rid = str(obj.get(id_field)) if id_field and obj.get(id_field) is not None else str(i)
                yield rid, obj.get(text_field) or "", obj.get(date_field)
    elif low.endswith(".json"):
        with open_maybe_gz(path) as fh:
            arr = orjson.loads(fh.read())
        if not isinstance(arr, list):
            raise ValueError(f"{path}: top-level JSON must be an array of objects")
        for i, obj in enumerate(arr):
            rid = str(obj.get(id_field)) if id_field and obj.get(id_field) is not None else str(i)
            yield rid, obj.get(text_field) or "", obj.get(date_field)
    elif low.endswith((".csv", ".tsv")):
        delim = "\t" if low.endswith(".tsv") else ","
        with open_maybe_gz(path) as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            if reader.fieldnames is None or text_field not in reader.fieldnames:
                raise ValueError(f"{path}: no '{text_field}' column (have: {reader.fieldnames})")
            for i, row in enumerate(reader):
                rid = row.get(id_field) or str(i) if id_field else str(i)
                yield rid, row.get(text_field) or "", row.get(date_field)
    elif low.endswith((".txt", ".txt.gz")):
        with open_maybe_gz(path) as fh:
            yield "", fh.read(), None
    else:
        raise ValueError(f"unsupported extension: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="canonical source name; becomes the parquet/<Source>/ subdir "
                         "and the value packing weights key on. Register it in "
                         "docs/PIPELINE.md before first use.")
    ap.add_argument("--input", required=True, help="input file, or directory scanned recursively")
    ap.add_argument("--out", required=True, help="corpus root; shards go to <out>/parquet/<source>/")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--date-field", default="date")
    ap.add_argument("--id-field", default="", help="optional stable id column/key in the input")
    ap.add_argument("--eval-year", type=int, default=2021)
    ap.add_argument("--eval-frac", type=float, default=0.003)
    ap.add_argument("--min-chars", type=int, default=32)
    ap.add_argument("--shard-rows", type=int, default=50000, help="rows per output parquet shard")
    ap.add_argument("--append", action="store_true",
                    help="continue shard numbering in a non-empty source dir "
                         "(default: refuse, to prevent accidental double-ingest)")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(p for p in glob.glob(os.path.join(args.input, "**", "*"), recursive=True)
                       if p.lower().endswith(TEXT_EXTS))
    else:
        files = [args.input]
    if not files:
        sys.exit(f"[abort] no ingestible files under {args.input} (supported: {TEXT_EXTS})")

    out_dir = os.path.join(args.out, "parquet", args.source)
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, f"{args.source}_*.parquet")))
    if existing and not args.append:
        sys.exit(f"[abort] {out_dir} already has {len(existing)} shards; "
                 f"pass --append to continue numbering, or clear the dir first")
    shard_no = len(existing)

    rows = {k: [] for k in SCHEMA.names}
    stats = {"docs": 0, "chars": 0, "dropped": 0, "train": 0,
             "eval_temporal": 0, "eval_random": 0}

    def flush():
        nonlocal shard_no, rows
        if not rows["id"]:
            return
        path = os.path.join(out_dir, f"{args.source}_{shard_no:05d}.parquet")
        pq.write_table(pa.table(rows, schema=SCHEMA), path, compression="zstd")
        print(f"  wrote {os.path.basename(path)} rows={len(rows['id']):,}", flush=True)
        shard_no += 1
        rows = {k: [] for k in SCHEMA.names}

    base = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    for f in files:
        rel = os.path.relpath(f, base).replace("\\", "/")
        try:
            records = iter_records(f, args.text_field, args.date_field, args.id_field)
            for rid, raw_text, raw_date in records:
                text = clean_text(raw_text if isinstance(raw_text, str) else str(raw_text))
                if len(text) < args.min_chars:
                    stats["dropped"] += 1
                    continue
                h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).hexdigest()
                year = parse_year(raw_date)
                split = assign_split(h, year, args.eval_year, args.eval_frac)
                rows["id"].append(f"{args.source}:{rel}:{rid}" if rid else f"{args.source}:{rel}")
                rows["source"].append(args.source)
                rows["date"].append(str(raw_date or "")[:32])
                rows["year"].append(year if year is not None else -1)
                rows["n_chars"].append(len(text))
                rows["n_words"].append(text.count(" ") + 1)
                rows["text"].append(text)
                rows["text_hash"].append(h)
                rows["split"].append(split)
                stats["docs"] += 1
                stats["chars"] += len(text)
                stats[split] += 1
                if len(rows["id"]) >= args.shard_rows:
                    flush()
        except ValueError as e:
            print(f"  [skip] {e}", flush=True)
    flush()

    print(f"\nINGEST_DONE source={args.source} docs={stats['docs']:,} "
          f"chars={stats['chars']/1e9:.2f}G dropped={stats['dropped']:,} "
          f"splits: train={stats['train']:,} eval_temporal={stats['eval_temporal']:,} "
          f"eval_random={stats['eval_random']:,} -> {out_dir}")
    print("next: python src/data/corpus_stats.py --corpus "
          f"{os.path.join(args.out, 'parquet')} --out analysis/data")


if __name__ == "__main__":
    main()
