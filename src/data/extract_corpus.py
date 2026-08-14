#!/usr/bin/env python
"""
Stage 1: extract the EN-Politics conflict corpus into clean columnar shards.

Each input file is a `.json.tar.gz` wrapping a JSONL file whose lines are
{"title","date","text","url"} article records. We stream every shard in
parallel, clean the text, assign a train / eval split, and write one Parquet
shard per input file with columns:

    id, source, date, year, n_chars, n_words, text, text_hash, split

A lightweight metadata index (everything except `text`) is derived from these
in a later step, so we do not duplicate the text twice on disk.

Splits (for MLM / pseudo-perplexity only; downstream tasks are separate data):
    eval_temporal : articles dated in --eval-year (default 2021)  -> future holdout
    eval_random   : deterministic ~--eval-frac of the rest        -> in-domain holdout
    train         : everything else

Usage (inside the WSL venv):
    python extract_corpus.py --corpus-root "/mnt/f/Confli_2/Corpus" \
        --out "/mnt/f/Confli_2/Corpus/conflibert-v2/outputs/corpus" [--limit N] [--workers 14]
"""
from __future__ import annotations
import argparse, glob, hashlib, os, re, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import orjson
import ftfy
import pyarrow as pa
import pyarrow.parquet as pq

# ---- source folder -> canonical source name ---------------------------------
SOURCE_MAP = {
    "1.News": "News", "2.Organization": "Organization", "3.Gigaword": "Gigaword",
    "4.UTDstory": "UTDstory", "5.wikipedia": "Wikipedia",
}
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")

SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("date", pa.string()),
    ("year", pa.int16()), ("n_chars", pa.int32()), ("n_words", pa.int32()),
    ("text", pa.large_string()), ("text_hash", pa.string()), ("split", pa.string()),
])


def source_of(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p in SOURCE_MAP:
            return SOURCE_MAP[p]
    return "Unknown"


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = ftfy.fix_text(t)              # repair cp1252 mojibake, mixed encodings
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = WS_RE.sub(" ", t)
    t = NL_RE.sub("\n\n", t)
    return t.strip()


def parse_year(date_val):
    if date_val is None or date_val == "":
        return None
    s = str(date_val).strip()
    m = DATE_RE.match(s)
    if m:
        y = int(m.group(1))
    elif s[:4].isdigit():          # e.g. int dates like 199405 or "1994"
        y = int(s[:4])
    else:
        return None
    return y if 1900 <= y <= 2100 else None


def assign_split(text_hash: str, year, eval_year: int, eval_frac: float) -> str:
    if year == eval_year:
        return "eval_temporal"
    # deterministic hash bucket in [0,1) for the random holdout
    bucket = int(text_hash[:8], 16) / 0xFFFFFFFF
    return "eval_random" if bucket < eval_frac else "train"


def process_shard(args) -> dict:
    path, out_dir, eval_year, eval_frac = args
    source = source_of(path)
    stem = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]  # strip .json.tar.gz
    out_path = os.path.join(out_dir, source, f"{stem}.parquet")
    if os.path.exists(out_path):
        return {"shard": stem, "source": source, "skipped": True}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = {k: [] for k in SCHEMA.names}
    stats = {"shard": stem, "source": source, "n": 0, "train": 0,
             "eval_temporal": 0, "eval_random": 0, "empty": 0, "chars": 0, "skipped": False}
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(".json")), None)
            if member is None:
                return {**stats, "error": "no .json in tar"}
            data = tar.extractfile(member).read()
    except Exception as e:  # corrupt archive -> report, do not crash the pool
        return {**stats, "error": f"read: {e}"}

    for i, line in enumerate(data.split(b"\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
            text = clean_text(obj.get("text") or "")
        except Exception:
            continue
        if len(text) < 32:                 # drop near-empty stubs
            stats["empty"] += 1
            continue
        h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).hexdigest()
        year = parse_year(obj.get("date"))
        split = assign_split(h, year, eval_year, eval_frac)
        rows["id"].append(f"{stem}:{i}")
        rows["source"].append(source)
        rows["date"].append(str(obj.get("date") or "")[:32])
        rows["year"].append(year if year is not None else -1)
        rows["n_chars"].append(len(text))
        rows["n_words"].append(text.count(" ") + 1)
        rows["text"].append(text)
        rows["text_hash"].append(h)
        rows["split"].append(split)
        stats["n"] += 1
        stats["chars"] += len(text)
        stats[split] += 1

    if stats["n"]:
        pq.write_table(pa.table(rows, schema=SCHEMA), out_path, compression="zstd")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default="/mnt/f/Confli_2/Corpus")
    ap.add_argument("--out", default="/mnt/f/Confli_2/Corpus/conflibert-v2/outputs/corpus")
    ap.add_argument("--eval-year", type=int, default=2021)
    ap.add_argument("--eval-frac", type=float, default=0.003)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="process only first N shards (smoke test)")
    args = ap.parse_args()

    pattern = os.path.join(args.corpus_root, "1945-2021 json_files_with_metadata", "**", "*.json.tar.gz")
    shards = sorted(glob.glob(pattern, recursive=True))
    if args.limit:
        shards = shards[:args.limit]
    out_dir = os.path.join(args.out, "parquet")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[extract] {len(shards)} shards | workers={args.workers} | out={out_dir}", flush=True)

    tasks = [(p, out_dir, args.eval_year, args.eval_frac) for p in shards]
    agg = {"n": 0, "train": 0, "eval_temporal": 0, "eval_random": 0, "empty": 0, "chars": 0}
    per_source = {}
    errors, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_shard, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                errors.append(("<worker>", str(e))); done += 1; continue
            done += 1
            if r.get("error"):
                errors.append((r["shard"], r["error"]))
            if not r.get("skipped"):
                for k in agg:
                    agg[k] += r.get(k, 0)
                s = per_source.setdefault(r["source"], {"n": 0, "chars": 0})
                s["n"] += r.get("n", 0); s["chars"] += r.get("chars", 0)
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} shards | {agg['n']:,} articles | "
                      f"{agg['chars']/1e9:.2f}G chars | {el:.0f}s | {done/max(el,1):.1f} shards/s",
                      flush=True)

    print("\n=== per-source ===")
    for src, s in sorted(per_source.items()):
        print(f"  {src:14s} {s['n']:>10,} articles  {s['chars']/1e9:6.2f}G chars")
    print("\n=== splits ===")
    for k in ("train", "eval_temporal", "eval_random"):
        print(f"  {k:14s} {agg[k]:>10,}")
    print(f"  empty/dropped  {agg['empty']:>10,}")
    print(f"\ntotal articles: {agg['n']:,} | total chars: {agg['chars']/1e9:.2f}G")
    if errors:
        print(f"\n{len(errors)} shard errors (first 10): {errors[:10]}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    main()
