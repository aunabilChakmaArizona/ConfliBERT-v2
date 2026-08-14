#!/usr/bin/env python
"""
Build the CrisisWatch trend-assessment task from parsed entries.

Each example: a country-month entry plus (when available) the same country's
previous entries as trailing context; the label is Crisis Group's own verdict
for the month (unchanged / deteriorated / improved). The verdict is a judgment
over the whole narrative, month-over-month, which is what makes this the test
where window-pooling has no principled answer.

Temporal split (no leakage, post-cutoff LLM test):
  train: through 2024-12    dev: 2025-01..2025-05    test: 2025-06 onward

Output: external/crisiswatch/task.jsonl
Usage:  python src/data/crisiswatch_task.py
"""
from __future__ import annotations
import collections, json

MAX_HISTORY = 3           # trailing entries used as context
MAX_GAP_MONTHS = 4        # a prior entry older than this is not "last month" context
CLASSES = ["unchanged", "deteriorated", "improved"]


def ym_index(y, m):
    return y * 12 + (m - 1)


def split_of(y, m):
    if (y, m) <= (2024, 12):
        return "train"
    if (y, m) <= (2025, 5):
        return "dev"
    return "test"


def main():
    entries = [json.loads(l) for l in
               open("external/crisiswatch/entries.jsonl", encoding="utf-8")]
    entries = [e for e in entries if e["status"] in CLASSES and len(e["text"]) > 200]

    by_country = collections.defaultdict(list)
    for e in entries:
        by_country[e["country"]].append(e)
    for v in by_country.values():
        v.sort(key=lambda e: ym_index(e["year"], e["month"]))

    n_hist = collections.Counter()
    out = []
    for country, es in by_country.items():
        for i, e in enumerate(es):
            hist = []
            for prev in reversed(es[:i]):
                gap = ym_index(e["year"], e["month"]) - ym_index(prev["year"], prev["month"])
                if gap > MAX_GAP_MONTHS or len(hist) >= MAX_HISTORY:
                    break
                hist.append(f"[{prev['year']}-{prev['month']:02d}] {prev['text']}")
            hist.reverse()
            n_hist[len(hist)] += 1
            out.append({
                "country": country, "year": e["year"], "month": e["month"],
                "region": e["region"], "label": e["status"],
                "split": split_of(e["year"], e["month"]),
                "n_history": len(hist),
                "text_current": f"[{e['year']}-{e['month']:02d}] {e['text']}",
                "text_history": "\n".join(hist),
            })

    with open("external/crisiswatch/task.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"examples: {len(out)}  history depth: {dict(sorted(n_hist.items()))}")
    for sp in ("train", "dev", "test"):
        rows = [r for r in out if r["split"] == sp]
        bal = collections.Counter(r["label"] for r in rows)
        print(f"  {sp:5s} n={len(rows):5d}  {dict(bal)}")
    words = [len((r["text_history"] + " " + r["text_current"]).split()) for r in out]
    words.sort()
    print(f"full-input words: median={words[len(words)//2]} "
          f"p90={words[int(len(words)*0.9)]}")


if __name__ == "__main__":
    main()
