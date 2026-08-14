#!/usr/bin/env python
"""
E2 (descriptive half): where does the evidence sit inside IndiaPoliceEvents articles?

For every document, tokenize sentence-by-sentence with a given tokenizer and record,
for each event label, the token offset at which the FIRST and LAST evidence sentence
(sentence carrying that label) begins. A 512-capped encoder cannot see evidence whose
first token lies beyond its window; if ALL evidence for a label starts past the cap,
the document is invisible to the model regardless of fine-tuning quality.

Output: analysis/data/ipe_evidence_position.csv, one row per (doc, label) with
positions, plus a printed summary of the share of gold-positive docs whose evidence
is partially / entirely beyond 512 and 1024 tokens.
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

LABELS = ["KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="external/IndiaPoliceEvents/data/final")
    ap.add_argument("--tokenizer", default="eventdata-utd/ConfliBERT-scr-uncased")
    ap.add_argument("--tok-name", default="ConfliBERT-2021")
    ap.add_argument("--out", default="analysis/data/ipe_evidence_position.csv")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sents = defaultdict(list)
    with open(os.path.join(args.corpus, "sents.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            labs = s["sent_labels"]
            if not isinstance(labs, list):
                labs = json.loads(labs.replace("'", '"'))
            sents[s["doc_id"]].append((int(s["sent_id"]), s["sent_text"], labs))

    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for doc_id, ss in sents.items():
        ss.sort(key=lambda t: t[0])
        # sentence token lengths without special tokens; +2 once for [CLS]/[SEP]
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for _, t, _ in ss]
        starts = np.concatenate([[1], 1 + np.cumsum(lens[:-1])])  # offset 1 for [CLS]
        n_tok = int(1 + sum(lens) + 1)
        for L in LABELS:
            ev = [int(st) for st, (_, _, labs) in zip(starts, ss) if L in labs]
            if not ev:
                continue
            rows.append([doc_id, args.tok_name, L, n_tok, len(ev),
                         min(ev), max(ev)])
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["doc_id", "tokenizer", "label", "doc_tokens",
                        "n_evidence_sents", "first_evidence_tok", "last_evidence_tok"])
        w.writerows(rows)

    for cap in (512, 1024):
        print(f"--- cap {cap} ({args.tok_name}) ---")
        for L in LABELS:
            rl = [r for r in rows if r[2] == L]
            n = len(rl)
            all_beyond = sum(1 for r in rl if r[5] >= cap)
            some_beyond = sum(1 for r in rl if r[6] >= cap)
            print(f"{L:11s} pos_docs={n:4d}  all_evidence_beyond={all_beyond:3d} "
                  f"({100*all_beyond/n:.1f}%)  some_evidence_beyond={some_beyond:3d} "
                  f"({100*some_beyond/n:.1f}%)")


if __name__ == "__main__":
    main()
