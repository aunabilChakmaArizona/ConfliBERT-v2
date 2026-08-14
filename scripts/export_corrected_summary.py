#!/usr/bin/env python
"""Export tidy per-(task,model) summary for R plotting.

Base numbers come from corrected_bench.csv. For the two NER tasks, rows from
ner_psfix.csv SUPERSEDE the bench rows for ModernBERT-family models: the bench
NER runs fed pre-split words to the ByteLevel BPE without the space prefix
(out-of-distribution vs pretraining) and capped the LR grid at 8e-5; the psfix
protocol fixes both. Models there carry a -psfix suffix, stripped here so each
model keeps one identity across figures. ConfliBERT-2021 (WordPiece, unaffected
by the bug, interior LR peak) keeps its bench rows.
"""
import csv, json, statistics as st, collections

csv.field_size_limit(2**31 - 1)
TASKS = ["re3d", "IndiaPoliceEvents_docs", "insightCrime", "satp_relevant",
         "cameo_class", "cameo_ner", "IndiaPoliceEvents_sents", "20news", "BBC_News"]
NER_TASKS = {"re3d", "cameo_ner"}


def summarize(rows):
    """dev-LR-selection summary: {(task, model): (best_lr, n, mean, sd)}"""
    by = collections.defaultdict(list)
    for r in rows:
        mj = json.loads(r["metrics_json"])
        by[(r["task"], r["model"])].append(
            (int(r["seed"]), mj.get("lr"), mj.get("dev_primary"), float(r["primary"])))
    out = {}
    for (t, m), rs in sorted(by.items()):
        sel = [r for r in rs if r[0] == 123 and r[2] is not None]
        if not sel:
            continue
        best = max(sel, key=lambda x: x[2])[1]
        finals = [r[3] for r in rs if r[1] == best]
        sd = st.stdev(finals) if len(finals) > 1 else 0.0
        out[(t, m)] = (best, len(finals), round(st.mean(finals), 5), round(sd, 5))
    return out


summ = summarize(list(csv.DictReader(open("analysis/data/corrected_bench.csv",
                                          encoding="utf-8"))))
try:
    psfix = summarize(list(csv.DictReader(open("analysis/data/ner_psfix.csv",
                                               encoding="utf-8"))))
except FileNotFoundError:
    psfix = {}
n_over = 0
for (t, m), v in psfix.items():
    if t in NER_TASKS and m.endswith("-psfix"):
        summ[(t, m[: -len("-psfix")])] = v
        n_over += 1

with open("analysis/data/corrected_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "model", "best_lr", "n_seeds", "mean_primary", "sd_primary"])
    for (t, m), (best, n, mean, sd) in sorted(summ.items()):
        w.writerow([t, m, best, n, mean, sd])
print(f"wrote analysis/data/corrected_summary.csv ({n_over} NER rows superseded by psfix)")
