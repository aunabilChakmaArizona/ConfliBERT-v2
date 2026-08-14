#!/usr/bin/env python
"""Summarize corrected_bench.csv: per (task, model) pick best-dev LR (selection
seed 123), report mean test primary over all seeds run at that LR."""
import csv, json, statistics as st, collections, sys

csv.field_size_limit(2**31 - 1)
PATH = sys.argv[1] if len(sys.argv) > 1 else "analysis/data/corrected_bench.csv"
TASKS = ["re3d", "IndiaPoliceEvents_docs", "insightCrime", "satp_relevant",
         "cameo_class", "cameo_ner", "IndiaPoliceEvents_sents", "20news", "BBC_News"]

rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
by = collections.defaultdict(list)
models = []
for r in rows:
    mj = json.loads(r["metrics_json"])
    by[(r["task"], r["model"])].append(
        (int(r["seed"]), mj.get("lr"), mj.get("dev_primary"), float(r["primary"])))
    if r["model"] not in models:
        models.append(r["model"])

summary = {}
print(f"{'task':<24}{'model':<24}{'bestLR':>8}{'n':>3}{'test mean':>10}{'sd':>6}")
for t in TASKS:
    for m in models:
        rs = by.get((t, m))
        if not rs:
            continue
        sel = [r for r in rs if r[0] == 123 and r[2] is not None]
        if not sel:
            continue
        best = max(sel, key=lambda x: x[2])[1]
        finals = [r[3] for r in rs if r[1] == best]
        mu = st.mean(finals)
        sd = st.stdev(finals) if len(finals) > 1 else 0.0
        summary[(t, m)] = (mu, len(finals))
        print(f"{t:<24}{m:<24}{best:>8}{len(finals):>3}{mu*100:>10.2f}{sd*100:>6.2f}")

print()
head = "task".ljust(24) + "".join(m[:14].rjust(16) for m in models) + "   diff(last-first)"
print(head)
means = collections.defaultdict(list)
for t in TASKS:
    vals = [summary.get((t, m)) for m in models]
    if all(v is not None for v in vals):
        line = t.ljust(24) + "".join(f"{v[0]*100:16.2f}" for v in vals)
        if len(vals) >= 2:
            line += f"   {100*(vals[-1][0]-vals[0][0]):+6.2f}"
        print(line)
        for m, v in zip(models, vals):
            means[m].append(v[0])
print()
for m in models:
    if means[m]:
        print(f"MEAN over {len(means[m])} shared tasks  {m:<24}{st.mean(means[m])*100:6.2f}")
