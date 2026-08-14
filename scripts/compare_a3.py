#!/usr/bin/env python
"""Compare A3 (WSD) downstream vs A2 (native cosine) and base, old protocol."""
import csv, collections, statistics as st

def means(fn):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(fn)):
        by[(r["task"], r["model"], r["seq_len"])].append(float(r["primary"]))
    return {k: st.mean(v) for k, v in by.items()}

new = means("analysis/data/downstream_wsd.csv")
old = means("analysis/data/downstream_results.csv")
nat = means("analysis/data/downstream_native.csv")
tasks = ["satp_relevant", "IndiaPoliceEvents_sents", "IndiaPoliceEvents_docs",
         "insightCrime", "cameo_class", "BBC_News", "re3d"]
print(f"{'task':<26}{'base':>7}{'A2nat':>7}{'A3wsd':>7}{'A3-A2':>8}{'A3-base':>8}")
s = [0.0, 0.0, 0.0]
for t in tasks:
    b = next((100*v for (tt, m, sl), v in old.items() if tt == t and m == "ModernBERT-base"), 0)
    a2 = next((100*v for (tt, m, sl), v in nat.items() if tt == t), 0)
    a3 = next((100*v for (tt, m, sl), v in new.items() if tt == t), 0)
    s[0] += b; s[1] += a2; s[2] += a3
    print(f"{t:<26}{b:>7.2f}{a2:>7.2f}{a3:>7.2f}{a3-a2:>+8.2f}{a3-b:>+8.2f}")
print(f"{'MEAN':<26}{s[0]/7:>7.2f}{s[1]/7:>7.2f}{s[2]/7:>7.2f}{(s[2]-s[1])/7:>+8.2f}{(s[2]-s[0])/7:>+8.2f}")
print()
print("truncation sweep A3 (task, seqlen, mean):")
tr = means("analysis/data/truncation_wsd.csv")
for (t, m, sl), v in sorted(tr.items(), key=lambda x: (x[0][0], int(x[0][2]))):
    print(f"  {t:<24}{sl:>5}  {100*v:.2f}")
