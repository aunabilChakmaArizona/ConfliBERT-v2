#!/usr/bin/env python
"""Export per-LR seed-123 rows (test + dev primary) from corrected_bench.csv."""
import csv, json

csv.field_size_limit(2**31 - 1)
rows = list(csv.DictReader(open("analysis/data/corrected_bench.csv", encoding="utf-8")))
with open("analysis/data/lr_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "model", "lr", "test_primary", "dev_primary"])
    for r in rows:
        if int(r["seed"]) != 123:
            continue
        mj = json.loads(r["metrics_json"])
        w.writerow([r["task"], r["model"], mj.get("lr"),
                    round(float(r["primary"]), 5), mj.get("dev_primary")])
print("wrote analysis/data/lr_curves.csv")
