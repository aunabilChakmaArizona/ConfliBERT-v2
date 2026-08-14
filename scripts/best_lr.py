#!/usr/bin/env python
"""Print the best-dev learning rate recorded for (task, model) in corrected_summary.csv."""
import csv, sys

task, model = sys.argv[1], sys.argv[2]
for r in csv.DictReader(open("analysis/data/corrected_summary.csv")):
    if r["task"] == task and r["model"] == model:
        print(r["best_lr"])
        break
else:
    print("3e-05")   # fallback: old default
