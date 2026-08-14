#!/usr/bin/env python
"""Aggregate per-source token contributions from a pack_tokens log."""
import re, sys, collections

path = sys.argv[1]
tot = collections.defaultdict(float)
pat = re.compile(r"^\s*(News|Organization|UTDstory|Gigaword|Wikipedia)\s+\S+\s+"
                 r"blocks=([\d,]+) tokens=([\d.]+)M")
for line in open(path, errors="replace"):
    m = pat.match(line)
    if m:
        tot[m.group(1)] += float(m.group(3))
grand = sum(tot.values())
for s, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"{s:<14}{v/1000:8.2f}B  {100*v/grand:5.1f}%")
print(f"{'TOTAL':<14}{grand/1000:8.2f}B")
