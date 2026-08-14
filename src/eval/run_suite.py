#!/usr/bin/env python
"""
Orchestrate the downstream benchmark: loop tasks x models x seeds by subprocess
(each run isolated for clean CUDA memory). Avoids inline shell loops.

Examples:
  # full suite, all models, config-default seeds
  python run_suite.py --models ConfliBERT=eventdata-utd/ConfliBERT-scr-uncased \
      ModernBERT-base=answerdotai/ModernBERT-base ConfliBERT-v2=outputs/models/conflibert-v2 \
      --tasks all --out analysis/data/downstream_results.csv

  # truncation-gap sweep on the long-doc tasks only
  python run_suite.py --models ConfliBERT-v2=outputs/models/conflibert-v2 \
      --tasks IndiaPoliceEvents_docs insightCrime --seq-lens 512 1024 2048 \
      --out analysis/data/truncation_downstream.csv --seeds 123 124
"""
from __future__ import annotations
import argparse, os, subprocess, sys

ALL_TASKS = ["satp_relevant", "IndiaPoliceEvents_sents", "IndiaPoliceEvents_docs",
             "insightCrime", "20news", "BBC_News", "cameo_class", "re3d", "cameo_ner"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[0], help="0 = task-config default")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--script", default="src/eval/finetune_benchmark.py")
    args = ap.parse_args()

    tasks = ALL_TASKS if args.tasks == ["all"] else args.tasks
    total = len(tasks) * len(args.models) * len(args.seq_lens)
    i = 0
    for task in tasks:
        for spec in args.models:
            name, path = spec.split("=", 1)
            for sl in args.seq_lens:
                i += 1
                cmd = [sys.executable, args.script, "--task", task, "--model", path,
                       "--model-name", name, "--repo", args.repo, "--out", args.out]
                if args.seeds:
                    cmd += ["--seeds"] + [str(s) for s in args.seeds]
                if sl:
                    cmd += ["--seq-len", str(sl)]
                if args.epochs:
                    cmd += ["--epochs", str(args.epochs)]
                tag = f"[{i}/{total}] {task} | {name} | seq_len={sl or 'default'}"
                print("\n" + "=" * 80 + f"\n{tag}\n" + "=" * 80, flush=True)
                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    print(f"!! FAILED ({rc}): {tag}", flush=True)
    print("\nSUITE_DONE")


if __name__ == "__main__":
    main()
