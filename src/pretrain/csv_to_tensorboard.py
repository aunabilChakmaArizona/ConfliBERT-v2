#!/usr/bin/env python
"""Replay a run's metrics.csv into a TensorBoard event dir (so CSV-only runs show up too).

Usage:
  python csv_to_tensorboard.py --csv outputs/models/.../metrics.csv --logdir outputs/tb/<run-name>
"""
from __future__ import annotations
import argparse, csv, os
from torch.utils.tensorboard import SummaryWriter

SCALARS = {  # csv column -> tensorboard tag
    "train_loss": "train/loss", "eval_loss": "eval/loss", "eval_ppl": "eval/pseudo_ppl",
    "lr": "train/lr", "grad_norm": "train/grad_norm", "tokens_per_sec": "perf/tokens_per_sec",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--logdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    w = SummaryWriter(log_dir=args.logdir)
    n = 0
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(row["step"])
            except (ValueError, KeyError):
                continue
            for col, tag in SCALARS.items():
                v = row.get(col, "")
                if v not in ("", None):
                    try:
                        w.add_scalar(tag, float(v), step)
                    except ValueError:
                        pass
            n += 1
    w.close()
    print(f"replayed {n} rows -> {args.logdir}")


if __name__ == "__main__":
    main()
