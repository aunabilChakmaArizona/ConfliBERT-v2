#!/usr/bin/env python
"""
Corrected-protocol benchmark driver.

For each (task, model): fine-tune at every LR in the grid with --dev-select
(per-epoch dev eval, best-primary epoch) on the selection seed, pick the LR with
the best dev primary, then run the remaining seeds at that LR. All rows land in
one CSV; analysis filters to best-dev-LR rows. Resumable: re-invoking skips
(seed, lr) combinations already present in the CSV.

The uniform LR grid {2e-5, 3e-5, 5e-5, 8e-5} spans both BERT-era optima (2-3e-5)
and ModernBERT optima (5-8e-5), so no model family is handicapped.
"""
import argparse, csv, json, os, subprocess, sys

csv.field_size_limit(2**31 - 1)


def rows_for(out, task, name):
    if not os.path.exists(out):
        return []
    with open(out, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["task"] == task and r["model"] == name]


def have_set(rows):
    out = set()
    for r in rows:
        try:
            out.add((int(r["seed"]), float(json.loads(r["metrics_json"])["lr"])))
        except Exception:
            pass
    return out


def run(task, path, name, lr, seeds, args):
    cmd = [sys.executable, "src/eval/finetune_benchmark.py", "--task", task,
           "--model", path, "--model-name", name, "--repo", args.repo,
           "--out", args.out, "--lr", str(lr), "--dev-select",
           "--seeds"] + [str(s) for s in seeds]
    if args.prefix_space:
        cmd.append("--prefix-space")
    print(f"[driver] {task} | {name} | lr={lr} | seeds={seeds}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[driver][FAIL] {task} {name} lr={lr} rc={rc}", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5, 3e-5, 5e-5, 8e-5])
    ap.add_argument("--select-seed", type=int, default=123)
    ap.add_argument("--final-seeds", type=int, nargs="+", default=[124, 125])
    ap.add_argument("--out", default="analysis/data/corrected_bench.csv")
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--prefix-space", action="store_true",
                    help="pass --prefix-space to the harness (ByteLevel BPE models only)")
    args = ap.parse_args()
    models = [m.split("=", 1) for m in args.models]

    for task in args.tasks:                      # task-major: comparisons land task by task
        for name, path in models:
            have = have_set(rows_for(args.out, task, name))
            for lr in args.lrs:                  # stage 1: LR selection on dev
                if (args.select_seed, lr) not in have:
                    run(task, path, name, lr, [args.select_seed], args)
            sel, best_lr, best_dev = rows_for(args.out, task, name), None, -1.0
            for r in sel:
                if int(r["seed"]) != args.select_seed:
                    continue
                mj = json.loads(r["metrics_json"])
                dp = mj.get("dev_primary")
                if dp is not None and float(dp) > best_dev:
                    best_dev, best_lr = float(dp), float(mj["lr"])
            if best_lr is None:
                print(f"[driver][SKIP] no dev_primary rows for {task}/{name}", flush=True)
                continue
            print(f"[driver] {task}/{name}: best lr={best_lr} (dev={best_dev:.4f})", flush=True)
            have = have_set(rows_for(args.out, task, name))
            need = [s for s in args.final_seeds if (s, best_lr) not in have]
            if need:                             # stage 2: remaining seeds at best-dev LR
                run(task, path, name, best_lr, need, args)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
