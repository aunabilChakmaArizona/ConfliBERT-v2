#!/usr/bin/env python
"""
Master analysis for the IndiaPoliceEvents cross-fit experiments (E1/E2/E3).

Reads analysis/data/ipe_crossfit_preds.csv (+ ipe_llm_preds.csv and
ipe_evidence_position.csv) and writes:
  analysis/data/ipe_summary.csv        per config x label: recall/precision/counts
  analysis/data/ipe_strata.csv         recall by evidence-position stratum (E2)
  analysis/data/ipe_variance.csv       count spread: across-seed vs across-config (E3)
Run from repo root:  python analysis/ipe_analysis.py
"""
import csv
from collections import defaultdict
from statistics import mean, stdev

LAB = ["KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION"]
THR = 0.5


def load_encoder_rows():
    rows = list(csv.DictReader(open("analysis/data/ipe_crossfit_preds.csv")))
    for r in rows:
        r["config"] = f"{r['model']}@{r['seq_len']}"
    return rows


def load_llm_rows():
    rows = [r for r in csv.DictReader(open("analysis/data/ipe_llm_preds.csv"))
            if r["parse_error"] == "0"]
    for r in rows:
        r["config"] = "GPT-5.6"
        r["seed"] = "0"
        for L in LAB:
            r[f"prob_{L}"] = r[f"pred_{L}"]
    return rows


def pred(r, L):
    return float(r[f"prob_{L}"]) > THR


def summarize(rows_by_cfg_seed):
    out = []
    for (cfg, seed), rs in sorted(rows_by_cfg_seed.items()):
        for L in LAB:
            tp = sum(1 for r in rs if r[f"gold_{L}"] == "1" and pred(r, L))
            fp = sum(1 for r in rs if r[f"gold_{L}"] == "0" and pred(r, L))
            fn = sum(1 for r in rs if r[f"gold_{L}"] == "1" and not pred(r, L))
            out.append(dict(config=cfg, seed=seed, label=L, gold=tp + fn,
                            pred_count=tp + fp,
                            recall=tp / (tp + fn) if tp + fn else 0,
                            precision=tp / (tp + fp) if tp + fp else 0))
    return out


def agg(vals):
    return (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)


def main():
    enc = load_encoder_rows()
    llm = load_llm_rows()
    allr = enc + llm

    by_cs = defaultdict(list)
    for r in allr:
        by_cs[(r["config"], r["seed"])].append(r)
    per_seed = summarize(by_cs)

    # ---- E1 summary: aggregate over seeds ----
    by_cl = defaultdict(list)
    for s in per_seed:
        by_cl[(s["config"], s["label"])].append(s)
    with open("analysis/data/ipe_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "label", "gold", "pred_count_mean", "pred_count_sd",
                    "recall_mean", "recall_sd", "precision_mean", "precision_sd"])
        print(f"{'config':22s} {'label':11s} {'gold':>4s} {'count':>12s} "
              f"{'recall':>13s} {'precision':>13s}")
        for (cfg, L), ss in sorted(by_cl.items()):
            cm, csd = agg([s["pred_count"] for s in ss])
            rm, rsd = agg([s["recall"] for s in ss])
            pm, psd = agg([s["precision"] for s in ss])
            w.writerow([cfg, L, ss[0]["gold"], round(cm, 1), round(csd, 1),
                        round(rm, 4), round(rsd, 4), round(pm, 4), round(psd, 4)])
            print(f"{cfg:22s} {L:11s} {ss[0]['gold']:4d} {cm:6.1f}±{csd:4.1f} "
                  f"{rm:.3f}±{rsd:.3f}  {pm:.3f}±{psd:.3f}")

    # ---- E2: recall by evidence-position stratum (512-capped instruments) ----
    ev = defaultdict(dict)   # doc_id -> label -> first_evidence_tok
    for r in csv.DictReader(open("analysis/data/ipe_evidence_position.csv")):
        ev[r["doc_id"]][r["label"]] = int(r["first_evidence_tok"])
    with open("analysis/data/ipe_strata.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "stratum", "n_doc_labels", "recall_mean", "recall_sd"])
        print("\nE2: pooled recall by evidence position (all labels pooled)")
        for cfg in sorted({r["config"] for r in allr}):
            seeds = sorted({r["seed"] for r in allr if r["config"] == cfg})
            for strat, cond in [("evidence_within_512", lambda t: t < 512),
                                ("evidence_beyond_512", lambda t: t >= 512)]:
                recs, n_pairs = [], 0
                for seed in seeds:
                    rs = by_cs[(cfg, seed)]
                    tp = fn = 0
                    for r in rs:
                        for L in LAB:
                            if r[f"gold_{L}"] != "1":
                                continue
                            t = ev.get(r["doc_id"], {}).get(L)
                            if t is None or not cond(t):
                                continue
                            if pred(r, L):
                                tp += 1
                            else:
                                fn += 1
                    if tp + fn:
                        recs.append(tp / (tp + fn))
                        n_pairs = tp + fn
                if recs:
                    rm, rsd = agg(recs)
                    w.writerow([cfg, strat, n_pairs, round(rm, 4), round(rsd, 4)])
                    print(f"  {cfg:22s} {strat:22s} n={n_pairs:4d} "
                          f"recall={rm:.3f}±{rsd:.3f}")

    # ---- E3: count spread, seeds vs configs (encoders only) ----
    with open("analysis/data/ipe_variance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "config", "count_range_across_seeds",
                    "count_range_across_configs"])
        print("\nE3: total-count spread")
        for L in LAB:
            cfg_means = {}
            for cfg in sorted({s["config"] for s in per_seed if s["config"] != "GPT-5.6"}):
                counts = [s["pred_count"] for s in per_seed
                          if s["config"] == cfg and s["label"] == L]
                cfg_means[cfg] = mean(counts)
                w.writerow([L, cfg, max(counts) - min(counts), ""])
            spread = max(cfg_means.values()) - min(cfg_means.values())
            w.writerow([L, "ALL_CONFIGS", "", round(spread, 1)])
            seed_ranges = [max(c) - min(c) for c in
                           [[s["pred_count"] for s in per_seed
                             if s["config"] == cfg and s["label"] == L]
                            for cfg in cfg_means]]
            print(f"  {L:11s} across-configs spread={spread:6.1f}  "
                  f"across-seeds range per config: {seed_ranges}")


if __name__ == "__main__":
    main()
