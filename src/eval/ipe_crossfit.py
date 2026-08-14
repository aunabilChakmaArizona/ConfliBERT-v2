#!/usr/bin/env python
"""
Cross-fitted event coding of the full IndiaPoliceEvents corpus (Halterman et al. 2021).

Political-science downstream experiment E1: instead of reporting benchmark F1, code
EVERY document in the corpus out-of-sample (5-fold cross-fitting: each fold is
predicted by a model fine-tuned on the other four) and save per-document label
probabilities. Downstream analysis aggregates these to daily event counts over
March 2002 and compares each model's implied time series to the human gold standard.

Folds are deterministic (seeded permutation of sorted doc_ids) and IDENTICAL across
models and seeds, so model comparisons are paired at the document level.

Usage (from repo root, WSL venv):
  python src/eval/ipe_crossfit.py \
    --model outputs/models/conflibert-v2-wsd --model-name ConfliBERT-v2-wsd \
    --seq-len 2048 --lr 8e-5 --seeds 123 124 125 \
    --out analysis/data/ipe_crossfit_preds.csv
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_benchmark import ClsDS, ClsCollator

LABELS = ["KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION"]
N_FOLDS = 5


def load_corpus(corpus_dir):
    docs, meta = {}, {}
    with open(os.path.join(corpus_dir, "docs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    with open(os.path.join(corpus_dir, "metadata.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            meta[m["doc_id"]] = m
    ids = sorted(docs.keys(), key=int)
    texts = [docs[i]["doc_text"] for i in ids]
    raw = [docs[i]["doc_labels"] for i in ids]
    raw = [r if isinstance(r, list) else json.loads(r.replace("'", '"')) for r in raw]
    y = np.array([[int(L in r) for L in LABELS] for r in raw], dtype=np.int64)
    dates = [meta[i]["date"] for i in ids]
    return ids, texts, y, dates


def fold_of(ids):
    perm = np.random.RandomState(42).permutation(len(ids))
    fold = np.empty(len(ids), dtype=int)
    fold[perm] = np.arange(len(ids)) % N_FOLDS
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="external/IndiaPoliceEvents/data/final")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--tmp", default=os.environ.get("CB2_TMP", "/root/cb2_ft_tmp"),
                    help="fine-tune scratch dir (set CB2_TMP on HPC, e.g. node-local storage)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, texts, y, dates = load_corpus(args.corpus)
    fold = fold_of(ids)
    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = (["model", "seq_len", "lr", "seed", "fold", "doc_id", "date"]
              + [f"gold_{L}" for L in LABELS] + [f"prob_{L}" for L in LABELS])
    new = not os.path.exists(args.out)

    enc_all = tok(texts, truncation=True, max_length=args.seq_len, padding=False)
    eval_bsz = max(4, min(32, (32 * 512) // max(args.seq_len, 512)))

    for seed in args.seeds:
        for k in range(N_FOLDS):
            tr_idx = np.where(fold != k)[0]
            te_idx = np.where(fold == k)[0]
            torch.manual_seed(seed); np.random.seed(seed)

            def subset(idx):
                enc = {key: [enc_all[key][i] for i in idx] for key in enc_all}
                labels = [y[i].tolist() for i in idx]
                return ClsDS(enc, labels, multilabel=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=len(LABELS),
                problem_type="multi_label_classification")
            out_dir = os.path.join(args.tmp, f"ipe_cf_{args.model_name}_{seed}_{k}")
            targs = TrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=args.bsz,
                per_device_eval_batch_size=eval_bsz,
                num_train_epochs=args.epochs, learning_rate=args.lr,
                warmup_ratio=0.06, weight_decay=0.01, bf16=True,
                logging_steps=200, report_to="none", seed=seed,
                dataloader_num_workers=2, save_strategy="no", eval_strategy="no")
            trainer = Trainer(model=model, args=targs, train_dataset=subset(tr_idx),
                              data_collator=ClsCollator(tok, multilabel=True))
            trainer.train()
            pred = trainer.predict(subset(te_idx))
            probs = 1 / (1 + np.exp(-pred.predictions))

            with open(args.out, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(header); new = False
                for row_i, i in enumerate(te_idx):
                    w.writerow([args.model_name, args.seq_len, args.lr, seed, k,
                                ids[i], dates[i]]
                               + y[i].tolist()
                               + [round(float(p), 5) for p in probs[row_i]])
            print(f"[{args.model_name}|len{args.seq_len}|seed{seed}|fold{k}] "
                  f"n_test={len(te_idx)} done", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            del model, trainer
            torch.cuda.empty_cache()
    print("CROSSFIT_DONE")


if __name__ == "__main__":
    main()
