#!/usr/bin/env python
"""
Cross-fitted event coding for ConfliBERT-format TSV multilabel tasks (paper
experiment E1b: generalization corpus, e.g. insightCrime).

Same design as ipe_crossfit.py: concatenate the task's train/dev/test splits into
the full labeled corpus, partition into 5 deterministic folds (identical across
models and seeds), fine-tune on 4 folds, predict the held-out fold, and save
per-document label probabilities. Documents are identified as <split>:<row> since
these TSVs carry no IDs or dates.

Usage (from repo root, WSL venv):
  python src/eval/tsv_crossfit.py --task insightCrime \
    --model outputs/models/conflibert-v2-wsd --model-name ConfliBERT-v2 \
    --seq-len 2048 --lr 5e-5 --out analysis/data/ic_crossfit_preds.csv
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_benchmark import ClsDS, ClsCollator

N_FOLDS = 5


def load_tsv_corpus(repo, task):
    import csv as _csv
    _csv.field_size_limit(2**31 - 1)
    ids, texts, labels = [], [], []
    for split in ("train", "dev", "test"):
        path = os.path.join(repo, "data", task, f"{split}.tsv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for row_i, row in enumerate(_csv.reader(fh, delimiter="\t", quotechar='"')):
                if not row:
                    continue
                ids.append(f"{split}:{row_i:04d}")
                texts.append(row[0])
                labels.append([int(x) for x in row[1:]])
    y = np.array(labels, dtype=np.int64)
    return ids, texts, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--tmp", default=os.environ.get("CB2_TMP", "/root/cb2_ft_tmp"),
                    help="fine-tune scratch dir (set CB2_TMP on HPC, e.g. node-local storage)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, texts, y = load_tsv_corpus(args.repo, args.task)
    n_labels = y.shape[1]
    perm = np.random.RandomState(42).permutation(len(ids))
    fold = np.empty(len(ids), dtype=int)
    fold[perm] = np.arange(len(ids)) % N_FOLDS
    print(f"{args.task}: {len(ids)} docs, {n_labels} labels, "
          f"prevalence {y.sum(0).tolist()}")

    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    enc_all = tok(texts, truncation=True, max_length=args.seq_len, padding=False)
    eval_bsz = max(4, min(32, (32 * 512) // max(args.seq_len, 512)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = (["task", "model", "seq_len", "lr", "seed", "fold", "doc_id"]
              + [f"gold_{j}" for j in range(n_labels)]
              + [f"prob_{j}" for j in range(n_labels)])
    new = not os.path.exists(args.out)

    for seed in args.seeds:
        for k in range(N_FOLDS):
            tr_idx = np.where(fold != k)[0]
            te_idx = np.where(fold == k)[0]
            torch.manual_seed(seed); np.random.seed(seed)

            def subset(idx):
                enc = {key: [enc_all[key][i] for i in idx] for key in enc_all}
                return ClsDS(enc, [y[i].tolist() for i in idx], multilabel=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=n_labels,
                problem_type="multi_label_classification")
            out_dir = os.path.join(args.tmp, f"tsv_cf_{args.task}_{args.model_name}_{seed}_{k}")
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
                    w.writerow([args.task, args.model_name, args.seq_len, args.lr,
                                seed, k, ids[i]]
                               + y[i].tolist()
                               + [round(float(p), 5) for p in probs[row_i]])
            print(f"[{args.task}|{args.model_name}|len{args.seq_len}|seed{seed}|"
                  f"fold{k}] n_test={len(te_idx)} done", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            del model, trainer
            torch.cuda.empty_cache()
    print("TSV_CROSSFIT_DONE")


if __name__ == "__main__":
    main()
