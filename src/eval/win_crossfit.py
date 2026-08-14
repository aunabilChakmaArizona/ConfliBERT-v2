#!/usr/bin/env python
"""
Cross-fitted coding with the practitioner's 512-token workaround: sliding-window
("chunking") inference for a capped encoder (paper experiment: the v1-at-its-best
cell). Training is identical to the truncated 512 cells; at inference the FULL
document is split into overlapping 512-token windows, each window is scored, and
the document probability per label is the max over windows. Folds replicate
ipe_crossfit.py / tsv_crossfit.py exactly, so all cells are paired per document.

Usage (from repo root, WSL venv):
  python src/eval/win_crossfit.py --corpus-type ipe \
    --model eventdata-utd/ConfliBERT-scr-uncased --model-name ConfliBERT-2021-win \
    --lr 5e-5 --out analysis/data/ipe_crossfit_preds.csv
  python src/eval/win_crossfit.py --corpus-type tsv --task insightCrime \
    --model eventdata-utd/ConfliBERT-scr-uncased --model-name ConfliBERT-2021-win \
    --lr 8e-5 --bsz 8 --out analysis/data/ic_crossfit_preds.csv
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_benchmark import ClsDS, ClsCollator

WINDOW = 512
OVERLAP = 128        # HF `stride`: windows advance by WINDOW - OVERLAP tokens
N_FOLDS = 5


def window_probs(model, tok, texts, n_labels, batch=32):
    """Max-pooled per-label probabilities over overlapping 512-token windows."""
    out = np.zeros((len(texts), n_labels))
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tok(text, truncation=True, max_length=WINDOW, stride=OVERLAP,
                      return_overflowing_tokens=True, padding=True,
                      return_tensors="pt")
            probs = []
            for j in range(0, enc["input_ids"].shape[0], batch):
                logits = model(
                    input_ids=enc["input_ids"][j:j+batch].cuda(),
                    attention_mask=enc["attention_mask"][j:j+batch].cuda()).logits
                probs.append(torch.sigmoid(logits).cpu().numpy())
            out[i] = np.concatenate(probs).max(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-type", choices=["ipe", "tsv"], required=True)
    ap.add_argument("--corpus", default="external/IndiaPoliceEvents/data/final")
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--task", default="insightCrime")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--tmp", default=os.environ.get("CB2_TMP", "/root/cb2_ft_tmp"),
                    help="fine-tune scratch dir (set CB2_TMP on HPC, e.g. node-local storage)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.corpus_type == "ipe":
        from ipe_crossfit import load_corpus, fold_of, LABELS
        ids, texts, y, dates = load_corpus(args.corpus)
        fold = fold_of(ids)
        gold_cols = [f"gold_{L}" for L in LABELS]
        prob_cols = [f"prob_{L}" for L in LABELS]
        header = (["model", "seq_len", "lr", "seed", "fold", "doc_id", "date"]
                  + gold_cols + prob_cols)
        def row_meta(i):
            return [ids[i], dates[i]]
        task_tag = None
    else:
        from tsv_crossfit import load_tsv_corpus
        ids, texts, y = load_tsv_corpus(args.repo, args.task)
        perm = np.random.RandomState(42).permutation(len(ids))
        fold = np.empty(len(ids), dtype=int)
        fold[perm] = np.arange(len(ids)) % N_FOLDS
        n = y.shape[1]
        header = (["task", "model", "seq_len", "lr", "seed", "fold", "doc_id"]
                  + [f"gold_{j}" for j in range(n)] + [f"prob_{j}" for j in range(n)])
        def row_meta(i):
            return [ids[i]]
        task_tag = args.task
    n_labels = y.shape[1]

    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    enc_all = tok(texts, truncation=True, max_length=WINDOW, padding=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)

    for seed in args.seeds:
        for k in range(N_FOLDS):
            tr_idx = np.where(fold != k)[0]
            te_idx = np.where(fold == k)[0]
            torch.manual_seed(seed); np.random.seed(seed)

            enc = {key: [enc_all[key][i] for i in tr_idx] for key in enc_all}
            train_ds = ClsDS(enc, [y[i].tolist() for i in tr_idx], multilabel=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=n_labels,
                problem_type="multi_label_classification")
            out_dir = os.path.join(args.tmp, f"win_cf_{args.model_name}_{seed}_{k}")
            targs = TrainingArguments(
                output_dir=out_dir, per_device_train_batch_size=args.bsz,
                num_train_epochs=args.epochs, learning_rate=args.lr,
                warmup_ratio=0.06, weight_decay=0.01, bf16=True,
                logging_steps=200, report_to="none", seed=seed,
                dataloader_num_workers=2, save_strategy="no", eval_strategy="no")
            trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                              data_collator=ClsCollator(tok, multilabel=True))
            trainer.train()

            probs = window_probs(model.cuda(), tok, [texts[i] for i in te_idx],
                                 n_labels)
            with open(args.out, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(header); new = False
                for row_i, i in enumerate(te_idx):
                    base = ([task_tag] if task_tag else []) + \
                        [args.model_name, f"win{WINDOW}", args.lr, seed, k] + row_meta(i)
                    w.writerow(base + y[i].tolist()
                               + [round(float(p), 5) for p in probs[row_i]])
            print(f"[{args.model_name}|win{WINDOW}|seed{seed}|fold{k}] "
                  f"n_test={len(te_idx)} done", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            del model, trainer
            torch.cuda.empty_cache()
    print("WIN_CROSSFIT_DONE")


if __name__ == "__main__":
    main()
