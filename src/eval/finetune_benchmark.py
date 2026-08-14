#!/usr/bin/env python
"""
Unified native-HF fine-tuning harness for the ConfliBERT benchmark suite.

Runs ANY encoder (ModernBERT-base, ConfliBERT-scr-uncased, ConfliBERT-v2, ablation
checkpoints) through ONE identical pipeline on the original ConfliBERT tasks,
reading their exact data formats/configs and reproducing their metrics
(example-based F1 for multilabel, seqeval for NER, acc/F1/MCC for single-label).

This controls for the fine-tuning framework so cross-model comparison is fair, and
it exposes a `--seq-len` override so the same long-document task can be run at 512
(ConfliBERT's ceiling) vs 1024/2048/4096 (ModernBERT/v2) to quantify the
truncation gap. One (task, model) per invocation; loops seeds; appends to a CSV.

Usage:
  python finetune_benchmark.py --task IndiaPoliceEvents_docs \
    --model answerdotai/ModernBERT-base --model-name ModernBERT-base \
    --repo external/ConfliBERT --out analysis/data/downstream_results.csv \
    --seeds 123 124 125 [--seq-len 1024] [--epochs 5]
"""
from __future__ import annotations
import argparse, csv, json, os, shutil
import numpy as np

csv.field_size_limit(2**31 - 1)   # 20news posts exceed the 128KB default field limit
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, DataCollatorForTokenClassification)

# ----- task registry: type + defaults; overridden by repo configs when present ----
TASK_TYPE = {
    "20news": "binary", "BBC_News": "binary",
    "IndiaPoliceEvents_docs": "multilabel", "IndiaPoliceEvents_sents": "multilabel",
    "insightCrime": "multilabel", "satp_relevant": "multilabel",
    "cameo_class": "multiclass",
    "re3d": "ner", "cameo_ner": "ner",
}
# for tasks lacking a repo config file
DEFAULT_CFG = {
    "cameo_class": {"task": "multiclass", "num_of_seeds": 3, "initial_seed": 123,
                    "epochs_per_seed": 5, "train_batch_size": 16, "max_seq_length": 256},
    "cameo_ner": {"task": "ner", "num_of_seeds": 3, "initial_seed": 123,
                  "epochs_per_seed": 5, "train_batch_size": 16, "max_seq_length": 128},
}


# ---------------- example-based multilabel metrics (from ConfliBERT example_based.py) ----
def _eb(y_true, y_pred, mode):
    yt, yp = [], []
    for t, p in zip(y_true, y_pred):
        cond = (sum(t + p) > 0) if mode in ("acc", "f1") else (sum(t) > 0 if mode == "rec" else sum(p) > 0)
        if cond:
            yt.append(t); yp.append(p)
    if not yt:
        return 0.0
    yt, yp = np.array(yt), np.array(yp)
    inter = np.sum(np.logical_and(yt, yp), axis=1)
    if mode == "acc":
        return float(np.mean(inter / np.sum(np.logical_or(yt, yp), axis=1)))
    if mode == "rec":
        return float(np.mean(inter / np.sum(yt, axis=1)))
    if mode == "prec":
        return float(np.mean(inter / np.sum(yp, axis=1)))
    if mode == "f1":
        return float(np.mean((2 * inter) / (np.sum(yp, axis=1) + np.sum(yt, axis=1))))


# ---------------- data loading ----------------
def load_classification(data_dir, ttype, carve_dev=False):
    import csv as _csv
    def rd(fn):
        texts, labels = [], []
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for row in _csv.reader(fh, delimiter="\t", quotechar='"'):
                if not row:
                    continue
                texts.append(row[0])
                if ttype == "multilabel":
                    labels.append([int(x) for x in row[1:]])
                else:
                    labels.append(int(row[1]))
        return texts, labels
    train, dev, test = rd("train.tsv"), rd("dev.tsv"), rd("test.tsv")
    if dev is None:
        if carve_dev:   # dev-select mode: carve 15% of train; NEVER select on test
            tx, lb = train
            idx = np.random.RandomState(42).permutation(len(tx))
            cut = max(1, int(0.15 * len(tx)))
            dev = ([tx[i] for i in idx[:cut]], [lb[i] for i in idx[:cut]])
            train = ([tx[i] for i in idx[cut:]], [lb[i] for i in idx[cut:]])
        else:                                    # legacy fallback (dev unused)
            dev = test
    if ttype == "multilabel":
        num_labels = len(train[1][0])
    else:
        num_labels = max(max(train[1]), max(test[1])) + 1
    return train, dev, test, num_labels


def load_ner(data_dir, carve_dev=False):
    def rd(fn):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            return None
        sents, words, labs = [], [], []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if len(line.strip()) == 0:
                    if words:
                        sents.append((words, labs)); words, labs = [], []
                else:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        parts = line.split()   # space-separated NER files (cameo_ner)
                    if len(parts) >= 2:
                        words.append(parts[0]); labs.append(parts[1])
        if words:
            sents.append((words, labs))
        return sents
    train, dev, test = rd("train.txt"), rd("dev.txt"), rd("test.txt")
    if dev is None:
        if carve_dev:   # dev-select mode: carve 15% of train; NEVER select on test
            idx = np.random.RandomState(42).permutation(len(train))
            cut = max(1, int(0.15 * len(train)))
            dev = [train[i] for i in idx[:cut]]
            train = [train[i] for i in idx[cut:]]
        else:                                    # legacy fallback (dev unused)
            dev = test
    lp = os.path.join(data_dir, "labels.json")
    if os.path.exists(lp):
        label_list = json.load(open(lp))
    else:
        label_list = sorted({l for s in train for l in s[1]})
    return train, dev, test, label_list


# ---------------- dataset builders ----------------
class ClsDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels, multilabel):
        self.enc, self.labels, self.ml = enc, labels, multilabel

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = self.labels[i]          # raw; collator tensorizes
        return item


class ClsCollator:
    """Pad input fields via tokenizer; stack labels (float for multilabel, long else)."""
    def __init__(self, tok, multilabel):
        self.tok, self.ml = tok, multilabel

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tok.pad(features, return_tensors="pt")
        batch["labels"] = torch.tensor(
            labels, dtype=torch.float if self.ml else torch.long)
        return batch


def tokenize_ner(sents, tok, label2id, max_len):
    all_words = [w for w, _ in sents]
    enc = tok(all_words, is_split_into_words=True, truncation=True, max_length=max_len,
              padding=False)
    labels = []
    for i, (_, labs) in enumerate(sents):
        wids = enc.word_ids(i)
        prev, row = None, []
        for wid in wids:
            if wid is None:
                row.append(-100)
            elif wid != prev:
                row.append(label2id.get(labs[wid], label2id.get("O", 0)))
            else:
                row.append(-100)             # only first subword carries the label
            prev = wid
        labels.append(row)
    return enc, labels


class NerDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc.keys()}
        item["labels"] = self.labels[i]
        return item


# ---------------- dev-set metric for model/LR selection ----------------
def make_compute_metrics(ttype, id2label=None):
    """Primary metric on the dev set, matching each task's test-time primary."""
    def fn(ep):
        logits, labels = ep.predictions, ep.label_ids
        if ttype == "ner":
            from seqeval.metrics import f1_score
            preds = np.argmax(logits, axis=-1)
            y_true, y_pred = [], []
            for p_row, l_row in zip(preds, labels):
                tt, tp = [], []
                for pi, li in zip(p_row, l_row):
                    if li != -100:
                        tt.append(id2label[int(li)]); tp.append(id2label[int(pi)])
                y_true.append(tt); y_pred.append(tp)
            return {"primary": float(f1_score(y_true, y_pred, average="micro"))}
        if ttype == "multilabel":
            probs = 1 / (1 + np.exp(-logits))
            yp = (probs > 0.5).astype(int).tolist()
            yt = labels.astype(int).tolist()
            return {"primary": float(_eb(yt, yp, "f1"))}
        from sklearn.metrics import f1_score
        preds = np.argmax(logits, axis=-1)
        if ttype == "binary":
            return {"primary": float(f1_score(labels, preds, average="binary", zero_division=0))}
        return {"primary": float(f1_score(labels, preds, average="macro"))}
    return fn


# ---------------- one run ----------------
def run_once(args, cfg, ttype, seed):
    data_dir = os.path.join(args.repo, "data", args.task)
    max_len = args.seq_len or cfg["max_seq_length"]
    epochs = args.epochs or cfg["epochs_per_seed"]
    bsz = args.bsz or cfg["train_batch_size"]
    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    if args.prefix_space:
        # ByteLevel BPE tokenizes pre-split words WITHOUT the leading space (out of
        # distribution vs pretraining, where words carry the G-space prefix); force it on
        from tokenizers import pre_tokenizers
        assert type(tok.backend_tokenizer.pre_tokenizer).__name__ == "ByteLevel", \
            "--prefix-space only applies to ByteLevel BPE tokenizers"
        tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True, trim_offsets=True, use_regex=True)
    torch.manual_seed(seed); np.random.seed(seed)

    dev_ds, compute_metrics, id2label = None, None, None
    if ttype == "ner":
        train, dev, test, label_list = load_ner(data_dir, carve_dev=args.dev_select)
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for l, i in label2id.items()}
        tr_enc, tr_lab = tokenize_ner(train, tok, label2id, max_len)
        te_enc, te_lab = tokenize_ner(test, tok, label2id, max_len)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id)
        train_ds, test_ds = NerDS(tr_enc, tr_lab), NerDS(te_enc, te_lab)
        collator = DataCollatorForTokenClassification(tok)
        if args.dev_select:
            dv_enc, dv_lab = tokenize_ner(dev, tok, label2id, max_len)
            dev_ds = NerDS(dv_enc, dv_lab)
    else:
        train, dev, test, num_labels = load_classification(data_dir, ttype,
                                                           carve_dev=args.dev_select)
        ml = (ttype == "multilabel")
        tr_enc = tok(train[0], truncation=True, max_length=max_len, padding=False)
        te_enc = tok(test[0], truncation=True, max_length=max_len, padding=False)
        problem = "multi_label_classification" if ml else "single_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels, problem_type=problem)
        train_ds = ClsDS(tr_enc, train[1], ml)
        test_ds = ClsDS(te_enc, test[1], ml)
        collator = ClsCollator(tok, ml)
        if args.dev_select:
            dv_enc = tok(dev[0], truncation=True, max_length=max_len, padding=False)
            dev_ds = ClsDS(dv_enc, dev[1], ml)

    # dev-select: evaluate the dev set each epoch and keep the best-primary epoch,
    # instead of blindly taking the final epoch (de-noises small tasks like re3d)
    sel = {}
    if args.dev_select:
        compute_metrics = make_compute_metrics(ttype, id2label)
        sel = dict(eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                   load_best_model_at_end=True, metric_for_best_model="primary",
                   greater_is_better=True)
    out_dir = os.path.join(args.tmp, f"{args.task}_{args.model_name}_{seed}")
    # scale eval batch with sequence length: bsz 32 at 2048 tokens exhausts the
    # WSL GPU memory ceiling (dxg make_resident -12 thrash); throughput-only
    eval_bsz = max(4, min(32, (32 * 512) // max(max_len, 512)))
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bsz, per_device_eval_batch_size=eval_bsz,
        num_train_epochs=epochs, learning_rate=args.lr, warmup_ratio=0.06,
        weight_decay=0.01, bf16=True, logging_steps=200, report_to="none", seed=seed,
        dataloader_num_workers=2,
        **(sel or {"save_strategy": "no", "eval_strategy": "no"}),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=dev_ds, data_collator=collator,
                      compute_metrics=compute_metrics)
    trainer.train()
    dev_primary = None
    if args.dev_select:
        dev_primary = trainer.evaluate().get("eval_primary")   # best epoch, on dev
    shutil.rmtree(out_dir, ignore_errors=True)                 # drop epoch checkpoints

    # ---- predict + metrics ----
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    result = {}
    if ttype == "ner":
        from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
        preds = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for p_row, l_row in zip(preds, te_lab):
            tp, tt = [], []
            for pi, li in zip(p_row, l_row):
                if li != -100:
                    tt.append(id2label[li]); tp.append(id2label[int(pi)])
            y_true.append(tt); y_pred.append(tp)
        result["f1_micro"] = float(f1_score(y_true, y_pred, average="micro"))
        result["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        result["precision"] = float(precision_score(y_true, y_pred))
        result["recall"] = float(recall_score(y_true, y_pred))
        result["acc"] = float(accuracy_score(y_true, y_pred))
        result["primary"] = result["f1_micro"]
    elif ttype == "multilabel":
        probs = 1 / (1 + np.exp(-logits))
        yp = (probs > 0.5).astype(int).tolist()
        yt = [list(map(int, r)) for r in test[1]]
        result["eb_f1"] = _eb(yt, yp, "f1")
        result["eb_acc"] = _eb(yt, yp, "acc")
        result["eb_prec"] = _eb(yt, yp, "prec")
        result["eb_rec"] = _eb(yt, yp, "rec")
        from sklearn.metrics import f1_score as skf1
        result["f1_micro"] = float(skf1(yt, yp, average="micro", zero_division=0))
        result["f1_macro"] = float(skf1(yt, yp, average="macro", zero_division=0))
        result["primary"] = result["eb_f1"]
    else:  # binary / multiclass
        from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
        preds = np.argmax(logits, axis=-1)
        yt = test[1]
        result["acc"] = float(accuracy_score(yt, preds))
        result["f1_micro"] = float(f1_score(yt, preds, average="micro"))
        result["f1_macro"] = float(f1_score(yt, preds, average="macro"))
        if ttype == "binary":
            result["f1"] = float(f1_score(yt, preds, average="binary", zero_division=0))
            result["mcc"] = float(matthews_corrcoef(yt, preds))
            result["primary"] = result["f1"]
        else:
            result["primary"] = result["f1_macro"]
    result["lr"] = args.lr
    if dev_primary is not None:
        result["dev_primary"] = round(float(dev_primary), 5)
    del model, trainer
    torch.cuda.empty_cache()
    return result, max_len, epochs, bsz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=os.environ.get("CB2_TMP", "/root/cb2_ft_tmp"),
                    help="fine-tune scratch dir (set CB2_TMP on HPC, e.g. node-local storage)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seq-len", type=int, default=0, help="override config max_seq_length")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--bsz", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dev-select", action="store_true",
                    help="per-epoch dev eval, keep best-primary epoch; records dev_primary")
    ap.add_argument("--prefix-space", action="store_true",
                    help="force add_prefix_space=True on ByteLevel BPE (pre-split NER inputs)")
    args = ap.parse_args()

    cfg_path = os.path.join(args.repo, "configs", args.task + ".json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else DEFAULT_CFG[args.task]
    ttype = TASK_TYPE[args.task]
    seeds = args.seeds or [cfg["initial_seed"] + i for i in range(cfg["num_of_seeds"])]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = ["task", "task_type", "model", "seed", "seq_len", "epochs", "bsz",
              "primary", "metrics_json"]
    new = not os.path.exists(args.out)
    for seed in seeds:
        result, max_len, epochs, bsz = run_once(args, cfg, ttype, seed)
        print(f"[{args.task}|{args.model_name}|seed{seed}|len{max_len}] "
              f"primary={result['primary']:.4f} {result}", flush=True)
        with open(args.out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header); new = False
            w.writerow([args.task, ttype, args.model_name, seed, max_len, epochs, bsz,
                        round(result["primary"], 5), json.dumps(result)])
    print("FT_DONE")


if __name__ == "__main__":
    main()
