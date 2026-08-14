#!/usr/bin/env python
"""
CrisisWatch trend-assessment fine-tuning (paper experiment: the judgment task).

Three-class classification (unchanged / deteriorated / improved) of a country's
month entry, with the country's trailing entries as context. Temporal split
comes from the task file (train through 2024-12, dev 2025-01..05, test 2025-06+).
Class-weighted cross-entropy handles the heavy imbalance.

Input modes:
  --input-mode current   the month's entry only
  --input-mode full      trailing history + the month's entry (the long input)
Inference modes:
  default                single pass at --seq-len
  --window-infer         sliced: 512-token windows, mean-pooled class probs
                         (the practitioner's workaround for a 512 model)

Per-example test predictions append to analysis/data/cw_preds.csv.
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_benchmark import ClsDS, ClsCollator

CLASSES = ["unchanged", "deteriorated", "improved"]
WINDOW, OVERLAP = 512, 128


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def window_probs(model, tok, texts, batch=32):
    out = np.zeros((len(texts), len(CLASSES)))
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
                probs.append(torch.softmax(logits, -1).cpu().numpy())
            out[i] = np.concatenate(probs).mean(axis=0)   # mean-pool for multiclass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="external/crisiswatch/task.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--input-mode", choices=["current", "full"], required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--window-infer", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--tmp", default=os.environ.get("CB2_TMP", "/root/cb2_ft_tmp"),
                    help="fine-tune scratch dir (set CB2_TMP on HPC, e.g. node-local storage)")
    ap.add_argument("--out", default="analysis/data/cw_preds.csv")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.task_file, encoding="utf-8")]
    def text_of(r):
        if args.input_mode == "current" or not r["text_history"]:
            return r["text_current"]
        return r["text_history"] + "\n" + r["text_current"]
    lab2id = {c: i for i, c in enumerate(CLASSES)}
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] in ("dev", "test")]

    counts = np.bincount([lab2id[r["label"]] for r in tr], minlength=3)
    weights = torch.tensor((counts.sum() / (3.0 * counts)), dtype=torch.float32)
    print(f"train={len(tr)} eval={len(te)} class_weights={weights.tolist()}")

    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    enc_tr = tok([text_of(r) for r in tr], truncation=True,
                 max_length=args.seq_len, padding=False)
    te_texts = [text_of(r) for r in te]
    enc_te = tok(te_texts, truncation=True, max_length=args.seq_len, padding=False)
    eval_bsz = max(4, min(32, (32 * 512) // max(args.seq_len, 512)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = (["model", "input_mode", "infer", "seq_len", "seed", "split",
               "country", "year", "month", "gold"] +
              [f"prob_{c}" for c in CLASSES])
    new = not os.path.exists(args.out)
    infer = "window" if args.window_infer else "single"

    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=3)
        out_dir = os.path.join(args.tmp, f"cw_{args.model_name}_{args.input_mode}_{seed}")
        targs = TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=eval_bsz, num_train_epochs=args.epochs,
            learning_rate=args.lr, warmup_ratio=0.06, weight_decay=0.01,
            bf16=True, logging_steps=100, report_to="none", seed=seed,
            dataloader_num_workers=2, save_strategy="no", eval_strategy="no")
        trainer = WeightedTrainer(
            class_weights=weights, model=model, args=targs,
            train_dataset=ClsDS(dict(enc_tr), [lab2id[r["label"]] for r in tr],
                                multilabel=False),
            data_collator=ClsCollator(tok, multilabel=False))
        trainer.train()

        if args.window_infer:
            probs = window_probs(model.cuda(), tok, te_texts)
        else:
            pred = trainer.predict(ClsDS(dict(enc_te),
                                         [lab2id[r["label"]] for r in te],
                                         multilabel=False))
            e = np.exp(pred.predictions - pred.predictions.max(-1, keepdims=True))
            probs = e / e.sum(-1, keepdims=True)

        with open(args.out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header); new = False
            for r, p in zip(te, probs):
                w.writerow([args.model_name, args.input_mode, infer, args.seq_len,
                            seed, r["split"], r["country"], r["year"], r["month"],
                            r["label"]] + [round(float(x), 5) for x in p])
        print(f"[{args.model_name}|{args.input_mode}|{infer}|seed{seed}] done",
              flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        del model, trainer
        torch.cuda.empty_cache()
    print("CW_FINETUNE_DONE")


if __name__ == "__main__":
    main()
