#!/usr/bin/env python
"""
Task-adaptive pretraining (TAPT, Gururangan et al. 2020): continue masked-LM
training on a benchmark task's own unlabeled train text before fine-tuning.

Usage:
  python tapt.py --base /root/models/conflibert-v2-wsd --task satp_relevant \
    --repo external/ConfliBERT --out /root/models/tapt/satp_relevant
"""
import argparse, math, os, sys
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from finetune_benchmark import TASK_TYPE, load_classification, load_ner  # noqa: E402


class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: self.enc[k][i] for k in self.enc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--repo", default="external/ConfliBERT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps-cap", type=int, default=2000)
    args = ap.parse_args()

    data_dir = os.path.join(args.repo, "data", args.task)
    ttype = TASK_TYPE[args.task]
    if ttype == "ner":
        train, _, _, _ = load_ner(data_dir)
        texts = [" ".join(w) for w, _ in train]
    else:
        train, _, _, _ = load_classification(data_dir, ttype)
        texts = train[0]

    tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    enc = tok(texts, truncation=True, max_length=args.max_len, padding=False)
    ds = TextDS(enc)
    spe = math.ceil(len(ds) / args.bsz)
    max_steps = min(args.max_steps_cap, args.epochs * spe)
    print(f"[tapt] {args.task}: {len(ds)} texts, {spe} steps/epoch, max_steps={max_steps}",
          flush=True)

    model = AutoModelForMaskedLM.from_pretrained(args.base, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True,
                                              mlm_probability=args.mlm_prob)
    targs = TrainingArguments(
        output_dir=args.out + "_tmp", max_steps=max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.bsz, warmup_ratio=0.06, weight_decay=0.01,
        lr_scheduler_type="cosine", bf16=True, logging_steps=100,
        save_strategy="no", report_to="none", seed=13,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    import shutil
    shutil.rmtree(args.out + "_tmp", ignore_errors=True)
    print(f"TAPT_DONE {args.task} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
