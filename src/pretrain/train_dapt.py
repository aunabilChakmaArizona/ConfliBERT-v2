#!/usr/bin/env python
"""
Domain-adaptive pretraining (DAPT) of ModernBERT via continued MLM.

Reads packed uint16 token blocks (from pack_tokens.py), masks with a configurable
rate (ModernBERT default 0.30), and continues masked-LM training. bf16, SDPA
attention, gradient accumulation, checkpoint/resume, and a metrics CSV logging
tokens-seen / loss / lr / tokens-per-sec so ablations are budget-comparable and
R can plot the curves.

Usage (single GPU):
  python train_dapt.py \
    --train .../outputs/packed/train_1024 --eval .../outputs/packed/eval_random_1024 \
    --out   .../outputs/models/conflibert-v2 \
    --mlm-prob 0.30 --lr 5e-5 --bsz 16 --accum 4 --epochs 1 [--lora] [--max-steps N]

Usage (multi-GPU, e.g. one Delta A100x4 node; --bsz/--accum are PER DEVICE, so
divide the single-GPU --accum by the GPU count to keep the same global batch):
  torchrun --nproc_per_node 4 train_dapt.py ... --bsz 16 --accum 1
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, TrainerCallback)

MODERNBERT = "answerdotai/ModernBERT-base"


class PackedBlocks(Dataset):
    """Memmap-backed fixed-length token blocks with precomputed special-token masks."""
    def __init__(self, packed_dir, special_ids):
        meta = json.load(open(os.path.join(packed_dir, "meta.json")))
        self.seqlen = meta["seqlen"]
        self.n = meta["n_blocks"]
        self.arr = np.memmap(os.path.join(packed_dir, meta["path"]), dtype=np.uint16,
                             mode="r", shape=(self.n, self.seqlen))
        self.special = np.array(sorted(special_ids), dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = np.asarray(self.arr[i], dtype=np.int64)
        stm = np.isin(row, self.special).astype(np.int64)
        return {"input_ids": row, "special_tokens_mask": stm}


class NewEmbedWarmup(TrainerCallback):
    """For the first N optimizer steps, let ONLY the embedding rows >= first_new_row
    receive gradient (tied output head shares the tensor, so it is covered too).
    New FVT-initialized vocab rows then settle before full-model CPT starts, instead
    of the whole network adapting around bad rows."""
    def __init__(self, model, warmup_steps, first_new_row):
        self.model = model
        self.emb = model.get_input_embeddings().weight
        self.n = warmup_steps
        self.row0 = first_new_row
        self.done_msg = False

    def on_pre_optimizer_step(self, args, state, control, **kw):
        if state.global_step >= self.n:
            if not self.done_msg:
                print(f"[dapt] new-embed warmup finished at step {state.global_step}; "
                      f"full model now training", flush=True)
                self.done_msg = True
            return
        for p in self.model.parameters():
            if p.grad is None or p is self.emb:
                continue
            p.grad.zero_()
        if self.emb.grad is not None:
            self.emb.grad[: self.row0].zero_()


class MetricsCSV(TrainerCallback):
    """Log step/tokens/loss/lr/tokens-per-sec to a CSV for R plotting."""
    def __init__(self, path, tokens_per_step):
        self.path, self.tps = path, tokens_per_step
        self.t0 = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "tokens_seen", "train_loss", "eval_loss", "eval_ppl",
                 "lr", "grad_norm", "tokens_per_sec", "wall_s"])

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        if self.t0 is None:
            self.t0 = time.time()
        step = state.global_step
        toks = step * self.tps
        wall = time.time() - self.t0
        tps = toks / wall if wall > 0 else 0
        ppl = ""
        if "eval_loss" in logs:
            try:
                ppl = round(float(np.exp(min(logs["eval_loss"], 20))), 3)
            except Exception:
                ppl = ""
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, toks, logs.get("loss", ""), logs.get("eval_loss", ""), ppl,
                logs.get("learning_rate", ""), logs.get("grad_norm", ""),
                round(tps, 1), round(wall, 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=MODERNBERT)
    ap.add_argument("--mlm-prob", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--scheduler", default="cosine",
                    help='"cosine" (default) or "wsd" (warmup-stable-decay, ModernBERT CPT recipe)')
    ap.add_argument("--decay-ratio", type=float, default=0.20,
                    help="fraction of total steps for the WSD linear decay tail")
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--log-steps", type=int, default=50)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=0.82,
                    help="cap process VRAM to leave headroom for the desktop")
    ap.add_argument("--new-embed-warmup-steps", type=int, default=0,
                    help="train ONLY new vocab embedding rows for the first N optimizer "
                         "steps (augmented-tokenizer runs; incompatible with --lora)")
    ap.add_argument("--new-embed-first-row", type=int, default=50368,
                    help="first embedding row treated as new vocab (ModernBERT-base "
                         "tokenizer length before augmentation)")
    ap.add_argument("--lora", action="store_true", help="LoRA-DAPT (PEFT) instead of full FT")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume from the last checkpoint in --out if one exists, else "
                         "start fresh (safe default for requeueable Slurm jobs)")
    ap.add_argument("--report", default="tensorboard", help='"tensorboard" or "none"')
    ap.add_argument("--run-name", default="")
    ap.add_argument("--tb-root", default="outputs/tb", help="shared TensorBoard logdir")
    args = ap.parse_args()
    run_name = args.run_name or os.path.basename(args.out.rstrip("/\\"))
    logging_dir = os.path.join(args.tb_root, run_name)

    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; plain python leaves them unset (=1 GPU)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = int(os.environ.get("RANK", 0)) == 0

    if torch.cuda.is_available() and 0 < args.mem_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, local_rank)

    tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    # fp32 master weights + bf16 autocast (via TrainingArguments bf16=True) for stable optimization
    model = AutoModelForMaskedLM.from_pretrained(
        args.base, dtype=torch.float32, attn_implementation="sdpa")
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora:
        from peft import LoraConfig, get_peft_model  # installed on demand
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                         target_modules=["Wqkv", "Wo", "Wi"], task_type="TOKEN_CLS")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    train_ds = PackedBlocks(args.train, tok.all_special_ids)
    eval_ds = PackedBlocks(args.eval, tok.all_special_ids) if args.eval else None
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True,
                                              mlm_probability=args.mlm_prob)

    tokens_per_step = args.bsz * args.accum * world_size * train_ds.seqlen
    if is_main:
        print(f"[dapt] train blocks={len(train_ds):,} seqlen={train_ds.seqlen} "
              f"| world={world_size} | tokens/step={tokens_per_step:,} "
              f"| ~{len(train_ds)*train_ds.seqlen/1e9:.2f}B tokens/epoch", flush=True)

    # Warmup-Stable-Decay (ModernBERT continued-pretraining recipe): short warmup,
    # long stable hold at the peak LR, then a linear decay tail. num_decay_steps is
    # absolute, so we compute the total step count here (global batch = bsz * accum
    # * world_size under DDP).
    import math
    steps_per_epoch = math.ceil(len(train_ds) / (args.bsz * args.accum * world_size))
    total_steps = args.max_steps if args.max_steps and args.max_steps > 0 \
        else int(steps_per_epoch * args.epochs)
    if args.scheduler == "wsd":
        sched_type = "warmup_stable_decay"
        n_decay = int(args.decay_ratio * total_steps)
        sched_kwargs = {"num_decay_steps": n_decay, "decay_type": "linear",
                        "min_lr_ratio": 0.0}
        if is_main:
            print(f"[dapt] WSD schedule: total~{total_steps} warmup={int(args.warmup_ratio*total_steps)} "
                  f"stable={total_steps-int(args.warmup_ratio*total_steps)-n_decay} decay={n_decay} "
                  f"peak_lr={args.lr}", flush=True)
    else:
        sched_type = args.scheduler
        sched_kwargs = {}

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=sched_type,
        lr_scheduler_kwargs=sched_kwargs,
        bf16=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        report_to=[args.report] if args.report != "none" else "none",
        logging_dir=logging_dir,
        run_name=run_name,
        remove_unused_columns=False,
    )
    # metrics CSV only on the main process (other ranks would clobber the file)
    callbacks = []
    if is_main:
        callbacks.append(MetricsCSV(os.path.join(args.out, "metrics.csv"), tokens_per_step))
    if args.new_embed_warmup_steps > 0:
        assert not args.lora, "--new-embed-warmup-steps is incompatible with --lora"
        callbacks.append(NewEmbedWarmup(model, args.new_embed_warmup_steps,
                                        args.new_embed_first_row))
        print(f"[dapt] new-embed warmup: rows >= {args.new_embed_first_row} only, "
              f"first {args.new_embed_warmup_steps} steps", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator, callbacks=callbacks)

    resume = args.resume
    if args.auto_resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.out) is not None
        if is_main:
            print(f"[dapt] auto-resume: {'found checkpoint, resuming' if resume else 'no checkpoint, fresh start'}",
                  flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.out)
    dt = time.time() - t0
    if trainer.is_world_process_zero():
        tok.save_pretrained(args.out)
        summary = {"gpu_hours": round(dt * world_size / 3600, 3),
                   "tokens_per_step": tokens_per_step,
                   "final_step": trainer.state.global_step,
                   "tokens_seen": trainer.state.global_step * tokens_per_step,
                   "args": vars(args)}
        json.dump(summary, open(os.path.join(args.out, "run_summary.json"), "w"), indent=2)
        print(f"DAPT_DONE {json.dumps(summary)}")


if __name__ == "__main__":
    main()
