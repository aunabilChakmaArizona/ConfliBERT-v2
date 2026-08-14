#!/usr/bin/env python
"""
E4: measured coding throughput and cost per million documents (paper experiment).

Times end-to-end document classification (tokenization + bf16 inference) on the real
IndiaPoliceEvents length distribution for each instrument configuration, including the
sliding-window mode a 512-capped model needs to fully cover long documents. Weights are
base models with a fresh classification head; fine-tuning does not change inference
cost. Batch sizes follow the harness formula used in all benchmark runs.

Usage (from repo root, WSL venv, GPU idle):
  python src/eval/throughput_bench.py --out analysis/data/throughput.csv
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CONFIGS = [
    # (name, model, seq_len, mode)
    ("ConfliBERT-2021-trunc512", "eventdata-utd/ConfliBERT-scr-uncased", 512, "truncate"),
    ("ConfliBERT-2021-slide512", "eventdata-utd/ConfliBERT-scr-uncased", 512, "sliding"),
    ("ConfliBERT-v2-512",  "outputs/models/conflibert-v2-wsd",  512, "truncate"),
    ("ConfliBERT-v2-1024", "outputs/models/conflibert-v2-wsd", 1024, "truncate"),
    ("ConfliBERT-v2-2048", "outputs/models/conflibert-v2-wsd", 2048, "truncate"),
    ("ModernBERT-base-2048", "answerdotai/ModernBERT-base", 2048, "truncate"),
]
STRIDE = 448          # sliding-window stride (64-token overlap)
REPEATS = 3
REPLICAS = 4          # replicate the 1,257-doc corpus for stable timing


def load_texts(corpus_dir):
    texts = []
    with open(os.path.join(corpus_dir, "docs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            texts.append(json.loads(line)["doc_text"])
    return texts * REPLICAS


def batches(rows, bsz):
    for i in range(0, len(rows), bsz):
        yield rows[i:i + bsz]


def run_config(name, model_path, seq_len, mode, texts, out_rows):
    tok = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=5, problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16).cuda().eval()
    bsz = max(4, min(32, (32 * 512) // max(seq_len, 512)))

    times = []
    for rep in range(REPEATS + 1):          # first rep is warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_windows = 0
        with torch.no_grad():
            for chunk in batches(texts, 256):        # tokenize in chunks
                if mode == "truncate":
                    enc = tok(chunk, truncation=True, max_length=seq_len,
                              padding=True, return_tensors="pt")
                    rows_ids = enc["input_ids"]
                    rows_att = enc["attention_mask"]
                else:                                # sliding: full coverage at 512
                    enc = tok(chunk, truncation=True, max_length=seq_len,
                              stride=STRIDE, return_overflowing_tokens=True,
                              padding=True, return_tensors="pt")
                    rows_ids = enc["input_ids"]
                    rows_att = enc["attention_mask"]
                n_windows += rows_ids.shape[0]
                for j in range(0, rows_ids.shape[0], bsz):
                    model(input_ids=rows_ids[j:j+bsz].cuda(),
                          attention_mask=rows_att[j:j+bsz].cuda())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if rep > 0:
            times.append(dt)
        print(f"  [{name}] rep{rep} {dt:.1f}s windows={n_windows}", flush=True)

    dt = float(np.median(times))
    docs_per_sec = len(texts) / dt
    hours_per_million = 1e6 / docs_per_sec / 3600
    out_rows.append([name, model_path, seq_len, mode, bsz, len(texts),
                     round(n_windows / len(texts), 3), round(dt, 2),
                     round(docs_per_sec, 2), round(hours_per_million, 2)])
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="external/IndiaPoliceEvents/data/final")
    ap.add_argument("--out", default="analysis/data/throughput.csv")
    args = ap.parse_args()
    texts = load_texts(args.corpus)
    print(f"timing corpus: {len(texts)} docs")
    out_rows = []
    for cfg in CONFIGS:
        run_config(*cfg, texts, out_rows)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "model", "seq_len", "mode", "bsz", "n_docs",
                    "windows_per_doc", "median_sec", "docs_per_sec",
                    "gpu_hours_per_million_docs"])
        w.writerows(out_rows)
    print("THROUGHPUT_DONE")


if __name__ == "__main__":
    main()
