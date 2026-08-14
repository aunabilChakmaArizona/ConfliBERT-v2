#!/usr/bin/env bash
# Final experiment pair, run overnight:
#   Stage 1: context-length comparison under the corrected protocol. Document
#     tasks at 1024 and 2048 for ModernBERT-base and A3 (v2-wsd); learning rate
#     reuses each (task, model) best-dev choice from the 512 sweep; 3 seeds,
#     dev-select. ConfliBERT-2021's corrected 512 rows are the fixed anchor.
#   Stage 2: TAPT (A5). For each of the nine tasks: adapt A3 on the task's own
#     train text, then fine-tune under the corrected protocol; appends to
#     corrected_bench.csv as model ConfliBERT-v2-wsd-TAPT.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/final_experiments.log
exec > >(tee -a "$LOG") 2>&1
echo "======== final experiments started $(date) ========"
source /root/cb2-venv/bin/activate

A3=/root/models/conflibert-v2-wsd
CTX_OUT=analysis/data/context_corrected.csv

# ---------------- stage 1: context length ----------------
for TASK in IndiaPoliceEvents_docs insightCrime; do
  for LEN in 1024 2048; do
    LR=$(python scripts/best_lr.py "$TASK" ModernBERT-base)
    echo "== [ctx] $TASK len=$LEN ModernBERT-base lr=$LR =="
    python src/eval/finetune_benchmark.py --task "$TASK" \
      --model answerdotai/ModernBERT-base --model-name ModernBERT-base \
      --repo external/ConfliBERT --out "$CTX_OUT" --seq-len "$LEN" \
      --lr "$LR" --dev-select --seeds 123 124 125 \
      || echo "[FAIL] ctx $TASK $LEN base"
    LR=$(python scripts/best_lr.py "$TASK" ConfliBERT-v2-wsd)
    echo "== [ctx] $TASK len=$LEN ConfliBERT-v2-wsd lr=$LR =="
    python src/eval/finetune_benchmark.py --task "$TASK" \
      --model "$A3" --model-name ConfliBERT-v2-wsd \
      --repo external/ConfliBERT --out "$CTX_OUT" --seq-len "$LEN" \
      --lr "$LR" --dev-select --seeds 123 124 125 \
      || echo "[FAIL] ctx $TASK $LEN wsd"
  done
done
echo "======== stage 1 done $(date) ========"

# ---------------- stage 2: TAPT on A3 ----------------
mkdir -p /root/models/tapt
for TASK in satp_relevant insightCrime IndiaPoliceEvents_sents IndiaPoliceEvents_docs \
            cameo_class cameo_ner re3d 20news BBC_News; do
  echo "== [tapt] $TASK =="
  python src/pretrain/tapt.py --base "$A3" --task "$TASK" \
    --repo external/ConfliBERT --out "/root/models/tapt/$TASK" \
    || { echo "[FAIL] tapt $TASK"; continue; }
  LR=$(python scripts/best_lr.py "$TASK" ConfliBERT-v2-wsd)
  echo "== [tapt-ft] $TASK lr=$LR =="
  python src/eval/finetune_benchmark.py --task "$TASK" \
    --model "/root/models/tapt/$TASK" --model-name ConfliBERT-v2-wsd-TAPT \
    --repo external/ConfliBERT --out analysis/data/corrected_bench.csv \
    --lr "$LR" --dev-select --seeds 123 124 125 \
    || echo "[FAIL] tapt-ft $TASK"
done
echo "======== final experiments finished $(date) ========"
