#!/usr/bin/env bash
# Rerun-safe resume of the final experiments (context comparison + TAPT).
# Skips any (task, model, length) cell with 3 recorded seeds and any TAPT task
# with 3 recorded fine-tune rows. Safe to run any number of times.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/final_experiments.log
exec > >(tee -a "$LOG") 2>&1
echo "======== resume final experiments $(date) ========"
source /root/cb2-venv/bin/activate

A3=/root/models/conflibert-v2-wsd
CTX_OUT=analysis/data/context_corrected.csv
BENCH=analysis/data/corrected_bench.csv

ctx_done () {  # task model len -> 0 if 3 seeds recorded
  local n
  n=$(grep -c "^$1,[^,]*,$2,[0-9]*,$3," "$CTX_OUT" 2>/dev/null || echo 0)
  [ "$n" -ge 3 ]
}

# ---------------- stage 1: context length ----------------
for TASK in IndiaPoliceEvents_docs insightCrime; do
  for LEN in 1024 2048; do
    if ctx_done "$TASK" ModernBERT-base "$LEN"; then
      echo "== [ctx] skip $TASK $LEN ModernBERT-base (done) =="
    else
      LR=$(python scripts/best_lr.py "$TASK" ModernBERT-base)
      echo "== [ctx] $TASK len=$LEN ModernBERT-base lr=$LR =="
      python src/eval/finetune_benchmark.py --task "$TASK" \
        --model answerdotai/ModernBERT-base --model-name ModernBERT-base \
        --repo external/ConfliBERT --out "$CTX_OUT" --seq-len "$LEN" \
        --lr "$LR" --dev-select --seeds 123 124 125 \
        || echo "[FAIL] ctx $TASK $LEN base"
    fi
    if ctx_done "$TASK" ConfliBERT-v2-wsd "$LEN"; then
      echo "== [ctx] skip $TASK $LEN ConfliBERT-v2-wsd (done) =="
    else
      LR=$(python scripts/best_lr.py "$TASK" ConfliBERT-v2-wsd)
      echo "== [ctx] $TASK len=$LEN ConfliBERT-v2-wsd lr=$LR =="
      python src/eval/finetune_benchmark.py --task "$TASK" \
        --model "$A3" --model-name ConfliBERT-v2-wsd \
        --repo external/ConfliBERT --out "$CTX_OUT" --seq-len "$LEN" \
        --lr "$LR" --dev-select --seeds 123 124 125 \
        || echo "[FAIL] ctx $TASK $LEN wsd"
    fi
  done
done
echo "======== stage 1 done $(date) ========"

# ---------------- stage 2: TAPT on A3 ----------------
mkdir -p /root/models/tapt
for TASK in satp_relevant insightCrime IndiaPoliceEvents_sents IndiaPoliceEvents_docs \
            cameo_class cameo_ner re3d 20news BBC_News; do
  n=$(grep -c "^$TASK,[^,]*,ConfliBERT-v2-wsd-TAPT," "$BENCH" 2>/dev/null || echo 0)
  if [ "$n" -ge 3 ]; then
    echo "== [tapt] skip $TASK (done) =="
    continue
  fi
  if [ ! -f "/root/models/tapt/$TASK/model.safetensors" ]; then
    echo "== [tapt] $TASK =="
    python src/pretrain/tapt.py --base "$A3" --task "$TASK" \
      --repo external/ConfliBERT --out "/root/models/tapt/$TASK" \
      || { echo "[FAIL] tapt $TASK"; continue; }
  fi
  LR=$(python scripts/best_lr.py "$TASK" ConfliBERT-v2-wsd)
  echo "== [tapt-ft] $TASK lr=$LR =="
  python src/eval/finetune_benchmark.py --task "$TASK" \
    --model "/root/models/tapt/$TASK" --model-name ConfliBERT-v2-wsd-TAPT \
    --repo external/ConfliBERT --out "$BENCH" \
    --lr "$LR" --dev-select --seeds 123 124 125 \
    || echo "[FAIL] tapt-ft $TASK"
done
echo "======== final experiments finished $(date) ========"
