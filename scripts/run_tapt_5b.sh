#!/usr/bin/env bash
# TAPT on top of the 5B model (A8, model name ConfliBERT-v2-wsd-5b-TAPT).
# Same recipe as A5 (run_final_experiments.sh stage 2), parent swapped for the 5B
# model: per task, MLM on the task's own train text (lr 1e-4, <=2000 steps), then
# fine-tune under the corrected protocol. Classification reuses the parent's
# best-dev LR and appends to corrected_bench.csv; NER runs the repaired
# prefix-space protocol over the extended grid and appends to ner_psfix.csv.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/tapt5b.log
exec > >(tee -a "$LOG") 2>&1
echo "======== TAPT-5B started $(date) ========"
source /root/cb2-venv/bin/activate

BASE=/root/models/conflibert-v2-wsd-5b
test -d "$BASE" || { echo "[FATAL] missing $BASE"; exit 1; }
mkdir -p /root/models/tapt5b

for TASK in satp_relevant insightCrime IndiaPoliceEvents_sents IndiaPoliceEvents_docs \
            cameo_class 20news BBC_News; do
  echo "== [tapt5b] $TASK =="
  python src/pretrain/tapt.py --base "$BASE" --task "$TASK" \
    --repo external/ConfliBERT --out "/root/models/tapt5b/$TASK" \
    || { echo "[FAIL] tapt5b $TASK"; continue; }
  LR=$(python scripts/best_lr.py "$TASK" ConfliBERT-v2-wsd-5b)
  echo "== [tapt5b-ft] $TASK lr=$LR =="
  python src/eval/finetune_benchmark.py --task "$TASK" \
    --model "/root/models/tapt5b/$TASK" --model-name ConfliBERT-v2-wsd-5b-TAPT \
    --repo external/ConfliBERT --out analysis/data/corrected_bench.csv \
    --lr "$LR" --dev-select --seeds 123 124 125 \
    || echo "[FAIL] tapt5b-ft $TASK"
done

for TASK in cameo_ner re3d; do
  echo "== [tapt5b] $TASK =="
  python src/pretrain/tapt.py --base "$BASE" --task "$TASK" \
    --repo external/ConfliBERT --out "/root/models/tapt5b/$TASK" \
    || { echo "[FAIL] tapt5b $TASK"; continue; }
  python -u scripts/corrected_bench.py --prefix-space --lrs 5e-5 8e-5 1.6e-4 2.4e-4 \
    --out analysis/data/ner_psfix.csv --tasks "$TASK" \
    --models "ConfliBERT-v2-wsd-5b-TAPT-psfix=/root/models/tapt5b/$TASK" \
    || echo "[FAIL] tapt5b-ner $TASK"
done

echo "======== TAPT-5B finished $(date) ========"
echo TAPT5B_DONE
