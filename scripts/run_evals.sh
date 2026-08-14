#!/usr/bin/env bash
# Auto-eval orchestrator: waits for the headline DAPT run to finish, finalizes the
# model, then runs the three evaluation stages in sequence. Each stage is tolerant:
# a failure is logged as [FAIL] and the chain continues. Launched in the background.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/auto_evals.log
mkdir -p outputs analysis/data
exec > >(tee -a "$LOG") 2>&1
echo "==================== auto-evals started $(date) ===================="

OUT=/root/cb2_out/conflibert-v2
V2=/root/models/conflibert-v2
INIT=outputs/models/modernbert-conflivocab-init

# ---------------- 1) wait for training to finish (poll, up to 4h) ----------------
echo "[wait] polling for training completion (run_summary.json or RC= marker)..."
for i in $(seq 1 240); do
  if [ -f "$OUT/run_summary.json" ]; then echo "[wait] run_summary.json found after ${i} min"; break; fi
  if grep -qa 'RC=' outputs/dapt_v2.log 2>/dev/null; then echo "[wait] RC marker found after ${i} min"; break; fi
  sleep 60
done

if [ -f "$OUT/run_summary.json" ] || grep -qa 'RC=0' outputs/dapt_v2.log 2>/dev/null; then
  echo "[wait] training completed cleanly."
else
  echo "[abort] training did not finish cleanly: $(grep -oa 'RC=[0-9]*' outputs/dapt_v2.log | tail -1). Skipping evals."
  exit 1
fi

# ---------------- 2) finalize model -> fast ext4 copy ----------------
echo "[finalize] assembling final model at $V2"
mkdir -p "$V2"
if [ -f "$OUT/model.safetensors" ]; then
  SRC="$OUT"
else
  SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  echo "[finalize] no top-level model; using latest checkpoint: $SRC"
fi
cp -f "$SRC"/*.json "$V2"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V2"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V2"/ 2>/dev/null
cp -f "$SRC"/*.txt "$V2"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2
cp -f "$V2"/* outputs/models/conflibert-v2/ 2>/dev/null   # persist to /mnt/f
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics.csv 2>/dev/null  # stage for the R curve
echo "[finalize] contents of $V2:"; ls -la "$V2"

source /root/cb2-venv/bin/activate

# ---------------- 3) intrinsic pseudo-perplexity ----------------
echo "==================== [1/3] pseudo-PPL $(date) ===================="
python src/eval/pseudo_ppl.py --eval /root/packed/eval_random_aug_1024 \
  --models init="$INIT" v2="$V2" --tokenizer "$V2" \
  --out analysis/data/pseudo_ppl.csv \
  && echo "[ok] pseudo_ppl" || echo "[FAIL] pseudo_ppl rc=$?"

# ---------------- 4) downstream benchmark (all models, all tasks) ----------------
echo "==================== [2/3] downstream benchmark $(date) ===================="
python src/eval/run_suite.py \
  --models ConfliBERT=eventdata-utd/ConfliBERT-scr-uncased ModernBERT-base=answerdotai/ModernBERT-base ConfliBERT-v2="$V2" \
  --tasks all --out analysis/data/downstream_results.csv \
  && echo "[ok] downstream" || echo "[FAIL] downstream rc=$?"

# ---------------- 5) truncation-gap downstream sweep ----------------
echo "==================== [3/3] truncation sweep $(date) ===================="
python src/eval/run_suite.py \
  --models ConfliBERT-v2="$V2" ModernBERT-base=answerdotai/ModernBERT-base \
  --tasks IndiaPoliceEvents_docs insightCrime --seq-lens 512 1024 2048 \
  --out analysis/data/truncation_downstream.csv --seeds 123 124 \
  && echo "[ok] truncation" || echo "[FAIL] truncation rc=$?"

echo "==================== auto-evals finished $(date) ===================="
echo "[result] CSVs in analysis/data:"; ls -la analysis/data/*.csv
