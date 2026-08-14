#!/usr/bin/env bash
# Native-tokenizer headline retrain: the ONLY change vs the augmented run is the tokenizer
# (answerdotai/ModernBERT-base instead of the +4000-token augmented one). Everything else
# (sampling, budget, hyperparameters) is identical, so augmented-vs-native is a clean
# single-variable ablation. Pipeline: pack -> train -> finalize -> eval. Fully autonomous.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/native_pipeline.log
mkdir -p outputs analysis/data
exec > >(tee -a "$LOG") 2>&1
echo "======== native-tokenizer pipeline started $(date) ========"
source /root/cb2-venv/bin/activate

CORPUS=outputs/corpus/parquet
TRAIN=/root/packed/train_native_1024
EVAL=/root/packed/eval_random_native_1024
BASE=answerdotai/ModernBERT-base
OUT=/root/cb2_out/conflibert-v2-native
V2N=/root/models/conflibert-v2-native

# ---------------- 1) pack with NATIVE tokenizer (same sampling as train_aug_1024) ----------------
echo "======== [pack] eval_random native $(date) ========"
python src/data/pack_tokens.py --corpus "$CORPUS" --split eval_random --seqlen 1024 \
  --tokenizer "$BASE" --max-tokens 5000000 --dedup --seed 7 --out "$EVAL" \
  || { echo "[abort] eval pack failed"; exit 1; }

echo "======== [pack] train native $(date) ========"
python src/data/pack_tokens.py --corpus "$CORPUS" --split train --seqlen 1024 \
  --tokenizer "$BASE" \
  --source-weights "News=1.0,Organization=1.0,UTDstory=1.0,Gigaword=0.7,Wikipedia=0.25" \
  --max-tokens 2500000000 --dedup --seed 7 --out "$TRAIN" \
  || { echo "[abort] train pack failed"; exit 1; }

NTOK=$(python -c "import json;print(json.load(open('$TRAIN/meta.json'))['n_tokens'])" 2>/dev/null || echo 0)
echo "[guard] train n_tokens=$NTOK"
if [ "$NTOK" -lt 2000000000 ]; then echo "[abort] train pack too small ($NTOK); not training"; exit 1; fi

# ---------------- 2) DAPT with native tokenizer, SAME config as the augmented run ----------------
echo "======== [train] native DAPT $(date) ========"
python -u src/pretrain/train_dapt.py --base "$BASE" --train "$TRAIN" --eval "$EVAL" --out "$OUT" \
  --mlm-prob 0.30 --lr 5e-5 --bsz 8 --accum 32 --epochs 1 --warmup-ratio 0.05 \
  --save-steps 500 --eval-steps 1000 --log-steps 25 --mem-fraction 0 \
  || { echo "[abort] training failed"; exit 1; }

# ---------------- 3) finalize -> fast ext4 copy + persist to /mnt/f + stage metrics ----------------
echo "======== [finalize] $(date) ========"
mkdir -p "$V2N"
if [ -f "$OUT/model.safetensors" ]; then SRC="$OUT"; else SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
echo "[finalize] source: $SRC"
cp -f "$SRC"/*.json "$V2N"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V2N"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V2N"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2-native
cp -f "$V2N"/* outputs/models/conflibert-v2-native/ 2>/dev/null
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics_native.csv 2>/dev/null
ls -la "$V2N"

# ---------------- 4) evals ----------------
# intrinsic: base vs native-DAPT (same native tokenizer -> comparable)
echo "======== [eval 1/3] pseudo-PPL native $(date) ========"
python src/eval/pseudo_ppl.py --eval "$EVAL" --models base="$BASE" v2native="$V2N" \
  --tokenizer "$BASE" --out analysis/data/pseudo_ppl_native.csv \
  && echo "[ok] ppl" || echo "[FAIL] ppl rc=$?"

# downstream: only the NEW model on the 7 working tasks (ConfliBERT + ModernBERT-base already logged, same config)
echo "======== [eval 2/3] downstream native $(date) ========"
python src/eval/run_suite.py --models ConfliBERT-v2-native="$V2N" \
  --tasks satp_relevant IndiaPoliceEvents_sents IndiaPoliceEvents_docs insightCrime cameo_class BBC_News re3d \
  --out analysis/data/downstream_native.csv \
  && echo "[ok] downstream" || echo "[FAIL] downstream rc=$?"

# truncation sweep for the new model (ModernBERT-base sweep already logged, comparable)
echo "======== [eval 3/3] truncation sweep native $(date) ========"
python src/eval/run_suite.py --models ConfliBERT-v2-native="$V2N" \
  --tasks IndiaPoliceEvents_docs insightCrime --seq-lens 512 1024 2048 \
  --out analysis/data/truncation_native.csv --seeds 123 124 \
  && echo "[ok] truncation" || echo "[FAIL] truncation rc=$?"

echo "======== native pipeline finished $(date) ========"
ls -la analysis/data/*native*.csv
