#!/usr/bin/env bash
# R1 headline: resume CPT from the ModernBERT pre-decay STABLE checkpoint (converted to
# HF) with a Warmup-Stable-Decay schedule -- the ModernBERT-correct continued-pretraining
# recipe. Native tokenizer, REUSING the native 2.5B pack (no repack). This is the fix for
# the two prior runs that cold-cosine-restarted a fully-decayed checkpoint and stayed flat
# downstream. Pipeline: (validate inputs) -> train (WSD) -> finalize -> 3 evals. Autonomous.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/r1_wsd_pipeline.log
mkdir -p outputs analysis/data
exec > >(tee -a "$LOG") 2>&1
echo "======== R1 WSD pipeline started $(date) ========"
source /root/cb2-venv/bin/activate

BASE=/root/models/modernbert-stable-hf          # converted pre-decay stable checkpoint
TRAIN=/root/packed/train_native_1024            # reuse native pack (same tokenizer)
EVAL=/root/packed/eval_random_native_1024
OUT=/root/cb2_out/conflibert-v2-wsd
V2W=/root/models/conflibert-v2-wsd

# ---------------- 0) preflight guards ----------------
[ -f "$BASE/model.safetensors" ] || { echo "[abort] missing converted stable ckpt $BASE"; exit 1; }
[ -f "$TRAIN/meta.json" ] || { echo "[abort] missing native train pack $TRAIN"; exit 1; }
[ -f "$EVAL/meta.json" ]  || { echo "[abort] missing native eval pack $EVAL"; exit 1; }
NTOK=$(python -c "import json;print(json.load(open('$TRAIN/meta.json'))['n_tokens'])" 2>/dev/null || echo 0)
echo "[guard] reusing native train pack n_tokens=$NTOK"
[ "$NTOK" -ge 2000000000 ] || { echo "[abort] train pack too small ($NTOK)"; exit 1; }

# ---------------- 1) DAPT from stable ckpt with WSD ----------------
# peak LR 2e-4 (stable-phase regime, 4x the failed 5e-5), 3% warmup to re-seat fresh Adam
# moments, 20% linear decay tail. Same 2.5B budget / batch as the prior runs for comparability.
echo "======== [train] WSD DAPT from stable ckpt $(date) ========"
python -u src/pretrain/train_dapt.py --base "$BASE" --train "$TRAIN" --eval "$EVAL" --out "$OUT" \
  --scheduler wsd --lr 2e-4 --decay-ratio 0.20 --warmup-ratio 0.03 \
  --mlm-prob 0.30 --bsz 8 --accum 32 --epochs 1 --weight-decay 0.01 \
  --save-steps 1000 --eval-steps 1000 --log-steps 25 --mem-fraction 0 \
  || { echo "[abort] training failed"; exit 1; }

# ---------------- 2) finalize ----------------
echo "======== [finalize] $(date) ========"
mkdir -p "$V2W"
if [ -f "$OUT/model.safetensors" ]; then SRC="$OUT"; else SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
echo "[finalize] source: $SRC"
cp -f "$SRC"/*.json "$V2W"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V2W"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V2W"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2-wsd
cp -f "$V2W"/* outputs/models/conflibert-v2-wsd/ 2>/dev/null
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics_wsd.csv 2>/dev/null
ls -la "$V2W"

# ---------------- 3) evals ----------------
echo "======== [eval 1/3] pseudo-PPL wsd $(date) ========"
python src/eval/pseudo_ppl.py --eval "$EVAL" --models base=answerdotai/ModernBERT-base v2wsd="$V2W" \
  --tokenizer answerdotai/ModernBERT-base --out analysis/data/pseudo_ppl_wsd.csv \
  && echo "[ok] ppl" || echo "[FAIL] ppl rc=$?"

echo "======== [eval 2/3] downstream wsd $(date) ========"
python src/eval/run_suite.py --models ConfliBERT-v2-wsd="$V2W" \
  --tasks satp_relevant IndiaPoliceEvents_sents IndiaPoliceEvents_docs insightCrime cameo_class BBC_News re3d \
  --out analysis/data/downstream_wsd.csv \
  && echo "[ok] downstream" || echo "[FAIL] downstream rc=$?"

echo "======== [eval 3/3] truncation sweep wsd $(date) ========"
python src/eval/run_suite.py --models ConfliBERT-v2-wsd="$V2W" \
  --tasks IndiaPoliceEvents_docs insightCrime --seq-lens 512 1024 2048 \
  --out analysis/data/truncation_wsd.csv --seeds 123 124 \
  && echo "[ok] truncation" || echo "[FAIL] truncation rc=$?"

echo "======== R1 WSD pipeline finished $(date) ========"
ls -la analysis/data/*wsd*.csv
