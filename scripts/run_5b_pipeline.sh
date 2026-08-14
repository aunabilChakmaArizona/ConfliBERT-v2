#!/usr/bin/env bash
# Run B: the 5B-token headline extension. Same model (ModernBERT-base, 150M params),
# same WSD-from-stable recipe and source weights as v2-wsd (A3) -- the ONLY change is
# the continued-pretraining budget: 5B tokens instead of 2.5B, still one pass over
# fresh text (weighted corpus pool ~6.5B tokens). Rationale: A3's eval loss was still
# falling at the 2.5B cutoff, and the corrected benchmark puts v2 ~1.5 pts behind
# ConfliBERT-2021 on the 9-task average, all of it on non-NER tasks.
# Pipeline: pack 5B -> train (~28h) -> finalize -> pseudo-PPL + corrected bench + NER.
# RUN ALONE: do not start while any other GPU job or pack build is running.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/wsd5b_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "======== 5B pipeline started $(date) ========"
source /root/cb2-venv/bin/activate

CORPUS=outputs/corpus/parquet
STABLE=/root/models/modernbert-stable-hf
TRAIN=/root/packed/train_native_1024_5b
EVAL=/root/packed/eval_random_native_1024
OUT=/root/cb2_out/conflibert-v2-wsd-5b
V5=/root/models/conflibert-v2-wsd-5b

[ -f "$STABLE/model.safetensors" ] || { echo "[abort] missing stable ckpt"; exit 1; }
[ -f "$EVAL/meta.json" ] || { echo "[abort] missing native eval pack"; exit 1; }

# ---------------- 1) pack 5B with the native tokenizer (same sampling as A3) ----------------
if [ ! -f "$TRAIN/meta.json" ]; then
  echo "======== [pack] 5B native $(date) ========"
  python src/data/pack_tokens.py --corpus "$CORPUS" --split train --seqlen 1024 \
    --tokenizer answerdotai/ModernBERT-base \
    --source-weights "News=1.0,Organization=1.0,UTDstory=1.0,Gigaword=0.7,Wikipedia=0.25" \
    --max-tokens 5000000000 --dedup --seed 7 --out "$TRAIN" \
    || { echo "[abort] pack failed"; exit 1; }
fi
NTOK=$(python -c "import json;print(json.load(open('$TRAIN/meta.json'))['n_tokens'])" 2>/dev/null || echo 0)
echo "[guard] 5B pack n_tokens=$NTOK"
[ "$NTOK" -ge 4500000000 ] || { echo "[abort] pack too small ($NTOK); corpus pool exhausted?"; exit 1; }

# ---------------- 2) WSD CPT from stable ckpt, 5B budget ----------------
echo "======== [train] 5B WSD DAPT $(date) ========"
python -u src/pretrain/train_dapt.py --base "$STABLE" --train "$TRAIN" --eval "$EVAL" --out "$OUT" \
  --scheduler wsd --lr 2e-4 --decay-ratio 0.20 --warmup-ratio 0.03 \
  --mlm-prob 0.30 --bsz 8 --accum 32 --epochs 1 --weight-decay 0.01 \
  --save-steps 1000 --eval-steps 1000 --log-steps 25 --mem-fraction 0 \
  || { echo "[abort] training failed"; exit 1; }

# ---------------- 3) finalize ----------------
echo "======== [finalize] $(date) ========"
mkdir -p "$V5"
if [ -f "$OUT/model.safetensors" ]; then SRC="$OUT"; else SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
echo "[finalize] source: $SRC"
cp -f "$SRC"/*.json "$V5"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V5"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V5"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2-wsd-5b
cp -f "$V5"/* outputs/models/conflibert-v2-wsd-5b/ 2>/dev/null
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics_wsd_5b.csv 2>/dev/null

# ---------------- 4) evals ----------------
echo "======== [eval 1/3] pseudo-PPL $(date) ========"
python src/eval/pseudo_ppl.py --eval "$EVAL" \
  --models base=answerdotai/ModernBERT-base v2wsd=/root/models/conflibert-v2-wsd v2wsd5b="$V5" \
  --tokenizer answerdotai/ModernBERT-base --out analysis/data/pseudo_ppl_wsd_5b.csv \
  && echo "[ok] ppl" || echo "[FAIL] ppl rc=$?"

echo "======== [eval 2/3] corrected bench, classification $(date) ========"
python -u scripts/corrected_bench.py \
  --models ConfliBERT-v2-wsd-5b="$V5" \
  --tasks satp_relevant IndiaPoliceEvents_sents IndiaPoliceEvents_docs insightCrime \
          cameo_class BBC_News 20news \
  --out analysis/data/corrected_bench.csv \
  && echo "[ok] bench" || echo "[FAIL] bench rc=$?"

echo "======== [eval 3/3] NER, fixed protocol $(date) ========"
python -u scripts/corrected_bench.py --prefix-space --lrs 5e-5 8e-5 1.6e-4 2.4e-4 \
  --models ConfliBERT-v2-wsd-5b-psfix="$V5" \
  --tasks re3d cameo_ner \
  --out analysis/data/ner_psfix.csv \
  && echo "[ok] ner" || echo "[FAIL] ner rc=$?"

echo "======== 5B pipeline finished $(date) ========"
echo WSD5B_DONE
