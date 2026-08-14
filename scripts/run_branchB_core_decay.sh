#!/usr/bin/env bash
# Branch B of the annealing-data ablation: repack a genuinely conflict-core
# annealing set (Organization + UTDstory ONLY; the previous attempt was ~90%
# News by availability), then run the 1,907-step decay phase from the preserved
# stable-trunk checkpoint (step 7000), then finalize and eval.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/branchB_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "======== branch B pipeline started $(date) ========"
source /root/cb2-venv/bin/activate

TRUNK=/root/cb2_out/stable_ckpt7000
CORE=/root/packed/train_core2_1024
EVAL=/root/packed/eval_random_native_1024
OUT=/root/cb2_out/conflibert-v2-wsd-core
V2C=/root/models/conflibert-v2-wsd-core

[ -f "$TRUNK/model.safetensors" ] || { echo "[abort] missing trunk"; exit 1; }

# ---- 1) conflict-core pack: Organization + UTDstory only ----
echo "======== [pack] core2 $(date) ========"
python src/data/pack_tokens.py --corpus outputs/corpus/parquet --split train --seqlen 1024 \
  --tokenizer answerdotai/ModernBERT-base \
  --source-weights "Organization=1.0,UTDstory=1.0" \
  --max-tokens 550000000 --dedup --seed 7 --out "$CORE" \
  || { echo "[abort] core pack failed"; exit 1; }
NTOK=$(python -c "import json;print(json.load(open('$CORE/meta.json'))['n_tokens'])" 2>/dev/null || echo 0)
echo "[guard] core2 n_tokens=$NTOK"
[ "$NTOK" -ge 500000000 ] || { echo "[abort] core pack too small ($NTOK)"; exit 1; }

# ---- 2) decay-only training from the stable trunk ----
# 1,907 steps of linear decay 2e-4 -> 0, matching branch A's decay length exactly.
echo "======== [train] decay on conflict core $(date) ========"
python -u src/pretrain/train_dapt.py --base "$TRUNK" --train "$CORE" --eval "$EVAL" --out "$OUT" \
  --scheduler wsd --lr 2e-4 --warmup-ratio 0.0 --decay-ratio 1.0 --max-steps 1907 \
  --mlm-prob 0.30 --bsz 8 --accum 32 --epochs 1 --weight-decay 0.01 \
  --save-steps 1000 --eval-steps 250 --log-steps 25 --mem-fraction 0 \
  || { echo "[abort] decay training failed"; exit 1; }

# ---- 3) finalize ----
echo "======== [finalize] $(date) ========"
mkdir -p "$V2C"
if [ -f "$OUT/model.safetensors" ]; then SRC="$OUT"; else SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
cp -f "$SRC"/*.json "$V2C"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V2C"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V2C"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2-wsd-core
cp -f "$V2C"/* outputs/models/conflibert-v2-wsd-core/ 2>/dev/null
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics_wsd_core.csv 2>/dev/null

# ---- 4) evals: intrinsic + corrected-protocol benchmark ----
echo "======== [eval] pseudo-PPL core branch $(date) ========"
python src/eval/pseudo_ppl.py --eval "$EVAL" --models base=answerdotai/ModernBERT-base v2wsdcore="$V2C" \
  --tokenizer answerdotai/ModernBERT-base --out analysis/data/pseudo_ppl_wsd_core.csv \
  && echo "[ok] ppl" || echo "[FAIL] ppl rc=$?"

echo "======== [eval] corrected bench core branch $(date) ========"
python -u scripts/corrected_bench.py \
  --models ConfliBERT-v2-wsd-core="$V2C" \
  --tasks re3d IndiaPoliceEvents_docs insightCrime satp_relevant cameo_class \
          cameo_ner IndiaPoliceEvents_sents 20news BBC_News \
  --out analysis/data/corrected_bench.csv \
  && echo "[ok] corrected bench" || echo "[FAIL] corrected bench rc=$?"

echo "======== branch B pipeline finished $(date) ========"
