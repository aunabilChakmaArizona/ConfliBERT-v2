#!/usr/bin/env bash
# Corrected augmented-tokenizer ablation (DO NOT launch alongside other GPU work; ~14h).
#
# The existing augmented model (A1) was trained with the broken cold-cosine-restart
# recipe, so "augmentation costs 1-2 F1" is confounded. This gives the augmented
# tokenizer the same corrected recipe as v2-wsd (A3), plus three fixes aimed at the
# measured failure modes:
#   1. augment the pre-decay STABLE checkpoint (not the decayed release),
#   2. FVT init with --rescale-norms (added rows measured at 0.67x norm and stuck
#      there through DAPT; short rows + tied head = low logits = no gradient),
#   3. --new-embed-warmup-steps 400 (~105M tokens): only the 3,973 new rows train
#      first, so the settled network does not adapt around garbage rows.
# Single-variable vs v2-wsd: same pack budget, schedule, and steps; only the
# tokenizer (and its supporting init/warmup) differs.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/aug_wsd_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "======== aug-WSD ablation started $(date) ========"
source /root/cb2-venv/bin/activate

STABLE=/root/models/modernbert-stable-hf
INIT=outputs/models/modernbert-stable-conflivocab
TRAIN=/root/packed/train_aug_1024
EVAL=/root/packed/eval_random_aug_1024
OUT=/root/cb2_out/conflibert-v2-aug-wsd
V2A=/root/models/conflibert-v2-aug-wsd

[ -f "$STABLE/model.safetensors" ] || { echo "[abort] missing stable ckpt"; exit 1; }
[ -f "$TRAIN/meta.json" ] || { echo "[abort] missing aug pack"; exit 1; }

# ---------------- 1) augment the STABLE checkpoint, norm-corrected FVT ----------------
python src/pretrain/augment_model.py --base "$STABLE" \
  --vocab outputs/vocab/domain_vocab.txt --out "$INIT" --rescale-norms \
  || { echo "[abort] augment failed"; exit 1; }

# guard: token ids must match the tokenizer the pack was built with
python - <<'EOF' || exit 1
from transformers import AutoTokenizer
a = AutoTokenizer.from_pretrained("outputs/models/modernbert-conflivocab-init")
b = AutoTokenizer.from_pretrained("outputs/models/modernbert-stable-conflivocab")
assert a.get_added_vocab() == b.get_added_vocab() and len(a) == len(b), \
    "added-vocab mismatch vs the tokenizer the pack was built with"
print("[guard] tokenizer identical to pack tokenizer")
EOF

# ---------------- 2) WSD CPT, new-embed warmup first ----------------
python -u src/pretrain/train_dapt.py --base "$INIT" --train "$TRAIN" --eval "$EVAL" --out "$OUT" \
  --scheduler wsd --lr 2e-4 --decay-ratio 0.20 --warmup-ratio 0.03 \
  --new-embed-warmup-steps 400 \
  --mlm-prob 0.30 --bsz 8 --accum 32 --epochs 1 --weight-decay 0.01 \
  --save-steps 1000 --eval-steps 1000 --log-steps 25 --mem-fraction 0 \
  || { echo "[abort] training failed"; exit 1; }

# ---------------- 3) finalize ----------------
mkdir -p "$V2A"
if [ -f "$OUT/model.safetensors" ]; then SRC="$OUT"; else SRC=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
cp -f "$SRC"/*.json "$V2A"/ 2>/dev/null
cp -f "$SRC"/*.safetensors "$V2A"/ 2>/dev/null
cp -f "$SRC"/tokenizer* "$V2A"/ 2>/dev/null
mkdir -p outputs/models/conflibert-v2-aug-wsd
cp -f "$V2A"/* outputs/models/conflibert-v2-aug-wsd/ 2>/dev/null
cp -f "$OUT/metrics.csv" analysis/data/dapt_metrics_aug_wsd.csv 2>/dev/null

# ---------------- 4) corrected-protocol benchmark (NER with --prefix-space) ----------------
python -u scripts/corrected_bench.py \
  --models ConfliBERT-v2-aug-wsd="$V2A" \
  --tasks satp_relevant IndiaPoliceEvents_sents IndiaPoliceEvents_docs insightCrime \
          cameo_class BBC_News 20news \
  --out analysis/data/corrected_bench.csv
python -u scripts/corrected_bench.py \
  --models ConfliBERT-v2-aug-wsd-psfix="$V2A" \
  --tasks re3d cameo_ner --prefix-space \
  --lrs 5e-5 8e-5 1.6e-4 2.4e-4 \
  --out analysis/data/ner_psfix.csv

echo "======== aug-WSD ablation finished $(date) ========"
echo AUG_WSD_DONE
