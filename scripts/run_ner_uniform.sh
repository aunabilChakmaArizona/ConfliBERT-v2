#!/usr/bin/env bash
# Uniform corrected-protocol NER benchmark for ALL ModernBERT-family variants:
# --prefix-space + LR grid extended to {5e-5, 8e-5, 1.6e-4, 2.4e-4} (1.6e-4 won at the
# old grid edge, so 2.4e-4 probes whether the peak is interior). Appends to
# ner_psfix.csv; the driver skips (seed, lr) rows already present, so the existing
# MB-base-psfix / v2-wsd-psfix rows are reused, not re-run. ConfliBERT-2021 needs no
# re-run (WordPiece is unaffected by the bug; its LR peak at 5e-5 is interior).
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/ner_uniform.log
OUT=analysis/data/ner_psfix.csv
exec > >(tee -a "$LOG") 2>&1
echo "======== uniform NER benchmark started $(date) ========"
source /root/cb2-venv/bin/activate

GRID="5e-5 8e-5 1.6e-4 2.4e-4"

# single-model variants over both NER tasks
python -u scripts/corrected_bench.py --prefix-space --lrs $GRID --out "$OUT" \
  --tasks re3d cameo_ner \
  --models ModernBERT-base-psfix=answerdotai/ModernBERT-base \
           ConfliBERT-v2-wsd-psfix=/root/models/conflibert-v2-wsd \
           ConfliBERT-v2-native-psfix=/root/models/conflibert-v2-native \
           ConfliBERT-v2-wsd-core-psfix=/root/models/conflibert-v2-wsd-core

# TAPT models are per-task
python -u scripts/corrected_bench.py --prefix-space --lrs $GRID --out "$OUT" \
  --tasks cameo_ner \
  --models ConfliBERT-v2-wsd-TAPT-psfix=/root/models/tapt/cameo_ner
python -u scripts/corrected_bench.py --prefix-space --lrs $GRID --out "$OUT" \
  --tasks re3d \
  --models ConfliBERT-v2-wsd-TAPT-psfix=/root/models/tapt/re3d

echo "======== uniform NER benchmark finished $(date) ========"
echo NER_UNIFORM_DONE
