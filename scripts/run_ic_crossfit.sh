#!/usr/bin/env bash
# E1b: cross-fitted coding of insightCrime, the generalization corpus (paper).
# Same 4 instrument configs as IPE; LRs = each model's dev-selected value from the
# corrected benchmark on insightCrime; bsz 8 per the task's repo config.
set -e
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
OUT=analysis/data/ic_crossfit_preds.csv

python -u src/eval/tsv_crossfit.py --task insightCrime \
  --model eventdata-utd/ConfliBERT-scr-uncased --model-name ConfliBERT-2021 \
  --seq-len 512 --lr 8e-5 --out $OUT

python -u src/eval/tsv_crossfit.py --task insightCrime \
  --model outputs/models/conflibert-v2-wsd --model-name ConfliBERT-v2 \
  --seq-len 512 --lr 5e-5 --out $OUT

python -u src/eval/tsv_crossfit.py --task insightCrime \
  --model outputs/models/conflibert-v2-wsd --model-name ConfliBERT-v2 \
  --seq-len 2048 --lr 5e-5 --out $OUT

python -u src/eval/tsv_crossfit.py --task insightCrime \
  --model answerdotai/ModernBERT-base --model-name ModernBERT-base \
  --seq-len 2048 --lr 8e-5 --out $OUT

echo IC_ALL_CONFIGS_DONE
