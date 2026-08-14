#!/usr/bin/env bash
# E1: cross-fitted event coding of the IndiaPoliceEvents corpus (paper experiment).
# 4 instrument configs x 3 seeds x 5 folds = 60 fine-tunes; per-doc probs appended to
# analysis/data/ipe_crossfit_preds.csv. 512-configs first for early signal.
set -e
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
OUT=analysis/data/ipe_crossfit_preds.csv

python -u src/eval/ipe_crossfit.py --model eventdata-utd/ConfliBERT-scr-uncased \
  --model-name ConfliBERT-2021 --seq-len 512 --lr 5e-5 --out $OUT

python -u src/eval/ipe_crossfit.py --model outputs/models/conflibert-v2-wsd \
  --model-name ConfliBERT-v2 --seq-len 512 --lr 8e-5 --out $OUT

python -u src/eval/ipe_crossfit.py --model outputs/models/conflibert-v2-wsd \
  --model-name ConfliBERT-v2 --seq-len 2048 --lr 8e-5 --out $OUT

python -u src/eval/ipe_crossfit.py --model answerdotai/ModernBERT-base \
  --model-name ModernBERT-base --seq-len 2048 --lr 3e-5 --out $OUT

echo ALL_CONFIGS_DONE
