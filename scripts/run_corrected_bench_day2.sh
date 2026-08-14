#!/usr/bin/env bash
# Day 2 of the corrected benchmark: ModernBERT-base (the proper baseline column)
# and A3 (ConfliBERT-v2-wsd), all nine tasks. Appends to corrected_bench.csv.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
python -u scripts/corrected_bench.py \
  --models ModernBERT-base=answerdotai/ModernBERT-base \
           ConfliBERT-v2-wsd=/root/models/conflibert-v2-wsd \
  --tasks re3d IndiaPoliceEvents_docs insightCrime satp_relevant cameo_class \
          cameo_ner IndiaPoliceEvents_sents 20news BBC_News \
  --out analysis/data/corrected_bench.csv
echo DAY2_DONE
