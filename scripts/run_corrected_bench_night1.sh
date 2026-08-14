#!/usr/bin/env bash
# Night 1 of the corrected benchmark: THE head-to-head the paper is about,
# ConfliBERT-2021 vs ConfliBERT-v2-native, all 9 tasks, uniform LR grid + dev-select.
# Runs on spare VRAM alongside the R1 DAPT job. v2-wsd joins tomorrow when R1 lands.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
python -u scripts/corrected_bench.py \
  --models ConfliBERT-2021=snowood1/ConfliBERT-scr-uncased \
           ConfliBERT-v2-native=/root/models/conflibert-v2-native \
  --tasks re3d IndiaPoliceEvents_docs insightCrime satp_relevant cameo_class \
          cameo_ner IndiaPoliceEvents_sents 20news BBC_News \
  --out analysis/data/corrected_bench.csv
echo NIGHT1_DONE
