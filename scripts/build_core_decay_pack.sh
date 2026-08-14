#!/usr/bin/env bash
# Build the conflict-core annealing pack for the decay-data ablation (branch B):
# Organization + UTDstory at full weight, thin slice of News, NO Gigaword/Wikipedia.
# 550M tokens covers the 1,907-step decay phase (500M) with margin. CPU-only.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
python src/data/pack_tokens.py --corpus outputs/corpus/parquet --split train --seqlen 1024 \
  --tokenizer answerdotai/ModernBERT-base \
  --source-weights "Organization=1.0,UTDstory=1.0,News=0.2" \
  --max-tokens 550000000 --dedup --seed 7 --out /root/packed/train_core_decay_1024
echo CORE_PACK_DONE
