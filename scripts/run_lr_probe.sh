#!/usr/bin/env bash
# LR-sensitivity probe: is the fixed 3e-5 fine-tune LR handicapping ModernBERT-family
# models? Runs alongside the DAPT job on spare VRAM. Seed 123 only, default protocol
# (no dev-select) so numbers are directly comparable to the logged 3e-5 runs.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
source /root/cb2-venv/bin/activate
OUT=analysis/data/lr_probe.csv
for LR in 5e-5 8e-5; do
  for TASK in re3d BBC_News; do
    echo "== probe $TASK lr=$LR $(date +%H:%M:%S) =="
    python src/eval/finetune_benchmark.py --task "$TASK" \
      --model answerdotai/ModernBERT-base --model-name "ModernBERT-base-lr$LR" \
      --repo external/ConfliBERT --out "$OUT" --seeds 123 --lr "$LR" \
      || echo "[FAIL] $TASK $LR"
  done
done
echo PROBE_DONE
