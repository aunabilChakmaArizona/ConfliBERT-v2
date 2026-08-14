#!/usr/bin/env bash
# GPU queue stage 3: after the windows stage, run v2 at 4096 tokens on
# insightCrime (covers 99.5% of its documents; bsz 4 for the 20GB WSL ceiling).
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/queue2_windows.log

while ! grep -aq "RC=" "$LOG"; do
  sleep 300
done
if ! grep -aq "QUEUE2_DONE" "$LOG"; then
  echo "stage-2 queue did not finish cleanly; aborting stage 3" >&2
  exit 1
fi

source /root/cb2-venv/bin/activate
python -u src/eval/tsv_crossfit.py --task insightCrime \
  --model outputs/models/conflibert-v2-wsd --model-name ConfliBERT-v2 \
  --seq-len 4096 --lr 5e-5 --bsz 4 --out analysis/data/ic_crossfit_preds.csv

echo QUEUE3_DONE
