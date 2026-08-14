#!/usr/bin/env bash
# GPU queue stage 4: the CrisisWatch trend-assessment cells (the judgment task).
# Waits for stage 3 (v2@4096 insightCrime), then runs four instrument configs:
# the 2021 model on the current entry (fits its window), the 2021 model sliced
# over the full history input, and the long-context models reading it whole.
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/queue3_ic4096.log

while ! grep -aq "RC=" "$LOG"; do
  sleep 300
done
if ! grep -aq "QUEUE3_DONE" "$LOG"; then
  echo "stage-3 queue did not finish cleanly; aborting stage 4" >&2
  exit 1
fi

source /root/cb2-venv/bin/activate
OUT=analysis/data/cw_preds.csv

python -u src/eval/cw_finetune.py --model eventdata-utd/ConfliBERT-scr-uncased \
  --model-name ConfliBERT-2021 --input-mode current --seq-len 512 --lr 3e-5 --out $OUT

python -u src/eval/cw_finetune.py --model eventdata-utd/ConfliBERT-scr-uncased \
  --model-name ConfliBERT-2021 --input-mode full --seq-len 512 --window-infer \
  --lr 3e-5 --out $OUT

python -u src/eval/cw_finetune.py --model outputs/models/conflibert-v2-wsd \
  --model-name ConfliBERT-v2 --input-mode full --seq-len 2048 --lr 5e-5 --out $OUT

python -u src/eval/cw_finetune.py --model answerdotai/ModernBERT-base \
  --model-name ModernBERT-base --input-mode full --seq-len 2048 --lr 5e-5 --out $OUT

echo QUEUE4_DONE
