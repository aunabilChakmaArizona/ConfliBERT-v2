#!/usr/bin/env bash
# GPU queue stage 2: after the insightCrime cross-fit + throughput queue finishes,
# run the v1-at-its-best cells: ConfliBERT-2021 with sliding-window (chunked)
# inference on both corpora. Completes the truncate vs chunk vs native-long design.
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/queue_after_e1.log

while ! grep -aq "RC=" "$LOG"; do
  sleep 300
done
if ! grep -aq "QUEUE_DONE" "$LOG"; then
  echo "stage-1 queue did not finish cleanly; aborting windows stage" >&2
  exit 1
fi

source /root/cb2-venv/bin/activate

python -u src/eval/win_crossfit.py --corpus-type ipe \
  --model eventdata-utd/ConfliBERT-scr-uncased --model-name ConfliBERT-2021-win \
  --lr 5e-5 --bsz 16 --out analysis/data/ipe_crossfit_preds.csv

python -u src/eval/win_crossfit.py --corpus-type tsv --task insightCrime \
  --model eventdata-utd/ConfliBERT-scr-uncased --model-name ConfliBERT-2021-win \
  --lr 8e-5 --bsz 8 --out analysis/data/ic_crossfit_preds.csv

echo QUEUE2_DONE
