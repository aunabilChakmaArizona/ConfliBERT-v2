#!/usr/bin/env bash
# GPU job queue: wait for the IPE cross-fit run (E1) to finish, then run the
# insightCrime cross-fit (E1b) and the throughput benchmark (E4).
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/ipe_crossfit.log

while ! grep -aq "RC=" "$LOG"; do
  sleep 300
done
if ! grep -aq "ALL_CONFIGS_DONE" "$LOG"; then
  echo "E1 did not finish cleanly (no ALL_CONFIGS_DONE); aborting queue" >&2
  exit 1
fi

bash scripts/run_ic_crossfit.sh

source /root/cb2-venv/bin/activate
python -u src/eval/throughput_bench.py --out analysis/data/throughput.csv

echo QUEUE_DONE
