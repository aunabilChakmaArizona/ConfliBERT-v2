#!/usr/bin/env bash
# One-shot status of the current pipeline (Run A: aug-WSD ablation -> Run B: 5B).
# Run from Windows PowerShell:
#   wsl -d Ubuntu-22.04 -u root bash /mnt/f/Confli_2/Corpus/conflibert-v2/scripts/status.sh
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
echo "==================== GPU ===================="
nvidia-smi --query-gpu=utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader
echo
echo "============ Run A: augmented-tokenizer ablation (14h) ============"
if [ -f outputs/aug_wsd_pipeline.log ]; then
  tr "\r" "\n" < outputs/aug_wsd_pipeline.log | grep -aoE "[0-9]+/9539 \[[^]]*\]" | tail -1
  grep -aE "^======== |\[abort\]|\[ok\]|\[FAIL\]|AUG_WSD_DONE" outputs/aug_wsd_pipeline.log | tail -4
  M=/root/cb2_out/conflibert-v2-aug-wsd/metrics.csv
  if [ -f "$M" ]; then
    echo "latest metrics (step, tokens, train_loss, eval_loss, ppl):"
    awk -F',' 'NR>1 && $3!="" {last=$1","$2","$3","$4","$5} END {print "  "last}' "$M"
    awk -F',' 'NR>1 && $4!="" {last=$1": eval_loss="$4" ppl="$5} END {if (last) print "  last eval  "last}' "$M"
  fi
else
  echo "  not started"
fi
echo
echo "============ Run B: 5B extension (starts after A) ============"
if [ -f outputs/wsd5b_pipeline.log ]; then
  tr "\r" "\n" < outputs/wsd5b_pipeline.log | grep -aoE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1
  grep -aE "^======== |\[abort\]|\[guard\]|WSD5B_DONE" outputs/wsd5b_pipeline.log | tail -3
else
  echo "  queued, not started"
fi
echo
echo "============ benchmark rows ============"
echo "ner_psfix.csv rows:       $(grep -c "" analysis/data/ner_psfix.csv 2>/dev/null || echo 0)"
echo "corrected_bench.csv rows: $(grep -c "" analysis/data/corrected_bench.csv 2>/dev/null || echo 0)"
