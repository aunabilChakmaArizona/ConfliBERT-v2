#!/usr/bin/env bash
# Submit wrapper: sources the path contract and picks the right Slurm account.
# Always submit from the repo root on Delta:
#
#   bash hpc/submit.sh hpc/pack_corpus.sbatch
#   bash hpc/submit.sh hpc/pretrain.sbatch
#   bash hpc/submit.sh hpc/eval.sbatch /scratch/.../models/conflibert-v2-hpc
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/paths.env
script="$1"; shift || true
case "$script" in
  *pack*) acct="$CB2_ACCOUNT_CPU" ;;
  *)      acct="$CB2_ACCOUNT_GPU" ;;
esac
mkdir -p logs
sbatch --account="$acct" --export=ALL "$script" "$@"
