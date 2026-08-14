#!/usr/bin/env bash
#SBATCH --job-name=cb2-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=logs/pack-%j.out
#
# Stage 2: tokenize + pack the Parquet corpus into fixed-length MLM blocks.
# CPU-only; runs on the cpu partition so it never burns GPU allocation.
# Submit from the repo root via: bash hpc/submit.sh hpc/pack_corpus.sbatch
#
# Tunables (export before submitting, or edit hpc/paths.env):
#   CB2_MAX_TOKENS  token budget for the train pack (default 0 = pack everything)
#   CB2_PACK_TAG    suffix for the pack dir name (default "hpc")
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/paths.env
source "$CB2_VENV/bin/activate"

TAG="${CB2_PACK_TAG:-hpc}"
MAXTOK="${CB2_MAX_TOKENS:-0}"
TRAIN_PACK="$CB2_PACKED/train_${CB2_SEQLEN}_${TAG}"
EVAL_PACK="$CB2_PACKED/eval_random_${CB2_SEQLEN}_${TAG}"

echo "== pack train -> $TRAIN_PACK (budget: $MAXTOK tokens, 0=all) =="
python src/data/pack_tokens.py \
  --corpus "$CB2_CORPUS" --split train --seqlen "$CB2_SEQLEN" \
  --tokenizer "$CB2_TOKENIZER" \
  --source-weights "$CB2_SOURCE_WEIGHTS" \
  --max-tokens "$MAXTOK" --dedup --seed 7 \
  --out "$TRAIN_PACK"

echo "== pack eval_random -> $EVAL_PACK =="
python src/data/pack_tokens.py \
  --corpus "$CB2_CORPUS" --split eval_random --seqlen "$CB2_SEQLEN" \
  --tokenizer "$CB2_TOKENIZER" \
  --dedup --seed 7 \
  --out "$EVAL_PACK"

echo "PACK_JOB_DONE train=$TRAIN_PACK eval=$EVAL_PACK"
