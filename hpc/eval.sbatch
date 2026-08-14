#!/usr/bin/env bash
#SBATCH --job-name=cb2-eval
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-%j.out
#
# Stage 4: evaluate a finalized model -- intrinsic pseudo-PPL vs the base model,
# then the 9-task downstream benchmark (classification + NER with the
# prefix-space fix). Single GPU is enough; each fine-tune is small.
#
# Usage (from the repo root):
#   bash hpc/submit.sh hpc/eval.sbatch <model_dir> [model_name]
# e.g.
#   bash hpc/submit.sh hpc/eval.sbatch "$CB2_MODELS/conflibert-v2-hpc" ConfliBERT-v2-hpc
#
# Prereq: benchmark task datasets present (see docs/PIPELINE.md "External data").
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/paths.env
source "$CB2_VENV/bin/activate"

MODEL_DIR="${1:?usage: eval.sbatch <model_dir> [model_name]}"
NAME="${2:-$(basename "$MODEL_DIR")}"
TAG="${CB2_PACK_TAG:-hpc}"
EVAL_PACK="$CB2_PACKED/eval_random_${CB2_SEQLEN}_${TAG}"
mkdir -p analysis/data

# fine-tune scratch on node-local storage, not Lustre (thousands of tiny writes)
export CB2_TMP="${SLURM_TMPDIR:-/tmp}/cb2_ft_tmp_${SLURM_JOB_ID}"

[ -d external/ConfliBERT/data ] || {
  echo "ABORT: external/ConfliBERT missing -- clone the benchmark repo first:"
  echo "  git clone https://github.com/eventdata/ConfliBERT external/ConfliBERT"
  exit 1
}

echo "== [1/3] pseudo-PPL: $NAME vs base =="
python src/eval/pseudo_ppl.py --eval "$EVAL_PACK" \
  --models base="$CB2_TOKENIZER" "$NAME=$MODEL_DIR" \
  --tokenizer "$CB2_TOKENIZER" \
  --out "analysis/data/pseudo_ppl_${NAME}.csv" \
  && echo "[ok] ppl" || echo "[FAIL] ppl rc=$?"

echo "== [2/3] downstream benchmark: classification tasks =="
python -u scripts/corrected_bench.py \
  --models "$NAME=$MODEL_DIR" \
  --tasks satp_relevant IndiaPoliceEvents_sents IndiaPoliceEvents_docs insightCrime \
          cameo_class BBC_News 20news \
  --out analysis/data/corrected_bench.csv \
  && echo "[ok] bench" || echo "[FAIL] bench rc=$?"

echo "== [3/3] NER (prefix-space fix, extended LR grid) =="
python -u scripts/corrected_bench.py --prefix-space --lrs 5e-5 8e-5 1.6e-4 2.4e-4 \
  --models "${NAME}-psfix=$MODEL_DIR" \
  --tasks re3d cameo_ner \
  --out analysis/data/ner_psfix.csv \
  && echo "[ok] ner" || echo "[FAIL] ner rc=$?"

echo "EVAL_JOB_DONE model=$NAME"
