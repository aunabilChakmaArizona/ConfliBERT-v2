#!/usr/bin/env bash
#SBATCH --job-name=cb2-dapt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=208g
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/dapt-%j.out
#
# Stage 3: DAPT (continued MLM) on one Delta A100x4 node via torchrun DDP.
# Submit from the repo root via: bash hpc/submit.sh hpc/pretrain.sbatch
#
# Resume-safe: --auto-resume picks up the last checkpoint in the run dir, so if
# the job hits the wall clock or is preempted, JUST SUBMIT IT AGAIN unchanged.
#
# Tunables (export before submitting; defaults reproduce the pilot v2-wsd recipe
# at the same global batch of 256 blocks/step = ~262k tokens/step @ seqlen 1024):
#   CB2_RUN        run name                     (default conflibert-v2-hpc)
#   CB2_BASE       base model or local ckpt dir (default answerdotai/ModernBERT-base;
#                  for ModernBERT-large also set CB2_BSZ=8, same packs work)
#   CB2_PACK_TAG   which packed set to train on (default "hpc", see pack_corpus.sbatch)
#   CB2_LR         peak learning rate           (default 2e-4, WSD)
#   CB2_SCHED      cosine | wsd                 (default wsd)
#   CB2_BSZ        per-device batch size        (default 16; A100-40G @ seqlen 1024)
#   CB2_GLOBAL     global blocks per opt step   (default 256)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/paths.env
source "$CB2_VENV/bin/activate"

RUN="${CB2_RUN:-conflibert-v2-hpc}"
BASE="${CB2_BASE:-answerdotai/ModernBERT-base}"
TAG="${CB2_PACK_TAG:-hpc}"
LR="${CB2_LR:-2e-4}"
SCHED="${CB2_SCHED:-wsd}"
BSZ="${CB2_BSZ:-16}"
GLOBAL="${CB2_GLOBAL:-256}"

TRAIN_PACK="$CB2_PACKED/train_${CB2_SEQLEN}_${TAG}"
EVAL_PACK="$CB2_PACKED/eval_random_${CB2_SEQLEN}_${TAG}"
OUT="$CB2_OUT/$RUN"

NGPU="$SLURM_GPUS_ON_NODE"
ACCUM=$(( GLOBAL / (BSZ * NGPU) ))
[ "$ACCUM" -ge 1 ] || { echo "ABORT: CB2_GLOBAL < BSZ*NGPU"; exit 1; }
[ -f "$TRAIN_PACK/meta.json" ] || { echo "ABORT: no train pack at $TRAIN_PACK (run pack_corpus.sbatch first)"; exit 1; }

export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NGPU ))
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "== $RUN: base=$BASE lr=$LR sched=$SCHED gpus=$NGPU bsz=$BSZ accum=$ACCUM (global $GLOBAL blocks/step) =="

torchrun --standalone --nproc_per_node "$NGPU" src/pretrain/train_dapt.py \
  --base "$BASE" \
  --train "$TRAIN_PACK" --eval "$EVAL_PACK" \
  --out "$OUT" \
  --scheduler "$SCHED" --lr "$LR" --decay-ratio 0.20 --warmup-ratio 0.03 \
  --mlm-prob 0.30 --bsz "$BSZ" --accum "$ACCUM" --epochs 1 --weight-decay 0.01 \
  --save-steps 500 --eval-steps 1000 --log-steps 25 \
  --mem-fraction 0 --tb-root "$CB2_SCRATCH/tb" \
  --auto-resume

# finalize: copy the exported model (not the checkpoints) to CB2_MODELS
if [ -f "$OUT/model.safetensors" ]; then
  mkdir -p "$CB2_MODELS/$RUN"
  cp -f "$OUT"/*.json "$OUT"/*.safetensors "$OUT"/tokenizer* "$CB2_MODELS/$RUN"/ 2>/dev/null || true
  cp -f "$OUT/metrics.csv" "$CB2_MODELS/$RUN/metrics.csv" 2>/dev/null || true
  echo "FINALIZED -> $CB2_MODELS/$RUN  (remember: /scratch is purged -- sync finished models to /projects or the Hub)"
fi
echo "DAPT_JOB_DONE"
