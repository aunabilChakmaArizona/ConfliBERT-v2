#!/usr/bin/env bash
# NER diagnosis A/B: does forcing add_prefix_space=True on ByteLevel BPE (so pre-split
# NER words tokenize identically to running text) close the ModernBERT-family NER gap,
# and does the LR grid need to extend past 8e-5? Controls included so the two effects
# are separable. All runs --dev-select, same protocol as corrected_bench.csv.
set -u
cd /mnt/f/Confli_2/Corpus/conflibert-v2
LOG=outputs/ner_psfix.log
OUT=analysis/data/ner_psfix.csv
exec > >(tee -a "$LOG") 2>&1
echo "======== NER psfix A/B started $(date) ========"
source /root/cb2-venv/bin/activate

MB=answerdotai/ModernBERT-base
V2W=/root/models/conflibert-v2-wsd
[ -f "$V2W/model.safetensors" ] || V2W=outputs/models/conflibert-v2-wsd

run () {  # task model name lr extra...
  local task=$1 model=$2 name=$3 lr=$4; shift 4
  python -u src/eval/finetune_benchmark.py --task "$task" --model "$model" \
    --model-name "$name" --out "$OUT" --lr "$lr" --dev-select \
    --seeds 123 124 125 "$@" \
    && echo "[ok] $task $name lr=$lr" || echo "[FAIL] $task $name lr=$lr"
}

# --- B: fix at the incumbent best LR (isolates the prefix-space effect) ---
run cameo_ner "$MB" ModernBERT-base-psfix 8e-5 --prefix-space
run re3d      "$MB" ModernBERT-base-psfix 8e-5 --prefix-space

# --- C: fix + extended LR (tests whether the grid was truncated) ---
run cameo_ner "$MB" ModernBERT-base-psfix 1.6e-4 --prefix-space
run re3d      "$MB" ModernBERT-base-psfix 1.6e-4 --prefix-space

# --- A: control, extended LR WITHOUT the fix (attributes any gain cleanly) ---
run cameo_ner "$MB" ModernBERT-base-hiLR 1.6e-4
run re3d      "$MB" ModernBERT-base-hiLR 1.6e-4

# --- D: the money runs, v2-wsd with the fix ---
run cameo_ner "$V2W" ConfliBERT-v2-wsd-psfix 8e-5   --prefix-space
run cameo_ner "$V2W" ConfliBERT-v2-wsd-psfix 1.6e-4 --prefix-space
run re3d      "$V2W" ConfliBERT-v2-wsd-psfix 8e-5   --prefix-space
run re3d      "$V2W" ConfliBERT-v2-wsd-psfix 1.6e-4 --prefix-space

echo "======== NER psfix A/B finished $(date) ========"
echo PSFIX_DONE
