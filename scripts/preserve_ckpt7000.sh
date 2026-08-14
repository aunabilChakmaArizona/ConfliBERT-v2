#!/usr/bin/env bash
# Preserve the last stable-phase checkpoint (step 7000, before the WSD decay starts at
# step 7630) for the decay-data ablation, before save_total_limit=3 rotates it away.
# Fail-safe: only ever copies; never touches the training process. Times out after 14h.
set -u
SRC=/root/cb2_out/conflibert-v2-wsd/checkpoint-7000
DST=/root/cb2_out/stable_ckpt7000
for i in $(seq 1 84); do
  if [ -f "$SRC/model.safetensors" ] && [ -f "$SRC/trainer_state.json" ]; then
    sleep 120   # let the trainer finish writing the checkpoint
    mkdir -p "$DST"
    cp -f "$SRC"/*.json "$SRC"/*.safetensors "$DST"/ 2>/dev/null
    ls -la "$DST"
    echo "CKPT7000_PRESERVED"
    exit 0
  fi
  sleep 600
done
echo "CKPT7000_TIMEOUT"
exit 1
