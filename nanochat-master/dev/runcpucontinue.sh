#!/usr/bin/env bash
set -euo pipefail

# match your previous run: you trained with device_batch_size=1
DEVICE_BATCH=1

# 1) MID-TRAIN
echo "==> mid-train (repo-native args only)"
torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --device_batch_size=${DEVICE_BATCH}

# 2) EVAL (mid)
echo "==> eval (mid)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid

# 3) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH}

# 4) EVAL (sft)
echo "==> eval (sft)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# 5) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
