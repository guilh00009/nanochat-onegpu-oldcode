#!/usr/bin/env bash
set -euo pipefail

# match your previous run: you trained with device_batch_size=1
DEVICE_BATCH=1

# 1) EVAL (mid)
#echo "==> eval (mid)"
#torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid

# 2) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH}

# 3) EVAL (sft)
#echo "==> eval (sft)"
#torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# 4) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
