#!/usr/bin/env bash
set -euo pipefail

# Max-fit nanochat for ~16GB RAM / 6GB VRAM
# tokenizer -> base -> loss/eval -> mid (repo-native) -> sft (repo-native) -> eval -> report

export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
mkdir -p "${NANOCHAT_BASE_DIR}"

# ---------- 1) Bootstrap ------------------------------------------------------
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "${HOME}/.cargo/bin/uv" ]; then export PATH="${HOME}/.cargo/bin:${PATH}"; fi
fi
[ -d ".venv" ] || uv venv

HAVE_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAVE_NVIDIA=1
fi

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  uv sync --extra cuda
else
  uv sync --extra cpu
fi

source .venv/bin/activate

: "${WANDB_RUN:=maxfit}"
python -m nanochat.report reset

# ---------- 2) Rust + rustbpe -------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck source=/dev/null
source "${HOME}/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# ---------- 3) Eval bundle + identity ----------------------------------------
EVAL_BUNDLE_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
if [ ! -d "${NANOCHAT_BASE_DIR}/eval_bundle" ]; then
  curl -L -o eval_bundle.zip "${EVAL_BUNDLE_URL}"
  unzip -q eval_bundle.zip && rm eval_bundle.zip
  mv eval_bundle "${NANOCHAT_BASE_DIR}"
fi
curl -L -o "${NANOCHAT_BASE_DIR}/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# ---------- 4) Tokenizer & data ----------------------------------------------
# (scaled for 16GB RAM / 6GB VRAM machine)
TOK_MAX_CHARS=500000000     # 500M chars
TOK_SHARDS=12
ALL_SHARDS=64

python -m nanochat.dataset -n "${TOK_SHARDS}"
python -m nanochat.dataset -n "${ALL_SHARDS}" &

python -m scripts.tok_train --max_chars="${TOK_MAX_CHARS}"
python -m scripts.tok_eval

# ---------- 5) Probe for max model that fits ---------------------------------
DEVICE_BATCH_SIZE=1
CANDIDATE_DEPTHS=(12 11 10 9 8 7 6)
CANDIDATE_SEQLENS=(1536 1408 1280 1152 1024 896 768 512)

BEST_DEPTH=""
BEST_SEQLEN=""

try_probe () {
  local depth="$1"
  local seqlen="$2"
  if [ "${HAVE_NVIDIA}" -eq 1 ]; then
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
      --depth="${depth}" \
      --max_seq_len="${seqlen}" \
      --device_batch_size="${DEVICE_BATCH_SIZE}" \
      --total_batch_size=4096 \
      --num_iterations=2 \
      --eval_every=999999 \
      --sample_every=999999 \
      --core_metric_every=999999 \
      --eval_tokens=2048 \
      >/dev/null 2>&1
  else
    python -m scripts.base_train \
      --depth="${depth}" \
      --max_seq_len="${seqlen}" \
      --device_batch_size="${DEVICE_BATCH_SIZE}" \
      --total_batch_size=2048 \
      --num_iterations=2 \
      --eval_every=999999 \
      --sample_every=999999 \
      --core_metric_every=999999 \
      --eval_tokens=2048 \
      >/dev/null 2>&1
  fi
}

echo "=== Probing maximum depth/seq-len that fit memory ==="
for depth in "${CANDIDATE_DEPTHS[@]}"; do
  for seqlen in "${CANDIDATE_SEQLENS[@]}"; do
    echo "Probe depth=${depth} seq_len=${seqlen} ..."
    if try_probe "${depth}" "${seqlen}"; then
      BEST_DEPTH="${depth}"
      BEST_SEQLEN="${seqlen}"
      echo "  -> OK"
      break
    else
      echo "  -> OOM / fail"
    fi
  done
  [ -n "${BEST_DEPTH}" ] && break
done

if [ -z "${BEST_DEPTH}" ]; then
  echo "Could not fit even the smallest candidate; falling back to depth=6, seq_len=512"
  BEST_DEPTH=6
  BEST_SEQLEN=512
fi

# ---------- 6) Global knobs ---------------------------------------------------
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  TOTAL_BATCH_TOKENS=16384   # base effective batch (via grad-accum)
else
  TOTAL_BATCH_TOKENS=8192
fi

BASE_ITERS=3500
EVAL_EVERY=250
EVAL_TOKENS=4096

if [ "${BEST_SEQLEN}" -le 768 ]; then
  BASE_ITERS=4500
fi

echo "=== FINAL CONFIG (6GB) ==="
echo "GPU:           ${HAVE_NVIDIA}"
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
echo "DEPTH:         ${BEST_DEPTH}"
echo "SEQ_LEN:       ${BEST_SEQLEN}"
echo "DEVICE_BATCH:  ${DEVICE_BATCH_SIZE}"
echo "TOTAL_BATCH:   ${TOTAL_BATCH_TOKENS}"
echo "BASE_ITERS:    ${BASE_ITERS}"
echo "=================================="

# ---------- 7) Base training --------------------------------------------------
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --depth="${BEST_DEPTH}" \
    --max_seq_len="${BEST_SEQLEN}" \
    --device_batch_size="${DEVICE_BATCH_SIZE}" \
    --total_batch_size="${TOTAL_BATCH_TOKENS}" \
    --eval_every="${EVAL_EVERY}" \
    --eval_tokens="${EVAL_TOKENS}" \
    --core_metric_every="${EVAL_EVERY}" \
    --core_metric_max_per_task=12 \
    --sample_every="${EVAL_EVERY}" \
    --num_iterations="${BASE_ITERS}"
else
  python -m scripts.base_train \
    --depth="${BEST_DEPTH}" \
    --max_seq_len="${BEST_SEQLEN}" \
    --device_batch_size="${DEVICE_BATCH_SIZE}" \
    --total_batch_size="${TOTAL_BATCH_TOKENS}" \
    --eval_every="${EVAL_EVERY}" \
    --eval_tokens="${EVAL_TOKENS}" \
    --core_metric_every="${EVAL_EVERY}" \
    --core_metric_max_per_task=12 \
    --sample_every="${EVAL_EVERY}" \
    --num_iterations="${BASE_ITERS}"
fi

# ---------- 8) base_loss / base_eval -----------------------------------------
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.base_loss -- --device_batch_size="${DEVICE_BATCH_SIZE}" --split_tokens="${EVAL_TOKENS}"
  torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --max-per-task=16
else
  python -m scripts.base_loss --device_batch_size="${DEVICE_BATCH_SIZE}" --split_tokens="${EVAL_TOKENS}"
  python -m scripts.base_eval --max-per-task=16
fi

# ======================================================================
# FROM HERE ON: use your working "runcpucontinue" style
# ======================================================================

# 9) MID-TRAIN (repo-native)
echo "==> mid-train (repo-native args only)"
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --device_batch_size=${DEVICE_BATCH_SIZE}
else
  python -m scripts.mid_train --device_batch_size=${DEVICE_BATCH_SIZE}
fi

# 10) EVAL (mid)
echo "==> eval (mid)"
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid
else
  python -m scripts.chat_eval -i mid
fi

# 11) SFT
echo "==> sft (same device_batch_size)"
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH_SIZE}
else
  python -m scripts.chat_sft --device_batch_size=${DEVICE_BATCH_SIZE}
fi

# 12) EVAL (sft)
echo "==> eval (sft)"
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
else
  python -m scripts.chat_eval -i sft
fi

# 13) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
