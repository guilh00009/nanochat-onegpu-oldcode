#!/usr/bin/env bash
set -euo pipefail

# Max-fit nanochat for ~40GB VRAM / ~110GB RAM
# tokenizer -> base -> loss/eval -> mid (repo-native) -> sft (repo-native) -> eval -> report
# IMPORTANT: seq_len will NEVER be < 2048. If 2048 doesn't fit, we abort.

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

: "${WANDB_RUN:=big40}"
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

# ---------- 4) Tokenizer & data (big RAM) ------------------------------------
TOK_MAX_CHARS=2000000000     # 2B chars
TOK_SHARDS=32
ALL_SHARDS=800               # like the full run

python -m nanochat.dataset -n "${TOK_SHARDS}"
python -m nanochat.dataset -n "${ALL_SHARDS}" &

python -m scripts.tok_train --max_chars="${TOK_MAX_CHARS}"
python -m scripts.tok_eval

# ---------- 5) Probe for max model (your order, min seq_len=2048) ------------
# device batch tries: force smallest batch for stability
CANDIDATE_DEVICE_BATCHES=(1)
# depth tries: start as high as 3× the original depth 32 and step down
CANDIDATE_DEPTHS=(96 92 88 84 80 76 72 68 64 60 56 52 48 44 40 36 32 30 28 26 24 22 20 18 16 14 12 10 8)
# seq_len tries: NEVER < 2048
CANDIDATE_SEQLENS=(4096 3584 3072 2560 2048)

BEST_DBSZ=""
BEST_DEPTH=""
BEST_SEQLEN=""

try_probe () {
  local db="$1"
  local depth="$2"
  local seqlen="$3"
  torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --depth="${depth}" \
    --max_seq_len="${seqlen}" \
    --device_batch_size="${db}" \
    --total_batch_size=32768 \
    --num_iterations=2 \
    --eval_every=999999 \
    --sample_every=999999 \
    --core_metric_every=999999 \
    --eval_tokens=2048 \
    >/dev/null 2>&1
}

echo "=== Probing maximum depth/seq-len/dbsz that fit memory (40GB) ==="
if [ "${HAVE_NVIDIA}" -ne 1 ]; then
  echo "No NVIDIA GPU detected; this script expects a 40GB GPU. Exiting."
  exit 1
fi

for db in "${CANDIDATE_DEVICE_BATCHES[@]}"; do
  for depth in "${CANDIDATE_DEPTHS[@]}"; do
    for seqlen in "${CANDIDATE_SEQLENS[@]}"; do
      echo "Probe dbsz=${db} depth=${depth} seq_len=${seqlen} ..."
      if try_probe "${db}" "${depth}" "${seqlen}"; then
        BEST_DBSZ="${db}"
        BEST_DEPTH="${depth}"
        BEST_SEQLEN="${seqlen}"
        echo "  -> OK"
        break 2
      else
        echo "  -> OOM / fail"
      fi
    done
  done
  [ -n "${BEST_DEPTH}" ] && break
done

# If we STILL don't have a model here, it means even depth=8 at seq_len=2048 didn't fit.
if [ -z "${BEST_DEPTH}" ]; then
  echo "FATAL: could not fit even depth=8 at seq_len=2048 on this GPU."
  echo "You asked to never go below 2048, so we won't auto-lower it."
  exit 1
fi

# ---------- 6) Global knobs for 40GB -----------------------------------------
BASE_TOTAL_BATCH=262144     # 256k tokens
EVAL_EVERY=250
EVAL_TOKENS=4096
BASE_ITERS=8000             # long run; adjust if needed

echo "=== FINAL CONFIG (40GB, seq_len >= 2048) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "DEPTH:          ${BEST_DEPTH}"
echo "SEQ_LEN:        ${BEST_SEQLEN}"
echo "DEVICE_BATCH:   ${BEST_DBSZ}"
echo "BASE_TOTAL:     ${BASE_TOTAL_BATCH}"
echo "BASE_ITERS:     ${BASE_ITERS}"
echo "====================================================="

# ---------- 7) Base training --------------------------------------------------
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
  --depth="${BEST_DEPTH}" \
  --max_seq_len="${BEST_SEQLEN}" \
  --device_batch_size="${BEST_DBSZ}" \
  --total_batch_size="${BASE_TOTAL_BATCH}" \
  --eval_every="${EVAL_EVERY}" \
  --eval_tokens="${EVAL_TOKENS}" \
  --core_metric_every="${EVAL_EVERY}" \
  --core_metric_max_per_task=12 \
  --sample_every="${EVAL_EVERY}" \
  --num_iterations="${BASE_ITERS}"

# ---------- 8) base_loss / base_eval -----------------------------------------
torchrun --standalone --nproc_per_node=1 -m scripts.base_loss -- --device_batch_size="${BEST_DBSZ}" --split_tokens="${EVAL_TOKENS}"
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --max-per-task=32

# ======================================================================
# FROM HERE ON: repo-native mid/SFT (your working style)
# ======================================================================

# 9) MID-TRAIN
echo "==> mid-train (repo-native args only)"
torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --device_batch_size=${BEST_DBSZ}

# 10) EVAL (mid)
echo "==> eval (mid)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid

# 11) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device_batch_size=${BEST_DBSZ}

# 12) EVAL (sft)
echo "==> eval (sft)"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# 13) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
