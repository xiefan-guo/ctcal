#!/usr/bin/env bash
# End-to-end CTCal-on-FLUX reproduction.
#
#   bash scripts/run_pipeline.sh data      # build the reward-selected training set
#   bash scripts/run_pipeline.sh diagnose  # test the paper's premise on FLUX
#   bash scripts/run_pipeline.sh train     # GORS baseline + CTCal
#   bash scripts/run_pipeline.sh eval      # score every checkpoint
#
# Categories default to the three BLIP-VQA (attribute binding) ones, which is what
# the evaluation stack here covers.
#
# Environment (all optional):
#   CONDA_ENV      conda env to activate; omit to use the current interpreter
#   HF_ENDPOINT    Hub mirror, e.g. https://hf-mirror.com
#   CTCAL_FLUX_PATH  a staged local FLUX.1-dev directory; omit to download
#
# Everything else is a knob with a default: CATEGORIES, N_GPUS, MAX_STEPS,
# EVAL_PROMPTS, K, TOP_N, NUM_PROMPTS, DIAG_SAMPLES.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

CATEGORIES="${CATEGORIES:-color shape texture}"
N_GPUS="${N_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-2000}"
EVAL_PROMPTS="${EVAL_PROMPTS:-100}"

mkdir -p logs

stage="${1:?usage: run_pipeline.sh {data|diagnose|train|eval}}"

case "$stage" in

data)
  # Generate candidates and keep the top-scoring ones. Sharded across all GPUs.
  for cat in $CATEGORIES; do
    echo "=== building $cat ==="
    for ((g = 0; g < N_GPUS; g++)); do
      CUDA_VISIBLE_DEVICES=$g python scripts/build_dataset.py \
        --category "$cat" --split train \
        --k "${K:-8}" --top-n "${TOP_N:-2}" \
        --num-prompts "${NUM_PROMPTS:-700}" \
        --shard "$g" --num-shards "$N_GPUS" \
        >"logs/data_${cat}_g${g}.log" 2>&1 &
      # Stagger launches: simultaneous cold starts race on the transformers module
      # cache. The builder resumes, so a retry is cheap anyway.
      sleep 5
    done
    wait
    n=$(cat "data/train/$cat"/metadata.shard*.jsonl 2>/dev/null | wc -l)
    echo "$cat: $n pairs"
  done
  ;;

diagnose)
  # Run before training so the teacher timestep comes from measurement rather than
  # assumption.
  roots=""
  for cat in $CATEGORIES; do roots="$roots data/train/$cat"; done
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_attention.py \
    --data-roots $roots --tag pretrained --num-samples "${DIAG_SAMPLES:-48}" \
    2>&1 | tee logs/diagnose_pretrained.log
  ;;

train)
  roots=""
  for cat in $CATEGORIES; do roots="$roots data/train/$cat"; done

  echo "=== GORS baseline ==="
  CUDA_VISIBLE_DEVICES=0 python scripts/train_ctcal_flux.py \
    --mode gors --data-roots $roots \
    --output-dir runs/gors --max-steps "$MAX_STEPS" \
    2>&1 | tee logs/train_gors.log

  echo "=== CTCal ==="
  CUDA_VISIBLE_DEVICES=1 python scripts/train_ctcal_flux.py \
    --mode ctcal --data-roots $roots \
    --output-dir runs/ctcal --max-steps "$MAX_STEPS" \
    2>&1 | tee logs/train_ctcal.log
  ;;

eval)
  gpu=0
  for cat in $CATEGORIES; do
    for tag in baseline gors ctcal; do
      lora_arg=""
      if [ "$tag" != "baseline" ]; then
        ckpt=$(ls -d "runs/$tag"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
        [ -z "$ckpt" ] && { echo "skip $tag ($cat): no checkpoint"; continue; }
        lora_arg="--lora $ckpt"
      fi
      CUDA_VISIBLE_DEVICES=$gpu python scripts/evaluate.py \
        --category "$cat" --split val --tag "$tag" \
        --num-prompts "$EVAL_PROMPTS" $lora_arg \
        >"logs/eval_${cat}_${tag}.log" 2>&1 &
      gpu=$(((gpu + 1) % N_GPUS))
    done
  done
  wait
  python scripts/collect_results.py
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 1
  ;;
esac
