#!/usr/bin/env bash
set -e

# Edit these settings for your experiment.
export N_GPUS=4
export NNODES=1
export ROLLOUT_TP_SIZE=1
export TEMPLATED_INPUT=0
export BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"  # or Qwen/Qwen3-VL-4B-Thinking
export DATA_DIR="data/mixed"                 # data/text, data/image, or data/mixed
export MAX_PROMPT_LENGTH=5120                # 2048 for text; 5120 for image/mixed
export EXPERIMENT_NAME="Qwen3-VL-8B-IH-mixed"
export OUTPUT_DIR="checkpoints/$EXPERIMENT_NAME"

bash ./scripts/train_grpo.sh "$@"
