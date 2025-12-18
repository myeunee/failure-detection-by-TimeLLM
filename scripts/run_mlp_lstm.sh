#!/bin/bash

#SBATCH --job-name=timellm_mlp_lstm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition=batch_ce_ugrad
#SBATCH -w moana-r5

# 작업 디렉토리
cd /data/myeunee/graduation_proj/failure-detection-by-TimeLLM

# Conda 환경 활성화
source ~/.bashrc
conda activate /data/myeunee/timellm

# Hugging Face 캐시 디렉토리를 /data로 강제 설정 (공간 부족 문제 해결)
export HF_HOME=/data/myeunee/.cache/huggingface
export TRANSFORMERS_CACHE=/data/myeunee/.cache/huggingface
export HF_DATASETS_CACHE=/data/myeunee/.cache/huggingface
mkdir -p $HF_HOME

set -e

# trace 데이터로 MLP+LSTM 실행

MODEL=TimeLLM
COMMENT_MLPLSTM="compare-trace-mlplstm"

# 데이터셋: trace
DATASET=${1:-trace}
DATAPATH=${2:-trace.csv}

COMMON_ARGS_BASE="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ${DATAPATH} \
  --model_id ${DATASET}_256_32 \
  --model $MODEL \
  --data Trace \
  --features M \
  --target avg_usage_memory \
  --reg_col avg_usage_memory \
  --cls_col fail_in_window \
  --trace_use_covariates \
  --seq_len 12 \
  --patch_len 4 \
  --stride 2 \
  --label_len 6 \
  --pred_len 3 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 2 \
  --batch_size 24 \
  --eval_batch_size 32 \
  --num_workers 0 \
  --learning_rate 0.0001 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 1 \
  --d_ff 64 \
  --multi_task \
  --cls_loss_weight 1.0 \
  --debug_samples 300000 \
  --train_epochs 15"

echo "========================================="
echo "[${DATASET}] With MLP+LSTM"
echo "========================================="
python3 run_main.py $COMMON_ARGS_BASE --model_comment $COMMENT_MLPLSTM --extra_head mlp_lstm



