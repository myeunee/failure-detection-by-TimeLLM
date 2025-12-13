#!/bin/bash

set -e

# trace 데이터로 baseline vs MLP+LSTM 비교

MODEL=TimeLLM
COMMENT_NONE="compare-trace-none"
COMMENT_MLPLSTM="compare-trace-mlplstm"

# 데이터셋: trace
DATASET=${1:-trace}
DATAPATH=${2:-trace.csv}

# trace.csv: Dataset_Trace 사용
# 두 타겟 변수: avg_usage_memory (regression), fail_in_window (classification)
# enc_in=2 (use_covariates=False일 때)
# 인스턴스 기반 분할 사용
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
  --batch_size 48 \
  --eval_batch_size 64 \
  --num_workers 0 \
  --learning_rate 0.0001 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 1 \
  --d_ff 64 \
  --use_amp \
  --multi_task \
  --cls_loss_weight 1.0 \
  --debug_samples 300000 \
  --train_epochs 20"

echo "========================================="
echo "[${DATASET}] Baseline (none)"
echo "========================================="
python3 run_main.py $COMMON_ARGS_BASE --model_comment $COMMENT_NONE --extra_head none

echo ""
echo "========================================="
echo "[${DATASET}] With MLP+LSTM"
echo "========================================="
python3 run_main.py $COMMON_ARGS_BASE --model_comment $COMMENT_MLPLSTM --extra_head mlp_lstm