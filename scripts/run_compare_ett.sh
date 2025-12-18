#!/bin/bash

#SBATCH --job-name=timellm_compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition=batch_ce_ugrad
#SBATCH -w moana-r4

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

# 로그 디렉토리 생성
mkdir -p logs

# 비교 실험 실행
bash scripts/compare_ett.sh | tee logs/compare_ett_batch_seq_256.log