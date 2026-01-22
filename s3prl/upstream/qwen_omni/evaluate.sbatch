#!/bin/bash
#SBATCH -J eval_asr_detailed
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --qos qmultiple9
#SBATCH --output=slurm-eval-%j.out

source /home/s2025244189/envs/qwen_omni/bin/activate
cd /home/s2025244189/s1025244003/s3prl/s3prl
export PYTHONPATH=/home/s2025244189/s1025244003/s3prl:$PYTHONPATH

LOG_DIR="/home/s2025244189/s1025244003/s3prl/s3prl/upstream/qwen_omni/log_evaluate"
RESULT_DIR="/home/s2025244189/s1025244003/s3prl/s3prl/upstream/qwen_omni/analysis_results"

BEST_CKPT="/home/s2025244189/s1025244003/s3prl/s3prl/result/downstream/dev-clean-best.ckpt"

# 评估并保存详细结果
echo "=== Evaluating test-clean ==="
CUDA_VISIBLE_DEVICES=4 python run_downstream.py \
    -m evaluate \
    -e $BEST_CKPT \
    -t test-clean \
    2>&1 | tee "${LOG_DIR}/eval_test_clean_best_960h_layer_all_online_hubert-large.log"

echo "=== Evaluating test-other ==="
CUDA_VISIBLE_DEVICES=4 python run_downstream.py \
    -m evaluate \
    -e $BEST_CKPT \
    -t test-other \
    2>&1 | tee "${LOG_DIR}/eval_test_other_best_960h_layer_all_online_hubert-large.log"

echo "=== Evaluation Done ==="