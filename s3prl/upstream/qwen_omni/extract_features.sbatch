#!/bin/bash
#SBATCH -J qwen_extract_fix
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --qos qmultiple9
#SBATCH --time=48:00:00

source /home/s2025244189/envs/qwen_omni/bin/activate
cd /home/s2025244189/s1025244003/s3prl/s3prl/upstream/qwen_omni

MODEL_PATH="/home/s2025244189/s1025244003/Qwen2.5-Omni-7B"
LIBRI_ROOT="/home/s2025244189/datas/LibriSpeech"
OUTPUT_ROOT="/home/s2025244189/s1025244003/s3prl/s3prl/data/hidden_layer_all"
LOG_DIR="/home/s2025244189/s1025244003/s3prl/s3prl/upstream/qwen_omni/log_extract"

echo "开始提取特征: $(date)"


# train-clean-100
# CUDA_VISIBLE_DEVICES=7 python extract_qwen_features.py \
#     --model_path="${MODEL_PATH}" \
#     --libri_root="${LIBRI_ROOT}" \
#     --output_root="${OUTPUT_ROOT}" \
#     --split=train-clean-100 \
#     --gpu=0 \
#     2>&1 | tee "${LOG_DIR}/extract_train-clean-100.log" &
# wait


# train-clean-360
CUDA_VISIBLE_DEVICES=4 python extract_qwen_features.py \
    --model_path="${MODEL_PATH}" \
    --libri_root="${LIBRI_ROOT}" \
    --output_root="${OUTPUT_ROOT}" \
    --split=train-clean-360 \
    --gpu=0 \
    2>&1 | tee "${LOG_DIR}/extract_train-clean-360.log" &
wait


# # train-other-500
# CUDA_VISIBLE_DEVICES=7 python extract_qwen_features.py \
#     --model_path="${MODEL_PATH}" \
#     --libri_root="${LIBRI_ROOT}" \
#     --output_root="${OUTPUT_ROOT}" \
#     --split=train-other-500 \
#     --gpu=0 \
#     2>&1 | tee "${LOG_DIR}/extract_train-other-500.log" &
# wait


# # dev-clean & other
# CUDA_VISIBLE_DEVICES=7 python extract_qwen_features.py \
#     --model_path="${MODEL_PATH}" \
#     --libri_root="${LIBRI_ROOT}" \
#     --output_root="${OUTPUT_ROOT}" \
#     --split=dev \
#     --gpu=0 \
#     2>&1 | tee "${LOG_DIR}/extract_dev.log" &
# wait


# # test-clean & other
# CUDA_VISIBLE_DEVICES=7 python extract_qwen_features.py \
#     --model_path="${MODEL_PATH}" \
#     --libri_root="${LIBRI_ROOT}" \
#     --output_root="${OUTPUT_ROOT}" \
#     --split=test \
#     --gpu=0 \
#     2>&1 | tee "${LOG_DIR}/extract_test.log" &
# wait


echo "完成: $(date)"

# 验证输出
echo ""
echo "验证输出文件..."
for SPLIT in dev-clean dev-other; do
    SPLIT_DIR=$(echo ${SPLIT} | tr '-' '_')
    OUTPUT_DIR="${OUTPUT_ROOT}/lib_hidden_${SPLIT_DIR}"
    if [ -d "${OUTPUT_DIR}" ]; then
        NUM_FILES=$(find ${OUTPUT_DIR} -name "*_seq.npz" 2>/dev/null | wc -l)
        echo "  ${SPLIT}: ${NUM_FILES} 个文件"
    fi
done