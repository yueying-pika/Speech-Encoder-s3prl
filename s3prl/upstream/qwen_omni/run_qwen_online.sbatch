#!/bin/bash
#SBATCH -J mytraining           # 作业名（随意取）
#SBATCH -p A800Z                 # 使用A800资源池
#SBATCH -N 1                     # 使用1个节点
#SBATCH --gres=gpu:2             # 每个节点申请3张GPU
#SBATCH --cpus-per-task=8        # 每个任务4核CPU
#SBATCH --qos qmultiple9         # 使用正式队列

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

source /home/s2025244189/envs/qwen_omni/bin/activate
cd /home/s2025244189/s1025244003/s3prl/s3prl

export PYTHONPATH=/home/s2025244189/s1025244003/s3prl:$PYTHONPATH
export PATH=/home/s2025244189/envs/qwen_omni/bin:$PATH
export LD_LIBRARY_PATH=/home/s2025244189/envs/qwen_omni/lib:$LD_LIBRARY_PATH

LOG_DIR="/home/s2025244189/s1025244003/s3prl/s3prl/upstream/qwen_omni/log_train"
export NCCL_TIMEOUT=1800

# 使用Singularity容器运行PyTorch程序
# 只需要修改-n，这是.pt保存的地址result/downstream/Qwen_ASR_960h_layer

CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 run_downstream.py \
    -m train \
    -u qwen_omni_online \
    -n Qwen_ASR_960h_layer_all_online_audio_tower \
    -d asr \
    -c downstream/asr/config_qwen_omni_960h_online_v2.yaml \
    -f \
    2>&1 | tee "${LOG_DIR}/qwen_ASR_960h_layer_all_online_audio_tower.log"


# -s last_hidden_state 最后一层特征
#  -e result/downstream/Qwen_ASR_cached_100h \   从上一次的模型开始训练
# parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')

# -u, hubconf.py 中 qwen_omni_cached 函数
# parser.add_argument('-u', '--upstream',)

# -n, .pt保存的地址result/downstream/Qwen_ASR_960h_layer
# parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')

# -c, .yaml配置文件
# parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

# -f,  True，表示要微调上游模型
# parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')