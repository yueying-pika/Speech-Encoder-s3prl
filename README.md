# Speech-Encoder-s3prl (Qwen-Omni upstream)

本项目基于 s3prl 接入 **Qwen-Omni** 上游特征

## Quick Start

### 1) 环境安装

conda create -n speech_encoder python=3.10
conda activate speech_encoder
pip install s3prl


### 2) 训练脚本
```bash
sbatch Speech-Encoder-s3prl/s3prl/upstream/qwen_omni/run_qwen_online.sbatch

配置文件（Downstream: ASR）
Speech-Encoder-s3prl/s3prl/s3prl/downstream/asr/config_qwen_omni_100h_online.yaml
Speech-Encoder-s3prl/s3prl/s3prl/downstream/asr/config_qwen_omni_960h_online.yaml

Upstream (speech encoder)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/qwen_omni/

Expert (online)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/qwen_omni/expert_online.py

Downstream (librispeech)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/asr

Downstream (asr_aishell_1)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/asr_aishell_1

Downstream (asr_seame)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/asr_seame

### 3) 评估脚本
sbatch Speech-Encoder-s3prl/s3prl/upstream/qwen_omni/evaluate.sbatch
