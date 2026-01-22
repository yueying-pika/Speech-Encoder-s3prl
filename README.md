# Speech-Encoder-s3prl (Qwen-Omni upstream)

本项目基于 s3prl 接入 **Qwen-Omni** 上游特征

## Quick Start

### 1) 运行脚本（SLURM）
```bash
sbatch Speech-Encoder-s3prl/s3prl/upstream/qwen_omni/run_qwen_online.sbatch

配置文件（Downstream: ASR）
Speech-Encoder-s3prl/s3prl/s3prl/downstream/asr/config_qwen_omni_960h_online_v2.yaml

Upstream (Qwen-Omni)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/qwen_omni/

Expert (online)
Speech-Encoder-s3prl/s3prl/s3prl/upstream/qwen_omni/expert_online.py


